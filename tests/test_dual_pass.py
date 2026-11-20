"""Unit tests for the Qwen3 training path (block construction, mask rule, Eq. 7 loss).

These deliberately avoid importing `src.model`, so they run without CUDA,
FlexAttention or transformers -- only torch is needed:

    python -m pytest tests/test_dual_pass.py -q
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from itertools import product
from types import SimpleNamespace

from src.train.dual_pass import build_diffusion_blocks, dual_pass_forward
from src.train.loss import IGNORE_INDEX, block_local_labels
from src.train.loss_kl import (
    fused_linear_accepted_length,
    fused_linear_cross_entropy,
    fused_linear_expected_rounds,
    fused_linear_forward_kl,
)

MASK_ID = 999


def _toy(batch=2, seq_len=64, block_size=8, num_anchors=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, 500, (batch, seq_len), generator=generator)
    # Ascending, >= 1, and leaving room for a full block.
    anchors = torch.stack([
        torch.sort(torch.randperm(seq_len - block_size, generator=generator)[:num_anchors]
                   + 1).values
        for _ in range(batch)
    ])
    return input_ids, anchors, block_size


# Eq. 5: block = [anchor, <mask> x (K-1)]

def test_blocks_keep_anchor_and_mask_the_rest():
    input_ids, anchors, block_size = _toy()
    batch, num_anchors = anchors.shape

    block_ids, positions, causal_limit = build_diffusion_blocks(
        input_ids, anchors, block_size, MASK_ID
    )
    blocks = block_ids.reshape(batch, num_anchors, block_size)

    expected_anchor_tokens = torch.gather(input_ids, 1, anchors)
    assert torch.equal(blocks[:, :, 0], expected_anchor_tokens)
    assert (blocks[:, :, 1:] == MASK_ID).all()
    assert blocks.shape[-1] == block_size


def test_block_positions_are_absolute():
    """RoPE must see a_b + j, not the packed offset."""
    input_ids, anchors, block_size = _toy()
    batch, num_anchors = anchors.shape

    _, positions, _ = build_diffusion_blocks(input_ids, anchors, block_size, MASK_ID)
    positions = positions.reshape(batch, num_anchors, block_size)

    offsets = torch.arange(block_size)
    assert torch.equal(positions, anchors.unsqueeze(-1) + offsets.reshape(1, 1, -1))


# Eq. 6: a block query attends to the STRICT clean prefix before its own anchor.

def test_causal_limit_encodes_strict_prefix():
    input_ids, anchors, block_size = _toy()
    batch, num_anchors = anchors.shape

    _, _, causal_limit = build_diffusion_blocks(input_ids, anchors, block_size, MASK_ID)
    causal_limit = causal_limit.reshape(batch, num_anchors, block_size)

    # Every position in block b carries the same limit, a_b - 1.
    assert torch.equal(causal_limit, (anchors - 1).unsqueeze(-1).expand_as(causal_limit))

    # `generate_dual_pass_mask` (src/model.py:62) tests `kv_idx <= causal_limit`.
    # That must be exactly equivalent to the Qwen3.5 twin's `kv_idx < anchors[...]`
    # (modeling_orthrus_qwen3_5.py:90) -- i.e. the anchor itself is NOT an AR key.
    seq_len = input_ids.shape[1]
    kv_idx = torch.arange(seq_len).reshape(1, 1, 1, -1)
    ours = kv_idx <= causal_limit.unsqueeze(-1)
    qwen35 = (kv_idx < anchors.reshape(batch, num_anchors, 1, 1)).expand_as(ours)
    assert torch.equal(ours, qwen35)

    # And no query may see its own anchor position or anything after it.
    assert not ours.gather(
        3, anchors.reshape(batch, num_anchors, 1, 1).expand(-1, -1, block_size, -1)
    ).any()


def test_anchors_never_reach_position_zero_underflow():
    """anchors >= 1 (sample_anchors' min_anchor), so causal_limit >= 0."""
    input_ids, anchors, block_size = _toy()
    _, _, causal_limit = build_diffusion_blocks(input_ids, anchors, block_size, MASK_ID)
    assert (causal_limit >= 0).all()


# Supervision grid: K-1 positions per block, aligned to AR row a_b + j.

def test_supervised_positions_are_block_size_minus_one():
    input_ids, anchors, block_size = _toy()
    batch, num_anchors = anchors.shape

    labels = block_local_labels(input_ids, anchors, block_size)
    assert labels.shape == (batch, num_anchors, block_size)

    # Last offset is never supervised: its target lies outside the block, and
    # generate() discards that row too (src/model.py:477).
    assert (labels[:, :, -1] == IGNORE_INDEX).all()
    assert (labels[:, :, :-1] != IGNORE_INDEX).all()

    # Offset j predicts the token after absolute position a_b + j.
    offsets = torch.arange(block_size - 1)
    target_positions = anchors.unsqueeze(-1) + offsets.reshape(1, 1, -1) + 1
    expected = torch.gather(
        input_ids.unsqueeze(1).expand(batch, num_anchors, -1), 2, target_positions
    )
    assert torch.equal(labels[:, :, :-1], expected)


def test_teacher_gather_matches_student_grid():
    """The teacher row for student offset j is the AR hidden state at a_b + j."""
    input_ids, anchors, block_size = _toy()
    batch, num_anchors = anchors.shape
    hidden_size = 5

    # Fake AR hidden states whose value encodes the position, so gathers are checkable.
    seq_len = input_ids.shape[1]
    ar_hidden = torch.arange(seq_len, dtype=torch.float32)
    ar_hidden = ar_hidden.reshape(1, seq_len, 1).expand(batch, seq_len, hidden_size)

    offsets = torch.arange(block_size)
    positions = anchors.unsqueeze(-1) + offsets.reshape(1, 1, -1)
    gather_index = positions.reshape(batch, -1, 1).expand(-1, -1, hidden_size)
    teacher = torch.gather(ar_hidden, 1, gather_index).reshape(
        batch, num_anchors, block_size, hidden_size
    )

    assert torch.equal(teacher[..., 0], positions.float())


# Eq. 7: forward KL against the frozen teacher.

def _naive_forward_kl(weight, student, teacher, keep):
    student, teacher = student[keep], teacher[keep]
    t_logprob = F.log_softmax(F.linear(teacher, weight).float(), dim=-1)
    s_logprob = F.log_softmax(F.linear(student, weight).float(), dim=-1)
    return (t_logprob.exp() * (t_logprob - s_logprob)).sum(-1).mean()


def test_chunked_kl_matches_naive_reference():
    torch.manual_seed(0)
    num_positions, hidden_size, vocab = 37, 8, 23
    weight = torch.randn(vocab, hidden_size)
    student = torch.randn(num_positions, hidden_size, requires_grad=True)
    teacher = torch.randn(num_positions, hidden_size)
    keep = torch.rand(num_positions) > 0.3

    # Chunk size deliberately smaller than the position count, and not a divisor.
    got = fused_linear_forward_kl(weight, student, teacher, keep, chunk_size=5)
    expected = _naive_forward_kl(weight, student, teacher, keep)
    assert torch.allclose(got, expected, atol=1e-6), (got.item(), expected.item())


def test_kl_is_zero_when_views_agree():
    torch.manual_seed(0)
    hidden = torch.randn(16, 8)
    weight = torch.randn(23, 8)
    loss = fused_linear_forward_kl(
        weight, hidden.clone().requires_grad_(True), hidden, torch.ones(16, dtype=torch.bool)
    )
    assert loss.abs().item() < 1e-6


def test_gradients_reach_student_only():
    torch.manual_seed(0)
    weight = torch.randn(23, 8, requires_grad=True)
    student = torch.randn(16, 8, requires_grad=True)
    teacher = torch.randn(16, 8)

    fused_linear_forward_kl(
        weight, student, teacher, torch.ones(16, dtype=torch.bool), chunk_size=4
    ).backward()

    assert student.grad is not None and student.grad.abs().sum() > 0
    # The teacher is detached inside the loss, so it can never accumulate grad.
    assert teacher.grad is None


def test_kl_rejects_a_teacher_that_carries_grad():
    weight = torch.randn(23, 8)
    student = torch.randn(4, 8, requires_grad=True)
    teacher = torch.randn(4, 8, requires_grad=True)
    try:
        fused_linear_forward_kl(weight, student, teacher, torch.ones(4, dtype=torch.bool))
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("expected a ValueError for a non-detached teacher")


def test_chunked_ce_matches_reference():
    torch.manual_seed(0)
    num_positions, hidden_size, vocab = 29, 8, 23
    weight = torch.randn(vocab, hidden_size)
    student = torch.randn(num_positions, hidden_size, requires_grad=True)
    labels = torch.randint(0, vocab, (num_positions,))
    labels[::4] = IGNORE_INDEX

    got = fused_linear_cross_entropy(weight, student, labels, chunk_size=6)
    expected = F.cross_entropy(
        F.linear(student, weight).float(), labels, ignore_index=IGNORE_INDEX
    )
    assert torch.allclose(got, expected, atol=1e-6), (got.item(), expected.item())


def test_empty_supervision_keeps_the_graph_alive():
    weight = torch.randn(23, 8)
    student = torch.randn(4, 8, requires_grad=True)
    teacher = torch.randn(4, 8)
    loss = fused_linear_forward_kl(weight, student, teacher, torch.zeros(4, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0.0
    assert student.grad is not None  # every rank still produces a gradient


# Stage 2: target-path accepted length for stochastic speculative verification.

def _naive_accepted_length(weight, student, teacher, labels, block_size):
    """Dense autograd reference: differentiate J directly, with no surrogate."""
    num_blocks = labels.shape[0] // block_size
    hidden_size = student.shape[-1]
    student = student.reshape(num_blocks, block_size, hidden_size)[:, :-1]
    teacher = teacher.reshape(num_blocks, block_size, hidden_size)[:, :-1]
    labels = labels.reshape(num_blocks, block_size)[:, :-1]

    prefix = (labels != IGNORE_INDEX).cumprod(dim=-1).bool()
    keep = prefix.any(dim=-1)
    student, teacher, labels, prefix = (
        student[keep], teacher[keep], labels[keep], prefix[keep]
    )
    labels = labels.masked_fill(~prefix, 0)

    with torch.no_grad():
        p = F.softmax(F.linear(teacher, weight).float(), dim=-1)
        p = p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    q = F.softmax(F.linear(student, weight).float(), dim=-1)
    q = q.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    accept = torch.where(prefix, (q / p).clamp_max(1.0), torch.ones_like(q))
    survival = accept.cumprod(dim=-1) * prefix
    return -survival.sum(dim=-1).mean()


def _accepted_toy():
    torch.manual_seed(7)
    num_blocks, block_size, hidden_size, vocab = 4, 5, 6, 17
    weight = torch.randn(vocab, hidden_size) * 0.4
    student = torch.randn(num_blocks * block_size, hidden_size, requires_grad=True)
    teacher = torch.randn(num_blocks * block_size, hidden_size) * 0.4
    labels = torch.randint(0, vocab, (num_blocks, block_size))
    labels[:, -1] = IGNORE_INDEX
    labels[0, 2] = IGNORE_INDEX       # this block has a supervised prefix of length 2
    labels[2, :] = IGNORE_INDEX       # invalid anchor: excluded from block normalization
    labels[3, 1] = IGNORE_INDEX       # only the first target survives in this block
    return weight, student, teacher, labels.reshape(-1), block_size


def test_accepted_length_matches_dense_value_and_gradient():
    weight, student, teacher, labels, block_size = _accepted_toy()
    reference_student = student.detach().clone().requires_grad_(True)

    # chunk_size < K-1 deliberately exercises whole-block chunking.
    got = fused_linear_accepted_length(
        weight, student, teacher, labels, block_size, chunk_size=3
    )
    expected = _naive_accepted_length(weight, reference_student, teacher, labels, block_size)
    assert torch.allclose(got, expected, atol=1e-6), (got.item(), expected.item())

    got.backward()
    expected.backward()
    assert torch.allclose(student.grad, reference_student.grad, atol=2e-6), (
        (student.grad - reference_student.grad).abs().max().item()
    )


def test_accepted_length_gradient_matches_finite_difference():
    torch.manual_seed(11)
    num_blocks, block_size, hidden_size, vocab = 1, 4, 3, 9
    weight = torch.randn(vocab, hidden_size) * 0.3
    student = torch.randn(num_blocks * block_size, hidden_size, requires_grad=True)
    teacher = torch.randn(num_blocks * block_size, hidden_size) * 0.3
    labels = torch.tensor([2, 4, 1, IGNORE_INDEX])

    loss = fused_linear_accepted_length(
        weight, student, teacher, labels, block_size, chunk_size=2
    )
    loss.backward()
    index = (1, 2)  # offset 1 is supervised; offset K-1 is not
    analytic = student.grad[index].item()

    epsilon = 1e-3
    values = []
    for sign in (-1.0, 1.0):
        perturbed = student.detach().clone()
        perturbed[index] += sign * epsilon
        values.append(fused_linear_accepted_length(
            weight, perturbed, teacher, labels, block_size, chunk_size=2
        ).item())
    numerical = (values[1] - values[0]) / (2.0 * epsilon)
    assert abs(analytic - numerical) < 2e-3, (analytic, numerical)


def test_accepted_length_masks_truncate_and_do_not_leak_gradients():
    weight, student, teacher, labels, block_size = _accepted_toy()
    baseline = fused_linear_accepted_length(
        weight, student, teacher, labels, block_size, chunk_size=16
    )

    # Neither an ignored tail nor a wholly invalid block may affect the objective.
    perturbed = student.detach().clone()
    grid = perturbed.reshape(num_blocks := 4, block_size, -1)
    grid[0, 2:] += 1000.0
    grid[2, :] -= 1000.0
    changed = fused_linear_accepted_length(
        weight, perturbed, teacher, labels, block_size, chunk_size=16
    )
    assert torch.allclose(baseline, changed, atol=1e-6)

    baseline.backward()
    grad = student.grad.reshape(num_blocks, block_size, -1)
    assert torch.count_nonzero(grad[0, 2:]) == 0
    assert torch.count_nonzero(grad[2]) == 0
    # The final offset is never a proposal consumed by generate().
    assert torch.count_nonzero(grad[:, -1]) == 0


def test_accepted_length_is_capped_and_has_zero_local_gradient_when_certain():
    # p is uniform; Q puts essentially all mass on every committed target token.
    weight = torch.eye(3)
    teacher = torch.zeros(4, 3)
    labels = torch.tensor([0, 1, 2, IGNORE_INDEX])
    student = torch.zeros(4, 3, requires_grad=True)
    with torch.no_grad():
        student[0, 0] = student[1, 1] = student[2, 2] = 20.0

    loss = fused_linear_accepted_length(
        weight, student, teacher, labels, block_size=4, chunk_size=3
    )
    assert torch.allclose(loss, torch.tensor(-3.0), atol=1e-6)
    loss.backward()
    assert student.grad.abs().max().item() < 1e-6


def test_accepted_length_empty_supervision_keeps_graph_alive():
    weight = torch.randn(11, 4)
    student = torch.randn(6, 4, requires_grad=True)
    teacher = torch.randn(6, 4)
    labels = torch.full((6,), IGNORE_INDEX)
    loss = fused_linear_accepted_length(
        weight, student, teacher, labels, block_size=3, chunk_size=2
    )
    loss.backward()
    assert loss.item() == 0.0
    assert student.grad is not None


def test_onpolicy_eal_matches_exact_enumeration_including_score_gradient():
    """Enumerate every draft path: a fixed-path derivative would fail this test."""
    torch.manual_seed(29)
    length, vocab = 3, 2
    block_size = length + 1
    base_logits = torch.randn(length, vocab) * 0.7
    exact_logits = base_logits.clone().requires_grad_(True)
    estimator_logits = base_logits.clone().requires_grad_(True)
    weight = torch.eye(vocab)

    # A separate AR conditional for every sampled prefix, stored as logits.
    teacher_logits = {}
    for path in product(range(vocab), repeat=length):
        for s in range(length):
            key = (s, path[:s])
            if key not in teacher_logits:
                teacher_logits[key] = torch.randn(vocab) * 0.7

    def path_probabilities(logits, path):
        q_values = []
        p_values = []
        for s, token in enumerate(path):
            q_values.append(F.softmax(logits[s] / 1.0, dim=-1)[token])
            p_values.append(F.softmax(
                teacher_logits[(s, path[:s])] / 1.0, dim=-1
            )[token])
        return q_values, p_values

    # Exact deployed expectation: sum_y Q(y) J(y), differentiated directly.
    exact_expected_j = exact_logits.new_zeros(())
    for path in product(range(vocab), repeat=length):
        q_values, p_values = path_probabilities(exact_logits, path)
        acceptance = [torch.minimum(
            torch.ones_like(p_values[s]), p_values[s] / q_values[s]
        ) for s in range(length)]
        survival = torch.cumprod(torch.stack(acceptance), dim=0)
        draft_probability = torch.stack(q_values).prod()
        exact_expected_j = exact_expected_j + draft_probability * survival.sum()
    (-exact_expected_j).backward()

    # The implementation is evaluated on every sampled path, weighted by the
    # fixed sampling law. Its surrogate must reproduce the FULL expectation
    # gradient, not merely the derivative through p/Q along one fixed path.
    estimated_loss = estimator_logits.new_zeros(())
    for path in product(range(vocab), repeat=length):
        q_detached = F.softmax(base_logits, dim=-1)
        draft_probability = torch.stack([
            q_detached[s, token] for s, token in enumerate(path)
        ]).prod().detach()
        student = torch.cat([
            estimator_logits,
            estimator_logits.new_zeros(1, vocab),
        ], dim=0)
        teacher = torch.cat([
            torch.stack([
                teacher_logits[(s, path[:s])] for s in range(length)
            ]),
            torch.zeros(1, vocab),
        ], dim=0)
        labels = torch.tensor([*path, IGNORE_INDEX])
        path_loss = fused_linear_accepted_length(
            weight, student, teacher, labels, block_size,
            chunk_size=2, temperature=1.0, on_policy=True,
        )
        estimated_loss = estimated_loss + draft_probability * path_loss

    assert torch.allclose(estimated_loss, -exact_expected_j, atol=2e-6)
    estimated_loss.backward()
    assert torch.allclose(estimator_logits.grad, exact_logits.grad, atol=3e-5), (
        estimator_logits.grad, exact_logits.grad
    )



def test_onpolicy_dual_pass_preserves_batch_anchor_shape():
    """Regression: sampled [B*A, K-1] tokens must be restored to [B, A, K-1]."""
    torch.manual_seed(41)
    batch, num_anchors, block_size = 2, 3, 4
    seq_len, hidden_size, vocab = 32, 5, 11
    input_ids = torch.randint(0, vocab, (batch, seq_len))
    anchors = torch.tensor([[1, 8, 15], [2, 9, 16]])

    class FakeModel:
        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, **kwargs):
            self.calls.append((input_ids.clone(), kwargs))
            hidden = torch.zeros(
                *input_ids.shape, hidden_size,
                requires_grad=kwargs.get("is_diffusion_pass", False),
            )
            cache = object() if kwargs.get("use_cache", False) else None
            return SimpleNamespace(hidden_states=(hidden,), past_key_values=cache)

    model = FakeModel()
    weight = torch.randn(vocab, hidden_size)
    student, teacher, keep, labels = dual_pass_forward(
        model,
        input_ids=input_ids,
        anchors=anchors,
        block_size=block_size,
        mask_token_id=MASK_ID,
        on_policy=True,
        lm_head_weight=weight,
        sampling_generator=torch.Generator().manual_seed(43),
        sampling_chunk_size=2,
    )

    num_positions = batch * num_anchors * block_size
    assert student.shape == (num_positions, hidden_size)
    assert teacher.shape == (num_positions, hidden_size)
    assert keep.shape == (num_positions,)
    assert labels.shape == (num_positions,)
    assert len(model.calls) == 3

    verification_ids, verification_kwargs = model.calls[2]
    assert verification_kwargs["is_verification_pass"]
    verification_blocks = verification_ids.reshape(batch, num_anchors, block_size)
    expected_anchors = torch.gather(input_ids, 1, anchors)
    assert torch.equal(verification_blocks[:, :, 0], expected_anchors)

    label_grid = labels.reshape(batch, num_anchors, block_size)
    assert (label_grid[:, :, :-1] != IGNORE_INDEX).all()
    assert (label_grid[:, :, -1] == IGNORE_INDEX).all()


# ---------------------------------------------------------------------------------
# RB-EAL, leave-one-out baselines, multi-draft trees, expected rounds
# ---------------------------------------------------------------------------------


def _random_prefix_teacher(length, vocab, scale=0.7):
    """A separate AR conditional (as logits) for every drafted prefix."""
    table = {}
    for s in range(length):
        for prefix in product(range(vocab), repeat=s):
            table[prefix] = torch.randn(vocab) * scale
    return table


def _exact_single_draft_ej(logits, teacher_logits):
    length, vocab = logits.shape
    q = F.softmax(logits, dim=-1)
    total = logits.new_zeros(())
    for path in product(range(vocab), repeat=length):
        prob, survival, j = logits.new_ones(()), logits.new_ones(()), logits.new_zeros(())
        for s, token in enumerate(path):
            p_token = F.softmax(teacher_logits[path[:s]], dim=-1)[token]
            survival = survival * torch.clamp(p_token / q[s, token], max=1.0)
            j = j + survival
            prob = prob * q[s, token]
        total = total + prob * j
    return total


def _single_draft_inputs(logits, teacher_logits, paths):
    """student [K, V], teacher [M*K, V], labels [M*K] for M drafts of one block."""
    vocab = logits.shape[1]
    student = torch.cat([logits, logits.new_zeros(1, vocab)], dim=0)
    teachers, labels = [], []
    for path in paths:
        length = len(path)
        teachers.append(torch.cat([
            torch.stack([teacher_logits[path[:s]] for s in range(length)]),
            torch.zeros(1, vocab),
        ], dim=0))
        labels.append(torch.tensor([*path, IGNORE_INDEX]))
    return student, torch.cat(teachers, dim=0), torch.cat(labels, dim=0)


def test_rb_eal_matches_exact_enumeration_including_score_gradient():
    torch.manual_seed(31)
    length, vocab = 3, 3
    base = torch.randn(length, vocab) * 0.8
    teacher_logits = _random_prefix_teacher(length, vocab, scale=0.9)
    exact_logits = base.clone().requires_grad_(True)
    exact = _exact_single_draft_ej(exact_logits, teacher_logits)
    (-exact).backward()

    est_logits = base.clone().requires_grad_(True)
    q_fixed = F.softmax(base, dim=-1)
    estimated = est_logits.new_zeros(())
    for path in product(range(vocab), repeat=length):
        weight = torch.stack([q_fixed[s, t] for s, t in enumerate(path)]).prod()
        student, teacher, labels = _single_draft_inputs(est_logits, teacher_logits, [path])
        estimated = estimated + weight * fused_linear_accepted_length(
            torch.eye(vocab), student, teacher, labels, length + 1,
            chunk_size=2, estimator="rb",
        )
    assert torch.allclose(estimated, -exact, atol=2e-6)
    estimated.backward()
    assert torch.allclose(est_logits.grad, exact_logits.grad, atol=3e-5), (
        est_logits.grad, exact_logits.grad
    )


def test_leave_one_out_baseline_is_unbiased_with_two_drafts():
    torch.manual_seed(37)
    length, vocab = 2, 3
    base = torch.randn(length, vocab) * 0.8
    teacher_logits = _random_prefix_teacher(length, vocab, scale=0.9)
    exact_logits = base.clone().requires_grad_(True)
    (-_exact_single_draft_ej(exact_logits, teacher_logits)).backward()

    q_fixed = F.softmax(base, dim=-1)
    paths = list(product(range(vocab), repeat=length))
    for estimator in ("onpolicy", "rb"):
        est_logits = base.clone().requires_grad_(True)
        estimated = est_logits.new_zeros(())
        for first in paths:
            for second in paths:
                weight = torch.stack(
                    [q_fixed[s, t] for s, t in enumerate(first)]
                    + [q_fixed[s, t] for s, t in enumerate(second)]
                ).prod()
                student, teacher, labels = _single_draft_inputs(
                    est_logits, teacher_logits, [first, second]
                )
                estimated = estimated + weight * fused_linear_accepted_length(
                    torch.eye(vocab), student, teacher, labels, length + 1,
                    chunk_size=4, estimator=estimator, num_drafts=2, baseline="loo",
                )
        estimated.backward()
        assert torch.allclose(est_logits.grad, exact_logits.grad, atol=3e-5), (
            estimator, est_logits.grad, exact_logits.grad
        )


def _residual_chain(p, q, count):
    residuals, masses = [p], []
    for _ in range(count):
        excess = torch.clamp(residuals[-1] - q, min=0.0)
        mass = excess.sum()
        masses.append(mass)
        residuals.append(excess / mass.clamp_min(1e-30))
    return residuals[:count], masses


def _exact_multidraft_el(logits, teacher_logits, num_drafts):
    """sum_Y Q(Y) E_coins[accepted length] of tree verification, by recursion."""
    length, vocab = logits.shape
    q = F.softmax(logits, dim=-1)

    def recurse(s, prefix, alive, chains):
        if s == length or not alive:
            return logits.new_zeros(())
        p = F.softmax(teacher_logits[tuple(prefix)], dim=-1)
        residuals, _ = _residual_chain(p, q[s], len(alive))
        total, reach = logits.new_zeros(()), logits.new_ones(())
        for j, chain in enumerate(alive):
            token = chains[chain][s]
            accept = torch.clamp(residuals[j][token] / q[s, token], max=1.0)
            survivors = [c for c in alive if chains[c][s] == token]
            total = total + reach * accept * (
                1.0 + recurse(s + 1, prefix + [token], survivors, chains)
            )
            reach = reach * (1.0 - accept)
        return total

    total = logits.new_zeros(())
    for flat in product(range(vocab), repeat=num_drafts * length):
        chains = [flat[c * length:(c + 1) * length] for c in range(num_drafts)]
        prob = torch.stack([q[s, chains[c][s]]
                            for c in range(num_drafts) for s in range(length)]).prod()
        total = total + prob * recurse(0, [], list(range(num_drafts)), chains)
    return total


def _decision_weight(q, teacher_logits, chains, decisions):
    """Probability of the tested accept/reject decisions; 1/2 per untested slot."""
    length = q.shape[0]
    num_drafts = len(chains)
    alive, prefix = list(range(num_drafts)), []
    weight, tested = 1.0, torch.zeros(length, num_drafts, dtype=torch.bool)
    for s in range(length):
        p = F.softmax(teacher_logits[tuple(prefix)], dim=-1)
        residuals, _ = _residual_chain(p, q[s], len(alive))
        accepted = None
        for j, chain in enumerate(alive):
            token = chains[chain][s]
            accept = min(1.0, (residuals[j][token] / q[s, token]).item())
            tested[s, chain] = True
            weight *= accept if decisions[s][chain] else 1.0 - accept
            if decisions[s][chain]:
                accepted = token
                break
        if accepted is None:
            break
        prefix.append(accepted)
        alive = [c for c in alive if chains[c][s] == accepted]
    return weight * 0.5 ** int((~tested).sum())


def test_multidraft_tree_matches_exact_enumeration_including_score_gradient():
    """Enumerate both drafted chains and every accept coin of the tree walk."""
    torch.manual_seed(43)
    length, vocab, num_drafts = 2, 3, 2
    base = torch.randn(length, vocab) * 0.8
    teacher_logits = _random_prefix_teacher(length, vocab, scale=0.9)
    exact_logits = base.clone().requires_grad_(True)
    exact = _exact_multidraft_el(exact_logits, teacher_logits, num_drafts)
    (-exact).backward()

    q_fixed = F.softmax(base, dim=-1)
    est_logits = base.clone().requires_grad_(True)
    estimated = est_logits.new_zeros(())
    for flat in product(range(vocab), repeat=num_drafts * length):
        chains = [flat[c * length:(c + 1) * length] for c in range(num_drafts)]
        token_weight = torch.stack([q_fixed[s, chains[c][s]]
                                    for c in range(num_drafts)
                                    for s in range(length)]).prod().item()
        student, teacher, labels = _single_draft_inputs(est_logits, teacher_logits, chains)
        for bits in product((False, True), repeat=length * num_drafts):
            decisions = [bits[s * num_drafts:(s + 1) * num_drafts] for s in range(length)]
            weight = token_weight * _decision_weight(q_fixed, teacher_logits, chains,
                                                     decisions)
            if weight == 0.0:
                continue
            # u = 0 accepts whenever the acceptance probability is positive; u = 1 never.
            uniforms = torch.tensor(
                [[0.0 if decisions[s][c] else 1.0 for c in range(num_drafts)]
                 for s in range(length)]
            ).unsqueeze(0)
            estimated = estimated + weight * fused_linear_accepted_length(
                torch.eye(vocab), student, teacher, labels, length + 1,
                chunk_size=8, estimator="multidraft", num_drafts=num_drafts,
                uniforms=uniforms,
            )
    assert torch.allclose(estimated, -exact, atol=2e-6), (estimated, exact)
    estimated.backward()
    assert torch.allclose(est_logits.grad, exact_logits.grad, atol=3e-5), (
        est_logits.grad, exact_logits.grad
    )
    # Two i.i.d. drafts verified as a tree accept more than one draft does.
    single = _exact_single_draft_ej(base, teacher_logits)
    assert exact.item() > single.item()


def test_multidraft_with_one_draft_has_single_draft_mean():
    torch.manual_seed(47)
    length, vocab = 3, 2
    base = torch.randn(length, vocab)
    teacher_logits = _random_prefix_teacher(length, vocab)
    q_fixed = F.softmax(base, dim=-1)
    estimated = 0.0
    for path in product(range(vocab), repeat=length):
        weight = torch.stack([q_fixed[s, t] for s, t in enumerate(path)]).prod().item()
        student, teacher, labels = _single_draft_inputs(base, teacher_logits, [path])
        for bits in product((False, True), repeat=length):
            decisions = [(b,) for b in bits]
            w = weight * _decision_weight(q_fixed, teacher_logits, [path], decisions)
            uniforms = torch.tensor([[0.0 if b else 1.0] for b in bits]).unsqueeze(0)
            estimated += w * fused_linear_accepted_length(
                torch.eye(vocab), student, teacher, labels, length + 1,
                estimator="multidraft", num_drafts=1, uniforms=uniforms,
            ).item()
    exact = _exact_single_draft_ej(base, teacher_logits).item()
    assert abs(estimated + exact) < 2e-6


def _exact_expected_rounds(draft_logits, target_logits, window_len, depth, vocab):
    """Simulate stochastic speculative decoding exactly; count rounds starting < H."""

    def rounds_from(prefix):
        if len(prefix) >= window_len:
            return draft_logits[()].new_zeros(())
        q = F.softmax(draft_logits[tuple(prefix)], dim=-1)            # [depth, vocab]

        def branch(j, committed, prob):
            p = F.softmax(target_logits[tuple(committed)], dim=-1)
            if j == depth:                                              # bonus token
                return sum(prob * p[v] * rounds_from(committed + [v]) for v in range(vocab))
            out = sum(
                branch(j + 1, committed + [v], prob * q[j, v] * torch.clamp(p[v] / q[j, v],
                                                                           max=1.0))
                for v in range(vocab)
            )
            residual = torch.clamp(p - q[j], min=0.0)
            return out + sum(prob * residual[v] * rounds_from(committed + [v])
                             for v in range(vocab))

        return 1.0 + branch(0, list(prefix), 1.0)

    return rounds_from([])


def test_expected_rounds_on_teacher_rollouts_equals_exact_speculative_decoding():
    """Coupling: averaging the occupancy objective over x ~ p is EXACT, value and grad."""
    torch.manual_seed(53)
    window_len, depth, vocab = 3, 2, 2
    block_size = depth + 1
    horizon = window_len + depth
    target_logits = {}
    for t in range(horizon + 2):
        for prefix in product(range(vocab), repeat=t):
            target_logits[prefix] = torch.randn(vocab) * 1.2
    base = {prefix: torch.randn(depth, vocab) * 1.2
            for t in range(window_len) for prefix in product(range(vocab), repeat=t)}

    exact_params = {k: v.clone().requires_grad_(True) for k, v in base.items()}
    exact = _exact_expected_rounds(exact_params, target_logits, window_len, depth, vocab)
    exact.backward()

    est_params = {k: v.clone().requires_grad_(True) for k, v in base.items()}
    estimated = 0.0
    for x in product(range(vocab), repeat=horizon):
        p_x = 1.0
        for t in range(horizon):
            p_x *= F.softmax(target_logits[x[:t]], dim=-1)[x[t]].item()
        # One window of anchors n = 0..H-1; block n drafts x[n .. n+depth-1].
        student = torch.cat([
            torch.cat([est_params[x[:n]], torch.zeros(1, vocab)], dim=0)
            for n in range(window_len)
        ], dim=0)
        teacher = torch.cat([
            torch.cat([torch.stack([target_logits[x[:n + j]] for j in range(depth)]),
                       torch.zeros(1, vocab)], dim=0)
            for n in range(window_len)
        ], dim=0)
        labels = torch.cat([
            torch.tensor([*x[n:n + depth], IGNORE_INDEX]) for n in range(window_len)
        ])
        estimated = estimated + p_x * window_len * fused_linear_expected_rounds(
            torch.eye(vocab), student, teacher, labels, block_size, window_len,
            chunk_size=2,
        )
    assert torch.allclose(estimated, exact, atol=1e-5), (estimated, exact)
    estimated.backward()
    for key in base:
        assert torch.allclose(est_params[key].grad, exact_params[key].grad, atol=1e-5), key


def test_expected_rounds_reduces_to_one_round_per_block_when_nothing_is_accepted():
    vocab, block_size, window_len = 3, 4, 5
    num_blocks = window_len
    weight = torch.eye(vocab)
    teacher = torch.zeros(num_blocks * block_size, vocab)
    teacher[:, 0] = 30.0                                   # p puts all mass on token 0
    student = torch.zeros(num_blocks * block_size, vocab, requires_grad=True)
    with torch.no_grad():
        student[:, 1] = 30.0                               # Q puts ~no mass on token 0
    labels = torch.zeros(num_blocks * block_size, dtype=torch.long)
    labels.reshape(num_blocks, block_size)[:, -1] = IGNORE_INDEX
    loss = fused_linear_expected_rounds(weight, student, teacher, labels,
                                        block_size, window_len)
    # Every round commits exactly one token: rounds/token = 1.
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


def test_window_anchors_are_contiguous_inside_assistant_runs():
    from src.utils.anchors import sample_window_anchors

    seq_len, block_size, window_len = 64, 4, 5
    input_ids = torch.zeros(2, seq_len, dtype=torch.long)
    mask = torch.zeros(2, seq_len, dtype=torch.bool)
    mask[0, 10:30] = True
    mask[0, 40:58] = True
    mask[1, 5:8] = True                                    # too short for any window
    anchors, valid = sample_window_anchors(
        input_ids, block_size, num_windows=3, window_len=window_len,
        generator=torch.Generator().manual_seed(0), assistant_mask=mask,
    )
    assert anchors.shape == valid.shape == (2, 3 * window_len)
    assert not valid[1].any()
    windows = anchors[0].reshape(3, window_len)
    window_valid = valid[0].reshape(3, window_len)
    assert window_valid.all()
    for window in windows:
        assert torch.equal(window, window[0] + torch.arange(window_len))
        assert mask[0, window].all()
        assert int(window[-1]) <= seq_len - block_size
    starts = windows[:, 0]
    assert torch.all(starts[1:] - starts[:-1] >= window_len)


def test_onpolicy_dual_pass_multiple_drafts_shapes():
    torch.manual_seed(59)
    batch, num_anchors, block_size, num_drafts = 2, 3, 4, 3
    seq_len, hidden_size, vocab = 32, 5, 11
    input_ids = torch.randint(0, vocab, (batch, seq_len))
    anchors = torch.tensor([[1, 8, 15], [2, 9, 16]])

    class FakeModel:
        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, **kwargs):
            self.calls.append(kwargs)
            hidden = torch.randn(
                *input_ids.shape, hidden_size,
            ).requires_grad_(kwargs.get("is_diffusion_pass", False))
            cache = object() if kwargs.get("use_cache", False) else None
            return SimpleNamespace(hidden_states=(hidden,), past_key_values=cache)

    model = FakeModel()
    student, teacher, keep, labels = dual_pass_forward(
        model, input_ids=input_ids, anchors=anchors, block_size=block_size,
        mask_token_id=MASK_ID, on_policy=True, lm_head_weight=torch.randn(vocab, hidden_size),
        sampling_generator=torch.Generator().manual_seed(61), num_drafts=num_drafts,
    )
    num_positions = batch * num_anchors * block_size
    assert len(model.calls) == 2 + num_drafts
    assert all(call["is_verification_pass"] for call in model.calls[2:])
    assert student.shape == (num_positions, hidden_size)
    assert teacher.shape == (num_drafts * num_positions, hidden_size)
    assert labels.shape == (num_drafts * num_positions,)
    assert keep.shape == (num_positions,)

    loss = fused_linear_accepted_length(
        torch.randn(vocab, hidden_size), student, teacher, labels, block_size,
        chunk_size=5, estimator="rb", num_drafts=num_drafts, baseline="loo",
    )
    loss.backward()
    assert torch.isfinite(loss)
