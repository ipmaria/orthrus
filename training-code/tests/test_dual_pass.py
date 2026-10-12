"""Unit tests for the Qwen3 training path (block construction, mask rule, Eq. 7 loss).

These deliberately avoid importing `src.model`, so they run without CUDA,
FlexAttention or transformers -- only torch is needed:

    python -m pytest tests/test_dual_pass.py -q
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.train.dual_pass import build_diffusion_blocks
from src.train.loss import IGNORE_INDEX, block_local_labels
from src.train.loss_kl import fused_linear_cross_entropy, fused_linear_forward_kl

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
