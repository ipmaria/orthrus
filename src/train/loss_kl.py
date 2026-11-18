"""Chunked KL, cross-entropy, and accepted-length training objectives.

The released `src/train/loss.py` trains the diffusion view with cross-entropy against
hard dataset tokens -- the variant the paper ablates in Table 5 (TPF 5.86). The method
the paper actually reports (TPF 6.35) minimises the FORWARD KL against the frozen AR
head's full predictive distribution:

    L = E[ sum_b sum_k D_KL( p_AR(.|x_<=a_b+k-1) || p_diff(.|x_<a_b, y~_b) ) ]

Materialising both distributions is the whole difficulty: at the paper's settings
(256 anchors x block_size 32 -> 7,936 supervised positions, vocab 151,936) a single
fp32 logit tensor is ~4.8 GB, and forward KL needs four of them (teacher/student
logits and log-probs). Liger's fused linear CE cannot help -- it is hard-label only.

So `fused_linear_forward_kl` streams the positions in chunks and recomputes each
chunk's logits during backward (`checkpoint(..., use_reentrant=False)`). Peak activation
memory becomes O(chunk_size x vocab) instead of O(num_positions x vocab), which is what
makes the paper's objective trainable at 256 anchors on a single node.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_forward_kl(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> torch.Tensor:
    """Per-position D_KL(p_teacher || p_student) for one chunk. Returns [n]."""
    # The teacher is the frozen AR head: no graph, ever (paper 3.2, "the AR backbone
    # remains strictly frozen"; gradients flow only through the diffusion module).
    with torch.no_grad():
        teacher_logits = F.linear(teacher_hidden, lm_head_weight).float()
        teacher_logprob = F.log_softmax(teacher_logits, dim=-1)
        teacher_prob = teacher_logprob.exp()

    student_logits = F.linear(student_hidden, lm_head_weight).float()
    student_logprob = F.log_softmax(student_logits, dim=-1)

    return (teacher_prob * (teacher_logprob - student_logprob)).sum(dim=-1)


def fused_linear_forward_kl(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    keep_mask: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Mean forward KL over supervised positions (Eq. 7).

    Args:
        lm_head_weight: frozen `lm_head.weight`, [vocab, hidden].
        student_hidden: diffusion-view hidden states, [N, hidden], requires grad.
        teacher_hidden: frozen AR hidden states at the matching causal positions,
            [N, hidden]. Must already be detached.
        keep_mask: [N] bool; False at padding / non-assistant / invalid-anchor slots.
        chunk_size: positions per recomputed chunk. Peak activation memory is
            ~4 x chunk_size x vocab x 4 bytes.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"student/teacher hidden mismatch: {tuple(student_hidden.shape)} vs "
            f"{tuple(teacher_hidden.shape)}"
        )
    if teacher_hidden.requires_grad:
        raise ValueError("teacher_hidden must be detached; the AR view is frozen.")

    student_hidden = student_hidden[keep_mask]
    teacher_hidden = teacher_hidden[keep_mask]

    num_positions = student_hidden.shape[0]
    if num_positions == 0:
        # Keep the graph alive so DDP/FSDP still see a gradient for every rank.
        return student_hidden.sum() * 0.0

    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, num_positions, chunk_size):
        stop = min(start + chunk_size, num_positions)
        chunk = checkpoint(
            _chunk_forward_kl,
            lm_head_weight,
            student_hidden[start:stop],
            teacher_hidden[start:stop],
            use_reentrant=False,
        )
        total = total + chunk.sum()
    return total / num_positions


def fused_linear_cross_entropy(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Hard-label CE over the same positions -- the Table 5 ablation arm.

    Chunked the same way as the KL arm so the two objectives differ only in the
    target, not in numerics or memory behaviour.
    """
    keep_mask = labels != ignore_index
    student_hidden = student_hidden[keep_mask]
    labels = labels[keep_mask]

    num_positions = student_hidden.shape[0]
    if num_positions == 0:
        return student_hidden.sum() * 0.0

    def _chunk_ce(weight, hidden, target):
        logits = F.linear(hidden, weight).float()
        return F.cross_entropy(logits, target, reduction="sum")

    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, num_positions, chunk_size):
        stop = min(start + chunk_size, num_positions)
        total = total + checkpoint(
            _chunk_ce,
            lm_head_weight,
            student_hidden[start:stop],
            labels[start:stop],
            use_reentrant=False,
        )
    return total / num_positions


def _chunk_accepted_length(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    prefix_mask: torch.Tensor,
    temperature: float,
    on_policy: bool,
) -> torch.Tensor:
    """Negative accepted length, summed over whole blocks.

    All tensors are block-major: hidden states are [blocks, K-1, hidden], labels
    and prefix_mask are [blocks, K-1]. The straight-through term preserves -J
    while installing the requested target-path or full on-policy derivative.
    """
    gather_index = labels.unsqueeze(-1)
    with torch.no_grad():
        teacher_logits = F.linear(teacher_hidden, lm_head_weight).float() / temperature
        teacher_logprob = F.log_softmax(teacher_logits, dim=-1)
        log_p = teacher_logprob.gather(-1, gather_index).squeeze(-1)

    student_logits = F.linear(student_hidden, lm_head_weight).float() / temperature
    student_logprob = F.log_softmax(student_logits, dim=-1)
    log_q = student_logprob.gather(-1, gather_index).squeeze(-1)

    # Compute the deployed acceptance ratio in log space. Neither the teacher
    # nor the survival estimate carries autograd; the exact derivative is
    # restored explicitly below.
    detached_log_q = log_q.detach()
    if on_policy:
        # y_s ~ Q_s: acceptance is min(1, p_s(y_s) / Q_s(y_s)). The
        # coefficient below is the combined pathwise + score-function gradient.
        log_ratio = log_p - detached_log_q
        active = detached_log_q <= log_p
    else:
        # x_s ~ p_s: target-path acceptance estimate is min(1, Q_s/p_s),
        # and only the pathwise derivative is needed because p is fixed.
        log_ratio = detached_log_q - log_p
        active = detached_log_q < log_p
    accept_prob = torch.exp(log_ratio.clamp_max(0.0))
    survival = torch.cumprod(accept_prob, dim=-1) * prefix_mask

    # Target path: dJ/dlogQ_s = 1[Q_s < p_s] sum_{t >= s} W_t.
    # On policy: the same-looking coefficient with p/Q survival and Q_s <= p_s
    # includes both dJ and J*dlogQ(path); the score-function term is not omitted.
    suffix_survival = torch.cumsum(survival.flip(-1), dim=-1).flip(-1)
    coefficient = active.to(survival.dtype) * suffix_survival
    coefficient = coefficient * prefix_mask

    accepted_length = survival.sum()
    straight_through = (coefficient * (log_q - log_q.detach())).sum()
    return -(accepted_length + straight_through)


def fused_linear_accepted_length(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    block_size: int,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    temperature: float = 1.0,
    on_policy: bool = False,
) -> torch.Tensor:
    """Mean negative accepted length over valid blocks (second-stage objective).

    With on_policy=True, labels are sampled draft tokens y_s ~ Q_s and teacher
    hidden states follow the sampled draft prefix. The estimator is

        f_t = prod_{s <= t} min(1, p_s(y_s) / Q_s(y_s)),
        J   = sum_{t <= T} f_t.

    Its coefficient 1[Q_s <= p_s] * sum_{t >= s} f_t includes BOTH the pathwise
    derivative and J times the draft path's score-function derivative.

    With on_policy=False, labels are committed target tokens x_s ~ p_s and the
    original target-path estimator is used:

        W_t = prod_{s <= t} min(1, Q_s(x_s) / p_s(x_s)),
        J   = sum_{t <= T} W_t.

    In both cases the returned scalar is -mean(J); only the exact gradient
    differs. p_s comes from the frozen AR teacher and Q_s from the parallel
    diffusion proposal.

    Args:
        lm_head_weight: frozen `lm_head.weight`, [vocab, hidden].
        student_hidden: flattened diffusion hidden states, [B*K, hidden].
        teacher_hidden: matching detached AR hidden states, [B*K, hidden].
        labels: flattened block-local next-token labels, [B*K]. Offset K-1 is
            ignored, as in generation. An ignored target truncates its block:
            later positions cannot survive a missing earlier verification.
        block_size: K, including the unsupervised final offset.
        chunk_size: approximate supervised positions per recomputed chunk.
            Chunks never split a block, so one chunk contains at least K-1
            positions even if chunk_size is smaller.
        ignore_index: label sentinel for padding / invalid supervision.
        temperature: positive sampling temperature applied to both distributions.
        on_policy: use sampled-draft p/Q survival and the combined gradient;
            False keeps the committed-target Q/p pathwise estimator.

    Returns:
        Scalar -J averaged over blocks with at least one supervised position.

    This estimator does not model greedy argmax consensus decoding.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"student/teacher hidden mismatch: {tuple(student_hidden.shape)} vs "
            f"{tuple(teacher_hidden.shape)}"
        )
    if student_hidden.ndim != 2 or labels.ndim != 1:
        raise ValueError("student/teacher hidden must be [N, hidden], labels [N].")
    if student_hidden.shape[0] != labels.shape[0]:
        raise ValueError("hidden states and labels must contain the same positions.")
    if teacher_hidden.requires_grad:
        raise ValueError("teacher_hidden must be detached; the AR view is frozen.")
    if block_size < 2:
        raise ValueError("block_size must be at least 2.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0.0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive.")
    if labels.shape[0] % block_size != 0:
        raise ValueError("the flattened supervision grid must contain whole blocks.")

    num_blocks = labels.shape[0] // block_size
    hidden_size = student_hidden.shape[-1]
    student = student_hidden.reshape(num_blocks, block_size, hidden_size)[:, :-1]
    teacher = teacher_hidden.reshape(num_blocks, block_size, hidden_size)[:, :-1]
    block_labels = labels.reshape(num_blocks, block_size)[:, :-1]

    # First-rejection semantics require a contiguous supervised target prefix.
    # Do not skip a masked token and accidentally join two separate spans.
    valid = block_labels != ignore_index
    prefix_mask = valid.cumprod(dim=-1).bool()
    block_keep = prefix_mask.any(dim=-1)
    student = student[block_keep]
    teacher = teacher[block_keep]
    block_labels = block_labels[block_keep]
    prefix_mask = prefix_mask[block_keep]

    num_kept_blocks = student.shape[0]
    if num_kept_blocks == 0:
        # Keep the graph alive so DDP/FSDP still see a gradient for every rank.
        return student_hidden.sum() * 0.0

    # Ignored tail entries are never scored, but gather still needs a legal id.
    block_labels = block_labels.masked_fill(~prefix_mask, 0)
    positions_per_block = block_size - 1
    blocks_per_chunk = max(1, chunk_size // positions_per_block)

    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, num_kept_blocks, blocks_per_chunk):
        stop = min(start + blocks_per_chunk, num_kept_blocks)
        total = total + checkpoint(
            _chunk_accepted_length,
            lm_head_weight,
            student[start:stop],
            teacher[start:stop],
            block_labels[start:stop],
            prefix_mask[start:stop],
            temperature,
            on_policy,
            use_reentrant=False,
        )
    return total / num_kept_blocks
