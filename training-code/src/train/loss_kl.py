"""Forward-KL distillation objective (paper Eq. 7).

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
