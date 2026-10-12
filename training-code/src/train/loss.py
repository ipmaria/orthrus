from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def block_local_labels(
    input_ids,
    anchors,
    block_size: int,
    supervise_mask=None,
    anchor_valid=None,
):
    device = input_ids.device
    batch_size, num_anchors = anchors.shape
    seq_len = input_ids.shape[1]

    offsets = torch.arange(block_size, device=device)
    positions = anchors.reshape(batch_size, num_anchors, 1) + offsets.reshape(1, 1, -1)
    clean_tokens = torch.gather(
        input_ids.unsqueeze(1).expand(batch_size, num_anchors, seq_len), 2, positions
    )

    labels = torch.full(
        (batch_size, num_anchors, block_size), IGNORE_INDEX, dtype=torch.long, device=device
    )
    labels[:, :, :-1] = clean_tokens[:, :, 1:]           # position t predicts token t+1

    target_positions = positions[:, :, 1:]
    keep = None
    if supervise_mask is not None:
        # Masks per (block, position), which is precisely the positions past end-of-turn.
        keep = torch.gather(
            supervise_mask.unsqueeze(1).expand(batch_size, num_anchors, seq_len),
            2, target_positions,
        ).bool()
    if anchor_valid is not None:
        slot_valid = anchor_valid.reshape(batch_size, num_anchors, 1).expand_as(
            labels[:, :, :-1]
        )
        keep = slot_valid if keep is None else (keep & slot_valid)

    if keep is not None:
        keep[0, 0, 0] = True
        labels[:, :, :-1] = labels[:, :, :-1].masked_fill(~keep, IGNORE_INDEX)
    return labels


def build_loss_fn(use_liger: bool = True):
    fused_ce = None
    if use_liger:
        try:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

            fused_ce = LigerFusedLinearCrossEntropyLoss(
                ignore_index=IGNORE_INDEX, reduction="mean"
            )
        except Exception:
            fused_ce = None

    def loss_fn(lm_head_weight, hidden, labels):
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_labels = labels.reshape(-1)
        if fused_ce is not None:
            return fused_ce(lm_head_weight, flat_hidden, flat_labels)
        logits = F.linear(flat_hidden, lm_head_weight).float()
        return F.cross_entropy(logits, flat_labels, ignore_index=IGNORE_INDEX)

    loss_fn.is_fused = fused_ce is not None
    return loss_fn
