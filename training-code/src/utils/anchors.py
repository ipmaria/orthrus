"""Anchor sampling, vendored so the Qwen3 stack does not depend on `fla`.

These three functions are copied verbatim from the official
`src/utils/init_model.py`. They are pure torch and have nothing to do with the Qwen3.5
backbone -- but that module also imports `src.models.modeling_orthrus_qwen3_5`, which
hard-imports `fla.modules.conv`. A Qwen3-only install should not need
flash-linear-attention just to sample block anchors.

`train_qwen3.py` prefers the official implementation when it is importable and only
falls back here, so the two cannot silently diverge in a full install.
"""

from __future__ import annotations

import torch


def assistant_anchor_candidates(assistant_mask, block_size, seq_len, min_anchor=1):
    """Anchor positions that start inside an assistant run
    """
    mask = assistant_mask.to(torch.int8).flatten()
    if mask.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=assistant_mask.device)
    # Run boundaries via a first difference on a zero-padded copy.
    zero = torch.zeros(1, dtype=torch.int8, device=mask.device)
    boundary = torch.diff(torch.cat([zero, mask, zero]).to(torch.int16))
    run_starts = (boundary == 1).nonzero(as_tuple=True)[0]
    run_ends = (boundary == -1).nonzero(as_tuple=True)[0]  # exclusive
    if run_starts.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=mask.device)
    candidates = torch.cat([
        torch.arange(int(run_start), int(run_end), device=mask.device)
        for run_start, run_end in zip(run_starts.tolist(), run_ends.tolist())
    ])
    fits_in_row = candidates <= seq_len - block_size
    return candidates[(candidates >= min_anchor) & fits_in_row]


def sample_anchors(
    seq_len: int,
    block_size: int,
    num_anchors: int,
    generator=None,
    device=None,
    assistant_mask=None,
):
    """Sorted block starts for ONE sequence, `1 <= anchor <= seq_len - block_size`.
    """
    if seq_len - block_size < 1:
        raise ValueError(f"seq_len {seq_len} too short for block_size {block_size}.")

    if assistant_mask is not None:
        candidates = assistant_anchor_candidates(assistant_mask[:seq_len], block_size, seq_len)
        if candidates.numel() == 0:
            return candidates
        num_to_draw = min(num_anchors, candidates.numel())
        picked = torch.randperm(
            candidates.numel(), generator=generator, device=candidates.device
        )[:num_to_draw]
        return torch.sort(candidates[picked]).values

    num_candidates = seq_len - block_size  # positions [1 .. seq_len - block_size]
    num_to_draw = min(num_anchors, num_candidates)
    picked = torch.randperm(num_candidates, generator=generator, device=device)[:num_to_draw]
    return torch.sort(picked + 1).values


def sample_batch_anchors(
    input_ids,
    block_size: int,
    num_anchors: int,
    generator=None,
    assistant_mask=None,
):
    batch_size, seq_len = input_ids.shape
    num_slots = min(num_anchors, seq_len - block_size)
    device = input_ids.device
    anchors = torch.zeros((batch_size, num_slots), dtype=torch.long, device=device)
    anchor_valid = torch.zeros((batch_size, num_slots), dtype=torch.bool, device=device)

    for batch in range(batch_size):
        row_mask = None if assistant_mask is None else assistant_mask[batch]
        row_anchors = sample_anchors(
            seq_len, block_size, num_slots,
            generator=generator, device=device, assistant_mask=row_mask,
        )
        if row_anchors.numel() == 0:
            # No assistant text in this row:
            row_anchors = sample_anchors(
                seq_len, block_size, num_slots, generator=generator, device=device
            )
            anchors[batch, : row_anchors.numel()] = row_anchors
            continue

        num_found = row_anchors.numel()
        anchors[batch, :num_found] = row_anchors
        anchor_valid[batch, :num_found] = True
        if num_found < num_slots:
            anchors[batch, num_found:] = row_anchors[-1]
    return anchors, anchor_valid
