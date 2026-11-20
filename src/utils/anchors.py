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


def sample_window_anchors(
    input_ids,
    block_size: int,
    num_windows: int,
    window_len: int,
    generator=None,
    assistant_mask=None,
):
    """Contiguous anchor windows for the multi-round (expected-rounds) objective.

    Each window is `window_len` consecutive anchors s, s+1, ..., s+H-1 that all lie in
    ONE assistant run (or anywhere in the row without a mask) and leave room for a
    full block. Windows within a row do not overlap and are sorted, so the flattened
    anchors stay ascending. Returns `(anchors, anchor_valid)`, each
    [batch, num_windows * window_len]; a row with fewer feasible windows repeats its
    first window (or position 1) with anchor_valid=False.
    """
    if num_windows <= 0 or window_len <= 0:
        raise ValueError("num_windows and window_len must be positive.")
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    if seq_len - block_size < window_len:
        raise ValueError(
            f"seq_len {seq_len} too short for a window of {window_len} anchors with "
            f"block_size {block_size}."
        )
    num_slots = num_windows * window_len
    anchors = torch.ones((batch_size, num_slots), dtype=torch.long, device=device)
    anchor_valid = torch.zeros((batch_size, num_slots), dtype=torch.bool, device=device)
    offsets = torch.arange(window_len, device=device)

    for batch in range(batch_size):
        eligible = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if assistant_mask is None:
            eligible[1 : seq_len - block_size + 1] = True
        else:
            candidates = assistant_anchor_candidates(
                assistant_mask[batch][:seq_len], block_size, seq_len
            )
            eligible[candidates] = True
        # A start s is feasible iff s .. s+H-1 are all eligible. Consecutive assistant
        # candidates come from one run, since a run boundary is a non-assistant token.
        counts = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                       eligible.to(torch.long)]),
            dim=0,
        )
        starts = torch.arange(seq_len - window_len + 1, device=device)
        feasible = starts[(counts[starts + window_len] - counts[starts]) == window_len]
        if feasible.numel() == 0:
            continue

        order = torch.randperm(feasible.numel(), generator=generator, device=device)
        chosen = []
        for index in order.tolist():
            start = int(feasible[index])
            if all(abs(start - other) >= window_len for other in chosen):
                chosen.append(start)
                if len(chosen) == num_windows:
                    break
        chosen.sort()
        row = torch.cat([start + offsets for start in chosen])
        anchors[batch, : row.numel()] = row
        anchor_valid[batch, : row.numel()] = True
        if row.numel() < num_slots:
            filler = (chosen[0] + offsets).repeat(num_windows - len(chosen))
            anchors[batch, row.numel():] = filler
    return anchors, anchor_valid
