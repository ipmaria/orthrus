"""Dual-pass block masking for the Qwen3 stack (paper 3.2).

`src/models/modeling_orthrus_qwen3_5.py` has `diffusion_train_forward` for the Qwen3.5
hybrid backbone. The Qwen3 model in `src/model.py` -- the one the paper actually
benchmarks and the one the released checkpoints use -- has the training branch wired
into `OrthrusModel.forward` (it builds the FlexAttention mask when
`self.training and is_diffusion_pass`) but nothing in the repo ever calls it.

This module drives that branch through the model's public signature, so `src/model.py`
stays untouched:

  1. One frozen AR pass over the clean packed row under `no_grad`, `use_cache=True`.
     This yields both (K_AR, V_AR) for every position AND the teacher hidden states.
  2. One diffusion pass over B corrupted blocks concatenated into a single sequence,
     reading that cache in place. Gradients flow only through the diffusion twins.

`logits_to_keep=1` on both passes is load-bearing: at 256 anchors x block 32 the full
logit tensor would be ~5 GB. The losses in `loss_kl.py` apply `lm_head` themselves,
chunked, on just the supervised positions.
"""

from __future__ import annotations

import torch

from src.train.loss import IGNORE_INDEX, block_local_labels


def build_diffusion_blocks(
    input_ids: torch.Tensor,
    anchors: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrupt B blocks per row (Eq. 5) and flatten them into one diffusion sequence.

    Returns `(block_input_ids, block_position_ids, causal_limit)`, each
    [batch, num_anchors * block_size].
    """
    batch_size, num_anchors = anchors.shape
    device = input_ids.device

    # Eq. 5: keep the anchor visible, mask the remaining K-1 positions.
    blocks = torch.full(
        (batch_size, num_anchors, block_size), mask_token_id, dtype=torch.long, device=device
    )
    blocks[:, :, 0] = torch.gather(input_ids, 1, anchors)

    # Block b occupies absolute positions a_b .. a_b + K - 1, so RoPE sees the true
    # sequence positions even though the blocks are packed contiguously.
    offsets = torch.arange(block_size, device=device)
    positions = anchors.unsqueeze(-1) + offsets.reshape(1, 1, -1)

    # Eq. 6, left term: a query in block b may attend to AR keys k <= a_b - 1, i.e. the
    # STRICT clean prefix before its own anchor. `generate_dual_pass_mask` compares with
    # `kv_idx <= causal_limit`, hence the -1 here. (The Qwen3.5 twin spells the same rule
    # as `kv_idx < anchors[...]`; the two conventions differ by exactly this offset.)
    causal_limit = (anchors - 1).unsqueeze(-1).expand(-1, -1, block_size)

    flat = (batch_size, num_anchors * block_size)
    return (
        blocks.reshape(flat),
        positions.reshape(flat),
        causal_limit.reshape(flat).contiguous(),
    )


def dual_pass_forward(
    model,
    input_ids: torch.Tensor,
    anchors: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    anchor_valid: torch.Tensor | None = None,
    supervise_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the frozen AR pass and the trainable diffusion pass over one micro-batch.

    Returns `(student_hidden, teacher_hidden, keep_mask, labels)`, where the hidden
    states are [N, hidden] aligned position-for-position and N = batch * B * K.

    Alignment: diffusion query at block offset j predicts the token after absolute
    position a_b + j, so its teacher is the AR row at a_b + j. Offset K-1 is never
    supervised -- its target lies outside the block, and `generate()` discards that
    row too (`src/model.py:477`, `logits[:, :-1]`). That is the `block_size - 1`
    supervised positions the official trainer already logs.
    """
    batch_size, seq_len = input_ids.shape
    num_anchors = anchors.shape[1]

    block_ids, block_positions, causal_limit = build_diffusion_blocks(
        input_ids, anchors, block_size, mask_token_id
    )

    # 1) Frozen AR pass: builds the shared cache and the teacher representations.
    with torch.no_grad():
        ar_outputs = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
    ar_hidden = ar_outputs.hidden_states[0].detach()  # [batch, seq_len, hidden]
    shared_cache = ar_outputs.past_key_values

    # 2) Diffusion pass over the same cache. use_cache=False so the corrupted blocks
    #    never contaminate (K_AR, V_AR) -- this is the "zero redundant cache" property.
    diff_outputs = model(
        input_ids=block_ids,
        position_ids=block_positions,
        past_key_values=shared_cache,
        use_cache=False,
        is_diffusion_pass=True,
        causal_limit=causal_limit,
        ar_seq_len=seq_len,
        logits_to_keep=1,
    )
    diff_hidden = diff_outputs.hidden_states[0]  # [batch, B * K, hidden]

    hidden_size = ar_hidden.shape[-1]
    student = diff_hidden.reshape(batch_size, num_anchors, block_size, hidden_size)

    # Teacher rows: AR hidden at absolute positions a_b + j, matching the student grid.
    offsets = torch.arange(block_size, device=input_ids.device)
    positions = anchors.unsqueeze(-1) + offsets.reshape(1, 1, -1)
    gather_index = positions.reshape(batch_size, -1, 1).expand(-1, -1, hidden_size)
    teacher = torch.gather(ar_hidden, 1, gather_index).reshape(
        batch_size, num_anchors, block_size, hidden_size
    )

    # Reuse the official label builder so the KL and CE arms supervise exactly the same
    # positions (it also masks non-assistant targets and invalid anchor slots).
    labels = block_local_labels(
        input_ids, anchors, block_size,
        supervise_mask=supervise_mask, anchor_valid=anchor_valid,
    )
    keep_mask = labels != IGNORE_INDEX

    return (
        student.reshape(-1, hidden_size),
        teacher.reshape(-1, hidden_size),
        keep_mask.reshape(-1),
        labels.reshape(-1),
    )
