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
import torch.nn.functional as F

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


@torch.no_grad()
def sample_draft_tokens(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    block_size: int,
    chunk_size: int = 1024,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample y_s ~ Q_s independently at the K-1 deployed proposal offsets.

    Returns [num_blocks, K-1]. The full vocabulary is streamed in chunks and no
    sampling graph is retained; differentiable log Q_s is recomputed in the loss.
    """
    if block_size < 2 or chunk_size <= 0:
        raise ValueError("block_size must be >= 2 and chunk_size must be positive.")
    if not 0.0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive.")
    if student_hidden.shape[0] % block_size != 0:
        raise ValueError("student_hidden must contain whole blocks.")

    num_blocks = student_hidden.shape[0] // block_size
    proposal_hidden = student_hidden.reshape(
        num_blocks, block_size, student_hidden.shape[-1]
    )[:, :-1].reshape(-1, student_hidden.shape[-1])
    sampled = torch.empty(
        proposal_hidden.shape[0], dtype=torch.long, device=proposal_hidden.device
    )
    for start in range(0, proposal_hidden.shape[0], chunk_size):
        stop = min(start + chunk_size, proposal_hidden.shape[0])
        logits = F.linear(
            proposal_hidden[start:stop], lm_head_weight
        ).float() / temperature
        probabilities = F.softmax(logits, dim=-1)
        sampled[start:stop] = torch.multinomial(
            probabilities, 1, generator=generator
        ).squeeze(-1)
    return sampled.reshape(num_blocks, block_size - 1)


def dual_pass_forward(
    model,
    input_ids: torch.Tensor,
    anchors: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    anchor_valid: torch.Tensor | None = None,
    supervise_mask: torch.Tensor | None = None,
    on_policy: bool = False,
    lm_head_weight: torch.Tensor | None = None,
    sampling_temperature: float = 1.0,
    sampling_generator: torch.Generator | None = None,
    sampling_chunk_size: int = 1024,
    num_drafts: int = 1,
    causal_drafting: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the frozen AR pass and the trainable diffusion pass over one micro-batch.

    Returns `(student_hidden, teacher_hidden, keep_mask, labels)`, where the hidden
    states are [N, hidden] aligned position-for-position and N = batch * B * K.

    Alignment: diffusion query at block offset j predicts the token after absolute
    position a_b + j, so its teacher is the AR row at a_b + j. Offset K-1 is never
    supervised -- its target lies outside the block, and `generate()` discards that
    row too (`src/model.py:477`, `logits[:, :-1]`). That is the `block_size - 1`
    supervised positions the official trainer already logs.

    With on_policy=True, proposal tokens are first sampled from the current
    diffusion view. A second frozen AR verification pass follows each sampled
    block causally, and the returned labels/teacher rows describe that draft path
    rather than the committed target path.

    With num_drafts=M > 1 (on-policy only), M independent drafts are sampled from the
    same proposal and each gets its own verification pass. student_hidden stays
    [N, hidden] (the proposal does not depend on the draft); teacher_hidden and labels
    become draft-major [M * N, ...]. keep_mask is the [N] position mask (identical for
    every draft).
    """
    if num_drafts < 1:
        raise ValueError("num_drafts must be >= 1.")
    if num_drafts > 1 and not on_policy:
        raise ValueError("num_drafts > 1 requires on_policy=True.")
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
        # Control arm for noise conditioning: same causal in-block mask, no noise.
        causal_within_block=causal_drafting,
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

    if on_policy:
        if lm_head_weight is None:
            raise ValueError("lm_head_weight is required for on-policy EAL.")
        # The target p_s must follow y_<s, not the committed dataset tokens.
        target_prefix = (labels[:, :, :-1] != IGNORE_INDEX).cumprod(dim=-1).bool()
        draft_teachers, draft_labels = [], []
        for _ in range(num_drafts):
            sampled_tokens = sample_draft_tokens(
                lm_head_weight,
                student.reshape(-1, hidden_size),
                block_size,
                chunk_size=sampling_chunk_size,
                temperature=sampling_temperature,
                generator=sampling_generator,
            ).reshape(batch_size, num_anchors, block_size - 1)
            draft_blocks = block_ids.reshape(batch_size, num_anchors, block_size).clone()
            draft_blocks[:, :, 1:] = sampled_tokens

            # This pass uses AR projections and a causal within-block mask; gradients
            # remain stopped and use_cache=False keeps the shared clean cache intact.
            with torch.no_grad():
                verification_outputs = model(
                    input_ids=draft_blocks.reshape(batch_size, num_anchors * block_size),
                    position_ids=block_positions,
                    past_key_values=shared_cache,
                    use_cache=False,
                    is_verification_pass=True,
                    causal_limit=causal_limit,
                    ar_seq_len=seq_len,
                    logits_to_keep=1,
                )
            draft_teachers.append(verification_outputs.hidden_states[0].detach().reshape(
                batch_size, num_anchors, block_size, hidden_size
            ))

            sampled_labels = torch.full_like(labels, IGNORE_INDEX)
            sampled_labels[:, :, :-1] = draft_blocks[:, :, 1:]
            sampled_labels[:, :, :-1] = sampled_labels[:, :, :-1].masked_fill(
                ~target_prefix, IGNORE_INDEX
            )
            draft_labels.append(sampled_labels)

        keep_mask = draft_labels[0] != IGNORE_INDEX
        return (
            student.reshape(-1, hidden_size),
            torch.stack(draft_teachers).reshape(-1, hidden_size),
            keep_mask.reshape(-1),
            torch.stack(draft_labels).reshape(-1),
        )

    return (
        student.reshape(-1, hidden_size),
        teacher.reshape(-1, hidden_size),
        keep_mask.reshape(-1),
        labels.reshape(-1),
    )


def noise_conditioned_dual_pass(
    model,
    input_ids: torch.Tensor,
    anchors: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    lm_head_weight: torch.Tensor,
    embed_weight: torch.Tensor,
    top_m: int,
    anchor_valid: torch.Tensor | None = None,
    supervise_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dual pass for noise-conditioned drafting (src/noise_coupling.py).

    1. Frozen AR pass (shared cache + teacher rows), as in `dual_pass_forward`.
    2. Noise-free diffusion pass under no_grad: its top-`top_m` tokens per position are
       the reveal support S, exactly as `generate()` chooses it.
    3. For every supervised position, G_S is drawn from the posterior given the committed
       token x (prior noise where unsupervised). On teacher rollouts this is the joint law
       of real decoding (coupling), so no drafts need to be sampled.
    4. Trainable diffusion pass with the revealed-noise features, CAUSAL within the
       block (a draft position may only see noise of positions <= its own).

    Returns `(student_hidden [N, H], log_target [N], keep_mask [N], labels [N])`, where
    log_target = log p(x | G_S), the law the verifier will check x against.
    """
    if top_m < 1:
        raise ValueError("top_m must be >= 1.")
    batch_size, seq_len = input_ids.shape
    num_anchors = anchors.shape[1]
    block_ids, block_positions, causal_limit = build_diffusion_blocks(
        input_ids, anchors, block_size, mask_token_id
    )
    with torch.no_grad():
        ar_outputs = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        ar_hidden = ar_outputs.hidden_states[0].detach()
        shared_cache = ar_outputs.past_key_values
        diff_kwargs = dict(
            input_ids=block_ids, position_ids=block_positions,
            past_key_values=shared_cache, use_cache=False, is_diffusion_pass=True,
            causal_limit=causal_limit, ar_seq_len=seq_len, logits_to_keep=1,
        )
        proposal_hidden = model(**diff_kwargs).hidden_states[0].detach()

    hidden_size = ar_hidden.shape[-1]
    offsets = torch.arange(block_size, device=input_ids.device)
    positions = anchors.unsqueeze(-1) + offsets.reshape(1, 1, -1)
    gather_index = positions.reshape(batch_size, -1, 1).expand(-1, -1, hidden_size)
    teacher = torch.gather(ar_hidden, 1, gather_index).reshape(-1, hidden_size)
    proposal = proposal_hidden.reshape(-1, hidden_size)
    labels = block_local_labels(
        input_ids, anchors, block_size,
        supervise_mask=supervise_mask, anchor_valid=anchor_valid,
    ).reshape(-1)
    valid = labels != IGNORE_INDEX
    safe_labels = labels.clamp_min(0)

    from src.noise_coupling import (
        noise_features, posterior_gumbel_on_support, revealed_log_law_at, sample_gumbel,
    )

    num_positions = labels.shape[0]
    log_target = torch.zeros(num_positions, dtype=torch.float32, device=input_ids.device)
    features = torch.zeros(num_positions, 2 * hidden_size, dtype=embed_weight.dtype,
                           device=input_ids.device)
    with torch.no_grad():
        for start in range(0, num_positions, chunk_size):
            stop = min(start + chunk_size, num_positions)
            support = (F.linear(proposal[start:stop], lm_head_weight).float()
                       / temperature).topk(top_m, dim=-1).indices
            teacher_logprob = F.log_softmax(
                F.linear(teacher[start:stop], lm_head_weight).float() / temperature, dim=-1
            )
            log_p_support = teacher_logprob.gather(-1, support)
            label = safe_labels[start:stop]
            log_p_label = teacher_logprob.gather(-1, label.unsqueeze(-1)).squeeze(-1)
            in_support = support == label.unsqueeze(-1)
            chunk_valid = valid[start:stop]
            posterior = posterior_gumbel_on_support(log_p_support, in_support, generator)
            prior = sample_gumbel(support.shape, generator=generator, device=support.device)
            gumbel = torch.where(chunk_valid.unsqueeze(-1), posterior, prior)
            log_target[start:stop] = torch.where(
                chunk_valid,
                revealed_log_law_at(log_p_support, gumbel, log_p_label, in_support),
                torch.zeros_like(log_p_label),
            )
            features[start:stop] = noise_features(embed_weight, support, gumbel).to(
                features.dtype)

    # The last block offset predicts nothing at inference: reveal nothing there.
    features = features.reshape(batch_size, num_anchors, block_size, -1)
    features[:, :, -1] = 0
    # Causal within the block: position i may only see the noise of positions <= i
    # (non-causal reveal breaks losslessness; see src/noise_coupling.py).
    student = model(
        **diff_kwargs, causal_within_block=True,
        noise_features=features.reshape(batch_size, num_anchors * block_size, -1),
    ).hidden_states[0].reshape(-1, hidden_size)
    return student, log_target, valid, labels
