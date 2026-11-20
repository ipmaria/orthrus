"""Lossless tree verification of several parallel drafts (inference side of `multidraft`).

A one-step diffusion drafter proposes the whole block from one forward pass and its
positions are factorized, so N drafts sampled from the same per-position distributions
Q_s are i.i.d. At every depth the chains that still agree with the committed prefix are
therefore i.i.d. candidates from Q_s, and SpecInfer-style recursive rejection keeps the
output distribution exactly equal to the target's:

    try alive chains in order: accept candidate v with prob min(1, r(v) / Q_s(v));
    on rejection r <- normalize([r - Q_s]_+); if every candidate is rejected, emit a
    correction token from the final residual. After acceptance only the chains carrying
    the accepted token stay alive; their next tokens are again i.i.d. from Q_{s+1}.

This is exactly the walk whose expected accepted length `loss_kl.multidraft` trains.
Torch-only (no transformers), so it is unit-tested directly.
"""

from __future__ import annotations

import torch

_TINY = 1e-12


def tree_verification_mask(
    num_chains: int,
    block_len: int,
    prefix_len: int,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Additive [1, 1, N*K, prefix_len + N*K] mask for N packed chains of length K.

    Every query sees the whole committed prefix (the shared KV cache) and, inside the
    packed region, only earlier-or-equal positions of its own chain.
    """
    total = num_chains * block_len
    chain = torch.arange(total, device=device) // block_len
    offset = torch.arange(total, device=device) % block_len
    allowed = (chain[:, None] == chain[None, :]) & (offset[None, :] <= offset[:, None])
    allowed = torch.cat(
        [torch.ones(total, prefix_len, dtype=torch.bool, device=device), allowed], dim=1
    )
    mask = torch.zeros(total, prefix_len + total, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None]


def _sample(probs: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return torch.multinomial(probs.float(), 1, generator=generator)


def tree_speculative_walk(
    chain_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[int, int, torch.Tensor]:
    """One verification round over N chains.

    Args:
        chain_tokens: [N, D] drafted tokens, chain n drawn i.i.d. from draft_probs.
        draft_probs: [D, V] proposal distributions Q_s (after any top-k/top-p).
        target_probs: [N, D+1, V]; target_probs[n, s] is the target distribution of the
            token at draft position s given the committed prefix followed by
            chain n's first s tokens; index D is the bonus distribution.
        generator: optional RNG for coins and the correction/bonus token.

    Returns:
        (chain, accepted, next_token): a chain carrying the accepted prefix, the number
        of accepted draft tokens, and the correction/bonus token as a [1] tensor.
    """
    num_chains, depth = chain_tokens.shape
    device = draft_probs.device
    alive = list(range(num_chains))
    for s in range(depth):
        base = target_probs[alive[0], s].float()
        residual = base
        q = draft_probs[s].float()
        accepted = None
        for chain in alive:
            token = int(chain_tokens[chain, s])
            ratio = min(1.0, float(residual[token]) / max(float(q[token]), _TINY))
            coin = float(torch.rand(1, generator=generator, device=device))
            if coin < ratio:
                accepted = token
                break
            residual = torch.clamp(residual - q, min=0.0)
            mass = float(residual.sum())
            # mass == 0 means residual == Q, so the next candidate is accepted surely;
            # the fallback only guards round-off.
            residual = residual / mass if mass > _TINY else base
        if accepted is None:
            return alive[0], s, _sample(residual, generator)
        alive = [chain for chain in alive if int(chain_tokens[chain, s]) == accepted]
    return alive[0], depth, _sample(target_probs[alive[0], depth], generator)
