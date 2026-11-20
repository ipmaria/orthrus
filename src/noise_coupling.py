"""Noise-conditioned parallel drafting: share the verifier's Gumbel noise with the drafter.

The target samples x_t = argmax_v (log p_t(v) + G_tv) with G i.i.d. standard Gumbel --
an exact sample of p_t. With G fixed, the target's continuation is a deterministic
function of the context and the noise, so a factorized (one-step parallel) drafter that
SEES the noise is not limited by the factorization ceiling of an ordinary parallel drafter.

Partial reveal. The drafter is shown G only on a support S_t of m tokens, chosen without
looking at G (here: the top-m of the drafter's own noise-free proposal). Conditioned on
G_S the target's law is closed form. With s* = max_{v in S} log p_v + G_v (at v*) and
p_out = p(S^c), the maximum over S^c is Gumbel(log p_out) and independent of which
S^c token attains it, so

    p(. | G_S) = w * delta_{v*} + (1 - w) * p restricted to S^c / p_out,
    w = P(max over S^c < s*) = exp(-p_out * exp(-s*)).

Averaging over G_S gives back p_t, so ordinary speculative sampling AGAINST p(. | G_S)
(accept with min(1, law/q), resample from the residual) is lossless for p. S = {} is
ordinary speculative decoding; S = V is deterministic exact-match decoding.

Causality (required). The draft at position i may depend on the revealed noise of
positions <= i only. If it also saw G_j for j > i, reaching position j in a round would
carry information about G_j, x_j would no longer be distributed as p_j, and the scheme
would stop being lossless. A Monte Carlo check of a full round confirms both directions
(non-causal: TV 0.031 >> noise; causal: TV 0.006 ~ noise). The diffusion pass that
consumes the noise therefore runs with a causal in-block mask. Nothing is lost in the
ideal case, because x_i is a function of G_{<=i} alone.

Training data. If x ~ p (a teacher rollout), G drawn from the posterior G | argmax = x is
distributed exactly as the prior, so (x, G_S) has the joint law of real decoding and the
target-path survival prod min(1, Q(x | G_S) / law(x | G_S)) is an unbiased, score-free
estimate of the accepted length under noise-conditioned verification (coupling).

Torch-only (no transformers); verified in tests/test_noise_coupling.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_TINY = 1e-30
_LOG_FLOOR = -1e4      # finite stand-in for log 0 (top-k/top-p processed targets)


def sample_gumbel(shape, generator=None, device=None, dtype=torch.float32) -> torch.Tensor:
    """i.i.d. standard Gumbel noise."""
    u = torch.rand(shape, generator=generator, device=device, dtype=dtype)
    u = u.clamp(min=torch.finfo(dtype).tiny, max=1.0 - torch.finfo(dtype).eps)
    return -torch.log(-torch.log(u))


def posterior_gumbel_on_support(
    log_p_support: torch.Tensor,
    label_in_support: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """G_S | argmax_v(log p_v + G_v) = x, restricted to the support S.

    log_p_support: [..., m] log p_v for v in S (normalized log-probabilities).
    label_in_support: [..., m] bool, True at the support slot that equals x (if any).
    The perturbed maximum M = log p_x + G_x is Gumbel(logsumexp log p) = Gumbel(0); every
    other token's perturbed score is an independent Gumbel(log p_v) truncated below M,
    i.e. -log(e^{-M} + e^{-phi}) with phi ~ Gumbel(log p_v). Only S needs sampling.
    """
    log_p = log_p_support.float().clamp_min(_LOG_FLOOR)
    shape = log_p.shape
    maximum = sample_gumbel(shape[:-1], generator=generator, device=log_p.device)
    phi = log_p + sample_gumbel(shape, generator=generator, device=log_p.device)
    truncated = -torch.logaddexp(-maximum.unsqueeze(-1), -phi)
    scores = torch.where(label_in_support, maximum.unsqueeze(-1), truncated)
    return scores - log_p


def _reveal_statistics(log_p_support: torch.Tensor, gumbel_support: torch.Tensor):
    log_p = log_p_support.float().clamp_min(_LOG_FLOOR)
    scores = log_p + gumbel_support.float()
    best, best_slot = scores.max(dim=-1)
    p_out = (1.0 - log_p.exp().sum(dim=-1)).clamp_min(0.0)
    log_w = -(p_out * torch.exp(-best))                     # log P(max over S^c < s*)
    return best_slot, p_out, log_w


def revealed_log_law_at(
    log_p_support: torch.Tensor,
    gumbel_support: torch.Tensor,
    log_p_token: torch.Tensor,
    token_in_support: torch.Tensor,
) -> torch.Tensor:
    """log p(x | G_S) for one token x per row (training: x = the committed token).

    log_p_support / gumbel_support / token_in_support: [..., m]; log_p_token: [...].
    """
    best_slot, p_out, log_w = _reveal_statistics(log_p_support, gumbel_support)
    in_support = token_in_support.any(dim=-1)
    is_best = token_in_support.gather(-1, best_slot.unsqueeze(-1)).squeeze(-1)
    # log(1 - w) = log(-expm1(log_w)), stable for w close to 0 and to 1.
    log_one_minus_w = torch.log((-torch.expm1(log_w)).clamp_min(_TINY))
    log_outside = (log_one_minus_w + log_p_token.float().clamp_min(_LOG_FLOOR)
                   - torch.log(p_out.clamp_min(_TINY)))
    log_inside = torch.where(is_best, log_w, torch.full_like(log_w, -float("inf")))
    return torch.where(in_support, log_inside, log_outside)


def revealed_law(p: torch.Tensor, support: torch.Tensor, gumbel_support: torch.Tensor) -> torch.Tensor:
    """Full conditional law p(. | G_S): p [..., V], support [..., m] long, gumbel [..., m]."""
    p = p.float()
    if support.shape[-1] == 0:          # nothing revealed: the plain target
        return p / p.sum(dim=-1, keepdim=True)
    p_support = p.gather(-1, support)
    best_slot, p_out, log_w = _reveal_statistics(torch.log(p_support.clamp_min(_TINY)),
                                                 gumbel_support)
    w = log_w.exp()
    law = p.scatter(-1, support, 0.0)
    law = law * ((1.0 - w) / p_out.clamp_min(_TINY)).unsqueeze(-1)
    best_token = support.gather(-1, best_slot.unsqueeze(-1))
    law = law.scatter_add(-1, best_token, w.unsqueeze(-1))
    return law / law.sum(dim=-1, keepdim=True)


def noise_features(embed_weight: torch.Tensor, support: torch.Tensor,
                   gumbel_support: torch.Tensor) -> torch.Tensor:
    """Drafter input for the revealed noise: [..., 2 * hidden].

    Two embedding-weighted summaries of (S, G_S): softmax(G_S)-weighted token embeddings
    (which revealed token carries the largest noise) and standardized-G-weighted ones
    (the spread). The model maps them to the hidden size with a zero-initialized
    projection, so a checkpoint without it starts exactly where it was.
    """
    embeddings = F.embedding(support, embed_weight).float()          # [..., m, H]
    g = gumbel_support.float()
    soft = torch.softmax(g, dim=-1).unsqueeze(-1)
    standardized = ((g - g.mean(dim=-1, keepdim=True))
                    / (g.std(dim=-1, keepdim=True, correction=0) + 1e-6) / g.shape[-1]).unsqueeze(-1)
    return torch.cat([(soft * embeddings).sum(-2), (standardized * embeddings).sum(-2)], -1)


def noise_conditioned_speculative_step(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    support: torch.Tensor,
    gumbel_support: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """One verification round against the revealed laws (inference side).

    draft_tokens [D]; draft_probs [D, V]; target_probs [D+1, V] along the draft (row D
    is the bonus distribution, which has no revealed noise); support / gumbel [D, m].
    Returns (accepted, next_token [1]).
    """
    depth = draft_tokens.shape[0]
    laws = revealed_law(target_probs[:depth], support, gumbel_support)
    for i in range(depth):
        token = int(draft_tokens[i])
        q = draft_probs[i].float()
        ratio = min(1.0, float(laws[i, token]) / max(float(q[token]), _TINY))
        if float(torch.rand(1, device=q.device)) < ratio:
            continue
        residual = torch.clamp(laws[i] - q, min=0.0)
        mass = float(residual.sum())
        residual = residual / mass if mass > 1e-12 else laws[i]
        return i, torch.multinomial(residual, 1)
    return depth, torch.multinomial(target_probs[depth].float(), 1)
