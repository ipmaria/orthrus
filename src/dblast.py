"""DBLAST re-implementation (Karimi, Gao, Hassanpour 2026, arXiv:2608.05448) for Orthrus.

Dependent block drafting with a categorical latent z in {1..K} per block:

    q_dep(x_{1:b} | y) = sum_z q(z | y) prod_i q_i(x_i | y, z)

* Category hidden expander: h_{i,z} = h_i + g_z(h_i), g_z a linear layer (with bias);
  the (frozen, shared) LM head maps every branch to q_i(. | y, z).
* Category prior head: a linear layer on the block-anchor hidden state -> K logits.
  Added parameters: K H^2 + 2 K H + K, as in the paper's Table 1.
  Here h is the Orthrus diffusion-view hidden state; the anchor state is block offset 0.
* Training objective (AL, paper Eqs. 14-16) on target-sampled prefixes x ~ p:
      r_l = q(x_{1:l}) / p(x_{1:l}),   A_j = min(1, p(x_j | x_<j) / q(x_j | x_<j)),
      S_l = sum_{i<=l} prod_{j<=i} A_j,
      J = sum_{l in T_tau} (log r_l + log S_l),  T_tau = {1..l_tau},
      l_tau = min({l : r_l < tau} u {b}),  tau = 0.1,  loss = -J.
  Membership in T_tau is a stop-gradient selection (the paper notes the estimator is
  neither unbiased nor a bound). q uses the differentiable soft mixture; A_j uses the
  soft conditional q(x_j | x_<j) = q(x_{1:j}) / q(x_{1:j-1}) (the paper does not say
  which conditional enters A_j in training; the soft one is the only differentiable one).
  p is the target AFTER the same temperature / top-k / top-p filter used to generate the
  training trajectories.
* Inference (greedy-branch decoder, used for all of the paper's evaluations):
  pi = softmax(prior / Z_T) (Z_T = 0 -> highest-prior category); sample z ~ pi; the draft
  is branch z's per-position argmax. The proposal is the mixture of the K deterministic
  branch blocks, so the verification conditional is
      q(v | x_<i) = sum_{z in C_i} pi_z 1{v = xhat_{i,z}} / sum_{z in C_i} pi_z,
  C_i = branches whose greedy prefix equals the accepted prefix (paper Eqs. 19-20).
  Accept with min(1, p/q); on rejection resample from [p - q]_+; bonus from p.

Details the paper leaves open and the choice made here: expander init (small random,
scaled by 1/sqrt(H), so the branches start near the base drafter but are not identical
-- zero init would keep them identical forever); prior head zero-init (uniform prior);
anchor state = diffusion hidden at block offset 0. Torch-only; tested in
tests/test_dblast.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_TINY = 1e-30


def branch_hidden(hidden: torch.Tensor, expander_weight: torch.Tensor | None,
                  expander_bias: torch.Tensor | None, num_categories: int) -> torch.Tensor:
    """[..., H] -> [..., K, H] with h_z = h + g_z(h) (K = 1: the base drafter, no head)."""
    if num_categories == 1:
        return hidden.unsqueeze(-2)
    delta = F.linear(hidden, expander_weight, expander_bias)
    return hidden.unsqueeze(-2) + delta.unflatten(-1, (num_categories, hidden.shape[-1]))


def category_log_prior(anchor_hidden: torch.Tensor, prior_weight: torch.Tensor | None,
                       prior_bias: torch.Tensor | None, num_categories: int) -> torch.Tensor:
    """log q(z | y): [..., K]."""
    if num_categories == 1:
        return anchor_hidden.new_zeros(*anchor_hidden.shape[:-1], 1).float()
    return F.log_softmax(F.linear(anchor_hidden, prior_weight, prior_bias).float(), dim=-1)


def filtered_log_probs(logits: torch.Tensor, temperature: float, top_k: int = 0,
                       top_p: float = 1.0) -> torch.Tensor:
    """log p after temperature, top-k, top-p -- the same rule as `generate()`'s sample()."""
    logits = logits.float() / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits = logits.masked_fill(remove.scatter(-1, sorted_indices, remove), -float("inf"))
    return F.log_softmax(logits, dim=-1)


def al_objective_terms(log_q_branch: torch.Tensor, log_prior: torch.Tensor,
                       log_p: torch.Tensor, prefix_mask: torch.Tensor, tau: float):
    """Per-block DBLAST quantities on the committed path.

    log_q_branch [B, D, K] = log q_i(x_i | z); log_prior [B, K]; log_p [B, D] (filtered
    target, -inf allowed only outside prefix_mask); prefix_mask [B, D] bool.
    Returns (J [B] with gradient, log q(x_{1:l}) [B, D], log r [B, D], A [B, D]).
    """
    mask = prefix_mask.to(log_q_branch.dtype)
    cumulative = log_prior.unsqueeze(1) + torch.cumsum(log_q_branch * mask.unsqueeze(-1), 1)
    log_q_prefix = torch.logsumexp(cumulative, dim=-1)                    # log q(x_{1:l})
    previous = torch.cat([log_q_prefix.new_zeros(log_q_prefix.shape[0], 1),
                          log_q_prefix[:, :-1]], dim=1)
    log_q_conditional = log_q_prefix - previous                          # log q(x_l | x_<l)
    safe_log_p = torch.where(prefix_mask, log_p, torch.zeros_like(log_p))
    log_r = log_q_prefix - torch.cumsum(safe_log_p, dim=1)
    accept = torch.exp((safe_log_p - log_q_conditional).clamp_max(0.0))
    survival = torch.cumprod(torch.where(prefix_mask, accept, torch.ones_like(accept)), 1)
    expected = torch.cumsum(survival * mask, dim=1)                       # S_l
    with torch.no_grad():
        below = (log_r < torch.log(torch.tensor(tau, device=log_r.device))) & prefix_mask
        depth = prefix_mask.shape[1]
        first = torch.where(below.any(1), below.to(torch.int64).argmax(1),
                            torch.full_like(below[:, 0], depth - 1, dtype=torch.int64))
        last_valid = prefix_mask.to(torch.int64).sum(1) - 1
        stop = torch.minimum(first, last_valid)
        selected = (torch.arange(depth, device=log_r.device).unsqueeze(0)
                    <= stop.unsqueeze(1)) & prefix_mask
    objective = ((log_r + torch.log(expected.clamp_min(_TINY))) * selected).sum(1)
    return objective, log_q_prefix, log_r, accept


def greedy_branch_step(branch_tokens: torch.Tensor, category_probs: torch.Tensor,
                       target_probs: torch.Tensor, chosen: int,
                       generator: torch.Generator | None = None) -> tuple[int, torch.Tensor]:
    """One verification round of the greedy-branch decoder (paper App. B).

    branch_tokens [K, D] (per-branch argmax drafts); category_probs [K] (pi); the draft is
    branch `chosen`; target_probs [D+1, V] along that draft (row D = bonus).
    Returns (accepted, next_token [1]).
    """
    num_categories, depth = branch_tokens.shape
    vocab = target_probs.shape[-1]
    device = target_probs.device
    draft = branch_tokens[chosen]
    consistent = (category_probs > 0)
    for i in range(depth):
        weights = category_probs * consistent
        q = torch.zeros(vocab, device=device, dtype=torch.float32).scatter_add(
            0, branch_tokens[:, i], weights.float())
        q = q / q.sum()
        token = int(draft[i])
        p = target_probs[i].float()
        ratio = min(1.0, float(p[token]) / max(float(q[token]), _TINY))
        if float(torch.rand(1, generator=generator, device=device)) < ratio:
            consistent = consistent & (branch_tokens[:, i] == token)
            continue
        residual = torch.clamp(p - q, min=0.0)
        mass = float(residual.sum())
        residual = residual / mass if mass > 1e-12 else p
        return i, torch.multinomial(residual, 1, generator=generator)
    return depth, torch.multinomial(target_probs[depth].float(), 1, generator=generator)
