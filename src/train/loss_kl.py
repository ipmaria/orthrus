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




# ---------------------------------------------------------------------------------
# Accepted-length objectives (stage 2)
# ---------------------------------------------------------------------------------
#
# Notation, per block: D = K-1 deployed proposal offsets, Q_s the (parallel, hence
# factorized) diffusion proposal at offset s, p_s(.|y_<s) the frozen AR teacher along a
# drafted prefix. Single-draft stochastic verification accepts y_s with probability
# r_s = min(1, p_s(y_s)/Q_s(y_s)); the number of accepted draft tokens has expectation
# E_y[J(y)], J = sum_t f_t, f_t = prod_{s<=t} r_s.
#
# Estimators (all unbiased for their objective; verified against exact enumeration in
# tests/test_dual_pass.py):
#
#   target-path  committed tokens x ~ data, W_t = prod min(1, Q/p), pathwise gradient.
#                EXACT (not only a proxy) when x are rollouts of the teacher at the same
#                temperature: under lossless verification the output is ~p whatever the
#                drafter, and P(accept_t | output x) = min(1, Q_t(x_t)/p_t(x_t)). On
#                off-policy data (dataset text written by other models) it is biased.
#   onpolicy     sampled drafts y ~ Q, value J(y), pathwise + score-function gradient.
#   rb           Rao-Blackwellised on-policy: the last token of every survival term is
#                integrated out analytically, E_{y_t}[r_t | y_<t] = 1 - TV(p_t, Q_t), so
#                J_RB = sum_t f_{t-1} (1 - TV_t) has the same mean and lower variance.
#                Score coefficient of log Q_s(y_s): sum_{t>s} f_{t-1}(1 - TV_t).
#   multidraft   N i.i.d. drafts verified as a tree (SpecInfer-style recursive rejection
#                among the alive chains at every depth -- lossless because a parallel
#                drafter's siblings are i.i.d. draws from the same Q_s). The acceptance
#                probability at each reached node, 1 - prod_j TV(p^(j), Q_s), is analytic;
#                the walk (tokens and accept coins) is sampled and enters via the score.
#
# Baselines: for onpolicy/rb with N >= 2 drafts per block, "loo" subtracts the mean
# score-return of the OTHER drafts of the same block (independent of this draft, so
# unbiased). For multidraft it uses the other blocks of the chunk at the same depth.

ACCEPTED_LENGTH_ESTIMATORS = ("target-path", "onpolicy", "rb", "multidraft")
ACCEPTED_LENGTH_BASELINES = ("none", "loo")
_TINY = 1e-30


def _suffix_sum(values: torch.Tensor) -> torch.Tensor:
    """sum_{t >= s} values[..., t] for every s."""
    return torch.flip(torch.cumsum(torch.flip(values, dims=(-1,)), dim=-1), dims=(-1,))


def _chunk_accepted_length(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    prefix_mask: torch.Tensor,
    temperature: float,
    estimator: str,
    baseline: str,
) -> torch.Tensor:
    """Negative accepted length, summed over whole blocks (averaged over drafts).

    Block-major tensors: student_hidden [blocks, D, hidden] (one proposal per block),
    teacher_hidden [N, blocks, D, hidden], labels [N, blocks, D] (N drafts or N=1
    committed path), prefix_mask [blocks, D]. The straight-through term keeps the
    scalar equal to -J while installing the estimator's exact gradient.
    """
    num_drafts = teacher_hidden.shape[0]
    mask = prefix_mask.to(torch.float32)

    student_logits = F.linear(student_hidden, lm_head_weight).float() / temperature
    student_logprob = F.log_softmax(student_logits, dim=-1)
    student_prob = student_logprob.exp() if estimator == "rb" else None

    log_qs, values, score_returns, actives = [], [], [], []
    for draft in range(num_drafts):
        gather_index = labels[draft].unsqueeze(-1)
        with torch.no_grad():
            teacher_logits = (
                F.linear(teacher_hidden[draft], lm_head_weight).float() / temperature
            )
            teacher_logprob = F.log_softmax(teacher_logits, dim=-1)
            log_p = teacher_logprob.gather(-1, gather_index).squeeze(-1)
        log_q = student_logprob.gather(-1, gather_index).squeeze(-1)
        detached_log_q = log_q.detach()

        if estimator == "target-path":
            # x_s ~ p_s: survival prod min(1, Q/p); dJ/dlogQ_s = 1[Q_s < p_s] sum_{t>=s} W_t.
            accept_prob = torch.exp((detached_log_q - log_p).clamp_max(0.0))
            survival = torch.cumprod(accept_prob, dim=-1) * mask
            value = survival.sum()
            score_return = _suffix_sum(survival)
            active = detached_log_q < log_p
        else:
            # y_s ~ Q_s: r_s = min(1, p/Q). Pathwise d r_s = -r_s dlogQ_s where Q_s > p_s;
            # adding the score term leaves the coefficient 1[Q_s <= p_s] * return_s.
            accept_prob = torch.exp((log_p - detached_log_q).clamp_max(0.0))
            active = detached_log_q <= log_p
            if estimator == "onpolicy":
                survival = torch.cumprod(accept_prob, dim=-1) * mask
                value = survival.sum()
                score_return = _suffix_sum(survival)               # sum_{t >= s} f_t
            else:  # rb
                survival_before = torch.cumprod(
                    torch.cat([torch.ones_like(accept_prob[:, :1]), accept_prob[:, :-1]],
                              dim=-1),
                    dim=-1,
                ) * mask                                            # f_{t-1}
                with torch.no_grad():
                    teacher_prob = teacher_logprob.exp()
                # 1 - TV(p_t, Q_t) = sum_v min(p_t(v), Q_t(v)); differentiable in Q_t.
                overlap = torch.minimum(teacher_prob, student_prob).sum(dim=-1)
                value = (survival_before * overlap).sum()           # pathwise via TV
                term = survival_before * overlap.detach()
                score_return = _suffix_sum(term) - term             # sum_{t > s}

        log_qs.append(log_q)
        values.append(value)
        score_returns.append(score_return * mask)
        actives.append(active.to(torch.float32))

    returns = torch.stack(score_returns)                            # [N, blocks, D]
    if baseline == "loo" and num_drafts > 1:
        # Mean score-return of the other drafts: independent of this draft's tokens.
        loo = (returns.sum(dim=0, keepdim=True) - returns) / (num_drafts - 1)
    else:
        loo = torch.zeros_like(returns)

    total = student_logits.new_zeros(())
    for draft in range(num_drafts):
        coefficient = (actives[draft] * returns[draft] - loo[draft]) * mask
        log_q = log_qs[draft]
        total = total + values[draft] + (coefficient * (log_q - log_q.detach())).sum()
    return -(total / num_drafts)


def _chunk_multidraft_accepted_length(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    prefix_mask: torch.Tensor,
    uniforms: torch.Tensor,
    temperature: float,
    baseline: str,
) -> torch.Tensor:
    """Negative expected accepted length of N-draft tree verification, summed over blocks.

    student_hidden [blocks, D, hidden]; teacher_hidden [N, blocks, D, hidden] where
    draft n's row s follows draft n's own prefix; labels [N, blocks, D] are the drafted
    tokens; prefix_mask [blocks, D]; uniforms [blocks, D, N] are the verifier's accept
    coins (passed in so that checkpoint recomputation replays the same walk).

    At depth s the alive chains are those agreeing with the committed prefix; their
    tokens are i.i.d. from Q_s. They are tried in chain order with recursive rejection
    (p^(j+1) = norm[p^(j) - Q_s]_+). The estimator is sum_s 1[reached s] a_s with the
    analytic a_s = 1 - prod_j TV(p^(j), Q_s); the sampled tokens and coins at depth s
    receive the score coefficient sum_{t > s} 1[reached t] a_t.
    """
    num_drafts, num_blocks, depth = labels.shape
    device = student_hidden.device

    student_logits = F.linear(student_hidden, lm_head_weight).float() / temperature
    student_logprob = F.log_softmax(student_logits, dim=-1)       # [B, D, V]
    student_prob = student_logprob.exp()
    with torch.no_grad():
        teacher_prob = F.softmax(
            F.linear(teacher_hidden, lm_head_weight).float() / temperature, dim=-1
        )                                                           # [N, B, D, V]

    tokens = labels.permute(1, 0, 2)                                # [B, N, D]
    valid = prefix_mask.bool()
    block_index = torch.arange(num_blocks, device=device)
    alive = torch.ones(num_blocks, num_drafts, dtype=torch.bool, device=device)
    reached = torch.ones(num_blocks, dtype=torch.bool, device=device)

    expected_accept, walk_log_prob, reached_mask = [], [], []
    for s in range(depth):
        reached = reached & valid[:, s]
        # Every alive chain shares the committed prefix, so any of them carries p_s.
        first_alive = alive.to(torch.int64).argmax(dim=1)
        residual = teacher_prob[first_alive, block_index, s]       # [B, V]
        q_s = student_prob[:, s]
        reject_all = q_s.new_ones(num_blocks)
        undecided = reached.clone()
        accepted_token = torch.full((num_blocks,), -1, dtype=torch.long, device=device)
        log_prob_s = q_s.new_zeros(num_blocks)

        for chain in range(num_drafts):
            candidate = alive[:, chain] & reached
            excess = (residual - q_s).clamp_min(0.0)
            reject_mass = excess.sum(dim=-1)                        # TV(p^(j), Q_s)
            reject_all = torch.where(candidate, reject_all * reject_mass, reject_all)

            token = tokens[:, chain, s]
            residual_token = residual.gather(-1, token.unsqueeze(-1)).squeeze(-1)
            q_token = q_s.gather(-1, token.unsqueeze(-1)).squeeze(-1)
            accept = (residual_token / q_token.clamp_min(_TINY)).clamp_max(1.0)

            tested = candidate & undecided
            take = uniforms[:, s, chain] < accept.detach()
            accept_now = tested & take
            reject_now = tested & ~take
            zero = torch.zeros_like(accept)
            log_prob_s = (
                log_prob_s
                + torch.where(accept_now, accept.clamp_min(_TINY).log(), zero)
                + torch.where(reject_now, (1.0 - accept).clamp_min(_TINY).log(), zero)
            )
            accepted_token = torch.where(accept_now, token, accepted_token)
            undecided = undecided & ~accept_now

            next_residual = excess / reject_mass.clamp_min(_TINY).unsqueeze(-1)
            residual = torch.where(candidate.unsqueeze(-1), next_residual, residual)

        token_log_q = student_logprob[:, s].gather(-1, tokens[:, :, s])     # [B, N]
        alive_here = (alive & reached.unsqueeze(-1)).to(token_log_q.dtype)
        log_prob_s = log_prob_s + (token_log_q * alive_here).sum(dim=-1)

        reached_float = reached.to(q_s.dtype)
        expected_accept.append((1.0 - reject_all) * reached_float)
        walk_log_prob.append(log_prob_s)
        reached_mask.append(reached_float)

        alive = alive & (tokens[:, :, s] == accepted_token.unsqueeze(-1))
        reached = reached & (accepted_token >= 0)

    expected_accept = torch.stack(expected_accept, dim=-1)         # [B, D]
    walk_log_prob = torch.stack(walk_log_prob, dim=-1)
    reached_mask = torch.stack(reached_mask, dim=-1)

    detached = expected_accept.detach()
    future = _suffix_sum(detached) - detached                       # sum_{t > s}
    if baseline == "loo":
        count = reached_mask.sum(dim=0, keepdim=True)
        others = (future * reached_mask).sum(dim=0, keepdim=True) - future * reached_mask
        loo = torch.where(count > 1, others / (count - 1).clamp_min(1.0),
                          torch.zeros_like(others))
    else:
        loo = torch.zeros_like(future)
    coefficient = (future - loo) * reached_mask
    straight_through = (coefficient * (walk_log_prob - walk_log_prob.detach())).sum()
    return -(expected_accept.sum() + straight_through)


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
    estimator: str | None = None,
    num_drafts: int = 1,
    baseline: str = "none",
    uniforms: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mean negative accepted length over valid blocks (second-stage objective).

    See the block comment above for the four estimators. The returned scalar is
    -mean_blocks(J) (for multidraft: the tree's expected accepted length); only the
    installed gradient differs between estimators.

    Args:
        lm_head_weight: frozen `lm_head.weight`, [vocab, hidden].
        student_hidden: flattened diffusion hidden states, [B*K, hidden]; shared by
            all drafts of a block (the parallel proposal does not see its own drafts).
        teacher_hidden: detached AR hidden states, [num_drafts*B*K, hidden], draft-major.
            For sampled estimators row s of draft n must follow draft n's prefix.
        labels: [num_drafts*B*K], draft-major; committed tokens (target-path) or drafted
            tokens. Offset K-1 is ignored, as in generation. An ignored target truncates
            its block: later positions cannot survive a missing earlier verification.
        block_size: K, including the unsupervised final offset.
        chunk_size: approximate positions (x drafts) per recomputed chunk. Chunks
            never split a block.
        ignore_index: label sentinel for padding / invalid supervision.
        temperature: positive sampling temperature applied to both distributions.
        on_policy: legacy switch; used only when `estimator` is None
            (True -> "onpolicy", False -> "target-path").
        estimator: one of ACCEPTED_LENGTH_ESTIMATORS.
        num_drafts: drafts per block (1 for target-path).
        baseline: "none" or "loo" (see above).
        uniforms: multidraft accept coins, [B, K-1, num_drafts] in [0, 1). Sampled
            with `generator` when omitted.
        generator: RNG for the multidraft coins.

    This objective does not model greedy argmax consensus decoding.
    """
    if estimator is None:
        estimator = "onpolicy" if on_policy else "target-path"
    if estimator not in ACCEPTED_LENGTH_ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}.")
    if baseline not in ACCEPTED_LENGTH_BASELINES:
        raise ValueError(f"unknown baseline {baseline!r}.")
    if num_drafts < 1:
        raise ValueError("num_drafts must be >= 1.")
    if estimator == "target-path" and num_drafts != 1:
        raise ValueError("target-path scores the single committed path (num_drafts=1).")
    if student_hidden.ndim != 2 or teacher_hidden.ndim != 2 or labels.ndim != 1:
        raise ValueError("student/teacher hidden must be [N, hidden], labels [N].")
    if student_hidden.shape[-1] != teacher_hidden.shape[-1]:
        raise ValueError("student/teacher hidden sizes differ.")
    if teacher_hidden.shape[0] != num_drafts * student_hidden.shape[0]:
        raise ValueError(
            f"teacher rows {teacher_hidden.shape[0]} != num_drafts x student rows "
            f"{num_drafts} x {student_hidden.shape[0]}."
        )
    if labels.shape[0] != teacher_hidden.shape[0]:
        raise ValueError("teacher hidden states and labels must contain the same positions.")
    if teacher_hidden.requires_grad:
        raise ValueError("teacher_hidden must be detached; the AR view is frozen.")
    if block_size < 2:
        raise ValueError("block_size must be at least 2.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0.0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive.")
    if student_hidden.shape[0] % block_size != 0:
        raise ValueError("the flattened supervision grid must contain whole blocks.")

    num_blocks = student_hidden.shape[0] // block_size
    depth = block_size - 1
    hidden_size = student_hidden.shape[-1]
    student = student_hidden.reshape(num_blocks, block_size, hidden_size)[:, :-1]
    teacher = teacher_hidden.reshape(num_drafts, num_blocks, block_size, hidden_size)[:, :, :-1]
    block_labels = labels.reshape(num_drafts, num_blocks, block_size)[:, :, :-1]

    if estimator == "multidraft":
        if uniforms is None:
            uniforms = torch.rand(
                (num_blocks, depth, num_drafts), generator=generator,
                device=student_hidden.device,
            )
        elif uniforms.shape != (num_blocks, depth, num_drafts):
            raise ValueError(
                f"uniforms must be [blocks, K-1, num_drafts] = "
                f"{(num_blocks, depth, num_drafts)}, got {tuple(uniforms.shape)}."
            )

    # First-rejection semantics require a contiguous supervised target prefix.
    # Do not skip a masked token and accidentally join two separate spans.
    valid = (block_labels != ignore_index).all(dim=0)
    prefix_mask = valid.to(torch.int64).cumprod(dim=-1).bool()
    block_keep = prefix_mask.any(dim=-1)
    student = student[block_keep]
    teacher = teacher[:, block_keep]
    block_labels = block_labels[:, block_keep]
    prefix_mask = prefix_mask[block_keep]
    if uniforms is not None:
        uniforms = uniforms[block_keep]

    num_kept_blocks = student.shape[0]
    if num_kept_blocks == 0:
        # Keep the graph alive so DDP/FSDP still see a gradient for every rank.
        return student_hidden.sum() * 0.0

    # Ignored tail entries are never scored, but gather still needs a legal id.
    block_labels = block_labels.masked_fill(~prefix_mask.unsqueeze(0), 0)
    blocks_per_chunk = max(1, chunk_size // (depth * num_drafts))

    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, num_kept_blocks, blocks_per_chunk):
        stop = min(start + blocks_per_chunk, num_kept_blocks)
        if estimator == "multidraft":
            total = total + checkpoint(
                _chunk_multidraft_accepted_length,
                lm_head_weight,
                student[start:stop],
                teacher[:, start:stop],
                block_labels[:, start:stop],
                prefix_mask[start:stop],
                uniforms[start:stop],
                temperature,
                baseline,
                use_reentrant=False,
            )
        else:
            total = total + checkpoint(
                _chunk_accepted_length,
                lm_head_weight,
                student[start:stop],
                teacher[:, start:stop],
                block_labels[:, start:stop],
                prefix_mask[start:stop],
                temperature,
                estimator,
                baseline,
                use_reentrant=False,
            )
    return total / num_kept_blocks


# ---------------------------------------------------------------------------------
# Multi-round objective: expected verification rounds per token
# ---------------------------------------------------------------------------------


def _chunk_target_path_survival(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    prefix_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """W[b, j] = prod_{i <= j} min(1, Q_i(x_i) / p_i(x_i)) along the committed path."""
    gather_index = labels.unsqueeze(-1)
    with torch.no_grad():
        teacher_logits = F.linear(teacher_hidden, lm_head_weight).float() / temperature
        log_p = F.log_softmax(teacher_logits, dim=-1).gather(-1, gather_index).squeeze(-1)
    student_logits = F.linear(student_hidden, lm_head_weight).float() / temperature
    log_q = F.log_softmax(student_logits, dim=-1).gather(-1, gather_index).squeeze(-1)
    accept = torch.exp((log_q - log_p).clamp_max(0.0))
    return torch.cumprod(accept, dim=-1) * prefix_mask.to(accept.dtype)


def expected_rounds_from_survival(survival: torch.Tensor) -> torch.Tensor:
    """Expected number of rounds that START inside each window.

    survival: [windows, H, D]; survival[w, n, j-1] = P(round started at window position n
    accepts >= j draft tokens | committed path). A round from n commits L + 1 tokens
    (L accepted drafts + one correction/bonus token), so the next round starts at
    n + L + 1. Occupancy recursion (cf. EDR, Zhao & Cai 2026):

        omega_0 = 1,   omega_{n+j+1} += omega_n * P(L = j),   rounds = sum_{n<H} omega_n.

    Differentiable in `survival` (exact gradient, no TD approximation).
    """
    num_windows, window_len, depth = survival.shape
    ones = survival.new_ones(num_windows, window_len, 1)
    zeros = survival.new_zeros(num_windows, window_len, 1)
    at_least = torch.cat([ones, survival, zeros], dim=-1)          # W_0 .. W_{D+1}
    stop_at = at_least[..., :-1] - at_least[..., 1:]                # P(L = j), j = 0..D

    total_len = window_len + depth + 1
    incoming = survival.new_zeros(num_windows, total_len)
    incoming[:, 0] = 1.0
    rounds = survival.new_zeros(num_windows)
    for n in range(window_len):
        occupancy = incoming[:, n]
        rounds = rounds + occupancy
        incoming = incoming + F.pad(
            occupancy.unsqueeze(-1) * stop_at[:, n], (n + 1, window_len - n - 1)
        )
    return rounds


def fused_linear_expected_rounds(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    block_size: int,
    window_len: int,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Mean expected verification rounds per token over windows of contiguous anchors.

    Unlike accepted length, this counts where the NEXT round starts: a round that ends
    just before a hard token costs differently from one that ends after it. Blocks must
    be laid out as consecutive windows of `window_len` anchors a, a+1, ..., a+H-1 (see
    `src.utils.anchors.sample_window_anchors`), with the usual target-path inputs:
    student = diffusion hidden per anchor, teacher = AR hidden at a+j, labels = the
    committed tokens.

    Exactness: the committed path enters only through the coupling
    P(accept | output x) = min(1, Q/p), which is exact when the rows are rollouts of the
    frozen teacher at `temperature` (e.g. scripts/generate_onpolicy.py with top-p/top-k
    disabled). On dataset text it is the same approximation as target-path EAL. No
    drafts are sampled, so no verification pass and no score-function variance.

    A window is used only if every one of its anchors has a supervised first target;
    later masked targets truncate their block (the round ends there).
    Returns (rounds / window_len) averaged over kept windows; 1 / value ~ tokens/round.
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
    if block_size < 2 or window_len < 1 or chunk_size <= 0:
        raise ValueError("need block_size >= 2, window_len >= 1, chunk_size > 0.")
    if not 0.0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive.")
    if labels.shape[0] % (block_size * window_len) != 0:
        raise ValueError("the grid must contain whole windows of whole blocks.")

    num_blocks = labels.shape[0] // block_size
    depth = block_size - 1
    hidden_size = student_hidden.shape[-1]
    student = student_hidden.reshape(num_blocks, block_size, hidden_size)[:, :-1]
    teacher = teacher_hidden.reshape(num_blocks, block_size, hidden_size)[:, :-1]
    block_labels = labels.reshape(num_blocks, block_size)[:, :-1]

    valid = block_labels != ignore_index
    prefix_mask = valid.to(torch.int64).cumprod(dim=-1).bool()
    window_keep = prefix_mask[:, 0].reshape(-1, window_len).all(dim=-1)
    block_keep = window_keep.repeat_interleave(window_len)
    num_windows = int(window_keep.sum().item())
    if num_windows == 0:
        return student_hidden.sum() * 0.0

    student = student[block_keep]
    teacher = teacher[block_keep]
    prefix_mask = prefix_mask[block_keep]
    block_labels = block_labels[block_keep].masked_fill(~prefix_mask, 0)

    blocks_per_chunk = max(1, chunk_size // depth)
    pieces = []
    for start in range(0, student.shape[0], blocks_per_chunk):
        stop = min(start + blocks_per_chunk, student.shape[0])
        pieces.append(checkpoint(
            _chunk_target_path_survival,
            lm_head_weight,
            student[start:stop],
            teacher[start:stop],
            block_labels[start:stop],
            prefix_mask[start:stop],
            temperature,
            use_reentrant=False,
        ))
    survival = torch.cat(pieces, dim=0).reshape(num_windows, window_len, depth)
    rounds = expected_rounds_from_survival(survival)
    return (rounds / window_len).mean()


# ---------------------------------------------------------------------------------
# Published baselines on the committed (target) path, for the stage-2 comparison
# ---------------------------------------------------------------------------------
#
#   lk          LK^alpha (Samarin et al., 2026): -log alpha_j, alpha_j = sum_v min(p, Q)
#               = 1 - TV, per supervised position.
#   lk-hybrid   LK^lambda: lambda_j KL(p||Q) + (1 - lambda_j) TV(p, Q),
#               lambda_j = exp(-eta * sg[mean alpha at offset j]) (per draft position,
#               averaged over the batch's blocks), eta = 3 by default.
#               Both LK variants weight offset j (0-based) by gamma^j, gamma = 0.8 -- the
#               paper's per-head weight, with block offsets playing the role of heads.
#   wtv         Window TV (Xia et al., 2026): -sum_k prod_{j<=k} (1 - TV_j) over the
#               block, on the committed prefix.
#   streak      SpecDiff-2 streak-distillation (Sandler et al., 2025):
#               -sum_m prod_{j<=m} Q_j(x_j), x = verifier samples (use teacher rollouts).
#
# All use the frozen AR teacher at the committed prefix and the same temperature.

BASELINE_OBJECTIVES = ("lk", "lk-hybrid", "wtv", "streak")


def _chunk_lk_statistics(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Per-offset sums over blocks: [4, D] = (KL, TV, -log alpha, count)."""
    with torch.no_grad():
        teacher_logits = F.linear(teacher_hidden, lm_head_weight).float() / temperature
        teacher_logprob = F.log_softmax(teacher_logits, dim=-1)
        teacher_prob = teacher_logprob.exp()
    student_logits = F.linear(student_hidden, lm_head_weight).float() / temperature
    student_logprob = F.log_softmax(student_logits, dim=-1)
    student_prob = student_logprob.exp()

    mask = valid_mask.to(torch.float32)
    kl = (teacher_prob * (teacher_logprob - student_logprob)).sum(dim=-1)
    alpha = torch.minimum(teacher_prob, student_prob).sum(dim=-1)
    tv = 1.0 - alpha
    neg_log_alpha = -alpha.clamp_min(1e-12).log()
    return torch.stack([
        (kl * mask).sum(dim=0),
        (tv * mask).sum(dim=0),
        (neg_log_alpha * mask).sum(dim=0),
        mask.sum(dim=0),
    ])


def _chunk_window_product(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    prefix_mask: torch.Tensor,
    temperature: float,
    kind: str,
) -> torch.Tensor:
    """-sum_k prod_{j<=k} a_j summed over blocks; a = 1 - TV (wtv) or Q(x) (streak)."""
    student_logits = F.linear(student_hidden, lm_head_weight).float() / temperature
    student_logprob = F.log_softmax(student_logits, dim=-1)
    if kind == "wtv":
        with torch.no_grad():
            teacher_prob = F.softmax(
                F.linear(teacher_hidden, lm_head_weight).float() / temperature, dim=-1
            )
        per_position = torch.minimum(teacher_prob, student_logprob.exp()).sum(dim=-1)
    else:
        per_position = student_logprob.gather(-1, labels.unsqueeze(-1)).squeeze(-1).exp()
    survival = torch.cumprod(per_position, dim=-1) * prefix_mask.to(per_position.dtype)
    return -survival.sum()


def fused_linear_baseline_loss(
    kind: str,
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    block_size: int,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    temperature: float = 1.0,
    lk_eta: float = 3.0,
    lk_gamma: float = 0.8,
) -> torch.Tensor:
    """One of BASELINE_OBJECTIVES on the target-path inputs (as for target-path EAL).

    lk / lk-hybrid: sum_j gamma^j * mean over blocks with a supervised offset j.
    wtv / streak: mean over blocks with a non-empty supervised prefix of -sum_k prod a.
    """
    if kind not in BASELINE_OBJECTIVES:
        raise ValueError(f"unknown baseline objective {kind!r}.")
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"student/teacher hidden mismatch: {tuple(student_hidden.shape)} vs "
            f"{tuple(teacher_hidden.shape)}"
        )
    if student_hidden.ndim != 2 or labels.ndim != 1 or labels.shape[0] != student_hidden.shape[0]:
        raise ValueError("hidden states must be [N, hidden] and labels [N].")
    if teacher_hidden.requires_grad:
        raise ValueError("teacher_hidden must be detached; the AR view is frozen.")
    if block_size < 2 or chunk_size <= 0 or labels.shape[0] % block_size != 0:
        raise ValueError("need block_size >= 2, chunk_size > 0 and whole blocks.")
    if not 0.0 < temperature < float("inf") or lk_eta <= 0.0 or lk_gamma <= 0.0:
        raise ValueError("temperature, lk_eta and lk_gamma must be positive.")

    num_blocks = labels.shape[0] // block_size
    depth = block_size - 1
    hidden_size = student_hidden.shape[-1]
    student = student_hidden.reshape(num_blocks, block_size, hidden_size)[:, :-1]
    teacher = teacher_hidden.reshape(num_blocks, block_size, hidden_size)[:, :-1]
    block_labels = labels.reshape(num_blocks, block_size)[:, :-1]
    valid = block_labels != ignore_index
    blocks_per_chunk = max(1, chunk_size // depth)

    if kind in ("lk", "lk-hybrid"):
        keep = valid.any(dim=-1)
        student, teacher, valid = student[keep], teacher[keep], valid[keep]
        if student.shape[0] == 0:
            return student_hidden.sum() * 0.0
        stats = student_hidden.new_zeros((4, depth), dtype=torch.float32)
        for start in range(0, student.shape[0], blocks_per_chunk):
            stop = min(start + blocks_per_chunk, student.shape[0])
            stats = stats + checkpoint(
                _chunk_lk_statistics, lm_head_weight, student[start:stop],
                teacher[start:stop], valid[start:stop], temperature,
                use_reentrant=False,
            )
        kl_sum, tv_sum, nla_sum, count = stats
        present = count > 0
        count = count.clamp_min(1.0)
        weights = lk_gamma ** torch.arange(depth, device=stats.device, dtype=stats.dtype)
        weights = weights * present.to(stats.dtype)
        if kind == "lk":
            per_offset = nla_sum / count
        else:
            mean_alpha = (1.0 - tv_sum / count).detach()
            lam = torch.exp(-lk_eta * mean_alpha)
            per_offset = (lam * kl_sum + (1.0 - lam) * tv_sum) / count
        return (weights * per_offset).sum()

    prefix_mask = valid.to(torch.int64).cumprod(dim=-1).bool()
    keep = prefix_mask.any(dim=-1)
    student, teacher = student[keep], teacher[keep]
    prefix_mask = prefix_mask[keep]
    block_labels = block_labels[keep].masked_fill(~prefix_mask, 0)
    if student.shape[0] == 0:
        return student_hidden.sum() * 0.0
    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, student.shape[0], blocks_per_chunk):
        stop = min(start + blocks_per_chunk, student.shape[0])
        total = total + checkpoint(
            _chunk_window_product, lm_head_weight, student[start:stop],
            teacher[start:stop], block_labels[start:stop], prefix_mask[start:stop],
            temperature, kind, use_reentrant=False,
        )
    return total / student.shape[0]


# ---------------------------------------------------------------------------------
# Noise-conditioned drafting: target-path EAL against the revealed law p(x | G_S)
# ---------------------------------------------------------------------------------


def _chunk_conditioned_survival(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    log_target: torch.Tensor,
    labels: torch.Tensor,
    prefix_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_logits = F.linear(student_hidden, lm_head_weight).float() / temperature
    log_q = F.log_softmax(student_logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    accept = torch.exp((log_q - log_target).clamp_max(0.0))
    survival = torch.cumprod(accept, dim=-1) * prefix_mask.to(accept.dtype)
    return -survival.sum()


def fused_linear_conditioned_target_path(
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    log_target: torch.Tensor,
    labels: torch.Tensor,
    block_size: int,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    temperature: float = 1.0,
) -> torch.Tensor:
    """-mean_blocks sum_t prod_{s<=t} min(1, Q_s(x_s) / target_s(x_s)).

    Target-path EAL with a per-position target given as log-probability of the committed
    token: log p(x | G_S) from `noise_conditioned_dual_pass`. On teacher rollouts with
    posterior noise this is unbiased for the accepted length of noise-conditioned
    decoding (coupling), with an exact pathwise gradient and no draft sampling.
    """
    if student_hidden.ndim != 2 or labels.ndim != 1 or log_target.shape != labels.shape:
        raise ValueError("student_hidden must be [N, H]; labels and log_target [N].")
    if student_hidden.shape[0] != labels.shape[0] or labels.shape[0] % block_size:
        raise ValueError("positions must align and form whole blocks.")
    if log_target.requires_grad:
        raise ValueError("log_target must be detached.")
    if block_size < 2 or chunk_size <= 0 or not 0.0 < temperature < float("inf"):
        raise ValueError("need block_size >= 2, chunk_size > 0, positive temperature.")
    num_blocks = labels.shape[0] // block_size
    depth = block_size - 1
    student = student_hidden.reshape(num_blocks, block_size, -1)[:, :-1]
    target = log_target.reshape(num_blocks, block_size)[:, :-1].float()
    block_labels = labels.reshape(num_blocks, block_size)[:, :-1]
    prefix_mask = (block_labels != ignore_index).to(torch.int64).cumprod(dim=-1).bool()
    keep = prefix_mask.any(dim=-1)
    if not keep.any():
        return student_hidden.sum() * 0.0
    student, target, prefix_mask = student[keep], target[keep], prefix_mask[keep]
    block_labels = block_labels[keep].masked_fill(~prefix_mask, 0)
    target = target.masked_fill(~prefix_mask, 0.0)
    blocks_per_chunk = max(1, chunk_size // depth)
    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, student.shape[0], blocks_per_chunk):
        stop = min(start + blocks_per_chunk, student.shape[0])
        total = total + checkpoint(
            _chunk_conditioned_survival, lm_head_weight, student[start:stop],
            target[start:stop], block_labels[start:stop], prefix_mask[start:stop],
            temperature, use_reentrant=False,
        )
    return total / student.shape[0]


# ---------------------------------------------------------------------------------
# DBLAST (re-implementation, see src/dblast.py)
# ---------------------------------------------------------------------------------

DBLAST_OBJECTIVES = ("al", "nll")


def _chunk_dblast(
    lm_head_weight, expander_weight, expander_bias, prior_weight, prior_bias,
    block_hidden, log_p, labels, prefix_mask, num_categories, kind, tau, temperature,
):
    from src.dblast import al_objective_terms, branch_hidden, category_log_prior

    branches = branch_hidden(block_hidden, expander_weight, expander_bias, num_categories)
    logits = F.linear(branches, lm_head_weight).float() / temperature        # [b, D, K, V]
    log_q_branch = F.log_softmax(logits, dim=-1).gather(
        -1, labels[:, :, None, None].expand(-1, -1, num_categories, 1)
    ).squeeze(-1)                                                           # [b, D, K]
    log_prior = category_log_prior(block_hidden[:, 0], prior_weight, prior_bias,
                                   num_categories)
    objective, log_q_prefix, _, _ = al_objective_terms(
        log_q_branch, log_prior, log_p, prefix_mask, tau
    )
    if kind == "al":
        return -objective.sum()
    last = prefix_mask.to(torch.int64).sum(1) - 1                           # dependent NLL
    return -log_q_prefix.gather(1, last.unsqueeze(1)).sum()


def fused_linear_dblast(
    kind: str,
    lm_head_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    block_size: int,
    num_categories: int,
    expander_weight: torch.Tensor | None = None,
    expander_bias: torch.Tensor | None = None,
    prior_weight: torch.Tensor | None = None,
    prior_bias: torch.Tensor | None = None,
    tau: float = 0.1,
    temperature: float = 1.0,
    target_top_k: int = 0,
    target_top_p: float = 1.0,
    chunk_size: int = 1024,
    ignore_index: int = -100,
) -> torch.Tensor:
    """DBLAST losses on target-path inputs, mean over blocks.

    kind="al": -J_AL (paper Eq. 16); kind="nll": dependent NLL (paper Eq. 8; with K=1
    the independent NLL of plain DFlash training). The target is filtered with the same
    temperature / top-k / top-p as the trajectories; a committed token outside that
    filter (p = 0, impossible on matching rollouts) truncates its block.
    """
    if kind not in DBLAST_OBJECTIVES:
        raise ValueError(f"unknown DBLAST objective {kind!r}.")
    if num_categories < 1:
        raise ValueError("num_categories must be >= 1.")
    if num_categories > 1 and any(t is None for t in (expander_weight, expander_bias,
                                                      prior_weight, prior_bias)):
        raise ValueError("K > 1 needs the expander and prior heads.")
    if student_hidden.shape != teacher_hidden.shape or labels.ndim != 1 \
            or labels.shape[0] != student_hidden.shape[0] or labels.shape[0] % block_size:
        raise ValueError("student/teacher [N, H] and labels [N] over whole blocks.")
    if teacher_hidden.requires_grad:
        raise ValueError("teacher_hidden must be detached; the AR view is frozen.")
    if block_size < 2 or chunk_size <= 0 or not 0.0 < tau < 1.0 \
            or not 0.0 < temperature < float("inf"):
        raise ValueError("invalid block_size / chunk_size / tau / temperature.")

    from src.dblast import filtered_log_probs

    num_blocks = labels.shape[0] // block_size
    depth = block_size - 1
    hidden_size = student_hidden.shape[-1]
    student = student_hidden.reshape(num_blocks, block_size, hidden_size)[:, :-1]
    teacher = teacher_hidden.reshape(num_blocks, block_size, hidden_size)[:, :-1]
    block_labels = labels.reshape(num_blocks, block_size)[:, :-1]
    valid = block_labels != ignore_index
    safe_labels = block_labels.masked_fill(~valid, 0)

    log_p = torch.empty(num_blocks, depth, dtype=torch.float32, device=labels.device)
    rows = max(1, chunk_size // depth)
    with torch.no_grad():
        for start in range(0, num_blocks, rows):
            stop = min(start + rows, num_blocks)
            log_p[start:stop] = filtered_log_probs(
                F.linear(teacher[start:stop], lm_head_weight), temperature,
                target_top_k, target_top_p,
            ).gather(-1, safe_labels[start:stop].unsqueeze(-1)).squeeze(-1)
    valid = valid & torch.isfinite(log_p)
    prefix_mask = valid.to(torch.int64).cumprod(dim=-1).bool()
    keep = prefix_mask.any(dim=-1)
    if not keep.any():
        return student_hidden.sum() * 0.0
    student, log_p = student[keep], log_p[keep]
    prefix_mask, safe_labels = prefix_mask[keep], safe_labels[keep]
    log_p = log_p.masked_fill(~prefix_mask, 0.0)

    blocks_per_chunk = max(1, chunk_size // (depth * num_categories))
    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, student.shape[0], blocks_per_chunk):
        stop = min(start + blocks_per_chunk, student.shape[0])
        total = total + checkpoint(
            _chunk_dblast, lm_head_weight, expander_weight, expander_bias, prior_weight,
            prior_bias, student[start:stop], log_p[start:stop], safe_labels[start:stop],
            prefix_mask[start:stop], num_categories, kind, tau, temperature,
            use_reentrant=False,
        )
    return total / student.shape[0]
