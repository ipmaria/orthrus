"""Partial Gumbel reveal for factorized drafters (toy). Order-free shared-noise coupling.

Target sampling: x_t = argmax_v log p_t(v) + G_tv, G i.i.d. Gumbel (exact sample of p_t).
The drafter is shown G_t on a set S_t (chosen before G is drawn: top-m of the drafter's
own marginal). Given G_S the target's law is closed-form:
    s* = max_{v in S} log p_v + G_v (at v*),  w = exp(-p(S^c) e^{-s*}),
    p(.|G_S) = w * delta_{v*} + (1 - w) * p restricted to S^c (renormalized).
Averaging over G_S returns p, so ordinary min(1, p(.|G_S)/q) verification stays lossless.
m = 0: standard SD.  m = V: deterministic exact match.
The factorized drafter outputs q_s = marginal of x_s under ITS OWN model conditioned on
the revealed noise (Bayes-optimal factorized for an exact drafter).
"""
import itertools
import numpy as np

V, D = 4, 4
SEQS = list(itertools.product(range(V), repeat=D))


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def cond_law(p, G, S):
    if len(S) == 0:
        return p
    scores = np.log(p[S]) + G[S]
    vstar = S[int(np.argmax(scores))]; sstar = scores.max()
    out_mask = np.ones(V, bool); out_mask[S] = False
    p_out = p[out_mask].sum()
    w = np.exp(-p_out * np.exp(-sstar)) if p_out > 0 else 1.0
    law = np.zeros(V)
    if p_out > 0:
        law[out_mask] = (1 - w) * p[out_mask] / p_out
    law[vstar] += w
    return law


class Chain:
    def __init__(self, P0, T): self.P0, self.T = P0, T
    def cond(self, prefix): return self.P0 if not prefix else self.T[prefix[-1]]
    def marginals(self):
        m = [self.P0]
        for _ in range(D - 1): m.append(m[-1] @ self.T)
        return np.array(m)


def expected_len(target, drafter, G, S):
    q = np.zeros((D, V))
    for x in SEQS:
        pr = 1.0
        for s in range(D):
            pr *= cond_law(drafter.cond(x[:s]), G[s], S[s])[x[s]]
        if pr > 0:
            for s in range(D): q[s, x[s]] += pr
    q = np.maximum(q, 1e-12); q /= q.sum(-1, keepdims=True)
    tot = 0.0
    for y in SEQS:
        prob, f, j = 1.0, 1.0, 0.0
        for s in range(D):
            prob *= q[s, y[s]]
            f *= min(1.0, cond_law(target.cond(y[:s]), G[s], S[s])[y[s]] / q[s, y[s]])
            j += f
        tot += prob * j
    return tot


rng = np.random.default_rng(3)
for scale in (1.0, 2.5):
    P0 = softmax(rng.normal(size=V) * scale); T = softmax(rng.normal(size=(V, V)) * scale)
    target = Chain(P0, T)
    print(f"\n=== logit scale {scale}  (D={D}, V={V})")
    for sigma in (0.0, 0.3, 0.6, 1.0):
        r = np.random.default_rng(11)
        drafter = target if sigma == 0 else Chain(
            softmax(np.log(P0) + r.normal(size=V) * sigma),
            softmax(np.log(T) + r.normal(size=(V, V)) * sigma))
        order = np.argsort(-drafter.marginals(), axis=-1)
        cells = []
        for m in range(V + 1):
            S = [order[s, :m] for s in range(D)]
            samples = 1 if m == 0 else 250
            vals = [expected_len(target, drafter, -np.log(-np.log(r.uniform(size=(D, V)))), S)
                    for _ in range(samples)]
            cells.append(np.mean(vals))
        print(f"  sigma={sigma}: " + "  ".join(f"m={m}:{v:.3f}" for m, v in enumerate(cells)))
