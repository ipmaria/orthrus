"""Torch-free verification of the stage-2 estimators (numpy, exact enumeration).

    python scripts/verify_estimators_numpy.py

Checks, on small random problems, by enumerating every draft / coin outcome and taking
central finite differences:
  * rb (Rao-Blackwellised EAL): unbiased value and gradient;
  * leave-one-out baselines for onpolicy and rb: unbiased gradient;
  * multidraft tree verification: unbiased value and gradient, and its gain over one draft;
  * the coupling identity behind expected-rounds: averaging the occupancy recursion over
    teacher rollouts equals the exact expected number of speculative-decoding rounds;
  * gradient variance of onpolicy vs rb vs target-path-on-teacher-rollouts.
"""

import itertools
import numpy as np

rng = np.random.default_rng(0)
EPS = 1e-6


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fd_grad(fun, theta, h=1e-5):
    g = np.zeros_like(theta)
    for idx in np.ndindex(theta.shape):
        tp, tm = theta.copy(), theta.copy()
        tp[idx] += h
        tm[idx] -= h
        g[idx] = (fun(tp) - fun(tm)) / (2 * h)
    return g


# ----------------------------------------------------------------- single draft
D, V = 3, 3
theta0 = rng.normal(size=(D, V)) * 0.8
teacher = {}
for s in range(D):
    for prefix in itertools.product(range(V), repeat=s):
        teacher[prefix] = softmax(rng.normal(size=V) * 0.9)


def p_of(prefix):
    return teacher[tuple(prefix)]


def exact_ej(theta):
    q = softmax(theta)
    total = 0.0
    for y in itertools.product(range(V), repeat=D):
        prob = np.prod([q[s, y[s]] for s in range(D)])
        f, j = 1.0, 0.0
        for s in range(D):
            f *= min(1.0, p_of(y[:s])[y[s]] / q[s, y[s]])
            j += f
        total += prob * j
    return total


def rb_parts(theta0, y):
    """Frozen quantities of the RB estimator on one sampled path."""
    q0 = softmax(theta0)
    r = np.array([min(1.0, p_of(y[:s])[y[s]] / q0[s, y[s]]) for s in range(D)])
    f_prev = np.concatenate([[1.0], np.cumprod(r)[:-1]])
    overlap0 = np.array([np.minimum(p_of(y[:t]), q0[t]).sum() for t in range(D)])
    g = f_prev * overlap0
    score_return = np.array([g[s + 1:].sum() for s in range(D)])   # sum_{t>s} g_t
    active = np.array([q0[s, y[s]] <= p_of(y[:s])[y[s]] for s in range(D)], float)
    return f_prev, score_return, active, g.sum()


def rb_surrogate(theta, y, f_prev, coeff):
    q = softmax(theta)
    overlap = np.array([np.minimum(p_of(y[:t]), q[t]).sum() for t in range(D)])
    logq = np.array([np.log(q[s, y[s]]) for s in range(D)])
    return (f_prev * overlap).sum() + (coeff * logq).sum()


def sampled_parts(theta0, y):
    """Frozen quantities of the existing on-policy (sampled-coin-free) estimator."""
    q0 = softmax(theta0)
    r = np.array([min(1.0, p_of(y[:s])[y[s]] / q0[s, y[s]]) for s in range(D)])
    f = np.cumprod(r)
    score_return = np.array([f[s:].sum() for s in range(D)])        # sum_{t>=s} f_t
    active = np.array([q0[s, y[s]] <= p_of(y[:s])[y[s]] for s in range(D)], float)
    return f.sum(), score_return, active


def sampled_surrogate(theta, y, coeff):
    q = softmax(theta)
    return (coeff * np.array([np.log(q[s, y[s]]) for s in range(D)])).sum()


paths = list(itertools.product(range(V), repeat=D))
q0 = softmax(theta0)
w = {y: np.prod([q0[s, y[s]] for s in range(D)]) for y in paths}
exact_val = exact_ej(theta0)
exact_grad = fd_grad(exact_ej, theta0)

# (1) RB, no baseline
val, grad = 0.0, np.zeros_like(theta0)
for y in paths:
    f_prev, sret, active, jrb = rb_parts(theta0, y)
    coeff = active * sret
    val += w[y] * jrb
    grad += w[y] * fd_grad(lambda th: rb_surrogate(th, y, f_prev, coeff), theta0)
print("RB value err", abs(val - exact_val), " grad err", np.abs(grad - exact_grad).max())

# (1b) RB variance vs. sampled estimator (per-sample J values)
j_rb = np.array([rb_parts(theta0, y)[3] for y in paths])
j_s = np.array([sampled_parts(theta0, y)[0] for y in paths])
pw = np.array([w[y] for y in paths])
var = lambda a: (pw * (a - (pw * a).sum()) ** 2).sum()
print("value variance: sampled", var(j_s), " RB", var(j_rb))

# (3a) LOO baseline with two iid drafts: RB and sampled estimators
for name in ("rb", "sampled"):
    grad = np.zeros_like(theta0)
    for y1 in paths:
        for y2 in paths:
            pair_w = w[y1] * w[y2]
            parts = {}
            for y in (y1, y2):
                if name == "rb":
                    f_prev, sret, active, _ = rb_parts(theta0, y)
                    parts[y] = (f_prev, sret, active)
                else:
                    _, sret, active = sampled_parts(theta0, y)
                    parts[y] = (None, sret, active)
            for me, other in ((y1, y2), (y2, y1)):
                f_prev, sret, active = parts[me]
                baseline = parts[other][1]
                coeff = active * sret - baseline
                if name == "rb":
                    fun = lambda th: rb_surrogate(th, me, f_prev, coeff)
                else:
                    fun = lambda th: sampled_surrogate(th, me, coeff)
                grad += 0.5 * pair_w * fd_grad(fun, theta0)
    print(f"LOO[{name}] grad err", np.abs(grad - exact_grad).max())

# ----------------------------------------------------------------- multi-draft
N, D2, V2 = 2, 2, 3
theta_m = rng.normal(size=(D2, V2)) * 0.8
teacher2 = {}
for s in range(D2):
    for prefix in itertools.product(range(V2), repeat=s):
        teacher2[prefix] = softmax(rng.normal(size=V2) * 0.9)


def residual_chain(p, q, k):
    """Residual targets p^(0..k-1) and per-candidate rejection masses TV_j."""
    res, tvs = [p], []
    for j in range(k):
        tv = np.maximum(res[-1] - q, 0.0).sum()
        tvs.append(tv)
        nxt = np.maximum(res[-1] - q, 0.0) / max(tv, 1e-30)
        res.append(nxt)
    return res[:k], tvs


def multidraft_exact(theta):
    q = softmax(theta)

    def rec(s, prefix, alive, Y):
        if s == D2 or not alive:
            return 0.0
        p = teacher2[tuple(prefix)]
        res, _ = residual_chain(p, q[s], len(alive))
        total, reach = 0.0, 1.0
        for j, c in enumerate(alive):
            v = Y[c][s]
            acc = min(1.0, res[j][v] / q[s, v])
            nxt = [cc for cc in alive if Y[cc][s] == v]
            total += reach * acc * (1.0 + rec(s + 1, prefix + [v], nxt, Y))
            reach *= 1.0 - acc
        return total

    total = 0.0
    for flat in itertools.product(range(V2), repeat=N * D2):
        Y = [flat[c * D2:(c + 1) * D2] for c in range(N)]
        prob = np.prod([q[s, Y[c][s]] for c in range(N) for s in range(D2)])
        total += prob * rec(0, [], list(range(N)), Y)
    return total


def multidraft_walk(theta, Y, dec):
    """Run the sampled walk. Returns per-depth (a_s, logP_s) for reached depths,
    plus the probability of the decision pattern actually used (tested slots)."""
    q = softmax(theta)
    alive, prefix, out, dec_prob = list(range(N)), [], [], 1.0
    tested = np.zeros((D2, N), bool)
    for s in range(D2):
        p = teacher2[tuple(prefix)]
        res, tvs = residual_chain(p, q[s], len(alive))
        a_s = 1.0 - np.prod(tvs)
        logp = sum(np.log(q[s, Y[c][s]]) for c in alive)
        accepted = None
        for j, c in enumerate(alive):
            v = Y[c][s]
            acc = min(1.0, res[j][v] / q[s, v])
            tested[s, j] = True
            d = dec[s, j]
            dec_prob *= acc if d else (1.0 - acc)
            logp += np.log(max(acc, 1e-300)) if d else np.log(max(1.0 - acc, 1e-300))
            if d:
                accepted = v
                break
        out.append((a_s, logp))
        if accepted is None:
            break
        prefix.append(accepted)
        alive = [c for c in alive if Y[c][s] == accepted]
    return out, dec_prob, tested


qm = softmax(theta_m)
exact_m = multidraft_exact(theta_m)
exact_m_grad = fd_grad(multidraft_exact, theta_m)
val, grad = 0.0, np.zeros_like(theta_m)
for flat in itertools.product(range(V2), repeat=N * D2):
    Y = [flat[c * D2:(c + 1) * D2] for c in range(N)]
    tok_w = np.prod([qm[s, Y[c][s]] for c in range(N) for s in range(D2)])
    for bits in itertools.product((0, 1), repeat=D2 * N):
        dec = np.array(bits).reshape(D2, N).astype(bool)
        out, dprob, tested = multidraft_walk(theta_m, Y, dec)
        weight = tok_w * dprob * 0.5 ** (~tested).sum()
        if weight == 0.0:
            continue
        a = np.array([o[0] for o in out])
        returns = np.array([a[s + 1:].sum() for s in range(len(a))])
        val += weight * a.sum()

        def surr(th, Y=Y, dec=dec, returns=returns):
            o2, _, _ = multidraft_walk(th, Y, dec)
            return sum(o2[s][0] + returns[s] * o2[s][1] for s in range(len(o2)))

        grad += weight * fd_grad(surr, theta_m)
print("multi-draft value err", abs(val - exact_m), " grad err",
      np.abs(grad - exact_m_grad).max())

# single-draft exact for the same toy, to see the multi-draft gain
def single_exact(theta):
    q = softmax(theta)
    tot = 0.0
    for y in itertools.product(range(V2), repeat=D2):
        prob = np.prod([q[s, y[s]] for s in range(D2)])
        f, j = 1.0, 0.0
        for s in range(D2):
            f *= min(1.0, teacher2[tuple(y[:s])][y[s]] / q[s, y[s]])
            j += f
        tot += prob * j
    return tot
print(f"E[L] single={single_exact(theta_m):.4f}  N={N} tree={exact_m:.4f}")

# ----------------------------------------------------------------- rounds coupling
H, D3, V3 = 3, 2, 2
L_TOT = H + D3 + 1
ptab, qtab = {}, {}
for t in range(L_TOT + 1):
    for prefix in itertools.product(range(V3), repeat=t):
        ptab[prefix] = softmax(rng.normal(size=V3) * 1.2)
        qtab[prefix] = softmax(rng.normal(size=(D3, V3)) * 1.2)  # drafter at start=prefix


def rounds_exact():
    def rec(prefix):
        n = len(prefix)
        if n >= H:
            return 0.0
        Q = qtab[tuple(prefix)]
        total = 1.0

        def branch(j, pre, prob):
            # j = number of draft tokens already accepted in this round
            if j == D3:  # bonus token
                p = ptab[tuple(pre)]
                return sum(prob * p[v] * rec(pre + [v]) for v in range(V3))
            p = ptab[tuple(pre)]
            out = 0.0
            for v in range(V3):
                qv = Q[j, v]
                acc = min(1.0, p[v] / qv)
                out += branch(j + 1, pre + [v], prob * qv * acc)
            resid = np.maximum(p - Q[j], 0.0)
            tv = resid.sum()
            if tv > 0:
                for v in range(V3):
                    out += prob * tv * (resid[v] / tv) * rec(pre + [v])
            return out

        return total + branch(0, list(prefix), 1.0)

    return rec([])


def rounds_coupled():
    total = 0.0
    for x in itertools.product(range(V3), repeat=H + D3):
        px = np.prod([ptab[tuple(x[:t])][x[t]] for t in range(H + D3)])
        omega = np.zeros(H + D3 + 2)
        omega[0] = 1.0
        for n in range(H):
            Q = qtab[tuple(x[:n])]
            W = [1.0]
            for j in range(D3):
                pos = n + j
                W.append(W[-1] * min(1.0, Q[j, x[pos]] / ptab[tuple(x[:pos])][x[pos]]))
            W.append(0.0)
            for j in range(D3 + 1):
                omega[n + j + 1] += omega[n] * (W[j] - W[j + 1])
        total += px * omega[:H].sum()
    return total


print("rounds exact", rounds_exact(), " coupled", rounds_coupled())

# ----------------------------------------------------------------- variance
def tp_surr(theta, x):
    q = softmax(theta); f, j = 1.0, 0.0
    for s in range(D):
        f *= min(1.0, q[s, x[s]] / p_of(x[:s])[x[s]]); j += f
    return j
px = {x: np.prod([p_of(x[:s])[x[s]] for s in range(D)]) for x in paths}
print("target-path on teacher rollouts: value err", abs(sum(px[x]*tp_surr(theta0,x) for x in paths)-exact_val))
def stats(items):  # list of (weight, grad)
    m = sum(w*g for w,g in items); v = sum(w*((g-m)**2).sum() for w,g in items)
    return np.abs(m-exact_grad).max(), v
samp=[];rb=[];tp=[]
for y in paths:
    _, sret, act = sampled_parts(theta0,y); samp.append((w[y], fd_grad(lambda th: sampled_surrogate(th,y,act*sret),theta0)))
    fp, sr, ac, _ = rb_parts(theta0,y); rb.append((w[y], fd_grad(lambda th: rb_surrogate(th,y,fp,ac*sr),theta0)))
    tp.append((px[y], fd_grad(lambda th: tp_surr(th,y),theta0)))
for n,it in (("onpolicy-sampled",samp),("rb",rb),("target-path@rollouts",tp)):
    b,v=stats(it); print(f"{n:22s} bias={b:.1e} grad-var(trace)={v:.4f}")
