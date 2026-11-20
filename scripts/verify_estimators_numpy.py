"""Torch-free verification of the stage-2 estimators (numpy, exact enumeration).

    python scripts/verify_estimators_numpy.py

Checks, on small random problems, by enumerating every draft / coin outcome and taking
central finite differences:
  * rb (Rao-Blackwellised EAL): unbiased value and gradient;
  * leave-one-out baselines for onpolicy and rb: unbiased gradient;
  * multidraft tree verification: unbiased value and gradient, and its gain over one draft;
  * the coupling identity behind expected-rounds: averaging the occupancy recursion over
    teacher rollouts equals the exact expected number of speculative-decoding rounds;
  * gradient variance of onpolicy vs rb vs target-path-on-teacher-rollouts;
  * losslessness of the tree verifier (src/tree_verify.py): exact output law = target;
  * noise-conditioned drafting (src/noise_coupling.py): posterior Gumbel sampler, the
    revealed law p(.|G_S), and a full decoding round with a causal noise-conditioned
    drafter (lossless; training estimator unbiased), plus the non-causal counterexample.
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

# ----------------------------------------------------------------- tree losslessness
rng = np.random.default_rng(5)
for (N, D, V) in [(1,2,3), (2,2,3), (3,2,2), (2,3,2)]:
    ptab = {pre: softmax(rng.normal(size=V)*1.3) for t in range(D+2) for pre in itertools.product(range(V), repeat=t)}
    Q = softmax(rng.normal(size=(D, V))*1.3)
    out = {}
    def add(seq, prob):
        # complete to length D+1 with the exact target law (later rounds are lossless)
        if len(seq) >= D+1:
            out[seq[:D+1]] = out.get(seq[:D+1], 0) + prob; return
        p = ptab[seq]
        for v in range(V): add(seq+(v,), prob*p[v])
    def walk(chains, prob):
        # recursion over depth / candidates with exact branch probabilities
        def depth(s, alive, committed, prob):
            if s == D:
                p = ptab[committed]
                for v in range(V): add(committed+(v,), prob*p[v])
                return
            base = ptab[committed]; residual = base.copy(); q = Q[s]
            def cand(j, residual, prob):
                if j == len(alive):
                    for v in range(V): add(committed+(v,), prob*residual[v])
                    return
                c = alive[j]; v = chains[c][s]
                ratio = min(1.0, residual[v]/q[v])
                if ratio > 0:
                    depth(s+1, [cc for cc in alive if chains[cc][s]==v], committed+(v,), prob*ratio)
                if ratio < 1:
                    r = np.maximum(residual-q, 0); m = r.sum()
                    cand(j+1, r/m if m > 1e-15 else base, prob*(1-ratio))
            cand(0, residual, prob)
        depth(0, list(range(N)), (), prob)
    for flat in itertools.product(range(V), repeat=N*D):
        chains = [flat[c*D:(c+1)*D] for c in range(N)]
        walk(chains, np.prod([Q[s, chains[c][s]] for c in range(N) for s in range(D)]))
    err = 0.0
    for seq in itertools.product(range(V), repeat=D+1):
        target = np.prod([ptab[seq[:t]][seq[t]] for t in range(D+1)])
        err = max(err, abs(out.get(seq, 0)-target))
    print(f"N={N} D={D} V={V}: max |P_tree - P_target| = {err:.2e}  total={sum(out.values()):.12f}")

# ----------------------------------------------------------------- noise conditioning
rng = np.random.default_rng(0)
V = 6
p = np.exp(rng.normal(size=V) * 1.3); p /= p.sum(); logp = np.log(p)

def gumbel(size, loc=0.0): return loc - np.log(-np.log(rng.uniform(size=size)))

def posterior(logp, x, n):
    """G | argmax(logp + G) = x, for n independent draws (full vocabulary)."""
    M = gumbel(n)                                   # max of logp+G ~ Gumbel(logsumexp)=Gumbel(0)
    phi = gumbel((n, V), loc=logp)                  # untruncated perturbed scores
    trunc = -np.log(np.exp(-M)[:, None] + np.exp(-phi))   # Gumbel(logp) truncated below M
    trunc[:, x] = M
    return trunc - logp

# (a) x ~ p then G ~ posterior  =>  G ~ iid Gumbel(0), argmax reproduces x
n = 400000
xs = rng.choice(V, size=n, p=p)
G = np.empty((n, V))
for v in range(V):
    idx = np.where(xs == v)[0]
    G[idx] = posterior(logp, v, len(idx))
print("argmax reproduces x:", np.all(np.argmax(logp + G, 1) == xs))
print("mean (want 0.5772):", G.mean(0).round(3), " var (want 1.645):", G.var(0).round(3))
print("max |corr| off-diagonal:", np.abs(np.corrcoef(G.T) - np.eye(V)).max().round(4))

# (b) closed-form law P(x | G_S) vs Monte Carlo over the unrevealed noise
def law(logp, G, S):
    S = np.asarray(S); p = np.exp(logp)
    scores = logp[S] + G[S]; vstar = S[np.argmax(scores)]; s = scores.max()
    out = np.ones(V, bool); out[S] = False; pout = p[out].sum()
    w = np.exp(-pout * np.exp(-s)) if pout > 0 else 1.0
    L = np.zeros(V); L[out] = (1 - w) * p[out] / max(pout, 1e-300); L[vstar] += w
    return L
S = [1, 4]
for trial in range(3):
    g = gumbel(V)
    Gs = gumbel((200000, V)); Gs[:, S] = g[S]
    emp = np.bincount(np.argmax(logp + Gs, 1), minlength=V) / len(Gs)
    print("law vs MC max err:", np.abs(law(logp, g, S) - emp).max().round(4))

# (c) sampling the posterior only on S u {x} matches the full posterior's marginal on S
x = 2
full = posterior(logp, x, 200000)[:, S]
M = gumbel(200000)
part = np.stack([(-np.log(np.exp(-M) + np.exp(-gumbel(200000, loc=logp[v])))) - logp[v]
                 for v in S], 1)
print("restricted posterior: mean diff", np.abs(full.mean(0) - part.mean(0)).round(3),
      " std diff", np.abs(full.std(0) - part.std(0)).round(3))


def noise_round(causal, n):
    rng = np.random.default_rng(2)
    V, D, m = 4, 3, 2
    def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
    def gumbel(size, loc=0.0): return loc - np.log(-np.log(rng.uniform(size=size)))
    P0 = softmax(rng.normal(size=V) * 1.5); T = softmax(rng.normal(size=(V, V)) * 1.5)
    cond = lambda pre: P0 if not pre else T[pre[-1]]
    S = [np.argsort(-rng.normal(size=V))[:m] for _ in range(D)]        # fixed noise-free supports
    theta = rng.normal(size=(D, V)); A = rng.normal(size=(D, D, m)) * 1.5
    if causal:  # q_s may only see the revealed noise of positions <= s
        A = A * (np.arange(D)[None, :, None] <= np.arange(D)[:, None, None])

    def law(p, G, Ss):
        sc = np.log(p[Ss]) + G; vs = Ss[np.argmax(sc)]; s = sc.max()
        out = np.ones(V, bool); out[Ss] = False; po = p[out].sum(); w = np.exp(-po * np.exp(-s))
        L = np.zeros(V); L[out] = (1 - w) * p[out] / po; L[vs] += w; return L

    def drafter(G):                       # q_s depends on the noise of every position
        q = theta.copy()
        for s in range(D):
            for j in range(D):
                q[s, S[j]] += A[s, j] * G[j]
        return softmax(q)

    # A) real decoding rounds
    counts = {}; lengths = []
    for _ in range(n):
        G = np.stack([gumbel(m) for _ in range(D)]); q = drafter(G)
        y = [rng.choice(V, p=q[s]) for s in range(D)]
        out, acc = [], 0
        for s in range(D):
            Ls = law(cond(y[:s]), G[s], S[s])
            if rng.uniform() < min(1, Ls[y[s]] / q[s, y[s]]):
                out.append(y[s]); acc += 1; continue
            r = np.maximum(Ls - q[s], 0); out.append(rng.choice(V, p=r / r.sum())); break
        else:
            out.append(rng.choice(V, p=cond(out)))
        while len(out) < D + 1: out.append(rng.choice(V, p=cond(out)))
        counts[tuple(out[:D + 1])] = counts.get(tuple(out[:D + 1]), 0) + 1; lengths.append(acc)
    tv = 0.5 * sum(abs(counts.get(x, 0) / n - np.prod([cond(x[:t])[x[t]] for t in range(D + 1)]))
                   for x in itertools.product(range(V), repeat=D + 1))
    print(f"[causal={causal}] lossless: TV(output, target) = {tv:.4f}  (MC noise ~ {0.5*np.sqrt(V**(D+1)/n)*0.8:.4f})")

    # B) training estimator: x ~ p, G ~ posterior(.|x) on S, survival with min(1, q/law)
    def posterior(logp_S, x_in, M):
        phi = logp_S + gumbel(len(logp_S))
        tr = -np.logaddexp(-M, -phi); tr[x_in] = M; return tr - logp_S
    est = []
    for _ in range(n):
        x = []
        for s in range(D): x.append(rng.choice(V, p=cond(x)))
        G = np.stack([posterior(np.log(cond(x[:s])[S[s]]), S[s] == x[s], gumbel(1)[0]) for s in range(D)])
        q = drafter(G); f = 1.0; j = 0.0
        for s in range(D):
            f *= min(1.0, q[s, x[s]] / law(cond(x[:s]), G[s], S[s])[x[s]]); j += f
        est.append(j)
    print(f"E[accepted]: decoding {np.mean(lengths):.4f} ± {np.std(lengths)/np.sqrt(n):.4f} | "
          f"training estimator {np.mean(est):.4f} ± {np.std(est)/np.sqrt(len(est)):.4f}")


noise_round(causal=False, n=150000)
noise_round(causal=True, n=150000)
