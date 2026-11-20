"""GPU checks for noise-conditioned decoding on a real checkpoint.

    python scripts/check_noise_decoding.py --model-dir ck/noise-B/final
    python scripts/check_noise_decoding.py --model-dir $S1 --enable-top-m 32   # untrained

With --enable-top-m on a checkpoint without the noise projection, a zero projection is
attached: the drafter then ignores the noise, which still exercises the whole
verification path (support, revealed laws, causal in-block mask) for losslessness.

1. Losslessness (statistical), two tests on N continuations per sampler:
   a. per-position marginal TV at generated positions [--check-offset, +--check-span),
      i.e. AFTER the near-deterministic thinking preamble ("<think>" ...), against the
      AR-vs-AR noise floor, with the empirical entropy of those positions (a test on
      zero-entropy positions passes trivially);
   b. a z-test on the mean log-likelihood of the whole continuation under the AR
      teacher at the sampling temperature. Any sampler bias (e.g. toward drafter modes)
      shifts it; |z| < 3 is a pass. AR-vs-AR z is printed as a calibration check.
2. Acceptance / speed (after a warm-up): plain decoding, causal in-block mask only, and
   noise decoding. With an untrained (zero) projection "causal only" isolates the cost of
   the mask and "noise" adds the cost of verifying against the revealed laws with a
   drafter that ignores the noise (by convexity of TV it can only lose).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_tree_decoding import PROMPT, first_tokens  # noqa: E402
from src.eval.metrics import ForwardPassCounter  # noqa: E402
from src.eval.runner import build_input_ids, load_model  # noqa: E402


def tv(a: Counter, b: Counter, n: int) -> float:
    return 0.5 * sum(abs(a[k] - b[k]) for k in set(a) | set(b)) / n


def entropy(counts: Counter, n: int) -> float:
    return -sum(c / n * math.log(c / n) for c in counts.values() if c)


@torch.inference_mode()
def continuations(model, input_ids, n_tokens, temperature, samples, use_diffusion):
    return [first_tokens(model, input_ids, n_tokens, temperature, 1,
                         use_diffusion=use_diffusion) for _ in range(samples)]


def position_counts(seqs, offset, span):
    return [Counter(seq[j] for seq in seqs if len(seq) > j) for j in range(offset, offset + span)]


@torch.inference_mode()
def log_likelihoods(model, input_ids, seqs, temperature, batch=64):
    """sum_t log p_T(x_t | prompt, x_<t) under the AR view, per continuation."""
    length = max(len(seq) for seq in seqs)
    full = [seq for seq in seqs if len(seq) == length]       # drop early-EOS samples
    scores = []
    prompt = input_ids[0]
    for start in range(0, len(full), batch):
        chunk = torch.tensor(full[start:start + batch], device=input_ids.device)
        ids = torch.cat([prompt.expand(chunk.shape[0], -1), chunk], dim=1)
        logits = model(input_ids=ids).logits[:, prompt.shape[0] - 1:-1].float() / temperature
        logp = torch.log_softmax(logits, dim=-1).gather(-1, chunk.unsqueeze(-1)).squeeze(-1)
        scores.extend(logp.sum(-1).tolist())
    return scores


def mean_position_tv(group_a, group_b, offset, span):
    ca, cb = position_counts(group_a, offset, span), position_counts(group_b, offset, span)
    n = min(len(group_a), len(group_b))
    return sum(tv(x, y, n) for x, y in zip(ca, cb)) / span


def permutation_p_value(group_a, group_b, offset, span, rounds, seed=0):
    """P(statistic >= observed) under random relabelling of the pooled samples."""
    import random

    observed = mean_position_tv(group_a, group_b, offset, span)
    pooled = list(group_a) + list(group_b)
    rng = random.Random(seed)
    hits = 0
    for _ in range(rounds):
        rng.shuffle(pooled)
        split = pooled[:len(group_a)], pooled[len(group_a):]
        hits += mean_position_tv(*split, offset, span) >= observed
    return observed, (hits + 1) / (rounds + 1)


def z_score(a, b):
    mean = lambda x: sum(x) / len(x)
    var = lambda x: sum((v - mean(x)) ** 2 for v in x) / (len(x) - 1)
    return (mean(a) - mean(b)) / math.sqrt(var(a) / len(a) + var(b) / len(b))


@torch.inference_mode()
def run(model, input_ids, args, noise_m, causal=False):
    model.config.noise_reveal_top_m = noise_m
    model.config.causal_drafting = causal
    for _ in range(2):                    # untimed warm-up at the measured length
        model.generate(input_ids=input_ids, max_new_tokens=args.speed_tokens,
                       temperature=args.temperature, top_k=0, top_p=1.0)
    tokens = passes = seconds = 0.0
    counter = ForwardPassCounter(model)
    for _ in range(args.speed_runs):
        with counter:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            out = model.generate(input_ids=input_ids, max_new_tokens=args.speed_tokens,
                                 temperature=args.temperature, top_k=0, top_p=1.0)
            torch.cuda.synchronize()
        seconds += time.perf_counter() - begin
        tokens += out.shape[1] - input_ids.shape[1]
        passes += counter.count
    per_cycle = 3 if noise_m else 2
    cycles = (passes - args.speed_runs) / per_cycle
    model.config.causal_drafting = False
    return tokens / seconds, tokens / max(cycles, 1) - 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--enable-top-m", type=int, default=None)
    p.add_argument("--check-offset", type=int, default=16,
                   help="Skip this many generated tokens (thinking preamble) for the TV test.")
    p.add_argument("--check-span", type=int, default=8)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--speed-tokens", type=int, default=512)
    p.add_argument("--speed-runs", type=int, default=5)
    args = p.parse_args()

    model, tokenizer = load_model(args.model_dir, dtype=args.dtype, attn_implementation="sdpa")
    m = args.enable_top_m or getattr(model.config, "noise_reveal_top_m", None)
    if not m:
        raise ValueError("checkpoint has no noise projection: pass --enable-top-m M.")
    if not hasattr(model, "noise_proj_diff"):
        hidden = model.config.hidden_size
        model.noise_proj_diff = nn.Linear(2 * hidden, hidden, bias=False).to(
            device=model.device, dtype=model.dtype)
        nn.init.zeros_(model.noise_proj_diff.weight)
    input_ids = build_input_ids(tokenizer, PROMPT).to(model.device)

    n_tokens = args.check_offset + args.check_span
    print(f"== 1. losslessness: {args.samples} continuations of {n_tokens} tokens per sampler, "
          f"noise decoding (top-{m}) vs AR, T={args.temperature}")
    model.config.noise_reveal_top_m = m
    model.config.causal_drafting = False
    ar_a = continuations(model, input_ids, n_tokens, args.temperature, args.samples, False)
    ar_b = continuations(model, input_ids, n_tokens, args.temperature, args.samples, False)
    noise = continuations(model, input_ids, n_tokens, args.temperature, args.samples, True)

    ca, cb, cn = (position_counts(x, args.check_offset, args.check_span)
                  for x in (ar_a, ar_b, noise))
    tv_noise = sum(tv(n_, a_, args.samples) for n_, a_ in zip(cn, ca)) / args.check_span
    tv_floor = sum(tv(b_, a_, args.samples) for b_, a_ in zip(cb, ca)) / args.check_span
    ent = sum(entropy(a_, args.samples) for a_ in ca) / args.check_span
    print(f"a. mean per-position TV over positions [{args.check_offset}, {n_tokens}): "
          f"noise vs AR {tv_noise:.4f} | AR vs AR floor {tv_floor:.4f} | "
          f"mean empirical entropy {ent:.2f} nats")
    if ent < 0.3:
        print("   WARNING: these positions are nearly deterministic; raise --check-offset.")
    # The floor above is one random draw; the permutation test asks how often a random
    # split of the pooled samples gives a TV at least this large (p > 0.01 passes).
    _, p_noise = permutation_p_value(noise, ar_a, args.check_offset, args.check_span,
                                     args.permutations)
    _, p_ar = permutation_p_value(ar_b, ar_a, args.check_offset, args.check_span,
                                  args.permutations, seed=1)
    print(f"   permutation test ({args.permutations} splits): noise vs AR p={p_noise:.3f} | "
          f"AR vs AR p={p_ar:.3f}  (p > 0.01 passes)")

    la = log_likelihoods(model, input_ids, ar_a, args.temperature)
    lb = log_likelihoods(model, input_ids, ar_b, args.temperature)
    ln = log_likelihoods(model, input_ids, noise, args.temperature)
    print(f"b. AR log-likelihood z-test: noise vs AR z={z_score(ln, la):+.2f} | "
          f"AR vs AR z={z_score(lb, la):+.2f}  (|z| < 3 passes; mean logL "
          f"AR {sum(la) / len(la):.2f}, noise {sum(ln) / len(ln):.2f})")

    print("== 2. acceptance and speed")
    # 'causal only' separates the two costs of noise decoding on an untrained projection:
    # the causal in-block mask (the drafter was trained bidirectional) and verifying
    # against the sharper revealed laws with a drafter that ignores the noise.
    # "noise off" runs again at the end: if its two speeds differ a lot, the timing is
    # dominated by order / warm-up effects on this machine, not by the decoding mode.
    for label, setting, causal in (("noise off", None, False),
                                   ("causal only", None, True),
                                   (f"noise top-{m}", m, False),
                                   ("noise off (again)", None, False)):
        speed, accept = run(model, input_ids, args, setting, causal)
        print(f"{label:>18}: accepted/cycle={accept:.2f}  {speed:.1f} tok/s")


if __name__ == "__main__":
    main()
