"""GPU checks for noise-conditioned decoding on a real checkpoint.

    python scripts/check_noise_decoding.py --model-dir ck/noise-B/final
    python scripts/check_noise_decoding.py --model-dir $S1 --enable-top-m 32   # untrained

With --enable-top-m on a checkpoint without the noise projection, a zero projection is
attached: the drafter then ignores the noise, which still exercises the whole
verification path (support, revealed laws, causal in-block mask) for losslessness.

1. Losslessness (statistical): first --check-tokens generated tokens, noise decoding vs
   pure AR sampling, with the AR-vs-AR noise floor.
2. Acceptance / speed: tokens per cycle and tok/s with noise decoding on and off.
"""

from __future__ import annotations

import argparse
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


@torch.inference_mode()
def run(model, input_ids, args, noise_m):
    model.config.noise_reveal_top_m = noise_m
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
    return tokens / seconds, tokens / max(cycles, 1) - 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--enable-top-m", type=int, default=None)
    p.add_argument("--check-tokens", type=int, default=2)
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

    print(f"== 1. first {args.check_tokens} tokens: noise decoding (top-{m}) vs AR")
    model.config.noise_reveal_top_m = m
    ar = lambda: Counter(first_tokens(model, input_ids, args.check_tokens, args.temperature,
                                      1, use_diffusion=False) for _ in range(args.samples))
    reference, again = ar(), ar()
    noise = Counter(first_tokens(model, input_ids, args.check_tokens, args.temperature, 1,
                                 use_diffusion=True) for _ in range(args.samples))
    print(f"TV(noise, AR)={tv(noise, reference, args.samples):.4f}   "
          f"TV(AR, AR) noise floor={tv(again, reference, args.samples):.4f}")

    print("== 2. acceptance and speed")
    for label, setting in (("noise off", None), (f"noise top-{m}", m)):
        speed, accept = run(model, input_ids, args, setting)
        print(f"{label:>14}: accepted/cycle={accept:.2f}  {speed:.1f} tok/s")


if __name__ == "__main__":
    main()
