"""GPU checks for tree decoding (`generate(num_drafts=N)`) on a real checkpoint.

    python scripts/check_tree_decoding.py --model-dir checkpoints/orthrus-qwen3-1.7b/final

1. KV commit: after one tree round, the cache entries copied from the packed
   verification pass must match a plain AR forward over prompt + accepted tokens.
2. Losslessness (statistical): the distribution of the first --check-tokens generated
   tokens with num_drafts=N vs pure AR sampling, same temperature, no top-k/top-p.
   Reports TV distance and the TV expected from sampling noise alone.
3. Speed: tokens/s and tokens per cycle for each N in --num-drafts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.runner import build_input_ids, load_model  # noqa: E402

PROMPT = "Natalia sold clips to 48 of her friends in April, and then she sold half as " \
         "many clips in May. How many clips did Natalia sell altogether in April and May?"


def temperature_sampler(temperature):
    def sample(logits):
        probs = F.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(
            probs.shape[:-1]), probs
    return sample


@torch.inference_mode()
def check_kv_commit(model, input_ids, num_drafts, temperature):
    cache = DynamicCache(config=model.config)
    n = input_ids.shape[1]
    out = model(input_ids=input_ids[:, :-1], past_key_values=cache, use_cache=True)
    start = n - 1
    block_size = model.config.block_size
    positions = torch.arange(start, start + block_size, device=input_ids.device)[None]
    block = torch.full((1, block_size), model.config.mask_token_id, device=input_ids.device)
    block[:, 0] = input_ids[:, -1]
    diff = model(input_ids=block, position_ids=positions, past_key_values=cache,
                 use_cache=False, is_diffusion_pass=True, ar_seq_len=start)
    draft_probs = F.softmax(diff.logits[0, :-1].float() / temperature, dim=-1)
    accepted, _, proposed = model._tree_verify_round(
        input_ids[:, -1:], draft_probs, positions, cache, start, num_drafts,
        temperature_sampler(temperature),
    )
    committed = torch.cat([input_ids[:, :-1], proposed[:, :accepted + 1]], dim=1)
    reference = DynamicCache(config=model.config)
    model(input_ids=committed, past_key_values=reference, use_cache=True)
    worst = 0.0
    for layer, ref_layer in zip(cache.layers, reference.layers):
        assert layer.keys.shape == ref_layer.keys.shape, (layer.keys.shape, ref_layer.keys.shape)
        worst = max(worst, float((layer.keys - ref_layer.keys).abs().max()),
                    float((layer.values - ref_layer.values).abs().max()))
    return accepted, worst, committed.shape[1]


@torch.inference_mode()
def first_tokens(model, input_ids, k, temperature, num_drafts, use_diffusion):
    if use_diffusion:
        out = model.generate(input_ids=input_ids, max_new_tokens=k, temperature=temperature,
                             top_k=0, top_p=1.0, num_drafts=num_drafts)
    else:
        out = model.generate(input_ids=input_ids, max_new_tokens=k, temperature=temperature,
                             top_k=0, top_p=1.0, do_sample=True, use_diffusion_mode=False)
    return tuple(out[0, input_ids.shape[1]:input_ids.shape[1] + k].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num-drafts", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--check-tokens", type=int, default=2)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--speed-tokens", type=int, default=512)
    p.add_argument("--speed-runs", type=int, default=5)
    args = p.parse_args()

    model, tokenizer = load_model(args.model_dir, dtype=args.dtype, attn_implementation="sdpa")
    input_ids = build_input_ids(tokenizer, PROMPT).to(model.device)
    tree_sizes = [n for n in args.num_drafts if n > 1] or [2]

    print("== 1. KV commit vs plain AR forward")
    for n in tree_sizes:
        for trial in range(3):
            accepted, worst, length = check_kv_commit(model, input_ids, n, args.temperature)
            print(f"N={n} trial={trial} accepted={accepted} cache_len={length} "
                  f"max|dKV|={worst:.3e}")

    print(f"== 2. first {args.check_tokens} tokens: tree vs AR, {args.samples} samples each")
    reference = Counter(first_tokens(model, input_ids, args.check_tokens, args.temperature,
                                     1, use_diffusion=False) for _ in range(args.samples))
    for n in tree_sizes:
        tree = Counter(first_tokens(model, input_ids, args.check_tokens, args.temperature,
                                    n, use_diffusion=True) for _ in range(args.samples))
        keys = set(reference) | set(tree)
        tv = 0.5 * sum(abs(tree[k] - reference[k]) for k in keys) / args.samples
        # Noise floor: TV between two independent AR sample sets of the same size.
        again = Counter(first_tokens(model, input_ids, args.check_tokens, args.temperature,
                                     1, use_diffusion=False) for _ in range(args.samples))
        floor = 0.5 * sum(abs(again[k] - reference[k])
                          for k in set(reference) | set(again)) / args.samples
        print(f"N={n}: TV(tree, AR)={tv:.4f}   TV(AR, AR) noise floor={floor:.4f}")

    print("== 3. speed")
    for n in args.num_drafts:
        tokens = seconds = 0.0
        for _ in range(args.speed_runs):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            out = model.generate(input_ids=input_ids, max_new_tokens=args.speed_tokens,
                                 temperature=args.temperature, top_k=0, top_p=1.0,
                                 num_drafts=n)
            torch.cuda.synchronize()
            seconds += time.perf_counter() - begin
            tokens += out.shape[1] - input_ids.shape[1]
        print(f"N={n}: {tokens / seconds:.1f} tok/s over {int(tokens)} tokens")


if __name__ == "__main__":
    main()
