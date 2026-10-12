"""Reproduce Figure 6: peak GPU memory and the O(1) KV-cache overhead.

    python -m src.eval.run_memory --model-dir chiennv/Orthrus-Qwen3-8B \
      --context-lengths 1024 4096 16384 40960

The paper's claim (Appendix B) is that the diffusion view adds a CONSTANT KV overhead
because it reads the AR cache in place and only materialises K_diff/V_diff for the
fixed-size parallel block. Analytically, per decoding step:

    bytes = 2 (K and V) x block_size x num_kv_heads x head_dim x dtype_bytes x layers

For Qwen3-8B at K=32, bf16: 2 x 32 x 8 x 128 x 2 x 36 = 4,718,592 B = 4.50 MiB,
which is exactly the "fixed delta ~4.5 MiB" Figure 6(b) reports. This script prints
that prediction alongside a measured peak-memory sweep, so the constant-overhead claim
is checkable both ways.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from src.eval.runner import generate_autoregressive, generate_orthrus, load_model

MIB = 1024.0 * 1024.0


def analytic_diffusion_cache_bytes(config, dtype_bytes: int) -> int:
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return (
        2  # keys and values
        * config.block_size
        * config.num_key_value_heads
        * head_dim
        * dtype_bytes
        * config.num_hidden_layers
    )


def analytic_ar_cache_bytes(config, seq_len: int, dtype_bytes: int) -> int:
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return (
        2 * seq_len * config.num_key_value_heads * head_dim * dtype_bytes
        * config.num_hidden_layers
    )


def measure(model, input_ids, max_new_tokens: int, mode: str) -> dict:
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated() if device.type == "cuda" else 0

    if mode == "orthrus":
        _, passes, seconds = generate_orthrus(model, input_ids, max_new_tokens)
    else:
        eos = getattr(model.config, "eos_token_id", None)
        _, passes, seconds = generate_autoregressive(
            model, input_ids, max_new_tokens, eos_token_id=eos
        )

    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    return {
        "mode": mode,
        "peak_mib": round(peak / MIB, 2),
        "baseline_allocated_mib": round(before / MIB, 2),
        "forward_passes": passes,
        "seconds": round(seconds, 3),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Orthrus memory-footprint sweep (Figure 6).")
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--context-lengths", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--attn-implementation", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2", "flash_attention_3"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="results/memory")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model(
        args.model_dir, dtype=args.dtype,
        attn_implementation=args.attn_implementation, device=args.device,
    )
    dtype_bytes = torch.finfo(model.dtype).bits // 8
    config = model.config

    diffusion_bytes = analytic_diffusion_cache_bytes(config, dtype_bytes)
    print(f"Loaded {args.model_dir} | block_size={config.block_size} | "
          f"layers={config.num_hidden_layers} | kv_heads={config.num_key_value_heads}")
    print(f"\nAnalytic diffusion-view KV overhead (constant in sequence length): "
          f"{diffusion_bytes:,} B = {diffusion_bytes / MIB:.2f} MiB")

    if model.device.type != "cuda":
        print("\nWARNING: not on CUDA -- peak-memory numbers will all be 0. "
              "The analytic overhead above is still valid.")

    rows = []
    for context_length in args.context_lengths:
        # Synthetic context; content is irrelevant to memory.
        input_ids = torch.randint(
            0, config.vocab_size - 1000, (1, context_length), device=model.device
        )
        ar_cache = analytic_ar_cache_bytes(config, context_length, dtype_bytes)
        entry = {
            "context_length": context_length,
            "ar_kv_cache_mib": round(ar_cache / MIB, 2),
            "diffusion_kv_overhead_mib": round(diffusion_bytes / MIB, 2),
            "overhead_fraction": round(diffusion_bytes / ar_cache, 6),
            "autoregressive": measure(model, input_ids, args.max_new_tokens, "ar"),
            "orthrus": measure(model, input_ids, args.max_new_tokens, "orthrus"),
        }
        entry["peak_delta_mib"] = round(
            entry["orthrus"]["peak_mib"] - entry["autoregressive"]["peak_mib"], 2
        )
        rows.append(entry)
        print(f"  L={context_length:>6} | AR KV {entry['ar_kv_cache_mib']:>9.2f} MiB | "
              f"diff overhead {entry['diffusion_kv_overhead_mib']:.2f} MiB "
              f"({100 * entry['overhead_fraction']:.3f}%) | "
              f"peak delta {entry['peak_delta_mib']:>8.2f} MiB", flush=True)

    payload = {
        "model_dir": args.model_dir,
        "block_size": config.block_size,
        "dtype_bytes": dtype_bytes,
        "analytic_diffusion_overhead_bytes": diffusion_bytes,
        "rows": rows,
    }
    with open(os.path.join(args.output_dir, "memory.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote {args.output_dir}/memory.json")


if __name__ == "__main__":
    main()
