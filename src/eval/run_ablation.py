"""Reproduce the Section 5 block-size ablation: TPF and accuracy as K varies.

    python -m src.eval.run_ablation --model-dir chiennv/Orthrus-Qwen3-8B \
      --task gsm8k --block-sizes 4 8 16 32 --limit 200

The model is loaded once and `config.block_size` is mutated between sweeps -- the
decoding loop reads it per call (`src/model.py:418`), so no reload is needed.

Caveat worth stating: the diffusion view was distilled at one K (32 for the released
checkpoints). Decoding at K' < K is well-defined -- the block mask is block-diagonal
and RoPE uses absolute positions -- but K' > K extrapolates past what was trained, so
falling TPF there is expected and is not evidence against the method.
"""

from __future__ import annotations

import argparse
import json
import os

from src.eval.run_benchmark import evaluate_task
from src.eval.runner import load_model
from src.eval.tasks import TASKS


class _EvalArgs:
    """Minimal stand-in for run_benchmark's argparse namespace."""

    def __init__(self, source, **overrides):
        for key, value in vars(source).items():
            setattr(self, key, value)
        for key, value in overrides.items():
            setattr(self, key, value)


def parse_args():
    p = argparse.ArgumentParser(description="Orthrus block-size (K) ablation.")
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--task", type=str, default="gsm8k", choices=sorted(TASKS))
    p.add_argument("--block-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--allow-code-execution", action="store_true")
    p.add_argument("--code-timeout", type=float, default=10.0)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--attn-implementation", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2", "flash_attention_3"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="results/ablation")
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model(
        args.model_dir, dtype=args.dtype,
        attn_implementation=args.attn_implementation, device=args.device,
    )
    trained_block_size = model.config.block_size
    task = TASKS[args.task]
    print(f"Loaded {args.model_dir} | checkpoint block_size={trained_block_size}")

    rows = []
    for block_size in args.block_sizes:
        model.config.block_size = block_size
        model.model.block_size = block_size
        # The AR baseline is identical across K, so run it only once (at the first K)
        # and reuse it for the speedup column.
        eval_args = _EvalArgs(
            args,
            no_baseline=(block_size != args.block_sizes[0]),
            save_samples=False,
            hf_path=None,
            hf_config=None,
        )
        entry = evaluate_task(model, tokenizer, task, eval_args)
        entry["block_size"] = block_size
        entry["extrapolating_past_trained_K"] = block_size > trained_block_size
        rows.append(entry)
        print(f"  K={block_size:>3} | TPF={entry['orthrus']['tpf']:.2f} | "
              f"acc={entry.get('accuracy_orthrus', float('nan')):.3f}"
              + ("  (K > trained K)" if block_size > trained_block_size else ""),
              flush=True)

    model.config.block_size = trained_block_size
    model.model.block_size = trained_block_size

    payload = {
        "model_dir": args.model_dir,
        "task": args.task,
        "trained_block_size": trained_block_size,
        "rows": rows,
    }
    path = os.path.join(args.output_dir, f"block_size_{args.task}.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
