"""Reproduce Table 1 (TPF + speedup) and the accuracy tables, with a losslessness check.

    python -m src.eval.run_benchmark --model-dir chiennv/Orthrus-Qwen3-8B \
      --tasks gsm8k math500 humaneval --temperature 0 --output-dir results/qwen3-8b

What it measures per task:
  * TPF for the dual-view decoder and for the sequential AR baseline (Eq. 9).
  * Average acceptance length -- accepted draft tokens per consensus cycle, the
    Figure 4 (left) metric. This is NOT the TPF column: MATH-500/Qwen3-8B is TPF
    6.35 against an acceptance length of 11.7. See `metrics.py` for the identity.
  * Measured wall-clock speedup (Table 1's speedup column is not the TPF ratio).
  * Accuracy for both decoders.
  * Whether the two token streams are IDENTICAL at temperature 0 -- the paper's
    central claim (3.3) is exact distribution matching, so any divergence here is a
    bug, not a rounding artefact.

`--no-baseline` skips the AR run when you only want efficiency numbers; TPF for the
baseline is 1.0 by construction in that case, and no losslessness check is possible.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from src.eval.metrics import RunStats, speedup_table
from src.eval.runner import (
    build_input_ids,
    completion_text,
    generate_autoregressive,
    generate_orthrus,
    load_model,
)
from src.eval.tasks import TASKS, load_task_rows, run_code_sample, score_math


def evaluate_task(model, tokenizer, task, args) -> dict:
    rows = load_task_rows(
        task, limit=args.limit, hf_path=args.hf_path, hf_config=args.hf_config
    )
    max_new_tokens = args.max_new_tokens or task.max_new_tokens
    eos_token_id = getattr(model.config, "eos_token_id", None) or tokenizer.eos_token_id

    # passes_per_cycle=2 (diffusion + AR verification) is what makes
    # `acceptance_length` computable; the AR baseline has no cycles.
    orthrus_stats = RunStats("orthrus", passes_per_cycle=2)
    baseline_stats = RunStats("autoregressive")
    orthrus_correct = baseline_correct = 0
    identical = mismatched = 0
    samples = []

    for index, row in enumerate(rows):
        prompt = task.build_prompt(row)
        input_ids = build_input_ids(
            tokenizer, prompt, enable_thinking=args.enable_thinking
        ).to(model.device)
        prompt_len = input_ids.shape[1]

        output_ids, passes, seconds = generate_orthrus(
            model, input_ids, max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        )
        orthrus_tokens = output_ids.shape[1] - prompt_len
        orthrus_stats.add(orthrus_tokens, passes, seconds)
        orthrus_text = completion_text(tokenizer, output_ids, prompt_len)

        baseline_text = None
        if not args.no_baseline:
            base_ids, base_passes, base_seconds = generate_autoregressive(
                model, input_ids, max_new_tokens,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                eos_token_id=eos_token_id,
            )
            baseline_stats.add(base_ids.shape[1] - prompt_len, base_passes, base_seconds)
            baseline_text = completion_text(tokenizer, base_ids, prompt_len)

            if args.temperature < 1e-5:
                a = output_ids[0, prompt_len:].tolist()
                b = base_ids[0, prompt_len:].tolist()
                if a == b:
                    identical += 1
                else:
                    mismatched += 1

        if task.kind == "math":
            gold = task.gold(row)
            orthrus_correct += int(score_math(orthrus_text, gold))
            if baseline_text is not None:
                baseline_correct += int(score_math(baseline_text, gold))
        elif task.kind == "code" and args.allow_code_execution:
            orthrus_correct += int(
                run_code_sample(task.test_program(row, orthrus_text), args.code_timeout)
            )
            if baseline_text is not None:
                baseline_correct += int(
                    run_code_sample(task.test_program(row, baseline_text), args.code_timeout)
                )

        if args.save_samples:
            samples.append({
                "index": index,
                "prompt": prompt,
                "orthrus": orthrus_text,
                "autoregressive": baseline_text,
            })

        if args.verbose and (index + 1) % args.log_every == 0:
            print(f"  [{task.name}] {index + 1}/{len(rows)} | "
                  f"TPF={orthrus_stats.tpf:.2f}", flush=True)

    num_rows = len(rows)
    scored = task.kind == "math" or (task.kind == "code" and args.allow_code_execution)
    result = {
        "task": task.name,
        "num_samples": num_rows,
        "max_new_tokens": max_new_tokens,
        "temperature": args.temperature,
        "block_size": model.config.block_size,
        "orthrus": orthrus_stats.as_dict(),
    }
    if scored:
        result["accuracy_orthrus"] = round(orthrus_correct / max(1, num_rows), 4)
    if not args.no_baseline:
        result["autoregressive"] = baseline_stats.as_dict()
        result["efficiency"] = speedup_table(baseline_stats, orthrus_stats)
        if scored:
            result["accuracy_autoregressive"] = round(baseline_correct / max(1, num_rows), 4)
        if args.temperature < 1e-5:
            result["lossless"] = {
                "identical": identical,
                "mismatched": mismatched,
                "exact_match_rate": round(identical / max(1, identical + mismatched), 4),
            }
    if args.save_samples:
        result["samples"] = samples
    return result


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 104)
    print(f"{'task':<18} {'TPF':>7} {'accept':>7} {'TPF x':>7} {'wall x':>8} "
          f"{'acc(orth)':>10} {'acc(AR)':>9} {'lossless':>18}")
    print("-" * 104)
    for entry in results:
        efficiency = entry.get("efficiency", {})
        lossless = entry.get("lossless")
        lossless_text = "n/a"
        if lossless is not None:
            lossless_text = f"{lossless['identical']}/{lossless['identical'] + lossless['mismatched']} exact"
        print(
            f"{entry['task']:<18} "
            f"{entry['orthrus']['tpf']:>7.2f} "
            f"{entry['orthrus'].get('acceptance_length', float('nan')):>7.2f} "
            f"{efficiency.get('tpf_ratio', float('nan')):>7.2f} "
            f"{efficiency.get('wallclock_speedup', float('nan')):>8.2f} "
            f"{entry.get('accuracy_orthrus', float('nan')):>10.3f} "
            f"{entry.get('accuracy_autoregressive', float('nan')):>9.3f} "
            f"{lossless_text:>18}"
        )
    print("=" * 104)


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce Orthrus Table 1 / accuracy tables.")
    p.add_argument("--model-dir", type=str, required=True,
                   help="Local checkpoint dir or hub id, e.g. chiennv/Orthrus-Qwen3-8B.")
    p.add_argument("--tasks", type=str, nargs="+", default=["gsm8k", "math500"],
                   choices=sorted(TASKS), metavar="TASK")
    p.add_argument("--hf-path", type=str, default=None,
                   help="Override the dataset id (single-task runs only).")
    p.add_argument("--hf-config", type=str, default=None,
                   help="Override the dataset config (single-task runs only). For "
                        "livecodebench_v5 this selects the release, e.g. release_v5 "
                        "(cumulative v1..v5) or v5 (that window only).")
    p.add_argument("--limit", type=int, default=None, help="Cap examples per task.")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override the per-task default.")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="Qwen3 thinking-mode default is 0.6. 0 = greedy: enables the "
                        "losslessness check, but thinking traces loop under greedy "
                        "decoding, which inflates TPF -- use it with --limit as a sanity "
                        "check only.")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--block-size", type=int, default=None,
                   help="Override K at inference (Section 5 ablation).")
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false",
                   help="Thinking mode is the default; this disables it.")
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip the AR run (no speedup or losslessness numbers).")
    p.add_argument("--allow-code-execution", action="store_true",
                   help="Required to score HumanEval/MBPP; runs model output locally.")
    p.add_argument("--code-timeout", type=float, default=10.0)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--attn-implementation", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2", "flash_attention_3"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--save-samples", action="store_true")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()
    if (args.hf_path or args.hf_config) and len(args.tasks) > 1:
        raise ValueError(
            "--hf-path / --hf-config apply to a single task; pass one --tasks entry."
        )
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model(
        args.model_dir, dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device=args.device, block_size=args.block_size,
    )
    print(f"Loaded {args.model_dir} | block_size={model.config.block_size} | "
          f"device={model.device} | dtype={args.dtype}")

    results = []
    for task_name in args.tasks:
        print(f"\n=== {task_name} ===", flush=True)
        task = TASKS[task_name]
        if task.kind == "code" and not args.allow_code_execution:
            print(f"  skipping accuracy for {task_name}: pass --allow-code-execution "
                  f"to execute generated programs (efficiency is still measured).")
        elif task.kind == "generation":
            print(f"  {task_name} is efficiency-only: the paper reports it in Table 1 "
                  f"(TPF / speedup) and in no accuracy table. TPF, wall-clock speedup "
                  f"and the losslessness check are still measured.")
        entry = evaluate_task(model, tokenizer, task, args)
        results.append(entry)
        with open(os.path.join(args.output_dir, f"{task_name}.json"), "w") as handle:
            json.dump(entry, handle, indent=2)

    print_summary(results)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump({"model_dir": args.model_dir, "results": results}, handle, indent=2)
    print(f"\nWrote {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
