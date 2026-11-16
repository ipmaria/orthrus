"""AR baseline, once, through vLLM.

The Orthrus AR path is the frozen base model (freeze_to_diffusion trains only the
`*_diff` twins, init_model_qwen3 copies the AR weights byte-for-byte), so its accuracy is
the same for every checkpoint built from the same init. Run this once, then evaluate each
checkpoint with `run_benchmark.py --no-baseline`.

Prompts and scoring are taken from src/eval/tasks.py unchanged (MATH_INSTRUCTION, the
chat-template call with enable_thinking, score_math), so the number lands in the same
column as `accuracy_autoregressive` in run_benchmark's JSON.

    # from the repo root, same base model you passed to init_model_qwen3
    python scripts/eval_ar_baseline_vllm.py --model Qwen/Qwen3-0.6B \
        --tasks gsm8k math500 --n 4 --output-dir results/ar-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from src.eval.tasks import TASKS, load_task_rows, score_math


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-0.6B",
                   help="The base model the Orthrus init was built from (vLLM cannot load "
                        "the Orthrus checkpoint itself, and does not need to).")
    p.add_argument("--tasks", nargs="+", default=["gsm8k", "math500"],
                   choices=[n for n, t in TASKS.items() if t.kind == "math"])
    p.add_argument("--limit", type=int, default=None, help="Cap examples per task.")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override the per-task default (4096 gsm8k, 8192 math500, ...).")
    p.add_argument("--n", type=int, default=1,
                   help="Samples per problem; accuracy is averaged over them. Cheap here, "
                        "and it tames the sampling noise at T=0.6.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false",
                   help="Thinking mode is the default, as in run_benchmark.py.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                   help="On a shared GPU lower it (0.5): vLLM refuses to start if less than "
                        "this fraction of the card is free at startup.")
    p.add_argument("--kv-cache-gb", type=float, default=None,
                   help="Fixed KV-cache size in GiB (vLLM's kv_cache_memory_bytes). Skips the "
                        "free-memory profiling that fails when another process on the GPU "
                        "allocates or frees memory during startup. 16-24 is plenty for a 0.6B "
                        "model (~112 KB per token). Overrides --gpu-memory-utilization for "
                        "sizing the cache.")
    p.add_argument("--output-dir", default="results/ar-baseline")
    p.add_argument("--save-samples", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    # FlashInfer JIT-builds its top-k/top-p sampling kernel on first use and needs nvcc
    # for that; vLLM's native sampler is as good for a 0.6B model. `export
    # VLLM_USE_FLASHINFER_SAMPLER=1` overrides this if nvcc is available.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Prompt tokens exactly as src/eval/runner.py::build_input_ids makes them.
    jobs = {}
    for name in args.tasks:
        task = TASKS[name]
        rows = load_task_rows(task, limit=args.limit)
        prompt_ids = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": task.build_prompt(row)}],
                tokenize=True, return_dict=True, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )["input_ids"]
            for row in rows
        ]
        jobs[name] = (task, rows, prompt_ids, args.max_new_tokens or task.max_new_tokens)

    max_model_len = max(
        max(len(ids) for ids in prompt_ids) + max_new_tokens
        for _, _, prompt_ids, max_new_tokens in jobs.values()
    )
    engine_kwargs = {}
    if args.kv_cache_gb is not None:
        engine_kwargs["kv_cache_memory_bytes"] = int(args.kv_cache_gb * 2**30)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enable_prefix_caching=True, seed=args.seed, **engine_kwargs)
    stop_ids = sorted({tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id})

    for name, (task, rows, prompt_ids, max_new_tokens) in jobs.items():
        params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                                top_k=args.top_k, max_tokens=max_new_tokens,
                                stop_token_ids=stop_ids, seed=args.seed)
        outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], params)

        correct = total = truncated = generated = 0
        samples = []
        for row, out in zip(rows, outputs):
            gold = task.gold(row)
            for completion in out.outputs:
                ok = score_math(completion.text, gold)
                correct += int(ok)
                total += 1
                truncated += int(completion.finish_reason != "stop")
                generated += len(completion.token_ids)
                if args.save_samples:
                    samples.append({"prompt": task.build_prompt(row), "gold": gold,
                                    "completion": completion.text, "correct": ok})

        result = {
            "task": name,
            "model": args.model,
            "num_samples": len(rows),
            "n_per_prompt": args.n,
            "max_new_tokens": max_new_tokens,
            "temperature": args.temperature,
            "accuracy_autoregressive": round(correct / max(1, total), 4),
            "truncated_fraction": round(truncated / max(1, total), 4),
            "mean_generated_tokens": round(generated / max(1, total), 1),
        }
        if args.save_samples:
            result["samples"] = samples
        print(f"{name}: accuracy={result['accuracy_autoregressive']:.4f} | "
              f"truncated={result['truncated_fraction']:.1%} | "
              f"mean_tokens={result['mean_generated_tokens']}", flush=True)
        with open(os.path.join(args.output_dir, f"{name}.json"), "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
