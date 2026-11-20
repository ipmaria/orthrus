"""Gradient variance of the stage-2 estimators on the real model (single GPU).

Uses the trainer's own `compute_objective_loss`, so the gradient measured is exactly
the one training would apply. All trainer flags are accepted (objective, estimator,
drafts, baseline, temperature, anchors, chunk size, ...); the flags below are added.

    python scripts/grad_variance.py --model-dir checkpoints/orthrus-qwen3-1.7b-kl/final \
        --packed-cache-path data/A-packed --seq-len 2048 \
        --objective accepted-length --eal-estimator rb --eal-num-drafts 4 \
        --repeats 16 --num-batches 4 --vary sampling --out results/gradvar-rb4.json

What varies between the `--repeats` gradients of one batch:
  sampling  fixed rows and anchors; only drafts / accept coins are resampled. The pure
            estimator noise (zero for target-path, lk, wtv, streak, expected-rounds).
  anchors   fixed rows; anchors and drafts are resampled.
  all       a different micro-batch for every repeat: total minibatch gradient noise,
            the fair axis for comparing sampled estimators with target-path on
            teacher rollouts (whose only randomness IS the data).

Reported per batch and averaged: tr Var[g] (unbiased, summed over the selected
parameters), |E g|^2, SNR = |E g|^2 / tr Var[g], relative noise tr Var / |E g|^2, mean
per-sample |g|, and mean / std of the loss value. Accumulation is Welford in fp32 on the
device, so memory is 2x the selected parameters; restrict with --param-filter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configuration import OrthrusConfig  # noqa: E402
from src.model import OrthrusDecoderLayer, OrthrusLM  # noqa: E402
from src.train.train_qwen3 import compute_objective_loss, parse_args  # noqa: E402
from src.utils.data_utils import create_packed_dataloader  # noqa: E402
from src.utils.init_model_qwen3 import freeze_to_diffusion  # noqa: E402
from src.utils.logging import mix_seed  # noqa: E402


def own_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--repeats", type=int, default=16, help="Gradients per batch (R >= 2).")
    p.add_argument("--num-batches", type=int, default=4)
    p.add_argument("--vary", choices=["sampling", "anchors", "all"], default="sampling")
    p.add_argument("--param-filter", type=str, default=None,
                   help="Regex on parameter names; default = every trainable parameter.")
    p.add_argument("--out", type=str, default=None, help="JSON output path.")
    return p.parse_known_args()


class Welford:
    def __init__(self, params):
        self.params = params
        self.count = 0
        self.mean = [torch.zeros_like(q, dtype=torch.float32) for q in params]
        self.m2 = [torch.zeros_like(q, dtype=torch.float32) for q in params]
        self.norms = []

    def add(self):
        self.count += 1
        squared_norm = 0.0
        for param, mean, m2 in zip(self.params, self.mean, self.m2):
            grad = (param.grad if param.grad is not None
                    else torch.zeros_like(param)).float()
            squared_norm += float(grad.pow(2).sum())
            delta = grad - mean
            mean.add_(delta / self.count)
            m2.add_(delta * (grad - mean))
        self.norms.append(math.sqrt(squared_norm))

    def summary(self):
        trace_var = sum(float(m2.sum()) for m2 in self.m2) / max(1, self.count - 1)
        mean_sq = sum(float(mean.pow(2).sum()) for mean in self.mean)
        # |mean|^2 overestimates |E g|^2 by tr Var / R; report the debiased value too.
        mean_sq_debiased = mean_sq - trace_var / self.count
        return {
            "repeats": self.count,
            "trace_var": trace_var,
            "mean_grad_sq": mean_sq,
            "mean_grad_sq_debiased": mean_sq_debiased,
            "snr": mean_sq_debiased / trace_var if trace_var > 0 else float("inf"),
            "relative_noise": trace_var / mean_sq_debiased if mean_sq_debiased > 0
            else float("inf"),
            "mean_sample_grad_norm": sum(self.norms) / len(self.norms),
        }


def main():
    mine, rest = own_args()
    if mine.repeats < 2:
        raise ValueError("--repeats must be >= 2.")
    sys.argv = [sys.argv[0], *rest]
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The diffusion pass needs CUDA (FlexAttention).")
    device = torch.device("cuda")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    cfg = OrthrusConfig.from_pretrained(args.model_dir)
    cfg.use_cache = False
    cfg._attn_implementation = args.attn_implementation
    model = OrthrusLM.from_pretrained(args.model_dir, config=cfg, dtype=dtype).to(device)
    freeze_to_diffusion(model)
    if args.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
        )
        apply_activation_checkpointing(
            model, check_fn=lambda module: isinstance(module, OrthrusDecoderLayer)
        )
    model.train()

    pattern = re.compile(mine.param_filter) if mine.param_filter else None
    named = [(name, param) for name, param in model.named_parameters()
             if param.requires_grad and (pattern is None or pattern.search(name))]
    if not named:
        raise ValueError("no trainable parameter matches --param-filter.")
    params = [param for _, param in named]
    print(f"measuring {sum(q.numel() for q in params):,} parameters in {len(params)} "
          f"tensors | objective={args.objective} estimator={args.eal_estimator} "
          f"drafts={args.eal_num_drafts} baseline={args.eal_baseline} vary={mine.vary}")

    loader = create_packed_dataloader(
        cache_path=args.packed_cache_path, seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size, num_workers=0, distributed=False,
        shuffle=True, seed=args.seed,
        require_assistant_mask=not args.allow_missing_assistant_mask,
    )
    batches = iter(loader)
    use_amp = dtype in (torch.float16, torch.bfloat16)
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if use_amp else nullcontext
    anchor_gen = torch.Generator(device=device)
    sample_gen = torch.Generator(device=device)

    def to_device(batch):
        ids = batch["input_ids"].to(device)
        mask = batch.get("assistant_mask")
        return ids, (mask.to(device) if mask is not None else None)

    groups = 1 if mine.vary == "all" else mine.num_batches
    results = []
    for group in range(groups):
        stats = Welford(params)
        losses = []
        fixed = None if mine.vary == "all" else to_device(next(batches))
        for repeat in range(mine.repeats):
            input_ids, assistant_mask = fixed if fixed is not None else to_device(next(batches))
            anchor_seed = mix_seed(args.seed, group, 0, 0 if mine.vary == "sampling" else repeat)
            anchor_gen.manual_seed(anchor_seed)
            sample_gen.manual_seed(mix_seed(args.seed + 1, group, 1, repeat))
            model.zero_grad(set_to_none=True)
            with amp():
                loss, _ = compute_objective_loss(
                    args, model, model, cfg, input_ids, assistant_mask, anchor_gen,
                    sample_gen=sample_gen,
                )
            loss.backward()
            losses.append(float(loss))
            stats.add()
        summary = stats.summary()
        mean_loss = sum(losses) / len(losses)
        summary["loss_mean"] = mean_loss
        summary["loss_std"] = math.sqrt(
            sum((x - mean_loss) ** 2 for x in losses) / (len(losses) - 1)
        )
        results.append(summary)
        print(f"[group {group}] trVar={summary['trace_var']:.4e} "
              f"|Eg|^2={summary['mean_grad_sq_debiased']:.4e} SNR={summary['snr']:.4f} "
              f"loss={mean_loss:.4f}±{summary['loss_std']:.4f}", flush=True)

    keys = [k for k in results[0] if isinstance(results[0][k], float)]
    average = {k: sum(r[k] for r in results) / len(results) for k in keys}
    print("average:", json.dumps(average, indent=2))
    if mine.out:
        os.makedirs(os.path.dirname(mine.out) or ".", exist_ok=True)
        with open(mine.out, "w") as handle:
            json.dump({"config": {**vars(args), **vars(mine)}, "groups": results,
                       "average": average}, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
