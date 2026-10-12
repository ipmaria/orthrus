"""Train the Orthrus diffusion view on a frozen Qwen3 backbone (paper 3.2 / Eq. 7).

    torchrun --nproc_per_node=8 -m src.train.train_qwen3 \
      --model-dir pretrained-models/orthrus-qwen3-8b-init \
      --packed-cache-path data/processed/orthrus-packed-2048-asst \
      --output-dir checkpoints/orthrus-qwen3-8b --fsdp2 --activation-checkpointing

Companion to the official `src/train/train.py`, which trains the Qwen3.5 hybrid
backbone with hard-label cross-entropy. Two differences, both deliberate:

  * It drives `src/model.py` -- the Qwen3 model the paper benchmarks and the released
    `chiennv/Orthrus-Qwen3-*` checkpoints actually run.
  * `--objective kl` (default) minimises the forward KL against the frozen AR head
    (Eq. 7). `--objective ce` reproduces the Table 5 ablation arm.

Defaults follow Table 4 (L=2048, 256 anchors, K=32, 2 epochs, lr 2e-4, warmup 0.05,
micro-batch 1 x grad-accum 16 x 8 ranks = global batch 128), not the official script's
smoke-run defaults.
"""

from __future__ import annotations

import argparse
import os

# BEFORE torch initializes its CUDA allocator: the diffusion pass fragments the pool
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

import src.utils.distributed as dist_utils
from src.configuration import OrthrusConfig
from src.model import OrthrusDecoderLayer, OrthrusLM
from src.train.checkpoint import (
    latest_checkpoint_dir,
    load_trainer_state,
    resume_batch_idx_is_valid,
    save_checkpoint,
    save_final,
)
from src.train.dual_pass import dual_pass_forward
from src.train.loss_kl import fused_linear_cross_entropy, fused_linear_forward_kl
from src.utils.data_utils import create_packed_dataloader

try:
    # Prefer the official implementation so the two cannot diverge...
    from src.utils.init_model import sample_batch_anchors
except ImportError:
    # ...but that module imports the Qwen3.5 backbone, which hard-imports `fla`.
    from src.utils.anchors import sample_batch_anchors

from src.utils.init_model_qwen3 import (
    copy_diff_from_ar,
    freeze_to_diffusion,
    write_release_style_config,
)
from src.utils.logging import (
    configure_run_logger,
    format_human_count,
    get_cosine_lr_scale,
    log_main,
    mix_seed,
)

try:
    import wandb
except ImportError:
    wandb = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _lm_head_weight(base_model):
    """`lm_head.weight`, following the tie to `embed_tokens` on the 1.7B/4B configs."""
    weight = base_model.lm_head.weight
    if getattr(base_model.config, "tie_word_embeddings", False):
        weight = base_model.model.embed_tokens.weight
    if hasattr(weight, "device_mesh"):  # a still-sharded DTensor
        raise RuntimeError(
            "lm_head.weight is sharded. The loss applies it outside the model's "
            "forward, so it must be replicated -- shard the decoder layers only "
            "(see shard_decoder_layers)."
        )
    return weight


def shard_decoder_layers(model, activation_checkpointing: bool) -> list:
    """FSDP2 over the decoder layers only, leaving root parameters replicated.

    `dist_utils.apply_fsdp2` also calls `fully_shard(model)` at the root, which shards
    `embed_tokens` / `lm_head` / `norm`. The official Qwen3.5 trainer can live with that
    because it computes its loss *inside* `forward`, where FSDP has unsharded the
    parameters. This trainer keeps `src/model.py` untouched and so computes the loss
    after `forward` returns, by which point root parameters are resharded and
    `lm_head.weight` is an unusable DTensor shard.

    Leaving the root replicated costs one copy of the frozen embedding / head per rank
    (~2.5 GB for Qwen3-8B) and nothing in optimizer state, since every root parameter is
    frozen. All trainable parameters live inside the decoder layers, so their gradients
    are still reduce-scattered by FSDP.

    Returns the sharded modules, for `set_requires_gradient_sync` during accumulation.
    """
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed._tensor import DeviceMesh
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
    )

    if activation_checkpointing:
        apply_activation_checkpointing(
            model, check_fn=lambda submodule: isinstance(submodule, OrthrusDecoderLayer)
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    mesh = DeviceMesh(device_type="cuda", mesh=list(range(world_size)))
    config = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
        ),
        "mesh": mesh,
        "reshard_after_forward": True,
    }
    sharded = []
    for module in model.modules():
        if isinstance(module, OrthrusDecoderLayer):
            fully_shard(module, **config)
            sharded.append(module)
    return sharded


@record
def train(args):
    distributed, rank, local_rank, world_size = dist_utils.setup_distributed(
        args.dist_timeout_minutes
    )
    is_main = rank == 0
    log_file = None
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, "train.log")
    configure_run_logger(is_main, log_file)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    resume_dir = args.resume_from or (
        latest_checkpoint_dir(args.output_dir) if args.auto_resume else None
    )
    if resume_dir is not None and not os.path.isdir(resume_dir):
        raise FileNotFoundError(f"--resume-from does not exist: {resume_dir}")
    weights_dir = resume_dir or args.model_dir
    if resume_dir:
        log_main(is_main, f"Resuming from {resume_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    cfg = OrthrusConfig.from_pretrained(weights_dir)
    cfg.use_cache = False
    cfg._attn_implementation = args.attn_implementation
    if cfg.block_size is None or cfg.mask_token_id is None:
        raise ValueError(
            f"{weights_dir}/config.json is missing block_size or mask_token_id; build the "
            f"init checkpoint with `python -m src.utils.init_model_qwen3`."
        )
    model = OrthrusLM.from_pretrained(weights_dir, config=cfg, dtype=dtype)

    if resume_dir is None:
        copy_diff_from_ar(model)
        log_main(is_main, "Warm-started diffusion twins from frozen AR weights.")
    n_train, n_total = freeze_to_diffusion(model)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    model = model.to(device)

    fsdp_modules: list = []
    if args.fsdp2:
        if not distributed:
            raise ValueError("--fsdp2 requires a torchrun launch (WORLD_SIZE > 1).")
        fsdp_modules = shard_decoder_layers(model, args.activation_checkpointing)
        log_main(is_main, f"FSDP2 initialized over {len(fsdp_modules)} decoder layers "
                          f"(root parameters left replicated for the loss).")
    elif distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    base_model = model.module if (distributed and not args.fsdp2) else model

    wandb_run = None
    if args.wandb and is_main:
        if wandb is None:
            raise ImportError("wandb is not installed; `pip install wandb` or drop --wandb.")
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                               config=vars(args))

    block_size = cfg.block_size
    log_main(is_main, "=" * 88)
    log_main(is_main, f"Orthrus-Qwen3 training | model_dir={args.model_dir} | "
                      f"world_size={world_size} | dtype={args.dtype} | "
                      f"attn={args.attn_implementation}")
    log_main(is_main, f"Objective | {args.objective.upper()}" + (
        "  (Eq. 7: forward KL vs the frozen AR teacher)" if args.objective == "kl"
        else "  (Table 5 ablation arm: hard-label cross-entropy)"))
    log_main(is_main, f"Config | seq_len={args.seq_len} | block_size={block_size} | "
                      f"micro_bsz={args.micro_batch_size} | grad_accum={args.grad_accum_steps} | "
                      f"anchors/seq={args.num_anchor_blocks} | lr={args.lr:.2e}")
    log_main(is_main, f"Supervised positions/row = {args.num_anchor_blocks * (block_size - 1):,} "
                      f"(num_anchor_blocks x (block_size - 1))")
    tokens_per_step = (args.seq_len * args.micro_batch_size * args.grad_accum_steps * world_size)
    log_main(is_main, f"Global tokens/step = {tokens_per_step:,} | "
                      f"global batch = {args.micro_batch_size * args.grad_accum_steps * world_size}")
    log_main(is_main, f"Params | total={format_human_count(total_params)} | "
                      f"trainable={format_human_count(trainable_params)} "
                      f"({100.0 * trainable_params / max(1, total_params):.2f}%) | "
                      f"{n_train}/{n_total} tensors")
    log_main(is_main, "=" * 88)

    loader = create_packed_dataloader(
        cache_path=args.packed_cache_path, seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size, num_workers=args.num_workers,
        distributed=distributed, shuffle=True, seed=args.seed,
        require_assistant_mask=not args.allow_missing_assistant_mask,
    )
    sampler = loader.sampler

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    updates_per_epoch = max(1, len(loader) // args.grad_accum_steps)
    total_steps = updates_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda cur: get_cosine_lr_scale(cur, warmup_steps, total_steps),
    )
    log_main(is_main, f"LR schedule | cosine | warmup={warmup_steps} | total={total_steps}")

    step, start_epoch, resume_batch_idx = 0, 0, -1
    if resume_dir is not None:
        step, start_epoch, resume_batch_idx, saved_world, saved_ga = load_trainer_state(
            resume_dir, model, optimizer, scheduler, args.fsdp2, distributed
        )
        log_main(is_main, f"Restored | step={step} epoch={start_epoch + 1} "
                          f"after batch_idx={resume_batch_idx}")
        if not resume_batch_idx_is_valid(saved_world, saved_ga, world_size, args.grad_accum_steps):
            log_main(is_main,
                     f"WARNING: checkpoint was written at world_size={saved_world or '?'} "
                     f"grad_accum={saved_ga or '?'} but this run is world_size={world_size} "
                     f"grad_accum={args.grad_accum_steps}. The per-rank batch_idx does not "
                     f"transfer; restarting this epoch from batch 0 (step/LR preserved).")
            resume_batch_idx = -1

    if args.seq_len < block_size + 1:
        raise ValueError(f"seq_len {args.seq_len} too short for block_size {block_size}.")

    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    amp_context = (lambda: torch.autocast(device_type="cuda", dtype=dtype)) if use_amp \
        else nullcontext
    anchor_gen = torch.Generator(device=device)

    train_start = last_log = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        epoch_iter = (tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
                      if (tqdm is not None and is_main) else loader)

        for batch_idx, batch in enumerate(epoch_iter):
            if epoch == start_epoch and batch_idx <= resume_batch_idx:
                continue
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            assistant_mask = batch.get("assistant_mask")
            if assistant_mask is not None:
                assistant_mask = assistant_mask.to(device, non_blocking=True)

            with amp_context():
                anchor_gen.manual_seed(mix_seed(args.seed, epoch, rank, batch_idx))
                anchors, anchor_valid = sample_batch_anchors(
                    input_ids, block_size, args.num_anchor_blocks,
                    generator=anchor_gen, assistant_mask=assistant_mask,
                )

                student, teacher, keep_mask, labels = dual_pass_forward(
                    model,
                    input_ids=input_ids,
                    anchors=anchors,
                    block_size=block_size,
                    mask_token_id=cfg.mask_token_id,
                    anchor_valid=anchor_valid,
                    supervise_mask=assistant_mask,
                )

                head_weight = _lm_head_weight(base_model)
                if args.objective == "kl":
                    loss = fused_linear_forward_kl(
                        head_weight, student, teacher, keep_mask,
                        chunk_size=args.loss_chunk_size,
                    )
                else:
                    loss = fused_linear_cross_entropy(
                        head_weight, student, labels, chunk_size=args.loss_chunk_size,
                    )
                loss = loss / args.grad_accum_steps

            is_sync_step = (batch_idx + 1) % args.grad_accum_steps == 0
            if distributed and args.fsdp2:
                # Root is not an FSDPModule here, so arm each sharded layer directly.
                for sharded_module in fsdp_modules:
                    sharded_module.set_requires_gradient_sync(is_sync_step)
                sync_ctx = nullcontext()
            elif distributed:
                sync_ctx = nullcontext() if is_sync_step else model.no_sync()
            else:
                sync_ctx = nullcontext()
            with sync_ctx:
                loss.backward()

            if not is_sync_step:
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_main and step % args.log_every == 0:
                now = time.perf_counter()
                elapsed = max(now - last_log, 1e-6)
                last_log = now
                tokens_logged = (args.seq_len * args.micro_batch_size * args.grad_accum_steps
                                 * world_size * args.log_every)
                loss_value = loss.item() * args.grad_accum_steps
                frac_valid = anchor_valid.float().mean().item()
                log_main(is_main, f"[train] epoch={epoch + 1}/{args.epochs} step={step} "
                                  f"{args.objective}_loss={loss_value:.4f} "
                                  f"lr={optimizer.param_groups[0]['lr']:.2e} "
                                  f"tok/s={tokens_logged / elapsed:,.0f} "
                                  f"anchor_valid={frac_valid:.2f}")
                if wandb_run is not None:
                    wandb_run.log({f"train/{args.objective}_loss": loss_value,
                                   "train/lr": optimizer.param_groups[0]["lr"],
                                   "train/tokens_per_sec": tokens_logged / elapsed,
                                   "train/anchor_valid": frac_valid,
                                   "train/epoch": epoch + 1}, step=step)

            if args.save_every > 0 and step % args.save_every == 0:
                save_dir = os.path.join(args.output_dir, f"step-{step}")
                save_checkpoint(save_dir, model, base_model, tokenizer, optimizer,
                                scheduler, step, epoch, batch_idx, args, distributed, is_main)
                if is_main:
                    write_release_style_config(Path(save_dir))

            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break

    final_dir = save_final(args.output_dir, model, base_model, tokenizer,
                           args.fsdp2, distributed, is_main)
    if is_main:
        write_release_style_config(Path(final_dir))
        mins = (time.perf_counter() - train_start) / 60.0
        log_main(is_main, "=" * 88)
        log_main(is_main, f"Training complete | steps={step} | {mins:.2f} min | {final_dir}")
        log_main(is_main, "=" * 88)
        if wandb_run is not None:
            wandb_run.finish()
    dist_utils.cleanup_distributed(distributed)


def parse_args():
    p = argparse.ArgumentParser(description="Train Orthrus-Qwen3 (dual-view diffusion).")
    # Data
    p.add_argument("--packed-cache-path", type=str, required=True,
                   help="Prebuilt packed dataset (input_ids + assistant_mask).")
    p.add_argument("--allow-missing-assistant-mask", action="store_true",
                   help="Accept a cache without assistant_mask. Anchors then land anywhere "
                        "in the row, so the diffusion view also trains on prompt tokens.")
    p.add_argument("--num-workers", type=int, default=8)
    # Model / IO
    p.add_argument("--model-dir", type=str, required=True,
                   help="Orthrus-Qwen3 init checkpoint (see src/utils/init_model_qwen3.py).")
    p.add_argument("--output-dir", type=str, default="checkpoints/orthrus-qwen3-8b")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--auto-resume", action="store_true",
                   help="Resume from the newest step-* in --output-dir (for preempted jobs).")
    p.add_argument("--keep-last-checkpoints", type=int, default=5)
    p.add_argument("--attn-implementation", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "flash_attention_3", "sdpa", "eager"],
                   help="For the AR pass only; the diffusion pass always uses FlexAttention. "
                        "Unlike the Qwen3.5 stack, sdpa/eager are correct here too -- "
                        "OrthrusModel.forward builds a real causal mask for them.")
    # Objective
    p.add_argument("--objective", type=str, default="kl", choices=["kl", "ce"],
                   help="kl = paper Eq. 7 (forward KL vs the frozen AR head). "
                        "ce = the hard-label arm ablated in Table 5.")
    p.add_argument("--loss-chunk-size", type=int, default=1024,
                   help="Positions per recomputed logit chunk. Peak activation memory is "
                        "~4 x chunk x vocab x 4B; lower it if the loss OOMs.")
    # Training (defaults = paper Table 4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=0, help="0 = run full epochs")
    p.add_argument("--num-anchor-blocks", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="0 disables clipping.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    # Parallelism
    p.add_argument("--fsdp2", action="store_true")
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--dist-timeout-minutes", type=int, default=180)
    # Logging
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="orthrus-qwen3")
    p.add_argument("--wandb-run-name", type=str, default=None)

    args = p.parse_args()
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0.0, 1.0).")
    if args.num_anchor_blocks <= 0:
        raise ValueError("--num-anchor-blocks must be > 0.")
    if args.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be > 0.")
    return args


if __name__ == "__main__":
    try:
        train(parse_args())
    except Exception:
        print(f"[rank{os.environ.get('RANK', '0')}] Unhandled training exception:")
        traceback.print_exc()
        raise
