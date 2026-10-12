"""Checkpoint save/load/prune.
"""

from __future__ import annotations

import os
import re
import shutil

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from src.utils.logging import log_main

TRAINER_STATE_FILE = "trainer_state.pt"


def latest_checkpoint_dir(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None
    best_step, best_path = -1, None
    for name in os.listdir(output_dir):
        match = re.fullmatch(r"step-(\d+)", name)
        full = os.path.join(output_dir, name)
        if match and os.path.isdir(full) and int(match.group(1)) > best_step:
            best_step, best_path = int(match.group(1)), full
    return best_path


def prune_old_checkpoints(output_dir: str, keep_last: int) -> list[str]:
    if keep_last <= 0:
        return []
    step_dirs = []
    for name in os.listdir(output_dir):
        match = re.fullmatch(r"step-(\d+)", name)
        full = os.path.join(output_dir, name)
        if match and os.path.isdir(full):
            step_dirs.append((int(match.group(1)), full))
    step_dirs.sort()
    removed = []
    for _, path in step_dirs[:-keep_last] if len(step_dirs) > keep_last else []:
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def resume_batch_idx_is_valid(saved_world, saved_grad_accum, world_size, grad_accum_steps):
    return saved_world == world_size and saved_grad_accum == grad_accum_steps


def save_trainer_state(
    save_dir,
    model,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    batch_idx: int,
    fsdp2: bool,
    distributed: bool,
    is_main: bool,
    grad_accum_steps: int,
):
    # Under FSDP2 the optimizer state is sharded, gather to rank 0 (cpu_offload bounds host memory).
    if fsdp2 and distributed:
        optim_state = get_optimizer_state_dict(
            model, optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    else:
        optim_state = optimizer.state_dict()
    if not is_main:
        return
    torch.save(
        {
            "optimizer": optim_state,
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "world_size": dist.get_world_size() if distributed else 1,
            "grad_accum_steps": grad_accum_steps,
        },
        os.path.join(save_dir, TRAINER_STATE_FILE),
    )


def load_trainer_state(
    checkpoint_dir,
    model,
    optimizer,
    scheduler,
    fsdp2: bool,
    distributed: bool,
):
    state_path = os.path.join(checkpoint_dir, TRAINER_STATE_FILE)
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"{checkpoint_dir} has no {TRAINER_STATE_FILE}.")

    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if fsdp2 and distributed:
        set_optimizer_state_dict(
            model, optimizer, optim_state_dict=payload["optimizer"],
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    else:
        optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return (
        int(payload["step"]),
        int(payload["epoch"]),
        int(payload["batch_idx"]),
        int(payload.get("world_size", 0)),
        int(payload.get("grad_accum_steps", 0)),
    )


def save_checkpoint(
    save_dir,
    model,
    base_model,
    tokenizer,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    batch_idx: int,
    args,
    distributed: bool,
    is_main: bool,
):
    if args.fsdp2 and distributed:
        full_state = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        if is_main:
            os.makedirs(save_dir, exist_ok=True)
            base_model.save_pretrained(save_dir, state_dict=full_state)
            tokenizer.save_pretrained(save_dir)
        save_trainer_state(save_dir, model, optimizer, scheduler, step, epoch, batch_idx,
                           fsdp2=True, distributed=True, is_main=is_main,
                           grad_accum_steps=args.grad_accum_steps)
    elif is_main:
        os.makedirs(save_dir, exist_ok=True)
        (model.module if distributed else model).save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        save_trainer_state(
            save_dir,
            model,
            optimizer,
            scheduler,
            step,
            epoch,
            batch_idx,
            fsdp2=False,
            distributed=distributed,
            is_main=is_main,
            grad_accum_steps=args.grad_accum_steps,
        )
    if is_main:
        log_main(is_main, f"[checkpoint] saved to {save_dir}")
        for removed in prune_old_checkpoints(args.output_dir, args.keep_last_checkpoints):
            log_main(is_main, f"[checkpoint] pruned {removed}")


def save_final(
    output_dir,
    model,
    base_model,
    tokenizer,
    fsdp2: bool,
    distributed: bool,
    is_main: bool,
):
    final_dir = os.path.join(output_dir, "final")
    if fsdp2 and distributed:
        final_state = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        if is_main:
            os.makedirs(final_dir, exist_ok=True)
            base_model.save_pretrained(final_dir, state_dict=final_state)
            tokenizer.save_pretrained(final_dir)
    elif is_main:
        os.makedirs(final_dir, exist_ok=True)
        (model.module if distributed else model).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
    return final_dir
