from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)


def setup_distributed(timeout_minutes: int = 180) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_minutes))
    return True, rank, local_rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def apply_fsdp2(model, modules_to_shard, activation_checkpointing: bool = False) -> None:
    if activation_checkpointing:
        targets = tuple(modules_to_shard)
        apply_activation_checkpointing(
            model, check_fn=lambda submodule: isinstance(submodule, targets)
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    mesh = DeviceMesh(device_type="cuda", mesh=list(range(world_size)))
    config = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        ),
        "mesh": mesh,
        "reshard_after_forward": True,
    }
    for module in model.modules():
        if any(isinstance(module, target) for target in modules_to_shard):
            fully_shard(module, **config)
    fully_shard(model, **config)
