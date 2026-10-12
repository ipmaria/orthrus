"""Rank-zero logging, LR schedule, and seed mixing."""

from __future__ import annotations

import logging
import math

_LOGGER = logging.getLogger("orthrus")


def configure_run_logger(is_main: bool, log_file: str | None = None) -> None:

    _LOGGER.setLevel(logging.INFO if is_main else logging.ERROR)
    _LOGGER.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    _LOGGER.addHandler(stream)
    
    if is_main and log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        _LOGGER.addHandler(file_handler)


def log_main(is_main: bool, message: str) -> None:
    if is_main:
        _LOGGER.info(message)


def get_cosine_lr_scale(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.1,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def mix_seed(*fields: int) -> int:
    """Hash fields into a seed
    """
    digest = 1469598103934665603  # FNV-1a 64-bit offset basis
    for field in fields:
        for byte in int(field).to_bytes(8, "little", signed=True):
            digest = ((digest ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return digest % (2**31 - 1)


def format_human_count(count: int) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if count >= scale:
            return f"{count / scale:.2f}{unit}"
    return str(count)
