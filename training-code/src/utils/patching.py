"""Liger-Kernel patching
"""

from __future__ import annotations

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP, Qwen3_5RMSNorm

from src.utils.logging import log_main


def apply_liger(
    model,
    rms_norm: bool = True,
    swiglu: bool = True,
    is_main: bool = True,
) -> dict:
    counts = {"rms_norm": 0, "swiglu": 0}
    if not (rms_norm or swiglu):
        return counts

    from functools import partial

    from liger_kernel.transformers.monkey_patch import (
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP

    patch_norm = partial(
        _patch_rms_norm_module,
        offset=1.0,
        casting_mode="gemma",
        in_place=False,
    )

    def patch_all_norms(module):
        num_patched = 0
        for submodule in module.modules():
            if isinstance(submodule, Qwen3_5RMSNorm):
                patch_norm(submodule)
                num_patched += 1
        return num_patched

    if rms_norm:
        counts["rms_norm"] = patch_all_norms(model)

    if swiglu:
        for layer in model.model.layers:
            if isinstance(layer.mlp, Qwen3_5MLP):
                _patch_swiglu_module(layer.mlp, LigerQwen3MoeSwiGLUMLP)
                counts["swiglu"] += 1

    n_layers = len(model.model.layers)
    if rms_norm and counts["rms_norm"] == 0:
        raise RuntimeError("Liger RMSNorm patch matched no modules; check class bindings.")
    if swiglu and counts["swiglu"] != n_layers:
        raise RuntimeError(
            f"Liger SwiGLU patch covered {counts['swiglu']}/{n_layers} layers; expected all."
        )

    log_main(is_main, f"[liger] patched rms_norm={counts['rms_norm']} modules, "
                      f"swiglu={counts['swiglu']} layers")
    return counts


def _resolved_impl(dispatching_fn):
    for cell in getattr(dispatching_fn, "__closure__", None) or ():
        value = cell.cell_contents
        if callable(value) and not isinstance(value, type):
            return getattr(value, "__module__", "?")
    return "?"


def kernel_report() -> dict:
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    from src.models.modeling_orthrus_qwen3_5 import fla_causal_conv1d

    chunk_impl = _resolved_impl(torch_chunk_gated_delta_rule)
    conv_impl = getattr(fla_causal_conv1d, "__module__", "?")
    return {
        "gated_delta_rule": chunk_impl,
        "gated_delta_rule_is_fla": chunk_impl.startswith("fla"),
        "causal_conv1d": conv_impl,
        "causal_conv1d_is_kernel": conv_impl.startswith("fla"),
    }
