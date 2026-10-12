"""Build an Orthrus-Qwen3 init checkpoint, and the Qwen3 freeze/warm-start helpers.

    python -m src.utils.init_model_qwen3 --base Qwen/Qwen3-8B --out pretrained-models/orthrus-qwen3-8b-init

The official `src/utils/init_model.py` does this for the Qwen3.5 hybrid backbone. The
paper benchmarks Qwen3 (1.7B / 4B / 8B), whose model lives in `src/model.py`, so this
is the matching initialiser: it produces a checkpoint whose `config.json` mirrors the
released `chiennv/Orthrus-Qwen3-*` layout (`model_type: qwen3`, `auto_map` ->
`modeling_orthrus.OrthrusLM`, `block_size`, `mask_token_id`) and copies `src/model.py`
in as `modeling_orthrus.py`. A checkpoint trained from this init therefore loads with
the exact README quickstart snippet.

Deliberately does NOT import `src.utils.init_model`: that module pulls in the Qwen3.5
modeling file, which hard-imports `fla`. Qwen3-only users should not need it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from src.configuration import OrthrusConfig
from src.model import OrthrusLM

# The diffusion twins injected by `OrthrusAttention.__init__` (src/model.py:92-104).
# Note this is QKV *and* the output projection plus the q/k norms -- which is what
# yields the paper's "approximately 16% of the total model parameters" (15.6% for
# Qwen3-8B). QKV alone would be 10.0%, so 3.1's shorter enumeration is the loose one.
DIFF_PARAM_SUFFIXES = (
    "q_proj_diff", "k_proj_diff", "v_proj_diff", "o_proj_diff",
    "q_norm_diff", "k_norm_diff",
)

REMOTE_CODE_FILENAME = "modeling_orthrus.py"


@torch.no_grad()
def _copy_module_params(dst: nn.Module, src: nn.Module) -> None:
    src_params = dict(src.named_parameters())
    for name, param in dst.named_parameters():
        param.copy_(src_params[name])
    src_buffers = dict(src.named_buffers())
    for name, buffer in dst.named_buffers():
        if name in src_buffers:
            buffer.copy_(src_buffers[name])


@torch.no_grad()
def copy_diff_from_ar(model: OrthrusLM) -> None:
    """Warm-start every `*_diff` twin from its frozen AR counterpart (paper 3.1)."""
    for layer in model.model.layers:
        attention = layer.self_attn
        for diff_module, ar_module in (
            (attention.q_proj_diff, attention.q_proj),
            (attention.k_proj_diff, attention.k_proj),
            (attention.v_proj_diff, attention.v_proj),
            (attention.o_proj_diff, attention.o_proj),
            (attention.q_norm_diff, attention.q_norm),
            (attention.k_norm_diff, attention.k_norm),
        ):
            _copy_module_params(diff_module, ar_module)


def freeze_to_diffusion(model: nn.Module) -> tuple[int, int]:
    """Freeze everything but the diffusion twins. Returns (n_trainable, n_total)."""
    num_trainable = num_total = 0
    for name, param in model.named_parameters():
        num_total += 1
        param.requires_grad = any(suffix in name for suffix in DIFF_PARAM_SUFFIXES)
        num_trainable += int(param.requires_grad)
    return num_trainable, num_total


def _load_base_state_dict(model_name_or_path: str) -> dict:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = model_name_or_path
    if not os.path.isdir(path):
        path = snapshot_download(model_name_or_path, allow_patterns=["*.safetensors", "*.json"])
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors found under {path}")
    state_dict = {}
    for shard in shards:
        state_dict.update(load_file(shard))
    return state_dict


def write_release_style_config(save_dir: Path) -> None:
    """Match the released checkpoints: model_type `qwen3` + remote-code auto_map."""
    config_path = save_dir / "config.json"
    with open(config_path) as handle:
        config = json.load(handle)
    config["model_type"] = "qwen3"
    config["architectures"] = ["OrthrusLM"]
    config["auto_map"] = {
        "AutoModelForCausalLM": f"{REMOTE_CODE_FILENAME[:-3]}.OrthrusLM"
    }
    with open(config_path, "w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    source = Path(__file__).resolve().parents[1] / "model.py"
    shutil.copyfile(source, save_dir / REMOTE_CODE_FILENAME)


def init_orthrus_qwen3(
    base_model_name_or_path: str,
    out_dir: str,
    block_size: int = 32,
    mask_token_id: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    force: bool = False,
):
    save_dir = Path(out_dir)
    if save_dir.exists() and (save_dir / "config.json").exists() and not force:
        raise ValueError(f"{out_dir} already holds a checkpoint; pass --force to overwrite.")
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    base_cfg = AutoConfig.from_pretrained(base_model_name_or_path)
    cfg = OrthrusConfig.from_dict(getattr(base_cfg, "text_config", base_cfg).to_dict())
    cfg.block_size = block_size
    cfg.use_cache = False

    if mask_token_id is None:
        # The released checkpoints use 151669 for Qwen3 == len(tokenizer): the first
        # reserved id past the tokenizer's vocabulary but inside the embedding matrix.
        mask_token_id = getattr(tokenizer, "mask_token_id", None) or len(tokenizer)
    if not mask_token_id < cfg.vocab_size:
        raise ValueError(
            f"mask_token_id {mask_token_id} >= vocab_size {cfg.vocab_size}; "
            f"pick a reserved id or widen the vocab."
        )
    cfg.mask_token_id = mask_token_id

    model = OrthrusLM(cfg).to(dtype)

    base_sd = _load_base_state_dict(base_model_name_or_path)
    missing_keys, unexpected_keys = model.load_state_dict(base_sd, strict=False)
    non_diff_missing = [
        key for key in missing_keys
        if not any(suffix in key for suffix in DIFF_PARAM_SUFFIXES) and key != "lm_head.weight"
    ]
    if non_diff_missing:
        raise RuntimeError(
            f"Base weights did not fully load: {len(non_diff_missing)} non-diff keys "
            f"missing, e.g. {non_diff_missing[:5]}. Check the checkpoint key layout."
        )

    copy_diff_from_ar(model)

    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    write_release_style_config(save_dir)

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(
        param.numel() for name, param in model.named_parameters()
        if any(suffix in name for suffix in DIFF_PARAM_SUFFIXES)
    )
    print(f"[init] saved to {save_dir}")
    print(f"[init] block_size={block_size} mask_token_id={mask_token_id} "
          f"vocab_size={cfg.vocab_size} tied={cfg.tie_word_embeddings}")
    print(f"[init] params: total={total/1e9:.2f}B trainable(diff)={trainable/1e9:.2f}B "
          f"({100.0 * trainable / total:.1f}%)")
    print(f"[init] unexpected keys dropped: {len(unexpected_keys)}")
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Build an Orthrus-Qwen3 init checkpoint.")
    parser.add_argument("--base", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=32, help="Paper Table 4: K=32.")
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_orthrus_qwen3(
        args.base, args.out,
        block_size=args.block_size,
        mask_token_id=args.mask_token_id,
        dtype={"float32": torch.float32, "float16": torch.float16,
               "bfloat16": torch.bfloat16}[args.dtype],
        force=args.force,
    )
