"""Model loading and single-sample generation for both decoding views."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer
from transformers.generation import GenerationMixin

from src.configuration import OrthrusConfig
from src.eval.metrics import BenchmarkTimer, ForwardPassCounter

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_model(model_dir: str, dtype: str = "bfloat16", attn_implementation: str = "sdpa",
               device: str | None = None, block_size: int | None = None):
    """Load an Orthrus-Qwen3 checkpoint (local dir or hub id) plus its tokenizer."""
    from src.model import OrthrusLM

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OrthrusConfig.from_pretrained(model_dir)
    cfg._attn_implementation = attn_implementation
    if block_size is not None:
        # Section 5 sweeps K at inference without retraining.
        cfg.block_size = block_size
    if cfg.block_size is None or cfg.mask_token_id is None:
        raise ValueError(f"{model_dir}/config.json lacks block_size / mask_token_id.")

    model = OrthrusLM.from_pretrained(model_dir, config=cfg, dtype=DTYPES[dtype])
    model = model.to(resolved_device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def build_input_ids(tokenizer, prompt: str, enable_thinking: bool = True,
                    system_prompt: str = "") -> torch.Tensor:
    # No empty system turn: the on-policy training data (generate_onpolicy.py) carries a
    # system message only when the source prompt had one, and the drafter should see the
    # same prefix format at eval time.
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
               [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    # Depending on the transformers version this is a tensor or a BatchEncoding.
    return encoded if isinstance(encoded, torch.Tensor) else encoded.input_ids


@torch.inference_mode()
def generate_orthrus(model, input_ids: torch.Tensor, max_new_tokens: int,
                     temperature: float = 0.0, top_k: int = 20, top_p: float = 0.8):
    """Dual-view decoding: 1 prefill + 2 passes per consensus cycle."""
    device = input_ids.device
    with ForwardPassCounter(model) as counter, BenchmarkTimer(device) as timer:
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_diffusion_mode=True,
        )
    return output_ids, counter.count, timer.elapsed


@torch.inference_mode()
def generate_autoregressive(model, input_ids: torch.Tensor, max_new_tokens: int,
                            temperature: float = 0.0, top_k: int = 20, top_p: float = 0.8,
                            eos_token_id: int | None = None):
    """Sequential AR baseline: exactly one forward pass per token (TPF = 1.0).

    Calls `GenerationMixin.generate` directly rather than
    `OrthrusLM.generate(use_diffusion_mode=False)`, which would forward
    `temperature=0.0` into HF's sampler and trip its "temperature must be strictly
    positive" validation. `use_cache=True` is explicit because the Orthrus configs
    ship with `use_cache: false`.
    """
    device = input_ids.device
    sampling = (
        {"do_sample": False}
        if temperature < 1e-5
        else {"do_sample": True, "temperature": temperature, "top_k": top_k, "top_p": top_p}
    )
    with ForwardPassCounter(model) as counter, BenchmarkTimer(device) as timer:
        output_ids = GenerationMixin.generate(
            model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            **sampling,
        )
    return output_ids, counter.count, timer.elapsed


def completion_text(tokenizer, output_ids: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
