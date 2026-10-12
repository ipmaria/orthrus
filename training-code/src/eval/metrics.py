"""Efficiency instrumentation: Tokens Per Forward Pass (paper Eq. 9) and wall clock.

    TPF = total generated tokens / total forward passes

Counting convention, chosen so the paper's own bounds come out exact:

  * EVERY `OrthrusLM.forward` call counts, prefill included. An AR run generating N
    tokens does 1 prefill (which yields the first token) + N-1 decode steps = N passes,
    so the AR baseline lands at exactly TPF 1.0 -- the bound stated in 4.2.
  * An Orthrus run does 1 prefill + 2 passes per consensus cycle. A cycle that accepts
    nothing still commits one token, so TPF -> 0.5 in the worst case, which is the
    "strict theoretical lower bound of 0.5 TPF" the paper quotes.

Note that Table 1's speedup column is NOT the TPF ratio (GSM8K/Qwen3-1.7B is TPF 4.20
vs 4.37x; AIME24 is 4.33 vs 5.65x), so `BenchmarkTimer` reports measured wall-clock
speedup separately rather than deriving it from TPF.

Average acceptance length (Figure 4, left) is a THIRD number again, and it needs no
extra instrumentation -- it follows from the counts above. A dual-view run spends
1 prefill + 2 passes per consensus cycle, so over `S` samples:

    cycles           = (forward_passes - S) / 2
    acceptance_length = generated_tokens / cycles - 1

The -1 removes the AR correction token each cycle commits, leaving the accepted DRAFT
tokens, which is what the paper plots. Checked against all four bars of Figure 4 using
Table 1's TPF for the same Qwen3-8B T=0 runs:

    task        TPF     2*TPF-1    Figure 4
    GSM8K       5.30    9.6        9.6
    MATH-500    6.35    11.7       11.7
    AIME25      5.25    9.5        9.5
    MBPP        3.95    6.9        6.9

Exact on all four, which pins down both the paper's definition and the prefill-counting
convention above. (`2*TPF-1` is the single-sample form; `as_dict` uses the aggregate one.)

The metric is undefined for the sequential AR baseline -- there are no cycles -- so it
is emitted only when `passes_per_cycle` is set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


class ForwardPassCounter:
    """Counts forward calls on a module for the duration of a `with` block."""

    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._handle = None
        self.count = 0

    def _hook(self, *_args, **_kwargs):
        self.count += 1

    def __enter__(self) -> "ForwardPassCounter":
        self.count = 0
        self._handle = self._model.register_forward_pre_hook(
            lambda module, inputs: self._hook(), with_kwargs=False
        )
        return self

    def __exit__(self, *_exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@dataclass
class RunStats:
    """Aggregated efficiency numbers for one (model, task, decoding mode) run."""

    label: str
    generated_tokens: int = 0
    forward_passes: int = 0
    seconds: float = 0.0
    num_samples: int = 0
    per_sample_tokens: list[int] = field(default_factory=list)
    # Forward passes per consensus cycle, for a dual-view run: 2 (diffusion + AR
    # verification). Leave None for the sequential AR baseline, which has no cycles
    # and for which acceptance length is meaningless rather than merely 0.
    passes_per_cycle: int | None = None

    @property
    def tpf(self) -> float:
        return self.generated_tokens / max(1, self.forward_passes)

    @property
    def consensus_cycles(self) -> float:
        """Cycles run, backing out the one prefill pass each sample spends."""
        if self.passes_per_cycle is None:
            return float("nan")
        return (self.forward_passes - self.num_samples) / self.passes_per_cycle

    @property
    def acceptance_length(self) -> float:
        """Mean accepted draft tokens per cycle (Figure 4, left)."""
        cycles = self.consensus_cycles
        if not cycles > 0:  # NaN or no cycles at all
            return float("nan")
        return self.generated_tokens / cycles - 1.0

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / max(1e-9, self.seconds)

    def add(self, tokens: int, passes: int, seconds: float) -> None:
        self.generated_tokens += tokens
        self.forward_passes += passes
        self.seconds += seconds
        self.num_samples += 1
        self.per_sample_tokens.append(tokens)

    def as_dict(self) -> dict:
        payload = {
            "label": self.label,
            "generated_tokens": self.generated_tokens,
            "forward_passes": self.forward_passes,
            "seconds": round(self.seconds, 3),
            "num_samples": self.num_samples,
            "tpf": round(self.tpf, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
        }
        acceptance = self.acceptance_length
        if acceptance == acceptance:  # not NaN -> a dual-view run
            payload["consensus_cycles"] = round(self.consensus_cycles, 1)
            payload["acceptance_length"] = round(acceptance, 4)
        return payload


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


class BenchmarkTimer:
    """Device-synchronised wall-clock timing for a single generation call."""

    def __init__(self, device: torch.device):
        self.device = device
        self.elapsed = 0.0

    def __enter__(self) -> "BenchmarkTimer":
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        synchronize(self.device)
        self.elapsed = time.perf_counter() - self._start


def speedup_table(baseline: RunStats, orthrus: RunStats) -> dict:
    """Table 1 row: TPF plus both the algorithmic and the measured speedup."""
    return {
        "tpf_ar": round(baseline.tpf, 4),
        "tpf_orthrus": round(orthrus.tpf, 4),
        "tpf_ratio": round(orthrus.tpf / max(1e-9, baseline.tpf), 4),
        "wallclock_speedup": round(
            orthrus.tokens_per_second / max(1e-9, baseline.tokens_per_second), 4
        ),
        "tokens_per_second_ar": round(baseline.tokens_per_second, 2),
        "tokens_per_second_orthrus": round(orthrus.tokens_per_second, 2),
    }
