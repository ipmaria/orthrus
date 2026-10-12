# Reproducing Orthrus (arXiv:2605.12825)

The upstream repository ships a faithful **inference** implementation of the paper plus
released checkpoints, but its training stack targets a different backbone (Qwen3.5) with
a different objective (hard-label cross-entropy), and there is no evaluation code. This
directory adds what is missing, without modifying any upstream file.

| Paper component | Upstream | Added here |
| :--- | :--- | :--- |
| Dual-view attention (Eq. 4), mask (Eq. 6) | `src/model.py` | — |
| Consensus decoding (Eq. 8, 3.3) | `src/model.py:387` | — |
| Training objective (Eq. 7, forward KL) | ✗ CE only | `src/train/loss_kl.py` |
| Training loop for the paper's Qwen3 models | ✗ Qwen3.5 only | `src/train/train_qwen3.py` |
| Init checkpoint for Qwen3 | ✗ Qwen3.5 only | `src/utils/init_model_qwen3.py` |
| Corpus construction (Appendix A) | ✗ | `scripts/build_orthrus_corpus.py` |
| TPF / speedup (Table 1, Eq. 9) — all 8 tasks | ✗ | `src/eval/run_benchmark.py` |
| Acceptance length (Figure 4, left) | ✗ | `src/eval/metrics.py` |
| Losslessness verification (3.3) | ✗ | `src/eval/run_benchmark.py` |
| Memory scaling (Figure 6) | ✗ | `src/eval/run_memory.py` |
| Block-size ablation (Section 5) | ✗ | `src/eval/run_ablation.py` |
| Objective ablation (Table 5) | ✗ | `--objective {kl,ce}` |

---

## 0. Environment

**Reproduction environment — Linux + CUDA, Python 3.10–3.12:**

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .                 # torch>=2.12, transformers>=5.15 — both current (2.13.0 / 5.15.1)
uv pip install ninja packaging
uv pip install flash-attn --no-build-isolation   # or: uv pip install "flash-attn-4[cu13]"  (4.0.0b27, beta)
uv pip install datasets pytest wandb tqdm
```

**3.10 is a hard floor**, not a preference: `src/model.py` and `src/model_mlx.py` use
PEP-604 `X | None` in runtime signatures without `from __future__ import annotations`,
so they raise `TypeError` at import on 3.9. (`pyproject.toml` said `>=3.9` until this
was corrected.) Everything added here carries the future import and is 3.9-clean, but
the model file is the one you cannot avoid.

Two of the declared dependencies are dead weight on the Qwen3 path and can be dropped
if resolution fights you: `flash-linear-attention` is imported only by
`src/models/modeling_orthrus_qwen3_5.py` (the Qwen3.5 backbone), and `liger-kernel`
only by `src/train/loss.py`, behind a `try/except`. Neither is reachable from
`train_qwen3.py`, `init_model_qwen3.py`, or `src/eval/`.

`flash-attn` is optional for evaluation — `--attn-implementation sdpa` is the default
and is correct here. Training defaults to `flash_attention_2` for the AR pass; pass
`--attn-implementation sdpa` to skip the build. The diffusion pass always uses
FlexAttention regardless.

Hardware: evaluation needs ~20 GB for the 8B in bf16, so one 80 GB card is ample and a
48 GB card fits. Training is 8×H200 per Appendix A, under 24 h for 2 epochs.

### Running on a machine that cannot host the model

Only the unit tests run off-CUDA, and they need nothing but torch:

```bash
python -m venv .venv-tests
.venv-tests/bin/pip install torch pytest      # do NOT `pip install -e .` here
.venv-tests/bin/python -m pytest tests/test_dual_pass.py -q
```

`tests/test_dual_pass.py` deliberately never imports `src.model`, so no transformers,
no FlexAttention, no CUDA. Run it from the repo root — `tests/__init__.py` makes pytest
prepend the root to `sys.path`, which is what makes `import src` resolve.

Two platform notes that are easy to lose an afternoon to:

* **Intel macOS cannot run this at all.** PyTorch's last macOS x86_64 wheel is 2.2.2;
  every release from 2.3.0 on is arm64-only, so `torch>=2.12` is unsatisfiable. Pin
  `torch==2.2.2` for the test venv above and stop there.
* **The MLX path needs Apple Silicon.** `mlx` ships arm64 wheels only, so
  `src/model_mlx.py` is unavailable on Intel Macs regardless of the torch situation.

---

## 1. Evaluation (no training required)

The released checkpoints are enough to check the paper's headline claims.

```bash
# Table 1: TPF, speedup, accuracy, and exact-match against the AR baseline
python -m src.eval.run_benchmark \
  --model-dir chiennv/Orthrus-Qwen3-8B \
  --tasks gsm8k math500 aime24 aime25 \
  --temperature 0 \
  --output-dir results/qwen3-8b-greedy

# Code tasks execute model output, so they are opt-in
python -m src.eval.run_benchmark \
  --model-dir chiennv/Orthrus-Qwen3-8B \
  --tasks humaneval mbpp --allow-code-execution \
  --output-dir results/qwen3-8b-code

# Table 1's remaining two rows: Pseudo2code and LiveCodeBench-v5.
# Both are efficiency-only (see below), so no --allow-code-execution is involved.
python -m src.eval.run_benchmark \
  --model-dir chiennv/Orthrus-Qwen3-8B \
  --tasks pseudo2code livecodebench_v5 \
  --output-dir results/qwen3-8b-longform

# Table 1's T=1 column
python -m src.eval.run_benchmark --model-dir chiennv/Orthrus-Qwen3-8B \
  --tasks gsm8k math500 --temperature 1.0 --output-dir results/qwen3-8b-sampled
```

**Pseudo2code and LiveCodeBench-v5 are measured for efficiency only** — TPF, wall-clock
speedup, and the exact-match losslessness check — and report no accuracy. That is what
the paper asks of them: both appear in Table 1 and in neither accuracy table. Scoring
them would mean two execution backends the paper never exercises — a C++ toolchain for
Pseudo2code, whose references are C++ judged against stdin/stdout cases, and
LiveCodeBench's zlib+base64 private-test decoder plus its sandbox. The losslessness
check still runs on both, and these are the rows where it bites hardest: long-horizon
code generation is exactly where conditional drift would surface.

Two dataset notes. Pseudo2code comes from LongProc, which ships `pseudo_to_code_0.5k`
and `pseudo_to_code_2k`; the paper does not say which, so 2k is the default and
`--hf-config pseudo_to_code_0.5k` selects the other. LiveCodeBench's hub repo is a
loading-script dataset and `datasets` 4.0 removed script execution, so the harness reads
the jsonl shards directly and streams them — `--hf-config release_v5` (the default) is
cumulative v1..v5, `--hf-config v5` is that release window alone. **Pass `--limit` unless
you want the full stream**: release_v5 is ~4.3 GB on the wire, almost all of it private
test-case blobs the harness immediately discards, and streaming stops early when limited.

**What the numbers mean.** `TPF = generated tokens / forward passes` (Eq. 9), counting
*every* forward pass including prefill. That convention makes the paper's own bounds come
out exactly: the AR baseline is 1.0, and Orthrus's floor is 0.5 (one prefill plus two
passes per consensus cycle, each cycle committing at least one token).

Table 1's speedup column is **not** the TPF ratio — GSM8K/Qwen3-1.7B is TPF 4.20 against
a 4.37× speedup, and AIME24 is 4.33 against 5.65×. The harness therefore reports
`tpf_ratio` and measured `wallclock_speedup` as separate columns rather than conflating
them. Wall-clock numbers are hardware-dependent; TPF is not.

**Average acceptance length** (Figure 4, left) is a third number again, and reading it
off the TPF column is the easy mistake: MATH-500/Qwen3-8B is TPF **6.35** against an
acceptance length of **11.7**. The harness reports it as its own `accept` column and as
`orthrus.acceptance_length`. It needs no instrumentation — a dual-view run spends
1 prefill + 2 passes per cycle, so `cycles = (forward_passes − samples) / 2` and
`acceptance_length = generated_tokens / cycles − 1`, the `−1` dropping the AR correction
token each cycle commits. In single-sample form that is `2 × TPF − 1`, which reproduces
all four of Figure 4's Orthrus bars exactly from Table 1's TPF (GSM8K 9.6, MATH-500 11.7,
AIME25 9.5, MBPP 6.9). The metric is undefined for the AR baseline, which has no cycles,
so it is omitted there rather than reported as zero.

**The losslessness check is the important one.** At `--temperature 0` the harness decodes
every prompt twice — once through the consensus loop, once through the plain AR path —
and compares token ids exactly. Section 3.3's claim is that Eq. 8 accepts a draft token
only when it equals the greedy AR prediction, so in exact arithmetic the two streams are
identical.

In bf16 they are identical *up to argmax tie-breaking*, and the distinction matters when
you read the number. The AR baseline's decode step is a GEMM with `M=1`; Orthrus's
verification pass is `M=32`. Different shapes select different kernels and tile sizes, so
the reductions over hidden and over the 151,936-wide vocab projection accumulate in a
different order, and attention takes a split-K decode kernel in one path but not the
other. Logits for the same position therefore differ by around 1e-3 relative. Where the
top-2 logits are closer than that, `argmax` can land differently — and since the
comparison is exact token-id equality over the whole completion, one flipped token
diverges the rest of that sample. A 500-problem run at 2048 tokens is ~1e6 argmax
decisions, so a per-token flip rate of 1e-6 already yields about one mismatched sample.

So judge the **shape**, not the count:

* **Numerics** — a handful of samples; the first divergence sits at a position where the
  AR top-2 logit gap is ~1e-3 or below; the rate is roughly flat in `--block-size`.
* **A real bug** — a rate above ~1%, or mismatches clustering at the start of generation,
  or a rate that grows with `--block-size`. That points at the KV-cache truncation
  (`src/model.py:532`), the position ids, or a mask leak: the consensus check accepted a
  token greedy AR would not have, which *is* a violation of Eq. 8.

The operational test: on a mismatch, log the first differing index and the AR top-2 logit
gap there. A tiny gap is rounding; a large gap is a bug.

No exact-match check runs at `--temperature > 0`, and none can — the two paths draw
different random numbers, and the guarantee there is distributional (rejection sampling
against `p_AR`), not token-identity.

Memory (Figure 6) and the K sweep (Section 5):

```bash
python -m src.eval.run_memory --model-dir chiennv/Orthrus-Qwen3-8B \
  --context-lengths 1024 4096 16384 40960

python -m src.eval.run_ablation --model-dir chiennv/Orthrus-Qwen3-8B \
  --task gsm8k --block-sizes 4 8 16 32 --limit 200
```

`run_memory` prints the analytic KV overhead alongside the measured peak. For Qwen3-8B at
K=32 in bf16 that is `2 × 32 × 8 × 128 × 2 × 36 = 4,718,592 B = 4.50 MiB`, independent of
sequence length — which is exactly the constant ≈4.5 MiB Figure 6(b) reports, and is the
concrete content of the O(1) claim.

---

## 2. Training

### 2.1 Init checkpoint

```bash
python -m src.utils.init_model_qwen3 \
  --base Qwen/Qwen3-8B \
  --out pretrained-models/orthrus-qwen3-8b-init \
  --block-size 32
```

Copies the frozen AR weights, creates the diffusion twins warm-started from their AR
counterparts (3.1), and writes a `config.json` matching the released layout
(`model_type: qwen3`, `auto_map → modeling_orthrus.OrthrusLM`, `mask_token_id = 151669`).
A checkpoint trained from this init loads with the README quickstart snippet unchanged.

Reported trainable fraction should be ~15.6% for Qwen3-8B, i.e. the paper's "approximately
16%". Note this counts `q/k/v/o_proj_diff` plus the `q/k_norm_diff` twins; §3.1 lists only
Q/K/V, which would be 10.0%, so the code — not the prose — is what matches the 16% figure.

### 2.2 Data (Appendix A)

```bash
python scripts/build_orthrus_corpus.py --inspect          # check the source schema first
python scripts/build_orthrus_corpus.py --out data/orthrus-sft.jsonl --per-domain 200000

python scripts/build_packed_cache.py \
  --dataset-path data/orthrus-sft.jsonl \
  --out data/processed/orthrus-packed-2048-asst \
  --model-dir pretrained-models/orthrus-qwen3-8b-init \
  --seq-len 2048
```

Appendix A samples Nemotron-Post-Training-Dataset-v2 balanced 1:1:1 across math / code /
chat, packed to L=2048, yielding 471,952 rows ≈ 0.96B tokens. The upstream category
naming has shifted over time, so run `--inspect` first and extend `DOMAIN_PATTERNS` if a
category comes back `UNMAPPED`.

`build_packed_cache.py` is upstream code and emits the `assistant_mask` column that
confines anchors to assistant spans.

### 2.3 Train

```bash
torchrun --nproc_per_node=8 -m src.train.train_qwen3 \
  --model-dir pretrained-models/orthrus-qwen3-8b-init \
  --packed-cache-path data/processed/orthrus-packed-2048-asst \
  --output-dir checkpoints/orthrus-qwen3-8b \
  --objective kl --fsdp2 --activation-checkpointing
```

or `sbatch scripts/submit_train_qwen3.sbatch`. Defaults follow Table 4 exactly: L=2048,
256 anchors/row, K=32, 2 epochs, lr 2e-4 cosine with 5% warmup, grad clip 1.0, micro-batch
1 × grad-accum 16 × 8 ranks = global batch 128, bf16. (The upstream Qwen3.5 script's
defaults — L=4096, 4 epochs, 2% warmup, grad-accum 2 — do not reproduce Table 4.)

**How the objective works.** One frozen AR pass builds `(K_AR, V_AR)` *and* the teacher
hidden states; one diffusion pass over B corrupted blocks reads that cache in place
(`use_cache=False`, so the blocks never contaminate it). Gradients reach only the
diffusion twins. Per block, offsets `0..K-2` are supervised against the AR row at
`a_b + j` — K−1 positions, matching what `generate()` actually consumes.

The memory problem the KL objective creates is real: at 256 anchors × K=32 there are 7,936
supervised positions, and a single fp32 logit tensor over Qwen3's 151,936-token vocabulary
is ~4.8 GB — with forward KL needing four of them. Liger's fused linear CE cannot help,
being hard-label only. `fused_linear_forward_kl` therefore streams positions in chunks and
recomputes each chunk's logits in backward, making peak activation memory
`O(chunk × vocab)` instead of `O(positions × vocab)`. Lower `--loss-chunk-size` if it OOMs.

### 2.4 Table 5 (objective ablation)

Train twice, changing one flag, then evaluate both:

```bash
OBJECTIVE=kl OUT_DIR=checkpoints/orthrus-qwen3-8b-kl sbatch scripts/submit_train_qwen3.sbatch
OBJECTIVE=ce OUT_DIR=checkpoints/orthrus-qwen3-8b-ce sbatch scripts/submit_train_qwen3.sbatch

for arm in kl ce; do
  python -m src.eval.run_benchmark --model-dir checkpoints/orthrus-qwen3-8b-$arm/final \
    --tasks math500 --output-dir results/objective-$arm
done
```

The paper predicts **identical accuracy** (86.2% either way — the consensus mechanism
guarantees it) and a TPF gap (6.35 for KL against 5.86 for CE). That accuracy is invariant
while TPF is not is the sharpest single test of the whole architecture: consensus makes
the objective irrelevant to *what* is produced and relevant only to *how fast*.

---

## 3. Tests

```bash
python -m pytest tests/test_dual_pass.py -q
```

Covers block corruption (Eq. 5), absolute-position RoPE for packed blocks, the Eq. 6 mask
rule, the K−1 supervision grid, teacher/student alignment, and the chunked KL against a
naive full-vocabulary reference. Torch-only — no CUDA, FlexAttention or transformers
needed, so it runs on a laptop.

---

## Caveats

* Training needs CUDA. The diffusion pass uses FlexAttention with the FLASH backend
  (`src/model.py:34`); it will not run on CPU or MPS. Evaluation runs anywhere the model
  fits, but `--attn-implementation sdpa` is required off CUDA.
* Unlike the Qwen3.5 stack, the Qwen3 AR pass is correct under `sdpa`/`eager` too —
  `OrthrusModel.forward` builds a real causal mask for those. Flash is still faster.
* `--fsdp2` here shards the decoder layers but leaves root parameters replicated. The
  loss applies `lm_head` *after* `forward` returns, where a root-sharded weight would be
  an unusable DTensor shard; upstream avoids this only by computing loss inside `forward`.
  Cost is one frozen embedding/head copy per rank (~2.5 GB on 8B) and no optimizer state.
* `generate_dual_pass_mask` calls `create_block_mask` uncompiled, so the mask is rebuilt
  every step (anchors are resampled per row). The Qwen3.5 twin wraps it in
  `torch.compile`. Compiling it is the obvious throughput win if training is mask-bound.
* Math scoring uses `\boxed{}` extraction plus a light normaliser, not symbolic
  equivalence. Expect small deltas against harnesses using `math_verify`; the AR-vs-Orthrus
  *comparison* is unaffected, since both are scored identically.
* Section 5's multi-step denoising ablation is not implemented — `generate()` is
  single-step by construction, and adding iterative denoising would mean forking the
  official decode loop.
