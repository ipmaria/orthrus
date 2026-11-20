# Go / no-go experiments for noise-conditioned drafting
# (Qwen3-0.6B, thinking, L = 8192, ONE GPU)

Two experiments decide whether this is a strong paper:

* **E1**: is there a signal, and does it come from the claimed mechanism?
* **E2**: does it beat the direct competitor (DBLAST), in real speed?

Everything else (scale, more benchmarks, ablations over m) is only worth doing if both
are positive. All commands are written out in full, for a single GPU (80 GB A100/H100).
Run them from the repository root, one after another.

| what | value |
|---|---|
| stage-1 checkpoint every arm starts from | `checkpoints/orthrus-qwen3-0.6b-think-8192/step-15000` |
| Orthrus init checkpoint (only its tokenizer is used) | `pretrained-models/orthrus-qwen3-0.6b-init` |
| teacher (the AR model the Orthrus init copies) | `Qwen/Qwen3-0.6B` |
| reasoning | on (thinking mode) in generation, training data and evaluation |
| sequence length | 8192 tokens everywhere |
| hardware | 1 GPU, no FSDP, no torchrun |

Every stage-2 arm starts from the **same** stage-1 checkpoint, with the same data,
steps, learning rate and seed. Take `--num-anchor-blocks` from your stage-1 launch
command (or its wandb config). The commands below use 256 (the trainer's default); if
stage 1 used another value, use that value everywhere.

**Budget on one GPU (rough, A100).** Rollouts ~4-5 h. Each training arm ~3 h
(`noise-B` ~1.5x that). Decoding at T = 1 runs at ~60 tok/s, so evaluation uses the
first 200 problems of each task (~4 h per run for GSM8K + MATH-500). That is enough for
acceptance length: a few hundred thousand cycles per run pin it down. E1 is ~2 days end
to end, E2 another ~2 days. Run E2 only if E1 says go.

```bash
mkdir -p logs data results checkpoints/stage2
```

---

## Step 0. Sanity checks (blocking)

```bash
python -m pytest tests/ -q

python scripts/verify_estimators_numpy.py

python scripts/check_noise_decoding.py \
  --model-dir checkpoints/orthrus-qwen3-0.6b-think-8192/step-15000 \
  --enable-top-m 32 \
  --temperature 1.0 \
  --samples 2000 \
  > logs/check-noise-s1.log 2>&1
```

In `logs/check-noise-s1.log`:

* **Losslessness.** (a) the per-position TV at positions 16-23 must be at the AR-vs-AR
  floor, and the printed entropy must not be ~0 (else the test is vacuous; raise
  `--check-offset`). (b) the log-likelihood z-test must give `|z| < 3`, with the AR-vs-AR
  z also within ±3. **If either fails, stop: it is a bug.**
* **Starting point.** `causal only` vs `noise off` is the cost of the causal mask alone;
  `noise top-32` additionally pays for verifying against the revealed laws with an
  untrained (zero) projection. Training has to win this back. If `causal only` already
  loses most of the acceptance, expect `causal-B` and `noise-B` in E1 to suffer.

---

## Step 1. Training data: teacher rollouts in thinking mode, T = 1, no truncation

All stage-2 arms train on this one corpus. Why these settings:

* **T = 1, `--top-p 1.0`, `--top-k -1`.** Genuine samples from the teacher at the
  temperature the losses use, with no top-p / top-k. This makes the target-path losses
  exact (coupling) and gives the posterior noise its prior law. (`-1` disables top-k in
  vLLM; if your version rejects it, use `0`.)
* **`--min-distinct-ngram 0`.** The default 0.3 drops repetitive responses, which biases
  the corpus away from the teacher's distribution.
* **`--kv-cache-dtype auto`.** fp8 would shift trajectories away from the exact model.
* **`--max-model-len 8192 --max-tokens 8192`.** Prompt plus response fill the 8192-token
  rows. Responses cut by the budget are kept as prefixes; a prefix of a sample is a
  sample.

Sizing for one GPU: training below is 1000 steps x global batch 32 (micro-batch 1 x
grad-accum 32) = **32,000 rows** of 8192 tokens (~260M tokens). 20,000 prompts per domain,
with 3 samples per code prompt, give ~100,000 responses, enough to fill that.

### 1a. Prompts

```bash
python scripts/build_orthrus_corpus.py \
  --out data/stage2-prompts.jsonl \
  --per-domain 20000 \
  --seed 7
```

No `--balance`: code has few unique prompts, so it gets 3 samples per prompt instead.

### 1b. Rollouts (one process, the whole GPU)

```bash
python scripts/generate_onpolicy.py \
  --in data/stage2-prompts.jsonl \
  --out data/stage2-rollouts-t1.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples-per-prompt 1 code=3 \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k -1 \
  --max-model-len 8192 \
  --max-tokens 8192 \
  --min-response-tokens 256 \
  --min-distinct-ngram 0 \
  --kv-cache-dtype auto \
  --gpu-memory-utilization 0.9 \
  --seed 42 \
  > logs/rollouts.log 2>&1
```

Without `--num-shards` the output is written straight to `data/stage2-rollouts-t1.jsonl`
(no shard files, nothing to concatenate). Generation runs in thinking mode
(`enable_thinking=True`) and stores the exact generated token ids.

### 1c. Pack into 8192-token rows

```bash
python scripts/build_packed_cache_ids.py \
  --dataset-path data/stage2-rollouts-t1.jsonl \
  --out data/processed/stage2-rollouts-t1-8192 \
  --model-dir pretrained-models/orthrus-qwen3-0.6b-init \
  --seq-len 8192
```

The packer supervises the whole assistant span, `<think>…</think>` included.
**Check the row count it prints: you need ≥ 32,000 rows.** With fewer, add `--epochs 2`
to every training command, or regenerate with a larger `--per-domain`.

---

## E1. Is there a signal, and is it the claimed mechanism?

| arm | output dir | what it isolates |
|---|---|---|
| `s1` | (stage-1 checkpoint itself) | reference |
| `tp-B` | `checkpoints/stage2/tp-B` | "just fine-tuned on teacher rollouts" |
| `causal-B` | `checkpoints/stage2/causal-B` | cost of the causal in-block mask alone (no noise) |
| `noise-B` | `checkpoints/stage2/noise-B` | the method |

### Train (one after another)

Single process: `python -m src.train.train_qwen3`, no `torchrun`, no `--fsdp2`.
Activation checkpointing works without FSDP. If you run out of memory, lower
`--loss-chunk-size` (default 1024) to 512 or 256.

```bash
python -m src.train.train_qwen3 \
  --model-dir checkpoints/orthrus-qwen3-0.6b-think-8192/step-15000 \
  --packed-cache-path data/processed/stage2-rollouts-t1-8192 \
  --output-dir checkpoints/stage2/tp-B \
  --objective accepted-length \
  --eal-estimator target-path \
  --eal-temperature 1.0 \
  --seq-len 8192 \
  --num-anchor-blocks 256 \
  --micro-batch-size 1 \
  --grad-accum-steps 32 \
  --epochs 1 \
  --max-steps 1000 \
  --lr 2e-5 \
  --warmup-ratio 0.05 \
  --max-grad-norm 1.0 \
  --dtype bfloat16 \
  --activation-checkpointing \
  --log-every 10 \
  --save-every 500 \
  --seed 42 \
  > logs/train-tp-B.log 2>&1
```

```bash
python -m src.train.train_qwen3 \
  --model-dir checkpoints/orthrus-qwen3-0.6b-think-8192/step-15000 \
  --packed-cache-path data/processed/stage2-rollouts-t1-8192 \
  --output-dir checkpoints/stage2/causal-B \
  --objective accepted-length \
  --eal-estimator target-path \
  --causal-drafting \
  --eal-temperature 1.0 \
  --seq-len 8192 \
  --num-anchor-blocks 256 \
  --micro-batch-size 1 \
  --grad-accum-steps 32 \
  --epochs 1 \
  --max-steps 1000 \
  --lr 2e-5 \
  --warmup-ratio 0.05 \
  --max-grad-norm 1.0 \
  --dtype bfloat16 \
  --activation-checkpointing \
  --log-every 10 \
  --save-every 500 \
  --seed 42 \
  > logs/train-causal-B.log 2>&1
```

```bash
python -m src.train.train_qwen3 \
  --model-dir checkpoints/orthrus-qwen3-0.6b-think-8192/step-15000 \
  --packed-cache-path data/processed/stage2-rollouts-t1-8192 \
  --output-dir checkpoints/stage2/noise-B \
  --objective noise-eal \
  --noise-reveal-top-m 32 \
  --eal-temperature 1.0 \
  --seq-len 8192 \
  --num-anchor-blocks 256 \
  --micro-batch-size 1 \
  --grad-accum-steps 32 \
  --epochs 1 \
  --max-steps 1000 \
  --lr 2e-5 \
  --warmup-ratio 0.05 \
  --max-grad-norm 1.0 \
  --dtype bfloat16 \
  --activation-checkpointing \
  --log-every 10 \
  --save-every 500 \
  --seed 42 \
  > logs/train-noise-B.log 2>&1
```

`noise-B` is ~1.5x slower per step than the others (one extra diffusion pass).
After training, confirm it is still lossless:

```bash
python scripts/check_noise_decoding.py \
  --model-dir checkpoints/stage2/noise-B/final \
  --temperature 1.0 \
  --samples 2000 \
  > logs/check-noise-B.log 2>&1
```

### Evaluate (one after another)

Thinking mode is on by default in `run_benchmark`. `--top-k 0 --top-p 1.0` is
mandatory: the losses use the untruncated distribution. `--no-baseline` skips the AR
run. One dtype for every arm (bfloat16). Subsets: the first 200 problems of each task
(`--limit` is per task; task default token budgets: GSM8K 4096, MATH-500 8192).

**Priority 1: T = 1.0** (decides go / no-go; 4 runs x ~4 h).

```bash
python -m src.eval.run_benchmark \
  --model-dir checkpoints/orthrus-qwen3-0.6b-think-8192/step-15000 \
  --tasks gsm8k math500 --limit 200 \
  --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --output-dir results/e1/s1-T1.0 > logs/e1-s1-T1.0.log 2>&1

python -m src.eval.run_benchmark \
  --model-dir checkpoints/stage2/tp-B/final \
  --tasks gsm8k math500 --limit 200 \
  --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --output-dir results/e1/tp-B-T1.0 > logs/e1-tp-B-T1.0.log 2>&1

python -m src.eval.run_benchmark \
  --model-dir checkpoints/stage2/causal-B/final \
  --tasks gsm8k math500 --limit 200 \
  --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --output-dir results/e1/causal-B-T1.0 > logs/e1-causal-B-T1.0.log 2>&1

python -m src.eval.run_benchmark \
  --model-dir checkpoints/stage2/noise-B/final \
  --tasks gsm8k math500 --limit 200 \
  --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --output-dir results/e1/noise-B-T1.0 > logs/e1-noise-B-T1.0.log 2>&1
```

**Priority 2: T = 0.7** (only if T = 1.0 is promising): the same four commands with
`--temperature 0.7` and output dirs `results/e1/<arm>-T0.7`.

**Optional:** the step-500 checkpoints (`checkpoints/stage2/<arm>/step-500`), to see
whether 1000 steps is enough.

**What to read.** The JSON files in each output directory give
`orthrus.acceptance_length` (accepted draft tokens per cycle, bonus excluded) and
`orthrus.survival` (P(L ≥ i) for draft position i). The survival curve is also printed
in the log.

**Decision.**

* `noise-B` beats `tp-B` by **≥ 15–20 %** in acceptance length at T = 1, and the gap
  **grows along the survival curve** (late positions, where the factorization error
  lives) → go to E2.
* Gain ≤ ~5 % → the idea does not survive at scale. Stop.
* `noise-B` < `tp-B`, and `causal-B` is worse still → the causal mask is the problem,
  not the idea. Do not go to E2.
* The gain is only at the first positions → something other than the factorization
  ceiling is at work. Investigate before claiming the mechanism.

---

## E2. Does it beat DBLAST, in real speed?

Two more arms, same data and budget. DBLAST's loss uses the same untruncated target as
the trajectories (`--dblast-target-top-k 0 --dblast-target-top-p 1.0`), not the paper's
T = 0.7 / top-p 0.8 filter, so every arm is trained against the same law.

### Train (one after another)

```bash
python -m src.train.train_qwen3 \
  --model-dir checkpoints/orthrus-qwen3-0.6b-think-8192/step-15000 \
  --packed-cache-path data/processed/stage2-rollouts-t1-8192 \
  --output-dir checkpoints/stage2/dblast-B \
  --objective dblast-al \
  --dblast-k 4 \
  --dblast-tau 0.1 \
  --dblast-target-top-k 0 \
  --dblast-target-top-p 1.0 \
  --eal-temperature 1.0 \
  --seq-len 8192 \
  --num-anchor-blocks 256 \
  --micro-batch-size 1 \
  --grad-accum-steps 32 \
  --epochs 1 \
  --max-steps 1000 \
  --lr 2e-5 \
  --warmup-ratio 0.05 \
  --max-grad-norm 1.0 \
  --dtype bfloat16 \
  --activation-checkpointing \
  --log-every 10 \
  --save-every 500 \
  --seed 42 \
  > logs/train-dblast-B.log 2>&1
```

```bash
python -m src.train.train_qwen3 \
  --model-dir checkpoints/orthrus-qwen3-0.6b-think-8192/step-15000 \
  --packed-cache-path data/processed/stage2-rollouts-t1-8192 \
  --output-dir checkpoints/stage2/dblast1-B \
  --objective dblast-al \
  --dblast-k 1 \
  --dblast-tau 0.1 \
  --dblast-target-top-k 0 \
  --dblast-target-top-p 1.0 \
  --eal-temperature 1.0 \
  --seq-len 8192 \
  --num-anchor-blocks 256 \
  --micro-batch-size 1 \
  --grad-accum-steps 32 \
  --epochs 1 \
  --max-steps 1000 \
  --lr 2e-5 \
  --warmup-ratio 0.05 \
  --max-grad-norm 1.0 \
  --dtype bfloat16 \
  --activation-checkpointing \
  --log-every 10 \
  --save-every 500 \
  --seed 42 \
  > logs/train-dblast1-B.log 2>&1
```

`dblast1-B` (K = 1) is the paper's "DFlash + AL" arm: the AL loss without the latent.
DBLAST with K = 4 computes 4 branch logits per position; if it runs out of memory, lower
`--loss-chunk-size`.

### Choose Z_T for DBLAST on a held-out split

As in the paper: a grid of category temperatures, chosen on held-out examples and then
frozen. Held-out split here: GSM8K problems 201-260 (`--offset 200 --limit 60`), disjoint
from the first 200 used for every reported number. Coarse grid to save time.

```bash
python -m src.eval.run_benchmark --model-dir checkpoints/stage2/dblast-B/final \
  --tasks gsm8k --offset 200 --limit 60 --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --dblast-category-temperature 0.0 --output-dir results/e2/zt-0.0 > logs/e2-zt-0.0.log 2>&1

python -m src.eval.run_benchmark --model-dir checkpoints/stage2/dblast-B/final \
  --tasks gsm8k --offset 200 --limit 60 --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --dblast-category-temperature 0.4 --output-dir results/e2/zt-0.4 > logs/e2-zt-0.4.log 2>&1

python -m src.eval.run_benchmark --model-dir checkpoints/stage2/dblast-B/final \
  --tasks gsm8k --offset 200 --limit 60 --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --dblast-category-temperature 0.8 --output-dir results/e2/zt-0.8 > logs/e2-zt-0.8.log 2>&1

python -m src.eval.run_benchmark --model-dir checkpoints/stage2/dblast-B/final \
  --tasks gsm8k --offset 200 --limit 60 --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --dblast-category-temperature 1.2 --output-dir results/e2/zt-1.2 > logs/e2-zt-1.2.log 2>&1
```

Pick the Z_T with the highest `orthrus.acceptance_length`; call it `ZT`. If two
neighbouring values are close, refine between them on the same held-out problems.

### Compare at T = 1.0 and T = 1.5

Replace `ZT` with the chosen value. The T = 1.0 numbers of `noise-B` and `tp-B` are
reused from E1.

```bash
python -m src.eval.run_benchmark --model-dir checkpoints/stage2/dblast-B/final \
  --tasks gsm8k math500 --limit 200 --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --dblast-category-temperature ZT --output-dir results/e2/dblast-B-T1.0 > logs/e2-dblast-B-T1.0.log 2>&1

python -m src.eval.run_benchmark --model-dir checkpoints/stage2/dblast1-B/final \
  --tasks gsm8k math500 --limit 200 --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --output-dir results/e2/dblast1-B-T1.0 > logs/e2-dblast1-B-T1.0.log 2>&1

python -m src.eval.run_benchmark --model-dir checkpoints/stage2/noise-B/final \
  --tasks gsm8k math500 --limit 200 --temperature 1.5 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --output-dir results/e2/noise-B-T1.5 > logs/e2-noise-B-T1.5.log 2>&1

python -m src.eval.run_benchmark --model-dir checkpoints/stage2/tp-B/final \
  --tasks gsm8k math500 --limit 200 --temperature 1.5 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --output-dir results/e2/tp-B-T1.5 > logs/e2-tp-B-T1.5.log 2>&1

python -m src.eval.run_benchmark --model-dir checkpoints/stage2/dblast-B/final \
  --tasks gsm8k math500 --limit 200 --temperature 1.5 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --dblast-category-temperature ZT --output-dir results/e2/dblast-B-T1.5 > logs/e2-dblast-B-T1.5.log 2>&1

python -m src.eval.run_benchmark --model-dir checkpoints/stage2/dblast1-B/final \
  --tasks gsm8k math500 --limit 200 --temperature 1.5 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --no-baseline \
  --output-dir results/e2/dblast1-B-T1.5 > logs/e2-dblast1-B-T1.5.log 2>&1
```

### Speed: batch 1, with the AR baseline

Without `--no-baseline`: each run also decodes with plain AR on the same GPU and reports
the speedup. Nothing else may run on the GPU meanwhile (check `nvidia-smi`).

```bash
python -m src.eval.run_benchmark --model-dir checkpoints/stage2/tp-B/final \
  --tasks gsm8k --limit 50 --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 \
  --output-dir results/e2/speed-tp-B > logs/e2-speed-tp-B.log 2>&1

python -m src.eval.run_benchmark --model-dir checkpoints/stage2/dblast-B/final \
  --tasks gsm8k --limit 50 --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 --dblast-category-temperature ZT \
  --output-dir results/e2/speed-dblast-B > logs/e2-speed-dblast-B.log 2>&1

python -m src.eval.run_benchmark --model-dir checkpoints/stage2/noise-B/final \
  --tasks gsm8k --limit 50 --temperature 1.0 --top-k 0 --top-p 1.0 \
  --attn-implementation sdpa --dtype bfloat16 \
  --output-dir results/e2/speed-noise-B > logs/e2-speed-noise-B.log 2>&1
```

**Speed of a future 2-pass version, estimated without implementing it.** Noise decoding
now costs 3 passes per cycle. A 2-pass variant (support from an intermediate layer of the
same pass) costs about what plain Orthrus pays per cycle, so

    tok/s(2-pass noise) ≈ tok/s(tp-B) × (L_noise + 1) / (L_tp + 1),

with L the measured `acceptance_length`. Report both the measured 3-pass speed and this
estimate, clearly labelled.

**Decision.**

* `noise-B` > `dblast-B` in acceptance length at T = 1.0 / 1.5 **and** measured speedup
  over AR above `tp-B` → strong paper. Scale up (more GPUs), add DBLAST's benchmarks, and
  the original DFlash as a system baseline.
* Wins on acceptance length but the measured 3-pass speed does not → implement the
  2-pass version first. It is the gate to any submission.
* Loses to DBLAST → the idea is real but not the best fix. A compatibility result (noise
  on top of DBLAST) remains possible, but that is a weaker paper.
