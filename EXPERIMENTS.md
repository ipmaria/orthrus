# Go / no-go experiments for noise-conditioned drafting

Two experiments decide whether this is a strong paper: **E1** "is there a signal, and
does it come from the claimed mechanism", **E2** "does it beat the direct competitor
(DBLAST) in real speed". Everything else (scale, more benchmarks, ablations over m) is
only worth doing if both are positive.

Setup used below: Qwen3-1.7B target, 1 node x 8 GPUs, stage-1 Orthrus checkpoint `$S1`,
init checkpoint `$INIT` (tokenizer). All arms start from `$S1` with the same data, steps,
learning rate and seed.

```bash
export S1=checkpoints/orthrus-qwen3-1.7b-kl/final
export INIT=checkpoints/orthrus-qwen3-1.7b-init
export TEACHER=Qwen/Qwen3-1.7B
```

## 0. Sanity (blocking)

```bash
python -m pytest tests/ -q
python scripts/verify_estimators_numpy.py            # torch-free math checks
python scripts/check_noise_decoding.py --model-dir $S1 --enable-top-m 32 --samples 4000
```

The last command attaches a zero noise projection to the untrained checkpoint and checks
the whole noise verification path. **Stop if `TV(noise, AR)` is clearly above the
`TV(AR, AR)` noise floor**: that is a bug and every later number would be wrong.
With `--fsdp2`, after the first ~10 steps of any `noise-eal` / `dblast-*` run, check
that `noise_proj_diff` / `dblast_*` weights are identical on all ranks (they are
replicated and all-reduced by hand).

## 1. Data: teacher rollouts at T = 1, no truncation (corpus B)

Every arm trains on the same corpus. Untruncated sampling makes the target-path losses
exact (coupling), matches what the losses compute, and gives the posterior noise its
prior law.

```bash
python scripts/build_orthrus_corpus.py --per-domain 50000 --balance --out data/prompts.jsonl
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python scripts/generate_onpolicy.py \
    --in data/prompts.jsonl --out data/B.$i.jsonl --model $TEACHER \
    --temperature 1.0 --top-p 1.0 --top-k -1 \
    --num-shards 8 --shard-id $i &      # vLLM: top-k -1 (or 0 on newer versions) = off
done; wait
cat data/B.?.jsonl > data/B.jsonl
python scripts/build_packed_cache_ids.py --dataset-path data/B.jsonl \
  --out data/B-packed --model-dir $INIT --seq-len 2048
```

Shared helpers:

```bash
train() { torchrun --nproc_per_node=8 -m src.train.train_qwen3 --model-dir $S1 \
  --packed-cache-path data/B-packed --seq-len 2048 --epochs 1 --max-steps 2000 \
  --lr 2e-5 --eal-temperature 1.0 --micro-batch-size 1 --grad-accum-steps 16 \
  --fsdp2 --activation-checkpointing --seed 42 "$@"; }

# $1 = checkpoint dir, $2 = temperature, rest = extra flags. No top-k / top-p.
evalm() { local ck=$1 t=$2; shift 2
  python -m src.eval.run_benchmark --model-dir $ck --tasks gsm8k math500 \
    --temperature $t --top-k 0 --top-p 1.0 --attn-implementation sdpa --no-baseline \
    --output-dir results/$(basename $(dirname $ck))-T$t "$@"; }
```

`--no-baseline` skips the AR run during the sweep. For the speed numbers in E2, run
once without it.

## E1. Is there a signal, and is it the claimed mechanism?

Four arms:

| arm | what it isolates |
|---|---|
| `s1` | reference: stage-1, no stage 2 |
| `tp-B` | "just fine-tuned on teacher rollouts" (same data, same budget) |
| `causal-B` | the cost of the causal in-block mask alone (no noise) |
| `noise-B` | the method |

```bash
train --objective accepted-length --eal-estimator target-path --output-dir ck/tp-B
train --objective accepted-length --eal-estimator target-path --causal-drafting \
      --output-dir ck/causal-B
train --objective noise-eal --noise-reveal-top-m 32 --output-dir ck/noise-B

for T in 1.0 0.7; do
  evalm $S1 $T
  for r in tp-B causal-B noise-B; do evalm ck/$r/final $T; done
done
python scripts/check_noise_decoding.py --model-dir ck/noise-B/final    # still lossless?
```

Read in the JSON files `run_benchmark` writes to each output dir: `orthrus.acceptance_length` (accepted draft tokens
per cycle, bonus excluded) and `orthrus.survival` (P(L ≥ i) for draft position i; the
same curve is printed during the run).

**Decision.**
- `noise-B` beats `tp-B` by **≥ 15–20 %** in acceptance length at T = 1, and the gap
  **grows along the survival curve** (late positions, where the factorization error
  lives) → go to E2.
- Gain ≤ ~5 % → the idea does not survive at scale. Stop.
- `noise-B` < `tp-B` and `causal-B` is worse still → the causal mask is the problem, not
  the idea. Next step: inject the noise without the mask (needs a different lossless
  argument) or a cheaper causal variant. Do not go to E2.
- The gain shows up only at the first positions → something other than the
  factorization ceiling is at work. Investigate before claiming the mechanism.

## E2. Does it beat DBLAST, in real speed?

Two more arms on the same data and budget. For an apples-to-apples comparison, DBLAST's
loss uses the same untruncated target as the trajectories (corpus B), not the paper's
T = 0.7 / top-p 0.8 filter.

```bash
train --objective dblast-al --dblast-k 4 --dblast-target-top-k 0 --dblast-target-top-p 1.0 \
      --output-dir ck/dblast-B
train --objective dblast-al --dblast-k 1 --dblast-target-top-k 0 --dblast-target-top-p 1.0 \
      --output-dir ck/dblast1-B          # 'DFlash + AL' arm of the paper

# Z_T for DBLAST: choose on a held-out split (paper: 10 %), then freeze.
for z in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4; do
  evalm ck/dblast-B/final 1.0 --limit 130 --dblast-category-temperature $z \
        --output-dir results/zt-$z
done
# -> pick the best Z_T by acceptance_length, then evaluate on the rest with it.

for T in 1.0 1.5; do
  evalm ck/dblast-B/final $T --dblast-category-temperature $ZT
  evalm ck/dblast1-B/final $T
  evalm ck/noise-B/final $T
  evalm ck/tp-B/final $T
done

# Speed: batch 1, same GPU, WITH the AR baseline (drop --no-baseline): speedup vs AR.
for r in tp-B dblast-B noise-B; do
  extra=(); [ "$r" = dblast-B ] && extra=(--dblast-category-temperature "$ZT")
  python -m src.eval.run_benchmark --model-dir ck/$r/final --tasks gsm8k --limit 200 \
    --temperature 1.0 --top-k 0 --top-p 1.0 --attn-implementation sdpa \
    --output-dir results/speed-$r "${extra[@]}"
done
```

Held-out split: the Z_T sweep uses the first 130 GSM8K examples (`--limit 130`).
Exclude them from the final DBLAST evaluation, or use another split.

**Speed of a future 2-pass version without implementing it.** Noise decoding now costs
3 passes per cycle. A 2-pass variant (support taken from an intermediate layer of the
same pass) would cost about what plain Orthrus pays per cycle, so

    tok/s(2-pass noise) ≈ tok/s(tp-B) × (L_noise + 1) / (L_tp + 1),

with L the measured acceptance lengths. Report both the measured 3-pass speed and this
estimate, clearly labelled.

**Decision.**
- `noise-B` > `dblast-B` in acceptance length at T = 1.0 / 1.5 **and** measured
  speedup over AR above `tp-B` → strong paper. Scale to 4B / 8B, add DBLAST's
  benchmarks (HumanEval, MT-Bench) and the original DFlash system baseline.
- Wins on acceptance length but the measured 3-pass speed does not → implement the
  2-pass version first; it is the gate to any submission.
- Loses to DBLAST → the idea is real but not the best fix. A compatibility result
  (noise on top of DBLAST) is still possible, but that is a weaker paper.

## Budget

E1: 3 trainings of 2000 steps + 7 eval runs per temperature. E2: 2 trainings + the Z_T
sweep + evals. Roughly a few days on one 8-GPU node at 1.7B. E1 alone answers the first
question; run E2 only if E1 says go.
