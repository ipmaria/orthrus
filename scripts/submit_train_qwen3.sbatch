#!/bin/bash
# Orthrus-Qwen3 training at the paper's Table 4 configuration.
#
#   MODEL_DIR=pretrained-models/orthrus-qwen3-8b-init sbatch scripts/submit_train_qwen3.sbatch
#
# Global batch = micro_batch(1) x grad_accum(16) x 8 ranks = 128 sequences, matching
# Table 4. Appendix A reports under 24h on one 8xH200 node for 2 epochs over ~472K
# packed rows (0.96B tokens).
#SBATCH --job-name=orthrus-qwen3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.out

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-.cache/torchinductor}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" slurm-logs

MODEL_DIR="${MODEL_DIR:-pretrained-models/orthrus-qwen3-8b-init}"
CACHE_PATH="${CACHE_PATH:-data/processed/orthrus-packed-2048-asst}"
OUT_DIR="${OUT_DIR:-checkpoints/orthrus-qwen3-8b}"
OBJECTIVE="${OBJECTIVE:-kl}"   # `ce` reproduces the Table 5 ablation arm

echo "node=$(hostname) model=$MODEL_DIR objective=$OBJECTIVE out=$OUT_DIR"

srun torchrun --nproc_per_node=8 -m src.train.train_qwen3 \
  --model-dir "$MODEL_DIR" \
  --packed-cache-path "$CACHE_PATH" \
  --output-dir "$OUT_DIR" \
  --objective "$OBJECTIVE" \
  --seq-len 2048 \
  --num-anchor-blocks 256 \
  --micro-batch-size 1 \
  --grad-accum-steps 16 \
  --epochs 2 \
  --lr 2e-4 \
  --warmup-ratio 0.05 \
  --max-grad-norm 1.0 \
  --dtype bfloat16 \
  --fsdp2 \
  --activation-checkpointing \
  --log-every 10 \
  --save-every 2500 \
  --keep-last-checkpoints 3 \
  --auto-resume
