#!/usr/bin/env bash
# Stage 1 — SFT on teacher trajectories. One node, data-parallel, LoRA, ZeRO-2.
#
#   sbatch --gres=gpu:h200:4 ... scripts/arc_sft.slurm
#   EPOCHS=1 sbatch ... scripts/arc_sft.slurm --max-steps 50     # calibration
#
# Teacher rows are built on the login node first (CPU only):
#   longctx teacher --instances data/v1/train.jsonl --out data/v1/sft.jsonl
#SBATCH --job-name=lcn-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --output=logs/slurm/sft-%j.out
#SBATCH --error=logs/slurm/sft-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

set -euo pipefail
mkdir -p logs/slurm
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/arc_env.sh"
trap arc_notify_finish EXIT
arc_notify "<b>START</b> sft ${SLURM_JOB_ID:-?} · ${LCN_DATA:-?} · ${SLURM_JOB_NODELIST:-?}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DS="${DS:-configs/deepspeed_zero2.json}"
TAG="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's|.*/||')"

arc_torchrun -m longctx.train_sft \
  --model "$MODEL" --deepspeed "$DS" \
  --sft-data "data/${LCN_DATA}/sft.jsonl" \
  --out "checkpoints/${LCN_DATA}/sft-${TAG}" \
  ${EPOCHS:+--epochs "$EPOCHS"} ${LR:+--learning-rate "$LR"} \
  ${PER_DEVICE_BS:+--per-device-batch-size "$PER_DEVICE_BS"} ${GRAD_ACCUM:+--grad-accum "$GRAD_ACCUM"} \
  "$@"
