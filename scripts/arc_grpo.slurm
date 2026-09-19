#!/usr/bin/env bash
# Stage 2 — GRPO. One node, one process per GPU; each rank rolls out its own
# instances (HF generate, same weights as the update) and gradients are
# averaged data-parallel. Plain DDP by default; set DS=configs/deepspeed_zero2.json
# to shard optimizer state through accelerate's DeepSpeed plugin.
#
#   SFT_CKPT=checkpoints/v1/sft-qwen2.5-7b-instruct sbatch --gres=gpu:h200:4 ... scripts/arc_grpo.slurm
#   ... scripts/arc_grpo.slurm --max-steps 5          # calibrate wall-clock first
#   CURRICULUM=1 ... scripts/arc_grpo.slurm            # start at 2 hops, escalate
#
# Wall-clock is rollouts: each optimiser step is B instances x G episodes x
# ~min_steps generate rounds (batched). Calibrate with --max-steps 5 before a
# long allocation; a run killed by walltime keeps its last saved adapter
# (--save-every) and resumes with RESUME=<adapter dir>.
#SBATCH --job-name=lcn-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=23:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --output=logs/slurm/grpo-%j.out
#SBATCH --error=logs/slurm/grpo-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

set -euo pipefail
mkdir -p logs/slurm
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/arc_env.sh"
trap arc_notify_finish EXIT
arc_notify "<b>START</b> grpo ${SLURM_JOB_ID:-?} · ${LCN_DATA:-?} · ${SLURM_JOB_NODELIST:-?}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TAG="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's|.*/||')"
SFT_CKPT="${SFT_CKPT:-checkpoints/${LCN_DATA}/sft-${TAG}}"
OUT="${OUT:-checkpoints/${LCN_DATA}/grpo-${TAG}}"
if [ -n "${DS:-}" ]; then
  export ACCELERATE_USE_DEEPSPEED=true
  export ACCELERATE_DEEPSPEED_CONFIG_FILE="$DS"
fi

arc_torchrun -m longctx.train_grpo \
  --model "$MODEL" --adapter "$SFT_CKPT" --out "$OUT" \
  --instances "data/${LCN_DATA}/train.jsonl" \
  ${G:+--group-size "$G"} ${B:+--instances-per-step "$B"} ${LR:+--learning-rate "$LR"} \
  ${KL_BETA:+--kl-beta "$KL_BETA"} ${CURRICULUM:+--curriculum} ${RESUME:+--resume "$RESUME"} \
  ${SAVE_EVERY:+--save-every "$SAVE_EVERY"} \
  "$@"
