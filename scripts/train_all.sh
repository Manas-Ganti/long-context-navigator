#!/usr/bin/env bash
# Submit sft -> grpo -> eval as one afterok chain. RUN this; do not sbatch it.
#
#   SBATCH_ACCOUNT=ece-6524-spring2026 MAIL_USER=you@vt.edu BASELINES_OK=1 ./scripts/train_all.sh
#   DRY_RUN=1 ./scripts/train_all.sh          # print the sbatch lines, submit nothing
#
# Refuses to run without BASELINES_OK=1: the LLM baselines (no-read at chance,
# single-chunk low, full-document high) must be on disk BEFORE any training,
# or they end up tuned until the trained policy looks good.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${SBATCH_ACCOUNT:?export SBATCH_ACCOUNT=<allocation> first}"
export LCN_DATA="${LCN_DATA:-v1}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export CONDA_ENV="${CONDA_ENV:-$HOME/miniconda3/envs/lcn}"
export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PARTITION="${PARTITION:-a100_normal_q}"     # H200s are hard to get at VT; 7B fits an A100-80 easily
GPU="${GPU:-a100}"
NGPU="${NGPU:-4}"
QOS="${QOS:-tc_${GPU}_normal_short}"
MAIL="${MAIL_USER:+--mail-user=$MAIL_USER}"
TAG="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's|.*/||')"

if [ -z "${BASELINES_OK:-}" ]; then
  cat >&2 <<MSG
refusing to submit: set BASELINES_OK=1 once results/$LCN_DATA/baselines_llm_base.json exists and shows
  no-read ~ chance, single-chunk well below full-document, full-document high.
  JOB=baselines sbatch --account=$SBATCH_ACCOUNT --partition=$PARTITION --qos=$QOS \\
      --gres=gpu:$GPU:1 --mem=96G --time=02:00:00 $MAIL scripts/arc_infer.slurm
MSG
  exit 2
fi
[ -s "data/$LCN_DATA/sft.jsonl" ] || { echo "data/$LCN_DATA/sft.jsonl missing: run 'longctx teacher' on the login node first" >&2; exit 2; }

submit() {  # submit <gres> <time> <script> [sbatch args...]
  local gres="$1" time="$2" script="$3"; shift 3
  local cmd=(sbatch --parsable --account="$SBATCH_ACCOUNT" --partition="$PARTITION" --qos="$QOS"
             --gres="$gres" --time="$time" $MAIL "$@" "$script")
  if [ -n "${DRY_RUN:-}" ]; then echo "DRY ${cmd[*]}" >&2; echo 000000; return; fi
  "${cmd[@]}"
}
mkdir -p logs/slurm
S=$(submit "gpu:$GPU:$NGPU" "${T_SFT:-06:00:00}" scripts/arc_sft.slurm --mem=192G)
echo "sft   $S -> checkpoints/$LCN_DATA/sft-$TAG"
G=$(SFT_CKPT="checkpoints/$LCN_DATA/sft-$TAG" submit "gpu:$GPU:$NGPU" "${T_GRPO:-23:00:00}" scripts/arc_grpo.slurm --mem=192G --dependency=afterok:$S)
echo "grpo  $G -> checkpoints/$LCN_DATA/grpo-$TAG"
E1=$(JOB=eval ADAPTER="checkpoints/$LCN_DATA/sft-$TAG" TAG=sft submit "gpu:$GPU:1" "${T_EVAL:-03:00:00}" scripts/arc_infer.slurm --mem=96G --dependency=afterok:$S)
MERGED="checkpoints/$LCN_DATA/sft-merged-$TAG"
E2=$(JOB=eval MODEL="$MERGED" ADAPTER="checkpoints/$LCN_DATA/grpo-$TAG" TAG=grpo submit "gpu:$GPU:1" "${T_EVAL:-03:00:00}" scripts/arc_infer.slurm --mem=96G --dependency=afterok:$G)
E3=$(JOB=scale MODEL="$MERGED" ADAPTER="checkpoints/$LCN_DATA/grpo-$TAG" TAG=grpo submit "gpu:$GPU:1" "${T_EVAL:-03:00:00}" scripts/arc_infer.slurm --mem=96G --dependency=afterok:$G)
echo "eval  sft=$E1 grpo=$E2 scale=$E3"
echo "cancel all: scancel $S $G $E1 $E2 $E3"
squeue -u "$USER" -o "%.10i %.12j %.2t %.10M %.10L %.6D %R"
