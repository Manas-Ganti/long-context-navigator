#!/usr/bin/env bash
# Offline inference on ARC: LLM baselines and evaluation through vLLM. One GPU
# is enough for a 7B policy (TP=1); the 7B-class model fits on an A100-80 or H200.
#
#   JOB=baselines sbatch ... scripts/arc_infer.slurm                    # base model, 3 conditions, answer-only
#   JOB=baselines REASONING=1 sbatch ... scripts/arc_infer.slurm        # same, model may reason first
#   JOB=eval      sbatch ... scripts/arc_infer.slurm                    # base model through the env
#   JOB=eval ADAPTER=checkpoints/v1/sft-qwen2.5-7b-instruct sbatch ...  # SFT policy
#   # GRPO adapters sit on the MERGED SFT model, so pass that as MODEL:
#   JOB=eval MODEL=checkpoints/v1/sft-merged-qwen2.5-7b-instruct \
#       ADAPTER=checkpoints/v1/grpo-qwen2.5-7b-instruct TAG=grpo sbatch ...
#
# Always pass on the submit line (see arc_quickref.md): --account, --partition,
# --qos=*_short, --gres=gpu:<type>:1, an explicit --mem, --time, --mail-user.
#SBATCH --job-name=lcn-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --output=logs/slurm/infer-%j.out
#SBATCH --error=logs/slurm/infer-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

set -euo pipefail
mkdir -p logs/slurm
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/arc_env.sh"
scontrol update jobid="${SLURM_JOB_ID}" name="lcn-${JOB:-eval}" >/dev/null 2>&1 || true
trap arc_notify_finish EXIT
arc_notify "<b>START</b> ${JOB:-eval} ${SLURM_JOB_ID:-?} · ${LCN_DATA:-?} · ${SLURM_JOB_NODELIST:-?}"

JOB="${JOB:-eval}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TP="${TP:-1}"
DATA="data/${LCN_DATA}"
INSTANCES="${INSTANCES:-$DATA/id_test.jsonl $DATA/ood_test.jsonl}"
TAG="${TAG:-${ADAPTER:+adapter}}"; TAG="${TAG:-base}"
mkdir -p "results/${LCN_DATA}"

case "$JOB" in
  baselines)
    "$PY" -m longctx.cli baselines --mode llm --backend vllm --model "$MODEL" --tp "$TP" \
      ${ADAPTER:+--adapter "$ADAPTER"} --instances $INSTANCES ${REASONING:+--reasoning} \
      --out "results/${LCN_DATA}/baselines_llm_${TAG}${REASONING:+_reasoning}.json" "$@"
    ;;
  eval)
    "$PY" -m longctx.cli evaluate --policy llm --backend vllm --model "$MODEL" --tp "$TP" \
      ${ADAPTER:+--adapter "$ADAPTER"} --instances $INSTANCES --batch-size "${BATCH:-64}" \
      ${GREEDY:+--greedy} --out "results/${LCN_DATA}/eval_${TAG}" "$@"
    "$PY" -m longctx.cli audit --instances $INSTANCES \
      --trajectories "results/${LCN_DATA}/eval_${TAG}.trajectories.jsonl" \
      --out "results/${LCN_DATA}/audit_${TAG}.json"
    ;;
  scale)
    "$PY" -m longctx.cli evaluate --policy llm --backend vllm --model "$MODEL" --tp "$TP" \
      ${ADAPTER:+--adapter "$ADAPTER"} --instances "$DATA/scale_test.jsonl" --batch-size "${BATCH:-32}" \
      ${GREEDY:+--greedy} --out "results/${LCN_DATA}/scale_${TAG}" "$@"
    ;;
  *)
    echo "JOB must be one of: baselines | eval | scale" >&2; exit 2 ;;
esac
