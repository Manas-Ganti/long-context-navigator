#!/usr/bin/env bash
# One-time: build the `lcn` conda env on the ARC login node.
#
#   bash scripts/arc_setup_env.sh                 # -> /home/$USER/miniconda3/envs/lcn
#
# One env is enough here (unlike the sibling VLM project): nothing depends on
# TRL, so vLLM's transformers pin does not fight anything. vLLM and torch must
# come from the same resolve — install vllm FIRST and let it pick torch.
# Do NOT `import deepspeed` on the login node to verify (Triton needs a GPU);
# `pip list` is the check for that one.
set -euo pipefail
PREFIX="${PREFIX:-/home/$USER/miniconda3/envs/lcn}"
module reset >/dev/null 2>&1 || true
module load Miniforge3 >/dev/null 2>&1 || module load Anaconda3 >/dev/null 2>&1 || true

if [ ! -x "$PREFIX/bin/python" ]; then
  conda create -y -p "$PREFIX" python=3.11
fi
PY="$PREFIX/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "vllm>=0.8"
"$PY" -m pip install "transformers>=4.51" "peft>=0.14" "accelerate>=1.4" \
                     "deepspeed>=0.16" "wandb>=0.18" "hf-transfer>=0.1" \
                     "datasets>=3.0"          # real-text substrates (MuSiQue)
"$PY" -m pip install -e "$(dirname "${BASH_SOURCE[0]}")/.."[test]

echo "--- verify (no GPU needed) ---"
"$PY" -V
"$PY" -c "import torch, vllm, transformers, peft, accelerate; print('torch', torch.__version__, torch.version.cuda, '| vllm', vllm.__version__, '| transformers', transformers.__version__)"
"$PY" -m pip list 2>/dev/null | grep -i deepspeed
"$PY" -c "import longctx; from longctx.cli import main; print('longctx OK')"
echo "--- next: HF_HOME=/home/$USER/hf_cache $PY -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')\""
