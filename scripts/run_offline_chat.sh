#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${LAB_ROOT}/configs/qwen3-0.6b.env"

export HF_HOME MODEL_ID MODEL_REVISION SERVED_MODEL_NAME
export LAB_DTYPE LAB_MAX_MODEL_LEN LAB_GPU_MEMORY_UTILIZATION LAB_MAX_NUM_SEQS
export VLLM_USE_FLASHINFER_SAMPLER

exec "${LAB_ROOT}/src/vllm/.venv/bin/python" \
  "${LAB_ROOT}/experiments/01_offline_chat.py"
