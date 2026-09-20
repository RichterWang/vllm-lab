#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${LAB_ROOT}/configs/qwen3-0.6b.env"

export HF_HOME VLLM_USE_FLASHINFER_SAMPLER
exec "${LAB_ROOT}/src/vllm/.venv/bin/vllm" serve "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${LAB_HOST}" \
  --port "${LAB_PORT}" \
  --dtype "${LAB_DTYPE}" \
  --max-model-len "${LAB_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${LAB_GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${LAB_MAX_NUM_SEQS}" \
  --enforce-eager \
  --generation-config vllm
