#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${LAB_ROOT}/configs/qwen3-0.6b.env"

HF_BIN="${LAB_ROOT}/src/vllm/.venv/bin/hf"
if [[ ! -x "${HF_BIN}" ]]; then
  echo "Missing Hugging Face CLI: ${HF_BIN}" >&2
  exit 1
fi

export HF_HOME
exec "${HF_BIN}" download "${MODEL_ID}" --revision "${MODEL_REVISION}"
