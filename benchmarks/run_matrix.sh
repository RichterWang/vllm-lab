#!/usr/bin/env bash
set -euo pipefail

# big letter represent env params
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${LAB_ROOT}/configs/qwen3-0.6b.env"

PYTHON="${LAB_ROOT}/src/vllm/.venv/bin/python"
VLLM="${LAB_ROOT}/src/vllm/.venv/bin/vllm"
MATRIX="${1:-${LAB_ROOT}/benchmarks/matrix.yaml}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git -C "${LAB_ROOT}/src/vllm" rev-parse --short HEAD)" # hash time stamp
RUN_DIR="${LAB_ROOT}/results/benchmark/${RUN_ID}"
SERVER_PID=""

# create rusult dir
mkdir -p "${RUN_DIR}"

# cleanup_server cleans up the server process if it is running
cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

# wait_for_server waits for the server to become healthy, returning 0 if it does, or 1 if it fails
wait_for_server() {
  local attempt
  for attempt in $(seq 1 180); do
    if curl --silent --fail "http://${LAB_HOST}:${LAB_PORT}/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

# port check: if the port is already in use, exit with an error
if curl --silent --fail "http://${LAB_HOST}:${LAB_PORT}/health" >/dev/null 2>&1; then
  echo "Port ${LAB_HOST}:${LAB_PORT} already has a healthy service; stop it before running the matrix." >&2
  exit 1
fi

# get modes from the matrix and run each mode
mapfile -t MODES < <("${PYTHON}" "${LAB_ROOT}/benchmarks/run_benchmark.py" \
  --matrix "${MATRIX}" --print-modes)

# main loop: run each mode in the matrix
for mode in "${MODES[@]}"; do
  # get mode args from the matrix
  MODE_ARGS=()
  while IFS= read -r argument; do
    if [[ -n "${argument}" ]]; then
      MODE_ARGS+=("${argument}")
    fi
  done < <("${PYTHON}" "${LAB_ROOT}/benchmarks/run_benchmark.py" \
    --matrix "${MATRIX}" --print-mode-args "${mode}")

  # start new server for this mode
  LOG_PATH="${RUN_DIR}/server_${mode}.log"
  echo "Starting mode ${mode}; log: ${LOG_PATH}"

  export HF_HOME VLLM_USE_FLASHINFER_SAMPLER
  setsid "${VLLM}" serve "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${LAB_HOST}" \
    --port "${LAB_PORT}" \
    --dtype "${LAB_DTYPE}" \
    --max-model-len "${LAB_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${LAB_GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${LAB_MAX_NUM_SEQS}" \
    --generation-config vllm \
    "${MODE_ARGS[@]}" >"${LOG_PATH}" 2>&1 &
  SERVER_PID=$!

  # wait for the server be ready
  if ! wait_for_server; then
    echo "Server failed to become healthy in mode ${mode}. See ${LOG_PATH}." >&2
    cleanup_server
    exit 1
  fi

  # conduct benchmark test
  "${PYTHON}" "${LAB_ROOT}/benchmarks/run_benchmark.py" \
    --matrix "${MATRIX}" \
    --mode "${mode}" \
    --run-dir "${RUN_DIR}"
  cleanup_server
done

# analyze results
"${PYTHON}" "${LAB_ROOT}/benchmarks/analyze_results.py" "${RUN_DIR}"
echo "Benchmark complete: ${RUN_DIR}"
