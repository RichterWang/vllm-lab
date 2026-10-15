#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LAB_ROOT}/configs/qwen3-0.6b.env"
PYTHON="${LAB_ROOT}/src/vllm/.venv/bin/python"
VLLM="${LAB_ROOT}/src/vllm/.venv/bin/vllm"
RUN_DIR="${LAB_ROOT}/results/issue56455/$(date -u +%Y%m%dT%H%M%SZ)"
SERVER_PID=""

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

if curl --silent --fail "http://${LAB_HOST}:${LAB_PORT}/health" >/dev/null 2>&1; then
  echo "Port ${LAB_HOST}:${LAB_PORT} already has a vLLM service" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
for arm in stock pr_disabled pr_slots2; do
  case "${arm}" in
    stock)
      TREE="${LAB_ROOT}/worktrees/vllm-56455-base"
      ARGS=()
      ;;
    pr_disabled)
      TREE="${LAB_ROOT}/worktrees/vllm-56455-pr"
      ARGS=(--prefill-admission-slots 0)
      ;;
    pr_slots2)
      TREE="${LAB_ROOT}/worktrees/vllm-56455-pr"
      ARGS=(--prefill-admission-slots 2)
      ;;
  esac

  SHA="$(git -C "${TREE}" rev-parse HEAD)"
  echo "${arm}: ${SHA}"
  printf '%s\n' "${SHA}" > "${RUN_DIR}/${arm}.sha"
  "${LAB_ROOT}/benchmarks/issue56455/link_extensions.sh" "${TREE}"
  export PYTHONPATH="${TREE}${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME VLLM_USE_FLASHINFER_SAMPLER
  setsid "${VLLM}" serve "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${LAB_HOST}" --port "${LAB_PORT}" \
    --dtype "${LAB_DTYPE}" \
    --max-model-len "${LAB_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${LAB_GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs 8 --max-num-batched-tokens 512 \
    --generation-config vllm \
    "${ARGS[@]}" > "${RUN_DIR}/${arm}.server.log" 2>&1 &
  SERVER_PID=$!

  healthy=0
  for attempt in $(seq 1 180); do
    if curl --silent --fail "http://${LAB_HOST}:${LAB_PORT}/health" >/dev/null; then
      healthy=1
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if [[ "${healthy}" != 1 ]]; then
    echo "${arm} failed to start; see ${RUN_DIR}/${arm}.server.log" >&2
    exit 1
  fi

  "${PYTHON}" "${LAB_ROOT}/benchmarks/issue56455/stream_probe.py" \
    --url "http://${LAB_HOST}:${LAB_PORT}" \
    --model "${SERVED_MODEL_NAME}" \
    --arm "${arm}" \
    --requests "${ISSUE56455_REQUESTS:-32}" \
    --concurrency "${ISSUE56455_CONCURRENCY:-16}" \
    --output "${RUN_DIR}/${arm}.json"
  cleanup_server
  unset PYTHONPATH
done

echo "Results: ${RUN_DIR}"
