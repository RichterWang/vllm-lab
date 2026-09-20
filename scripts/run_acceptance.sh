#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${LAB_ROOT}/configs/qwen3-0.6b.env"

PYTHON_BIN="${LAB_ROOT}/src/vllm/.venv/bin/python"
SERVER_PID=""

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT "${SERVER_PID}" 2>/dev/null || true
    for _ in {1..80}; do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 0.25
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

trap cleanup_server EXIT INT TERM

"${PYTHON_BIN}" - <<'PY'
import os
import socket

host = os.environ["LAB_HOST"]
if host in {"0.0.0.0", "::"}:
    host = "127.0.0.1"
port = int(os.environ["LAB_PORT"])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.settimeout(0.5)
    if client.connect_ex((host, port)) == 0:
        raise SystemExit(f"Refusing to start: {host}:{port} is already in use")
PY

"${LAB_ROOT}/scripts/download_model.sh"
"${LAB_ROOT}/scripts/run_offline_chat.sh"

mkdir -p "${LAB_ROOT}/results"
"${LAB_ROOT}/scripts/serve_qwen3_0_6b.sh" \
  >"${LAB_ROOT}/results/server.log" 2>&1 &
SERVER_PID=$!

"${PYTHON_BIN}" "${LAB_ROOT}/scripts/smoke_openai_api.py"
"${PYTHON_BIN}" "${LAB_ROOT}/scripts/concurrent_batch_test.py"

cleanup_server
"${PYTHON_BIN}" "${LAB_ROOT}/scripts/verify_acceptance.py"
trap - EXIT INT TERM
