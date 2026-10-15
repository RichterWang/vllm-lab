#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LAB_ROOT}/src/vllm/.venv/bin/python"
VLLM="${LAB_ROOT}/src/vllm/.venv/bin/vllm"
MODEL="/workspace/issue56455/model"
DATASET="/workspace/issue56455/fixed/sharegpt_2000.json"
RESULTS="${ISSUE56455_RUN_ROOT:-${LAB_ROOT}/results/issue56455/h100_$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${ISSUE56455_PORT:-8001}"
PROMPTS="${ISSUE56455_NUM_PROMPTS:-2000}"
CONCURRENCY="${ISSUE56455_CONCURRENCY:-256}"
ROUNDS="${ISSUE56455_ROUNDS:-3}"
ARMS="${ISSUE56455_ARMS:-stock,pr0,pr25}"
LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH
SERVER_PID=""
MONITOR_PID=""

cleanup() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill -TERM "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap cleanup EXIT INT TERM

[[ -x "$PYTHON" && -x "$VLLM" && -f "$DATASET" ]] || {
  echo "Missing Python, vLLM, or frozen dataset" >&2
  exit 1
}
mkdir -p "$RESULTS"
printf 'run_root=%s\nprompts=%s\nconcurrency=%s\nrounds=%s\narms=%s\n' \
  "$RESULTS" "$PROMPTS" "$CONCURRENCY" "$ROUNDS" "$ARMS" > "$RESULTS/run_config.txt"
sha256sum "$DATASET" > "$RESULTS/dataset.sha256"

run_arm() {
  local round="$1" arm="$2" tree slots=()
  case "$arm" in
    stock) tree="$LAB_ROOT/worktrees/vllm-56455-base" ;;
    pr0) tree="$LAB_ROOT/worktrees/vllm-56455-pr"; slots=(--prefill-admission-slots 0) ;;
    pr25) tree="$LAB_ROOT/worktrees/vllm-56455-pr"; slots=(--prefill-admission-slots 25) ;;
    *) echo "Unknown arm: $arm" >&2; exit 2 ;;
  esac
  local output="$RESULTS/r${round}_${arm}"
  mkdir -p "$output"
  git -C "$tree" rev-parse HEAD > "$output/source.sha"
  "$LAB_ROOT/benchmarks/issue56455/link_extensions.sh" "$tree"
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "Port $PORT already serves a model" >&2
    exit 1
  fi

  local server=("$VLLM" serve "$MODEL" --served-model-name Qwen/Qwen2.5-7B-Instruct
    --host 127.0.0.1 --port "$PORT" --dtype bfloat16
    --max-num-seqs 128 --max-num-batched-tokens 4096
    --enable-chunked-prefill --generation-config vllm "${slots[@]}")
  printf '%q ' "${server[@]}" > "$output/server.command"
  printf '\n' >> "$output/server.command"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$output/server_start_utc.txt"
  setsid env PYTHONPATH="$tree" VLLM_USE_V2_MODEL_RUNNER=1 \
    HF_HOME=/workspace/issue56455/hf_cache "${server[@]}" \
    > "$output/server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0
  for ((attempt=0; attempt<900; attempt++)); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > "$output/health.txt" 2>/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "$arm server failed: $output/server.log" >&2
    exit 1
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$output/server_ready_utc.txt"
  curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$output/models.json"
  curl -fsS "http://127.0.0.1:$PORT/metrics" > "$output/metrics_before.prom"
  curl -fsSN "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen2.5-7B-Instruct","prompt":"vLLM health probe","max_tokens":8,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}' \
    > "$output/stream_smoke.txt"
  grep -q '^data:' "$output/stream_smoke.txt"

  "$PYTHON" "$LAB_ROOT/benchmarks/issue56455/metrics_monitor.py" \
    --url "http://127.0.0.1:$PORT/metrics" \
    --output "$output/metrics.jsonl" --interval 0.5 &
  MONITOR_PID=$!
  local benchmark=("$VLLM" bench serve --backend openai
    --host 127.0.0.1 --port "$PORT" --endpoint /v1/completions
    --model Qwen/Qwen2.5-7B-Instruct --tokenizer "$MODEL"
    --dataset-name sharegpt --dataset-path "$DATASET" --disable-shuffle
    --num-prompts "$PROMPTS" --seed 618 --temperature 0
    --request-rate inf --max-concurrency "$CONCURRENCY"
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99
    --save-result --save-detailed --result-dir "$output"
    --result-filename bench.json)
  printf '%q ' "${benchmark[@]}" > "$output/benchmark.command"
  printf '\n' >> "$output/benchmark.command"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$output/benchmark_start_utc.txt"
  env PYTHONPATH="$LAB_ROOT/src/vllm" HF_HOME=/workspace/issue56455/hf_cache \
    "${benchmark[@]}" > "$output/benchmark.log" 2>&1
  date -u +%Y-%m-%dT%H:%M:%SZ > "$output/benchmark_end_utc.txt"
  cleanup
  sleep 3
  echo "Completed r${round}_${arm}: $output"
}

for ((round=1; round<=ROUNDS; round++)); do
  case "$round" in
    1) order=(stock pr0 pr25) ;;
    2) order=(pr0 pr25 stock) ;;
    *) order=(pr25 stock pr0) ;;
  esac
  for arm in "${order[@]}"; do
    if [[ ",$ARMS," == *",$arm,"* ]]; then
      run_arm "$round" "$arm"
    fi
  done
done
echo "Results: $RESULTS"
