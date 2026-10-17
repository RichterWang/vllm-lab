#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible H100 serving comparison for PR #61991. Run inside the H100 Pod.
LAB=/workspace/vllm_lab
RUN=/tmp/issue61991-run
MODEL="$RUN/model"
PARENT_ENV="$RUN/env-parent"
MAIN_ENV="$RUN/env-main"
PR_TREE="$LAB/worktrees/vllm-61991-pr"
RESULTS="$LAB/results/serve/issue61991/h100_20261016T152655Z"
PORT=8000
MODEL_NAME=qwen3-0.6b

PARENT_SHA=fa75d3456d60a5ffe923818c4b495295675b38e1
PR_SHA=b2625fd3fd7ea896940c075b8cc132d9e6bbaee3
MAIN_SHA=cffc0cfa311f4997be58cfe8f5f6473ab4898269

PKG="$PARENT_ENV/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/logprob.py"
PARENT_LOGPROB="$RUN/logprob-parent.py"
PR_LOGPROB="$RUN/logprob-pr.py"
SERVER_PID_FILE="$RUN/server.pid"
FAIL=0

mkdir -p "$RESULTS/logs" "$RESULTS/raw" "$RESULTS/telemetry"
if [[ -e "$RESULTS/run_status.csv" ]]; then
  echo "Refusing to overwrite an existing matrix run: $RESULTS/run_status.csv" >&2
  exit 2
fi
cp "$PKG" "$PARENT_LOGPROB"
cp "$PR_TREE/vllm/v1/worker/gpu/sample/logprob.py" "$PR_LOGPROB"

server_env() {
  local env_dir=$1
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH="$env_dir/bin:/usr/local/cuda-12.8/bin:$PATH"
  export HF_HOME="$RUN/hf-cache"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_FLASHINFER_SAMPLER=0
  export TRITON_CACHE_DIR="$RUN/triton-cache"
}

stop_server() {
  if [[ -f "$SERVER_PID_FILE" ]]; then
    local pid
    pid=$(cat "$SERVER_PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid"
      for _ in $(seq 1 90); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$pid" 2>/dev/null; then
        echo "Server PID $pid did not exit after SIGTERM" >&2
        kill -KILL "$pid" || true
        FAIL=1
      fi
    fi
    rm -f "$SERVER_PID_FILE"
  fi
  for _ in $(seq 1 30); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [[ "${used:-0}" -lt 1000 ]] && break
    sleep 1
  done
  if [[ "${used:-0}" -ge 1000 ]]; then
    echo "GPU memory did not clear after stopping the server (used=${used} MiB)" >&2
    FAIL=1
    return 1
  fi
}

start_server() {
  local arm=$1 round=$2 env_dir=$3
  local log="$RESULTS/logs/${round}_${arm}_server.log"
  server_env "$env_dir"
  stop_server
  nohup "$env_dir/bin/python" -m vllm.entrypoints.launchers.api_server.entry \
    "$MODEL_NAME" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --served-model-name "$MODEL_NAME" --tokenizer "$MODEL" --dtype float16 \
    --max-model-len 4096 --max-num-seqs 64 --gpu-memory-utilization 0.80 \
    --enforce-eager --disable-log-requests >"$log" 2>&1 </dev/null &
  echo $! > "$SERVER_PID_FILE"
  for _ in $(seq 1 360); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "ready arm=$arm round=$round pid=$(cat "$SERVER_PID_FILE")"
      return 0
    fi
    if ! kill -0 "$(cat "$SERVER_PID_FILE")" 2>/dev/null; then
      echo "Server exited before readiness; see $log" >&2
      tail -80 "$log" >&2
      return 1
    fi
    sleep 1
  done
  echo "Server readiness timeout; see $log" >&2
  tail -80 "$log" >&2
  return 1
}

run_arm() {
  local arm=$1 round=$2 env_dir=$3 commit=$4
  local -a concurrencies
  case "$round" in
    1) concurrencies=(1 16 64) ;;
    2) concurrencies=(64 1 16) ;;
    3) concurrencies=(16 64 1) ;;
    *) echo "Invalid round: $round" >&2; return 2 ;;
  esac

  case "$arm" in
    parent) cp "$PARENT_LOGPROB" "$PKG" ;;
    pr) cp "$PR_LOGPROB" "$PKG" ;;
    main) : ;;
    *) echo "Invalid arm: $arm" >&2; return 2 ;;
  esac

  if ! start_server "$arm" "r$round" "$env_dir"; then
    FAIL=1
    return 0
  fi

  local telemetry="$RESULTS/telemetry/r${round}_${arm}.csv"
  server_env "$env_dir"
  nvidia-smi --query-gpu=timestamp,name,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader -l 5 >"$telemetry" 2>&1 &
  local telemetry_pid=$!

  for concurrency in "${concurrencies[@]}"; do
    local file="r${round}-c${concurrency}.json"
    local log="$RESULTS/logs/r${round}_${arm}_c${concurrency}_bench.log"
    if "$PARENT_ENV/bin/python" "$RUN/run_vllm_bench.py" \
      --backend openai --host 127.0.0.1 --port "$PORT" \
      --endpoint /v1/completions --model "$MODEL_NAME" --tokenizer "$MODEL" \
      --dataset-name random --random-input-len 128 --random-output-len 128 \
      --num-prompts 200 --num-warmups 4 --request-rate inf \
      --max-concurrency "$concurrency" --ignore-eos --seed 618 \
      --temperature 0 --logprobs 5 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --save-result --save-detailed --disable-tqdm \
      --metadata "arm=$arm" "round=$round" "concurrency=$concurrency" "commit=$commit" \
      --result-dir "$RESULTS/raw/$arm" --result-filename "$file" >"$log" 2>&1; then
      echo "ok,$arm,$round,$concurrency,$commit" | tee -a "$RESULTS/run_status.csv"
    else
      echo "failed,$arm,$round,$concurrency,$commit" | tee -a "$RESULTS/run_status.csv"
      FAIL=1
    fi
  done

  kill -TERM "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  stop_server
}

printf 'status,arm,round,concurrency,commit\n' > "$RESULTS/run_status.csv"

# Alternating three-arm order limits time-order and thermal bias.
run_arm parent 1 "$PARENT_ENV" "$PARENT_SHA"
run_arm pr     1 "$PARENT_ENV" "$PR_SHA"
run_arm main   1 "$MAIN_ENV"   "$MAIN_SHA"
run_arm pr     2 "$PARENT_ENV" "$PR_SHA"
run_arm main   2 "$MAIN_ENV"   "$MAIN_SHA"
run_arm parent 2 "$PARENT_ENV" "$PARENT_SHA"
run_arm main   3 "$MAIN_ENV"   "$MAIN_SHA"
run_arm parent 3 "$PARENT_ENV" "$PARENT_SHA"
run_arm pr     3 "$PARENT_ENV" "$PR_SHA"

stop_server
exit "$FAIL"
