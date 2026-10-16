#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="${VLLM_LAB_ROOT:-/workspace/vllm_lab}"
PYTHON="$LAB_ROOT/src/vllm/.venv/bin/python"
BENCHMARK="$LAB_ROOT/benchmarks/kernels/issue60095/bench_token_logprobs.py"
BASELINE="$LAB_ROOT/worktrees/vllm-60097-base"
RUN_ROOT="${VLLM_RUN_ROOT:?Set VLLM_RUN_ROOT to the existing three-way result directory}"
VOCABS="131072,151936,262144"
BATCHES="1,8,16,24,32,47,64,96,128,192,256"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

[[ "$(git -C "$BASELINE" rev-parse HEAD)" == "d09ff7763747bbe7cd153edd8017479804f47be7" ]]
[[ -z "$(git -C "$BASELINE" status --porcelain)" ]]
{
  echo "baseline=$BASELINE"
  echo "baseline_head=$(git -C "$BASELINE" rev-parse HEAD)"
  echo "baseline_status=clean"
  echo "vocabs=$VOCABS"
  echo "K=1,6,32; dtype=bfloat16"
  echo "rounds=3; rotated K order"
  echo "benchmark_sha256=$(sha256sum "$BENCHMARK" | awk '{print $1}')"
  nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader
} | tee "$RUN_ROOT/baseline_experiment.txt"

sample_gpu() {
  local tag="$1"
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits \
    | sed "s/^/$tag,/" >> "$RUN_ROOT/gpu_telemetry_baseline.csv"
}

run_one() {
  local round="$1" k="$2" batches
  if [[ "$k" == "6" ]]; then batches="$BATCHES"; else batches="96,128,192,256"; fi
  local tag="r${round}_baseline_k${k}"
  sample_gpu "${tag}_before"
  "$PYTHON" "$BENCHMARK" \
    --binary-root "$LAB_ROOT/src/vllm" \
    --vllm-root "$BASELINE" \
    --select "$batches" \
    --vocab-sizes "$VOCABS" \
    --splits auto \
    --num-logprobs "$k" \
    --dtype bfloat16 \
    --output-dir "$RUN_ROOT/$tag" \
    2>&1 | tee "$RUN_ROOT/$tag.log"
  sample_gpu "${tag}_after"
}

run_one 1 6
run_one 1 1
run_one 1 32
run_one 2 1
run_one 2 32
run_one 2 6
run_one 3 32
run_one 3 6
run_one 3 1
