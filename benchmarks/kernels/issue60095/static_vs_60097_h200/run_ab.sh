#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="${VLLM_LAB_ROOT:-/workspace/vllm_lab}"
PYTHON="$LAB_ROOT/src/vllm/.venv/bin/python"
BENCHMARK="$LAB_ROOT/benchmarks/kernels/issue60095/bench_token_logprobs.py"
ANALYZER="$LAB_ROOT/benchmarks/kernels/issue60095/static_vs_60097_h200/analyze.py"
CANDIDATE="$LAB_ROOT/worktrees/vllm-60095-pr"
REFERENCE="$LAB_ROOT/worktrees/vllm-60097-pr"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="$LAB_ROOT/results/kernel/issue60095/static_vs_60097_h200_$RUN_ID"
VOCABS="131072,151936,262144"
BATCHES="1,8,16,24,32,47,64,96,128,192,256"
mkdir -p "$RUN_ROOT"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

candidate_head="$(git -C "$CANDIDATE" rev-parse HEAD)"
reference_head="$(git -C "$REFERENCE" rev-parse HEAD)"
[[ "$candidate_head" == "8730e4e1d2d6975cb3c8908fa998921f2161781a" ]]
[[ "$reference_head" == "0ddd6017793c943a2091a5800f0cbac1daaa3e02" ]]
[[ -z "$(git -C "$CANDIDATE" status --porcelain)" ]]
[[ -z "$(git -C "$REFERENCE" status --porcelain)" ]]

{
  echo "run_id=$RUN_ID"
  echo "candidate=$CANDIDATE"
  echo "candidate_head=$candidate_head"
  echo "candidate_parent=$(git -C "$CANDIDATE" rev-parse HEAD^)"
  echo "candidate_status=clean"
  echo "reference=$REFERENCE"
  echo "reference_head=$reference_head"
  echo "reference_status=clean"
  echo "vocabs=$VOCABS"
  echo "batches=$BATCHES"
  echo "dtype=bfloat16"
  echo "K=1,6,32 (num-logprobs)"
  echo "rounds=3; per-K arm order: C/R, R/C, C/R"
  echo "benchmark=$BENCHMARK"
  echo "benchmark_sha256=$(sha256sum "$BENCHMARK" | awk '{print $1}')"
  echo "torch=$($PYTHON -c 'import torch; print(torch.__version__, torch.version.cuda)')"
  echo "triton=$($PYTHON -c 'import triton; print(triton.__version__)')"
  nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader
} | tee "$RUN_ROOT/experiment.txt"

sample_gpu() {
  local tag="$1"
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits \
    | sed "s/^/$tag,/" >> "$RUN_ROOT/gpu_telemetry.csv"
}

run_one() {
  local round="$1" arm="$2" k="$3" root shape_batches
  if [[ "$arm" == "candidate" ]]; then root="$CANDIDATE"; else root="$REFERENCE"; fi
  if [[ "$k" == "6" ]]; then shape_batches="$BATCHES"; else shape_batches="96,128,192,256"; fi
  local tag="r${round}_${arm}_k${k}"
  sample_gpu "${tag}_before"
  "$PYTHON" "$BENCHMARK" \
    --binary-root "$LAB_ROOT/src/vllm" \
    --vllm-root "$root" \
    --select "$shape_batches" \
    --vocab-sizes "$VOCABS" \
    --splits auto \
    --num-logprobs "$k" \
    --dtype bfloat16 \
    --output-dir "$RUN_ROOT/$tag" \
    2>&1 | tee "$RUN_ROOT/$tag.log"
  sample_gpu "${tag}_after"
}

# Rotate K order across rounds and alternate which implementation runs first.
run_one 1 candidate 6
run_one 1 reference 6
run_one 1 candidate 1
run_one 1 reference 1
run_one 1 candidate 32
run_one 1 reference 32

run_one 2 reference 1
run_one 2 candidate 1
run_one 2 reference 32
run_one 2 candidate 32
run_one 2 reference 6
run_one 2 candidate 6

run_one 3 candidate 32
run_one 3 reference 32
run_one 3 candidate 6
run_one 3 reference 6
run_one 3 candidate 1
run_one 3 reference 1

"$PYTHON" "$ANALYZER" --input-dir "$RUN_ROOT"
echo "Completed A/B run: $RUN_ROOT"
