#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="$LAB_ROOT/src/vllm/.venv/bin/python"
BINARY_ROOT="${VLLM_BINARY_ROOT:-$LAB_ROOT/src/vllm}"
BENCHMARK="$SCRIPT_DIR/bench_token_logprobs.py"
ANALYZER="$SCRIPT_DIR/analyze_repeats.py"
BASE_ROOT="$LAB_ROOT/worktrees/vllm-60097-base"
PR_ROOT="$LAB_ROOT/worktrees/vllm-60097-pr"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing benchmark Python environment: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$BASE_ROOT" || ! -d "$PR_ROOT" ]]; then
  echo "Expected base and PR worktrees under $LAB_ROOT/worktrees" >&2
  exit 1
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%S_%N)"
RUN_ROOT="$LAB_ROOT/results/kernel/issue60095/repeats/run_$RUN_ID"
mkdir -p "$RUN_ROOT"

run_bench() {
  local label="$1"
  local worktree_path="$2"

  "$PYTHON" "$BENCHMARK" \
    --binary-root "$BINARY_ROOT" \
    --vllm-root "$worktree_path" \
    --select 1,16,32,64,128 \
    --splits auto \
    --output-dir "$RUN_ROOT/$label"
}

# Alternate the order to reduce bias from GPU temperature and clock changes.
run_bench r1_base "$BASE_ROOT"
run_bench r1_pr "$PR_ROOT"
run_bench r2_pr "$PR_ROOT"
run_bench r2_base "$BASE_ROOT"
run_bench r3_base "$BASE_ROOT"
run_bench r3_pr "$PR_ROOT"

"$PYTHON" "$ANALYZER" --input-dir "$RUN_ROOT"
echo "Repeated A/B results: $RUN_ROOT"
