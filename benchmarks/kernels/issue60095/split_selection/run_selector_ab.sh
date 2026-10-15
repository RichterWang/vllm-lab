#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISSUE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LAB_ROOT="$(cd -- "$ISSUE_DIR/../../.." && pwd)"
PYTHON="$LAB_ROOT/src/vllm/.venv/bin/python"
BENCHMARK="$ISSUE_DIR/bench_token_logprobs.py"
COMPARE="$SCRIPT_DIR/compare_auto_ab.py"
CANDIDATE="$LAB_ROOT/worktrees/vllm-60097-pr"
PROFILE="${1:-key}"
REFERENCE_KIND="${2:-pr}"
case "$REFERENCE_KIND" in
  pr)
    REFERENCE="$LAB_ROOT/worktrees/vllm-60097-pr-reference"
    ;;
  baseline)
    REFERENCE="$LAB_ROOT/worktrees/vllm-60097-base"
    ;;
  *)
    echo "Usage: $0 [key|holdout|guarded_holdout|fallback_check|autotune_vocab] [pr|baseline]" >&2
    exit 2
    ;;
esac

case "$PROFILE" in
  key)
    BATCHES=1,8,16,24,32,47,64,96,128
    VOCABS=8192,32768,65536,131072,151936
    ;;
  holdout)
    BATCHES=1,9,17,33,49,65,127,192,256
    VOCABS=49152,98304,200000
    ;;
  guarded_holdout)
    BATCHES=7,15,19,27,39,55,73,111,173,224
    VOCABS=55296,114688,180224
    ;;
  fallback_check)
    BATCHES=7,15,19
    VOCABS=114688,180224
    ;;
  autotune_vocab)
    BATCHES=1,8,16,24,32,47,64,96,128
    VOCABS=151936
    ;;
  *)
    echo "Usage: $0 [key|holdout|guarded_holdout|fallback_check|autotune_vocab] [pr|baseline]" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON" || ! -d "$REFERENCE" || ! -d "$CANDIDATE" ]]; then
  echo "Missing benchmark Python or PR reference/candidate worktree" >&2
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%S_%N)"
RUN_ROOT="$LAB_ROOT/results/kernel/issue60095/split-selection/ab_${REFERENCE_KIND}_${PROFILE}_$STAMP"
mkdir -p "$RUN_ROOT"

run_arm() {
  local label="$1"
  local source="$2"
  echo "Running $label from $source"
  local extra=()
  if [[ "$PROFILE" == autotune_vocab && "$source" == "$CANDIDATE" ]]; then
    extra=(--initialize-autotune --autotune-cache-dir "$LAB_ROOT/results/kernel/issue60095/autotune/cache")
  fi
  "$PYTHON" "$BENCHMARK" \
    --binary-root "$LAB_ROOT/src/vllm" \
    --vllm-root "$source" \
    --select "$BATCHES" \
    --vocab-sizes "$VOCABS" \
    --splits auto \
    --wall-repetitions "${WALL_REPETITIONS:-10}" \
    "${extra[@]}" \
    --output-dir "$RUN_ROOT/$label" \
    > "$RUN_ROOT/$label.log" 2>&1
}

run_arm r1_reference "$REFERENCE"
run_arm r1_candidate "$CANDIDATE"
run_arm r2_candidate "$CANDIDATE"
run_arm r2_reference "$REFERENCE"
run_arm r3_reference "$REFERENCE"
run_arm r3_candidate "$CANDIDATE"

"$PYTHON" "$COMPARE" \
  --reference "$RUN_ROOT/r1_reference" "$RUN_ROOT/r2_reference" "$RUN_ROOT/r3_reference" \
  --candidate "$RUN_ROOT/r1_candidate" "$RUN_ROOT/r2_candidate" "$RUN_ROOT/r3_candidate" \
  --output-dir "$RUN_ROOT/analysis"
echo "A/B results: $RUN_ROOT"
