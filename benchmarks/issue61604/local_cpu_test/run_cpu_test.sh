#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
PYTHON="$ROOT/src/vllm/.venv/bin/python"
SCRIPT="$ROOT/benchmarks/issue61604/reproduce.py"
OUT="$ROOT/results/issue61604/local_cpu_test"
mkdir -p "$OUT"

for variant in base pr; do
  worktree="$ROOT/worktrees/issue-61604-$variant"
  PYTHONPATH="$worktree" VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 \
    "$PYTHON" -m pytest --noconftest \
    "$worktree/tests/v1/core/test_prefix_caching.py" -q \
    > "$OUT/${variant}-prefix-cache-tests.log" 2>&1
done

for round in 1 2 3 4 5; do
  for variant in base pr; do
    worktree="$ROOT/worktrees/issue-61604-$variant"
    PYTHONPATH="$worktree" VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 \
      "$PYTHON" "$SCRIPT" --output "$OUT/${variant}-${round}.json" \
      > "$OUT/${variant}-${round}.log" 2>&1
  done
done

"$PYTHON" "$ROOT/benchmarks/issue61604/local_cpu_test/summarize.py" \
  --input "$OUT" --output "$OUT/summary.json"
