#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
PYTHON="$ROOT/src/vllm/.venv/bin/python"
SCRIPT="$ROOT/benchmarks/issue61604/reproduce.py"
OUT="$ROOT/results/issue61604/local_cpu_test/dynamic"
mkdir -p "$OUT"

for round in 1 2 3; do
  if (( round % 2 )); then variants=(base pr); else variants=(pr base); fi
  for variant in "${variants[@]}"; do
    worktree="$ROOT/worktrees/issue-61604-$variant"
    PYTHONPATH="$worktree" VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 \
      "$PYTHON" "$SCRIPT" --steps 500 --evict-every 5 --arrival-every 5 \
      --output "$OUT/${variant}-${round}.json" \
      > "$OUT/${variant}-${round}.log" 2>&1
  done
done
"$PYTHON" "$ROOT/benchmarks/issue61604/local_cpu_test/summarize_dynamic.py" \
  --input "$OUT" --output "$OUT/summary.json"
