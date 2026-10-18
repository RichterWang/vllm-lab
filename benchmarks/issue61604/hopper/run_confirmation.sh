#!/usr/bin/env bash
set -euo pipefail

# Five paired, alternating confirmation rounds for the calibrated long-shared
# prefix workload. This script is intentionally fixed to the H100 Pod layout
# used for the 2026-10-18 run; edit only after preserving the original manifest.
ROOT=/workspace/vllm_lab
OUT_ROOT="$ROOT/results/issue61604/hopper/20261017T230937Z"
PYTHON="$ROOT/envs/issue61991-parent/bin/python"
PARENT_SITE="$ROOT/envs/issue61991-parent/lib/python3.12/site-packages"
TORCH_SITE=/tmp/issue61604-torch213-site
AUDIT_SITE="$ROOT/benchmarks/issue61604/hopper/audit_site"
CUDA_HOME="$PARENT_SITE/nvidia/cu13"
MODEL="$ROOT/models/Qwen3-0.6B-c1899de289a04d12100db370d81485cdf75e47ca"
BASE_TREE="$ROOT/worktrees/issue-61604-base"
PATCH_TREE="$ROOT/worktrees/issue-61604-pr"
BASE_SITE=/tmp/issue61604-runtime/base
PATCH_SITE=/tmp/issue61604-runtime/patch

port=19010
for pair in 1 2 3 4 5; do
    seed=$((6160410 + pair))
    if (( pair % 2 )); then
        implementations=(base patch)
    else
        implementations=(patch base)
    fi
    for implementation in "${implementations[@]}"; do
        if [[ "$implementation" == base ]]; then
            worktree=$BASE_TREE
            package_site=$BASE_SITE
        else
            worktree=$PATCH_TREE
            package_site=$PATCH_SITE
        fi
        run_id="confirm-sync-pair${pair}-${implementation}-seed${seed}"
        out="$OUT_ROOT/$run_id"
        echo "START $(date -u +%FT%TZ) pair=$pair arm=$implementation seed=$seed port=$port out=$out"
        "$PYTHON" "$ROOT/benchmarks/issue61604/hopper/run_case.py" \
            --python "$PYTHON" \
            --worktree "$worktree" \
            --package-site "$package_site" \
            --parent-site "$PARENT_SITE" \
            --torch-site "$TORCH_SITE" \
            --audit-site "$AUDIT_SITE" \
            --cuda-home "$CUDA_HOME" \
            --model-path "$MODEL" \
            --output-dir "$out" \
            --run-id "$run_id" \
            --implementation "$implementation" \
            --scenario long-shared \
            --mode sync \
            --port "$port" \
            --requests 200 \
            --concurrency 32 \
            --prefix-tokens 16384 \
            --suffix-tokens 128 \
            --output-tokens 256 \
            --arrival-interval-ms 0 \
            --seed "$seed" \
            --max-model-len 16896 \
            --max-num-seqs 128 \
            --max-num-batched-tokens 65536 \
            --gpu-memory-utilization 0.90 \
            --num-gpu-blocks-override 1130 \
            --startup-timeout-s 1200
        echo "DONE  $(date -u +%FT%TZ) pair=$pair arm=$implementation seed=$seed"
        port=$((port + 1))
    done
done
