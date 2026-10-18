#!/usr/bin/env bash
set -euo pipefail
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

run_one() {
    local pair="$1" implementation="$2" seed="$3" port="$4"
    local worktree package_site run_id out
    if [[ "$implementation" == base ]]; then
        worktree=$BASE_TREE
        package_site=$BASE_SITE
    else
        worktree=$PATCH_TREE
        package_site=$PATCH_SITE
    fi
    run_id="confirm-sync-pair${pair}-${implementation}-seed${seed}"
    out="$OUT_ROOT/$run_id"
    echo "START $(date -u +%FT%TZ) pair=$pair arm=$implementation seed=$seed port=$port"
    "$PYTHON" "$ROOT/benchmarks/issue61604/hopper/run_case.py" \
        --python "$PYTHON" --worktree "$worktree" \
        --package-site "$package_site" --parent-site "$PARENT_SITE" \
        --torch-site "$TORCH_SITE" --audit-site "$AUDIT_SITE" \
        --cuda-home "$CUDA_HOME" --model-path "$MODEL" \
        --output-dir "$out" --run-id "$run_id" \
        --implementation "$implementation" --scenario long-shared \
        --mode sync --port "$port" --requests 200 --concurrency 32 \
        --prefix-tokens 16384 --suffix-tokens 128 --output-tokens 256 \
        --arrival-interval-ms 0 --seed "$seed" --max-model-len 16896 \
        --max-num-seqs 128 --max-num-batched-tokens 65536 \
        --gpu-memory-utilization 0.90 --num-gpu-blocks-override 1130 \
        --startup-timeout-s 1200
    echo "DONE  $(date -u +%FT%TZ) pair=$pair arm=$implementation seed=$seed"
}

# Pair 4's patch arm completed before the long-lived SSH channel closed.
run_one 4 base 6160414 19017
# Pair 5 alternates back to base -> patch.
run_one 5 base 6160415 19018
run_one 5 patch 6160415 19019
