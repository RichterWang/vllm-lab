#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/vllm_lab
RUN_ROOT="$ROOT/results/issue61604/hopper/20261018T050837Z"
PYTHON="$ROOT/envs/issue61991-parent/bin/python"
SITE="$ROOT/envs/issue61991-parent/lib/python3.12/site-packages"
LOCAL_SITE=/tmp/issue61604-rerun/local-site
AUDIT_SITE=/tmp/issue61604-rerun/diagnostic_v2
MODEL="$ROOT/models/Qwen3-0.6B-c1899de289a04d12100db370d81485cdf75e47ca"
BASE_TREE="$ROOT/worktrees/issue-61604-base"
PATCH_TREE="$ROOT/worktrees/issue-61604-pr"

RUN_ID=${1:?run id required}
IMPLEMENTATION=${2:?base or patch required}
BLOCKS=${3:?KV block budget or none required}
SEED=${4:?seed required}
REQUESTS=${5:-200}
DIAGNOSTIC=${6:-1}
MODE=${7:-sync}
CONCURRENCY=${8:-32}
PREFIX_TOKENS=${9:-16384}
SUFFIX_TOKENS=${10:-128}
OUTPUT_TOKENS=${11:-256}
INPUT_TRACE=${12:-}
PREFIX_CACHING=${13:-1}

if [[ "$IMPLEMENTATION" == base ]]; then
    WORKTREE=$BASE_TREE
elif [[ "$IMPLEMENTATION" == patch ]]; then
    WORKTREE=$PATCH_TREE
else
    echo "invalid implementation: $IMPLEMENTATION" >&2
    exit 2
fi

OUT="$RUN_ROOT/exploration/$RUN_ID"
if [[ -e "$OUT" ]]; then
    echo "refusing to overwrite existing result path: $OUT" >&2
    exit 3
fi
mkdir -p "$OUT"
sampler_pid=
cleanup() {
    if [[ -n "${sampler_pid:-}" ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader -l 2 > "$OUT/gpu-samples.csv" &
sampler_pid=$!

args=(
    --python "$PYTHON"
    --worktree "$WORKTREE"
    --runtime-cwd /tmp
    --package-site "$WORKTREE"
    --parent-site "$SITE"
    --binary-site "$SITE"
    --torch-site "$LOCAL_SITE"
    --audit-site "$AUDIT_SITE"
    --cuda-home "$SITE/nvidia/cu13"
    --model-path "$MODEL"
    --output-dir "$OUT"
    --run-id "$RUN_ID"
    --implementation "$IMPLEMENTATION"
    --scenario long-shared
    --mode "$MODE"
    --port 19211
    --requests "$REQUESTS"
    --concurrency "$CONCURRENCY"
    --prefix-tokens "$PREFIX_TOKENS"
    --suffix-tokens "$SUFFIX_TOKENS"
    --output-tokens "$OUTPUT_TOKENS"
    --arrival-interval-ms 0
    --seed "$SEED"
    --max-model-len 16896
    --max-num-seqs 128
    --max-num-batched-tokens 65536
    --gpu-memory-utilization 0.90
    --startup-timeout-s 1200
    --metric-interval-s 2
)
if [[ -n "$INPUT_TRACE" ]]; then
    args+=(--input-trace "$INPUT_TRACE")
fi
if [[ "$BLOCKS" != none ]]; then
    args+=(--num-gpu-blocks-override "$BLOCKS")
fi
if [[ "$DIAGNOSTIC" == 1 ]]; then
    args+=(--diagnostic)
fi
if [[ "$PREFIX_CACHING" == 0 ]]; then
    args+=(--no-enable-prefix-caching)
fi

{
    date -u +%FT%TZ
    printf 'run_id=%s\nimplementation=%s\nblocks=%s\nseed=%s\nrequests=%s\ndiagnostic=%s\nmode=%s\n' \
        "$RUN_ID" "$IMPLEMENTATION" "$BLOCKS" "$SEED" "$REQUESTS" "$DIAGNOSTIC" "$MODE"
    printf 'worktree=%s\nbase_sha=' "$WORKTREE"
    git -C "$WORKTREE" rev-parse HEAD
    git -C "$WORKTREE" status --short
} > "$OUT/launch-manifest.txt"

"$PYTHON" "$AUDIT_SITE/run_case_v2.py" "${args[@]}"
