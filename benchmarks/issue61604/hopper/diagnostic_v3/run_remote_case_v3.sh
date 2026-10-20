#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/vllm_lab
RUN_ID_ROOT=${RUN_ID_ROOT:-}
RUN_ROOT="$ROOT/results/issue61604/hopper/$RUN_ID_ROOT"
PYTHON="$ROOT/envs/issue61991-parent/bin/python"
SITE="$ROOT/envs/issue61991-parent/lib/python3.12/site-packages"
LOCAL_SITE=/tmp/issue61604-rerun/local-site
AUDIT_SITE=${ISSUE61604_AUDIT_SITE:-/tmp/issue61604-rerun/diagnostic_v3}
MODEL="$ROOT/models/Qwen3-0.6B-c1899de289a04d12100db370d81485cdf75e47ca"
BASE_TREE="$ROOT/worktrees/issue-61604-base"
PATCH_TREE="$ROOT/worktrees/issue-61604-pr"

RUN_ID=${1:?run id required}
IMPLEMENTATION=${2:?base or patch required}
BLOCKS=${3:?KV block budget or none required}
SEED=${4:?seed required}
SCENARIO=${5:-long-shared}
MODE=${6:-sync}
REQUESTS=${7:-200}
CONCURRENCY=${8:-32}
PREFIX_TOKENS=${9:-16384}
SUFFIX_TOKENS=${10:-128}
OUTPUT_TOKENS=${11:-256}
DIAGNOSTIC=${12:-0}
DIAG_MODE=${13:-count}
INPUT_TRACE=${14:-}
LOGPROBS=${15:-}
PREFIX_CACHING=${16:-1}
CANCEL_INDEX=${17:-}
CANCEL_AFTER_TOKENS=${18:-16}
METRIC_INTERVAL_S=${19:-2}
GPU_SAMPLE_INTERVAL_S=${20:-2}

if [[ -z "$RUN_ID_ROOT" ]]; then
    echo 'RUN_ID_ROOT must point to the new result bundle' >&2
    exit 2
fi
if [[ "$IMPLEMENTATION" == base ]]; then WORKTREE=$BASE_TREE; elif [[ "$IMPLEMENTATION" == patch ]]; then WORKTREE=$PATCH_TREE; else echo "invalid implementation: $IMPLEMENTATION" >&2; exit 2; fi
if [[ "$MODE" != sync && "$MODE" != async-default ]]; then echo "invalid mode: $MODE" >&2; exit 2; fi
if [[ "$SCENARIO" != long-shared && "$SCENARIO" != default-capacity && "$SCENARIO" != short-unshared && "$SCENARIO" != churn ]]; then echo "invalid scenario: $SCENARIO" >&2; exit 2; fi
OUT="$RUN_ROOT/$SCENARIO/$MODE/$RUN_ID"
if [[ -e "$OUT" ]]; then echo "refusing to overwrite existing result path: $OUT" >&2; exit 3; fi
mkdir -p "$OUT"
sampler_pid=
cleanup() { if [[ -n "${sampler_pid:-}" ]]; then kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
if [[ "$GPU_SAMPLE_INTERVAL_S" != 0 ]]; then
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader -l "$GPU_SAMPLE_INTERVAL_S" > "$OUT/gpu-samples.csv" &
    sampler_pid=$!
else
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader > "$OUT/gpu-samples.csv"
fi

args=(
  --python "$PYTHON" --worktree "$WORKTREE" --runtime-cwd /tmp
  --package-site "$WORKTREE" --parent-site "$SITE" --binary-site "$SITE"
  --torch-site "$LOCAL_SITE" --audit-site "$AUDIT_SITE"
  --cuda-home "$SITE/nvidia/cu13" --model-path "$MODEL"
  --output-dir "$OUT" --run-id "$RUN_ID" --implementation "$IMPLEMENTATION"
  --scenario "$SCENARIO" --mode "$MODE" --port 19211 --requests "$REQUESTS"
  --concurrency "$CONCURRENCY" --prefix-tokens "$PREFIX_TOKENS"
  --suffix-tokens "$SUFFIX_TOKENS" --output-tokens "$OUTPUT_TOKENS"
  --arrival-interval-ms 0 --seed "$SEED" --max-model-len 16896
  --metric-interval-s "$METRIC_INTERVAL_S"
  --max-num-seqs 128 --max-num-batched-tokens 65536
  --gpu-memory-utilization 0.90 --block-size 16 --startup-timeout-s 1200
  --prefix-families 8 --churn-a-requests 32
)
[[ -n "$INPUT_TRACE" ]] && args+=(--input-trace "$INPUT_TRACE")
[[ "$BLOCKS" != none ]] && args+=(--num-gpu-blocks-override "$BLOCKS")
[[ "$DIAGNOSTIC" == 1 ]] && args+=(--diagnostic --diag-mode "$DIAG_MODE")
[[ -n "$LOGPROBS" ]] && args+=(--logprobs "$LOGPROBS")
[[ -n "$CANCEL_INDEX" ]] && args+=(--cancel-index "$CANCEL_INDEX" --cancel-after-tokens "$CANCEL_AFTER_TOKENS")
[[ "$PREFIX_CACHING" == 0 ]] && args+=(--no-enable-prefix-caching)
{
  date -u +%FT%TZ
  printf 'run_id=%s\nimplementation=%s\nblocks=%s\nseed=%s\nrequests=%s\ndiagnostic=%s\ndiag_mode=%s\nmode=%s\nscenario=%s\n' \
    "$RUN_ID" "$IMPLEMENTATION" "$BLOCKS" "$SEED" "$REQUESTS" "$DIAGNOSTIC" "$DIAG_MODE" "$MODE" "$SCENARIO"
  printf 'worktree=%s\nhead=' "$WORKTREE"; git -C "$WORKTREE" rev-parse HEAD
  git -C "$WORKTREE" status --short
  printf 'cpuset='; taskset -pc $$
  printf 'numa='; numactl --show 2>/dev/null | tr '\n' ';' || true
} > "$OUT/launch-manifest.txt"
"$PYTHON" "$AUDIT_SITE/run_case_v2.py" "${args[@]}"
