# PR #61991 H100 end-to-end validation

This folder contains the runner and archived evidence for a three-arm serving
comparison on an NVIDIA H100. It is separate from the H200 kernel
microbenchmark: this test starts an OpenAI-compatible server and measures
requests that ask for token logprobs.

## Pinned inputs

| Arm | Source commit | Role |
|---|---|---|
| `parent` | `fa75d3456d60a5ffe923818c4b495295675b38e1` | Direct parent of PR #61991; attribution baseline |
| `pr` | `b2625fd3fd7ea896940c075b8cc132d9e6bbaee3` | PR #61991 implementation |
| `main` | `cffc0cfa311f4997be58cfe8f5f6473ab4898269` | Upstream main pinned when the matrix began |

The tested model is `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`, downloaded once and served from a
local Pod path. The server uses FP16, eager execution, max model length 4096,
max sequences 64, and 0.80 GPU memory utilization for all arms.

The official `vllm.benchmarks.serve.main` implementation is called by
[`run_vllm_bench.py`](run_vllm_bench.py). Each measurement uses the random
dataset, 128 input tokens, 128 output tokens, `logprobs=5`, temperature 0,
seed 618, four warmups, 200 measured prompts, and concurrency 1, 16, or 64.
Detailed JSON retains per-request latency, ITL samples, generated text, and
errors. The matrix alternates arm order across three rounds and rotates the
concurrency order.

## Run on the prepared H100 Pod

The Pod-specific matrix script is
[`run_h100_matrix.sh`](run_h100_matrix.sh). It expects the three pinned
worktrees, the local model, and the two prepared environments under
`/tmp/issue61991-run`. It writes raw data, server/client logs, and five-second
GPU telemetry under the unique remote result directory recorded in the script.
It refuses to overwrite an existing matrix result.

The completed H100 archive is stored under
[`results/h100_20261016T152655Z/`](results/h100_20261016T152655Z/). Its
[`REPORT.md`](results/h100_20261016T152655Z/REPORT.md) contains the findings,
environment caveats, shutdown-log interpretation, and pointers to raw evidence.
The archive includes all 27 raw benchmark results, detailed logs, telemetry,
build diagnostics, provenance checks, and generated plots.

Regenerate the tables and plots with:

```bash
src/vllm/.venv/bin/python benchmarks/kernels/issue60095/h100_e2e_61991/analyze_h100_e2e.py \
  --results-dir benchmarks/kernels/issue60095/h100_e2e_61991/results/h100_20261016T152655Z
```

If `--results-dir` is omitted, the analyzer uses the latest `results/h100_*`
directory. It produces per-arm medians and ranges, PR speedup ratios, ordered
generated-text checks and same-arm repeatability checks in `analysis/`, with
two comparison figures.

## Interpretation

The parent-to-PR comparison isolates the PR's contribution more closely. The
current-main comparison is a whole-snapshot comparison because main contains
many changes after the PR parent; it is useful as a current-state reference but
does not attribute the full difference to PR #61991. Three repeats are an
initial stability check, not a large-sample tail-latency characterization.
