# H200 reference: token-logprob split implementations

This directory preserves the complete local evidence for the 2026-10-16 H200
kernel microbenchmark, so the Runpod instance can be released without losing
results or the analysis path.

## What was compared

Three clean vLLM source trees were measured with the same benchmark harness:

| Arm | Commit | Meaning |
|---|---|---|
| Pre-PR baseline | `d09ff7763747bbe7cd153edd8017479804f47be7` | Direct parent of #60097; original unsplit token-logprob kernel |
| #60097 | `0ddd6017793c943a2091a5800f0cbac1daaa3e02` | Upstream online reduction plus vocabulary splitting |
| Candidate | `8730e4e1d2d6975cb3c8908fa998921f2161781a` | `vllm-60095-pr`, static split selection, based on `d6f0b7a4166b325b4dda6512e2e06dcb581923ba` |

The baseline is the direct parent of #60097, making baseline-to-#60097 the
closest attribution comparison. The candidate is based on a newer mainline
snapshot and has a different kernel implementation, so candidate comparisons
are full-snapshot comparisons and do not isolate the split selector alone.

## Headline results

All 57 V/B/K shapes passed correctness checks in each arm, with three runs per
shape. Geometric speedup is computed from per-shape medians; a ratio above 1
means the implementation named after the arrow is faster.

| Comparison | CUDA Graph device time | Synchronized wall time |
|---|---:|---:|
| Pre-PR baseline → #60097 | 3.446× | 1.610× |
| #60097 → candidate | 3.540× | 1.361× |
| Pre-PR baseline → candidate | 12.200× | 2.191× |

These are synthetic-logits kernel microbenchmark results, not model-serving
TTFT or throughput results. The baseline round ran after the candidate/#60097
A/B rounds instead of being interleaved with them; GPU clocks remained fixed
at 1980 MHz (SM) and 3201 MHz (memory), temperature was 20–30°C, and no other
GPU work was observed. See `results/report_three_way.md` for interpretation
and limits.

## Reproduction inputs

- `bench_token_logprobs.py`: exact harness used; SHA256 is recorded in
  `results/experiment.txt` and `results/baseline_experiment.txt`.
- `run_ab.sh`: candidate versus #60097 three-round alternating runner.
- `run_baseline.sh`: three-round pre-PR baseline runner.
- `analyze.py`, `analyze_three_way.py`: two-way and three-way result analysis.
- `results/`: raw per-run CSV/JSON, logs, environment snapshots, telemetry,
  aggregate CSV/JSON, reports, and figures.
- `SHA256SUMS`: integrity hashes for archived files.

The benchmark ran on NVIDIA H200 (132 SM, 143771 MiB), driver 570.195.03,
Python 3.12.3, PyTorch 2.13.0+cu130, CUDA runtime 13.0, and Triton 3.7.1.
The Pod required `LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat` for the
PyTorch CUDA runtime. Full details and source worktree state are in
`environment.txt` and `source_commits.txt`.

To regenerate the three-way analysis from the archived raw results:

```bash
python analyze_three_way.py --input-dir results
```

Matplotlib is required to regenerate `results/three_way_speedup.png`.
