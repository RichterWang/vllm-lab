# H100 token-logprob split selection: reproducible kernel results

This is the evidence package for the split-selector change built on vLLM
PR [#60097](https://github.com/vllm-project/vllm/pull/60097). It separates
the original PR's contribution from this candidate's incremental improvement.

## Environment and source

- GPU: NVIDIA H100 80GB HBM3, 132 SMs, compute capability 9.0.
- Driver: 570.195.03. PyTorch: 2.13.0+cu130. Triton: 3.7.1.
- The container used `cuda-compat-13-0` with
  `LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat` because the mounted PyTorch
  build uses CUDA 13.0 and the Pod driver reports CUDA 12.8.
- Pre-PR baseline: `d09ff7763747bbe7cd153edd8017479804f47be7`.
- PR #60097: `0ddd6017793c943a2091a5800f0cbac1daaa3e02`.
- Candidate: `c00ded17476992df717086687e25033342807e8e`.
- Harness: vllm-lab `87660b0a22e32a1851253615340b0e331daac9b2`.

Each arm used the same precompiled vLLM binary root. Shapes were V=151936,
bf16, K=6, and B=1/8/16/24/32/47/64/96/128. The reference and candidate
were alternated for three rounds, and every row passed the FP64 oracle check.
Latency below is warm CUDA Graph device time per call. The `analysis/` folders
also report synchronized wall time, which includes Python dispatch and sync.

## Result

| B | Baseline auto µs | PR auto split/µs | Candidate auto split/µs | PR to candidate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 110.948 | 16 / 8.225 | 16 / 8.218 | 1.001× |
| 8 | 111.643 | 16 / 8.425 | 16 / 8.417 | 1.001× |
| 16 | 111.716 | 16 / 8.755 | 16 / 8.769 | 0.998× |
| 24 | 111.773 | 8 / 13.999 | 8 / 14.000 | 1.000× |
| 32 | 113.325 | 8 / 14.052 | 8 / 14.078 | 0.998× |
| 47 | 113.517 | 4 / 25.106 | 4 / 25.154 | 0.998× |
| 64 | 113.429 | 4 / 25.367 | 4 / 25.370 | 1.000× |
| 96 | 113.473 | 2 / 46.928 | 32 / 14.185 | 3.308× |
| 128 | 114.175 | 2 / 65.741 | 32 / 18.004 | 3.652× |

Across these nine shapes, the candidate versus PR device-time geometric mean
is **1.318×** and synchronized-wall geometric mean is **1.119×**. There was
no regression over 5%. Baseline versus candidate device-time geometric mean
is **8.105×**, but this includes the original PR's contribution and must not
be attributed to the candidate selector. The forced-split scan independently
shows about 18 µs at B=128 with 16/32 splits, versus about 66 µs with the PR's
auto choice of two splits.

## Startup and limits

The first K=6 calibration took 1.783 s. Five independent cache-hit startup
measurements were 66/93/84/77 ms plus one 3.594 s outlier. A separate run of
30 cache hits within one process measured 40.4/44.3/66.3 ms
(minimum/median/maximum); it did not reproduce the outlier. These are startup
measurements, not request P99. The run reused Triton compilation cache and
did not measure cold compilation or full model serving. Only one V/K/dtype
combination was tested here; H100 results do not establish H200 performance.

## Reproduce and inspect

From a lab checkout with the candidate at `worktrees/vllm-60097-pr`, the clean
PR at `worktrees/vllm-60097-pr-reference`, and the baseline at
`worktrees/vllm-60097-base`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
benchmarks/kernels/issue60095/split_selection/run_selector_ab.sh autotune_vocab pr
benchmarks/kernels/issue60095/split_selection/run_selector_ab.sh autotune_vocab baseline
```

The command above uses `src/vllm/.venv/bin/python`. The H100 Pod kept the
candidate in a separate `vllm-60097-candidate` worktree and linked it to the
expected `vllm-60097-pr` path in a separate lab checkout. Source SHAs and
full run settings are recorded in each `result.json`.

- [PR versus candidate report](data/pr_ab/analysis/report.md) and
  [comparison CSV](data/pr_ab/analysis/comparison.csv).
- [Baseline versus candidate report](data/baseline_ab/analysis/report.md) and
  [comparison CSV](data/baseline_ab/analysis/comparison.csv).
- [Forced-split scan](data/h100_split_scan/result.csv) and
  [autotune cache table](data/autotune_cache/).
- [Environment snapshot](data/h100_handoff/environment.txt) and
  [30-hit cache diagnostic](data/h100_handoff/cache_hit_30.log) with its
  [script](data/h100_handoff/diagnose_cache_hit.py).
- [Three-way latency](figures/latency_three_way.png),
  [speedup by batch](figures/speedup_by_batch.png), and
  [selected splits](figures/selected_splits.png).
- [B=96](figures/round_variation_b96.png) and
  [B=128](figures/round_variation_b128.png) round variation.

The plot generator is `../split_selection/plot_three_way_ab.py`.
