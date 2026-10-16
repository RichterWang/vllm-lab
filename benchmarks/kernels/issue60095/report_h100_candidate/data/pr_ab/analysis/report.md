# Reference auto vs candidate auto

Reference: pr at `0ddd6017793c943a2091a5800f0cbac1daaa3e02`; candidate at `c00ded17476992df717086687e25033342807e8e`.
GPU: NVIDIA H100 80GB HBM3 (132 SMs).
3 rounds, 9 shapes; geometric mean speedup 1.32×; worst 1.00×; >5% regressions 0.
Metric: CUDA Graph device time per call, lower is better.
Warm wall time includes Python dispatch, launch, and final GPU sync; each run uses 10 calls, so treat small differences cautiously.
Wall geomean speedup 1.12×; worst 0.98×; >5% wall regressions 0.

| B | V | reference split | candidate split | reference device µs | candidate device µs | device speedup | reference wall µs | candidate wall µs | wall speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 151936 | 16 | 16 | 8.23 | 8.22 | 1.00× | 47.93 | 48.16 | 1.00× |
| 8 | 151936 | 16 | 16 | 8.43 | 8.42 | 1.00× | 51.65 | 47.05 | 1.10× |
| 16 | 151936 | 16 | 16 | 8.76 | 8.77 | 1.00× | 54.75 | 46.88 | 1.17× |
| 24 | 151936 | 8 | 8 | 14.00 | 14.00 | 1.00× | 48.11 | 49.19 | 0.98× |
| 32 | 151936 | 8 | 8 | 14.05 | 14.08 | 1.00× | 45.74 | 46.16 | 0.99× |
| 47 | 151936 | 4 | 4 | 25.11 | 25.15 | 1.00× | 48.14 | 47.49 | 1.01× |
| 64 | 151936 | 4 | 4 | 25.37 | 25.37 | 1.00× | 47.56 | 47.01 | 1.01× |
| 96 | 151936 | 2 | 32 | 46.93 | 14.19 | 3.31× | 61.64 | 47.33 | 1.30× |
| 128 | 151936 | 2 | 32 | 65.74 | 18.00 | 3.65× | 76.88 | 46.01 | 1.67× |
