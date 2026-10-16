# Reference auto vs candidate auto

Reference: base at `d09ff7763747bbe7cd153edd8017479804f47be7`; candidate at `c00ded17476992df717086687e25033342807e8e`.
GPU: NVIDIA H100 80GB HBM3 (132 SMs).
3 rounds, 9 shapes; geometric mean speedup 8.11×; worst 4.48×; >5% regressions 0.
Metric: CUDA Graph device time per call, lower is better.
Warm wall time includes Python dispatch, launch, and final GPU sync; each run uses 10 calls, so treat small differences cautiously.
Wall geomean speedup 2.59×; worst 2.51×; >5% wall regressions 0.

| B | V | reference split | candidate split | reference device µs | candidate device µs | device speedup | reference wall µs | candidate wall µs | wall speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 151936 | None | 16 | 110.95 | 8.22 | 13.49× | 117.40 | 46.60 | 2.52× |
| 8 | 151936 | None | 16 | 111.64 | 8.41 | 13.27× | 117.69 | 44.75 | 2.63× |
| 16 | 151936 | None | 16 | 111.72 | 8.77 | 12.74× | 117.90 | 45.31 | 2.60× |
| 24 | 151936 | None | 8 | 111.77 | 13.96 | 8.00× | 117.82 | 46.57 | 2.53× |
| 32 | 151936 | None | 8 | 113.33 | 14.06 | 8.06× | 119.30 | 44.64 | 2.67× |
| 47 | 151936 | None | 4 | 113.52 | 25.18 | 4.51× | 119.26 | 47.52 | 2.51× |
| 64 | 151936 | None | 4 | 113.43 | 25.35 | 4.48× | 119.33 | 45.60 | 2.62× |
| 96 | 151936 | None | 32 | 113.47 | 14.16 | 8.01× | 119.32 | 45.77 | 2.61× |
| 128 | 151936 | None | 32 | 114.17 | 17.99 | 6.35× | 121.36 | 46.25 | 2.62× |
