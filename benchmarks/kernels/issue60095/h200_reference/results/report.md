# H200: static vllm-60095-pr vs #60097 auto split

This report compares the full static candidate at commit `8730e4e1d2d6975cb3c8908fa998921f2161781a` with #60097 at `0ddd6017793c943a2091a5800f0cbac1daaa3e02`. The candidate is based on a newer main commit (`d6f0b7a4166b325b4dda6512e2e06dcb581923ba`), so this measures the two implementations as checked out, not an isolated selector-only patch.

All 57 shapes have three measurements per arm. All successful correctness flags were checked. Speedup is reference (#60097) latency divided by candidate latency; values above 1 favor the candidate.

## Overall

- CUDA Graph device-time geometric speedup: **3.540×**; minimum 1.674×; >5% candidate regressions 0/57.
- Synchronized Python wall-time geometric speedup: **1.361×**; minimum 1.035×; >5% candidate regressions 0/57.

## Per vocabulary and K

| V | K | Shapes | Device geo speedup | Device min | Device >5% regressions | Wall geo speedup | Wall min | Wall >5% regressions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 131072 | 1 | 4 | 4.231× | 3.038× | 0 | 1.210× | 1.054× | 0 |
| 131072 | 6 | 11 | 2.772× | 1.676× | 0 | 1.156× | 1.051× | 0 |
| 131072 | 32 | 4 | 4.103× | 2.961× | 0 | 1.213× | 1.035× | 0 |
| 151936 | 1 | 4 | 3.987× | 2.742× | 0 | 1.395× | 1.078× | 0 |
| 151936 | 6 | 11 | 2.756× | 1.753× | 0 | 1.192× | 1.043× | 0 |
| 151936 | 32 | 4 | 3.987× | 2.954× | 0 | 1.394× | 1.050× | 0 |
| 262144 | 1 | 4 | 5.540× | 4.624× | 0 | 2.155× | 1.542× | 0 |
| 262144 | 6 | 11 | 3.443× | 1.674× | 0 | 1.403× | 1.064× | 0 |
| 262144 | 32 | 4 | 5.424× | 4.552× | 0 | 2.145× | 1.557× | 0 |

## Round-to-round repeatability

| Round | Device geo speedup | Device min | Device >5% regressions | Wall geo speedup | Wall min | Wall >5% regressions |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.541× | 1.674× | 0 | 1.344× | 0.686× | 1 |
| 2 | 3.536× | 1.673× | 0 | 1.358× | 0.989× | 0 |
| 3 | 3.540× | 1.660× | 0 | 1.359× | 0.939× | 1 |

## Representative V=151936, K=6

| B | Candidate splits | #60097 splits | Candidate device µs | #60097 device µs | Device speedup | Candidate wall ms | #60097 wall ms | Wall speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64,64,64 | 16,16,16 | 3.51 | 8.05 | 2.294× | 0.0733 | 0.0789 | 1.075× |
| 8 | 64,64,64 | 16,16,16 | 4.18 | 8.28 | 1.978× | 0.0754 | 0.0787 | 1.043× |
| 16 | 64,64,64 | 16,16,16 | 4.94 | 8.65 | 1.753× | 0.0730 | 0.0781 | 1.070× |
| 24 | 64,64,64 | 8,8,8 | 5.90 | 14.03 | 2.378× | 0.0757 | 0.0811 | 1.072× |
| 32 | 64,64,64 | 8,8,8 | 6.73 | 14.03 | 2.084× | 0.0742 | 0.0822 | 1.107× |
| 47 | 64,64,64 | 4,4,4 | 8.29 | 25.11 | 3.030× | 0.0722 | 0.0807 | 1.118× |
| 64 | 64,64,64 | 4,4,4 | 10.05 | 25.31 | 2.520× | 0.0716 | 0.0799 | 1.115× |
| 96 | 64,64,64 | 2,2,2 | 13.57 | 46.88 | 3.456× | 0.0743 | 0.0804 | 1.082× |
| 128 | 64,64,64 | 2,2,2 | 17.35 | 46.91 | 2.704× | 0.0737 | 0.0801 | 1.088× |
| 192 | 32,32,32 | 1,1,1 | 23.06 | 128.15 | 5.557× | 0.0743 | 0.1348 | 1.814× |
| 256 | 32,32,32 | 1,1,1 | 28.91 | 128.73 | 4.453× | 0.0744 | 0.1350 | 1.816× |

## Per-shape results

See `comparison.csv` for per-round measurements, median device and wall latency, effective split choices, and speedups.

## Interpretation limits

- This is a kernel microbenchmark with synthetic logits, not a model-serving or TTFT test.
- Geometric wall-time speedup is consistent across the three rounds, but individual wall-time outliers occurred (including one large candidate slowdown in round 1 at V=131072/B=1/K=6). The headline per-shape wall comparisons use the median of three rounds; inspect the round columns in `comparison.csv` before making per-shape claims.
- The candidate commit is based on a newer main snapshot than the original #60097 PR commit. Results compare those full source snapshots; they do not isolate only the static split rule.
- The optional main baseline was not run; it is not required to answer candidate versus #60097.
