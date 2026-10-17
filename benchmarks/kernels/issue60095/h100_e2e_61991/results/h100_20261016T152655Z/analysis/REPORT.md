# H100 end-to-end results: PR #61991

The values below summarize per-run medians across the available rounds.
For latency, speedup is comparator / PR; for throughput it is PR / comparator.
Ratios divide each arm's three-run median; they are not medians of same-round paired ratios.
A value above 1 favors the PR. See the raw JSON for request-level data.

## Run completeness

Loaded 27 benchmark JSON files.
Expected matrix: 3 arms × 3 rounds × 3 concurrency levels = 27 runs; each run targets 200 requests.

## Per-arm medians

| Arm | Concurrency | Runs | Success median | Failed total | Output tok/s | TPOT p50 ms | ITL p99 ms | E2E p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 1 | 3 | 200.000 | 0 | 85.675 | 11.208 | 17.750 | 2215.550 |
| parent | 16 | 3 | 200.000 | 0 | 1119.224 | 12.004 | 20.790 | 2476.230 |
| parent | 64 | 3 | 200.000 | 0 | 3223.929 | 13.847 | 24.325 | 2529.417 |
| pr | 1 | 3 | 200.000 | 0 | 89.675 | 10.215 | 17.884 | 2011.282 |
| pr | 16 | 3 | 200.000 | 0 | 1060.098 | 12.147 | 22.559 | 2460.424 |
| pr | 64 | 3 | 200.000 | 0 | 3564.041 | 12.571 | 22.416 | 2310.087 |
| main | 1 | 3 | 200.000 | 0 | 84.815 | 11.290 | 19.925 | 2537.636 |
| main | 16 | 3 | 200.000 | 0 | 1201.667 | 11.990 | 18.549 | 2414.138 |
| main | 64 | 3 | 200.000 | 0 | 3096.225 | 16.609 | 26.618 | 2359.833 |

## PR speedup ratios

| Comparator | Concurrency | Output tok/s | TPOT p50 | ITL p99 | E2E p99 |
|---|---:|---:|---:|---:|---:|
| pr_vs_parent | 1 | 1.047 | 1.097 | 0.992 | 1.102 |
| pr_vs_parent | 16 | 0.947 | 0.988 | 0.922 | 1.006 |
| pr_vs_parent | 64 | 1.105 | 1.101 | 1.085 | 1.095 |
| pr_vs_main | 1 | 1.057 | 1.105 | 1.114 | 1.262 |
| pr_vs_main | 16 | 0.882 | 0.987 | 0.822 | 0.981 |
| pr_vs_main | 64 | 1.151 | 1.321 | 1.187 | 1.022 |

## Ordered generated-text agreement against parent

| Arm | Concurrency | Paired requests | Exact matches |
|---|---:|---:|---:|
| pr | 1 | 600 | 600/600 |
| pr | 16 | 600 | 571/600 |
| pr | 64 | 600 | 571/600 |
| main | 1 | 600 | 600/600 |
| main | 16 | 600 | 578/600 |
| main | 64 | 600 | 572/600 |

Text matching is by benchmark result order, not by a stable request ID. Same-arm cross-round matches are also imperfect; see `same_arm_repeatability.csv`. Treat this as a consistency signal, not a correctness proof.

Request-level p99 metrics (TTFT, TPOT, E2E) use 200 request samples per run and are sensitive to a few tail requests.
ITL p99 pools token inter-token intervals across successful requests (about 25,000 intervals at 128 output tokens); intervals within a request are correlated. Three repetitions provide only an initial stability check.
