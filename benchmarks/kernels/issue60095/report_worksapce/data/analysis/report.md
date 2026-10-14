# Issue #60095 repeated A/B benchmark analysis

- GPU: NVIDIA H200
- PyTorch / CUDA / Triton: 2.13.0+cu130 / 13.0 / 3.7.1
- Metric: CUDA Graph latency in microseconds; lower is better.
- PR rows use `requested_splits=auto`; base rows use its unchanged implementation.
- Speedup is `base latency / PR auto latency`; values above 1 mean PR is faster.

## Per-round comparison

| Round | Batch | Base (µs) | PR auto (µs) | Speedup | PR latency reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 111.29 | 8.02 | 13.88× | 92.8% |
| 1 | 16 | 112.15 | 8.65 | 12.97× | 92.3% |
| 1 | 32 | 113.72 | 14.02 | 8.11× | 87.7% |
| 1 | 64 | 113.60 | 25.26 | 4.50× | 77.8% |
| 1 | 128 | 113.69 | 47.12 | 2.41× | 58.6% |
| 2 | 1 | 111.25 | 8.02 | 13.88× | 92.8% |
| 2 | 16 | 112.14 | 8.68 | 12.92× | 92.3% |
| 2 | 32 | 113.93 | 13.98 | 8.15× | 87.7% |
| 2 | 64 | 113.83 | 25.29 | 4.50× | 77.8% |
| 2 | 128 | 113.71 | 46.97 | 2.42× | 58.7% |
| 3 | 1 | 111.27 | 8.02 | 13.88× | 92.8% |
| 3 | 16 | 112.17 | 8.67 | 12.94× | 92.3% |
| 3 | 32 | 113.69 | 13.98 | 8.13× | 87.7% |
| 3 | 64 | 113.66 | 25.27 | 4.50× | 77.8% |
| 3 | 128 | 113.72 | 46.99 | 2.42× | 58.7% |

## Three-round median and variation

| Implementation | Batch | Median (µs) | Min–max (µs) | Range / median |
|---|---:|---:|---:|---:|
| Base | 1 | 111.27 | 111.25–111.29 | 0.03% |
| Base | 16 | 112.15 | 112.14–112.17 | 0.02% |
| Base | 32 | 113.72 | 113.69–113.93 | 0.22% |
| Base | 64 | 113.66 | 113.60–113.83 | 0.20% |
| Base | 128 | 113.71 | 113.69–113.72 | 0.03% |
| PR auto | 1 | 8.02 | 8.02–8.02 | 0.03% |
| PR auto | 16 | 8.67 | 8.65–8.68 | 0.41% |
| PR auto | 32 | 13.98 | 13.98–14.02 | 0.24% |
| PR auto | 64 | 25.27 | 25.26–25.29 | 0.15% |
| PR auto | 128 | 46.99 | 46.97–47.12 | 0.32% |

## Charts

- Round 1 comparison: `round_1_comparison.png`
- Round 2 comparison: `round_2_comparison.png`
- Round 3 comparison: `round_3_comparison.png`
- Base variation: `base_round_variation.png`
- PR auto variation: `pr_round_variation.png`

Three repeats are useful for an initial stability check, but too few to claim a robust statistical distribution. The variation charts show the observed runs, not a histogram or confidence interval.
