# H200 three-way comparison: baseline, #60097, and vllm-60095-pr

Baseline is the parent of #60097 (`d09ff7763747bbe7cd153edd8017479804f47be7`). #60097 is `0ddd6017793c943a2091a5800f0cbac1daaa3e02`; candidate is `8730e4e1d2d6975cb3c8908fa998921f2161781a` based on `d6f0b7a4166b325b4dda6512e2e06dcb581923ba`. Baseline was measured in three additional rounds with the same benchmark/environment after the candidate/#60097 run. All ratios use reference latency divided by the named implementation; >1 favors the named implementation.

Each arm passed correctness on 57 shapes over three rounds. Candidate and #60097 each have 3 interleaved rounds; baseline has 3 additional rounds. The source commits are not all on one linear base: baseline→#60097 is a direct parent/child pair, while candidate is on a newer main snapshot. Thus baseline→candidate is a useful full-implementation comparison but does not isolate a single selector change.

## Overall geometric speedup and worst shape

| Comparison (speedup >1 favors right side) | Geometric speedup | Minimum | Shapes with >5% right-side slowdown |
|---|---:|---:|---:|
| baseline / candidate device | 12.200× | 6.407× | 0 |
| baseline / candidate wall | 2.191× | 1.307× | 0 |
| baseline / #60097 device | 3.446× | 1.459× | 0 |
| baseline / #60097 wall | 1.610× | 1.202× | 0 |
| #60097 / candidate device | 3.540× | 1.674× | 0 |
| #60097 / candidate wall | 1.361× | 1.035× | 0 |

## Per-shape table

The complete `comparison_three_way.csv` includes all V/B/K shapes, per-round times, split selections, and all three pairwise speed ratios.

## Interpretation

The baseline→#60097 comparison isolates the PR commit against its direct parent. The candidate currently has a different kernel implementation and is based on newer main; candidate comparisons show which complete implementation is faster on these microbenchmark shapes, not the isolated effect of the static split rule. The test uses synthetic logits and does not establish model-serving TTFT or throughput.
