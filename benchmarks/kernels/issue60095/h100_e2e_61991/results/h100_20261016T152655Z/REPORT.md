# PR #61991 H100 end-to-end serving comparison

## Result

The PR implementation shows a mixed result. Against its direct parent, the PR's median output throughput is **4.7% higher at concurrency 1**, **5.3% lower at concurrency 16**, and **10.5% higher at concurrency 64**. At concurrency 64, median TPOT p50 improves by 9.2% and p99 ITL improves by 7.8%. At concurrency 16, median p99 ITL is 8.5% worse, although the three-run ranges overlap. This supports a high-concurrency benefit for this workload, not a general improvement across all loads.

The percentages above divide each arm's three-run median; they are not medians of same-round paired ratios. At concurrency 16, the round-paired throughput changes are +4.1%, −6.4%, and +1.3%, so the result is mixed rather than a repeatable regression. At concurrency 64, the round-paired changes are +9.9%, +3.4%, and +13.9%.

The comparison against current `main` also varies by concurrency. The PR has 15.1% higher median throughput at concurrency 64, but 11.8% lower throughput at concurrency 16. This is a broad snapshot comparison: changes since the PR parent are included, so it does not isolate PR #61991.

Each of the 27 measured runs completed 200 requests with zero benchmark-reported failures. The final output-handler `EngineDeadError` messages occur during the runner's intentional server shutdown after SIGTERM; they appear for multiple arms, after benchmark completion, and did not affect the result JSONs. The logs are retained for review.

## Pinned comparison and workload

| Arm | Commit | Purpose |
|---|---|---|
| Parent | `fa75d3456d60a5ffe923818c4b495295675b38e1` | Direct parent of the PR; closest attribution baseline |
| PR | `b2625fd3fd7ea896940c075b8cc132d9e6bbaee3` | PR #61991 static split implementation |
| Main | `cffc0cfa311f4997be58cfe8f5f6473ab4898269` | Pinned current-main snapshot for the experiment |

- GPU: NVIDIA H100 80 GB HBM3; driver 595.91.07.
- Serving environment: Python 3.12, PyTorch 2.14.0+cu130, CUDA runtime 13.0, Triton 3.8.0. CUDA was available in both serving environments.
- Model: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, downloaded once and served from the Pod's local disk.
- Server: FP16, eager mode, max model length 4096, max sequences 64, GPU memory utilization 0.80.
- Requests: `/v1/completions`, random 128-token input and 128-token output, `logprobs=5`, temperature 0, seed 618, ignore EOS, 4 warmups, 200 measured prompts; concurrency 1, 16, and 64.
- Ordering alternated across three rounds: parent→PR→main, PR→main→parent, main→parent→PR. Concurrency order was rotated within each round.
- Metrics: official vLLM serving benchmark TTFT, TPOT, ITL, E2E latency percentiles and throughput. Tables below use the median of three run-level values.

The parent and PR serving arms shared the parent-built environment. For the PR arm, the runner copied the PR checkout's `vllm/v1/worker/gpu/sample/logprob.py` into the installed parent package before starting the service. The PR commit recorded in benchmark metadata identifies the source file under test; it is not a separately installed PR wheel. This file-level substitution isolates the PR's runtime change, but should not be described as a full PR-checkout installation test.

## Median results

| Arm | Concurrency | Output tok/s | TPOT p50 (ms) | ITL p99 (ms) | E2E p99 (ms) |
|---|---:|---:|---:|---:|---:|
| Parent | 1 | 85.675 | 11.208 | 17.750 | 2215.550 |
| PR | 1 | 89.675 | 10.215 | 17.884 | 2011.282 |
| Main | 1 | 84.815 | 11.290 | 19.925 | 2537.636 |
| Parent | 16 | 1119.224 | 12.004 | 20.790 | 2476.230 |
| PR | 16 | 1060.098 | 12.147 | 22.559 | 2460.424 |
| Main | 16 | 1201.667 | 11.990 | 18.549 | 2414.138 |
| Parent | 64 | 3223.929 | 13.847 | 24.325 | 2529.417 |
| PR | 64 | 3564.041 | 12.571 | 22.416 | 2310.087 |
| Main | 64 | 3096.225 | 16.609 | 26.618 | 2359.833 |

### PR speedup ratios

Ratios above 1 favor the PR. Throughput uses PR / comparator; latency uses comparator / PR.

| Comparison | Concurrency | Output tok/s | TPOT p50 | ITL p99 | E2E p99 |
|---|---:|---:|---:|---:|---:|
| PR vs parent | 1 | 1.047 | 1.097 | 0.992 | 1.102 |
| PR vs parent | 16 | 0.947 | 0.988 | 0.922 | 1.006 |
| PR vs parent | 64 | 1.105 | 1.101 | 1.085 | 1.095 |
| PR vs main | 1 | 1.057 | 1.105 | 1.114 | 1.262 |
| PR vs main | 16 | 0.882 | 0.987 | 0.822 | 0.981 |
| PR vs main | 64 | 1.151 | 1.321 | 1.187 | 1.022 |

### Same-round output throughput: PR vs pinned main

This broad snapshot comparison is shown per round to make its variability visible; it does not isolate PR #61991.

| Concurrency | Round | Main → PR output tok/s (change) |
|---:|---:|---:|
| 1 | 1 | 76.853 → 89.675 (+16.7%) |
| 1 | 2 | 85.997 → 94.512 (+9.9%) |
| 1 | 3 | 84.815 → 87.791 (+3.5%) |
| 16 | 1 | 1121.787 → 1060.098 (−5.5%) |
| 16 | 2 | 1201.667 → 1047.094 (−12.9%) |
| 16 | 3 | 1230.796 → 1219.695 (−0.9%) |
| 64 | 1 | 3096.225 → 3564.041 (+15.1%) |
| 64 | 2 | 2686.366 → 3333.309 (+24.1%) |
| 64 | 3 | 3584.207 → 3568.675 (−0.4%) |

### Same-round parent-to-PR changes

Each cell shows parent → PR (percentage change). Positive throughput changes favor the PR; negative latency changes favor the PR.

| Concurrency | Round | Output tok/s | TPOT p50 (ms) | ITL p99 (ms) | E2E p99 (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 85.675 → 89.675 (+4.7%) | 10.812 → 10.215 (−5.5%) | 20.415 → 19.699 (−3.5%) | 2565.659 → 2368.635 (−7.7%) |
| 1 | 2 | 85.516 → 94.512 (+10.5%) | 11.374 → 10.075 (−11.4%) | 17.580 → 17.884 (+1.7%) | 2215.550 → 1944.999 (−12.2%) |
| 1 | 3 | 86.698 → 87.791 (+1.3%) | 11.208 → 11.141 (−0.6%) | 17.750 → 17.311 (−2.5%) | 2176.176 → 2011.282 (−7.6%) |
| 16 | 1 | 1018.799 → 1060.098 (+4.1%) | 12.461 → 13.532 (+8.6%) | 24.371 → 22.559 (−7.4%) | 2699.614 → 2460.424 (−8.9%) |
| 16 | 2 | 1119.224 → 1047.094 (−6.4%) | 12.004 → 12.147 (+1.2%) | 20.790 → 23.689 (+13.9%) | 2476.230 → 2762.625 (+11.6%) |
| 16 | 3 | 1203.635 → 1219.695 (+1.3%) | 11.636 → 11.318 (−2.7%) | 20.533 → 19.457 (−5.2%) | 2474.787 → 2114.159 (−14.6%) |
| 64 | 1 | 3243.882 → 3564.041 (+9.9%) | 13.847 → 12.571 (−9.2%) | 22.567 → 22.416 (−0.7%) | 2529.417 → 2310.087 (−8.7%) |
| 64 | 2 | 3223.929 → 3333.309 (+3.4%) | 14.185 → 12.461 (−12.2%) | 31.833 → 31.878 (+0.1%) | 2588.285 → 2911.855 (+12.5%) |
| 64 | 3 | 3133.522 → 3568.675 (+13.9%) | 13.412 → 13.389 (−0.2%) | 24.325 → 21.322 (−12.3%) | 2390.001 → 2187.781 (−8.5%) |

## Interpretation and limits

1. **The strongest result is at concurrency 64 against the PR parent.** All three PR output-throughput runs (3333–3569 tok/s) were above all three parent runs (3134–3244 tok/s). TPOT p50 and ITL p99 medians also improve there.
2. **Concurrency 16 does not support an improvement claim.** PR output throughput is lower and p99 ITL is higher than the parent median. The 3-run ranges overlap, so the size of the regression is uncertain; more repeats would be needed to characterize it.
3. **Request-level p99 has limited support here.** TTFT, TPOT, and E2E p99 are each calculated from 200 request samples per run and are sensitive to a few tail requests. ITL p99 instead pools token inter-token intervals across successful requests (about 25,000 intervals for 128-token outputs); those intervals are correlated within each request. Three repetitions are an initial stability check, not a definitive tail-latency characterization. Per-run ranges are in `analysis/summary.csv`, and individual values are in `raw/`.
4. **Generated text is not a robust correctness proof.** Ordered text agreement against parent is 600/600 at concurrency 1, 571/600 at 16, and 571/600 at 64 for the PR. But same-arm comparisons across rounds also differ (see `analysis/same_arm_repeatability.csv`). The benchmark JSON lacks a stable prompt/request identifier for aligning these results, and this test does not compare returned logprob values. Use the unit/reference tests for numerical correctness; treat these text counts only as a consistency signal.
5. **The main comparison is not causal attribution.** `main` includes unrelated code changes since the PR parent. Parent vs PR is the relevant comparison for the split change.
6. **Comparison ratios are ratios of arm medians.** They do not pair rounds. At concurrency 16, the median ratio is −5.3% throughput, while the same-round changes are +4.1%, −6.4%, and +1.3%; report both rather than presenting the median ratio as a repeated per-round effect.
7. **Kernel vs service evidence:** this measures full service requests with `logprobs=5`. It complements, but does not replace or merge with, the H200 kernel microbenchmark in the sibling `h200_reference/` archive.

## Environment record discrepancy

The archived `environment.txt` was captured at 2026-10-16 15:27 UTC. Its commit section records main as `1a5b187066c918b2e23184560d703174dd542828`, which conflicts with the matrix runner's pinned SHA and the run JSON metadata. For the actual test, the runner pinned `cffc0cfa311f4997be58cfe8f5f6473ab4898269`; all nine main JSON files contain that SHA; the main worktree was later confirmed at that SHA; and the built main package reports `0.32.0.dev0+gcffc0cfa31`. The serving environments also import PyTorch 2.14.0+cu130 and Triton 3.8.0. The same snapshot's top-level system Python reports PyTorch 2.8.0+cu128; that was not the interpreter used by the benchmark. The runner also exported `CUDA_HOME=/usr/local/cuda-12.8` and placed that toolkit on `PATH`, while the main extension build log shows the venv's CUDA 13.0 `nvcc`. Tests completed under this mixed setting; it is retained as a reproducibility caveat, not evidence of an arm-specific measured difference. The original snapshot remains unchanged in `environment.txt`; the cross-check and live verification are retained in `diagnostics/` and `provenance.md`.

## Build and shutdown notes

The current-main extension build took about 67 minutes after dependencies and the CUDA 13.0 toolchain were corrected. An initial build failed on missing `libdw` headers; a retry selected the wrong system CUDA 12.8 compiler and was stopped; the successful build used the CUDA 13.0 toolchain and H100 environment. Build logs and smoke response are retained under `diagnostics/`. The smoke test confirmed service startup and `top_logprobs` length 5 on generated tokens.

The runner sends SIGTERM after each arm completes. Eight of the nine server logs then show `EngineCore` shutdown followed by an `AsyncLLM output_handler failed` / `EngineDeadError` during teardown; the remaining parent shutdown completed without that message. Benchmark logs and JSONs show all requested measurements completed and `failed=0` before shutdown. It did not affect these recorded measurements, but remains a teardown-time log error.

## Reproduction and files

From the repository root, regenerate the tables and plots with:

```bash
src/vllm/.venv/bin/python benchmarks/kernels/issue60095/h100_e2e_61991/analyze_h100_e2e.py \
  --results-dir benchmarks/kernels/issue60095/h100_e2e_61991/results/h100_20261016T152655Z
```

- `analysis/summary.csv`: run medians and min/max across the three repetitions.
- `analysis/comparisons.csv`: PR speedup ratios.
- `analysis/output_consistency.csv` and `analysis/same_arm_repeatability.csv`: ordered text match checks.
- `analysis/metrics_by_concurrency.png` and `analysis/speedup_by_concurrency.png`: visual comparisons; error bars show the three-run min–max range in the first figure.
- `raw/`, `logs/`, `telemetry/`, `environment.txt`, and `diagnostics/`: original data and execution evidence.
- `checksums.sha256`: checksums for the archived files (excluding the checksum list itself).

This result is suitable as supplementary evidence in PR discussion with the limitations above. It is not evidence that the PR improves every workload or every concurrency setting.
