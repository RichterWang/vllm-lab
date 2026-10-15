# Split selection experiment

This directory holds the work added after the independent base/PR #60097
reproduction. The shared microbenchmark stays at
`../bench_token_logprobs.py`; the cumulative experiment log and H200 plan stay
at `../end.md`.

| File | Purpose |
| --- | --- |
| `analyze_split_sweeps.py` | Summarize repeated forced-split sweeps. |
| `fit_split_rule.py` | Fit a small offline split rule to sweep results. |
| `run_selector_ab.sh` | Run three alternating candidate/reference rounds. |
| `compare_auto_ab.py` | Compare those rounds by shape and metric. |
| `selector_candidate_sm86.patch` | Snapshot of the earlier 3070 Ti fixed-rule candidate; it does not include startup autotuning. |

Run from the lab root:

```bash
benchmarks/kernels/issue60095/split_selection/run_selector_ab.sh autotune_vocab pr
benchmarks/kernels/issue60095/split_selection/run_selector_ab.sh autotune_vocab baseline
```

The outer lab repository records the benchmark and analysis. The live vLLM
implementation is in the separate Git repository at
`worktrees/vllm-60097-pr/`, on `codex/issue60095-split-selection`. That
worktree has local candidate edits. The clean PR #60097 reference is
`worktrees/vllm-60097-pr-reference/`; the pre-PR baseline is
`worktrees/vllm-60097-base/`. Generated run data remains under the ignored
`results/kernel/issue60095/` directory.
