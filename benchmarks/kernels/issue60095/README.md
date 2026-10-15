# Issue #60095 token-logprob benchmark

## Directory map and current experiment

The shared benchmark entry point is `bench_token_logprobs.py`. The original
base/PR analysis is `run_repeated_ab.sh` and `analyze_repeats.py`, with its
published H200 evidence in `report_h200_60097/`. The newer split-selection
scripts and fixed-rule patch are in `split_selection/`; see its README for
the entry points. The cumulative results, startup-autotune experiment, and
H200 retest plan are in `end.md`. Raw new runs are under the ignored
`results/kernel/issue60095/` tree.

The lab repository and the edited vLLM candidate worktree currently use the
same branch name, `codex/issue60095-split-selection`, but they are **two
independent Git repositories**. `worktrees/vllm-60097-pr/` is the edited
candidate, while `worktrees/vllm-60097-pr-reference/` is the clean PR
reference. The `src/vllm/` checkout remains on its separate learning branch.
Open `issue60095.code-workspace` from the lab root in VS Code to see all four
folders and their Git state separately. No experiment files are committed or
pushed automatically.

This microbenchmark compares the original token-logprob kernel at the base
commit with PR #60097's automatic and forced vocabulary split counts. It uses
synthetic logits with the production Qwen vocabulary size and does not load a
model. Run the same command once per source worktree, on the same idle GPU and
with the same Python environment.

## Source worktrees

The active learning checkout remains untouched. These worktrees share the
underlying vLLM repository:

- `worktrees/vllm-60097-base`: `d09ff7763747bbe7cd153edd8017479804f47be7`
- `worktrees/vllm-60097-pr-reference`: clean detached PR reference at `0ddd6017793c943a2091a5800f0cbac1daaa3e02`.
- `worktrees/vllm-60097-pr`: edited candidate on `codex/issue60095-split-selection`, based on the PR commit. Its directory name is historical; it is not the clean PR reference.

Recreate the clean references after removal with:

```bash
LAB_ROOT="$(git rev-parse --show-toplevel)"
git -C "$LAB_ROOT/src/vllm" fetch origin refs/pull/60097/head:refs/remotes/origin/pr/60097
git -C "$LAB_ROOT/src/vllm" worktree add --detach \
  "$LAB_ROOT/worktrees/vllm-60097-base" d09ff7763747bbe7cd153edd8017479804f47be7
git -C "$LAB_ROOT/src/vllm" worktree add --detach \
  "$LAB_ROOT/worktrees/vllm-60097-pr-reference" 0ddd6017793c943a2091a5800f0cbac1daaa3e02
```

The candidate has **uncommitted local source edits**. Creating a new worktree
from the PR commit will not recreate them. Keep it until its full changes are
committed or exported. The older `split_selection/selector_candidate_sm86.patch`
contains only the fixed-rule stage, not startup autotuning.

On a fresh machine, clone the outer lab repository first, then clone vLLM into
its ignored `src/vllm/` directory. The following commands assume the outer
repository is the current directory:

```bash
LAB_ROOT="$(git rev-parse --show-toplevel)"
mkdir -p "$LAB_ROOT/src" "$LAB_ROOT/worktrees"
git clone https://github.com/vllm-project/vllm.git "$LAB_ROOT/src/vllm"
git -C "$LAB_ROOT/src/vllm" fetch origin refs/pull/60097/head:refs/remotes/origin/pr/60097
git -C "$LAB_ROOT/src/vllm" worktree add --detach \
  "$LAB_ROOT/worktrees/vllm-60097-base" d09ff7763747bbe7cd153edd8017479804f47be7
git -C "$LAB_ROOT/src/vllm" worktree add --detach \
  "$LAB_ROOT/worktrees/vllm-60097-pr-reference" 0ddd6017793c943a2091a5800f0cbac1daaa3e02
```

The exact source commits above are pinned so a later move of the PR branch does
not silently change the experiment. Create the Python environment under
`src/vllm/.venv` and build/install the vLLM extensions from the base source on
that machine. If the extensions are built in the base worktree, point
`VLLM_BINARY_ROOT` there; the local environment defaults to `src/vllm`. Both
arms must use the same binary root.

The benchmark loads Python files from the selected worktree and reuses the
compiled vLLM extension modules from the installed checkout. This keeps the
compiled extensions constant while isolating the Python/Triton implementation
under study. The result records both source and extension roots; for final
serving validation, build/install the tested source commit in its own
environment.

## Run

First check the local driver from the same machine/container where the GPU is
visible. Then run the two arms separately:

```bash
LAB_ROOT="$(git rev-parse --show-toplevel)"
PYTHON="$LAB_ROOT/src/vllm/.venv/bin/python"
BINARY_ROOT="${VLLM_BINARY_ROOT:-$LAB_ROOT/src/vllm}"
"$PYTHON" "$LAB_ROOT/benchmarks/kernels/issue60095/bench_token_logprobs.py" \
  --binary-root "$BINARY_ROOT" \
  --vllm-root "$LAB_ROOT/worktrees/vllm-60097-base"

"$PYTHON" "$LAB_ROOT/benchmarks/kernels/issue60095/bench_token_logprobs.py" \
  --binary-root "$BINARY_ROOT" \
  --vllm-root "$LAB_ROOT/worktrees/vllm-60097-pr" \
  --splits auto,1,2,4,8,16,32
```

By default the benchmark runs the full batch sweep
`1,2,4,8,16,32,64,128,256`. Pass `--full` to request that sweep explicitly,
or use `--select` with a comma-separated list to run only selected batch sizes.
For example, these commands run the key sizes `1,16,32,64,128` on both base
and PR:

```bash
LAB_ROOT="$(git rev-parse --show-toplevel)"
PYTHON="$LAB_ROOT/src/vllm/.venv/bin/python"
BINARY_ROOT="${VLLM_BINARY_ROOT:-$LAB_ROOT/src/vllm}"
"$PYTHON" "$LAB_ROOT/benchmarks/kernels/issue60095/bench_token_logprobs.py" \
  --binary-root "$BINARY_ROOT" \
  --vllm-root "$LAB_ROOT/worktrees/vllm-60097-base" \
  --select 1,16,32,64,128

"$PYTHON" "$LAB_ROOT/benchmarks/kernels/issue60095/bench_token_logprobs.py" \
  --binary-root "$BINARY_ROOT" \
  --vllm-root "$LAB_ROOT/worktrees/vllm-60097-pr" \
  --select 1,16,32,64,128 \
  --splits auto,1,2,4,8,16,32
```

`--full` and `--select` are mutually exclusive. The selected sizes are recorded
in result metadata.

The benchmark uses vocabulary size 151,936 and six returned token logprobs. It checks outputs against a float64
`torch.log_softmax` reference, measures first-call wall time after CUDA context
initialization, warmed synchronized call time, and CUDA Graph device time.
Each run writes `result.json` and `result.csv` under
`results/kernel/issue60095/` and records the GPU, source commit, PyTorch,
CUDA, and Triton versions.

The first-call measurement includes compilation for each new kernel variant
only when the Triton cache is empty. To measure cold compilation, use a new
`TRITON_CACHE_DIR` for each process. Keep cache state fixed when comparing
warm timings. The kernel benchmark does not measure server TTFT; that requires
a later end-to-end run with a real request that asks for logprobs.

On a GeForce RTX 3070 Ti this experiment can validate the code path and show
Ampere behavior. It cannot reproduce the H200 result or establish the best
heuristic for Hopper.

## Analyze repeated base/PR runs

The repeated runner below creates `r1_base`, `r1_pr`, `r2_base`, `r2_pr`,
`r3_base`, and `r3_pr` under a timestamped `run_<timestamp>/` folder. To
analyze the newest run automatically, use:

```bash
LAB_ROOT="$(git rev-parse --show-toplevel)"
"$LAB_ROOT/src/vllm/.venv/bin/python" \
  "$LAB_ROOT/benchmarks/kernels/issue60095/analyze_repeats.py"
```

The script uses only each PR run's `requested_splits=auto` rows, compares them
with base, and creates a new `analysis_<timestamp>/` directory inside that run
folder. It writes `comparison.csv`, `report.md`, metadata JSON, three
per-round grouped bar charts, and separate base/PR latency variation charts.
Three repeats provide an initial consistency check, not a robust statistical
distribution.

To analyze a specific run instead of the newest one, pass its path with
`--input-dir`, for example `--input-dir "$LAB_ROOT/results/kernel/issue60095/repeats/run_<timestamp>"`.

To run the three alternating base/PR rounds and analyze them in one step, use:

```bash
benchmarks/kernels/issue60095/run_repeated_ab.sh
```

Each invocation creates a unique `run_<timestamp>/` under the repeats directory
so it does not overwrite earlier measurements. Set `VLLM_BINARY_ROOT` first if
the compiled extensions are not in `src/vllm`.
