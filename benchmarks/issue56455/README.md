# vLLM PR #56455: bounded prefill admission

This directory holds the independent reproduction harness and notes for
[PR #56455](https://github.com/vllm-project/vllm/pull/56455). The upstream
source stays in separate Git worktrees under `worktrees/`.

## Pinned source

| Arm | Worktree | SHA |
| --- | --- | --- |
| Stock | `worktrees/vllm-56455-base` | `4b839c3378c6a40fa613328a9b97069ed83b181b` |
| PR, slots 0 or 2 | `worktrees/vllm-56455-pr` | `418ccdb5d2f8e6854516514594f8d483b3902257` |

The base is the PR commit's direct parent. Recheck the upstream PR before a
later experiment because its head can move. The PR was open on 2026-10-15.

## First local experiment

Run `./benchmarks/issue56455/run_local.sh` from the lab root. It starts a fresh
Qwen3-0.6B service for each of three arms (stock, PR slots=0, PR slots=2),
then sends 32 fixed, streaming requests at concurrency 16. The 8 GiB 3070 Ti
settings use `max_num_seqs=8`, `max_num_batched_tokens=512`, 256 input tokens,
and 64 output tokens. Raw request timing and summaries are saved below
`results/issue56455/` (ignored by Git). Set `ISSUE56455_REQUESTS` or
`ISSUE56455_CONCURRENCY` to change the local load.

The runner links the existing environment's compiled CUDA extensions into both
worktrees. These ignored symlinks keep the same binary build in each arm. The
binary build came from the `src/vllm` checkout, so any binary interface mismatch
must be resolved before interpreting benchmark results.

The client records stream chunk gaps as an ITL proxy. Text chunks can contain
more than one token, so these gaps are not exact token-level ITL. This first
experiment checks service compatibility and whether tail-gap behavior appears
on the local GPU. It does not reproduce the author's Qwen2.5-7B, ShareGPT,
2000-request, concurrency-256 result. A later full run needs the same dataset,
large-memory GPU, repeated cold-server A/B, and a token-accurate ITL method.
