# Issue #61604: waiting prefix lookup memo

Date: 2026-10-18 (Asia/Singapore).
Baseline: `a2fb4827324bf023075799ef3d70d347834b8137`.
Implementation: `worktrees/issue-61604-pr`, branch `issue-61604-pr`.
Baseline worktree: `worktrees/issue-61604-base`.

## Duplicate-work check

GitHub CLI was unavailable; equivalent read-only GitHub REST searches were used.
Snapshots are under `evidence/`: issue and comments, PRs mentioning 61604,
open PRs matching prefix/memo and prefix/lookup (all three result pages), and
PRs by the issue author. No matching implementation was identified in these
searches. This is a search result, not proof that no unpublished patch exists.

- https://github.com/vllm-project/vllm/issues/61604 describes a local patch;
  the retrieved issue/comments do not provide that patch.
- https://github.com/vllm-project/vllm/pull/59322 avoids scans when updating
  offload keys, rather than caching GPU prefix-hit results across admission retries.
- https://github.com/vllm-project/vllm/pull/54366 changes eviction selection and
  memoizes within an allocation transaction, not across waiting retries.
- #58837 changes admission policy. These changes may interact with this work,
  but are not the same implementation.

Our implementation is independently written from the issue's described idea.
No author patch has been applied; no GitHub comment or PR has been posted.

## CPU reproduction

From `/home/minstrel/vllm_lab`:

```bash
PYTHONPATH=worktrees/issue-61604-base VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 \
  src/vllm/.venv/bin/python benchmarks/issue61604/reproduce.py \
  --output benchmarks/issue61604/base.json
PYTHONPATH=worktrees/issue-61604-pr VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 \
  src/vllm/.venv/bin/python benchmarks/issue61604/reproduce.py \
  --output benchmarks/issue61604/pr.json
```

Uses the existing virtual environment (torch 2.13.0+cu132) and cached
Qwen3-0.6B configuration; no model weights are executed and no GPU is needed.
The real scheduler and KV manager are exercised. A holder caches 8192 tokens;
a pool allocation occupies the remaining capacity. A waiting request has the
same prefix plus 128 tokens. Its admission fails on 1000 scheduler steps.
After capacity is released, the script asserts successful admission.

Counters and timings cover only the 1000 blocked steps, not final admission.
Each JSON includes source commit and manager file SHA256. Instrumentation counts
`get_cached_block` calls, not every low-level dictionary operation. Timings include
instrumentation overhead. The repeated empty scheduling trace is hashed;
this is a stable-wait scenario, not a proof of dynamic workload equivalence.
Three alternating baseline/patch pairs are in `base-1.json` through `pr-3.json`.
This deliberately stable workload favors memoization. Do not interpret its
CPU improvement as end-to-end serving improvement. Mixed/churning workloads,
first-lookup overhead, asynchronous serving, and GPU model evaluation remain pending.

## Validity and lifecycle

The fast path is restricted to the exact built-in Unitary coordinator and
FullAttentionSpec, no EAGLE group drop, DCP=PCP=1, caching enabled, and request
status WAITING or PREEMPTED. Other configurations keep the coordinator lookup.
Spec subclasses are conservatively excluded.

- Cache entries use request object identity via weak keys, not request IDs.
  Reusing an ID cannot inherit another request's memo.
- Each remembered block must still have the expected primary hash, and the
  request hash at that position must still match. Stop at the first mismatch.
- Only primary-hash matches are remembered. Alias matches are looked up again.
- Resume normal table lookups at the first unvalidated block, including past
  a fully valid old hit, so newly cached tails can extend the hit.
- Limit hits to `(request.num_tokens - 1) // block_size`, preserving final-token
  recomputation. Truncation shortens the remembered prefix.
- Eviction, hash movement, and reset clear/change the primary hash through
  BlockPool. A block reused with the same complete chained hash still represents
  the same prefix under the existing prefix-cache hash correctness assumption.
- Memo references do not touch block refcounts, pin blocks, or reorder the free
  queue. Allocation failure keeps the memo for a retry.
- Successful allocation clears the memo before either normal or delayed-cache
  return. `free(request)` clears it on release/preemption/termination.
- Successful prefix-cache reset clears all memos; failed reset leaves them valid.
  Weak keys also allow abandoned requests to be collected.
- Existing cache event emission remains after lookup, shared by both paths.

## Validation

```bash
PYTHONPATH=worktrees/issue-61604-pr VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 \
  src/vllm/.venv/bin/python -m pytest --noconftest \
  worktrees/issue-61604-pr/tests/v1/core/test_prefix_caching.py -q
```

180 passed, including 8 new cases. `--noconftest` avoids unrelated global/GPU
fixtures; this is CPU validation, not the upstream CI configuration.
Coverage includes prefix growth, middle-block eviction, reset, pool reuse
without pinning, admission/free cleanup, alias fallback, EAGLE comparison,
and DCP/PCP gating. Full integration/model correctness remains to be validated.
Ruff and `git diff --check` pass for the changed files.

## Measured stable-wait result

| Round | Baseline schedule total (ms) | Patch schedule total (ms) |
| --- | ---: | ---: |
| 1 | 269.43 | 105.00 |
| 2 | 264.02 | 105.32 |
| 3 | 264.20 | 106.85 |

Median: 264.20 → 105.32 ms (about 60.1% lower). Every pair had 513000 baseline
probes versus 1512 patch probes (about 99.7% fewer), 1000 blocked allocations,
identical blocked scheduling traces, and successful admission after release.
The patch still validates every remembered block: this is a constant-factor
reduction in O(prefix blocks) work, not an O(1) lookup.
