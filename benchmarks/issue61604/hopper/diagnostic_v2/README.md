# H100 causal diagnosis, v2

This runner addresses counter-definition errors in the first H100 attempt. It
does not modify vLLM production source. It measures the following separately:

- `schedule_cpu_ns` and `schedule_wall_ns`: time inside each real
  `Scheduler.schedule()` call, measured with the scheduler thread CPU clock and
  monotonic wall clock. Samples are saved so p50/p95/p99 can be calculated.
- `ordinary_lookup_*`: inclusive CPU/wall time in the concrete coordinator's
  `find_longest_cache_hit`; map probes and successful map probes are counted
  separately through `BlockPool.get_cached_block`.
- `memo_lookup_*`: inclusive CPU/wall time in `_find_memoized_prefix`.
  `memo_validated_blocks` is returned memo-path blocks minus successful fallback
  map probes; it is a block count, not a token count or a saved-work estimate.
- `memo_fallback_*`: map probes, hits, and inclusive CPU time from
  `BlockPool.get_cached_block` while the memo method is active. The remainder
  `memo_lookup_cpu_ns - memo_fallback_cpu_ns` is memo validation/list assembly
  time; it is not summed again with the inclusive memo time.
- `memo_build_*`: CPU/wall time in `_remember_failed_prefix`. An entry is
  counted only when the request key is absent before the call and present after
  it. Calls that find an existing entry are not creations.
- `memo_invalidation_checks`: calls to `_invalidate_prefix_memos`; this is a
  check count only. `memo_entries_removed_on_reset` increments only by the
  before/after weak-table length difference when a successful reset clears
  records or the block-map object changes. Successful reset calls, map-pointer
  resynchronizations, allocation cleanup, free cleanup, and lazy validation
  truncation are reported as distinct events.
- `allocate_slots_failures`: actual calls that returned `None`, indexed by
  request id. `repeated_failure_requests` counts request ids with more than one
  failed allocation. The scheduler trace records running/waiting queue lengths,
  scheduled requests/tokens, and preemption events per schedule step.
- `memo_entry_count_peak` and `memo_block_records_peak`: weak memo table and
  stored record peaks. They do not imply KV block references; values are weakly
  held by the request-keyed table.

Diagnostics are deliberately opt-in and start only after the separate warmup
request. The runner creates a shared marker file; each instrumented Python
process watches for it, so activation does not depend on signals that vLLM may
replace during startup. The same watcher atomically checkpoints counters every
five seconds because normal API shutdown can force-terminate EngineCore before
its exit hook runs. The activation manifest and resulting scheduler file must
show that the measured workload recorded nonzero calls; otherwise the
diagnostic run is invalid. Base-version instrumentation must treat memo fields
as optional because the base `KVCacheManager` has no memo table. The service
runner can disable prefix caching with `--no-enable-prefix-caching` for a small
output-stability control; this arm is not an optimization A/B.
The API metrics sampler also starts after warmup. For clean performance runs,
omit `--diagnostic`; the source overlay remains identical, but no method
wrappers are installed. Compare one same-trace instrumented and uninstrumented
patch run to quantify probe overhead before interpreting diagnostic timings.

The source overlay takes Python from the selected base/patch worktree and native
extensions from the same installed vLLM wheel. This is necessary on the current
Pod because `nvcc` is absent. The wheel's stable-libtorch extensions load under
the installed Torch 2.14.0+cu130, but were not rebuilt from base SHA
`a2fb4827324bf023075799ef3d70d347834b8137`; every report must preserve that
limitation. The local `/tmp` cache copies the runtime Torch Python files (not
headers/tests) only to avoid FUSE metadata latency; it does not alter package
contents or the workspace checkout.

`load_client.py` writes compressed exact input token-id traces before sending
requests and request-level output token ids/finish reasons after completion.
The service warmup uses the first request from that same trace with only eight
decode tokens, so paired runs enter measurement with the same shared prefix in
KV cache. The patch arm can load the saved base trace directly; this avoids
slow repeated Transformers imports on the Pod's FUSE-mounted package volume.
Exploration and confirmation directories must remain distinct. No run is
excluded because its measured result is unfavorable.
