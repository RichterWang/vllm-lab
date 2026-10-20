# Issue #61604 H100 diagnostic v3

This diagnostic version separates mechanism counting from CPU/wall timing. It is opt-in and does not modify vLLM production source.

## Modes

- `--diagnostic --diag-mode count`: counts real `allocate_slots` failures, failed requests/retries, memo lookup/build/reuse calls, successfully created weak entries, actual memo records validated, fallback block-map probes/hits, map deletions, actual memo truncations and first mismatch reason, lifecycle integrity checks, and reference acquisition/release for blocks returned from the memo. It uses no per-block clocks. It does copy the memo list in the rare/diagnostic lookup path to classify truncations; this mode is for behavior/correctness coverage, never timing claims.
- `--diagnostic --diag-mode timing`: measures only outer `Scheduler.schedule`, `KVCacheManager.get_computed_blocks`, ordinary `find_longest_cache_hit`, `_find_memoized_prefix`, and `_remember_failed_prefix` calls. CPU is `thread_time_ns`; elapsed time is `perf_counter_ns`. Every `*_calls` field is incremented once by the wrapper; `_add_timing` never changes call counts. `BlockPool.get_cached_block` has no wrapper in this mode: no per-block clock, counter, memo copy, or set construction. Nested timings are inclusive and overlap; do not sum them.
- No `--diagnostic`: no wrappers and no instrumentation; use this for throughput results.

The diagnostic watcher starts after warmup, checkpoints at a low rate, and writes a final post-drain snapshot. `run_case_v2.py` saves environment, exact server command, client outputs, metrics, GPU samples, and diagnostic records.

## Correctness hooks

Count mode checks that each reused memo block is the exact selected physical object, matches the request's complete block key and lifecycle version, remains present in the cache map, and appears in contiguous prefix order. At `allocate_new_computed_blocks`, it checks each selected memo block is passed through and has exactly one reference acquired. At coordinator free, it checks the matching reference is released. Any failed assertion aborts the run and writes a diagnostic event first. These checks are extra instrumentation, not production assertions.

A natural churn case can use `--scenario churn`, `--num-gpu-blocks-override 1130`, and `--cancel-index 35 --cancel-after-tokens 16`. The trace places the first 32 requests on long prefix A, then sends unrelated prefix families B through normal queueing. The cancellation is a client disconnect; verify server metrics and final queue state before counting it as observed cancellation.

Count mode also wraps `Scheduler._preempt_request` once per event and records request IDs. This wrapper adds no clocks and exists only to verify actual natural preemption in the bounded churn run.

## Logprobs

`--logprobs 5` records generated-token logprobs and top alternatives from `/v1/completions`. It does not request `prompt_logprobs`. The vLLM request path only skips prefix-cache reads when `SamplingParams.skip_reading_prefix_cache` is true; output `logprobs` alone does not set that flag. A high-pressure count-mode run with `--logprobs 5` must still show actual failures and memo reuse before it can be used as evidence that this logging mode retains the memo path.

Run `python -m py_compile *.py` and `bash -n run_remote_case_v3.sh` before upload. Use a new result root and never overwrite prior runs.
