# Remote Experiment Agent Prompt: vLLM Issue #61604

You are responsible for continuing the remote H100/H200 experiment for vLLM issue #61604. Your task is to determine why the current implementation shows only a small and uncertain end-to-end gain, and whether the main limitation is (A) the memo lookup path itself saves little CPU time or (B) it saves CPU time, but scheduler work is too small a fraction of total service time in the tested shape. If B is supported, adjust the workload shape and continue experiments. Do not optimize production code merely because end-to-end throughput is small.

## User intent and decision rule

The optimization memoizes a long prefix-cache hit after a failed KV allocation, so a waiting request can avoid repeatedly querying the cache map for the same prefix. Establish the causal chain:

`memo lookup -> prefix lookup CPU saved -> scheduler CPU saved -> GPU/service idle time or end-to-end metric changed`

Use this decision rule:

1. If a matched mechanism measurement shows that memo materially reduces prefix-lookup CPU time and total `schedule()` CPU time, but lookup/scheduler time is only a small fraction of the service critical path, classify the issue as a low scheduler-time share for this shape. **Keep the production implementation unchanged; adjust the experimental shape and continue testing** to find a naturally occurring, repeatable workload where scheduler CPU is more visible. Report the shape search and keep all tried shapes, not only the winner.
2. If memo is actually exercised but does not materially reduce prefix-lookup time, or adds enough validation/recording work that the memo path itself is neutral or slower, inspect and profile the production path before proposing implementation changes. Identify the measured operations responsible. Make the smallest change only after evidence points to a specific cost, then rerun correctness and matched mechanism measurements.
3. If memo is rarely exercised, the shape is invalid for judging this optimization. Fix the workload/queue/capacity configuration until real failed admission and repeated long-prefix retries occur naturally. Never fake allocation failure or inject sleeps.
4. If the measured intervals overlap run-to-run variation, report the result as unresolved. Do not tune the implementation to noise, cherry-pick a favorable round, or claim equivalence from an underpowered test.

## Existing experiment and known caveats

Read these before running anything:

- Main experiment protocol: `benchmarks/issue61604/AGENT.md`.
- Current H100 stage report: `results/issue61604/hopper/20261017T230937Z/REPORT_H100_E2E.md`.
- Existing scripts: `benchmarks/issue61604/hopper/`.
- Existing raw result bundle: `results/issue61604/hopper/20261017T230937Z/`.
- Current candidate patch: `benchmarks/issue61604/hopper/issue61604-lifecycle-20261017T230937Z.patch` and its source manifest. Verify hashes and the actual source tree before use; do not assume branch names alone identify the tested code.

The prior H100 run used Qwen/Qwen3-0.6B, one H100 80GB, synchronous scheduling, 16,384 shared prefix tokens, 128 unique suffix tokens, 256 output tokens, 32 closed-loop concurrent requests, block size 16, and 1,130 KV blocks. It completed five base/patch pairs. Patch throughput averaged +3.36%, but the paired 95% interval was -0.57% to +7.29%, crossing zero. Four pairs improved and one regressed. A same-seed Base rerun differed from the first Base run by about -4.55%, so the current measured gain is comparable to observed run variation.

The prior shape also caused heavy pressure: about 304–324 preemptions per 200 requests. A rough capacity estimate is 1,024 blocks for the shared prefix, leaving about 106 blocks from a 1,130-block budget; each request needs roughly 24 additional blocks for its unique suffix and 256-token generation. This can leave the H100 underfilled and add preemption/recomputation work unrelated to memo lookup. Treat this as a hypothesis to measure, not as a proven explanation.

The previous diagnostic counters are not trustworthy for invalidation or saved-work accounting. `audit_site/sitecustomize.py` increments `memo_invalidations` whenever `_invalidate_prefix_memos` is called, without checking whether any entry was removed; it adds the pre-call table size to `memo_entries_invalidated` regardless of the post-call size. Its `memoized_tokens_returned` includes the returned prefix length, which can include fallback map lookup. `remember_failed_prefix_calls` counts calls, not distinct entries built. Do not quote the old 16,404 invalidations, 132.4M memoized tokens, or 8,040 memo creations as literal actual invalidations, saved tokens, or created entries.

The previous performance environment reused the same 13 native extensions from a later vLLM build because a clean build failed due to CUDA/nvcc mismatch. Both arms shared them, but this is not a clean build of each source snapshot. Prefer a matched clean environment. If impossible, preserve the limitation, verify imports and hashes, and do not present it as a clean reproducible upstream benchmark.

The old output report wording is also too strong. Same-seed Base/Base had 5.88% differing token positions; pair 5 Base/Patch had 5.96%, and pair 1 had substantially more divergence. This does not establish KV corruption, but it also does not establish correctness. Preserve this as an open investigation and compare first-divergence positions/logprobs and, when practical, a prefix-cache-disabled control.

## Non-negotiable safety and execution rules

1. Do not modify the user's unrelated workspace, commit, push, open a PR, or message maintainers. Do not provision/stop paid resources unless separately authorized. Do not run on a machine/GPU unless the user has provided or explicitly authorized access to it.
2. Before a run, record host/GPU identity, available CPU cores, affinity/NUMA, clocks/power if available, software versions, model revision, Python import paths, source commit/patch hash, backend, scheduler mode, model/cache spec and whether the memo gate is enabled.
3. Use the same host, GPU, model files, tokenizer, environment, attention backend, runtime options, warmup, request trace, and instrumentation for Base and Patch. Run one server at a time. Alternate AB/BA order. Separate diagnostic runs from clean performance runs.
4. Keep the patch and base source immutable for the experiment. At start and end, verify source hashes and git status. Save a copy of the full source diff and manifests with each result bundle.
5. Do not run a broad parameter grid. Change one workload dimension at a time, use a base calibration to locate valid natural waiting, then test a small number of candidate shapes.
6. Do not tune thresholds per shape or after looking at confirmation data. Keep the current threshold fixed at 4 for the initial diagnosis. Any threshold experiment must be a separate, disclosed tuning phase, and then the selected threshold must be frozen before confirmation.
7. Do not add high-frequency Python probes to timed performance runs. Use low-overhead clean timing runs for throughput and separate diagnostic/profile runs for mechanism attribution. Quantify probe overhead with an instrumentation-off/on control if instrumentation is necessary.
8. Preserve every valid run, including unfavorable ones. Predefine exclusion reasons; never discard a run because its result is inconvenient.
9. Do not add tests or production code changes unless needed to investigate a measured defect. If code changes, state the evidence and rerun the relevant correctness tests before remote performance testing.

## Phase 0: Audit the inputs and reproduce the current result

1. Read `AGENT.md`, the H100 report, all scripts, source manifest, patch, and raw summaries. Inspect the actual A/B commands and metric definitions.
2. Verify the base SHA and exact complete patch. Confirm the candidate imports from the intended source roots. Confirm the runtime `UnitaryKVCacheCoordinator`, exact `FullAttentionSpec`, no EAGLE groups, DCP=PCP=1, and `prefix_memo_enabled=true` in the patch. If any gate is false, stop treating the run as an #61604 patch run.
3. Check CPU allocation/affinity and GPU health. Record CPU model, number of usable cores, NUMA node, scheduler process affinity, GPU model and power/clock behavior. If the host differs from the previous result, describe the difference.
4. Reproduce one or two prior pairs only as a smoke check. Do not spend the full budget repeating five pairs before fixing the mechanism counters and understanding the shape.

## Phase 1: Measure the memo's direct value

Build a low-overhead, matched measurement for Base and Patch at the existing 16K/1,130-block shape, plus one less severe capacity point if needed. Keep the timed run uninstrumented. Run separate diagnostic/profile cases with bounded sampling.

Collect at least:

- End-to-end: completed requests, failures/timeouts, actual generated tokens, wall time, output tokens/s, requests/s, TTFT and E2E p50/p95/p99. State whether drain time is included.
- Scheduler: total scheduler CPU time and `schedule()` call count, mean/p50/p95/p99 schedule duration, schedule CPU time per completed request and per scheduler step. Distinguish CPU time from wall-clock duration.
- Prefix lookup: count and cumulative CPU time for ordinary `find_longest_cache_hit` and `_find_memoized_prefix`, separately. Measure memo validation time, fallback map lookup time, memo build time, and the time in the unmodified base path. Use a profiler or carefully sampled instrumentation; document overhead.
- Real trigger coverage: failed `allocate_slots` count, failed requests, repeated failures per request, request wait duration/steps, actual prefix blocks found, memo entries created, memo hits, validated memo blocks, first stale block index, fallback blocks/probes, genuine entries removed, and why they were removed (reset vs lifecycle mismatch vs request cleanup). Each count must have a precise event definition.
- Serving state: running/waiting request counts over time, active sequences, scheduled tokens, preemptions, KV used/free blocks, cache insertions/evictions, GPU iteration time/idle gaps/utilization, and model-forward/decode time where reliable.
- Memory/lifecycle: memo record count and retained block references over time, RSS, and whether records return to baseline after requests finish and cache drains. Memo must not pin KV blocks.

For invalidation, count actual before/after entries and identify the removal reason; do not count calls as removals. For saved lookup work, compare probes/time against the corresponding ordinary lookup; returned token count is not saved work. Avoid double-counting fallback work as a memo hit.

Use an additional Patch-with-memo-disabled arm only if it can be produced without changing other code or environment. This isolates public lifecycle-field maintenance and common-path overhead from memo benefit. Label it as an ablation, not as Base.

### Phase 1 decision

Compute and report the observed fractions:

- prefix lookup CPU / scheduler CPU
- scheduler CPU / total server CPU
- scheduler CPU / end-to-end critical-path time (with a clear measurement caveat because CPU and GPU overlap)
- time saved by memo / total service wall time

If the direct memo path is materially faster and `schedule()` improves, but scheduler time is only a small share of service time, proceed to Phase 2. If the memo path itself does not reduce lookup or scheduler cost, stay on this shape long enough to profile the exact overhead, then decide whether a narrowly scoped implementation change is justified. If the memo does not trigger, repair the scenario before drawing conclusions.

## Phase 2: Adjust shape only when scheduler share is the limiter

This phase is required if Phase 1 shows useful lookup/scheduler savings but small end-to-end effect. Keep production code unchanged while searching.

The previous 1,130-block setup may be too aggressive because it produced many preemptions and little KV headroom. Calibrate with Base first, then try a small sequential set of block budgets around a capacity that:

- naturally creates repeated failed admissions for requests with long cache hits;
- retains several active requests so the GPU continues doing useful work;
- does not produce pathological preemption/recomputation;
- does not hit `max_num_seqs` or token-budget limits instead of KV allocation limits.

Do not assume a higher budget is better: it may eliminate the target retry. Do not assume lower budget increases signal: it may starve the GPU. Use observed queue, wait, preemption, free-block, and GPU-step data to choose the next point.

Suggested bounded search, changing one variable at a time:

1. Fix model/revision, prefix length, suffix, output, concurrency, and token budget. Adjust only KV block budget around the measured natural-wait threshold to reduce preemption while retaining repeated memo opportunities.
2. At the best few block budgets, compare a small set of shared-prefix lengths (for example 8K, 16K, and, only if the model context supports it, 32K). Use the base to confirm each produces a long hit and real retries.
3. Only then adjust concurrency or unique suffix/output length if necessary to maintain useful GPU occupancy and controlled waiting. Do not run a full Cartesian product.
4. Keep at least one neighboring shape to determine whether the result is a narrow isolated point.

Use a short exploratory screen first (for example 2–3 paired runs per candidate shape) and identify it as exploratory. Select by a predeclared rule that jointly considers direct memo savings, enough trigger frequency, completed throughput, preemption, and latency. Do not select solely by the highest single tokens/s number. Record all candidates and why each was retained/rejected.

Once a shape is selected, freeze all parameters and use new seeds/traces for at least 5 paired confirmation runs, preferably 10 if the gain is small or noisy. Alternate AB/BA. Do not include exploratory runs in confirmation statistics. Measure the same selected shape with default asynchronous scheduling and at least one neighboring shape. Keep default-capacity and short/no-shared-prefix cases as regressions; a synthetic high-signal shape does not establish general service benefit.

## Phase 3: Correctness and performance interpretation

- Compare base-base variability with base-patch variability under the same seeds and runtime. Save exact prompt IDs, output IDs, finish reasons, and first divergence positions. Temperature zero is not proof of bitwise determinism.
- Investigate output differences with first-divergence logits/logprobs if available and a prefix-cache-disabled control for a small fixed request set. Do not call differences harmless solely because Base/Base also differs; do not call them KV corruption without evidence.
- Check admissions, preemption/resume, block IDs where useful, equivalent replicas, refcounts/free queue operations, cancellations, and post-drain zero/leak behavior. Allow a different valid physical replica only when its complete key and reference lifecycle are correct.
- Report throughput gains separately from elapsed-time reductions. Use actual output-token counts. Analyze at the run-pair level; 200 requests within one run are not 200 independent A/B replicates.
- Report confidence intervals and run ranges. If the interval crosses zero or the effect is within measured base repeatability, say that the result is inconclusive.
- Distinguish the best measured synthetic high-signal shape from default/representative workloads. It is acceptable for the best shape to show a clear gain while default async shows no measurable improvement, provided both are reported honestly.

## Required artifacts and final response

Write results under `results/issue61604/hopper/<new-UTC-run-id>/` without overwriting or deleting the existing bundle. Include:

- immutable source manifest and full patch/diff;
- host/GPU/runtime/model manifests and exact commands;
- per-run case JSON, service/client logs, metrics snapshots, request-level output records, GPU/CPU samples;
- separate clean performance and diagnostic/profile output;
- precise instrumentation definitions and probe-overhead estimate;
- all exploratory and confirmation runs, including failed/invalid runs with reasons;
- scripts used to summarize and calculate paired intervals;
- a report with direct answers to the decision rule above.

Final report must state:

1. Did the memo path itself reduce lookup CPU time? By how much, with uncertainty?
2. Did it reduce total `schedule()` CPU time? What fraction of total serving work was that?
3. Was the small E2E effect explained mainly by low scheduler share, preemption/recomputation, run noise, or an actual memo-path overhead? Label unproven explanations as hypotheses.
4. If low scheduler share was the cause, what shapes were tried, which was selected, and did the adjusted shape preserve natural retries while improving the signal? Include the best measured shape and independent confirmation, even if the outcome is negative.
5. What happened under default async, default capacity, and adverse workloads?
6. Were output differences and memory/reference lifecycle issues explained?
7. Is there evidence to change production code, continue shape exploration, freeze a candidate, or stop the experiment?

Do not claim a positive or negative conclusion beyond what the measurements establish. The key task is causal diagnosis first, then shape adjustment if the memo is useful but hidden by the rest of serving time.
