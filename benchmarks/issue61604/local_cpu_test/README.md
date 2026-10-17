# Local CPU validation for issue #61604

The latest CPU results and scope limits are in
[REPORT_2026-10-18.md](/home/minstrel/vllm_lab/results/issue61604/local_cpu_test/REPORT_2026-10-18.md).
The common-path and threshold ablation is in
[ABLATION_REPORT_2026-10-18.md](/home/minstrel/vllm_lab/results/issue61604/local_cpu_test/ABLATION_REPORT_2026-10-18.md).

This round ran on the Intel Core i7-12700H with CPU scheduler/KV manager only:

- test_prefix_caching.py -k waiting_prefix_memo: 17 passed.
- Full test_prefix_caching.py: 189 passed.
- Adverse lookup A/B: actual baseline worktree vs patch memo-on and patch memo-off, three alternating rounds, 500 samples plus 50 warmups per scenario.
- Same-prefix scheduler replay: 12 requests, 6 preemptions, one cancellation; baseline and patch traces identical.
- Memo threshold: only create records after failed admission with at least four actual cached blocks; the 3-block and 4-block boundary tests pass.
- Thresholded four-prefix churn replay: 20 requests, 37 steps, 21 failed allocations, 17 cache-map removals, 17 preemptions; patch records a five-block memo and both worktrees have identical schedule/resource/free traces.
- Under fixed CPU affinity, short-prefix replay creates no memo and is +4.4% versus base; long-prefix churn creates a five-block memo and is +6.3% versus base (three rounds each). These are millisecond-scale scheduler CPU replays, not serving throughput.
- Stable long-prefix wait after threshold: five paired 1000-step full-scheduler runs, 8192-token hit; schedule median 269.70 ms base to 110.13 ms patch (-59.2%), with matching traces.
- Active waiter memory stress: 100 failed admissions made 100 entries/400 block references (~31 KB shallow Python object size); eviction cleared all memos; free and weak-key collection returned the entry count to zero.
- Latest ablation: status-first, single weak-map lookup change; base/memo-off and 4/8/16-block thresholds tested on short, five-block churn, and 8192-token stable-wait replays. Threshold 16 skipped five-block memo construction and retained the long-wait reduction (~59%); the production default remains four pending full correctness validation.

The corrected results are stored in
/home/minstrel/vllm_lab/results/issue61604/local_cpu_test/adverse-retest-20261018/
and
/home/minstrel/vllm_lab/results/issue61604/local_cpu_test/dynamic-retest-20261018/.
The latest combined replay, timing-only churn, and stable-wait results are in
/home/minstrel/vllm_lab/results/issue61604/local_cpu_test/combined-timing-retest-20261018/,
/home/minstrel/vllm_lab/results/issue61604/local_cpu_test/dynamic-retest-20261018/, and
/home/minstrel/vllm_lab/results/issue61604/local_cpu_test/stable-wait-retest-20261018/.
Threshold-specific results are in
/home/minstrel/vllm_lab/results/issue61604/local_cpu_test/threshold-retest-20261018/.
The erroneous earlier adverse result files were deleted. The currently valid performance run records the true baseline worktree in each JSON and manifest.

Candidate is not frozen. Memo-off is close to base on stable wait, while enabled thresholds retain the target gain. Threshold 16 reduces the measured five-block churn cost, but short-prefix common-path overhead remains. Validate the target serving workload on H100/H200 before deciding whether to keep this optimization. No GPU E2E was run.
