"""Compare base and patch under multi-prefix scheduler pressure and cache churn."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--requests", type=int, default=20)
parser.add_argument("--num-blocks", type=int, default=8)
parser.add_argument("--prompt-tokens", type=int, default=80)
parser.add_argument("--shared-prefix-tokens", type=int, default=64)
parser.add_argument("--timing-only", action="store_true")
parser.add_argument("--memo-off", action="store_true")
parser.add_argument("--memo-threshold", type=int)
parser.add_argument("--expect-no-memo", action="store_true")
parser.add_argument("--profile-segments", action="store_true")
args = parser.parse_args()
assert 0 < args.shared_prefix_tokens < args.prompt_tokens
assert args.shared_prefix_tokens % 16 == 0
assert args.prompt_tokens % 16 == 0

scheduler = create_scheduler(
    enable_prefix_caching=True,
    num_blocks=args.num_blocks,
    block_size=16,
    max_num_seqs=4,
    max_num_batched_tokens=128,
    max_model_len=128,
)
requests = create_requests(
    args.requests,
    num_tokens=args.prompt_tokens,
    max_tokens=3,
    same_prompt=True,
    block_size=16,
    req_ids=[f"mix-{i}" for i in range(args.requests)],
)
prefix_groups = set()
for index, request in enumerate(requests):
    group = 0 if index < 4 else 1 + (index % 3)
    prefix_groups.add(group)
    request.prompt_token_ids = [1000] * args.shared_prefix_tokens + [
        2000 + group
    ] * (args.prompt_tokens - args.shared_prefix_tokens)
    request._all_token_ids = request.prompt_token_ids.copy()
    request.block_hashes.clear()
    request.update_block_hashes()

manager = scheduler.kv_cache_manager
if hasattr(manager, "_memoize_prefix_hits"):
    manager._memoize_prefix_hits = not args.memo_off
    if args.memo_threshold is not None:
        manager_module = sys.modules[manager.__class__.__module__]
        manager_module.MIN_PREFIX_HIT_BLOCKS_FOR_MEMO = args.memo_threshold
generated = {r.request_id: 0 for r in requests}
next_request = 0
trace = []
cancelled = []
free_events = []
lookup_waiting_ids = set()
counts = {"failed_allocations": 0, "memo_fast_path_calls": 0, "cache_key_deletions": 0}
segments = {
    "prefix_lookup_total_ns": 0,
    "memo_invalidate_ns": 0,
    "memo_fast_path_ns": 0,
    "memo_establish_ns": 0,
    "allocate_slots_total_ns": 0,
    "failed_allocate_slots_ns": 0,
}
profile_counts = {"allocate_slots_calls": 0, "failed_allocate_slots_calls": 0}
previous_keys = set()
original_lookup = manager.get_computed_blocks
original_allocate = manager.allocate_slots
original_free = manager.free
cache_map = manager.block_pool.cached_block_hash_to_block
original_map_pop = cache_map.pop
if args.profile_segments:
    original_get_computed = manager.get_computed_blocks
    original_allocate = manager.allocate_slots

    def timed_invalidate(*call_args, **kwargs):
        start = time.perf_counter_ns()
        try:
            return original_invalidate(*call_args, **kwargs)
        finally:
            segments["memo_invalidate_ns"] += time.perf_counter_ns() - start

    def timed_remember(*call_args, **kwargs):
        start = time.perf_counter_ns()
        try:
            return original_remember(*call_args, **kwargs)
        finally:
            segments["memo_establish_ns"] += time.perf_counter_ns() - start

    def timed_memo_lookup(*call_args, **kwargs):
        start = time.perf_counter_ns()
        try:
            return original_memo_lookup(*call_args, **kwargs)
        finally:
            segments["memo_fast_path_ns"] += time.perf_counter_ns() - start

    def timed_get_computed(request):
        start = time.perf_counter_ns()
        try:
            return original_get_computed(request)
        finally:
            segments["prefix_lookup_total_ns"] += time.perf_counter_ns() - start

    def timed_allocate(*call_args, _original=original_allocate, **kwargs):
        start = time.perf_counter_ns()
        result = _original(*call_args, **kwargs)
        elapsed = time.perf_counter_ns() - start
        segments["allocate_slots_total_ns"] += elapsed
        profile_counts["allocate_slots_calls"] += 1
        if result is None:
            segments["failed_allocate_slots_ns"] += elapsed
            profile_counts["failed_allocate_slots_calls"] += 1
        return result

    if hasattr(manager, "_invalidate_prefix_memos"):
        original_invalidate = manager._invalidate_prefix_memos
        original_remember = manager._remember_failed_prefix
        original_memo_lookup = manager._find_memoized_prefix
        manager._invalidate_prefix_memos = timed_invalidate
        manager._remember_failed_prefix = timed_remember
        manager._find_memoized_prefix = timed_memo_lookup
    manager.get_computed_blocks = timed_get_computed
    manager.allocate_slots = timed_allocate
    original_lookup = manager.get_computed_blocks
    original_allocate = manager.allocate_slots


def record_map_pop(key, block_id):
    existed = cache_map.contain(key, block_id)
    result = original_map_pop(key, block_id)
    if existed and result is not None:
        counts["cache_key_deletions"] += 1
    return result


def record_lookup(request):
    if request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
        lookup_waiting_ids.add(request.request_id)
    return original_lookup(request)


def record_allocate(*call_args, **kwargs):
    result = original_allocate(*call_args, **kwargs)
    if result is None:
        counts["failed_allocations"] += 1
    return result


def record_free(request):
    block_ids = manager.get_block_ids(request.request_id)
    original_free(request)
    free_events.append({
        "request_id": request.request_id,
        "released_block_ids": block_ids,
        "pool_refcounts_after": [b.ref_cnt for b in manager.block_pool.blocks],
    })


manager.get_computed_blocks = record_lookup
manager.allocate_slots = record_allocate
manager.free = record_free
cache_map.pop = record_map_pop
if hasattr(manager, "_find_memoized_prefix") and not args.timing_only:
    original_memo_lookup = manager._find_memoized_prefix

    def record_memo_lookup(*call_args, **kwargs):
        counts["memo_fast_path_calls"] += 1
        return original_memo_lookup(*call_args, **kwargs)

    manager._find_memoized_prefix = record_memo_lookup

peak_memo_entries = 0
peak_memoized_blocks = 0
schedule_ns = 0
replay_start_ns = time.perf_counter_ns()
for step in range(300):
    arrivals = 0
    target = min(args.requests, 4 if step == 0 else 4 + step // 2)
    while next_request < target:
        scheduler.add_request(requests[next_request])
        next_request += 1
        arrivals += 1
    lookup_waiting_ids.clear()
    schedule_start_ns = time.perf_counter_ns()
    output = scheduler.schedule()
    schedule_ns += time.perf_counter_ns() - schedule_start_ns
    memo = getattr(manager, "_prefix_hits", {})
    peak_memo_entries = max(peak_memo_entries, len(memo))
    peak_memoized_blocks = max(
        peak_memoized_blocks, max((len(value) for value in memo.values()), default=0)
    )
    preemptions = {
        r.request_id: r.num_preemptions
        for r in requests[:next_request]
        if r.num_preemptions
    }
    waiting_ids = {r.request_id for r in scheduler.waiting}
    candidates = sorted(waiting_ids & lookup_waiting_ids)
    cancel_id = None
    if candidates and not cancelled:
        cancel_id = candidates[-1]
        scheduler.finish_requests(cancel_id, RequestStatus.FINISHED_ABORTED)
        cancelled.append(cancel_id)

    scheduled_ids = list(output.num_scheduled_tokens)
    trace.append({
        "step": step,
        "arrivals": arrivals,
        "scheduled_tokens": sorted(output.num_scheduled_tokens.items()),
        "new_ids": [r.req_id for r in output.scheduled_new_reqs],
        "cached_ids": list(output.scheduled_cached_reqs.req_ids),
        "cached_new_block_ids": output.scheduled_cached_reqs.new_block_ids,
        "finished_ids": sorted(output.finished_req_ids),
        "cancelled_id": cancel_id,
        "waiting_ids": [r.request_id for r in scheduler.waiting],
        "running_ids": [r.request_id for r in scheduler.running],
        "preemptions": preemptions,
        "request_blocks": {
            r.request_id: manager.get_block_ids(r.request_id)
            for r in requests[:next_request]
            if not r.is_finished() and r.request_id in scheduler.requests
        },
        "pool_refcounts": [b.ref_cnt for b in manager.block_pool.blocks],
        "cached_hash_keys": len(manager.block_pool.cached_block_hash_to_block),
    })

    if scheduled_ids:
        sampled = []
        for rid in scheduled_ids:
            generated[rid] += 1
            sampled.append([50256] if generated[rid] >= 3 else [123])
        model_output = ModelRunnerOutput(
            req_ids=scheduled_ids,
            req_id_to_index={rid: i for i, rid in enumerate(scheduled_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(output, model_output)

    if next_request == args.requests and not scheduler.running and not scheduler.waiting:
        break
else:
    raise AssertionError("scheduler replay did not drain before step limit")

summary = {
    "steps": len(trace),
    "requests": args.requests,
    "memo_enabled": getattr(manager, "_memoize_prefix_hits", False),
    "memo_threshold_blocks": (
        getattr(
            sys.modules[manager.__class__.__module__],
            "MIN_PREFIX_HIT_BLOCKS_FOR_MEMO",
            None,
        )
        if hasattr(manager, "_memoize_prefix_hits")
        else None
    ),
    "prefix_groups": sorted(prefix_groups),
    "completed_or_aborted": sum(r.is_finished() for r in requests),
    "preemptions": sum(r.num_preemptions for r in requests),
    "cancelled": cancelled,
    "counts": counts,
    "peak_memo_entries": peak_memo_entries,
    "peak_memoized_blocks": peak_memoized_blocks,
    "schedule_ns": schedule_ns,
    "profile_segments_ns": segments if args.profile_segments else None,
    "profile_counts": profile_counts if args.profile_segments else None,
    "replay_elapsed_ns": time.perf_counter_ns() - replay_start_ns,
    "free_events": free_events,
    "final_statuses": {r.request_id: r.status.name for r in requests},
    "final_pool_refcounts": [b.ref_cnt for b in manager.block_pool.blocks],
    "trace_sha256": hashlib.sha256(
        json.dumps(trace, sort_keys=True).encode()
    ).hexdigest(),
    "trace": trace,
    "source_root": str(Path(__import__("vllm").__file__).resolve().parent.parent),
    "kv_cache_manager_sha256": hashlib.sha256(
        (Path(__import__("vllm").__file__).resolve().parent / "v1/core/kv_cache_manager.py").read_bytes()
    ).hexdigest(),
}
assert counts["failed_allocations"] > 0
assert counts["cache_key_deletions"] > 0
assert summary["preemptions"] > 0
assert cancelled
if hasattr(manager, "_find_memoized_prefix") and not args.memo_off:
    if args.expect_no_memo:
        assert peak_memo_entries == 0, "short-prefix replay unexpectedly populated memo"
        assert counts["memo_fast_path_calls"] == 0
    else:
        import sys

        manager_module = sys.modules[manager.__class__.__module__]
        min_blocks = getattr(manager_module, "MIN_PREFIX_HIT_BLOCKS_FOR_MEMO", 1)
        assert peak_memo_entries > 0, "memo-enabled replay never populated a prefix memo"
        assert peak_memoized_blocks >= min_blocks, (
            "memoized prefix did not reach the configured block threshold"
        )
        if not args.timing_only:
            assert counts["memo_fast_path_calls"] > 0
assert all(ref == 0 for ref in summary["final_pool_refcounts"])
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k not in ("trace", "free_events")}, indent=2))

