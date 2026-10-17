"""Measure unrelated cache churn invalidating a long waiting-prefix memo."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from tests.v1.core.utils import create_requests, create_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--mode", choices=("no-churn", "b-churn"), required=True)
parser.add_argument(
    "--invalidation",
    choices=("lifecycle", "global"),
    default="lifecycle",
    help="Use block lifecycle validation or emulate the previous global epoch clear.",
)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--pool-blocks", type=int, default=640)
parser.add_argument("--a-hit-blocks", type=int, default=512)
parser.add_argument("--b-blocks", type=int, default=2)
args = parser.parse_args()
assert args.pool_blocks > args.a_hit_blocks + args.b_blocks

block_size = 16
a_hit_tokens = args.a_hit_blocks * block_size
a_prompt_tokens = 2 * a_hit_tokens
scheduler = create_scheduler(
    enable_prefix_caching=True,
    num_blocks=args.pool_blocks,
    block_size=block_size,
    max_num_seqs=4,
    max_num_batched_tokens=128,
    max_model_len=a_prompt_tokens,
)
manager = scheduler.kv_cache_manager

# Keep A's cached prefix pinned in a live producer so B can churn only its own
# unrelated mappings while the waiting request repeatedly fails admission.
a_producer = create_requests(
    1,
    num_tokens=a_prompt_tokens,
    max_tokens=2,
    same_prompt=True,
    block_size=block_size,
    req_ids=["A-producer"],
)[0]
a_waiter = create_requests(
    1,
    num_tokens=a_prompt_tokens,
    max_tokens=2,
    same_prompt=True,
    block_size=block_size,
    req_ids=["A-waiter"],
)[0]
assert manager.allocate_slots(a_producer, a_hit_tokens) is not None
expected_a_block_ids = [
    block.block_id
    for block in manager.get_computed_blocks(a_waiter)[0].blocks[0]
]
assert len(expected_a_block_ids) == args.a_hit_blocks
scheduler.add_request(a_waiter)
b_requests = create_requests(
    args.steps + 1,
    num_tokens=args.b_blocks * block_size,
    max_tokens=2,
    block_size=block_size,
    req_ids=[f"B-churn-{i}" for i in range(args.steps + 1)],
)[1:]

original_allocate = manager.allocate_slots
original_get_computed = manager.get_computed_blocks
original_map_pop = manager.block_pool.cached_block_hash_to_block.pop
cache_map = manager.block_pool.cached_block_hash_to_block
counts = {
    "a_full_lookups": 0,
    "a_fast_path_calls": 0,
    "a_memo_builds": 0,
    "a_global_invalidations": 0,
    "a_invalidated_entries": 0,
    "b_mapping_deletions": 0,
    "a_failed_admissions": 0,
    "a_hit_tokens": [],
}

if hasattr(manager, "_find_memoized_prefix"):
    original_find_memo = manager._find_memoized_prefix
    original_remember = manager._remember_failed_prefix
    original_invalidate = manager._invalidate_prefix_memos

    def count_find_memo(*call_args, **kwargs):
        request = call_args[0]
        if request is a_waiter:
            counts["a_fast_path_calls"] += 1
        return original_find_memo(*call_args, **kwargs)

    def count_remember(request, *call_args, **kwargs):
        before = request in manager._prefix_hits
        result = original_remember(request, *call_args, **kwargs)
        if request is a_waiter and not before and request in manager._prefix_hits:
            counts["a_memo_builds"] += 1
        return result

    def count_invalidate(*call_args, **kwargs):
        before = len(manager._prefix_hits)
        stale = cache_map is not manager._prefix_hit_map
        result = original_invalidate(*call_args, **kwargs)
        if stale and before:
            counts["a_global_invalidations"] += 1
            counts["a_invalidated_entries"] += before
        return result

    manager._find_memoized_prefix = count_find_memo
    manager._remember_failed_prefix = count_remember
    manager._invalidate_prefix_memos = count_invalidate

original_coordinator_lookup = manager.coordinator.find_longest_cache_hit


def count_full_lookup(block_hashes, max_cache_hit_length, *call_args, **kwargs):
    if block_hashes is a_waiter.block_hashes:
        counts["a_full_lookups"] += 1
    return original_coordinator_lookup(
        block_hashes, max_cache_hit_length, *call_args, **kwargs
    )


manager.coordinator.find_longest_cache_hit = count_full_lookup


def count_get_computed(request):
    result = original_get_computed(request)
    if request is a_waiter:
        counts["a_hit_tokens"].append(result[1])
    return result


def force_a_admission_failure(*call_args, **kwargs):
    request = call_args[0]
    if request is a_waiter:
        kwargs["reserved_blocks"] = manager.block_pool.get_num_free_blocks() + 1
    result = original_allocate(*call_args, **kwargs)
    if request is a_waiter and result is None:
        counts["a_failed_admissions"] += 1
    return result


def count_b_pop(key, block_id):
    existed = cache_map.contain(key, block_id)
    result = original_map_pop(key, block_id)
    if existed and result is not None:
        counts["b_mapping_deletions"] += 1
    return result


manager.get_computed_blocks = count_get_computed
manager.allocate_slots = force_a_admission_failure
cache_map.pop = count_b_pop

schedule_ns = 0
schedule_samples_ns = []
previous_deletion_count = counts["b_mapping_deletions"]
for step in range(args.steps):
    if (
        args.invalidation == "global"
        and hasattr(manager, "_prefix_hits")
        and counts["b_mapping_deletions"] != previous_deletion_count
    ):
        if manager._prefix_hits:
            counts["a_global_invalidations"] += 1
            counts["a_invalidated_entries"] += len(manager._prefix_hits)
        manager._prefix_hits.clear()
    previous_deletion_count = counts["b_mapping_deletions"]
    start = time.perf_counter_ns()
    output = scheduler.schedule()
    elapsed = time.perf_counter_ns() - start
    schedule_ns += elapsed
    schedule_samples_ns.append(elapsed)
    assert not output.num_scheduled_tokens, "A waiter unexpectedly admitted"
    assert a_waiter in scheduler.waiting
    assert all(
        manager.block_pool.blocks[block_id].ref_cnt > 0
        and manager.block_pool.blocks[block_id].block_hash is not None
        for block_id in expected_a_block_ids
    )

    if args.mode == "b-churn":
        b_request = b_requests[step]
        assert set(b_request.block_hashes).isdisjoint(a_waiter.block_hashes)
        assert original_allocate(b_request, args.b_blocks * block_size) is not None
        manager.free(b_request)

assert all(hit == a_hit_tokens for hit in counts["a_hit_tokens"])
assert counts["a_failed_admissions"] == args.steps
if args.mode == "b-churn":
    assert counts["b_mapping_deletions"] > 0, "B churn caused no cache deletions"
    if hasattr(manager, "_find_memoized_prefix"):
        if args.invalidation == "global":
            assert counts["a_global_invalidations"] > 0
            assert counts["a_memo_builds"] > 1
        else:
            assert counts["a_global_invalidations"] == 0
            assert counts["a_memo_builds"] == 1
            assert counts["a_full_lookups"] == 1
elif hasattr(manager, "_find_memoized_prefix"):
    assert counts["a_global_invalidations"] == 0
    assert counts["a_memo_builds"] == 1

summary = {
    "mode": args.mode,
    "invalidation": args.invalidation,
    "steps": args.steps,
    "pool_blocks": args.pool_blocks,
    "a_hit_blocks": args.a_hit_blocks,
    "a_hit_tokens": a_hit_tokens,
    "b_blocks_per_request": args.b_blocks,
    "memo_enabled": hasattr(manager, "_find_memoized_prefix"),
    "memo_threshold_blocks": getattr(
        __import__(manager.__class__.__module__, fromlist=["MIN_PREFIX_HIT_BLOCKS_FOR_MEMO"]),
        "MIN_PREFIX_HIT_BLOCKS_FOR_MEMO",
        None,
    ),
    "counts": counts,
    "schedule_ns": schedule_ns,
    "schedule_median_ns": sorted(schedule_samples_ns)[len(schedule_samples_ns) // 2],
    "schedule_p95_ns": sorted(schedule_samples_ns)[int(len(schedule_samples_ns) * 0.95)],
    "a_prefix_block_ids_sha256": hashlib.sha256(
        json.dumps(expected_a_block_ids).encode()
    ).hexdigest(),
    "remaining_waiting": [r.request_id for r in scheduler.waiting],
    "source_root": str(Path(__import__("vllm").__file__).resolve().parent.parent),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
