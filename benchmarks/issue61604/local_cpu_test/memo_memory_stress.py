"""Measure active waiting-request memo growth, eviction invalidation, and cleanup."""
import argparse
import gc
import json
import sys
from pathlib import Path

from tests.v1.core.utils import create_requests, create_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--requests", type=int, default=100)
args = parser.parse_args()
scheduler = create_scheduler(
    enable_prefix_caching=True,
    num_blocks=32,
    block_size=16,
    max_num_seqs=8,
    max_num_batched_tokens=512,
    max_model_len=512,
)
manager = scheduler.kv_cache_manager
holder = create_requests(
    1, num_tokens=129, same_prompt=True, block_size=16, req_ids=["holder"]
)[0]
assert manager.allocate_slots(holder, 64) is not None
manager.free(holder)
waiters = create_requests(
    args.requests,
    num_tokens=129,
    same_prompt=True,
    block_size=16,
    req_ids=[f"waiting-{i}" for i in range(args.requests)],
)
memo = getattr(manager, "_prefix_hits", {})
before_refs = [b.ref_cnt for b in manager.block_pool.blocks]
failed = 0
for request in waiters:
    blocks, hit, _ = manager.get_computed_blocks(request)
    assert hit == 64
    assert manager.allocate_slots(
        request,
        1,
        num_new_computed_tokens=hit,
        new_computed_blocks=blocks,
        reserved_blocks=manager.block_pool.get_num_free_blocks() + 1,
    ) is None
    failed += 1
active_snapshot = {
    "request_entries": len(memo),
    "remembered_blocks": sum(map(len, memo.values())),
    "shallow_memo_bytes": sys.getsizeof(memo) + sum(
        sys.getsizeof(items)
        + sum(sys.getsizeof(item) for item in items)
        for items in memo.values()
    ),
    "failed_admissions": failed,
    "pool_refcounts_unchanged": [b.ref_cnt for b in manager.block_pool.blocks] == before_refs,
}
if memo:
    victim_request = waiters[0]
    victim_block = next(iter(memo[victim_request]))[1]
    manager.block_pool._maybe_evict_cached_block(victim_block)
    assert victim_block.block_hash is None
    blocks, hit, _ = manager.get_computed_blocks(victim_request)
    after_eviction_snapshot = {
        "memo_entries_after_mapping_deletion": len(memo),
        "fresh_lookup_hit_tokens": hit,
        "same_as_manager_oracle": list(blocks.blocks[0])
        == manager.coordinator.find_longest_cache_hit(
            victim_request.block_hashes, victim_request.num_tokens - 1
        )[0][0],
    }
else:
    after_eviction_snapshot = {"skipped": "base implementation has no memo"}
for request in waiters:
    manager.free(request)
cancel_cleanup = {"memo_entries_after_free": len(memo)}
assert cancel_cleanup["memo_entries_after_free"] == 0

weak_waiters = create_requests(
    args.requests,
    num_tokens=129,
    same_prompt=True,
    block_size=16,
    req_ids=[f"weak-{i}" for i in range(args.requests)],
)
replacement_holder = create_requests(
    1, num_tokens=129, same_prompt=True, block_size=16, req_ids=["replacement"]
)[0]
assert manager.allocate_slots(replacement_holder, 64) is not None
manager.free(replacement_holder)
for request in weak_waiters:
    blocks, hit, _ = manager.get_computed_blocks(request)
    assert manager.allocate_slots(
        request,
        1,
        num_new_computed_tokens=hit,
        new_computed_blocks=blocks,
        reserved_blocks=manager.block_pool.get_num_free_blocks() + 1,
    ) is None
weak_entries_before_gc = len(memo)
del request, weak_waiters, waiters
gc.collect()
weak_cleanup = {
    "entries_before_gc": weak_entries_before_gc,
    "entries_after_gc": len(memo),
}
assert weak_cleanup["entries_after_gc"] == 0
result = {
    "requests": args.requests,
    "active_waiters": active_snapshot,
    "mapping_eviction": after_eviction_snapshot,
    "free_cleanup": cancel_cleanup,
    "weak_cleanup": weak_cleanup,
    "source_root": str(Path(__import__("vllm").__file__).resolve().parent.parent),
    "memo_supported": hasattr(manager, "_prefix_hits"),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

