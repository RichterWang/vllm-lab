"""CPU scheduler replay: long cached prefix held under KV capacity pressure.

Run with PYTHONPATH pointing to the selected vLLM worktree. No model forward.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from tests.v1.core.utils import create_requests, create_scheduler

import vllm

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--prefix-tokens", type=int, default=8192)
p.add_argument("--output", type=Path, required=True)
p.add_argument(
    "--evict-every",
    type=int,
    default=0,
    help="Evict one cached prefix block every N scheduler steps",
)
p.add_argument("--arrival-every", type=int, default=0)
p.add_argument("--memo-off", action="store_true")
p.add_argument("--memo-threshold", type=int)
a = p.parse_args()
block_size = 16
assert a.prefix_tokens % block_size == 0
s = create_scheduler(
    model="Qwen/Qwen3-0.6B",
    max_model_len=32768,
    enable_prefix_caching=True,
    num_blocks=a.prefix_tokens // 16 + 10,
    max_num_batched_tokens=32768,
    block_size=block_size,
)
m = s.kv_cache_manager
if hasattr(m, "_memoize_prefix_hits"):
    m._memoize_prefix_hits = not a.memo_off
    if a.memo_threshold is not None:
        manager_module = sys.modules[m.__class__.__module__]
        manager_module.MIN_PREFIX_HIT_BLOCKS_FOR_MEMO = a.memo_threshold
holder, waiter = create_requests(
    2, num_tokens=a.prefix_tokens + 128, same_prompt=True, req_ids=["holder", "waiter"]
)
assert m.allocate_slots(holder, a.prefix_tokens) is not None
# Hold the last free block to force allocation failure without evicting the prefix.
pressure = m.block_pool.get_new_blocks(m.block_pool.get_num_free_blocks())
s.add_request(waiter)
arrival_requests = []
if a.arrival_every:
    arrival_requests = create_requests(
        (a.steps + a.arrival_every - 1) // a.arrival_every,
        num_tokens=a.prefix_tokens + 128,
        same_prompt=True,
        req_ids=[
            f"arrival-{i}"
            for i in range((a.steps + a.arrival_every - 1) // a.arrival_every)
        ],
    )
# Release the holder while keeping its blocks in prefix cache. This makes them
# evictable without changing the block pool's KV-capacity pressure.
initial_blocks = [
    m.block_pool.get_cached_block(block_hash, [0])[0]
    for block_hash in waiter.block_hashes[: a.prefix_tokens // block_size]
]
m.free(holder)
counts = {"probes": 0, "lookups": 0, "blocked_allocations": 0, "lookup_ns": 0}
original_probe = m.block_pool.get_cached_block
original_lookup = m.get_computed_blocks
original_allocate = m.allocate_slots


def probe(*args, **kwargs):
    counts["probes"] += 1
    return original_probe(*args, **kwargs)


def lookup(*args, **kwargs):
    start = time.perf_counter_ns()
    result = original_lookup(*args, **kwargs)
    counts["last_hit_tokens"] = result[1]
    counts["lookup_ns"] += time.perf_counter_ns() - start
    counts["lookups"] += 1
    return result


def allocate(*args, **kwargs):
    result = original_allocate(*args, **kwargs)
    counts["blocked_allocations"] += result is None
    return result


m.block_pool.get_cached_block = probe
m.get_computed_blocks = lookup
m.allocate_slots = allocate
trace = []
hit_trace = []
queue_trace = []
evictions = 0
arrivals = 0
start = time.perf_counter_ns()
for step in range(a.steps):
    if a.arrival_every and (step + 1) % a.arrival_every == 0:
        s.add_request(arrival_requests[arrivals])
        arrivals += 1
    if a.evict_every and (step + 1) % a.evict_every == 0:
        target_index = len(initial_blocks) - 1 - evictions
        if target_index >= 0:
            evictions += m.block_pool._maybe_evict_cached_block(
                initial_blocks[target_index]
            )
    out = s.schedule()
    assert not out.num_scheduled_tokens
    trace.append(out.num_scheduled_tokens)
    hit_trace.append(counts.get("last_hit_tokens", None))
    queue_trace.append(len(s.waiting))
elapsed = time.perf_counter_ns() - start
measured_counts = counts.copy()
assert counts["blocked_allocations"] == a.steps
# Release capacity and ensure the waiting request can subsequently be admitted.
m.block_pool.free_blocks(pressure)
out = s.schedule()
assert "waiter" in out.num_scheduled_tokens
result = dict(
    steps=a.steps,
    prefix_tokens=a.prefix_tokens,
    memo_enabled=getattr(m, "_memoize_prefix_hits", False),
    memo_threshold_blocks=(
        getattr(
            sys.modules[m.__class__.__module__],
            "MIN_PREFIX_HIT_BLOCKS_FOR_MEMO",
            None,
        )
        if hasattr(m, "_memoize_prefix_hits")
        else None
    ),
    evict_every=a.evict_every,
    evictions=evictions,
    arrivals=arrivals,
    final_waiting_queue=len(s.waiting),
    schedule_ns=elapsed,
    **measured_counts,
    trace_sha256=hashlib.sha256(json.dumps(trace).encode()).hexdigest(),
    hit_trace_sha256=hashlib.sha256(json.dumps(hit_trace).encode()).hexdigest(),
    queue_trace_sha256=hashlib.sha256(json.dumps(queue_trace).encode()).hexdigest(),
    min_hit_tokens=min(hit_trace),
    max_hit_tokens=max(hit_trace),
    admitted_after_release=True,
    source_root=str(Path(vllm.__file__).resolve().parent.parent),
    source_commit=subprocess.check_output(
        [
            "git",
            "-C",
            str(Path(vllm.__file__).resolve().parent.parent),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip(),
    manager_sha256=hashlib.sha256(
        (
            Path(vllm.__file__).resolve().parent / "v1/core/kv_cache_manager.py"
        ).read_bytes()
    ).hexdigest(),
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
