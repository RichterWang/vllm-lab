"""Replay varied prefixes through the real CPU scheduler and sample memo size."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--requests", type=int, default=20)
args = parser.parse_args()
scheduler = create_scheduler(
    enable_prefix_caching=True,
    num_blocks=5,
    block_size=16,
    max_num_seqs=4,
    max_num_batched_tokens=128,
    max_model_len=128,
)
requests = create_requests(
    args.requests,
    num_tokens=48,
    max_tokens=3,
    same_prompt=False,
    block_size=16,
    req_ids=[f"varied-{i}" for i in range(args.requests)],
)
for index, request in enumerate(requests):
    request.prompt_token_ids = [1000] * 32 + [2000 + index % 4] * 16
    request._all_token_ids = request.prompt_token_ids.copy()
    request.block_hashes.clear()
    request.update_block_hashes()

generated = {r.request_id: 0 for r in requests}
next_request = 0
trace = []
cancelled = []
queried_waiting_ids = set()
manager = scheduler.kv_cache_manager
original_lookup = manager.get_computed_blocks


def record_lookup(request):
    if request.num_preemptions > 0 and request.status in (
        RequestStatus.WAITING,
        RequestStatus.PREEMPTED,
    ):
        queried_waiting_ids.add(request.request_id)
    return original_lookup(request)


def memo_snapshot():
    memo = getattr(manager, "_prefix_hits", {})
    values = list(memo.values())
    return {
        "requests": len(memo),
        "blocks": sum(map(len, values)),
        "shallow_bytes": sys.getsizeof(memo) + sum(
            sys.getsizeof(value) + sum(sys.getsizeof(item) for item in value)
            for value in values
        ),
    }


manager.get_computed_blocks = record_lookup
peak = {"requests": 0, "blocks": 0, "shallow_bytes": 0}
previous_cache_keys = set()
for step in range(300):
    arrivals = 0
    target = min(args.requests, 4 if step == 0 else 4 + step // 2)
    while next_request < target:
        scheduler.add_request(requests[next_request])
        next_request += 1
        arrivals += 1
    queried_waiting_ids.clear()
    output = scheduler.schedule()
    preemptions = {
        r.request_id: r.num_preemptions
        for r in requests[:next_request]
        if r.num_preemptions
    }
    waiting_ids = {r.request_id for r in scheduler.waiting}
    cancel_id = None
    candidates = sorted(waiting_ids & queried_waiting_ids)
    if candidates and not cancelled:
        cancel_id = candidates[-1]
        scheduler.finish_requests(cancel_id, RequestStatus.FINISHED_ABORTED)
        cancelled.append(cancel_id)
    memory = memo_snapshot()
    for key in peak:
        peak[key] = max(peak[key], memory[key])
    scheduled_ids = list(output.num_scheduled_tokens)
    current_cache_keys = {
        repr(key)
        for key in manager.block_pool.cached_block_hash_to_block._cache
    }
    evicted_hash_keys = len(previous_cache_keys - current_cache_keys)
    inserted_hash_keys = len(current_cache_keys - previous_cache_keys)
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
        "inserted_hash_keys": inserted_hash_keys,
        "evicted_hash_keys": evicted_hash_keys,
        "memo": memory,
    })
    previous_cache_keys = current_cache_keys
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
    "requests": args.requests,
    "steps": len(trace),
    "completed": sum(r.is_finished() for r in requests),
    "preemptions": sum(r.num_preemptions for r in requests),
    "cancelled": cancelled,
    "prefix_groups": 4,
    "peak_memo": peak,
    "final_memo": memo_snapshot(),
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
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "trace"}, indent=2))

