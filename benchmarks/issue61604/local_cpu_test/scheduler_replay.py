"""Drive the real CPU scheduler with deterministic synthetic model outputs."""

import argparse
import hashlib
import json
from pathlib import Path

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--requests", type=int, default=12)
a = p.parse_args()
s = create_scheduler(
    enable_prefix_caching=True,
    num_blocks=5,
    block_size=16,
    max_num_seqs=4,
    max_num_batched_tokens=128,
    max_model_len=128,
)
requests = create_requests(
    a.requests,
    num_tokens=48,
    max_tokens=3,
    same_prompt=True,
    block_size=16,
    req_ids=[f"req-{i}" for i in range(a.requests)],
)
generated = {r.request_id: 0 for r in requests}
next_request = 0
trace = []
cancelled = []
queried_waiting_ids = set()
original_lookup = s.kv_cache_manager.get_computed_blocks


def record_lookup(request):
    if request.num_preemptions > 0 and request.status in (
        RequestStatus.WAITING,
        RequestStatus.PREEMPTED,
    ):
        queried_waiting_ids.add(request.request_id)
    return original_lookup(request)


s.kv_cache_manager.get_computed_blocks = record_lookup
for step in range(200):
    arrivals_now = 0
    target_arrivals = min(a.requests, 4 if step == 0 else 4 + step // 2)
    while next_request < target_arrivals:
        s.add_request(requests[next_request])
        next_request += 1
        arrivals_now += 1

    queried_waiting_ids.clear()
    out = s.schedule()
    preemptions = {
        r.request_id: r.num_preemptions
        for r in requests[:next_request]
        if r.num_preemptions
    }
    waiting_ids = {r.request_id for r in s.waiting}
    memo_waiting = sorted(waiting_ids & queried_waiting_ids)
    cancel_id = None
    if memo_waiting and not cancelled:
        cancel_id = memo_waiting[-1]
        s.finish_requests(cancel_id, RequestStatus.FINISHED_ABORTED)
        cancelled.append(cancel_id)

    scheduled_ids = list(out.num_scheduled_tokens)
    trace.append(
        {
            "step": step,
            "arrivals": arrivals_now,
            "scheduled_tokens": sorted(out.num_scheduled_tokens.items()),
            "new_request_ids": [x.req_id for x in out.scheduled_new_reqs],
            "cached_request_ids": list(out.scheduled_cached_reqs.req_ids),
            "cached_new_block_ids": out.scheduled_cached_reqs.new_block_ids,
            "finished_ids": sorted(out.finished_req_ids),
            "cancelled_id": cancel_id,
            "waiting_ids": [r.request_id for r in s.waiting],
            "running_ids": [r.request_id for r in s.running],
            "preemptions": preemptions,
            "request_blocks": {
                r.request_id: s.kv_cache_manager.get_block_ids(r.request_id)
                for r in requests[:next_request]
                if not r.is_finished() and r.request_id in s.requests
            },
            "pool_refcounts": [b.ref_cnt for b in s.kv_cache_manager.block_pool.blocks],
        }
    )

    if scheduled_ids:
        sampled = []
        for rid in scheduled_ids:
            generated[rid] += 1
            sampled.append([50256] if generated[rid] >= 3 else [123])
        model_out = ModelRunnerOutput(
            req_ids=scheduled_ids,
            req_id_to_index={rid: i for i, rid in enumerate(scheduled_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        s.update_from_output(out, model_out)

    if next_request == a.requests and not s.running and not s.waiting:
        break
else:
    raise AssertionError("scheduler replay did not drain before step limit")

summary = {
    "steps": len(trace),
    "requests": a.requests,
    "completed": sum(r.is_finished() for r in requests),
    "preemptions": sum(r.num_preemptions for r in requests),
    "cancelled": cancelled,
    "final_statuses": {r.request_id: r.status.name for r in requests},
    "block_pool_refcounts": [b.ref_cnt for b in s.kv_cache_manager.block_pool.blocks],
    "trace_sha256": hashlib.sha256(
        json.dumps(trace, sort_keys=True).encode()
    ).hexdigest(),
    "trace": trace,
    "source_root": str(Path(__import__("vllm").__file__).resolve().parent.parent),
    "kv_cache_manager_sha256": hashlib.sha256(
        (
            Path(__import__("vllm").__file__).resolve().parent
            / "v1/core/kv_cache_manager.py"
        ).read_bytes()
    ).hexdigest(),
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "trace"}, indent=2))
