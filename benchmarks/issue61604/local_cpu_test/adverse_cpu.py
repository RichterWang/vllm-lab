"""Compare prefix lookup paths without mixing validation into timed loops."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from tests.v1.core.utils import create_requests, create_scheduler

import vllm

SCENARIOS = ("zero_miss", "short_hit", "medium_hit", "long_hit", "early_invalidation")
PREFIX_TOKENS = {
    "zero_miss": 512,
    "short_hit": 64,
    "medium_hit": 512,
    "long_hit": 8192,
    "early_invalidation": 8192,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("base", "memo_on", "memo_off"), required=True
    )
    return parser.parse_args()


def make_request(prefix_tokens: int, request_id: str):
    req = create_requests(
        1,
        num_tokens=prefix_tokens + 1,
        same_prompt=True,
        block_size=16,
        req_ids=[request_id],
    )[0]
    return req


def main():
    args = parse_args()
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    results = []
    for scenario in SCENARIOS:
        prefix = PREFIX_TOKENS[scenario]
        cache_tokens = prefix
        if scenario == "zero_miss":
            cache_tokens = 0
        scheduler = create_scheduler(
            enable_prefix_caching=True,
            num_blocks=max(600, prefix // 16 + 32),
            block_size=16,
            max_num_batched_tokens=16384,
            max_model_len=16384,
        )
        manager = scheduler.kv_cache_manager
        holder = None
        holder_first_hash = None
        invalidated_blocks = []
        if cache_tokens:
            holder = create_requests(
                1,
                num_tokens=cache_tokens + 16,
                same_prompt=True,
                block_size=16,
                req_ids=[f"holder-{scenario}"],
            )[0]
            holder_blocks = manager.allocate_slots(holder, cache_tokens)
            assert holder_blocks is not None
            if scenario == "early_invalidation":
                first_block = holder_blocks.blocks[0][0]
                holder_first_hash = first_block.block_hash
                invalidated_blocks = [
                    block
                    for block in manager.block_pool.blocks
                    if manager.block_pool.cached_block_hash_to_block.contain(
                        holder_first_hash, block.block_id
                    )
                ]
                assert invalidated_blocks

        manager._memoize_prefix_hits = args.mode == "memo_on"

        def prepare_request(
            index: int,
            prefix: int = prefix,
            scenario: str = scenario,
            mode: str = args.mode,
        ):
            request = make_request(prefix, f"{scenario}-{mode}-{index}")
            if scenario == "zero_miss":
                # Give the request real full-block hashes that are absent from
                # the pool; this exercises lookup misses rather than a zero limit.
                request.prompt_token_ids = [100_000 + index] * (prefix + 1)
                request._all_token_ids = request.prompt_token_ids.copy()
                request.block_hashes.clear()
                request.update_block_hashes()
            return request

        expected_hit = 0 if scenario in ("zero_miss", "early_invalidation") else prefix
        observed_hits = set()

        def seed_failed_admission(request):
            blocks, hit, _ = manager.get_computed_blocks(request)
            allocated = manager.allocate_slots(
                request,
                1,
                num_new_computed_tokens=hit,
                new_computed_blocks=blocks,
                reserved_blocks=manager.block_pool.get_num_free_blocks() + 1,
            )
            assert allocated is None
            return hit

        def run_once(
            request,
            admit: bool,
            timed: bool,
            scenario: str = scenario,
            expected_hit: int = expected_hit,
            manager=manager,
            holder_first_hash=holder_first_hash,
            invalidated_blocks=invalidated_blocks,
            observed_hits=observed_hits,
        ):
            if scenario == "early_invalidation":
                seed_failed_admission(request)
                for block in invalidated_blocks:
                    manager.block_pool._maybe_evict_cached_block(block)
                assert all(block.block_hash is None for block in invalidated_blocks)
            start_ns = time.perf_counter_ns() if timed else 0
            blocks, hit, _ = manager.get_computed_blocks(request)
            elapsed_ns = time.perf_counter_ns() - start_ns if timed else 0
            assert hit == expected_hit, (scenario, hit, expected_hit)
            observed_hits.add(hit)
            if admit:
                start_ns = time.perf_counter_ns() if timed else 0
                allocated = manager.allocate_slots(
                    request,
                    1,
                    num_new_computed_tokens=hit,
                    new_computed_blocks=blocks,
                )
                assert allocated is not None
                manager.free(request)
                if timed:
                    elapsed_ns += time.perf_counter_ns() - start_ns
            if scenario == "early_invalidation":
                for block in invalidated_blocks:
                    manager.block_pool._insert_block_hash(
                        holder_first_hash, block, num_tokens=16
                    )
            return elapsed_ns

        # Warm the Python/runtime paths before collecting measurements.
        for i in range(args.warmup):
            request = prepare_request(i)
            run_once(request, admit=True, timed=False)

        lookup_ns = 0
        lookup_admit_ns = 0
        for i in range(args.iterations):
            request = prepare_request(args.warmup + i)
            lookup_ns += run_once(request, admit=False, timed=True)
            del request

            request = prepare_request(args.warmup + args.iterations + i)
            lookup_admit_ns += run_once(request, admit=True, timed=True)
            del request

        # Count first-query and repeated-query probes outside timed loops.
        probes = [0]
        original_probe = manager.block_pool.get_cached_block

        def count_probe(*call_args, _probes=probes, _original=original_probe, **kwargs):
            _probes[0] += 1
            return _original(*call_args, **kwargs)

        manager.block_pool.get_cached_block = count_probe
        retry_request = prepare_request(args.warmup + 2 * args.iterations + 1)
        seeded_hit = seed_failed_admission(retry_request)
        seed_expected_hit = 0 if scenario == "zero_miss" else prefix
        assert seeded_hit == seed_expected_hit
        if scenario == "early_invalidation":
            for block in invalidated_blocks:
                manager.block_pool._maybe_evict_cached_block(block)
        probes[0] = 0
        blocks, retry_hit, _ = manager.get_computed_blocks(retry_request)
        retry_probe_count = probes[0]
        retry_oracle, retry_oracle_hit, _ = manager.coordinator.find_longest_cache_hit(
            retry_request.block_hashes, retry_request.num_tokens - 1
        )
        assert retry_hit == retry_oracle_hit == expected_hit
        assert list(blocks.blocks[0]) == list(retry_oracle[0])
        if scenario == "early_invalidation":
            for block in invalidated_blocks:
                manager.block_pool._insert_block_hash(
                    holder_first_hash, block, num_tokens=16
                )

        oracle_request = prepare_request(args.warmup + 2 * args.iterations + 2)
        probes[0] = 0
        oracle_blocks, oracle_hit, _ = manager.get_computed_blocks(oracle_request)
        baseline_probe_count = probes[0]
        oracle, coordinator_hit, _ = manager.coordinator.find_longest_cache_hit(
            oracle_request.block_hashes, oracle_request.num_tokens - 1
        )
        baseline_expected_hit = 0 if scenario == "zero_miss" else prefix
        assert oracle_hit == coordinator_hit == baseline_expected_hit
        assert list(oracle_blocks.blocks[0]) == list(oracle[0])
        manager.block_pool.get_cached_block = original_probe

        results.append(
            {
                "scenario": scenario,
                "prefix_tokens": prefix,
                "observed_hit_tokens": sorted(observed_hits),
                "iterations": args.iterations,
                "warmup_iterations": args.warmup,
                "lookup_elapsed_ns": lookup_ns,
                "lookup_admit_elapsed_ns": lookup_admit_ns,
                "average_lookup_us": lookup_ns / args.iterations / 1000,
                "average_lookup_admit_us": lookup_admit_ns / args.iterations / 1000,
                "baseline_probe_count": baseline_probe_count,
                "memo_retry_probe_count": retry_probe_count,
                "oracle_block_ids_match": True,
                "memo_enabled": args.mode == "memo_on",
            }
        )
        if holder is not None:
            manager.free(holder)

    source_root = Path(vllm.__file__).resolve().parent.parent
    summary = {
        "mode": args.mode,
        "results": results,
        "source_root": str(source_root),
        "kv_cache_manager_sha256": hashlib.sha256(
            (
                Path(vllm.__file__).resolve().parent / "v1/core/kv_cache_manager.py"
            ).read_bytes()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
