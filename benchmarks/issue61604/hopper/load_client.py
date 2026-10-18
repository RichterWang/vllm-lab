#!/usr/bin/env python3
"""Replay deterministic token-ID workloads against a vLLM completions API."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q + 0.5)))
    return ordered[index]


def ids_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(
        b",".join(str(token_id).encode() for token_id in token_ids)
    ).hexdigest()


def build_trace(args: argparse.Namespace, tokenizer: Any) -> tuple[list[dict], str]:
    rng = random.Random(args.seed)
    vocab_size = len(tokenizer)
    special_ids = set(tokenizer.all_special_ids)

    def token() -> int:
        while True:
            token_id = rng.randrange(16, vocab_size - 16)
            if token_id not in special_ids:
                return token_id

    families = max(1, args.prefix_families if args.scenario == "churn" else 1)
    shared = [[token() for _ in range(args.prefix_tokens)] for _ in range(families)]
    shared_hashes = [ids_hash(prefix) for prefix in shared]
    records = []
    for index in range(args.requests):
        if args.scenario in ("long-shared", "default-capacity"):
            family = 0
            prefix = shared[0]
            suffix_len = args.suffix_tokens
        elif args.scenario == "short-unshared":
            family = index
            prefix = []
            suffix_len = args.prompt_tokens
        else:
            # Keep the first family as the long-lived waiter cohort, then rotate
            # through unrelated prefixes so cache eviction occurs naturally.
            family = 0 if index < args.churn_a_requests else 1 + (
                (index - args.churn_a_requests) % max(1, families - 1)
            )
            prefix = shared[family]
            suffix_len = args.suffix_tokens

        prompt_ids = prefix + [token() for _ in range(suffix_len)]
        records.append(
            {
                "request_index": index,
                "family": family,
                "prompt_ids": prompt_ids,
                "prompt_sha256": ids_hash(prompt_ids),
                "arrival_offset_s": index * args.arrival_interval_ms / 1000.0,
                "seed": args.seed + index,
            }
        )

    return records, hashlib.sha256("".join(shared_hashes).encode()).hexdigest()


async def send_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    base_url: str,
    model: str,
    item: dict,
    trace_start: float,
    output_tokens: int,
) -> dict:
    await asyncio.sleep(max(0.0, trace_start + item["arrival_offset_s"] - time.perf_counter()))
    async with semaphore:
        started = time.perf_counter()
        result = {
            "request_index": item["request_index"],
            "family": item["family"],
            "prompt_tokens_requested": len(item["prompt_ids"]),
            "prompt_sha256": item["prompt_sha256"],
            "scheduled_offset_s": item["arrival_offset_s"],
            "client_start_offset_s": started - trace_start,
            "seed": item["seed"],
            "status": "error",
            "error": None,
            "output_token_ids": [],
            "stream_chunks": 0,
            "multitoken_chunks": 0,
            "max_chunk_token_count": 0,
            "text": "",
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "ttft_ms": None,
            "chunk_gaps_ms": [],
            "e2el_ms": None,
        }
        payload = {
            "model": model,
            "prompt": item["prompt_ids"],
            "max_tokens": output_tokens,
            "min_tokens": output_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "seed": item["seed"],
            "stream": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True},
        }
        last_chunk_at = None
        try:
            async with session.post(
                f"{base_url.rstrip('/')}/v1/completions", json=payload
            ) as response:
                if response.status != 200:
                    result["error"] = f"HTTP {response.status}: {await response.text()}"
                else:
                    async for raw_line in response.content:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        event = json.loads(data)
                        if event.get("error"):
                            result["error"] = json.dumps(event["error"], sort_keys=True)
                            continue
                        now = time.perf_counter()
                        usage = event.get("usage")
                        if usage:
                            result["prompt_tokens"] = usage.get("prompt_tokens")
                            result["completion_tokens"] = usage.get("completion_tokens")
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        chunk_ids = choice.get("token_ids") or []
                        if chunk_ids:
                            result["stream_chunks"] += 1
                            result["multitoken_chunks"] += int(len(chunk_ids) > 1)
                            result["max_chunk_token_count"] = max(
                                result["max_chunk_token_count"], len(chunk_ids)
                            )
                            if result["ttft_ms"] is None:
                                result["ttft_ms"] = (now - started) * 1000.0
                            elif last_chunk_at is not None:
                                result["chunk_gaps_ms"].append(
                                    (now - last_chunk_at) * 1000.0
                                )
                            last_chunk_at = now
                            result["output_token_ids"].extend(chunk_ids)
                        result["text"] += choice.get("text") or ""
                        if choice.get("finish_reason") is not None:
                            result["finish_reason"] = choice["finish_reason"]
                    if result["error"] is None:
                        if len(result["output_token_ids"]) == output_tokens:
                            result["status"] = "ok"
                        else:
                            result["error"] = (
                                f"expected {output_tokens} output token IDs, "
                                f"received {len(result['output_token_ids'])}"
                            )
        except Exception as exc:  # Preserve individual failures for the report.
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["e2el_ms"] = (time.perf_counter() - started) * 1000.0
        result["output_tokens_observed"] = len(result["output_token_ids"])
        result["output_ids_sha256"] = ids_hash(result["output_token_ids"])
        return result


async def run(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True, trust_remote_code=True
    )
    trace, shared_prefix_sha256 = build_trace(args, tokenizer)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s, connect=30)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=60)
    semaphore = asyncio.Semaphore(args.concurrency)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if args.warmup:
            if args.scenario == "short-unshared":
                warmup_args = argparse.Namespace(**vars(args))
                warmup_args.requests = 1
                warmup_args.seed = args.seed + 1
                warmup_trace, _ = build_trace(warmup_args, tokenizer)
                warmup_item = warmup_trace[0]
            else:
                warmup_item = trace[0]
            warmup = await send_one(
                session,
                semaphore,
                args.base_url,
                args.model,
                warmup_item,
                time.perf_counter(),
                min(8, args.output_tokens),
            )
            (out / "warmup.json").write_text(json.dumps(warmup, indent=2) + "\n")
            if warmup["status"] != "ok":
                raise RuntimeError(f"warmup request failed: {warmup['error']}")

        trace_start = time.perf_counter() + 1.0
        tasks = [
            asyncio.create_task(
                send_one(
                    session,
                    semaphore,
                    args.base_url,
                    args.model,
                    item,
                    trace_start,
                    args.output_tokens,
                )
            )
            for item in trace
        ]
        results = await asyncio.gather(*tasks)
        finished = time.perf_counter()
    first_start = min((r["client_start_offset_s"] for r in results), default=0.0)
    elapsed_s = max(
        0.001,
        max((r["client_start_offset_s"] + (r["e2el_ms"] or 0) / 1000 for r in results), default=0)
        - first_start,
    )
    successful = [r for r in results if r["status"] == "ok"]
    ttf = [r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]
    e2el = [r["e2el_ms"] for r in successful if r["e2el_ms"] is not None]
    chunk_gaps = [gap for r in successful for gap in r["chunk_gaps_ms"]]
    completion_tokens = sum(r["output_tokens_observed"] for r in successful)
    summary = {
        "run_id": args.run_id,
        "implementation": args.implementation,
        "scenario": args.scenario,
        "mode": args.mode,
        "seed": args.seed,
        "requests_requested": len(results),
        "requests_succeeded": len(successful),
        "requests_failed": len(results) - len(successful),
        "completion_tokens_observed": completion_tokens,
        "elapsed_s": elapsed_s,
        "requests_per_s": len(successful) / elapsed_s,
        "output_tokens_per_s": completion_tokens / elapsed_s,
        "ttft_ms_p50": percentile(ttf, 0.50),
        "ttft_ms_p95": percentile(ttf, 0.95),
        "ttft_ms_p99": percentile(ttf, 0.99),
        "e2el_ms_p50": percentile(e2el, 0.50),
        "e2el_ms_p95": percentile(e2el, 0.95),
        "e2el_ms_p99": percentile(e2el, 0.99),
        "inter_chunk_gap_ms_p50": percentile(chunk_gaps, 0.50),
        "inter_chunk_gap_ms_p95": percentile(chunk_gaps, 0.95),
        "inter_chunk_gap_ms_p99": percentile(chunk_gaps, 0.99),
        "inter_chunk_gap_samples": len(chunk_gaps),
        "stream_chunks": sum(r["stream_chunks"] for r in successful),
        "multitoken_chunks": sum(r["multitoken_chunks"] for r in successful),
        "max_chunk_token_count": max(
            (r["max_chunk_token_count"] for r in successful), default=0
        ),
        "token_level_itl_valid": bool(successful) and all(
            r["max_chunk_token_count"] <= 1
            and r["stream_chunks"] == r["output_tokens_observed"]
            for r in successful
        ),
        "shared_prefix_sha256": shared_prefix_sha256,
        "wall_s_including_client_setup": finished - trace_start,
        "limitations": [
            "ITL is measured as inter-stream-chunk gap; verify stream_interval=1 and one token per chunk before labeling it token-level ITL.",
            "Requests share a pre-generated token-ID trace; with nonzero arrival intervals the same schedule is replayed in each arm.",
        ],
    }
    stem = f"{args.run_id}-{args.implementation}-{args.scenario}-{args.mode}-seed{args.seed}"
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / f"{stem}-requests.jsonl").open("w") as f:
        for result in results:
            f.write(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--implementation", choices=("base", "patch"), required=True)
    parser.add_argument("--scenario", choices=("long-shared", "default-capacity", "short-unshared", "churn"), required=True)
    parser.add_argument("--mode", choices=("sync", "async-default"), required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--prefix-tokens", type=int, default=16384)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--prefix-families", type=int, default=8)
    parser.add_argument("--churn-a-requests", type=int, default=32)
    parser.add_argument("--arrival-interval-ms", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=61604)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
