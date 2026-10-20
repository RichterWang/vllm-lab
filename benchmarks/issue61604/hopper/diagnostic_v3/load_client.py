#!/usr/bin/env python3
"""Replay deterministic token-ID workloads against a vLLM completions API."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import aiohttp


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

    if args._cancel_request is not None:
        for item in records:
            if int(item["request_index"]) == args._cancel_request:
                item["cancel_after_tokens"] = args.cancel_after_tokens

    return records, hashlib.sha256("".join(shared_hashes).encode()).hexdigest()


def read_trace(path: str, args: argparse.Namespace) -> tuple[list[dict], str]:
    """Load an already materialized trace without importing Transformers."""
    trace = []
    with gzip.open(path, "rt", encoding="utf-8") as trace_file:
        for line in trace_file:
            if line.strip():
                trace.append(json.loads(line))
                if len(trace) == args.requests:
                    break
    if len(trace) < args.requests:
        raise ValueError(
            f"trace has {len(trace)} requests, expected at least {args.requests}"
        )
    trace = trace[: args.requests]
    if args._cancel_request is not None:
        for item in trace:
            if int(item["request_index"]) == args._cancel_request:
                item["cancel_after_tokens"] = args.cancel_after_tokens
    family_prefixes: dict[int, list[int]] = {}
    for item in trace:
        family = int(item["family"])
        family_prefixes.setdefault(
            family, item["prompt_ids"][: args.prefix_tokens]
        )
    shared_hashes = [ids_hash(family_prefixes[key]) for key in sorted(family_prefixes)]
    shared_prefix_sha256 = hashlib.sha256("".join(shared_hashes).encode()).hexdigest()
    return trace, shared_prefix_sha256


async def send_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    base_url: str,
    model: str,
    item: dict,
    trace_start: float,
    output_tokens: int,
    logprobs: int | None,
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
            "token_logprobs": [],
            "cancelled_after_tokens": None,
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
        if logprobs is not None:
            payload["logprobs"] = logprobs
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
                        logprobs = choice.get("logprobs") or {}
                        if logprobs:
                            token_texts = logprobs.get("tokens") or []
                            token_scores = logprobs.get("token_logprobs") or []
                            top_scores = logprobs.get("top_logprobs") or []
                            for index, token_id in enumerate(chunk_ids):
                                result["token_logprobs"].append(
                                    {
                                        "token_id": token_id,
                                        "token": token_texts[index]
                                        if index < len(token_texts)
                                        else None,
                                        "logprob": token_scores[index]
                                        if index < len(token_scores)
                                        else None,
                                        "top_logprobs": top_scores[index]
                                        if index < len(top_scores)
                                        else None,
                                    }
                                )
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
                            if (
                                item.get("cancel_after_tokens") is not None
                                and len(result["output_token_ids"])
                                >= item["cancel_after_tokens"]
                            ):
                                result["status"] = "cancelled"
                                result["cancelled_after_tokens"] = len(
                                    result["output_token_ids"]
                                )
                                response.close()
                                break
                        result["text"] += choice.get("text") or ""
                        if choice.get("finish_reason") is not None:
                            result["finish_reason"] = choice["finish_reason"]
                    if result["error"] is None and result["status"] != "cancelled":
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
    if args.prepare_trace:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, local_files_only=True, trust_remote_code=True
        )
        trace, shared_prefix_sha256 = build_trace(args, tokenizer)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        trace_path = out / "input-trace.jsonl.gz"
        with gzip.open(trace_path, "wt", encoding="utf-8") as trace_file:
            for item in trace:
                trace_file.write(json.dumps(item, separators=(",", ":")) + "\n")
        (out / "trace-manifest.json").write_text(
            json.dumps(
                {
                    "requests": len(trace),
                    "shared_prefix_sha256": shared_prefix_sha256,
                    "seed": args.seed,
                    "trace": str(trace_path),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(json.dumps({"prepared_trace": str(trace_path), "requests": len(trace)}))
        return

    if args.input_trace:
        trace, shared_prefix_sha256 = read_trace(args.input_trace, args)
    else:
        # Importing Transformers can be slow on the Pod's FUSE-mounted
        # site-packages. Reusing the first arm's saved trace avoids that cost
        # and guarantees the paired arm receives identical prompt token IDs.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, local_files_only=True, trust_remote_code=True
        )
        trace, shared_prefix_sha256 = build_trace(args, tokenizer)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s, connect=30)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=60)
    semaphore = asyncio.Semaphore(args.concurrency)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "input-trace.jsonl.gz", "wt", encoding="utf-8") as trace_file:
        for item in trace:
            trace_file.write(json.dumps(item, separators=(",", ":")) + "\n")
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if args.warmup:
            if args.scenario == "short-unshared":
                warmup_item = dict(trace[0])
                warmup_item["prompt_ids"] = [
                    (args.seed + i * 7919) % 150000
                    for i in range(args.prompt_tokens)
                ]
                warmup_item["prompt_sha256"] = ids_hash(warmup_item["prompt_ids"])
            else:
                warmup_item = trace[0]
            (out / "warmup-input.json").write_text(
                json.dumps(warmup_item, separators=(",", ":")) + "\n"
            )
            warmup = await send_one(
                session,
                semaphore,
                args.base_url,
                args.model,
                warmup_item,
                time.perf_counter(),
                min(8, args.output_tokens),
                args.logprobs,
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
                    args.logprobs,
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
    cancelled = [r for r in results if r["status"] == "cancelled"]
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
        "requests_cancelled": len(cancelled),
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
    parser.add_argument("--input-trace")
    parser.add_argument("--prepare-trace", action="store_true")
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
    parser.add_argument("--logprobs", type=int)
    parser.add_argument("--cancel-index", type=int)
    parser.add_argument("--cancel-after-tokens", type=int)
    args = parser.parse_args()
    if args.cancel_index is not None:
        if args.cancel_after_tokens is None or args.cancel_after_tokens < 1:
            parser.error("--cancel-index requires --cancel-after-tokens >= 1")
        args._cancel_request = args.cancel_index
    else:
        args._cancel_request = None
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
