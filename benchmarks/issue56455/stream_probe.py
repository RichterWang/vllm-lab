#!/usr/bin/env python3
"""Small fixed-load streaming probe for PR #56455."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import httpx


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


async def run(args: argparse.Namespace) -> dict:
    limits = httpx.Limits(max_connections=args.concurrency + 4)
    timeout = httpx.Timeout(180.0, connect=10.0)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()

    async with httpx.AsyncClient(base_url=args.url, timeout=timeout, limits=limits) as client:
        async def request(index: int) -> dict:
            async with semaphore:
                begin = time.perf_counter()
                chunks: list[float] = []
                usage = None
                error = None
                try:
                    prompt = (" vLLM scheduler benchmark" * 64) + f" request {index}"
                    async with client.stream(
                        "POST",
                        "/v1/completions",
                        json={
                            "model": args.model,
                            "prompt": prompt,
                            "max_tokens": 64,
                            "min_tokens": 64,
                            "ignore_eos": True,
                            "temperature": 0,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        },
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            event = json.loads(line[6:])
                            if event.get("usage"):
                                usage = event["usage"]
                            if any(choice.get("text") for choice in event.get("choices", [])):
                                chunks.append(time.perf_counter() - begin)
                except (httpx.HTTPError, json.JSONDecodeError) as exc:
                    error = str(exc)
                end = time.perf_counter()
                return {
                    "index": index,
                    "ok": error is None and bool(chunks) and bool(usage),
                    "error": error,
                    "ttft_seconds": chunks[0] if chunks else None,
                    "e2e_seconds": end - begin,
                    "chunk_gaps_seconds": [b - a for a, b in zip(chunks, chunks[1:])],
                    "stream_chunk_count": len(chunks),
                    "usage": usage,
                }

        rows = await asyncio.gather(*(request(i) for i in range(args.requests)))

    wall = time.perf_counter() - started
    valid = [row for row in rows if row["ok"]]
    gaps = [gap for row in valid for gap in row["chunk_gaps_seconds"]]
    return {
        "arm": args.arm,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "successes": len(valid),
        "wall_seconds": wall,
        "request_throughput": len(valid) / wall,
        "ttft_seconds": stats([row["ttft_seconds"] for row in valid]),
        "chunk_gap_seconds": stats(gaps),
        "chunk_gap_count": len(gaps),
        "request_details": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "request_details"}, indent=2))
    if result["successes"] != args.requests:
        raise SystemExit("Some requests failed")


if __name__ == "__main__":
    main()
