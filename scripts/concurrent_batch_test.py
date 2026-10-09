from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

LAB_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = LAB_ROOT / "src" / "vllm"
RESULT_PATH = LAB_ROOT / "results" / "concurrent_batch.json"
PROMPTS = [
    "生成一段编号为 A 的简短中文测试文本。",
    "生成一段编号为 B 的简短中文测试文本。",
    "生成一段编号为 C 的简短中文测试文本。",
    "生成一段编号为 D 的简短中文测试文本。",
]
TRACKED_METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
    "prompt_tokens_total": "vllm:prompt_tokens_total",
    "generation_tokens_total": "vllm:generation_tokens_total",
}

# get data stamp
def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SOURCE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def configured_base_url() -> str:
    if override := os.getenv("VLLM_BASE_URL"):
        return override.rstrip("/")
    host = os.getenv("LAB_HOST", "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{os.getenv('LAB_PORT', '8000')}"


def write_result(result: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(RESULT_PATH)


def parse_metric(payload: str, metric_name: str) -> float | None:
    values: list[float] = []
    for line in payload.splitlines():
        if line.startswith(f"{metric_name}{{") or line.startswith(f"{metric_name} "):
            values.append(float(line.rsplit(" ", 1)[1]))
    return max(values) if values else None


async def metrics_snapshot(client: httpx.AsyncClient) -> dict[str, float | None]:
    response = await client.get("/metrics")
    response.raise_for_status()
    return {
        label: parse_metric(response.text, metric_name)
        for label, metric_name in TRACKED_METRICS.items()
    }


async def poll_metrics(
    client: httpx.AsyncClient,
    stop: asyncio.Event,
    started: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    while not stop.is_set():
        snapshot = await metrics_snapshot(client)
        samples.append(
            {"elapsed_seconds": round(time.perf_counter() - started, 3), **snapshot}
        )
        await asyncio.sleep(0.01)
    samples.append(
        {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            **(await metrics_snapshot(client)),
        }
    )
    return samples


async def request_one(
    client: httpx.AsyncClient,
    model: str,
    index: int,
    prompt: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 128,
            "min_tokens": 96,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "index": index,
        "http_status": response.status_code,
        "prompt": prompt,
        "text": payload["choices"][0]["message"]["content"].strip(),
        "usage": payload.get("usage"),
        "finish_reason": payload["choices"][0].get("finish_reason"),
        "seconds": round(time.perf_counter() - started, 3),
    }


async def run(started_at: str) -> dict[str, Any]:
    base_url = configured_base_url()
    model = os.getenv("SERVED_MODEL_NAME", "qwen3-0.6b")
    started = time.perf_counter()
    limits = httpx.Limits(max_connections=16, max_keepalive_connections=8)
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=120.0,
        limits=limits,
    ) as client:
        stop = asyncio.Event()
        poller = asyncio.create_task(poll_metrics(client, stop, started))
        await asyncio.sleep(0)
        try:
            outputs = await asyncio.gather(
                *(
                    request_one(client, model, index, prompt)
                    for index, prompt in enumerate(PROMPTS)
                )
            )
        finally:
            stop.set()
            samples = await poller
        final_metrics = await metrics_snapshot(client)

    wall_seconds = time.perf_counter() - started
    observed_running = [
        sample["running"] for sample in samples if sample["running"] is not None
    ]
    observed_waiting = [
        sample["waiting"] for sample in samples if sample["waiting"] is not None
    ]
    observed_kv = [
        sample["kv_cache_usage"]
        for sample in samples
        if sample["kv_cache_usage"] is not None
    ]
    peak_running = max(observed_running, default=0.0)
    all_nonempty = len(outputs) == len(PROMPTS) and all(
        item["http_status"] == 200 and item["text"] for item in outputs
    )
    server_overlap_observed = peak_running >= 2.0

    return {
        "status": "passed" if all_nonempty and server_overlap_observed else "failed",
        "acceptance_scope": (
            "API liveness plus observed concurrent engine requests; this is not "
            "a semantic-quality or throughput benchmark"
        ),
        "started_at": started_at,
        "completed_at": utc_now(),
        "source_commit": source_commit(),
        "model_revision": os.environ["MODEL_REVISION"],
        "base_url": base_url,
        "served_model": model,
        "concurrency": len(PROMPTS),
        "wall_seconds": round(wall_seconds, 3),
        "server_side_overlap_observed": server_overlap_observed,
        "metrics_sampling": {
            "sample_count": len(samples),
            "peak_running_requests": peak_running,
            "peak_waiting_requests": max(observed_waiting, default=0.0),
            "peak_kv_cache_usage": max(observed_kv, default=0.0),
            "final": final_metrics,
        },
        "runtime_config": {
            "max_tokens_per_request": 128,
            "min_tokens_per_request": 96,
            "ignore_eos": True,
            "max_num_seqs": int(os.environ["LAB_MAX_NUM_SEQS"]),
            "sampler_backend": "native",
        },
        "requests": outputs,
    }


async def main() -> None:
    started_at = utc_now()
    RESULT_PATH.unlink(missing_ok=True)
    try:
        result = await run(started_at)
    except BaseException as error:
        write_result(
            {
                "status": "failed",
                "acceptance_scope": "concurrent API runtime smoke",
                "started_at": started_at,
                "completed_at": utc_now(),
                "base_url": configured_base_url(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise

    write_result(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit("Concurrent engine overlap test failed")


if __name__ == "__main__":
    asyncio.run(main())
