#!/usr/bin/env python3
"""Run request-level vLLM serving benchmarks against an already running server."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import httpx
import yaml
from transformers import AutoTokenizer

LAB_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = LAB_ROOT / "src" / "vllm"
DEFAULT_MATRIX = Path(__file__).with_name("matrix.yaml")
TRACKED_METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
    "prompt_tokens_total": "vllm:prompt_tokens_total",
    "generation_tokens_total": "vllm:generation_tokens_total",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()

# get git commit hash
def source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SOURCE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()

# get base url for the vLLM server
def configured_base_url() -> str:
    if override := os.getenv("VLLM_BASE_URL"):
        return override.rstrip("/")
    host = os.getenv("LAB_HOST", "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{os.getenv('LAB_PORT', '8000')}"

# load the test scenarios combinations
def load_matrix(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"Unsupported benchmark matrix: {path}")
    if not payload.get("modes") or not payload.get("scenarios"):
        raise ValueError("The matrix must define at least one mode and scenario")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def parse_metric(payload: str, metric_name: str) -> float | None:
    values: list[float] = []
    for line in payload.splitlines():
        if line.startswith(f"{metric_name}{{") or line.startswith(f"{metric_name} "):
            try:
                values.append(float(line.rsplit(" ", 1)[1]))
            except ValueError:
                continue
    return sum(values) if values and metric_name.endswith("_total") else (
        max(values) if values else None
    )

# coroutine(async/await) realize concurrency via single thread
async def metrics_snapshot(client: httpx.AsyncClient) -> dict[str, float | None]:
    response = await client.get("/metrics")
    response.raise_for_status()
    return {
        label: parse_metric(response.text, metric)
        for label, metric in TRACKED_METRICS.items()
    }

# continuously poll engine metrics until stop event is set, return all samples
async def poll_engine_metrics(
    client: httpx.AsyncClient,
    stop: asyncio.Event,
    started: float,
    interval: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    while not stop.is_set():
        try:
            samples.append(
                {
                    "elapsed_seconds": time.perf_counter() - started,
                    **(await metrics_snapshot(client)),
                }
            )
        except (httpx.HTTPError, ValueError) as error:
            samples.append(
                {
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": str(error),
                }
            )
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass
    return samples


async def gpu_snapshot(gpu_index: int) -> dict[str, float]:
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode().strip() or "nvidia-smi failed")
    fields = [float(item.strip()) for item in stdout.decode().strip().split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"Unexpected nvidia-smi output: {stdout.decode().strip()}")
    return dict(zip(("memory_used_mib", "memory_total_mib", "gpu_utilization_pct", "power_watts"), fields))

# continuously poll GPU metrics until stop event is set, return all samples
async def poll_gpu(
    stop: asyncio.Event,
    started: float,
    interval: float,
    gpu_index: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    while not stop.is_set():
        try:
            sample: dict[str, Any] = await gpu_snapshot(gpu_index)
        except (OSError, RuntimeError) as error:
            sample = {"error": str(error)}
        samples.append({"elapsed_seconds": time.perf_counter() - started, **sample})
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass
    return samples

# generate certain testing prompt（fixed length）
def make_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    """Create a deterministic completion prompt with the requested token count."""
    if target_tokens < 8:
        raise ValueError("input_tokens must be at least 8")
    seed_ids = tokenizer.encode(" benchmark", add_special_tokens=False)
    if len(seed_ids) == 1:
        token_ids = seed_ids * target_tokens
        exact_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        if tokenizer.encode(exact_text, add_special_tokens=False) == token_ids:
            return exact_text, target_tokens

    # Fallback for tokenizers where no stable one-token seed is available.
    unit = " vLLM benchmark request"
    low, high = 1, max(2, target_tokens)
    best_text = unit
    best_count = len(tokenizer.encode(best_text, add_special_tokens=False))
    while low <= high:
        middle = (low + high) // 2
        candidate = (unit * middle).strip()
        count = len(tokenizer.encode(candidate, add_special_tokens=False))
        if abs(count - target_tokens) < abs(best_count - target_tokens):
            best_text, best_count = candidate, count
        if count < target_tokens:
            low = middle + 1
        elif count > target_tokens:
            high = middle - 1
        else:
            return candidate, count
    return best_text, best_count

@dataclass(frozen=True)
class Scenario:
    name: str
    concurrency: int
    input_tokens: int
    output_tokens: int
    warmup_requests: int
    measured_requests: int
    request_timeout_seconds: float
    metrics_interval_seconds: float
    gpu_interval_seconds: float
    temperature: float
    ignore_eos: bool
    seed: int

    @classmethod
    def from_config(cls, defaults: dict[str, Any], raw: dict[str, Any]) -> "Scenario":
        values = {**defaults, **raw}
        values.pop("enabled", None)
        scenario = cls(**values)
        if scenario.concurrency < 1 or scenario.measured_requests < 1:
            raise ValueError(f"Invalid request counts in scenario {scenario.name}")
        if scenario.output_tokens < 2:
            raise ValueError("output_tokens must be at least 2 to calculate TPOT")
        return scenario


async def stream_request(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    scenario: Scenario,
    request_index: int,
    measured_started: float,
) -> dict[str, Any]:
    started = time.perf_counter() # start counter

    # initialize variables to track request state
    first_text_at: float | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    token_events: list[dict[str, Any]] = []
    text_parts: list[str] = []
    error: str | None = None
    status_code: int | None = None

    # send the request to the server and stream the response
    try:
        async with client.stream( # here send http request
            "POST",
            "/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": scenario.temperature,
                "max_tokens": scenario.output_tokens,
                "min_tokens": scenario.output_tokens,
                "ignore_eos": scenario.ignore_eos,
                "seed": scenario.seed + request_index,
                "stream": True, # set streaming response + SSE
                "stream_options": {"include_usage": True},
            },
        ) as response:
            status_code = response.status_code
            response.raise_for_status()

            # read the streaming response line by line and process the data
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue # skip non-data lines
                data = line[6:]
                if data == "[DONE]":
                    break # end of stream
                payload = json.loads(data) # parse JSON payload
                if payload.get("usage"):
                    usage = payload["usage"] # only the last payload contains usage info
                for choice in payload.get("choices", []):
                    text = choice.get("text") or ""
                    if text:
                        now = time.perf_counter()

                        # TTFT
                        if first_text_at is None:
                            first_text_at = now

                        # record text
                        text_parts.append(text)

                        # token event for throughput calculation
                        token_events.append(
                            {
                                "elapsed_seconds": now - started,
                                "text_characters": len(text),
                            }
                        )

                    # check if the request has finished and record the finish reason
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]

    except (httpx.HTTPError, json.JSONDecodeError) as caught:
        error = f"{type(caught).__name__}: {caught}"

    # record the end time of the request
    ended = time.perf_counter()
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    ttft = first_text_at - started if first_text_at is not None else None # TTFT
    e2e = ended - started # E2E
    tpot = (
        (ended - first_text_at) / (completion_tokens - 1)
        if first_text_at is not None and completion_tokens > 1
        else None
    ) # TPOT

    return {
        "request_index": request_index,
        "started_offset_seconds": started - measured_started,
        "http_status": status_code,
        "ok": error is None and first_text_at is not None and completion_tokens > 0,
        "error": error,
        "ttft_seconds": ttft,
        "e2e_seconds": e2e,
        "tpot_seconds": tpot,
        "usage": usage,
        "finish_reason": finish_reason,
        "response_characters": sum(map(len, text_parts)),
        "stream_events": token_events,
    }


# concurrently run multiple requests with a limit on concurrency
async def run_requests(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    scenario: Scenario,
    count: int,
    concurrency: int,
    measured_started: float,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency) # ues semaphore to limit concurrency

    async def limited(index: int) -> dict[str, Any]:
        async with semaphore: # semaphore - 1
            return await stream_request(
                client, model, prompt, scenario, index, measured_started
            ) # index represents the request index

    return await asyncio.gather(*(limited(index) for index in range(count)))


def metric_delta(
    samples: list[dict[str, Any]], key: str
) -> float | None:
    values = [float(row[key]) for row in samples if row.get(key) is not None]
    return values[-1] - values[0] if len(values) >= 2 else None


def peak(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in samples if row.get(key) is not None]
    return max(values) if values else None


# run each scenario and collect metrics, write results to output directory
async def run_scenario(
    mode: str,
    scenario: Scenario,
    output_dir: Path,
    tokenizer: Any,
) -> dict[str, Any]:
    base_url = configured_base_url()
    model = os.getenv("SERVED_MODEL_NAME", "qwen3-0.6b")
    prompt, local_prompt_tokens = make_prompt(tokenizer, scenario.input_tokens) # return actual prompt & nums of tokens

    # set http connection pool limits based on concurrency
    limits = httpx.Limits(
        max_connections=max(16, scenario.concurrency + 4),
        max_keepalive_connections=max(8, scenario.concurrency),
    )
    async with httpx.AsyncClient(
        # init https client
        base_url=base_url,
        timeout=scenario.request_timeout_seconds,
        limits=limits,
    ) as client:
        
        # check server health
        health = await client.get("/health")
        health.raise_for_status()

        # check server model info
        if scenario.warmup_requests:
            warmups = await run_requests(
                client,
                model,
                prompt,
                scenario,
                scenario.warmup_requests,
                min(scenario.concurrency, scenario.warmup_requests),
                time.perf_counter(),
            )
            if not all(row["ok"] for row in warmups):
                raise RuntimeError(f"Warm-up failed for scenario {scenario.name}")

        # conduct formal test
        started_at = utc_now() # time stamp
        initial_engine = await metrics_snapshot(client) # engine metrics before test
        started = time.perf_counter() # counter start
        stop = asyncio.Event() # stop signal

        # start backend profiling tasks
        engine_task = asyncio.create_task(
            poll_engine_metrics(
                client, stop, started, scenario.metrics_interval_seconds
            )
        )
        gpu_task = asyncio.create_task(
            poll_gpu(
                stop,
                started,
                scenario.gpu_interval_seconds,
                int(os.getenv("LAB_GPU_INDEX", "0")),
            )
        )

        ## concurrently run requests and collect metrics until all requests are done
        try:
            requests = await run_requests(
                client,
                model,
                prompt,
                scenario,
                scenario.measured_requests,
                scenario.concurrency,
                started,
            )
        finally:
            stop.set() # signal to stop metrics polling
            engine_samples, gpu_samples = await asyncio.gather(engine_task, gpu_task)

        # calculate wall time and final engine metrics
        wall_seconds = time.perf_counter() - started
        final_engine = await metrics_snapshot(client)

    # padding initial and final engine metrics to the samples for completeness
    engine_samples.insert(0, {"elapsed_seconds": 0.0, **initial_engine})
    engine_samples.append({"elapsed_seconds": wall_seconds, **final_engine})

    # filter out valid requests and compute metrics
    valid = [row for row in requests if row["ok"]]

    # get performance metrics from valid requests
    ttft = [float(row["ttft_seconds"]) for row in valid]
    e2e = [float(row["e2e_seconds"]) for row in valid]
    tpot = [float(row["tpot_seconds"]) for row in valid if row["tpot_seconds"] is not None]

    # compute token counts from valid requests
    prompt_tokens = sum(int(row["usage"]["prompt_tokens"]) for row in valid)
    completion_tokens = sum(int(row["usage"]["completion_tokens"]) for row in valid)

    # write scenario results to output directory
    scenario_dir = output_dir / scenario.name
    write_jsonl(scenario_dir / "requests.jsonl", requests)
    write_jsonl(scenario_dir / "engine_metrics.jsonl", engine_samples)
    write_jsonl(scenario_dir / "gpu_metrics.jsonl", gpu_samples)

    # summarize the scenario results and write to summary.json
    summary = {
        "status": "passed" if len(valid) == scenario.measured_requests else "failed",
        "mode": mode,
        "scenario": asdict(scenario),
        "started_at": started_at,
        "completed_at": utc_now(),
        "wall_seconds": wall_seconds,
        "sample_count": len(valid),
        "failed_request_count": len(requests) - len(valid),
        "local_prompt_tokens": local_prompt_tokens,
        "observed_prompt_tokens_per_request": (
            prompt_tokens / len(valid) if valid else None
        ),
        "metrics": {
            "ttft_seconds": distribution(ttft),
            "tpot_seconds": distribution(tpot),
            "e2e_seconds": distribution(e2e),
            "request_throughput_per_second": len(valid) / wall_seconds,
            "output_token_throughput_per_second": completion_tokens / wall_seconds,
            "total_token_throughput_per_second": (
                prompt_tokens + completion_tokens
            ) / wall_seconds,
        },
        "engine": {
            "sample_count": len(engine_samples),
            "peak_running_requests": peak(engine_samples, "running"),
            "peak_waiting_requests": peak(engine_samples, "waiting"),
            "peak_kv_cache_usage": peak(engine_samples, "kv_cache_usage"),
            "prompt_token_counter_delta": metric_delta(
                engine_samples, "prompt_tokens_total"
            ),
            "generation_token_counter_delta": metric_delta(
                engine_samples, "generation_tokens_total"
            ),
        },
        "gpu": {
            "sample_count": len(gpu_samples),
            "peak_memory_used_mib": peak(gpu_samples, "memory_used_mib"),
            "peak_utilization_pct": peak(gpu_samples, "gpu_utilization_pct"),
            "peak_power_watts": peak(gpu_samples, "power_watts"),
        },
        "p95_warning": (
            None if len(valid) >= 20 else "Fewer than 20 valid requests; p95 is exploratory."
        ),
    }
    atomic_json(scenario_dir / "summary.json", summary)
    return summary


# main benchmark execution function
async def benchmark_mode(args: argparse.Namespace) -> int:
    # get test scenarios from matrix.yaml
    matrix = load_matrix(args.matrix)
    if args.mode not in matrix["modes"]:
        raise ValueError(f"Unknown mode: {args.mode}")
    defaults = matrix.get("defaults", {})

    # load scenarios(fron matrix.yaml)
    scenarios = [
        Scenario.from_config(defaults, raw)
        for raw in matrix["scenarios"]
        if raw.get("enabled", True)
        and (args.scenario is None or raw["name"] == args.scenario)
    ]
    if not scenarios:
        raise ValueError("No enabled scenarios selected")

    # create output directory for the benchmark run
    output_dir = args.run_dir / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    # load tokenizer for prompt generation(tokenization_auto.py)
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["MODEL_ID"],
        revision=os.environ["MODEL_REVISION"],
        trust_remote_code=False,
    )

    manifest = {
        "schema_version": 1,
        "mode": args.mode,
        "mode_config": matrix["modes"][args.mode],
        "source_commit": source_commit(),
        "model_id": os.environ["MODEL_ID"],
        "model_revision": os.environ["MODEL_REVISION"],
        "base_url": configured_base_url(),
        "started_at": utc_now(),
        "matrix": str(args.matrix.resolve()),
    }

    # write in manifest.json(original data)
    atomic_json(output_dir / "manifest.json", manifest)
    summaries: list[dict[str, Any]] = []

    ## for each scenario, run the benchmark and collect summaries
    for scenario in scenarios:
        print(f"[{args.mode}] running {scenario.name}", flush=True)
        summaries.append(
            await run_scenario(args.mode, scenario, output_dir, tokenizer)
        )
    manifest["completed_at"] = utc_now()

    # update manifest status
    manifest["status"] = (
        "passed" if all(item["status"] == "passed" for item in summaries) else "failed"
    )
    atomic_json(output_dir / "manifest.json", manifest)
    return 0 if manifest["status"] == "passed" else 1


# check command line arguments and build parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--mode")
    parser.add_argument("--scenario")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--print-modes", action="store_true")
    parser.add_argument("--print-mode-args")
    return parser


# main entrance
def main() -> None:
    args = build_parser().parse_args() # parse command line arguments
    matrix = load_matrix(args.matrix) # load benchmark matrix from YAML file
    if args.print_modes: # print all available modes
        print("\n".join(matrix["modes"]))
        return
    if args.print_mode_args: # print server arguments for a specific mode
        mode = matrix["modes"].get(args.print_mode_args)
        if mode is None:
            raise SystemExit(f"Unknown mode: {args.print_mode_args}")
        print("\n".join(mode.get("server_args", [])))
        return
    if not args.mode or not args.run_dir: # check params if valid
        raise SystemExit("--mode and --run-dir are required when running benchmarks")
    raise SystemExit(asyncio.run(benchmark_mode(args))) # core execute


if __name__ == "__main__":
    main()
