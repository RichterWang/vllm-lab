#!/usr/bin/env python3
"""Sample vLLM queue/KV metrics and H100 telemetry during a serving run."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
)
STOP = False


def stop(_signal: int, _frame: object) -> None:
    global STOP
    STOP = True


def read_metrics(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=3) as response:
        body = response.read().decode("utf-8")
    totals = {name: 0.0 for name in METRICS}
    seen = set()
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split(None, 1)[0].split("{", 1)[0]
        if name in totals:
            totals[name] += float(line.rsplit(" ", 1)[-1])
            seen.add(name)
    return {name: totals[name] for name in seen}


def read_gpu() -> dict[str, float]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, timeout=5).strip()
    utilization, memory, power = (float(part.strip()) for part in output.split(","))
    return {"gpu_utilization_pct": utilization, "gpu_memory_mib": memory,
            "gpu_power_w": power}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        sample_index = 0
        while not STOP:
            sample = {"wall_time": time.time(), "monotonic_time": time.monotonic()}
            try:
                sample["metrics"] = read_metrics(args.url)
            except Exception as error:
                sample["metrics_error"] = repr(error)
            if sample_index % 4 == 0:
                try:
                    sample["gpu"] = read_gpu()
                except Exception as error:
                    sample["gpu_error"] = repr(error)
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            sample_index += 1
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
