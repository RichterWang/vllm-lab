#!/usr/bin/env python3
"""Aggregate per-scenario benchmark summaries into portable tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def milliseconds(value: float | None) -> float | None:
    return value * 1000 if value is not None else None


def flatten(summary: dict[str, Any]) -> dict[str, Any]:
    scenario = summary["scenario"]
    metrics = summary["metrics"]
    engine = summary["engine"]
    gpu = summary["gpu"]
    return {
        "mode": summary["mode"],
        "scenario": scenario["name"],
        "concurrency": scenario["concurrency"],
        "target_input_tokens": scenario["input_tokens"],
        "observed_input_tokens": summary["observed_prompt_tokens_per_request"],
        "output_tokens": scenario["output_tokens"],
        "samples": summary["sample_count"],
        "status": summary["status"],
        "ttft_p50_ms": milliseconds(metrics["ttft_seconds"]["p50"]),
        "ttft_p95_ms": milliseconds(metrics["ttft_seconds"]["p95"]),
        "tpot_p50_ms": milliseconds(metrics["tpot_seconds"]["p50"]),
        "tpot_p95_ms": milliseconds(metrics["tpot_seconds"]["p95"]),
        "e2e_p50_ms": milliseconds(metrics["e2e_seconds"]["p50"]),
        "e2e_p95_ms": milliseconds(metrics["e2e_seconds"]["p95"]),
        "request_throughput_per_second": metrics["request_throughput_per_second"],
        "output_token_throughput_per_second": metrics[
            "output_token_throughput_per_second"
        ],
        "total_token_throughput_per_second": metrics[
            "total_token_throughput_per_second"
        ],
        "peak_running_requests": engine["peak_running_requests"],
        "peak_waiting_requests": engine["peak_waiting_requests"],
        "peak_kv_cache_usage": engine["peak_kv_cache_usage"],
        "peak_gpu_memory_mib": gpu["peak_memory_used_mib"],
        "peak_gpu_utilization_pct": gpu["peak_utilization_pct"],
    }

# just write into csv
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.run_dir.glob("*/*/summary.json"))
    ]
    if not summaries:
        raise SystemExit(f"No scenario summaries found under {args.run_dir}")
    rows = [flatten(summary) for summary in summaries]

    json_path = args.run_dir / "summary.json"
    json_path.write_text(
        json.dumps({"schema_version": 1, "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = args.run_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failed = [row for row in rows if row["status"] != "passed"]
    print(f"Aggregated {len(rows)} scenarios into {csv_path}")
    if failed:
        raise SystemExit(f"{len(failed)} benchmark scenarios failed")


if __name__ == "__main__":
    main()
