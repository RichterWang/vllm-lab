"""Validate and summarize dynamic arrival/eviction CPU scheduler pairs."""

import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
pairs = []
for i in range(1, 4):
    base = json.loads((a.input / f"base-{i}.json").read_text())
    patch = json.loads((a.input / f"pr-{i}.json").read_text())
    for key in ("trace_sha256", "hit_trace_sha256", "queue_trace_sha256"):
        if base[key] != patch[key]:
            raise SystemExit(f"{key} differs in pair {i}")
    for run in (base, patch):
        if run["evictions"] != 100 or run["arrivals"] != 100:
            raise SystemExit(
                f"arrival/eviction count differs from workload in pair {i}"
            )
        if run["blocked_allocations"] != 500 or not run["admitted_after_release"]:
            raise SystemExit(f"blocked/admission check failed in pair {i}")
    pairs.append(
        {
            "round": i,
            "base_schedule_ms": base["schedule_ns"] / 1e6,
            "pr_schedule_ms": patch["schedule_ns"] / 1e6,
            "base_lookup_ms": base["lookup_ns"] / 1e6,
            "pr_lookup_ms": patch["lookup_ns"] / 1e6,
            "base_probes": base["probes"],
            "pr_probes": patch["probes"],
            "evictions": base["evictions"],
            "arrivals": base["arrivals"],
            "minimum_hit_tokens": base["min_hit_tokens"],
            "maximum_hit_tokens": base["max_hit_tokens"],
        }
    )
summary = {
    "pairs": pairs,
    "base_schedule_median_ms": statistics.median(x["base_schedule_ms"] for x in pairs),
    "pr_schedule_median_ms": statistics.median(x["pr_schedule_ms"] for x in pairs),
    "base_lookup_median_ms": statistics.median(x["base_lookup_ms"] for x in pairs),
    "pr_lookup_median_ms": statistics.median(x["pr_lookup_ms"] for x in pairs),
}
a.output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
