#!/usr/bin/env python3
"""Summarize repeated PR auto/forced token-logprob split sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Directories with result.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_run(path: Path) -> tuple[dict, dict[tuple, dict]]:
    data = json.loads((path / "result.json").read_text(encoding="utf-8"))
    by_case = {}
    for row in data["results"]:
        if row["status"] != "ok" or not row["correct"]:
            raise ValueError(f"Failed result in {path}: {row}")
        key = (
            row["batch_size"],
            row["vocab_size"],
            row["num_logprobs"],
            row["dtype"],
            row["requested_splits"],
        )
        if key in by_case:
            raise ValueError(f"Duplicate case {key} in {path}")
        by_case[key] = row
    return data, by_case


def analyze(runs: list[Path]) -> tuple[dict, list[dict]]:
    loaded = [load_run(path) for path in runs]
    first_meta, first_rows = loaded[0]
    env_fields = ("gpu", "gpu_sm_count", "torch", "cuda_runtime", "triton")
    for path, (meta, rows) in zip(runs[1:], loaded[1:]):
        for field in env_fields:
            if meta.get(field) != first_meta.get(field):
                raise ValueError(f"{field} differs in {path}")
        if rows.keys() != first_rows.keys():
            raise ValueError(f"Shape/split set differs in {path}")

    by_shape = defaultdict(dict)
    for key in first_rows:
        shape = key[:4]
        split = key[4]
        values = [float(rows[key]["cuda_graph_us"]) for _, rows in loaded]
        effective = {rows[key].get("effective_splits") for _, rows in loaded}
        if len(effective) != 1:
            raise ValueError(f"Effective split changed across runs: {key}")
        by_shape[shape][split] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "effective": effective.pop(),
        }

    output = []
    for shape, splits in sorted(by_shape.items()):
        if "auto" not in splits:
            raise ValueError(f"Missing auto result: {shape}")
        forced = {name: info for name, info in splits.items() if name != "auto"}
        best_name, best = min(forced.items(), key=lambda item: item[1]["median"])
        auto = splits["auto"]
        output.append(
            {
                "sm_count": first_meta.get("gpu_sm_count"),
                "batch_size": shape[0],
                "vocab_size": shape[1],
                "num_logprobs": shape[2],
                "dtype": shape[3],
                "auto_splits": auto["effective"],
                "auto_median_us": round(auto["median"], 4),
                "auto_min_us": round(auto["min"], 4),
                "auto_max_us": round(auto["max"], 4),
                "best_splits": best_name,
                "best_median_us": round(best["median"], 4),
                "auto_over_best_x": round(auto["median"] / best["median"], 4),
                "forced_us": json.dumps(
                    {name: round(info["median"], 4) for name, info in forced.items()},
                    sort_keys=True,
                ),
            }
        )
    return {field: first_meta.get(field) for field in env_fields}, output


def main() -> None:
    args = parse_args()
    environment, rows = analyze(args.runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "split_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PR #60097 split sweep",
        "",
        f"- GPU: {environment['gpu']} ({environment['gpu_sm_count']} SMs)",
        f"- PyTorch/CUDA/Triton: {environment['torch']} / "
        f"{environment['cuda_runtime']} / {environment['triton']}",
        f"- Runs: {len(args.runs)}; metric: median CUDA Graph device time per call.",
        "- Best forced split is selected from measured candidates; it is an "
        "oracle for analysis, not a deployable online autotuner.",
        "",
        "| B | V | auto split | auto µs | best split | best µs | auto / best |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['vocab_size']} | {row['auto_splits']} "
            f"| {row['auto_median_us']:.2f} | {row['best_splits']} "
            f"| {row['best_median_us']:.2f} | {row['auto_over_best_x']:.2f}× |"
        )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {"runs": [str(path.resolve()) for path in args.runs], "environment": environment},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Analyzed {len(rows)} shapes into {args.output_dir}")


if __name__ == "__main__":
    main()
