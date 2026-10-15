#!/usr/bin/env python3
"""Compare repeated original-PR and candidate auto-split benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", nargs="+", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_run(path: Path) -> tuple[dict, dict[tuple, dict]]:
    metadata = json.loads((path / "result.json").read_text(encoding="utf-8"))
    rows = {}
    for row in metadata["results"]:
        if row["requested_splits"] != "auto":
            raise ValueError(f"Expected auto-only run in {path}")
        if row["status"] != "ok" or not row["correct"]:
            raise ValueError(f"Failed result in {path}: {row}")
        key = (
            row["batch_size"],
            row["vocab_size"],
            row["num_logprobs"],
            row["dtype"],
        )
        if key in rows:
            raise ValueError(f"Duplicate case {key} in {path}")
        rows[key] = row
    return metadata, rows


def main() -> None:
    args = parse_args()
    if len(args.reference) != len(args.candidate):
        raise ValueError("Reference and candidate must have the same number of runs")
    reference = [load_run(path) for path in args.reference]
    candidate = [load_run(path) for path in args.candidate]
    all_runs = reference + candidate
    environment = all_runs[0][0]
    env_keys = (
        "gpu",
        "gpu_sm_count",
        "torch",
        "cuda_runtime",
        "triton",
        "benchmark_sha256",
    )
    expected_cases = all_runs[0][1].keys()
    for metadata, rows in all_runs[1:]:
        if rows.keys() != expected_cases:
            raise ValueError("A/B runs do not cover the same shapes")
        if any(metadata.get(key) != environment.get(key) for key in env_keys):
            raise ValueError("A/B runs use different hardware or runtimes")
        if metadata["settings"]["binary_root"] != environment["settings"][
            "binary_root"
        ]:
            raise ValueError("A/B runs use different compiled-extension roots")
    for label, runs in (("reference", reference), ("candidate", candidate)):
        fingerprints = {metadata.get("source_module_sha256") for metadata, _ in runs}
        if len(fingerprints) != 1:
            raise ValueError(f"{label} source file changed between rounds")
    if reference[0][0].get("source_module_sha256") == candidate[0][0].get(
        "source_module_sha256"
    ):
        raise ValueError("A/B arms loaded identical logprob source files")
    if any(metadata.get("source_dirty") for metadata, _ in reference):
        raise ValueError("The original PR reference worktree is dirty")

    rows = []
    speedups = []
    wall_speedups = []
    for key in sorted(expected_cases):
        ref_values = [float(data[key]["cuda_graph_us"]) for _, data in reference]
        cand_values = [float(data[key]["cuda_graph_us"]) for _, data in candidate]
        ref_median = statistics.median(ref_values)
        cand_median = statistics.median(cand_values)
        speedup = ref_median / cand_median
        speedups.append(speedup)
        ref_wall = statistics.median(
            float(data[key]["warm_wall_ms"]) * 1000 for _, data in reference
        )
        cand_wall = statistics.median(
            float(data[key]["warm_wall_ms"]) * 1000 for _, data in candidate
        )
        wall_speedup = ref_wall / cand_wall
        wall_speedups.append(wall_speedup)
        rows.append(
            {
                "batch_size": key[0],
                "vocab_size": key[1],
                "num_logprobs": key[2],
                "dtype": key[3],
                "reference_splits": reference[0][1][key].get("effective_splits"),
                "candidate_splits": candidate[0][1][key].get("effective_splits"),
                "reference_median_us": round(ref_median, 4),
                "candidate_median_us": round(cand_median, 4),
                "speedup": round(speedup, 4),
                "reference_wall_median_us": round(ref_wall, 4),
                "candidate_wall_median_us": round(cand_wall, 4),
                "wall_speedup": round(wall_speedup, 4),
                "reference_min_us": round(min(ref_values), 4),
                "reference_max_us": round(max(ref_values), 4),
                "candidate_min_us": round(min(cand_values), 4),
                "candidate_max_us": round(max(cand_values), 4),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "reference_runs": [str(path.resolve()) for path in args.reference],
        "candidate_runs": [str(path.resolve()) for path in args.candidate],
        "reference_source_sha256": reference[0][0].get("source_module_sha256"),
        "candidate_source_sha256": candidate[0][0].get("source_module_sha256"),
        "reference_commit": reference[0][0]["commit"],
        "candidate_commit": candidate[0][0]["commit"],
        "reference_implementation": reference[0][1][next(iter(expected_cases))][
            "implementation"
        ],
        "gpu": environment["gpu"],
        "sm_count": environment["gpu_sm_count"],
        "shapes": len(rows),
        "rounds": len(reference),
        "geomean_speedup": statistics.geometric_mean(speedups),
        "worst_speedup": min(speedups),
        "regressions_over_5pct": sum(value < 1 / 1.05 for value in speedups),
        "wall_geomean_speedup": statistics.geometric_mean(wall_speedups),
        "wall_worst_speedup": min(wall_speedups),
        "wall_regressions_over_5pct": sum(
            value < 1 / 1.05 for value in wall_speedups
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Reference auto vs candidate auto",
        "",
        f"Reference: {summary['reference_implementation']} at "
        f"`{summary['reference_commit']}`; candidate at "
        f"`{summary['candidate_commit']}`.",
        f"GPU: {summary['gpu']} ({summary['sm_count']} SMs).",
        f"{summary['rounds']} rounds, {summary['shapes']} shapes; "
        f"geometric mean speedup {summary['geomean_speedup']:.2f}×; "
        f"worst {summary['worst_speedup']:.2f}×; "
        f">5% regressions {summary['regressions_over_5pct']}.",
        "Metric: CUDA Graph device time per call, lower is better.",
        "Warm wall time includes Python dispatch, launch, and final GPU sync; "
        "each run uses 10 calls, so treat small differences cautiously.",
        f"Wall geomean speedup {summary['wall_geomean_speedup']:.2f}×; "
        f"worst {summary['wall_worst_speedup']:.2f}×; "
        f">5% wall regressions {summary['wall_regressions_over_5pct']}.",
        "",
        "| B | V | reference split | candidate split | reference device µs | candidate device µs | device speedup | reference wall µs | candidate wall µs | wall speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['vocab_size']} | "
            f"{row['reference_splits']} | {row['candidate_splits']} | "
            f"{row['reference_median_us']:.2f} | "
            f"{row['candidate_median_us']:.2f} | {row['speedup']:.2f}× | "
            f"{row['reference_wall_median_us']:.2f} | "
            f"{row['candidate_wall_median_us']:.2f} | "
            f"{row['wall_speedup']:.2f}× |"
        )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
