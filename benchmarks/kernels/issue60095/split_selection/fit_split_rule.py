#!/usr/bin/env python3
"""Score simple split heuristics against offline forced-split measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def choose_splits(
    batch_size: int,
    vocab_size: int,
    sm_count: int,
    target_per_sm: int,
    min_chunk: int,
    max_splits: int,
    min_vocab: int,
) -> int:
    if vocab_size < min_vocab:
        target_per_sm, min_chunk, max_splits = 2, 8192, 64
    limit = min(
        target_per_sm * sm_count // batch_size,
        math.ceil(vocab_size / min_chunk),
        max_splits,
    )
    return 1 << (limit.bit_length() - 1) if limit >= 2 else 1


def read_shapes(paths: list[Path]) -> list[dict]:
    shapes = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                shapes.append(
                    {
                        "sm_count": int(row["sm_count"]),
                        "batch_size": int(row["batch_size"]),
                        "vocab_size": int(row["vocab_size"]),
                        "auto_us": float(row["auto_median_us"]),
                        "best_us": float(row["best_median_us"]),
                        "forced": {
                            int(k): float(v)
                            for k, v in json.loads(row["forced_us"]).items()
                        },
                    }
                )
    return shapes


def score(
    shapes: list[dict], target: int, min_chunk: int, cap: int, min_vocab: int
) -> dict:
    speedups = []
    oracle_gaps = []
    selected = []
    for shape in shapes:
        n = choose_splits(
            shape["batch_size"],
            shape["vocab_size"],
            shape["sm_count"],
            target,
            min_chunk,
            cap,
            min_vocab,
        )
        measured_us = shape["forced"][n]
        selected.append(n)
        speedups.append(shape["auto_us"] / measured_us)
        oracle_gaps.append(measured_us / shape["best_us"])
    return {
        "target_per_sm": target,
        "min_chunk": min_chunk,
        "max_splits": cap,
        "min_vocab": min_vocab,
        "geomean_speedup_vs_pr_auto": statistics.geometric_mean(speedups),
        "worst_speedup_vs_pr_auto": min(speedups),
        "regressions_over_5pct": sum(value < 1 / 1.05 for value in speedups),
        "geomean_gap_to_best": statistics.geometric_mean(oracle_gaps),
        "used_splits": ",".join(map(str, sorted(set(selected)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    shapes = read_shapes(args.summaries)
    candidates = [
        score(shapes, target, min_chunk, cap, min_vocab)
        for target in (2, 4, 8, 16, 32, 64, 128)
        for min_chunk in (1024, 2048, 4096, 8192, 16384)
        for cap in (16, 32, 64)
        for min_vocab in (0, 4096, 8192, 16384, 32768)
    ]
    candidates.sort(
        key=lambda row: (
            row["regressions_over_5pct"],
            -row["worst_speedup_vs_pr_auto"],
            -row["geomean_speedup_vs_pr_auto"],
            row["geomean_gap_to_best"],
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    print(f"Scored {len(candidates)} rules on {len(shapes)} shapes into {args.output}")
    for row in candidates[:10]:
        print(row)


if __name__ == "__main__":
    main()
