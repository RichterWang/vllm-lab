#!/usr/bin/env python3
"""Analyze same-source split-cache runs and produce shareable H200 figures."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    rows = json.loads((root / "results.json").read_text())
    initialization = json.loads((root / "initialization.json").read_text())
    environment = json.loads((root / "environment.json").read_text())
    main_rows = [row for row in rows if row["k"] == 6]
    if not main_rows or not all(row["correct"] for row in rows):
        raise RuntimeError("no valid K=6 results or an incorrect candidate")
    for row in rows:
        row["device_speedup"] = row["static_device_us"] / row["cached_device_us"]
        row["wall_speedup"] = row["static_wall_us"] / row["cached_wall_us"]
        row["cached_vs_best"] = row["cached_device_us"] / row["best_device_us"]
    worst = min(main_rows, key=lambda row: row["device_speedup"])
    changed = [row for row in main_rows if row["cached_split"] != row["fallback_split"]]
    over_five = [row for row in main_rows if abs(row["device_speedup"] - 1) > 0.05]
    geom_device = geometric_mean([row["device_speedup"] for row in main_rows])
    geom_wall = geometric_mean([row["wall_speedup"] for row in main_rows])
    break_even = []
    for vocab in sorted({row["vocab_size"] for row in main_rows}):
        init = next(item for item in initialization if item["vocab_size"] == vocab)
        for row in (row for row in main_rows if row["vocab_size"] == vocab):
            saved_us = row["static_wall_us"] - row["cached_wall_us"]
            break_even.append({"vocab_size": vocab, "batch_size": row["batch_size"],
                               "saved_wall_us_per_call": saved_us,
                               "calls_to_pay_cold_calibration": (
                                   math.ceil(init["cold_wall_s"] * 1e6 / saved_us)
                                   if saved_us > 0 else None)})
    summary = {
        "source_head": environment["head"],
        "source_dirty": bool(environment["status_porcelain"]),
        "k6_shapes": len(main_rows),
        "device_geometric_speedup": geom_device,
        "wall_geometric_speedup": geom_wall,
        "worst_device_speedup": worst,
        "shapes_changed_choice": changed,
        "shapes_over_five_percent": over_five,
        "break_even": break_even,
        "all_rows": rows,
    }
    (root / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")

    vocabs = sorted({row["vocab_size"] for row in main_rows})
    batches = sorted({row["batch_size"] for row in main_rows})
    fig, axes = plt.subplots(len(vocabs), 1, figsize=(11, 3 * len(vocabs)), sharex=True)
    if len(vocabs) == 1:
        axes = [axes]
    for axis, vocab in zip(axes, vocabs):
        sub = sorted((row for row in main_rows if row["vocab_size"] == vocab),
                     key=lambda row: row["batch_size"])
        x = [row["batch_size"] for row in sub]
        axis.plot(x, [row["static_device_us"] for row in sub], "o-", label="static")
        axis.plot(x, [row["cached_device_us"] for row in sub], "s-", label="cache")
        axis.plot(x, [row["best_device_us"] for row in sub], "^-", label="best scanned")
        axis.set_ylabel(f"V={vocab}\nGPU µs")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xticks(batches)
    axes[-1].set_xlabel("Batch size B (K=6)")
    fig.tight_layout()
    fig.savefig(root / "latency_by_shape.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 4))
    for vocab in vocabs:
        sub = sorted((row for row in main_rows if row["vocab_size"] == vocab),
                     key=lambda row: row["batch_size"])
        axis.plot([row["batch_size"] for row in sub],
                  [row["device_speedup"] for row in sub], "o-", label=f"V={vocab}")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(batches)
    axis.set_xlabel("Batch size B (K=6)")
    axis.set_ylabel("Static / cache GPU time (higher is better)")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "speedup_by_shape.png", dpi=160)
    plt.close(fig)

    lines = [
        "# H200 same-source split-cache evaluation", "",
        f"Source HEAD: `{environment['head']}`; dirty: `{bool(environment['status_porcelain'])}`.",
        f"K=6 shapes: {len(main_rows)}; device geometric speedup: **{geom_device:.3f}×**; "
        f"wall geometric speedup: **{geom_wall:.3f}×**.",
        f"Worst device speedup: **{worst['device_speedup']:.3f}×** at "
        f"V={worst['vocab_size']}, B={worst['batch_size']}.",
        f"Selection changed in {len(changed)} shapes; absolute speed change exceeded "
        f"5% in {len(over_five)} shapes.", "",
        "| V | B | K | Static split | Cache split | Best split | Static GPU µs | Cache GPU µs | Speedup |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda row: (row["vocab_size"], row["batch_size"], row["k"])):
        lines.append(
            f"| {row['vocab_size']} | {row['batch_size']} | {row['k']} | "
            f"{row['fallback_split']} | {row['cached_split']} | {row['best_split']} | "
            f"{row['static_device_us']:.2f} | {row['cached_device_us']:.2f} | "
            f"{row['device_speedup']:.3f}× |"
        )
    lines += ["", "## Initialization cost", "",
              "| V | Cold calibration s | Cache hit s |", "| ---: | ---: | ---: |"]
    for item in initialization:
        lines.append(f"| {item['vocab_size']} | {item['cold_wall_s']:.3f} | {item['hit_wall_s']:.3f} |")
    lines += ["", "`analysis.json` contains the per-shape break-even estimates and full "
              "candidate scan. These call counts assume one shape repeats exclusively; "
              "a deployment needs its actual shape mix.", ""]
    (root / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:7]))


if __name__ == "__main__":
    main()
