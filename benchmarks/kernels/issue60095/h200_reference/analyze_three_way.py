#!/usr/bin/env python3
"""Add PR-parent baseline results to the existing H200 candidate/#60097 A/B."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def geo(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(x) for x in values))


def load_baseline(root: Path) -> dict[tuple[int, int, int], dict[str, float | str]]:
    rounds = {}
    for rnd in (1, 2, 3):
        merged = {}
        for k in (1, 6, 32):
            folder = root / f"r{rnd}_baseline_k{k}"
            metadata = json.loads((folder / "result.json").read_text())
            if metadata["commit"] != "d09ff7763747bbe7cd153edd8017479804f47be7":
                raise ValueError(f"Unexpected baseline commit in {folder}")
            rows = csv.DictReader((folder / "result.csv").open(newline=""))
            for row in rows:
                if row["status"] != "ok" or row["correct"].lower() != "true":
                    raise ValueError(f"Baseline correctness/status failure in {folder}: {row}")
                key = (int(row["vocab_size"]), int(row["batch_size"]), int(row["num_logprobs"]))
                merged[key] = {
                    "warm_wall_ms": float(row["warm_wall_ms"]),
                    "cuda_graph_us": float(row["cuda_graph_us"]),
                    # The pre-PR kernel is one unsplit program per row; it does
                    # not expose or select a vocabulary-split count.
                    "splits": int(row["effective_splits"]) if row["effective_splits"] else "unsplit",
                }
        if len(merged) != 57:
            raise ValueError(f"Expected 57 baseline shapes in round {rnd}; found {len(merged)}")
        rounds[rnd] = merged
    return rounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    two_way = list(csv.DictReader((root / "comparison.csv").open(newline="")))
    baseline_rounds = load_baseline(root)
    baseline_median = {}
    for key in baseline_rounds[1]:
        baseline_median[key] = {
            field: statistics.median(baseline_rounds[r][key][field] for r in (1, 2, 3))
            for field in ("cuda_graph_us", "warm_wall_ms")
        }
    output = []
    for old in two_way:
        key = (int(old["vocab_size"]), int(old["batch_size"]), int(old["num_logprobs_K"]))
        base = baseline_median[key]
        candidate_device = float(old["candidate_cuda_graph_us_median"])
        pr_device = float(old["60097_cuda_graph_us_median"])
        candidate_wall = float(old["candidate_wall_ms_median"])
        pr_wall = float(old["60097_wall_ms_median"])
        row = dict(old)
        row.update({
            "baseline_splits": ",".join(str(baseline_rounds[r][key]["splits"]) for r in (1, 2, 3)),
            "baseline_cuda_graph_us_median": base["cuda_graph_us"],
            "baseline_wall_ms_median": base["warm_wall_ms"],
            "baseline_over_candidate_device_speedup": base["cuda_graph_us"] / candidate_device,
            "baseline_over_candidate_wall_speedup": base["warm_wall_ms"] / candidate_wall,
            "baseline_over_60097_device_speedup": base["cuda_graph_us"] / pr_device,
            "baseline_over_60097_wall_speedup": base["warm_wall_ms"] / pr_wall,
        })
        for rnd in (1, 2, 3):
            row[f"r{rnd}_baseline_cuda_graph_us"] = baseline_rounds[rnd][key]["cuda_graph_us"]
            row[f"r{rnd}_baseline_wall_ms"] = baseline_rounds[rnd][key]["warm_wall_ms"]
        output.append(row)

    columns = list(output[0])
    with (root / "comparison_three_way.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output)

    pairings = {
        "baseline_over_candidate_device": "baseline_over_candidate_device_speedup",
        "baseline_over_candidate_wall": "baseline_over_candidate_wall_speedup",
        "baseline_over_60097_device": "baseline_over_60097_device_speedup",
        "baseline_over_60097_wall": "baseline_over_60097_wall_speedup",
        "60097_over_candidate_device": "60097_over_candidate_device_speedup",
        "60097_over_candidate_wall": "60097_over_candidate_wall_speedup",
    }
    summary = {}
    for name, field in pairings.items():
        ratios = [float(row[field]) for row in output]
        summary[name] = {
            "geomean_speedup": geo(ratios),
            "min_speedup": min(ratios),
            "max_speedup": max(ratios),
            "median_shapes_with_gt_5pct_slowdown": sum(x < 1 / 1.05 for x in ratios),
        }
    (root / "analysis_three_way.json").write_text(json.dumps(summary, indent=2) + "\n")

    labels = {
        "baseline_over_candidate_device": "baseline / candidate device",
        "baseline_over_candidate_wall": "baseline / candidate wall",
        "baseline_over_60097_device": "baseline / #60097 device",
        "baseline_over_60097_wall": "baseline / #60097 wall",
        "60097_over_candidate_device": "#60097 / candidate device",
        "60097_over_candidate_wall": "#60097 / candidate wall",
    }
    lines = [
        "# H200 three-way comparison: baseline, #60097, and vllm-60095-pr",
        "",
        "Baseline is the parent of #60097 (`d09ff7763747bbe7cd153edd8017479804f47be7`). #60097 is `0ddd6017793c943a2091a5800f0cbac1daaa3e02`; candidate is `8730e4e1d2d6975cb3c8908fa998921f2161781a` based on `d6f0b7a4166b325b4dda6512e2e06dcb581923ba`. Baseline was measured in three additional rounds with the same benchmark/environment after the candidate/#60097 run. All ratios use reference latency divided by the named implementation; >1 favors the named implementation.",
        "",
        "Each arm passed correctness on 57 shapes over three rounds. Candidate and #60097 each have 3 interleaved rounds; baseline has 3 additional rounds. The source commits are not all on one linear base: baseline→#60097 is a direct parent/child pair, while candidate is on a newer main snapshot. Thus baseline→candidate is a useful full-implementation comparison but does not isolate a single selector change.",
        "",
        "## Overall geometric speedup and worst shape",
        "",
        "| Comparison (speedup >1 favors right side) | Geometric speedup | Minimum | Shapes with >5% right-side slowdown |",
        "|---|---:|---:|---:|",
    ]
    # Ratios have reference/implementation. Translate each label explicitly.
    for key, label in labels.items():
        s = summary[key]
        lines.append(f"| {label} | {s['geomean_speedup']:.3f}× | {s['min_speedup']:.3f}× | {s['median_shapes_with_gt_5pct_slowdown']} |")
    lines += [
        "",
        "## Per-shape table",
        "",
        "The complete `comparison_three_way.csv` includes all V/B/K shapes, per-round times, split selections, and all three pairwise speed ratios.",
        "",
        "## Interpretation",
        "",
        "The baseline→#60097 comparison isolates the PR commit against its direct parent. The candidate currently has a different kernel implementation and is based on newer main; candidate comparisons show which complete implementation is faster on these microbenchmark shapes, not the isolated effect of the static split rule. The test uses synthetic logits and does not establish model-serving TTFT or throughput.",
        "",
    ]
    (root / "report_three_way.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels_x = [
        f"{int(r['vocab_size'])//1024}k/B{r['batch_size']}/K{r['num_logprobs_K']}"
        for r in output
    ]
    x = list(range(len(output)))
    fig, axes = plt.subplots(2, 1, figsize=(19, 11), sharex=True)
    for ax, fields, title in (
        (axes[0], ("baseline_over_candidate_device_speedup", "baseline_over_60097_device_speedup"), "CUDA Graph device speedup"),
        (axes[1], ("baseline_over_candidate_wall_speedup", "baseline_over_60097_wall_speedup"), "Synchronized wall speedup"),
    ):
        ax.plot(x, [float(r[fields[0]]) for r in output], marker=".", linewidth=1, label="baseline / candidate")
        ax.plot(x, [float(r[fields[1]]) for r in output], marker=".", linewidth=1, label="baseline / #60097")
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("Speedup")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels_x, rotation=90, fontsize=7)
    axes[-1].set_xlabel("V / batch / K")
    fig.tight_layout()
    fig.savefig(root / "three_way_speedup.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
