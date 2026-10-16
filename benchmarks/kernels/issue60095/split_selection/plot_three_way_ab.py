#!/usr/bin/env python3
"""Plot repeated baseline, PR, and candidate token-logprob measurements."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-run-dir", required=True, type=Path)
    parser.add_argument("--baseline-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def comparison(run_dir: Path) -> dict[int, dict]:
    path = run_dir / "analysis" / "comparison.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {int(row["batch_size"]): row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"No comparison rows in {path}")
    return rows


def rounds(run_dir: Path, arm: str) -> dict[int, dict[int, float]]:
    found = {}
    for path in sorted(run_dir.glob(f"r?_{arm}/result.json")):
        round_id = int(path.parent.name[1])
        data = json.loads(path.read_text(encoding="utf-8"))
        values = {}
        for row in data["results"]:
            if row["requested_splits"] != "auto":
                continue
            if row["status"] != "ok" or not row["correct"]:
                raise ValueError(f"Invalid result in {path}: {row}")
            values[int(row["batch_size"])] = float(row["cuda_graph_us"])
        found[round_id] = values
    if sorted(found) != [1, 2, 3]:
        raise ValueError(f"Expected three {arm} rounds in {run_dir}")
    return found


def main() -> None:
    args = arguments()
    pr_rows = comparison(args.pr_run_dir)
    baseline_rows = comparison(args.baseline_run_dir)
    if pr_rows.keys() != baseline_rows.keys():
        raise ValueError("PR and baseline comparisons cover different batch sizes")
    batches = sorted(pr_rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    colors = {"baseline": "#71717a", "pr": "#2563eb", "candidate": "#ea580c"}
    baseline = [float(baseline_rows[b]["reference_median_us"]) for b in batches]
    pr = [float(pr_rows[b]["reference_median_us"]) for b in batches]
    candidate = [float(pr_rows[b]["candidate_median_us"]) for b in batches]
    candidate_baseline = [
        float(baseline_rows[b]["candidate_median_us"]) for b in batches
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for name, values in (("baseline", baseline), ("pr", pr), ("candidate", candidate)):
        ax.plot(batches, values, marker="o", linewidth=2, color=colors[name], label=name)
    ax.set(yscale="log", xlabel="Batch size B", ylabel="CUDA Graph latency (µs; log scale)",
           title="H100 token-logprob latency: three-round medians")
    ax.set_xticks(batches)
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.savefig(output / "latency_three_way.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    ax.plot(batches, [a / b for a, b in zip(pr, candidate)], marker="o",
            linewidth=2, color=colors["pr"], label="PR / candidate")
    ax.plot(batches, [a / b for a, b in zip(baseline, candidate_baseline)], marker="s",
            linewidth=2, color=colors["baseline"], label="baseline / candidate")
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Batch size B", ylabel="Speedup (higher is better)",
           title="Candidate speedup by batch size")
    ax.set_xticks(batches)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output / "speedup_by_batch.png", dpi=180)
    plt.close(fig)

    reference_rounds = rounds(args.pr_run_dir, "reference")
    candidate_rounds = rounds(args.pr_run_dir, "candidate")
    for batch in (96, 128):
        if batch not in pr_rows:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.7), layout="constrained")
        for name, data in (("PR auto", reference_rounds), ("candidate auto", candidate_rounds)):
            values = [data[round_id][batch] for round_id in (1, 2, 3)]
            color = colors["pr" if name.startswith("PR") else "candidate"]
            ax.plot((1, 2, 3), values, marker="o", linewidth=2, color=color, label=name)
        ax.set(xlabel="A/B round", ylabel="CUDA Graph latency (µs)",
               title=f"B={batch}: variation across three rounds")
        ax.set_xticks((1, 2, 3))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(output / f"round_variation_b{batch}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.7), layout="constrained")
    ax.step(batches, [int(pr_rows[b]["reference_splits"]) for b in batches],
            where="mid", marker="o", linewidth=2, color=colors["pr"], label="PR auto")
    ax.step(batches, [int(pr_rows[b]["candidate_splits"]) for b in batches],
            where="mid", marker="o", linewidth=2, color=colors["candidate"],
            label="candidate auto")
    ax.set(xlabel="Batch size B", ylabel="Vocabulary splits", title="Selected split count")
    ax.set_xticks(batches)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output / "selected_splits.png", dpi=180)
    plt.close(fig)

    report = [
        "# H100 three-way token-logprob plots", "",
        "Latency values are three-round medians in microseconds. Each round uses",
        "CUDA Graph replay after warmup. The speedup chart separates the original",
        "PR contribution from the candidate's incremental improvement.", "",
        f"PR/candidate geometric-mean speedup: "
        f"{statistics.geometric_mean(a / b for a, b in zip(pr, candidate)):.3f}×.",
        f"Baseline/candidate geometric-mean speedup: "
        f"{statistics.geometric_mean(a / b for a, b in zip(baseline, candidate_baseline)):.3f}×.",
        "", "- `latency_three_way.png`: three implementations by batch size",
        "- `speedup_by_batch.png`: speedup against PR and baseline",
        "- `round_variation_b96.png`, `round_variation_b128.png`: repeated A/B values",
        "- `selected_splits.png`: PR and candidate split choices", "",
    ]
    (output / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
