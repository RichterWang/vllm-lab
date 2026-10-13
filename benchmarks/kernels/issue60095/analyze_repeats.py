#!/usr/bin/env python3
"""Compare repeated base/PR token-logprob benchmark runs and plot results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = LAB_ROOT / "results" / "kernel" / "issue60095" / "repeats"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to a new timestamped analysis directory under --input-dir.",
    )
    return parser.parse_args()


def read_result(path: Path) -> tuple[dict, list[dict]]:
    json_path = path / "result.json"
    csv_path = path / "result.csv"
    if not json_path.is_file() or not csv_path.is_file():
        raise ValueError(f"Missing result.json or result.csv in {path}")

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selected = []
    for row in rows:
        if row.get("requested_splits") != "auto":
            continue
        if row.get("status") != "ok" or row.get("correct", "").lower() != "true":
            raise ValueError(f"Non-successful auto result in {csv_path}: {row}")
        row["batch_size"] = int(row["batch_size"])
        row["cuda_graph_us"] = float(row["cuda_graph_us"])
        selected.append(row)
    if not selected:
        raise ValueError(f"No successful requested_splits=auto rows in {csv_path}")
    return metadata, selected


def discover_runs(input_dir: Path) -> tuple[list[int], dict, dict]:
    # The repeat runner keeps each invocation in a timestamped directory.
    # Accept either one run directory directly or the parent repeats folder.
    if not any((input_dir / f"r{round_id}_{arm}").is_dir() for round_id in range(1, 4) for arm in ("base", "pr")):
        candidates = sorted(input_dir.glob("run_*/"), key=lambda path: path.name)
        if candidates:
            input_dir = candidates[-1]

    rounds: dict[int, dict[str, tuple[dict, list[dict]]]] = {}
    for round_id in range(1, 1000):
        for arm in ("base", "pr"):
            run_dir = input_dir / f"r{round_id}_{arm}"
            if run_dir.exists():
                rounds.setdefault(round_id, {})[arm] = read_result(run_dir)

    round_ids = sorted(rounds)
    if len(round_ids) != 3:
        raise ValueError(f"Expected exactly 3 rounds under {input_dir}; found {round_ids}")
    for round_id in round_ids:
        if set(rounds[round_id]) != {"base", "pr"}:
            raise ValueError(f"Round {round_id} must contain both base and pr result folders")

    metadata = {arm: [] for arm in ("base", "pr")}
    measurements = {arm: {} for arm in ("base", "pr")}
    expected_batches = None
    expected_gpu = None
    expected_runtime = None
    for round_id in round_ids:
        for arm in ("base", "pr"):
            run_metadata, rows = rounds[round_id][arm]
            if run_metadata.get("gpu"):
                if expected_gpu is None:
                    expected_gpu = run_metadata["gpu"]
                elif run_metadata["gpu"] != expected_gpu:
                    raise ValueError(f"GPU mismatch in round {round_id} {arm}: {run_metadata['gpu']}")
            runtime = (
                run_metadata.get("torch"),
                run_metadata.get("cuda_runtime"),
                run_metadata.get("triton"),
            )
            if expected_runtime is None:
                expected_runtime = runtime
            elif runtime != expected_runtime:
                raise ValueError(f"Torch/CUDA/Triton version mismatch in round {round_id} {arm}")

            by_batch = {row["batch_size"]: row["cuda_graph_us"] for row in rows}
            if len(by_batch) != len(rows):
                raise ValueError(f"Duplicate auto batch size in round {round_id} {arm}")
            batches = sorted(by_batch)
            if expected_batches is None:
                expected_batches = batches
            elif batches != expected_batches:
                raise ValueError(
                    f"Batch-size mismatch in round {round_id} {arm}: {batches}; "
                    f"expected {expected_batches}"
                )
            metadata[arm].append(run_metadata)
            measurements[arm][round_id] = by_batch

    assert expected_batches is not None
    return round_ids, measurements, {
        "batches": expected_batches,
        "gpu": expected_gpu,
        "runtime": expected_runtime,
        "metadata": metadata,
    }


def write_comparison_csv(path: Path, round_ids: list[int], measurements: dict) -> list[dict]:
    output_rows = []
    for round_id in round_ids:
        for batch in sorted(measurements["base"][round_id]):
            base_us = measurements["base"][round_id][batch]
            pr_us = measurements["pr"][round_id][batch]
            output_rows.append(
                {
                    "round": round_id,
                    "batch_size": batch,
                    "base_cuda_graph_us": base_us,
                    "pr_auto_cuda_graph_us": pr_us,
                    "speedup_base_over_pr": base_us / pr_us,
                    "pr_latency_reduction_percent": (1 - pr_us / base_us) * 100,
                }
            )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def make_plots(output_dir: Path, round_ids: list[int], measurements: dict, batches: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"base": "#4C78A8", "pr": "#F58518"}
    labels = {"base": "Base", "pr": "PR auto"}
    x = np.arange(len(batches))
    width = 0.36

    all_values = [
        value
        for arm in ("base", "pr")
        for round_id in round_ids
        for value in measurements[arm][round_id].values()
    ]
    shared_ymax = max(all_values) * 1.18

    for round_id in round_ids:
        fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
        base = [measurements["base"][round_id][batch] for batch in batches]
        pr = [measurements["pr"][round_id][batch] for batch in batches]
        bars_base = ax.bar(x - width / 2, base, width, label="Base", color=colors["base"])
        bars_pr = ax.bar(x + width / 2, pr, width, label="PR auto", color=colors["pr"])
        ax.bar_label(bars_base, fmt="%.1f", padding=2, fontsize=8)
        ax.bar_label(bars_pr, fmt="%.1f", padding=2, fontsize=8)
        ax.set_title(f"Round {round_id}: Base vs PR auto")
        ax.set_xlabel("Batch size (logits rows)")
        ax.set_ylabel("CUDA Graph latency (µs; lower is better)")
        ax.set_xticks(x, [str(batch) for batch in batches])
        ax.set_ylim(0, shared_ymax)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.savefig(output_dir / f"round_{round_id}_comparison.png", dpi=160)
        plt.close(fig)

    for arm in ("base", "pr"):
        fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
        for batch in batches:
            values = [measurements[arm][round_id][batch] for round_id in round_ids]
            ax.plot(
                round_ids,
                values,
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=f"Batch {batch}",
            )
        ax.set_title(f"{labels[arm]} latency variation across rounds")
        ax.set_xlabel("Test round")
        ax.set_ylabel("CUDA Graph latency (µs)")
        ax.set_xticks(round_ids, [f"R{round_id}" for round_id in round_ids])
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        fig.savefig(output_dir / f"{arm}_round_variation.png", dpi=160)
        plt.close(fig)


def build_report(
    path: Path,
    round_ids: list[int],
    rows: list[dict],
    measurements: dict,
    batches: list[int],
    environment: dict,
) -> None:
    lines = [
        "# Issue #60095 repeated A/B benchmark analysis",
        "",
        f"- GPU: {environment['gpu']}",
        f"- PyTorch / CUDA / Triton: {' / '.join(environment['runtime'])}",
        "- Metric: CUDA Graph latency in microseconds; lower is better.",
        "- PR rows use `requested_splits=auto`; base rows use its unchanged implementation.",
        "- Speedup is `base latency / PR auto latency`; values above 1 mean PR is faster.",
        "",
        "## Per-round comparison",
        "",
        "| Round | Batch | Base (µs) | PR auto (µs) | Speedup | PR latency reduction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['round']} | {row['batch_size']} | {row['base_cuda_graph_us']:.2f} "
            f"| {row['pr_auto_cuda_graph_us']:.2f} | {row['speedup_base_over_pr']:.2f}× "
            f"| {row['pr_latency_reduction_percent']:.1f}% |"
        )

    lines += [
        "",
        "## Three-round median and variation",
        "",
        "| Implementation | Batch | Median (µs) | Min–max (µs) | Range / median |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, label in (("base", "Base"), ("pr", "PR auto")):
        for batch in batches:
            values = [measurements[arm][round_id][batch] for round_id in round_ids]
            median = statistics.median(values)
            spread = max(values) - min(values)
            lines.append(
                f"| {label} | {batch} | {median:.2f} | {min(values):.2f}–{max(values):.2f} "
                f"| {spread / median * 100:.2f}% |"
            )

    lines += [
        "",
        "## Charts",
        "",
        *[f"- Round {round_id} comparison: `round_{round_id}_comparison.png`" for round_id in round_ids],
        "- Base variation: `base_round_variation.png`",
        "- PR auto variation: `pr_round_variation.png`",
        "",
        "Three repeats are useful for an initial stability check, but too few to claim a robust statistical distribution. The variation charts show the observed runs, not a histogram or confidence interval.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        output_dir = input_dir / f"analysis_{stamp}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    round_ids, measurements, environment = discover_runs(input_dir)
    rows = write_comparison_csv(output_dir / "comparison.csv", round_ids, measurements)
    make_plots(output_dir, round_ids, measurements, environment["batches"])
    build_report(
        output_dir / "report.md",
        round_ids,
        rows,
        measurements,
        environment["batches"],
        environment,
    )
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "rounds": round_ids,
        "batches": environment["batches"],
        "gpu": environment["gpu"],
        "runtime": environment["runtime"],
        "comparison_rows": len(rows),
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Analysis written to {output_dir}")


if __name__ == "__main__":
    main()
