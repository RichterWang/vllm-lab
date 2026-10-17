#!/usr/bin/env python3
"""Summarize the official vLLM serving benchmark JSON for PR #61991."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

ARMS = ("parent", "pr", "main")
METRICS = (
    "request_throughput",
    "output_throughput",
    "p50_ttft_ms",
    "p99_ttft_ms",
    "p50_tpot_ms",
    "p99_tpot_ms",
    "p50_itl_ms",
    "p99_itl_ms",
    "p50_e2el_ms",
    "p99_e2el_ms",
)
FILE_RE = re.compile(r"r(?P<round>\d+)-c(?P<concurrency>\d+)\.json$")


def load_runs(results_dir: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    runs: dict[tuple[str, int, int], dict[str, Any]] = {}
    for arm in ARMS:
        arm_dir = results_dir / "raw" / arm
        if not arm_dir.exists():
            continue
        for path in arm_dir.glob("r*-c*.json"):
            match = FILE_RE.fullmatch(path.name)
            if not match:
                continue
            round_id = int(match.group("round"))
            concurrency = int(match.group("concurrency"))
            data = json.loads(path.read_text())
            runs[(arm, round_id, concurrency)] = data
    return runs


def med(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def write_analysis(results_dir: Path) -> None:
    runs = load_runs(results_dir)
    concurrencies = sorted({key[2] for key in runs})
    out_dir = results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for concurrency in concurrencies:
            records = [
                runs[(arm, r, concurrency)]
                for r in range(1, 4)
                if (arm, r, concurrency) in runs
            ]
            if not records:
                continue
            row: dict[str, Any] = {
                "arm": arm,
                "concurrency": concurrency,
                "runs": len(records),
                "successful_median": med([float(d.get("completed", 0)) for d in records]),
                "failed_total": sum(int(d.get("failed", 0)) for d in records),
            }
            for metric in METRICS:
                values = [float(d[metric]) for d in records if d.get(metric) is not None]
                row[f"median_{metric}"] = med(values)
                row[f"min_{metric}"] = min(values) if values else None
                row[f"max_{metric}"] = max(values) if values else None
            summary_rows.append(row)

    if summary_rows:
        columns = list(summary_rows[0])
        with (out_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summary_rows)

    comparison_rows: list[dict[str, Any]] = []
    for comparator in ("parent", "main"):
        for concurrency in concurrencies:
            row: dict[str, Any] = {"comparison": f"pr_vs_{comparator}", "concurrency": concurrency}
            for metric in METRICS:
                pr_value = next(
                    (r[f"median_{metric}"] for r in summary_rows if r["arm"] == "pr" and r["concurrency"] == concurrency),
                    None,
                )
                baseline_value = next(
                    (r[f"median_{metric}"] for r in summary_rows if r["arm"] == comparator and r["concurrency"] == concurrency),
                    None,
                )
                if pr_value and baseline_value:
                    # Latency speedup is baseline / PR; throughput speedup is PR / baseline.
                    ratio = (
                        pr_value / baseline_value
                        if "throughput" in metric
                        else baseline_value / pr_value
                    )
                    row[f"speedup_{metric}"] = ratio
                else:
                    row[f"speedup_{metric}"] = None
            comparison_rows.append(row)

    if comparison_rows:
        columns = list(comparison_rows[0])
        with (out_dir / "comparisons.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(comparison_rows)

    consistency: list[dict[str, Any]] = []
    repeat_stability: list[dict[str, Any]] = []
    for arm in ("pr", "main"):
        for round_id in range(1, 4):
            for concurrency in concurrencies:
                base = runs.get(("parent", round_id, concurrency))
                candidate = runs.get((arm, round_id, concurrency))
                if not base or not candidate:
                    continue
                base_text = base.get("generated_texts", [])
                candidate_text = candidate.get("generated_texts", [])
                count = min(len(base_text), len(candidate_text))
                matches = sum(a == b for a, b in zip(base_text, candidate_text))
                consistency.append(
                    {
                        "arm": arm,
                        "round": round_id,
                        "concurrency": concurrency,
                        "paired_requests": count,
                        "exact_matches": matches,
                        "match_fraction": matches / count if count else None,
                        "parent_failed": int(base.get("failed", 0)),
                        "arm_failed": int(candidate.get("failed", 0)),
                    }
                )
    # Compare text lists positionally. The benchmark JSON does not include a
    # stable prompt/request ID to prove cross-run alignment, so this is only a
    # repeatability signal, not a per-request correctness check.
    for arm in ARMS:
        for concurrency in concurrencies:
            for round_a, round_b in ((1, 2), (1, 3), (2, 3)):
                first = runs.get((arm, round_a, concurrency), {}).get("generated_texts", [])
                second = runs.get((arm, round_b, concurrency), {}).get("generated_texts", [])
                count = min(len(first), len(second))
                matches = sum(a == b for a, b in zip(first, second))
                repeat_stability.append(
                    {
                        "arm": arm,
                        "round_a": round_a,
                        "round_b": round_b,
                        "concurrency": concurrency,
                        "paired_requests": count,
                        "exact_matches": matches,
                        "match_fraction": matches / count if count else None,
                    }
                )
    if consistency:
        with (out_dir / "output_consistency.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(consistency[0]))
            writer.writeheader()
            writer.writerows(consistency)

    if repeat_stability:
        with (out_dir / "same_arm_repeatability.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(repeat_stability[0]))
            writer.writeheader()
            writer.writerows(repeat_stability)

    lines = [
        "# H100 end-to-end results: PR #61991",
        "",
        "The values below summarize per-run medians across the available rounds.",
        "For latency, speedup is comparator / PR; for throughput it is PR / comparator.",
        "Ratios divide each arm's three-run median; they are not medians of same-round paired ratios.",
        "A value above 1 favors the PR. See the raw JSON for request-level data.",
        "",
        "## Run completeness",
        "",
        f"Loaded {len(runs)} benchmark JSON files.",
        "Expected matrix: 3 arms × 3 rounds × 3 concurrency levels = 27 runs; each run targets 200 requests.",
        "",
        "## Per-arm medians",
        "",
        "| Arm | Concurrency | Runs | Success median | Failed total | Output tok/s | TPOT p50 ms | ITL p99 ms | E2E p99 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {arm} | {concurrency} | {runs} | {successful_median} | {failed_total} | {output} | {tpot} | {itl} | {e2el} |".format(
                arm=row["arm"],
                concurrency=row["concurrency"],
                runs=row["runs"],
                successful_median=fmt(row["successful_median"]),
                failed_total=row["failed_total"],
                output=fmt(row.get("median_output_throughput")),
                tpot=fmt(row.get("median_p50_tpot_ms")),
                itl=fmt(row.get("median_p99_itl_ms")),
                e2el=fmt(row.get("median_p99_e2el_ms")),
            )
        )
    lines += ["", "## PR speedup ratios", "", "| Comparator | Concurrency | Output tok/s | TPOT p50 | ITL p99 | E2E p99 |", "|---|---:|---:|---:|---:|---:|"]
    for row in comparison_rows:
        lines.append(
            "| {comparison} | {concurrency} | {output} | {tpot} | {itl} | {e2el} |".format(
                comparison=row["comparison"],
                concurrency=row["concurrency"],
                output=fmt(row.get("speedup_output_throughput")),
                tpot=fmt(row.get("speedup_p50_tpot_ms")),
                itl=fmt(row.get("speedup_p99_itl_ms")),
                e2el=fmt(row.get("speedup_p99_e2el_ms")),
            )
        )
    if consistency:
        lines += ["", "## Ordered generated-text agreement against parent", "", "| Arm | Concurrency | Paired requests | Exact matches |", "|---|---:|---:|---:|"]
        for arm in ("pr", "main"):
            for concurrency in concurrencies:
                rows = [r for r in consistency if r["arm"] == arm and r["concurrency"] == concurrency]
                if not rows:
                    continue
                total = sum(r["paired_requests"] for r in rows)
                matches = sum(r["exact_matches"] for r in rows)
                lines.append(f"| {arm} | {concurrency} | {total} | {matches}/{total} |")
    lines += [
        "",
        "Text matching is by benchmark result order, not by a stable request ID. Same-arm cross-round matches are also imperfect; see `same_arm_repeatability.csv`. Treat this as a consistency signal, not a correctness proof.",
        "",
        "Request-level p99 metrics (TTFT, TPOT, E2E) use 200 request samples per run and are sensitive to a few tail requests.",
        "ITL p99 pools token inter-token intervals across successful requests (about 25,000 intervals at 128 output tokens); intervals within a request are correlated. Three repetitions provide only an initial stability check.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")

    # Save a compact JSON manifest for machine-readable downstream charts.
    (out_dir / "analysis.json").write_text(
        json.dumps(
            {
                "benchmark_files": len(runs),
                "summary": summary_rows,
                "comparisons": comparison_rows,
                "output_consistency": consistency,
                "same_arm_repeatability": repeat_stability,
            },
            indent=2,
            default=lambda value: float(value) if isinstance(value, float) else value,
        )
        + "\n"
    )
    write_plots(out_dir, summary_rows, comparison_rows, concurrencies)


def write_plots(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    concurrencies: list[int],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("median_output_throughput", "median_p50_tpot_ms", "median_p99_itl_ms"),
        ("Output tok/s", "TPOT p50 (ms)", "ITL p99 (ms)"),
    ):
        for arm in ARMS:
            rows = {
                row["concurrency"]: row
                for row in summary_rows
                if row["arm"] == arm
            }
            xs = [c for c in concurrencies if c in rows]
            ys = [rows[c].get(metric) for c in xs]
            lows = [rows[c].get(metric) - rows[c].get(metric.replace("median_", "min_")) for c in xs]
            highs = [rows[c].get(metric.replace("median_", "max_")) - rows[c].get(metric) for c in xs]
            axis.errorbar(xs, ys, yerr=[lows, highs], marker="o", capsize=3, label=f"{arm} (median, min–max)")
        axis.set_title(title)
        axis.set_xlabel("Maximum request concurrency")
        axis.grid(True, alpha=0.3)
        axis.set_xticks(concurrencies)
    axes[0].set_ylabel("Higher is better for throughput")
    axes[1].set_ylabel("Milliseconds; lower is better")
    axes[2].set_ylabel("Milliseconds; lower is better")
    axes[0].legend()
    fig.savefig(out_dir / "metrics_by_concurrency.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    speedup_metrics = (
        ("speedup_output_throughput", "Output tok/s", False),
        ("speedup_p50_tpot_ms", "TPOT p50", True),
        ("speedup_p99_itl_ms", "ITL p99", True),
    )
    for axis, comparator in zip(axes, ("parent", "main")):
        rows = {
            row["concurrency"]: row
            for row in comparison_rows
            if row["comparison"] == f"pr_vs_{comparator}"
        }
        for metric, label, _ in speedup_metrics:
            xs = [c for c in concurrencies if c in rows and rows[c].get(metric) is not None]
            ys = [rows[c][metric] for c in xs]
            axis.plot(xs, ys, marker="o", label=label)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"PR speedup vs {comparator}")
        axis.set_xlabel("Maximum request concurrency")
        axis.set_ylabel("Speedup ratio (above 1 favors PR)")
        axis.set_xticks(concurrencies)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.savefig(out_dir / "speedup_by_concurrency.png", dpi=160)
    plt.close(fig)


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.3f}" if isinstance(value, (float, int)) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    results_dir = args.results_dir
    if results_dir is None:
        candidates = sorted((Path(__file__).parent / "results").glob("h100_*"))
        if not candidates:
            raise SystemExit("No results found; pass --results-dir explicitly")
        results_dir = candidates[-1]
    write_analysis(results_dir)


if __name__ == "__main__":
    main()
