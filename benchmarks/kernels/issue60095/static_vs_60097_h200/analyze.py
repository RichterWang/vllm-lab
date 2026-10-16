#!/usr/bin/env python3
"""Pair the three-round H200 static-candidate vs #60097 microbenchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def load_arm(path: Path) -> tuple[dict, dict[tuple[int, int, int], dict]]:
    metadata = json.loads((path / "result.json").read_text())
    rows = list(csv.DictReader((path / "result.csv").open(newline="")))
    result = {}
    for row in rows:
        if row["requested_splits"] != "auto":
            continue
        if row["status"] != "ok" or row["correct"].lower() != "true":
            raise ValueError(f"Unsuccessful result in {path}: {row}")
        key = (int(row["vocab_size"]), int(row["batch_size"]), int(row["num_logprobs"]))
        if key in result:
            raise ValueError(f"Duplicate shape {key} in {path}")
        result[key] = row
    if not result:
        raise ValueError(f"No auto-split rows in {path}")
    return metadata, result


def geo_mean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(v) for v in values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    rounds = (1, 2, 3)
    arms = ("candidate", "reference")
    metadata = {}
    measurements = {}
    for rnd in rounds:
        for arm in arms:
            by_shape_path = root / f"r{rnd}_{arm}_k6"
            # Each K is saved in its own folder; load the union below.
            merged: dict[tuple[int, int, int], dict] = {}
            per_k_meta = []
            for k in (1, 6, 32):
                meta, rows = load_arm(root / f"r{rnd}_{arm}_k{k}")
                per_k_meta.append(meta)
                merged.update(rows)
            metadata[(rnd, arm)] = per_k_meta
            measurements[(rnd, arm)] = merged

    expected_shapes = set(measurements[(1, "candidate")])
    if len(expected_shapes) != 57:
        raise ValueError(f"Expected 57 shapes, found {len(expected_shapes)}")
    for key, rows in measurements.items():
        if set(rows) != expected_shapes:
            raise ValueError(f"Shape mismatch in {key}")
        for meta in metadata[key]:
            if meta["gpu"] != "NVIDIA H200":
                raise ValueError(f"Unexpected GPU: {meta['gpu']}")
            if (meta["torch"], meta["cuda_runtime"], meta["triton"]) != (
                "2.13.0+cu130", "13.0", "3.7.1"
            ):
                raise ValueError(f"Runtime mismatch in {key}: {meta}")

    fields = ("cuda_graph_us", "warm_wall_ms")
    medians: dict[tuple[int, str, tuple[int, int, int]], dict[str, float]] = {}
    output_rows = []
    for arm in arms:
        for shape in sorted(expected_shapes):
            medians[(0, arm, shape)] = {
                field: statistics.median(float(measurements[(rnd, arm)][shape][field]) for rnd in rounds)
                for field in fields
            }
    for shape in sorted(expected_shapes):
        vocab, batch, k = shape
        candidate = medians[(0, "candidate", shape)]
        reference = medians[(0, "reference", shape)]
        cand_splits = [int(measurements[(rnd, "candidate")][shape]["effective_splits"]) for rnd in rounds]
        ref_splits = [int(measurements[(rnd, "reference")][shape]["effective_splits"]) for rnd in rounds]
        row = {
            "vocab_size": vocab,
            "batch_size": batch,
            "num_logprobs_K": k,
            "candidate_splits": ",".join(map(str, cand_splits)),
            "60097_splits": ",".join(map(str, ref_splits)),
            "candidate_cuda_graph_us_median": candidate["cuda_graph_us"],
            "60097_cuda_graph_us_median": reference["cuda_graph_us"],
            "60097_over_candidate_device_speedup": reference["cuda_graph_us"] / candidate["cuda_graph_us"],
            "candidate_wall_ms_median": candidate["warm_wall_ms"],
            "60097_wall_ms_median": reference["warm_wall_ms"],
            "60097_over_candidate_wall_speedup": reference["warm_wall_ms"] / candidate["warm_wall_ms"],
        }
        for rnd in rounds:
            for arm in arms:
                data = measurements[(rnd, arm)][shape]
                row[f"r{rnd}_{arm}_cuda_graph_us"] = data["cuda_graph_us"]
                row[f"r{rnd}_{arm}_wall_ms"] = data["warm_wall_ms"]
        output_rows.append(row)

    csv_path = root / "comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    groups = []
    for vocab in (131072, 151936, 262144):
        for k in (1, 6, 32):
            rows = [r for r in output_rows if r["vocab_size"] == vocab and r["num_logprobs_K"] == k]
            if not rows:
                continue
            for metric, field in (("device", "60097_over_candidate_device_speedup"),
                                  ("wall", "60097_over_candidate_wall_speedup")):
                ratios = [r[field] for r in rows]
                groups.append({
                    "vocab_size": vocab,
                    "K": k,
                    "metric": metric,
                    "shapes": len(rows),
                    "geomean_speedup_60097_over_candidate": geo_mean(ratios),
                    "minimum_speedup": min(ratios),
                    "maximum_speedup": max(ratios),
                    "candidate_over_60097_slowdown_gt_5pct": sum(r < 1 / 1.05 for r in ratios),
                })
    all_rows = output_rows
    overall = {}
    for metric, field in (("device", "60097_over_candidate_device_speedup"),
                          ("wall", "60097_over_candidate_wall_speedup")):
        ratios = [r[field] for r in all_rows]
        overall[metric] = {
            "geomean_speedup_60097_over_candidate": geo_mean(ratios),
            "minimum_speedup": min(ratios),
            "maximum_speedup": max(ratios),
            "candidate_over_60097_slowdown_gt_5pct": sum(r < 1 / 1.05 for r in ratios),
        }
    round_summary = {}
    for rnd in rounds:
        round_summary[str(rnd)] = {}
        for metric, field in (("device", "cuda_graph_us"), ("wall", "warm_wall_ms")):
            ratios = [
                float(measurements[(rnd, "reference")][shape][field])
                / float(measurements[(rnd, "candidate")][shape][field])
                for shape in sorted(expected_shapes)
            ]
            round_summary[str(rnd)][metric] = {
                "geomean_speedup_60097_over_candidate": geo_mean(ratios),
                "minimum_speedup": min(ratios),
                "candidate_over_60097_slowdown_gt_5pct": sum(r < 1 / 1.05 for r in ratios),
            }
    analysis = {
        "shapes": len(output_rows),
        "rounds": list(rounds),
        "overall": overall,
        "round_summary": round_summary,
        "groups": groups,
    }
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

    lines = [
        "# H200: static vllm-60095-pr vs #60097 auto split",
        "",
        "This report compares the full static candidate at commit `8730e4e1d2d6975cb3c8908fa998921f2161781a` with #60097 at `0ddd6017793c943a2091a5800f0cbac1daaa3e02`. The candidate is based on a newer main commit (`d6f0b7a4166b325b4dda6512e2e06dcb581923ba`), so this measures the two implementations as checked out, not an isolated selector-only patch.",
        "",
        f"All {len(output_rows)} shapes have three measurements per arm. All successful correctness flags were checked. Speedup is reference (#60097) latency divided by candidate latency; values above 1 favor the candidate.",
        "",
        "## Overall",
        "",
        f"- CUDA Graph device-time geometric speedup: **{overall['device']['geomean_speedup_60097_over_candidate']:.3f}×**; minimum {overall['device']['minimum_speedup']:.3f}×; >5% candidate regressions {overall['device']['candidate_over_60097_slowdown_gt_5pct']}/{len(output_rows)}.",
        f"- Synchronized Python wall-time geometric speedup: **{overall['wall']['geomean_speedup_60097_over_candidate']:.3f}×**; minimum {overall['wall']['minimum_speedup']:.3f}×; >5% candidate regressions {overall['wall']['candidate_over_60097_slowdown_gt_5pct']}/{len(output_rows)}.",
        "",
        "## Per vocabulary and K",
        "",
        "| V | K | Shapes | Device geo speedup | Device min | Device >5% regressions | Wall geo speedup | Wall min | Wall >5% regressions |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for vocab in (131072, 151936, 262144):
        for k in (1, 6, 32):
            d = next((g for g in groups if g["vocab_size"] == vocab and g["K"] == k and g["metric"] == "device"), None)
            w = next((g for g in groups if g["vocab_size"] == vocab and g["K"] == k and g["metric"] == "wall"), None)
            if d and w:
                lines.append(f"| {vocab} | {k} | {d['shapes']} | {d['geomean_speedup_60097_over_candidate']:.3f}× | {d['minimum_speedup']:.3f}× | {d['candidate_over_60097_slowdown_gt_5pct']} | {w['geomean_speedup_60097_over_candidate']:.3f}× | {w['minimum_speedup']:.3f}× | {w['candidate_over_60097_slowdown_gt_5pct']} |")
    lines += [
        "",
        "## Round-to-round repeatability",
        "",
        "| Round | Device geo speedup | Device min | Device >5% regressions | Wall geo speedup | Wall min | Wall >5% regressions |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rnd in rounds:
        d = round_summary[str(rnd)]["device"]
        w = round_summary[str(rnd)]["wall"]
        lines.append(f"| {rnd} | {d['geomean_speedup_60097_over_candidate']:.3f}× | {d['minimum_speedup']:.3f}× | {d['candidate_over_60097_slowdown_gt_5pct']} | {w['geomean_speedup_60097_over_candidate']:.3f}× | {w['minimum_speedup']:.3f}× | {w['candidate_over_60097_slowdown_gt_5pct']} |")
    lines += [
        "",
        "## Representative V=151936, K=6",
        "",
        "| B | Candidate splits | #60097 splits | Candidate device µs | #60097 device µs | Device speedup | Candidate wall ms | #60097 wall ms | Wall speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in output_rows:
        if r["vocab_size"] == 151936 and r["num_logprobs_K"] == 6:
            lines.append(
                f"| {r['batch_size']} | {r['candidate_splits']} | {r['60097_splits']} | "
                f"{r['candidate_cuda_graph_us_median']:.2f} | {r['60097_cuda_graph_us_median']:.2f} | "
                f"{r['60097_over_candidate_device_speedup']:.3f}× | "
                f"{r['candidate_wall_ms_median']:.4f} | {r['60097_wall_ms_median']:.4f} | "
                f"{r['60097_over_candidate_wall_speedup']:.3f}× |"
            )
    lines += [
        "",
        "## Per-shape results",
        "",
        "See `comparison.csv` for per-round measurements, median device and wall latency, effective split choices, and speedups.",
        "",
        "## Interpretation limits",
        "",
        "- This is a kernel microbenchmark with synthetic logits, not a model-serving or TTFT test.",
        "- Geometric wall-time speedup is consistent across the three rounds, but individual wall-time outliers occurred (including one large candidate slowdown in round 1 at V=131072/B=1/K=6). The headline per-shape wall comparisons use the median of three rounds; inspect the round columns in `comparison.csv` before making per-shape claims.",
        "- The candidate commit is based on a newer main snapshot than the original #60097 PR commit. Results compare those full source snapshots; they do not isolate only the static split rule.",
        "- The optional main baseline was not run; it is not required to answer candidate versus #60097.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [f"{r['vocab_size']//1024}k/B{r['batch_size']}/K{r['num_logprobs_K']}" for r in output_rows]
    x = list(range(len(output_rows)))
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    for ax, field, title in (
        (axes[0], "60097_over_candidate_device_speedup", "CUDA Graph device speedup (60097 / candidate)"),
        (axes[1], "60097_over_candidate_wall_speedup", "Synchronized Python wall speedup (60097 / candidate)"),
    ):
        vals = [r[field] for r in output_rows]
        ax.plot(x, vals, marker=".", linewidth=1)
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.axhline(1 / 1.05, color="red", linestyle=":", linewidth=1)
        ax.set_ylabel("Speedup")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[-1].set_xlabel("V / batch size / K")
    fig.tight_layout()
    fig.savefig(root / "speedup_by_shape.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    for ax, cand_field, ref_field, title in (
        (axes[0], "candidate_cuda_graph_us_median", "60097_cuda_graph_us_median", "CUDA Graph device latency (µs)"),
        (axes[1], "candidate_wall_ms_median", "60097_wall_ms_median", "Synchronized wall latency (ms)"),
    ):
        ax.plot(x, [r[ref_field] for r in output_rows], label="#60097", marker=".", linewidth=1)
        ax.plot(x, [r[cand_field] for r in output_rows], label="vllm-60095-pr", marker=".", linewidth=1)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[-1].set_xlabel("V / batch size / K")
    fig.tight_layout()
    fig.savefig(root / "latency_by_shape.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
