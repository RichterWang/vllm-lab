#!/usr/bin/env python3
"""Summarize paired high-pressure sync confirmation runs and plot results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, median, stdev


METRICS = {
    "output_tokens_per_s": "output_tokens_per_s",
    "requests_per_s": "requests_per_s",
    "ttft_ms_p50": "ttft_ms_p50",
    "ttft_ms_p99": "ttft_ms_p99",
    "e2el_ms_p99": "e2el_ms_p99",
    "inter_chunk_gap_ms_p99": "inter_chunk_gap_ms_p99",
}


def load_cases(root: Path) -> dict[tuple[int, str], dict]:
    cases: dict[tuple[int, str], dict] = {}
    for case_dir in sorted(root.glob("confirm-sync-pair*-*")):
        parts = case_dir.name.split("-")
        if len(parts) < 5 or parts[0] != "confirm" or parts[1] != "sync":
            continue
        try:
            pair = int(parts[2].removeprefix("pair"))
            implementation = parts[3]
        except (ValueError, IndexError):
            continue
        if implementation not in {"base", "patch"}:
            continue
        summaries = list(case_dir.glob("*-long-shared-sync-seed*.json"))
        if len(summaries) != 1:
            continue
        summary = json.loads(summaries[0].read_text())
        request_files = list(case_dir.glob("*-requests.jsonl"))
        requests = {}
        if len(request_files) == 1:
            for line in request_files[0].read_text().splitlines():
                request = json.loads(line)
                request_id = request["request_index"]
                output_bytes = ",".join(
                    str(token_id) for token_id in request["output_token_ids"]
                ).encode()
                requests[request_id] = {
                    "prompt_sha256": request["prompt_sha256"],
                    "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
                    "output_token_ids": request["output_token_ids"],
                    "status": request["status"],
                }
        summary["_requests"] = requests
        cases[pair, implementation] = summary
    return cases


def bootstrap_ci(values: list[float], seed: int = 61604, samples: int = 20000):
    if not values:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def student_t_ci_mean(values: list[float]) -> tuple[float | None, float | None]:
    """Small-sample paired t interval; useful alongside the bootstrap interval."""
    if not values:
        return None, None
    if len(values) == 1:
        return None, None
    # Two-sided 95% Student-t critical values, indexed by degrees of freedom.
    critical = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }.get(len(values) - 1, 1.96)
    half_width = critical * stdev(values) / len(values) ** 0.5
    center = mean(values)
    return center - half_width, center + half_width


def load_requests(case_dir: Path) -> dict[int, dict]:
    files = list(case_dir.glob("*-requests.jsonl"))
    if len(files) != 1:
        return {}
    requests = {}
    for line in files[0].read_text().splitlines():
        request = json.loads(line)
        ids = request.get("output_token_ids", [])
        output_hash = hashlib.sha256(
            ",".join(str(token_id) for token_id in ids).encode()
        ).hexdigest()
        requests[request["request_index"]] = {
            "prompt_sha256": request["prompt_sha256"],
            "output_sha256": output_hash,
            "output_token_ids": ids,
            "status": request["status"],
            "output_count": len(ids),
            "finish_reason": request.get("finish_reason"),
        }
    return requests


def compare_requests(label: str, left_name: str, right_name: str, root: Path) -> dict:
    left, right = load_requests(root / left_name), load_requests(root / right_name)
    common = sorted(left.keys() & right.keys())
    return {
        "comparison": label,
        "left_run": left_name,
        "right_run": right_name,
        "requests_compared": len(common),
        "prompt_ids_match_count": sum(
            left[i]["prompt_sha256"] == right[i]["prompt_sha256"] for i in common
        ),
        "output_ids_match_count": sum(
            left[i]["output_sha256"] == right[i]["output_sha256"] for i in common
        ),
        "output_ids_match_pct": (
            100 * sum(left[i]["output_sha256"] == right[i]["output_sha256"] for i in common)
            / len(common) if common else None
        ),
        "output_lengths_match_count": sum(
            left[i]["output_count"] == right[i]["output_count"] for i in common
        ),
        "statuses_match_count": sum(
            left[i]["status"] == right[i]["status"] for i in common
        ),
        "finish_reasons_match_count": sum(
            left[i]["finish_reason"] == right[i]["finish_reason"] for i in common
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.results)
    pairs = [
        pair for pair in sorted({pair for pair, _ in cases})
        if (pair, "base") in cases and (pair, "patch") in cases
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in pairs:
        base, patch = cases[pair, "base"], cases[pair, "patch"]
        row = {
            "pair": pair,
            "seed": base.get("seed"),
            "base_requests_succeeded": base.get("requests_succeeded"),
            "patch_requests_succeeded": patch.get("requests_succeeded"),
            "base_requests_failed": base.get("requests_failed"),
            "patch_requests_failed": patch.get("requests_failed"),
        }
        base_requests = base.get("_requests", {})
        patch_requests = patch.get("_requests", {})
        comparable = sorted(set(base_requests) & set(patch_requests))
        row["requests_compared"] = len(comparable)
        row["prompt_ids_match_count"] = sum(
            base_requests[i]["prompt_sha256"] == patch_requests[i]["prompt_sha256"]
            for i in comparable
        )
        row["output_ids_match_count"] = sum(
            base_requests[i]["output_sha256"] == patch_requests[i]["output_sha256"]
            for i in comparable
        )
        row["output_ids_match_pct"] = (
            100 * row["output_ids_match_count"] / len(comparable)
            if comparable else None
        )
        divergence_positions = []
        for request_id in comparable:
            base_ids = base_requests[request_id]["output_token_ids"]
            patch_ids = patch_requests[request_id]["output_token_ids"]
            if base_ids != patch_ids:
                divergence_positions.append(
                    next(
                        (i for i, (b, p) in enumerate(zip(base_ids, patch_ids)) if b != p),
                        min(len(base_ids), len(patch_ids)),
                    )
                )
        row["mismatched_output_count"] = len(divergence_positions)
        row["first_divergence_token_median"] = (
            median(divergence_positions) if divergence_positions else None
        )
        for metric in METRICS:
            b, p = base.get(metric), patch.get(metric)
            row[f"base_{metric}"] = b
            row[f"patch_{metric}"] = p
            row[f"patch_over_base_{metric}"] = p / b if b and p is not None else None
            row[f"change_pct_{metric}"] = (p / b - 1) * 100 if b and p is not None else None
        row["outputs_match"] = (
            bool(comparable)
            and row["output_ids_match_count"] == len(comparable)
        )
        rows.append(row)

    csv_path = args.output_dir / "paired_results.csv"
    fields = list(rows[0]) if rows else ["pair"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "paired_rounds": len(rows),
        "complete_pairs": pairs,
        "all_outputs_match": bool(rows) and all(row["outputs_match"] for row in rows),
        "metrics": {},
    }
    output_comparisons = []
    for pair in pairs:
        output_comparisons.append(
            compare_requests(
                f"pair-{pair}-base-vs-patch",
                f"confirm-sync-pair{pair}-base-seed616041{pair}",
                f"confirm-sync-pair{pair}-patch-seed616041{pair}",
                args.results,
            )
        )
    control_base = args.results / "confirm-sync-pair5-base-seed6160415"
    control_repeat = args.results / "base-base-control-seed6160415"
    if control_base.is_dir() and control_repeat.is_dir():
        output_comparisons.append(
            compare_requests(
                "base-base-same-seed-repeat",
                "confirm-sync-pair5-base-seed6160415",
                "base-base-control-seed6160415",
                args.results,
            )
        )
    report["output_consistency"] = output_comparisons
    (args.output_dir / "output_consistency.csv").write_text(
        "\n".join(
            [
                ",".join(output_comparisons[0].keys()),
                *(
                    ",".join(str(row[key]) for key in output_comparisons[0])
                    for row in output_comparisons
                ),
            ]
        )
        + "\n"
    )
    for metric in METRICS:
        changes = [row[f"change_pct_{metric}"] for row in rows if row[f"change_pct_{metric}"] is not None]
        low, high = bootstrap_ci(changes)
        t_low, t_high = student_t_ci_mean(changes)
        report["metrics"][metric] = {
            "pair_changes_pct": changes,
            "median_change_pct": median(changes) if changes else None,
            "mean_change_pct": sum(changes) / len(changes) if changes else None,
            "bootstrap_95pct_ci_mean_change_pct": [low, high],
            "paired_student_t_95pct_ci_mean_change_pct": [t_low, t_high],
            "interpretation": "higher_is_better" if metric in {"output_tokens_per_s", "requests_per_s"} else "lower_is_better",
        }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to generate the PNG") from exc
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    if rows:
        x = list(range(len(rows)))
        axes[0].plot(x, [row["base_output_tokens_per_s"] for row in rows], "o-", label="base")
        axes[0].plot(x, [row["patch_output_tokens_per_s"] for row in rows], "o-", label="patch")
        axes[0].set_xticks(x, [f"pair {row['pair']}" for row in rows])
        axes[0].set_ylabel("Output tokens/s")
        axes[0].set_title("Synchronized high-pressure throughput")
        axes[0].legend()

        latency = [row["patch_over_base_ttft_ms_p99"] for row in rows]
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[1].plot(x, latency, "o-", color="#c44e52")
        axes[1].set_xticks(x, [f"pair {row['pair']}" for row in rows])
        axes[1].set_ylabel("Patch / base p99 TTFT (lower is better)")
        axes[1].set_title("Paired tail-latency ratio")
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No complete pairs found", ha="center", va="center")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle("Issue #61604 H100 confirmation runs")
    fig.tight_layout()
    fig.savefig(args.output_dir / "confirmation.png", dpi=160)

    if output_comparisons:
        fig, axis = plt.subplots(figsize=(10, 4.8))
        labels = [row["comparison"].replace("-", " ") for row in output_comparisons]
        values = [row["output_ids_match_pct"] for row in output_comparisons]
        colors = ["#4c72b0" if "base-vs-patch" in row["comparison"] else "#dd8452"
                  for row in output_comparisons]
        axis.bar(labels, values, color=colors)
        axis.set_ylim(0, 105)
        axis.set_ylabel("Requests with exactly matching output token IDs (%)")
        axis.set_title("Output determinism: base/patch and base/base control")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "output_consistency.png", dpi=160)


if __name__ == "__main__":
    main()
