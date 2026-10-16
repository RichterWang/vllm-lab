#!/usr/bin/env python3
"""Compare static and cached split selection in one unchanged vLLM source tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BATCHES = (1, 8, 16, 24, 32, 47, 64, 96, 128, 192, 256)
LARGE_BATCHES = (96, 128, 192, 256)
VOCABS = (131072, 151936, 262144)
SPLITS = (1, 2, 4, 8, 16, 32, 64)


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def telemetry() -> dict:
    names = "temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw"
    try:
        result = command(
            "nvidia-smi", f"--query-gpu={names}", "--format=csv,noheader,nounits"
        )
        return dict(zip(names.split(","), (field.strip() for field in result.split(","))))
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": repr(error)}


def graph_us(torch, call, calls: int = 10, replays: int = 10) -> float:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            call()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end) * 1000 / (calls * replays)
    del graph
    return elapsed


def wall_us(torch, call, repetitions: int = 30) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(repetitions):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - start) / (repetitions * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--binary-root", type=Path, default=ROOT / "src/vllm")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-sizes", default=",".join(map(str, VOCABS)))
    parser.add_argument("--batch-sizes", default=",".join(map(str, BATCHES)))
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    vocabs = tuple(int(item) for item in args.vocab_sizes.split(","))
    batches = tuple(int(item) for item in args.batch_sizes.split(","))
    if args.rounds < 3:
        parser.error("at least three interleaved rounds are required")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n")

    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(618)
    device = torch.device("cuda:0")
    torch.empty(1, device=device).zero_()
    sys.path.insert(0, str(args.vllm_root.resolve()))
    vllm = importlib.import_module("vllm")
    extension_path = args.binary_root.resolve() / "vllm"
    vllm.__path__.append(str(extension_path))
    logprob = importlib.import_module("vllm.v1.worker.gpu.sample.logprob")
    autotune = importlib.import_module("vllm.v1.worker.gpu.sample.logprob_autotune")
    if not Path(logprob.__file__).resolve().is_relative_to(args.vllm_root.resolve()):
        raise RuntimeError(f"wrong logprob source: {logprob.__file__}")
    if not Path(autotune.__file__).resolve().is_relative_to(args.vllm_root.resolve()):
        raise RuntimeError(f"wrong autotune source: {autotune.__file__}")
    if not all(hasattr(logprob, name) for name in (
        "_AUTOTUNED_SPLITS", "_get_fallback_num_vocab_splits",
        "_get_num_vocab_splits", "compute_token_logprobs"
    )):
        raise RuntimeError("source lacks required same-tree selection API")
    if tuple(autotune._BATCHES) != BATCHES or tuple(autotune._SPLITS) != SPLITS:
        raise RuntimeError("calibration batch/split grid changed; inspect before testing")
    if autotune._CALIBRATION_K != 6:
        raise RuntimeError("calibration K changed; inspect before testing")

    source = args.vllm_root.resolve()
    props = torch.cuda.get_device_properties(device)
    environment = {
        "source_tree": str(source),
        "branch": command("git", "-C", str(source), "branch", "--show-current"),
        "head": command("git", "-C", str(source), "rev-parse", "HEAD"),
        "status_porcelain": command("git", "-C", str(source), "status", "--porcelain"),
        "logprob_path": str(Path(logprob.__file__).resolve()),
        "logprob_sha256": hashlib.sha256(Path(logprob.__file__).read_bytes()).hexdigest(),
        "autotune_path": str(Path(autotune.__file__).resolve()),
        "autotune_sha256": hashlib.sha256(Path(autotune.__file__).read_bytes()).hexdigest(),
        "gpu": props.name,
        "gpu_memory_bytes": props.total_memory,
        "sms": props.multi_processor_count,
        "nvidia_smi": command("nvidia-smi", "--query-gpu=name,memory.total,driver_version,uuid", "--format=csv,noheader"),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "cache_key_fields": list(autotune._fingerprint(vocabs[0], torch.bfloat16, device)),
        "calibration_k": autotune._CALIBRATION_K,
        "calibration_batches": list(autotune._BATCHES),
        "candidate_splits": list(autotune._SPLITS),
        "shape_vocabs": vocabs, "shape_batches": batches,
        "rounds": args.rounds,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (output / "source_diff.patch").write_text(
        command("git", "-C", str(source), "diff", "--binary", "HEAD") + "\n"
    )
    source_snapshot = output / "source_snapshot"
    source_snapshot.mkdir()
    for original in (Path(logprob.__file__), Path(autotune.__file__)):
        shutil.copy2(original, source_snapshot / original.name)
    print(json.dumps(environment), flush=True)
    cache_dir = output / "selection_cache"
    initialization = []
    tables = {}
    for vocab in vocabs:
        # One empty selection cache per experiment; Triton JIT cache is left unchanged.
        start = time.perf_counter()
        cold = autotune.initialize_split_autotune(vocab, torch.bfloat16, 6, device, cache_dir)
        cold_wall = time.perf_counter() - start
        if cold["status"] != "calibrated":
            raise RuntimeError(f"expected cold calibration for V={vocab}: {cold['status']}")
        key = (0, vocab, torch.bfloat16)
        tables[vocab] = dict(logprob._AUTOTUNED_SPLITS[key])
        start = time.perf_counter()
        hit = autotune.initialize_split_autotune(vocab, torch.bfloat16, 6, device, cache_dir)
        hit_wall = time.perf_counter() - start
        if hit["status"] != "cache_hit" or logprob._AUTOTUNED_SPLITS[key] != tables[vocab]:
            raise RuntimeError(f"cache hit failed for V={vocab}")
        initialization.append({"vocab_size": vocab, "cold_wall_s": cold_wall,
                               "hit_wall_s": hit_wall, "cold": cold, "hit": hit})
    (output / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")

    rows = []
    for vocab in vocabs:
        key = (0, vocab, torch.bfloat16)
        for batch in batches:
            ks = (1, 6, 32) if batch in LARGE_BATCHES else (6,)
            for k in ks:
                logits = torch.randn((batch, vocab), dtype=torch.bfloat16, device=device)
                token_ids = torch.randint(vocab, (batch, k), device=device)
                reference = torch.log_softmax(logits.double(), dim=-1).gather(
                    -1, token_ids.long()).float()
                fallback = logprob._get_fallback_num_vocab_splits(batch, vocab, device)
                logprob._AUTOTUNED_SPLITS.pop(key, None)
                assert logprob._get_num_vocab_splits(batch, vocab, device, torch.bfloat16, k) == fallback
                logprob._AUTOTUNED_SPLITS[key] = tables[vocab]
                cached = logprob._get_num_vocab_splits(batch, vocab, device, torch.bfloat16, k)
                valid = [s for s in SPLITS if s <= min(64, vocab // logprob._MIN_TILE_ELEMENTS)]
                if fallback not in valid or cached not in valid:
                    raise RuntimeError(f"selector returned unsupported split at V={vocab}, B={batch}, K={k}")
                # Compile every candidate and check its result before any timed run.
                for split in valid:
                    actual = logprob.compute_token_logprobs(logits, token_ids, split)
                    torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-3)
                torch.cuda.synchronize()
                before = telemetry()
                timings = {s: [] for s in valid}
                pair = {"static": [], "cache": []}
                for round_index in range(args.rounds):
                    order = ("static", "cache") if round_index % 2 == 0 else ("cache", "static")
                    for arm in order:
                        if arm == "static":
                            logprob._AUTOTUNED_SPLITS.pop(key, None)
                            selected = fallback
                        else:
                            logprob._AUTOTUNED_SPLITS[key] = tables[vocab]
                            selected = cached
                        call = lambda: logprob.compute_token_logprobs(logits, token_ids)
                        actual_selected = logprob._get_num_vocab_splits(batch, vocab, device, torch.bfloat16, k)
                        if actual_selected != selected:
                            raise RuntimeError("selector state mismatch")
                        pair[arm].append({"round": round_index + 1, "split": selected,
                                          "device_us": graph_us(torch, call),
                                          "wall_us": wall_us(torch, call)})
                    for split in (valid if round_index % 2 == 0 else tuple(reversed(valid))):
                        call = lambda s=split: logprob.compute_token_logprobs(logits, token_ids, s)
                        timings[split].append(graph_us(torch, call))
                logprob._AUTOTUNED_SPLITS[key] = tables[vocab]
                medians = {split: statistics.median(values) for split, values in timings.items()}
                best = min(medians, key=medians.__getitem__)
                row = {
                    "vocab_size": vocab, "batch_size": batch, "k": k,
                    "fallback_split": fallback, "cached_split": cached,
                    "best_split": best, "best_device_us": medians[best],
                    "candidate_device_us": medians,
                    "static_device_us": statistics.median(v["device_us"] for v in pair["static"]),
                    "cached_device_us": statistics.median(v["device_us"] for v in pair["cache"]),
                    "static_wall_us": statistics.median(v["wall_us"] for v in pair["static"]),
                    "cached_wall_us": statistics.median(v["wall_us"] for v in pair["cache"]),
                    "static_rounds": pair["static"], "cached_rounds": pair["cache"],
                    "temperature_before": before, "temperature_after": telemetry(),
                    "correct": True,
                }
                rows.append(row)
                with (output / "raw.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps({key: value for key, value in row.items()
                                  if key not in ("static_rounds", "cached_rounds", "candidate_device_us")}), flush=True)
                del logits, token_ids, reference
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    columns = ["vocab_size", "batch_size", "k", "fallback_split", "cached_split",
               "best_split", "best_device_us", "static_device_us", "cached_device_us",
               "static_wall_us", "cached_wall_us", "correct"]
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({name: row[name] for name in columns} for row in rows)


if __name__ == "__main__":
    main()
