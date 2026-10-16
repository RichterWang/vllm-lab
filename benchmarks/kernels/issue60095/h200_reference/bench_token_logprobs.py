#!/usr/bin/env python3
"""Microbenchmark vLLM token-logprob kernels from a selected source tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

LAB_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY_ROOT = LAB_ROOT / "src" / "vllm"
FULL_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--binary-root", type=Path, default=DEFAULT_BINARY_ROOT)
    parser.add_argument("--output-dir", type=Path)
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--full",
        action="store_true",
        help="Run the full default batch-size sweep (also the default behavior).",
    )
    batch_group.add_argument(
        "--select",
        type=parse_ints,
        metavar="BATCH_SIZES",
        help="Run only these comma-separated batch sizes, e.g. --select 1,2,4,8,16,32.",
    )
    parser.add_argument("--vocab-sizes", type=parse_ints, default=parse_ints("151936"))
    parser.add_argument("--splits", default="auto,1,2,4,8,16,32")
    parser.add_argument("--num-logprobs", type=int, default=6)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--wall-repetitions", type=int, default=10)
    parser.add_argument("--graph-calls", type=int, default=20)
    parser.add_argument("--graph-replays", type=int, default=10)
    parser.add_argument("--initialize-autotune", action="store_true")
    parser.add_argument("--autotune-cache-dir", type=Path)
    parser.add_argument("--logprob-source-override", type=Path)
    args = parser.parse_args()
    args.batch_sizes = args.select if args.select is not None else FULL_BATCH_SIZES
    return args


def load_kernel_module(
    vllm_root: Path, binary_root: Path, source_override: Path | None = None
):
    """Load Python sources from a worktree and compiled extensions from install."""
    sys.path.insert(0, str(vllm_root.resolve()))
    vllm = importlib.import_module("vllm")
    extension_path = (binary_root.resolve() / "vllm").as_posix()
    if extension_path not in vllm.__path__:
        vllm.__path__.append(extension_path)
    if source_override is None:
        module = importlib.import_module("vllm.v1.worker.gpu.sample.logprob")
    else:
        spec = importlib.util.spec_from_file_location(
            "vllm.v1.worker.gpu.sample.logprob_reference", source_override
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load logprob source: {source_override}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    module_path = Path(module.__file__).resolve()
    if source_override is None and not module_path.is_relative_to(
        vllm_root.resolve() / "vllm"
    ):
        raise RuntimeError(f"Loaded logprob module outside selected tree: {module_path}")
    return vllm, module


def make_call(fn: Callable, logits, token_ids, split: str) -> Callable[[], object]:
    if "num_vocab_splits" not in inspect.signature(fn).parameters:
        if split != "auto":
            raise ValueError("The selected base implementation does not support split overrides")
        return lambda: fn(logits, token_ids)
    split_value = None if split == "auto" else int(split)
    return lambda: fn(logits, token_ids, num_vocab_splits=split_value)


def cuda_graph_latency(call: Callable[[], object], graph_calls: int, replays: int) -> float:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            call()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (graph_calls * replays)


def synchronized_wall_latency(call: Callable[[], object], repetitions: int) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / repetitions


def main() -> None:
    global torch
    args = parse_args()
    vllm_root = args.vllm_root.resolve()
    binary_root = args.binary_root.resolve()

    print("[1/4] Importing PyTorch and Triton...", flush=True)
    import torch as torch_module

    torch = torch_module
    import triton

    print("[2/4] Checking and initializing CUDA...", flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this benchmark requires an NVIDIA GPU and driver")

    device = torch.device("cuda")
    torch.manual_seed(618)
    # Initialize CUDA before vLLM platform discovery and before measuring the
    # first Triton compilation latency.
    torch.empty(1, device=device).zero_()
    torch.cuda.synchronize()

    print(
        "[3/4] Loading the selected vLLM source (platform discovery may query NVML)...",
        flush=True,
    )
    vllm, module = load_kernel_module(
        vllm_root, binary_root, args.logprob_source_override
    )
    print(f"[4/4] Running benchmark for batch sizes {args.batch_sizes}...", flush=True)

    dtype = getattr(torch, args.dtype)
    autotune_result = None
    if args.initialize_autotune:
        from vllm.v1.worker.gpu.sample.logprob_autotune import (
            initialize_split_autotune,
        )

        autotune_result = []
        for vocab_size in args.vocab_sizes:
            result = initialize_split_autotune(
                vocab_size, dtype, args.num_logprobs, device,
                args.autotune_cache_dir,
            )
            autotune_result.append(result)
            print("Autotune:", json.dumps(result), flush=True)
    commit = subprocess.run(
        ["git", "-C", str(vllm_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output_dir = args.output_dir or (
        LAB_ROOT
        / "results"
        / "kernel"
        / "issue60095"
        / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{commit[:10]}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    fn = module.compute_token_logprobs

    for vocab_size in args.vocab_sizes:
        for batch_size in args.batch_sizes:
            logits = torch.randn((batch_size, vocab_size), dtype=dtype, device=device)
            token_ids = torch.randint(
                vocab_size, (batch_size, args.num_logprobs), device=device
            )
            reference = torch.log_softmax(logits.double(), dim=-1).gather(
                -1, token_ids.long()
            ).float()
            split_values = args.splits.split(",")
            if "num_vocab_splits" not in inspect.signature(fn).parameters:
                split_values = ["auto"]

            for split in split_values:
                selector = getattr(module, "_get_num_vocab_splits", None)
                effective_splits = (
                    selector(
                        batch_size, vocab_size, device,
                        *([dtype, args.num_logprobs]
                          if "dtype" in inspect.signature(selector).parameters else []),
                    )
                    if split == "auto" and selector is not None
                    else int(split) if split != "auto" else None
                )
                record: dict[str, object] = {
                    "commit": commit,
                    "implementation": "pr" if "num_vocab_splits" in inspect.signature(fn).parameters else "base",
                    "batch_size": batch_size,
                    "vocab_size": vocab_size,
                    "num_logprobs": args.num_logprobs,
                    "dtype": args.dtype,
                    "requested_splits": split,
                    "effective_splits": effective_splits,
                }
                call = make_call(fn, logits, token_ids, split)
                try:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    output = call()
                    torch.cuda.synchronize()
                    record["first_call_wall_ms"] = (time.perf_counter() - started) * 1000
                    torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-3)
                    record["correct"] = True
                    for _ in range(args.warmup):
                        call()
                    torch.cuda.synchronize()
                    record["warm_wall_ms"] = synchronized_wall_latency(
                        call, args.wall_repetitions
                    )
                    record["cuda_graph_us"] = cuda_graph_latency(
                        call, args.graph_calls, args.graph_replays
                    )
                    record["status"] = "ok"
                except (RuntimeError, AssertionError, ValueError) as error:
                    torch.cuda.synchronize()
                    record.update(correct=False, status="error", error=str(error))
                rows.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "vllm_root": str(vllm_root),
        "vllm_module": module.__file__,
        "source_module_sha256": hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_branch": subprocess.run(
            ["git", "-C", str(vllm_root), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_dirty": bool(
            subprocess.run(
                ["git", "-C", str(vllm_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "vllm_version": vllm.__version__,
        "commit": commit,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_properties": str(torch.cuda.get_device_properties(device)),
        "gpu_sm_count": torch.cuda.get_device_properties(device).multi_processor_count,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "autotune": autotune_result,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        | {"vllm_root": str(vllm_root), "binary_root": str(binary_root)},
        "results": rows,
    }
    (output_dir / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        with (output_dir / "result.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
