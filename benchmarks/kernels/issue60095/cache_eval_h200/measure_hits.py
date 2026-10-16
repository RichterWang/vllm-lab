#!/usr/bin/env python3
"""Repeat persistent split-cache loads without changing the vLLM source."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--binary-root", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    destination = args.result_dir / "hit_repeats.json"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    sys.path.insert(0, str(args.vllm_root.resolve()))
    vllm = importlib.import_module("vllm")
    vllm.__path__.append(str((args.binary_root.resolve() / "vllm")))
    autotune = importlib.import_module("vllm.v1.worker.gpu.sample.logprob_autotune")
    if not Path(autotune.__file__).resolve().is_relative_to(args.vllm_root.resolve()):
        raise RuntimeError("wrong autotune source")
    rows = []
    for vocab in (131072, 151936, 262144):
        for requested_k in (1, 6, 32):
            for index in range(args.repetitions):
                started = time.perf_counter()
                result = autotune.initialize_split_autotune(
                    vocab, torch.bfloat16, requested_k, torch.device("cuda:0"),
                    args.result_dir / "selection_cache",
                )
                elapsed = time.perf_counter() - started
                if result["status"] != "cache_hit":
                    raise RuntimeError(f"expected hit for V={vocab}, K={requested_k}")
                rows.append({"vocab_size": vocab, "requested_k": requested_k,
                             "index": index, "wall_ms": elapsed * 1000,
                             "cache_path": result["cache_path"],
                             "calibration_k": result["calibration_num_logprobs"]})
    destination.write_text(json.dumps(rows, indent=2) + "\n")
    for vocab in (131072, 151936, 262144):
        for k in (1, 6, 32):
            values = [row["wall_ms"] for row in rows
                      if row["vocab_size"] == vocab and row["requested_k"] == k]
            print(vocab, k, "min/median/max ms:",
                  round(min(values), 2), round(statistics.median(values), 2),
                  round(max(values), 2), flush=True)


if __name__ == "__main__":
    main()
