#!/usr/bin/env python3
"""Repeat K=6 calibration with warm JIT and fresh selection caches."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--binary-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.result_dir / "recalibration.json"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    sys.path.insert(0, str(args.vllm_root.resolve()))
    vllm = importlib.import_module("vllm")
    vllm.__path__.append(str(args.binary_root.resolve() / "vllm"))
    autotune = importlib.import_module("vllm.v1.worker.gpu.sample.logprob_autotune")
    if not Path(autotune.__file__).resolve().is_relative_to(args.vllm_root.resolve()):
        raise RuntimeError("wrong autotune source")

    rows = []
    for run in range(1, 4):
        for vocab in (131072, 151936, 262144):
            cache = args.result_dir / "calibration_repeats" / f"r{run}_v{vocab}"
            cache.mkdir(parents=True, exist_ok=False)
            started = time.perf_counter()
            result = autotune.initialize_split_autotune(
                vocab, torch.bfloat16, 6, torch.device("cuda:0"), cache
            )
            elapsed = time.perf_counter() - started
            if result["status"] != "calibrated":
                raise RuntimeError(f"not a fresh calibration: {cache}")
            row = {"run": run, "vocab_size": vocab, "wall_s": elapsed,
                   "choices": {r["batch_size"]: r["chosen_splits"] for r in result["rows"]},
                   "fallback": {r["batch_size"]: r["fallback_splits"] for r in result["rows"]},
                   "cache_path": result["cache_path"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
            destination.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
