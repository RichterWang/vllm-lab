#!/usr/bin/env python3
"""Freeze 2,000 valid ShareGPT conversations for the scheduler experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    selected = []
    lengths = []
    for entry in source:
        turns = entry.get("conversations", [])
        if len(turns) < 2:
            continue
        prompt = turns[0].get("value")
        completion = turns[1].get("value")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            continue
        prompt_len = len(tokenizer(prompt).input_ids)
        output_len = len(tokenizer(completion).input_ids)
        if prompt_len < 4 or output_len < 4 or prompt_len > 1024:
            continue
        if prompt_len + output_len > 2048:
            continue
        selected.append({"conversations": turns[:2]})
        lengths.append({
            "request_index": len(selected) - 1,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_tokens": prompt_len,
            "target_output_tokens": output_len,
        })
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"Found only {len(selected)} valid entries")
    frozen = args.output_dir / f"sharegpt_{args.count}.json"
    frozen.write_text(json.dumps(selected, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "request_lengths.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(lengths[0]))
        writer.writeheader()
        writer.writerows(lengths)
    manifest = {
        "source_path": str(args.source),
        "source_sha256": sha256(args.source),
        "frozen_path": str(frozen),
        "frozen_sha256": sha256(frozen),
        "tokenizer_path": str(args.tokenizer),
        "count": len(selected),
        "input_tokens_min": min(row["input_tokens"] for row in lengths),
        "input_tokens_max": max(row["input_tokens"] for row in lengths),
        "target_output_tokens_min": min(row["target_output_tokens"] for row in lengths),
        "target_output_tokens_max": max(row["target_output_tokens"] for row in lengths),
    }
    (args.output_dir / "request_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
