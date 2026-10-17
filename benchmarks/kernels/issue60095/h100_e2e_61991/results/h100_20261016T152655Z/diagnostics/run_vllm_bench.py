#!/usr/bin/env python3
"""Call vLLM's official serving benchmark API without the full CLI loader.

This uses the same ``vllm.benchmarks.serve.main`` entry point as
``vllm bench serve``. It exists because the root CLI eagerly imports every
vLLM command, which adds substantial startup time on the remote shared volume.
"""

import json

from vllm.benchmarks.serve import add_cli_args, main
from vllm.utils.argparse_utils import FlexibleArgumentParser


def main_cli() -> None:
    parser = FlexibleArgumentParser(description="vLLM serving benchmark")
    add_cli_args(parser)
    result = main(parser.parse_args())
    keys = (
        "completed",
        "failed",
        "request_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p50_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "p50_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
        "mean_e2el_ms",
        "p50_e2el_ms",
        "p95_e2el_ms",
        "p99_e2el_ms",
    )
    summary = {key: result[key] for key in keys if key in result}
    summary = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in summary.items()
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main_cli()
