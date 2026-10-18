#!/usr/bin/env python3
"""Start one isolated vLLM service, warm it, replay one deterministic case."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def fetch(url: str, timeout: float = 3.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


async def wait_ready(base_url: str, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {process.returncode}")
        try:
            status, _ = await asyncio.to_thread(fetch, f"{base_url}/health", 3.0)
            if status == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(2)
    raise TimeoutError(f"vLLM did not become ready within {timeout_s}s")


async def scrape_metrics(base_url: str, output: Path, interval_s: float | None) -> None:
    if interval_s is None:
        status, body = await asyncio.to_thread(fetch, f"{base_url}/metrics")
        (output / "metrics-after.txt").write_bytes(body if status == 200 else b"")
        return
    samples = []
    while True:
        try:
            status, body = await asyncio.to_thread(fetch, f"{base_url}/metrics")
            if status == 200:
                samples.append(
                    {
                        "sampled_at_unix": time.time(),
                        "text": body.decode("utf-8", errors="replace"),
                    }
                )
        except Exception:
            pass
        await asyncio.sleep(interval_s)
        if (output / ".stop-metrics").exists():
            break
    with (output / "metrics-samples.jsonl").open("w") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, separators=(",", ":")) + "\n")


def client_command(args: argparse.Namespace, requests: int, seed: int) -> list[str]:
    return [
        args.python,
        str(Path(__file__).with_name("load_client.py")),
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--model",
        args.served_model_name,
        "--tokenizer-path",
        args.model_path,
        "--output-dir",
        args.output_dir,
        "--run-id",
        args.run_id if requests > 1 else f"{args.run_id}-warmup",
        "--implementation",
        args.implementation,
        "--scenario",
        args.scenario,
        "--mode",
        args.mode,
        "--requests",
        str(requests),
        "--concurrency",
        str(args.concurrency),
        "--prefix-tokens",
        str(args.prefix_tokens),
        "--suffix-tokens",
        str(args.suffix_tokens),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--output-tokens",
        str(args.output_tokens),
        "--prefix-families",
        str(args.prefix_families),
        "--churn-a-requests",
        str(args.churn_a_requests),
        "--arrival-interval-ms",
        str(args.arrival_interval_ms),
        "--seed",
        str(seed),
    ]


async def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "case.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n"
    )
    base_url = f"http://127.0.0.1:{args.port}"
    env = os.environ.copy()
    package_path = str(Path(args.package_site).resolve())
    runtime_paths = [
        str(Path(args.audit_site).resolve()),
        str(Path(args.torch_site).resolve()),
        package_path,
        str(Path(args.parent_site).resolve()),
    ]
    env["PYTHONPATH"] = os.pathsep.join(runtime_paths)
    env["CUDA_HOME"] = args.cuda_home
    env["VLLM_TARGET_DEVICE"] = "cuda"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    # This Pod's FlashInfer sampler cubins are unavailable, and the installed
    # nvcc/headers are mismatched. Use vLLM's fallback sampler in both arms so
    # sampling does not attempt an unsupported JIT build on the first request.
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["ISSUE61604_SOURCE_AUDIT"] = "1"
    if args.diagnostic:
        env["ISSUE61604_DIAG_DIR"] = str((output / "diagnostic").resolve())
        (output / "diagnostic").mkdir(exist_ok=True)

    (output / "runtime-env.json").write_text(
        json.dumps(
            {
                key: env.get(key)
                for key in (
                    "PYTHONPATH",
                    "CUDA_HOME",
                    "VLLM_TARGET_DEVICE",
                    "VLLM_USE_FLASHINFER_SAMPLER",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "TOKENIZERS_PARALLELISM",
                    "ISSUE61604_DIAG_DIR",
                )
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    command = [
        args.python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--block-size",
        str(args.block_size),
        "--stream-interval",
        "1",
        "--enable-prefix-caching",
    ]
    if args.num_gpu_blocks_override is not None:
        command += ["--num-gpu-blocks-override", str(args.num_gpu_blocks_override)]
    if args.mode == "sync":
        command.append("--no-async-scheduling")
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.trust_remote_code:
        command.append("--trust-remote-code")

    (output / "server-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", args.port))
    with (output / "server.log").open("wb") as server_log:
        process = subprocess.Popen(
            command,
            # Do not let the source checkout shadow the staged package site.
            # This is important for the hybrid runtime, whose native extension
            # modules are shared from an installed package directory.
            cwd=args.runtime_cwd,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        metric_task = None
        try:
            await wait_ready(base_url, process, args.startup_timeout_s)
            status, body = await asyncio.to_thread(fetch, f"{base_url}/v1/models")
            (output / "models.json").write_bytes(body)
            if status != 200:
                raise RuntimeError(f"/v1/models returned HTTP {status}")

            if args.diagnostic:
                (output / "metrics-before.txt").write_bytes(
                    (await asyncio.to_thread(fetch, f"{base_url}/metrics"))[1]
                )
                metric_task = asyncio.create_task(
                    scrape_metrics(base_url, output, args.metric_interval_s)
                )
            else:
                status, body = await asyncio.to_thread(fetch, f"{base_url}/metrics")
                (output / "metrics-before.txt").write_bytes(body if status == 200 else b"")

            main_command = client_command(args, args.requests, args.seed)
            main_command.append("--warmup")
            with (output / "client.log").open("wb") as stream:
                client_result = await asyncio.to_thread(
                    subprocess.run, main_command, cwd=args.runtime_cwd, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            if client_result.returncode != 0:
                raise RuntimeError(f"load client failed with exit code {client_result.returncode}")
            if not args.diagnostic:
                status, body = await asyncio.to_thread(fetch, f"{base_url}/metrics")
                (output / "metrics-after.txt").write_bytes(body if status == 200 else b"")
        finally:
            if metric_task:
                (output / ".stop-metrics").touch()
                await metric_task
            if process.poll() is None:
                # Let the API server shut down its EngineCore workers cleanly.
                # Signalling the whole process group at once can kill the
                # diagnostic worker before its atexit/SIGTERM handler writes
                # the manager counters.
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
            (output / "server-exit.json").write_text(
                json.dumps({"return_code": process.returncode}) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--runtime-cwd", default="/tmp")
    parser.add_argument("--package-site", required=True)
    parser.add_argument("--parent-site", required=True)
    parser.add_argument("--torch-site", required=True)
    parser.add_argument("--audit-site", required=True)
    parser.add_argument("--cuda-home", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--implementation", choices=("base", "patch"), required=True)
    parser.add_argument("--scenario", choices=("long-shared", "default-capacity", "short-unshared", "churn"), required=True)
    parser.add_argument("--mode", choices=("sync", "async-default"), required=True)
    parser.add_argument("--port", type=int, default=18995)
    parser.add_argument("--served-model-name", default="qwen3-0.6b")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=18432)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-gpu-blocks-override", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--startup-timeout-s", type=float, default=1800)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--metric-interval-s", type=float, default=5.0)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--prefix-tokens", type=int, default=16384)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--prefix-families", type=int, default=8)
    parser.add_argument("--churn-a-requests", type=int, default=32)
    parser.add_argument("--arrival-interval-ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=61604)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
