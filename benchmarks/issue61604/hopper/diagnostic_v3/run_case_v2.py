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


def client_command(
    args: argparse.Namespace,
    requests: int,
    seed: int,
    output_dir: str | None = None,
    input_trace: str | None = None,
    output_tokens: int | None = None,
) -> list[str]:
    command = [
        args.python,
        str(Path(__file__).with_name("load_client.py")),
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--model",
        args.served_model_name,
        "--tokenizer-path",
        args.model_path,
        "--output-dir",
        output_dir or args.output_dir,
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
        str(args.output_tokens if output_tokens is None else output_tokens),
        "--prefix-families",
        str(args.prefix_families),
        "--churn-a-requests",
        str(args.churn_a_requests),
        "--arrival-interval-ms",
        str(args.arrival_interval_ms),
        "--seed",
        str(seed),
    ]
    if input_trace:
        command += ["--input-trace", input_trace]
    if args.logprobs is not None:
        command += ["--logprobs", str(args.logprobs)]
    if requests > 1 and args.cancel_index is not None:
        command += [
            "--cancel-index",
            str(args.cancel_index),
            "--cancel-after-tokens",
            str(args.cancel_after_tokens),
        ]
    return command


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
    env["ISSUE61604_SOURCE_ROOT"] = str(Path(args.worktree).resolve())
    binary_site = args.binary_site or args.parent_site
    env["ISSUE61604_BINARY_SITE"] = str(Path(binary_site).resolve())
    if args.diagnostic:
        env["ISSUE61604_DIAG_DIR"] = str((output / "diagnostic").resolve())
        env["ISSUE61604_DIAG_START_MANUAL"] = "1"
        env["ISSUE61604_DIAG_MODE"] = args.diag_mode
        env["ISSUE61604_DIAG_CONTROL_FILE"] = str(
            (output / ".diagnostics-started").resolve()
        )
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
                    "ISSUE61604_DIAG_START_MANUAL",
                    "ISSUE61604_DIAG_CONTROL_FILE",
                    "ISSUE61604_DIAG_MODE",
                    "ISSUE61604_SOURCE_ROOT",
                    "ISSUE61604_BINARY_SITE",
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
    ]
    command.append(
        "--enable-prefix-caching"
        if args.enable_prefix_caching
        else "--no-enable-prefix-caching"
    )
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

            if not args.input_trace:
                trace_path = output / "input-trace.jsonl.gz"
                prepare_command = client_command(
                    args,
                    args.requests,
                    args.seed,
                    output_dir=str(output),
                )
                prepare_command.append("--prepare-trace")
                with (output / "trace-prepare.log").open("wb") as stream:
                    prepare_result = await asyncio.to_thread(
                        subprocess.run,
                        prepare_command,
                        cwd=args.runtime_cwd,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if prepare_result.returncode != 0 or not trace_path.is_file():
                    raise RuntimeError(
                        "load client could not prepare the fixed input trace; "
                        f"exit code {prepare_result.returncode}"
                    )
                args.input_trace = str(trace_path)

            # Warm up the model separately so diagnostic counters and measured
            # service metrics cover only the fixed request trace below.
            warmup_dir = output / "warmup"
            warmup_dir.mkdir(exist_ok=True)
            warmup_command = client_command(
                # Warm the exact shared prefix used by the measured trace so
                # every version starts from the same cache state. Keep warmup
                # decode short; only prefix prefill/cache population matters.
                args,
                1,
                args.seed,
                output_dir=str(warmup_dir),
                input_trace=args.input_trace,
                output_tokens=min(8, args.output_tokens),
            )
            with (output / "warmup-client.log").open("wb") as stream:
                warmup_result = await asyncio.to_thread(
                    subprocess.run, warmup_command, cwd=args.runtime_cwd, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            if warmup_result.returncode != 0:
                raise RuntimeError(
                    f"warmup client failed with exit code {warmup_result.returncode}"
                )
            if args.diagnostic:
                # All instrumented Python processes watch this shared marker.
                # File activation avoids vLLM replacing or consuming signals.
                control_file = output / ".diagnostics-started"
                control_file.touch()
                (output / "diagnostic-start.json").write_text(
                    json.dumps(
                        {
                            "activation": "shared-file",
                            "path": str(control_file.resolve()),
                            "created_at_unix": time.time(),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                await asyncio.sleep(0.5)
            status, body = await asyncio.to_thread(fetch, f"{base_url}/metrics")
            (output / "metrics-before.txt").write_bytes(body if status == 200 else b"")
            if args.diagnostic and args.metric_interval_s > 0:
                metric_task = asyncio.create_task(
                    scrape_metrics(base_url, output, args.metric_interval_s)
                )

            main_command = client_command(
                args, args.requests, args.seed, input_trace=args.input_trace
            )
            with (output / "client.log").open("wb") as stream:
                client_result = await asyncio.to_thread(
                    subprocess.run, main_command, cwd=args.runtime_cwd, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            if client_result.returncode != 0:
                raise RuntimeError(f"load client failed with exit code {client_result.returncode}")
            if args.diagnostic:
                (output / ".diagnostics-started.stop").touch()
                await asyncio.sleep(1.0)
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

    if args.diagnostic:
        scheduler_files = list((output / "diagnostic").glob("scheduler-*.json"))
        schedule_calls = 0
        scheduler_records = []
        for scheduler_file in scheduler_files:
            try:
                record = json.loads(scheduler_file.read_text())
                scheduler_records.append(record)
                schedule_calls += int(record.get("schedule_calls", 0))
            except (OSError, ValueError, TypeError):
                continue
        manager_records = []
        for manager_file in (output / "diagnostic").glob("manager-*.json"):
            try:
                payload = json.loads(manager_file.read_text())
                manager_records.extend(payload.get("managers", []))
            except (OSError, ValueError, TypeError):
                continue
        manager_stats = [record.get("stats", {}) for record in manager_records]
        aggregate = {
            key: sum(int(stats.get(key, 0) or 0) for stats in manager_stats)
            for key in (
                "allocate_slots_calls",
                "allocate_slots_failures",
                "computed_lookup_calls",
                "computed_lookup_cpu_ns",
                "computed_lookup_wall_ns",
                "ordinary_lookup_calls",
                "ordinary_lookup_cpu_ns",
                "ordinary_lookup_wall_ns",
                "ordinary_block_map_probes",
                "ordinary_block_map_hits",
                "memo_lookup_calls",
                "memo_lookup_cpu_ns",
                "memo_lookup_wall_ns",
                "memo_validated_blocks",
                "memo_reuse_calls",
                "memo_reused_blocks",
                "memo_build_calls",
                "memo_build_cpu_ns",
                "memo_build_wall_ns",
                "memo_entries_created",
                "memo_build_success_existing",
                "memo_fallback_map_probes",
                "memo_fallback_map_hits",
                "memo_invalidation_checks",
                "memo_entries_removed_on_reset",
                "memo_entries_removed_on_allocation",
                "memo_entries_removed_on_free",
                "memo_entries_live",
                "memo_block_records_live",
                "memo_entry_truncation_events",
                "memo_block_records_removed_by_validation",
                "cache_mapping_deletions",
                "memo_integrity_checks",
                "memo_integrity_failures",
                "memo_reference_acquire_checks",
                "memo_reference_acquire_failures",
                "memo_reference_release_checks",
                "memo_reference_release_successes",
                "memo_pending_cleared_without_acquire",
                "failed_attempt_request_count",
                "repeated_failure_requests",
            )
        }
        timing_fields = (
            "schedule_cpu_ns",
            "schedule_wall_ns",
        )
        for key in timing_fields:
            aggregate[key] = sum(int(record.get(key, 0) or 0) for record in scheduler_records)
        aggregate["schedule_calls"] = schedule_calls
        aggregate["preempted_request_events"] = sum(
            int(record.get("preempted_request_events", 0) or 0)
            for record in scheduler_records
        )
        (output / "diagnostic-summary.json").write_text(
            json.dumps(
                {
                    "diag_mode": args.diag_mode,
                    "validation": {
                        "scheduler_observed": schedule_calls > 0,
                        "manager_observed": (
                            aggregate["computed_lookup_calls"] > 0
                            if args.diag_mode == "timing"
                            else aggregate["allocate_slots_calls"] > 0
                        ),
                        "base_failure_trigger_observed": (
                            None
                            if args.diag_mode == "timing"
                            else aggregate["allocate_slots_failures"] > 0
                        ),
                        "memo_path_observed": aggregate["memo_lookup_calls"] > 0,
                        "memo_integrity_failed": aggregate["memo_integrity_failures"] > 0,
                        "failure_trigger_is_required_for_efficacy_claim": True,
                    },
                    "aggregate_counters": aggregate,
                    "per_call_means_ns": {
                        f"{prefix}_{clock}": aggregate[f"{prefix}_{clock}_ns"] / calls
                        for prefix, calls in (
                            ("computed_lookup", aggregate["computed_lookup_calls"]),
                            ("ordinary_lookup", aggregate["ordinary_lookup_calls"]),
                            ("memo_lookup", aggregate["memo_lookup_calls"]),
                            ("memo_build", aggregate["memo_build_calls"]),
                            ("schedule", aggregate["schedule_calls"]),
                        )
                        for clock in ("cpu", "wall")
                        if calls and aggregate.get(f"{prefix}_{clock}_ns", 0)
                    },
                    "manager_records": len(manager_records),
                    "scheduler_records": len(scheduler_records),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if schedule_calls == 0:
            raise RuntimeError(
                "diagnostic did not record any Scheduler.schedule calls; "
                "preserve this run as invalid and do not use it for attribution"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--runtime-cwd", default="/tmp")
    parser.add_argument("--package-site", required=True)
    parser.add_argument("--parent-site", required=True)
    parser.add_argument("--binary-site")
    parser.add_argument("--torch-site", required=True)
    parser.add_argument("--audit-site", required=True)
    parser.add_argument("--cuda-home", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-trace")
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
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--startup-timeout-s", type=float, default=1800)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--diag-mode", choices=("count", "timing"), default="count")
    parser.add_argument("--logprobs", type=int)
    parser.add_argument("--cancel-index", type=int)
    parser.add_argument("--cancel-after-tokens", type=int, default=16)
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
