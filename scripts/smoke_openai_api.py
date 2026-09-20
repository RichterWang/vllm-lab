from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

LAB_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = LAB_ROOT / "src" / "vllm"
RESULT_PATH = LAB_ROOT / "results" / "openai_api.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SOURCE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def configured_base_url() -> str:
    if override := os.getenv("VLLM_BASE_URL"):
        return override.rstrip("/")
    host = os.getenv("LAB_HOST", "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{os.getenv('LAB_PORT', '8000')}"


def write_result(result: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(RESULT_PATH)


def wait_until_ready(client: httpx.Client, timeout_seconds: float = 180.0) -> int:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = client.get("/health")
            if response.is_success:
                return response.status_code
            last_error = RuntimeError(f"health returned HTTP {response.status_code}")
        except httpx.HTTPError as error:
            last_error = error
        time.sleep(0.25)
    raise TimeoutError(f"server did not become ready: {last_error}")


def run(started_at: str) -> dict[str, Any]:
    base_url = configured_base_url()
    model = os.getenv("SERVED_MODEL_NAME", "qwen3-0.6b")
    prompt = "用一句话说明 vLLM 的 KV cache 有什么作用。"

    started = time.perf_counter()
    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        health_status = wait_until_ready(client)

        models = client.get("/v1/models")
        models.raise_for_status()
        models_payload = models.json()

        version = client.get("/version")
        version.raise_for_status()
        version_payload = version.json()

        completion_started = time.perf_counter()
        completion = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        completion.raise_for_status()
        completion_seconds = time.perf_counter() - completion_started
        completion_payload = completion.json()

    text = completion_payload["choices"][0]["message"]["content"].strip()
    advertised_models = [item["id"] for item in models_payload["data"]]
    return {
        "status": "passed" if text and model in advertised_models else "failed",
        "acceptance_scope": "OpenAI API liveness; generated text is not scored",
        "started_at": started_at,
        "completed_at": utc_now(),
        "source_commit": source_commit(),
        "model_revision": os.environ["MODEL_REVISION"],
        "base_url": base_url,
        "served_model": model,
        "server_version": version_payload.get("version"),
        "runtime_config": {
            "dtype": os.environ["LAB_DTYPE"],
            "max_model_len": int(os.environ["LAB_MAX_MODEL_LEN"]),
            "gpu_memory_utilization": float(os.environ["LAB_GPU_MEMORY_UTILIZATION"]),
            "max_num_seqs": int(os.environ["LAB_MAX_NUM_SEQS"]),
            "enforce_eager": True,
            "generation_config": "vllm",
            "sampler_backend": "native",
        },
        "checks": {
            "health_http_status": health_status,
            "models_http_status": models.status_code,
            "version_http_status": version.status_code,
            "chat_http_status": completion.status_code,
        },
        "advertised_models": advertised_models,
        "prompt": prompt,
        "text": text,
        "usage": completion_payload.get("usage"),
        "finish_reason": completion_payload["choices"][0].get("finish_reason"),
        "completion_seconds": round(completion_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    started_at = utc_now()
    RESULT_PATH.unlink(missing_ok=True)
    try:
        result = run(started_at)
    except BaseException as error:
        write_result(
            {
                "status": "failed",
                "acceptance_scope": "OpenAI API liveness",
                "started_at": started_at,
                "completed_at": utc_now(),
                "base_url": configured_base_url(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise

    write_result(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit("OpenAI API smoke test failed")


if __name__ == "__main__":
    main()
