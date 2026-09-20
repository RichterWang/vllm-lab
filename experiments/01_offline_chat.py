from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = LAB_ROOT / "src" / "vllm"
RESULT_PATH = LAB_ROOT / "results" / "offline_chat.json"


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


def gpu_snapshot() -> dict[str, str]:
    fields = "name,driver_version,memory.total,memory.used,memory.free,utilization.gpu"
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={os.environ['LAB_GPU_INDEX']}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line = next(line for line in completed.stdout.splitlines() if line.strip())
    values = [value.strip() for value in line.split(",")]
    return dict(zip(fields.split(","), values, strict=True))


def write_result(result: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(RESULT_PATH)


def run(started_at: str) -> dict[str, Any]:
    from vllm import LLM, SamplingParams, __version__ as vllm_version

    model = os.environ["MODEL_ID"]
    revision = os.environ["MODEL_REVISION"]
    prompt = "用两句话解释 vLLM 的 continuous batching，并说明它为何提高吞吐量。"

    initial_gpu = gpu_snapshot()
    load_started = time.perf_counter()
    llm = LLM(
        model=model,
        revision=revision,
        dtype=os.environ["LAB_DTYPE"],
        max_model_len=int(os.environ["LAB_MAX_MODEL_LEN"]),
        gpu_memory_utilization=float(os.environ["LAB_GPU_MEMORY_UTILIZATION"]),
        max_num_seqs=int(os.environ["LAB_MAX_NUM_SEQS"]),
        enforce_eager=True,
        generation_config="vllm",
        trust_remote_code=False,
        seed=0,
    )
    load_seconds = time.perf_counter() - load_started
    loaded_gpu = gpu_snapshot()

    generation_started = time.perf_counter()
    requests = llm.chat(
        [{"role": "user", "content": prompt}],
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=96,
            seed=0,
        ),
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    generation_seconds = time.perf_counter() - generation_started
    output = requests[0].outputs[0]
    generated_gpu = gpu_snapshot()

    return {
        "status": "passed" if output.text.strip() else "failed",
        "acceptance_scope": "runtime smoke; generated text is not scored for accuracy",
        "started_at": started_at,
        "completed_at": utc_now(),
        "source_commit": source_commit(),
        "model": model,
        "revision": revision,
        "vllm_version": vllm_version,
        "runtime_config": {
            "dtype": os.environ["LAB_DTYPE"],
            "max_model_len": int(os.environ["LAB_MAX_MODEL_LEN"]),
            "gpu_memory_utilization": float(os.environ["LAB_GPU_MEMORY_UTILIZATION"]),
            "max_num_seqs": int(os.environ["LAB_MAX_NUM_SEQS"]),
            "gpu_index": int(os.environ["LAB_GPU_INDEX"]),
            "enforce_eager": True,
            "generation_config": "vllm",
            "sampler_backend": "native",
        },
        "prompt": prompt,
        "text": output.text.strip(),
        "prompt_tokens": len(requests[0].prompt_token_ids or []),
        "completion_tokens": len(output.token_ids),
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "gpu_before": initial_gpu,
        "gpu_after_load": loaded_gpu,
        "gpu_after_generation": generated_gpu,
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
                "acceptance_scope": "runtime smoke",
                "started_at": started_at,
                "completed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise

    write_result(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit("Offline generation returned empty text")


if __name__ == "__main__":
    main()
