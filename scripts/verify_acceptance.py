from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = LAB_ROOT / "results" / "acceptance.json"
REQUIRED_RESULTS = {
    "offline": LAB_ROOT / "results" / "offline_chat.json",
    "openai_api": LAB_ROOT / "results" / "openai_api.json",
    "concurrent": LAB_ROOT / "results" / "concurrent_batch.json",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_results() -> dict[str, dict[str, Any]]:
    return {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in REQUIRED_RESULTS.items()
    }


def port_is_open(host: str, port: int) -> bool:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.5)
        return client.connect_ex((connect_host, port)) == 0


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
    temporary_path = RESULT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(RESULT_PATH)


def main() -> None:
    RESULT_PATH.unlink(missing_ok=True)
    results = load_results()
    host = os.environ["LAB_HOST"]
    port = int(os.environ["LAB_PORT"])
    gpu = gpu_snapshot()
    checks = {
        "offline_passed": results["offline"].get("status") == "passed",
        "openai_api_passed": results["openai_api"].get("status") == "passed",
        "concurrent_passed": results["concurrent"].get("status") == "passed",
        "server_overlap_observed": results["concurrent"].get(
            "server_side_overlap_observed"
        )
        is True,
        "port_closed_after_shutdown": not port_is_open(host, port),
        "gpu_returned_to_idle": int(gpu["memory.used"]) <= 256,
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "acceptance_scope": (
            "runtime, API, observed server concurrency, and clean shutdown; "
            "not semantic quality or comparative performance"
        ),
        "completed_at": utc_now(),
        "checks": checks,
        "gpu_after_shutdown": gpu,
        "result_files": {
            label: str(path.relative_to(LAB_ROOT))
            for label, path in REQUIRED_RESULTS.items()
        },
    }
    write_result(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit("Acceptance verification failed")


if __name__ == "__main__":
    main()
