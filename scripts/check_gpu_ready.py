from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any


def torch_cuda_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment
        return {
            "ok": False,
            "available": False,
            "device_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    status: dict[str, Any] = {
        "ok": True,
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "devices": [],
    }
    for index in range(status["device_count"]):
        try:
            name = torch.cuda.get_device_name(index)
        except Exception as exc:  # pragma: no cover - depends on local driver state
            name = f"<unavailable: {type(exc).__name__}: {exc}>"
        status["devices"].append({"index": index, "name": name})
    return status


def nvidia_smi_status(timeout: float = 10.0) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {"ok": False, "error": "nvidia-smi not found", "command": command}
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": f"timeout after {exc.timeout}s", "command": command}

    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "command": command,
    }


def evaluate_gpu_readiness(
    *,
    torch_status: dict[str, Any],
    smi_status: dict[str, Any] | None,
    min_devices: int,
    require_nvidia_smi: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not torch_status.get("ok", False):
        reasons.append(f"torch CUDA check failed: {torch_status.get('error', 'unknown error')}")
    if not torch_status.get("available", False):
        reasons.append("torch.cuda.is_available() is false")
    if int(torch_status.get("device_count", 0)) < min_devices:
        reasons.append(
            f"torch.cuda.device_count()={torch_status.get('device_count', 0)} < min_devices={min_devices}"
        )
    if require_nvidia_smi and not (smi_status or {}).get("ok", False):
        error = (
            (smi_status or {}).get("stderr")
            or (smi_status or {}).get("stdout")
            or (smi_status or {}).get("error")
            or "unknown error"
        )
        reasons.append(f"nvidia-smi check failed: {error}")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "min_devices": min_devices,
        "require_nvidia_smi": require_nvidia_smi,
        "torch": torch_status,
        "nvidia_smi": smi_status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail fast when CUDA/NVML is not healthy enough for CoSER-DiC GPU experiments."
    )
    parser.add_argument("--min-devices", type=int, default=1)
    parser.add_argument("--nvidia-smi-timeout", type=float, default=10.0)
    parser.add_argument("--skip-nvidia-smi", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_devices < 1:
        raise ValueError("--min-devices must be >= 1")

    smi = None if args.skip_nvidia_smi else nvidia_smi_status(timeout=args.nvidia_smi_timeout)
    status = evaluate_gpu_readiness(
        torch_status=torch_cuda_status(),
        smi_status=smi,
        min_devices=args.min_devices,
        require_nvidia_smi=not args.skip_nvidia_smi,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    if not status["ready"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
