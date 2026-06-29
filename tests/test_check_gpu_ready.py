from __future__ import annotations

from scripts.check_gpu_ready import evaluate_gpu_readiness


def test_evaluate_gpu_readiness_accepts_healthy_cuda_and_smi() -> None:
    status = evaluate_gpu_readiness(
        torch_status={"ok": True, "available": True, "device_count": 1},
        smi_status={"ok": True},
        min_devices=1,
        require_nvidia_smi=True,
    )

    assert status["ready"] is True
    assert status["reasons"] == []


def test_evaluate_gpu_readiness_rejects_nvml_failure() -> None:
    status = evaluate_gpu_readiness(
        torch_status={"ok": True, "available": True, "device_count": 1},
        smi_status={"ok": False, "stderr": "Failed to initialize NVML: Unknown Error"},
        min_devices=1,
        require_nvidia_smi=True,
    )

    assert status["ready"] is False
    assert "nvidia-smi check failed" in status["reasons"][0]


def test_evaluate_gpu_readiness_reports_nvidia_smi_stdout_failure() -> None:
    status = evaluate_gpu_readiness(
        torch_status={"ok": True, "available": True, "device_count": 1},
        smi_status={"ok": False, "stdout": "Failed to initialize NVML: Unknown Error"},
        min_devices=1,
        require_nvidia_smi=True,
    )

    assert status["ready"] is False
    assert "Failed to initialize NVML" in status["reasons"][0]


def test_evaluate_gpu_readiness_rejects_missing_cuda_devices() -> None:
    status = evaluate_gpu_readiness(
        torch_status={"ok": True, "available": False, "device_count": 0},
        smi_status=None,
        min_devices=1,
        require_nvidia_smi=False,
    )

    assert status["ready"] is False
    assert "torch.cuda.is_available() is false" in status["reasons"]
    assert any("device_count" in reason for reason in status["reasons"])
