from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.sweep_stage5_condition_residual_guard import build_eval_command, float_tag, run_name_for


def test_guard_float_tag_and_run_name() -> None:
    assert float_tag(0.25) == "0p25"
    assert run_name_for(prefix="probe", guard="none", granularity="global", ratio=0.0, min_gate=0.0) == (
        "probe_guard_none"
    )
    assert run_name_for(prefix="probe", guard="rms_clip", granularity="channel", ratio=0.5, min_gate=0.1) == (
        "probe_guard_channel_r0p5_ming0p1"
    )


def test_build_eval_command_includes_guard_flags() -> None:
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        run_name="run",
        output_dir="out",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        guard="rms_clip",
        granularity="spatial",
        ratio=0.35,
        min_gate=0.0,
        extra_args=["--wandb-mode", "offline"],
    )

    assert "--condition-residual-guard" in command
    assert "rms_clip" in command
    assert "--condition-residual-guard-granularity" in command
    assert "spatial" in command
    assert "--condition-residual-max-rms-ratio" in command
    assert "0.35" in command
    assert command[-2:] == ["--wandb-mode", "offline"]


def test_cli_writes_guard_sweep_plan(tmp_path: Path) -> None:
    output = tmp_path / "guard_sweep.json"
    script = tmp_path / "guard_sweep.sh"

    subprocess.run(
        [
            sys.executable,
            "scripts/sweep_stage5_condition_residual_guard.py",
            "--checkpoint",
            "ckpt.pt",
            "--manifest",
            "manifest.jsonl",
            "--per-image-metrics",
            "metrics.jsonl",
            "--run-prefix",
            "stage5_guard",
            "--ratio",
            "0.25",
            "--granularity",
            "global",
            "--dry-run",
            "--output-json",
            str(output),
            "--output-sh",
            str(script),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    assert len(payload["settings"]) == 2
    assert payload["settings"][0]["run_name"] == "stage5_guard_guard_none"
    assert payload["settings"][1]["condition_residual_guard"] == "rms_clip"
    assert payload["settings"][1]["condition_residual_max_rms_ratio"] == 0.25
    text = script.read_text()
    assert "check_gpu_ready.py" in text
    assert "eval_stage4_cod_lite_adapter.py" in text
    assert "--condition-residual-guard rms_clip" in text
