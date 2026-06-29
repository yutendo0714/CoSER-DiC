from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.sweep_stage5_lora_backbone import lr_tag, plan_settings, run_name_for


def _plan() -> dict[str, object]:
    return {
        "settings": [
            {
                "preset": "denoiser_tail",
                "module_count": 2,
                "lora_param_count": 100,
                "base_param_count": 1000,
                "rationale": "tail",
            },
            {
                "preset": "y_decoder",
                "module_count": 1,
                "lora_param_count": 50,
                "base_param_count": 500,
                "rationale": "riskier",
            },
        ]
    }


def test_lr_tag_and_run_name_for() -> None:
    assert lr_tag(2e-5) == "2em05"
    assert run_name_for(prefix="run", preset="tail", rank=4, lora_lr=2e-5, max_steps=1000) == (
        "run_tail_r4_lr2em05_ft1000"
    )


def test_plan_settings_filters_y_embedder_and_param_count() -> None:
    rows = plan_settings(_plan(), presets=[], max_lora_params=80, include_y_embedder=False)
    assert rows == []

    rows = plan_settings(_plan(), presets=[], max_lora_params=120, include_y_embedder=False)
    assert [row["preset"] for row in rows] == ["denoiser_tail"]

    rows = plan_settings(_plan(), presets=[], max_lora_params=120, include_y_embedder=True)
    assert [row["preset"] for row in rows] == ["denoiser_tail", "y_decoder"]


def test_cli_writes_lora_train_eval_plan(tmp_path: Path) -> None:
    plan = tmp_path / "lora_plan.json"
    output = tmp_path / "sweep.json"
    script = tmp_path / "sweep.sh"
    plan.write_text(json.dumps(_plan()))

    subprocess.run(
        [
            sys.executable,
            "scripts/sweep_stage5_lora_backbone.py",
            "--lora-plan",
            str(plan),
            "--preset",
            "denoiser_tail",
            "--run-prefix",
            "stage5_lora",
            "--train-manifest",
            "train.jsonl",
            "--eval-manifest",
            "eval.jsonl",
            "--eval-per-image-metrics",
            "metrics.jsonl",
            "--init-checkpoint",
            "init.pt",
            "--max-steps",
            "123",
            "--condition-residual-rms-guard-weight",
            "0.05",
            "--condition-residual-rms-guard-ratio",
            "0.35",
            "--condition-residual-rms-guard-granularity",
            "channel",
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
    row = payload["settings"][0]
    assert row["preset"] == "denoiser_tail"
    assert row["run_name"] == "stage5_lora_denoiser_tail_r4_lr2em05_ft123"
    assert "--backbone-lora-plan" in row["train_command"]
    assert "--backbone-lora-preset" in row["train_command"]
    assert "--condition-residual-rms-guard-weight" in row["train_command"]
    assert "0.05" in row["train_command"]
    assert "--condition-residual-rms-guard-granularity" in row["train_command"]
    assert "channel" in row["train_command"]
    assert row["command"] == row["eval_command"]
    assert row["command_text"] == row["eval_command_text"]
    assert row["source_setting_metadata"]["lora_preset"] == "denoiser_tail"
    assert row["summary"].endswith("stage5_lora_denoiser_tail_r4_lr2em05_ft123_limit64_eval/summary.json")
    script_text = script.read_text()
    assert "check_gpu_ready.py" in script_text
    assert "train_stage4_cod_lite_adapter.py" in script_text
    assert "eval_stage4_cod_lite_adapter.py" in script_text
