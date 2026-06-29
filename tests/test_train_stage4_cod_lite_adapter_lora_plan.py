from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train_stage4_cod_lite_adapter import load_lora_plan_setting


def test_load_lora_plan_setting_selects_unique_preset(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "settings": [
                    {"preset": "denoiser_tail", "regex": "^tail$", "target_types": ["conv2d"]},
                    {"preset": "dec_net", "regex": "^dec$", "target_types": ["linear"]},
                ]
            }
        )
    )

    setting = load_lora_plan_setting(path, "dec_net")

    assert setting["regex"] == "^dec$"
    assert setting["target_types"] == ["linear"]


def test_load_lora_plan_setting_rejects_missing_or_duplicate_preset(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"settings": [{"preset": "a"}, {"preset": "a"}]}))

    with pytest.raises(ValueError, match="matched 0"):
        load_lora_plan_setting(path, "missing")
    with pytest.raises(ValueError, match="matched 2"):
        load_lora_plan_setting(path, "a")
