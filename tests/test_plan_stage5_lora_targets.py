from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.plan_stage5_lora_targets import block_index, escaped_exact_pattern, filter_modules, summarize_setting


def _fake_modules() -> list[dict[str, object]]:
    return [
        {"name": "blocks.0.block.conv1", "type": "Conv2d", "lora_supported": True, "lora_param_count": 10, "base_param_count": 100},
        {"name": "blocks.1.block.conv4", "type": "Conv2d", "lora_supported": True, "lora_param_count": 20, "base_param_count": 200},
        {"name": "blocks.2.block.conv5", "type": "Conv2d", "lora_supported": True, "lora_param_count": 30, "base_param_count": 300},
        {"name": "blocks.2.block.conv2", "type": "Conv2d", "lora_supported": False, "lora_param_count": 0, "base_param_count": 40},
        {"name": "dec_net.cond_embed", "type": "Linear", "lora_supported": True, "lora_param_count": 7, "base_param_count": 70},
        {"name": "y_embedder.decoder.conv_out", "type": "Conv2d", "lora_supported": True, "lora_param_count": 9, "base_param_count": 90},
    ]


def test_block_index_and_escaped_exact_pattern() -> None:
    assert block_index("blocks.12.block.conv1") == 12
    assert block_index("dec_net.cond_embed") is None
    assert escaped_exact_pattern(["a.b", "c+d"]) == r"^(a\.b|c\+d)$"


def test_filter_denoiser_tail_excludes_unsupported_and_y_embedder() -> None:
    rows, rationale = filter_modules(_fake_modules(), preset="denoiser_tail", last_blocks=2)

    assert "avoids y_embedder" in rationale
    assert [row["name"] for row in rows] == [
        "blocks.1.block.conv4",
        "blocks.2.block.conv5",
    ]


def test_summarize_setting_counts_params_and_builds_regex() -> None:
    rows, rationale = filter_modules(_fake_modules(), preset="dec_net", last_blocks=2)
    setting = summarize_setting(rows, preset="dec_net", rationale=rationale, rank=4)

    assert setting["module_count"] == 1
    assert setting["lora_param_count"] == 7
    assert setting["target_types"] == ["linear"]
    assert setting["regex"] == r"^(dec_net\.cond_embed)$"


def test_cli_writes_lora_plan(tmp_path: Path) -> None:
    inspect_json = tmp_path / "inspect.json"
    output_json = tmp_path / "plan.json"
    inspect_json.write_text(
        json.dumps(
            {
                "checkpoint": "ckpt.pt",
                "config": "config.yaml",
                "lora_rank": 4,
                "lora_candidate_modules": _fake_modules(),
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/plan_stage5_lora_targets.py",
            "--inspect-json",
            str(inspect_json),
            "--preset",
            "denoiser_tail",
            "--preset",
            "dec_net",
            "--last-blocks",
            "2",
            "--output-json",
            str(output_json),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text())
    assert payload["settings"][0]["preset"] == "denoiser_tail"
    assert payload["settings"][0]["lora_param_count"] == 50
    assert payload["settings"][1]["regex"] == r"^(dec_net\.cond_embed)$"
