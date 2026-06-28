from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.analyze_stage3_failure_modes import analyze_rows, pearson, semantic_delta_summary, worst_rows


def test_pearson_returns_none_for_constant_signal() -> None:
    assert pearson([1.0, 1.0], [0.0, 1.0]) is None


def test_semantic_delta_summary_respects_metric_direction() -> None:
    rows = [
        {"stage3_lpips_alex_delta_vs_semantic_only": -0.1, "stage3_psnr_delta_vs_semantic_only": 0.2},
        {"stage3_lpips_alex_delta_vs_semantic_only": 0.2, "stage3_psnr_delta_vs_semantic_only": -0.1},
    ]

    lpips = semantic_delta_summary(rows, "stage3_lpips_alex", "lower")
    psnr = semantic_delta_summary(rows, "stage3_psnr", "higher")

    assert lpips["improvement_count"] == 1
    assert lpips["regression_count"] == 1
    assert psnr["improvement_count"] == 1
    assert psnr["regression_count"] == 1


def test_worst_rows_for_lower_is_better_metric_sorts_high_values() -> None:
    rows = [
        {"index": 0, "path": "/dpl/kodak/kodim01.png", "stage3_lpips_alex": 0.5},
        {"index": 1, "path": "/dpl/kodak/kodim02.png", "stage3_lpips_alex": 0.8},
    ]

    selected = worst_rows(rows, "stage3_lpips_alex", "high", top_k=1)

    assert selected[0]["index"] == 1
    assert selected[0]["dataset"] == "kodak"


def test_analyze_rows_reports_driver_correlation() -> None:
    rows = [
        {
            "path": "/dpl/kodak/kodim01.png",
            "stage3_lpips_alex": 0.1,
            "stage3_lpips_alex_delta_vs_semantic_only": -0.1,
            "semantic_topk_hit_rate": 0.9,
        },
        {
            "path": "/dpl/kodak/kodim02.png",
            "stage3_lpips_alex": 0.2,
            "stage3_lpips_alex_delta_vs_semantic_only": -0.2,
            "semantic_topk_hit_rate": 0.8,
        },
        {
            "path": "/dpl/kodak/kodim03.png",
            "stage3_lpips_alex": 0.3,
            "stage3_lpips_alex_delta_vs_semantic_only": 0.1,
            "semantic_topk_hit_rate": 0.7,
        },
    ]

    analysis = analyze_rows(rows, top_k=2)

    corr = analysis["correlations"]["stage3_lpips_alex"]["semantic_topk_hit_rate"]
    assert corr["count"] == 3
    assert corr["pearson"] < 0.0
    assert len(analysis["quality"]["stage3_lpips_alex"]["worst_rows"]) == 2


def test_cli_quiet_writes_json_without_stdout(tmp_path: Path) -> None:
    input_path = tmp_path / "metrics.jsonl"
    output_path = tmp_path / "failure_modes.json"
    input_path.write_text(
        json.dumps(
            {
                "index": 0,
                "path": "/dpl/kodak/kodim01.png",
                "stage3_lpips_alex": 0.1,
                "stage3_lpips_alex_delta_vs_semantic_only": -0.1,
            }
        )
        + "\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_stage3_failure_modes.py",
            "--input",
            str(input_path),
            "--output-json",
            str(output_path),
            "--quiet",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""
    assert json.loads(output_path.read_text())["overall"]["count"] == 1
