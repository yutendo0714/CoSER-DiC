from __future__ import annotations

from scripts.compare_stage3_gain_sweep import metric_win_count, semantic_improvement_count


def test_metric_win_count_uses_direction() -> None:
    baseline = {
        "a": {"stage3_psnr": 10.0, "stage3_lpips_alex": 0.5},
        "b": {"stage3_psnr": 20.0, "stage3_lpips_alex": 0.4},
    }
    candidate = {
        "a": {"stage3_psnr": 11.0, "stage3_lpips_alex": 0.6},
        "b": {"stage3_psnr": 19.0, "stage3_lpips_alex": 0.3},
    }

    assert metric_win_count(baseline, candidate, "stage3_psnr", "higher") == 1
    assert metric_win_count(baseline, candidate, "stage3_lpips_alex", "lower") == 1


def test_semantic_improvement_count_uses_delta_direction() -> None:
    rows = {
        "a": {"stage3_psnr_delta_vs_semantic_only": 0.1, "stage3_lpips_alex_delta_vs_semantic_only": -0.1},
        "b": {"stage3_psnr_delta_vs_semantic_only": -0.1, "stage3_lpips_alex_delta_vs_semantic_only": 0.1},
    }

    assert semantic_improvement_count(rows, "stage3_psnr", "higher") == 1
    assert semantic_improvement_count(rows, "stage3_lpips_alex", "lower") == 1
