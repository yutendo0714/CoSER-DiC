from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.compute_bd_rate import RateMetricPoint, compute_bd_rate, load_points


def test_compute_bd_rate_reports_half_rate_as_minus_50_percent() -> None:
    reference = [
        RateMetricPoint(rate=0.10, metric=20.0),
        RateMetricPoint(rate=0.20, metric=25.0),
        RateMetricPoint(rate=0.40, metric=30.0),
    ]
    candidate = [
        RateMetricPoint(rate=0.05, metric=20.0),
        RateMetricPoint(rate=0.10, metric=25.0),
        RateMetricPoint(rate=0.20, metric=30.0),
    ]

    result = compute_bd_rate(reference, candidate, metric="psnr")

    assert result.status == "ok"
    assert result.bd_rate_percent == pytest.approx(-50.0, abs=1.0e-6)


def test_compute_bd_rate_refuses_one_point_candidate() -> None:
    reference = [
        RateMetricPoint(rate=0.10, metric=20.0),
        RateMetricPoint(rate=0.20, metric=25.0),
    ]
    candidate = [RateMetricPoint(rate=0.10, metric=22.0)]

    result = compute_bd_rate(reference, candidate, metric="lpips")

    assert result.status == "insufficient_points"
    assert result.bd_rate_percent is None


def test_compute_bd_rate_refuses_non_overlapping_metric_range() -> None:
    reference = [
        RateMetricPoint(rate=0.10, metric=20.0),
        RateMetricPoint(rate=0.20, metric=25.0),
    ]
    candidate = [
        RateMetricPoint(rate=0.10, metric=30.0),
        RateMetricPoint(rate=0.20, metric=35.0),
    ]

    result = compute_bd_rate(reference, candidate, metric="psnr")

    assert result.status == "no_metric_overlap"
    assert result.bd_rate_percent is None


def test_load_points_reads_stage4_summary_keys(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "actual_payload_bpp_mean": 0.014,
                "stage4_lpips_alex_mean": 0.43,
                "stage4_dists_mean": 0.30,
            }
        )
        + "\n"
    )

    points = load_points(path, metric="lpips")

    assert points == [RateMetricPoint(rate=0.014, metric=0.43, label="summary.json")]


def test_cli_writes_unavailable_status_for_single_point_curve(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(
        json.dumps({"points": [{"bpp": 0.1, "lpips": 0.5}, {"bpp": 0.2, "lpips": 0.4}]})
    )
    candidate.write_text(json.dumps({"actual_payload_bpp_mean": 0.15, "stage4_lpips_alex_mean": 0.45}))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_bd_rate.py",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--metric",
            "lpips",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["results"][0]["status"] == "insufficient_points"


def test_cli_promotion_threshold_fails_when_bd_rate_is_not_good_enough(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(
        json.dumps({"points": [{"bpp": 0.1, "lpips": 0.5}, {"bpp": 0.2, "lpips": 0.4}]})
    )
    candidate.write_text(
        json.dumps({"points": [{"bpp": 0.09, "lpips": 0.5}, {"bpp": 0.18, "lpips": 0.4}]})
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_bd_rate.py",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--metric",
            "lpips",
            "--max-bd-rate-percent",
            "-20",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 3
    payload = json.loads(result.stdout)
    assert payload["promotion"]["passed"] is False
    assert payload["results"][0]["bd_rate_percent"] == pytest.approx(-10.0)


def test_cli_promotion_threshold_passes_when_bd_rate_is_good_enough(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(
        json.dumps({"points": [{"bpp": 0.1, "lpips": 0.5}, {"bpp": 0.2, "lpips": 0.4}]})
    )
    candidate.write_text(
        json.dumps({"points": [{"bpp": 0.05, "lpips": 0.5}, {"bpp": 0.1, "lpips": 0.4}]})
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_bd_rate.py",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--metric",
            "lpips",
            "--max-bd-rate-percent",
            "-20",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["promotion"]["passed"] is True
    assert payload["results"][0]["bd_rate_percent"] == pytest.approx(-50.0)


def test_cli_reports_dataset_and_bpp_policy_warnings(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(
        json.dumps(
            {
                "name": "official_cod_lite_kodak512",
                "dataset": "Kodak512",
                "bpp_policy": "official nominal bpp",
                "points": [{"bpp": 0.1, "lpips": 0.5}, {"bpp": 0.2, "lpips": 0.4}],
            }
        )
    )
    candidate.write_text(
        json.dumps(
            {
                "name": "coserdic_full552",
                "dataset": "gencodec512_full552",
                "bpp_policy": "actual_payload_bpp / paper_bpp",
                "points": [{"bpp": 0.05, "lpips": 0.5}, {"bpp": 0.1, "lpips": 0.4}],
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_bd_rate.py",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--metric",
            "lpips",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    warning_kinds = {warning["kind"] for warning in payload["protocol"]["warnings"]}
    assert "dataset_mismatch" in warning_kinds
    assert "bpp_policy_mismatch" in warning_kinds
    assert payload["curve_metadata"]["reference"]["dataset"] == "Kodak512"
    assert payload["curve_metadata"]["candidate"]["dataset"] == "gencodec512_full552"


def test_cli_can_fail_on_dataset_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(
        json.dumps({"dataset": "Kodak512", "points": [{"bpp": 0.1, "lpips": 0.5}, {"bpp": 0.2, "lpips": 0.4}]})
    )
    candidate.write_text(
        json.dumps(
            {"dataset": "gencodec512_full552", "points": [{"bpp": 0.05, "lpips": 0.5}, {"bpp": 0.1, "lpips": 0.4}]}
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_bd_rate.py",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--metric",
            "lpips",
            "--fail-on-dataset-mismatch",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 4
    assert "dataset_mismatch" in result.stdout
