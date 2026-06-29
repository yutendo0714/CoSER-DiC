from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_collect_bd_curve_points_stage4_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    output = tmp_path / "curve.json"
    summary.write_text(
        json.dumps(
            {
                "actual_payload_bpp_mean": 0.014,
                "stage4_psnr_mean": 21.2,
                "stage4_ms_ssim_mean": 0.71,
                "stage4_lpips_alex_mean": 0.43,
                "stage4_dists_mean": 0.30,
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_bd_curve_points.py",
            "--input",
            f"point_a={summary}",
            "--name",
            "test_curve",
            "--dataset",
            "toy",
            "--bpp-policy",
            "toy payload policy",
            "--stage",
            "stage4",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    assert payload["name"] == "test_curve"
    assert payload["bpp_policy"] == "toy payload policy"
    assert payload["points"][0]["label"] == "point_a"
    assert payload["points"][0]["bpp"] == 0.014
    assert payload["points"][0]["lpips"] == 0.43


def test_collect_bd_curve_points_reads_promotion_json(tmp_path: Path) -> None:
    eval_dir = tmp_path / "evals"
    run_dir = eval_dir / "candidate_full552"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "actual_payload_bpp_mean": 0.0151,
                "stage4_psnr_mean": 21.8,
                "stage4_ms_ssim_mean": 0.73,
                "stage4_lpips_alex_mean": 0.36,
                "stage4_dists_mean": 0.24,
            }
        )
    )
    promotion = tmp_path / "promotion.json"
    promotion.write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "source_label": "candidate_limit64",
                        "promoted_run_name": "candidate_full552",
                        "command": [
                            sys.executable,
                            "scripts/eval_stage4_cod_lite_adapter.py",
                            "--run-name",
                            "candidate_full552",
                            "--output-dir",
                            str(eval_dir),
                        ],
                    }
                ]
            }
        )
    )
    output = tmp_path / "curve.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_bd_curve_points.py",
            "--promotion-json",
            str(promotion),
            "--name",
            "promotion_curve",
            "--dataset",
            "toy",
            "--stage",
            "stage4",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    assert payload["promotion_jsons"] == [str(promotion)]
    assert payload["points"][0]["label"] == "candidate_full552"
    assert payload["points"][0]["source"] == str(run_dir / "summary.json")
    assert payload["points"][0]["bpp"] == 0.0151
    assert payload["points"][0]["lpips"] == 0.36


def test_collect_bd_curve_points_merges_extra_fid_kid_json(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    output = tmp_path / "curve.json"
    metric_json = tmp_path / "patch_fid_kid.json"
    summary.write_text(
        json.dumps(
            {
                "actual_payload_bpp_mean": 0.014,
                "stage4_lpips_alex_mean": 0.43,
                "stage4_dists_mean": 0.30,
            }
        )
    )
    metric_json.write_text(
        json.dumps(
            {
                "metrics": {
                    "frechet_inception_distance": 88.5,
                    "kernel_inception_distance_mean": 0.021,
                    "kernel_inception_distance_std": 0.003,
                }
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_bd_curve_points.py",
            "--input",
            f"candidate={summary}",
            "--extra-metric-json",
            f"candidate={metric_json}",
            "--metric",
            "lpips",
            "--metric",
            "dists",
            "--metric",
            "fid",
            "--metric",
            "kid",
            "--stage",
            "stage4",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    point = payload["points"][0]
    assert point["fid"] == 88.5
    assert point["kid"] == 0.021
    assert point["extra_metric_sources"] == [str(metric_json)]


def test_collect_bd_curve_points_fails_on_missing_metric(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    output = tmp_path / "curve.json"
    summary.write_text(json.dumps({"actual_payload_bpp_mean": 0.014, "stage4_psnr_mean": 21.2}))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_bd_curve_points.py",
            "--input",
            str(summary),
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "missing metrics" in result.stderr


def test_collect_bd_curve_points_can_read_dataset_split_summary(tmp_path: Path) -> None:
    summary = tmp_path / "split_summary.json"
    output = tmp_path / "curve.json"
    summary.write_text(
        json.dumps(
            {
                "by_dataset": {
                    "kodak": {
                        "actual_payload_bpp": {"mean": 0.014},
                        "stage4_psnr": {"mean": 20.9},
                        "stage4_ms_ssim": {"mean": 0.70},
                        "stage4_lpips_alex": {"mean": 0.47},
                        "stage4_dists": {"mean": 0.32},
                    }
                }
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_bd_curve_points.py",
            "--input",
            f"kodak_point={summary}",
            "--stage",
            "stage4",
            "--dataset-key",
            "kodak",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    assert payload["dataset_key"] == "kodak"
    assert payload["points"][0]["dataset_key"] == "kodak"
    assert payload["points"][0]["lpips"] == 0.47
