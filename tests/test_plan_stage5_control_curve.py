from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.plan_stage5_control_curve import parse_band, plan_curve_settings


def _row(setting: str, *, bpp: float, retained: float, residual: float = 0.0) -> dict[str, object]:
    return {
        "setting": setting,
        "control_bpp": bpp,
        "basis_retained_energy_fraction_mean": retained,
        "basis_residual_energy_fraction_mean": residual,
    }


def test_parse_band_defaults_to_one_candidate() -> None:
    band = parse_band("0.0001:0.001")

    assert band["low"] == 0.0001
    assert band["high"] == 0.001
    assert band["count"] == 1


def test_plan_curve_settings_selects_one_per_band_by_retained_per_bpp() -> None:
    rows = [
        _row("a", bpp=0.0002, retained=0.10),
        _row("b", bpp=0.0004, retained=0.50),
        _row("c", bpp=0.0012, retained=0.70),
        _row("d", bpp=0.0018, retained=0.80),
    ]

    selected, bands = plan_curve_settings(
        rows,
        bands=[parse_band("0:0.001:1"), parse_band("0.001:0.002:1")],
        rank_by="retained_per_bpp",
    )

    assert [row["setting"] for row in selected] == ["b", "c"]
    assert bands[0]["selected_count"] == 1
    assert selected[0]["curve_band"] == "control_bpp_0_0.001"


def test_cli_writes_sweep_compatible_settings_json(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    output = tmp_path / "curve_settings.json"
    settings.write_text(
        json.dumps(
            {
                "settings": [
                    {
                        "setting": "mode=basis,groups=16,grid=8,coeffs=8,bits=4,range=0.5,scale=1,codec=fixed_bits,quantizer=uniform,mu=16",
                        "control_bpp": 0.0002,
                        "codec": "fixed_bits",
                        "quantizer": "uniform",
                        "quantile": "p95",
                        "components": 8,
                        "candidate_components": 8,
                        "selection": "prefix",
                        "basis_retained_energy_fraction_mean": 0.60,
                        "basis_residual_energy_fraction_mean": 0.40,
                        "quantization_rmse": 0.05,
                        "clipped_fraction": 0.01,
                    },
                    {
                        "setting": "mode=basis,groups=16,grid=8,coeffs=16,bits=4,range=0.5,scale=1,codec=fixed_bits,quantizer=uniform,mu=16",
                        "control_bpp": 0.0012,
                        "codec": "fixed_bits",
                        "quantizer": "uniform",
                        "quantile": "p95",
                        "components": 16,
                        "candidate_components": 16,
                        "selection": "prefix",
                        "basis_retained_energy_fraction_mean": 0.80,
                        "basis_residual_energy_fraction_mean": 0.20,
                        "quantization_rmse": 0.06,
                        "clipped_fraction": 0.02,
                    },
                ]
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/plan_stage5_control_curve.py",
            "--settings-json",
            str(settings),
            "--band",
            "0:0.001:1",
            "--band",
            "0.001:0.002:1",
            "--rank-by",
            "retained_per_bpp",
            "--min-basis-retained-energy",
            "0.5",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    assert len(payload["settings"]) == 2
    assert payload["settings"][0]["curve_band"] == "control_bpp_0_0.001"
    assert payload["settings"][1]["curve_band"] == "control_bpp_0.001_0.002"
    assert "setting" in payload["settings"][0]


def test_cli_reads_sweep_plan_dict_settings_for_affine_curve(tmp_path: Path) -> None:
    settings = tmp_path / "affine_sweep_plan.json"
    output = tmp_path / "affine_curve_settings.json"
    settings.write_text(
        json.dumps(
            {
                "settings": [
                    {
                        "setting": {
                            "mode": "affine",
                            "groups": 8,
                            "grid": 1,
                            "bits": 4,
                            "value_range": 0.25,
                            "gain_range": 1.0,
                            "bias_range": 0.25,
                            "scale": 1.0,
                            "coeffs": 0,
                            "codec": "fixed_bits",
                            "huffman_key": "",
                            "quantizer": "uniform",
                            "mu": 16.0,
                            "selection": "prefix",
                            "candidate_components": 0,
                        },
                        "control_bpp": 0.000244140625,
                    },
                    {
                        "setting": {
                            "mode": "affine",
                            "groups": 16,
                            "grid": 2,
                            "bits": 4,
                            "value_range": 0.25,
                            "gain_range": 1.5,
                            "bias_range": 0.25,
                            "scale": 1.0,
                            "coeffs": 0,
                            "codec": "fixed_bits",
                            "huffman_key": "",
                            "quantizer": "mu_law",
                            "mu": 16.0,
                            "selection": "prefix",
                            "candidate_components": 0,
                        },
                        "control_bpp": 0.001953125,
                    },
                ]
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/plan_stage5_control_curve.py",
            "--settings-json",
            str(settings),
            "--band",
            "0:0.001:1",
            "--band",
            "0.001:0.003:1",
            "--rank-by",
            "control_bpp",
            "--include-mode",
            "affine",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    assert [row["mode"] for row in payload["settings"]] == ["affine", "affine"]
    assert payload["settings"][0]["setting"].startswith("mode=affine")
    assert "gain_range=1.5" in payload["settings"][1]["setting"]
    assert payload["filters"]["include_mode"] == ["affine"]
