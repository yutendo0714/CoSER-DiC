from __future__ import annotations

import json
from pathlib import Path

from scripts.select_stage5_control_candidates import (
    add_anchor_deltas,
    add_scores,
    add_reference_curve_gaps,
    anchor_guard_pass,
    anchor_count_mismatch_warnings,
    basis_guard_pass,
    interpolate_bpp_for_metric,
    interpolate_metric_at_bpp,
    iter_summary_entries_from_sweep_plan,
    mark_pareto,
    normalize_candidate,
    sort_rows,
    stage3_guard_pass,
)


def _summary(label: str, *, bpp: float, lpips: float, dists: float, psnr: float = 22.0) -> dict[str, object]:
    return {
        "run_name": label,
        "count": 64,
        "actual_payload_bpp_mean": bpp,
        "stage3_actual_payload_bpp_mean": 0.014,
        "control_payload_bpp_mean": max(bpp - 0.014, 0.0),
        "stage3_lpips_alex_mean": 0.58,
        "stage3_dists_mean": 0.35,
        "stage3_psnr_mean": 22.0,
        "stage3_ms_ssim_mean": 0.735,
        "stage4_lpips_alex_mean": lpips,
        "stage4_dists_mean": dists,
        "stage4_psnr_mean": psnr,
        "stage4_ms_ssim_mean": 0.73,
    }


def test_normalize_candidate_adds_stage3_deltas() -> None:
    row = normalize_candidate(_summary("a", bpp=0.015, lpips=0.50, dists=0.32))

    assert row["actual_payload_bpp"] == 0.015
    assert abs(float(row["control_payload_bpp"]) - 0.001) < 1.0e-12
    assert row["lpips_delta_vs_stage3"] < 0.0
    assert row["dists_delta_vs_stage3"] < 0.0


def test_pareto_and_ranking_prefers_guarded_perceptual_gain() -> None:
    rows = [
        normalize_candidate(_summary("dominated", bpp=0.016, lpips=0.52, dists=0.33)),
        normalize_candidate(_summary("better", bpp=0.015, lpips=0.50, dists=0.32)),
        normalize_candidate(_summary("unsafe", bpp=0.0145, lpips=0.45, dists=0.30, psnr=20.0)),
    ]

    mark_pareto(rows, metrics=["lpips", "dists", "psnr"], eps=1.0e-12)
    add_scores(
        rows,
        lpips_weight=1.0,
        dists_weight=1.0,
        psnr_guard_drop=0.75,
        ms_ssim_guard_drop=0.025,
        lpips_guard_increase=0.0,
        dists_guard_increase=0.0,
        min_basis_retained_energy=0.0,
        max_basis_residual_energy=1.0,
    )
    ranked = sort_rows(rows)

    assert next(row for row in rows if row["label"] == "dominated")["pareto"] is False
    assert stage3_guard_pass(rows[2], max_psnr_drop=0.75, max_ms_ssim_drop=0.025, max_lpips_increase=0.0, max_dists_increase=0.0) is False
    assert ranked[0]["label"] == "better"


def test_anchor_deltas_and_guard_require_improvement_without_fidelity_collapse() -> None:
    anchor = normalize_candidate(_summary("anchor", bpp=0.014, lpips=0.50, dists=0.32, psnr=22.0))
    better = normalize_candidate(_summary("better", bpp=0.014, lpips=0.47, dists=0.30, psnr=21.9))
    collapsed = normalize_candidate(_summary("collapsed", bpp=0.014, lpips=0.44, dists=0.28, psnr=20.5))
    rows = [anchor, better, collapsed]
    for row in rows:
        add_anchor_deltas(row, anchor)

    mark_pareto(rows, metrics=["lpips", "dists", "psnr"], eps=1.0e-12)
    add_scores(
        rows,
        lpips_weight=1.0,
        dists_weight=1.0,
        psnr_guard_drop=2.0,
        ms_ssim_guard_drop=0.1,
        lpips_guard_increase=0.0,
        dists_guard_increase=0.0,
        min_basis_retained_energy=0.0,
        max_basis_residual_energy=1.0,
        max_anchor_psnr_drop=0.25,
        max_anchor_ms_ssim_drop=0.02,
        max_anchor_lpips_increase=0.0,
        max_anchor_dists_increase=0.0,
    )
    ranked = sort_rows(rows)

    assert better["lpips_delta_vs_anchor"] < 0.0
    assert better["dists_delta_vs_anchor"] < 0.0
    assert anchor_guard_pass(
        better,
        max_psnr_drop=0.25,
        max_ms_ssim_drop=0.02,
        max_lpips_increase=0.0,
        max_dists_increase=0.0,
    ) is True
    assert collapsed["anchor_guard_pass"] is False
    assert ranked[0]["label"] == "better"


def test_anchor_count_mismatch_warnings_detect_split_mixing() -> None:
    anchor = normalize_candidate({**_summary("anchor_full552", bpp=0.014, lpips=0.50, dists=0.32), "count": 552})
    candidate = normalize_candidate({**_summary("candidate_limit64", bpp=0.014, lpips=0.47, dists=0.30), "count": 64})

    add_anchor_deltas(candidate, anchor)
    warnings = anchor_count_mismatch_warnings([candidate], anchor)

    assert candidate["anchor_count_match"] is False
    assert candidate["count_delta_vs_anchor"] == -488
    assert warnings[0]["count"] == 64
    assert warnings[0]["anchor_count"] == 552


def test_sweep_plan_missing_summaries_are_recorded(tmp_path: Path) -> None:
    from scripts.select_stage5_control_candidates import iter_summary_paths_from_sweep_plan

    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "settings": [
                    {"summary": str(tmp_path / "missing_a" / "summary.json")},
                    {"summary": str(tmp_path / "missing_b" / "summary.json")},
                ]
            }
        )
    )

    paths = iter_summary_paths_from_sweep_plan(plan)

    assert paths == [tmp_path / "missing_a" / "summary.json", tmp_path / "missing_b" / "summary.json"]


def test_sweep_plan_entries_preserve_basis_metadata(tmp_path: Path) -> None:
    summary = tmp_path / "run_a" / "summary.json"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "settings": [
                    {
                        "summary": str(summary),
                        "basis_reconstruction_key": "topk_c64_k8",
                        "basis_retained_energy_fraction_mean": 0.75,
                        "basis_residual_energy_fraction_mean": 0.25,
                        "source_setting_metadata": {"selection": "topk"},
                    }
                ]
            }
        )
    )

    entries = iter_summary_entries_from_sweep_plan(plan)

    assert entries[0][0] == summary
    assert entries[0][1]["basis_reconstruction_key"] == "topk_c64_k8"
    assert entries[0][1]["basis_retained_energy_fraction_mean"] == 0.75
    assert entries[0][1]["source_setting_metadata"] == {"selection": "topk"}


def test_sweep_plan_entries_preserve_affine_control_metadata(tmp_path: Path) -> None:
    summary = tmp_path / "run_affine" / "summary.json"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "settings": [
                    {
                        "summary": str(summary),
                        "source_setting_metadata": {
                            "counted_control_mode": "condition_residual_affine",
                            "control_grid_size": 2,
                            "control_groups": 8,
                            "control_affine_gain_range": 1.5,
                            "control_affine_bias_range": 0.25,
                            "control_scale": 0.75,
                            "control_codec_type": "fixed_bits",
                            "control_quantizer": "mu_law",
                        },
                    }
                ]
            }
        )
    )

    entries = iter_summary_entries_from_sweep_plan(plan)
    row = normalize_candidate(
        {
            **_summary("run_affine", bpp=0.015, lpips=0.49, dists=0.31),
            **entries[0][1],
        }
    )

    assert row["counted_control_mode"] == "condition_residual_affine"
    assert row["control_grid_size"] == 2
    assert row["control_groups"] == 8
    assert row["control_affine_gain_range"] == 1.5
    assert row["control_affine_bias_range"] == 0.25
    assert row["control_scale"] == 0.75
    assert row["control_codec_type"] == "fixed_bits"
    assert row["control_quantizer"] == "mu_law"


def test_sweep_plan_entries_preserve_lora_metadata(tmp_path: Path) -> None:
    summary = tmp_path / "run_lora" / "summary.json"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "settings": [
                    {
                        "summary": str(summary),
                        "preset": "denoiser_tail",
                        "module_count": 30,
                        "lora_param_count": 147456,
                        "base_param_count": 25000000,
                        "source_setting_metadata": {
                            "lora_preset": "denoiser_tail",
                            "lora_rank": 4,
                            "lora_lr": 2.0e-5,
                            "lora_param_count": 147456,
                            "lora_module_count": 30,
                        },
                    }
                ]
            }
        )
    )

    entries = iter_summary_entries_from_sweep_plan(plan)
    row = normalize_candidate(
        {
            **_summary("run_lora", bpp=0.014, lpips=0.50, dists=0.31),
            **entries[0][1],
        }
    )

    assert entries[0][1]["preset"] == "denoiser_tail"
    assert entries[0][1]["lora_param_count"] == 147456
    assert row["lora_preset"] == "denoiser_tail"
    assert row["lora_rank"] == 4
    assert row["lora_param_count"] == 147456
    assert row["lora_module_count"] == 30


def test_sweep_plan_entries_preserve_condition_guard_metadata(tmp_path: Path) -> None:
    summary = tmp_path / "run_guard" / "summary.json"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "settings": [
                    {
                        "summary": str(summary),
                        "condition_residual_guard": "rms_clip",
                        "condition_residual_guard_granularity": "channel",
                        "condition_residual_max_rms_ratio": 0.25,
                        "condition_residual_min_gate": 0.0,
                    }
                ]
            }
        )
    )

    entries = iter_summary_entries_from_sweep_plan(plan)
    row = normalize_candidate(
        {
            **_summary("run_guard", bpp=0.014, lpips=0.50, dists=0.31),
            **entries[0][1],
        }
    )

    assert entries[0][1]["condition_residual_guard"] == "rms_clip"
    assert row["condition_residual_guard"] == "rms_clip"
    assert row["condition_residual_guard_granularity"] == "channel"
    assert row["condition_residual_max_rms_ratio"] == 0.25


def test_promotion_plan_commands_infer_summary_paths(tmp_path: Path) -> None:
    plan = tmp_path / "promotion.json"
    plan.write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "source_label": "limit64_candidate",
                        "promoted_run_name": "candidate_full552",
                        "command": [
                            "python",
                            "scripts/eval_stage4_cod_lite_adapter.py",
                            "--run-name",
                            "candidate_full552",
                            "--output-dir",
                            "results/full552_eval",
                        ],
                    }
                ]
            }
        )
    )

    entries = iter_summary_entries_from_sweep_plan(plan)

    assert entries[0][0] == Path("results/full552_eval/candidate_full552/summary.json")
    assert entries[0][1]["promoted_run_name"] == "candidate_full552"
    assert entries[0][1]["source_label"] == "limit64_candidate"


def test_basis_guard_pass_uses_retained_and_residual_energy() -> None:
    row = {
        "basis_retained_energy_fraction_mean": 0.70,
        "basis_residual_energy_fraction_mean": 0.30,
    }

    assert basis_guard_pass(row, min_retained_energy=0.60, max_residual_energy=0.40) is True
    assert basis_guard_pass(row, min_retained_energy=0.80, max_residual_energy=0.40) is False
    assert basis_guard_pass(row, min_retained_energy=0.60, max_residual_energy=0.20) is False


def test_reference_curve_gap_and_single_point_proxy() -> None:
    curve = {
        "name": "ref",
        "points": [
            {"bpp": 0.01, "lpips": 0.30, "dists": 0.20},
            {"bpp": 0.02, "lpips": 0.20, "dists": 0.10},
        ],
    }
    row = normalize_candidate(_summary("candidate", bpp=0.015, lpips=0.28, dists=0.16))

    assert interpolate_metric_at_bpp(curve, "lpips", 0.015) == 0.25
    assert abs(float(interpolate_bpp_for_metric(curve, "lpips", 0.28)) - 0.012) < 1.0e-12

    add_reference_curve_gaps([row], curve, metrics=["lpips", "dists"])

    assert row["reference_lpips_at_bpp"] == 0.25
    assert row["lpips_gap_vs_reference_at_bpp"] > 0.0
    assert row["lpips_beats_reference_at_bpp"] is False
    assert row["lpips_single_point_rate_proxy_percent"] > 0.0
