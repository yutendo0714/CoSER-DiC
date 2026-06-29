from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.sweep_stage5_counted_control import build_eval_command, parse_setting, planned_control_bytes
from scripts.sweep_stage5_counted_control import load_setting_rows_from_json, load_setting_texts_from_json
from scripts.sweep_stage5_counted_control import override_setting_mode


def test_parse_dct_control_setting_and_bpp() -> None:
    setting = parse_setting("mode=dct,groups=8,grid=8,coeffs=4,bits=4,range=0.25,scale=0.75")

    assert setting.label == "dct_g8_s8_k4_b4_r0.25_sc0.75_uniform"
    assert setting.eval_mode == "condition_residual_dct"
    assert setting.control_bytes() == 16
    assert setting.control_bpp(512, 512) == 8 * 16 / (512 * 512)


def test_parse_grid_control_setting_and_command() -> None:
    setting = parse_setting("mode=grid,groups=4,grid=4,bits=4,range=0.25,scale=1")
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=["--wandb-mode", "offline"],
    )

    assert setting.eval_mode == "condition_residual_grid"
    assert setting.control_bytes() == 32
    assert "--counted-control-mode" in command
    assert "condition_residual_grid" in command
    assert "--control-dct-coeffs-per-group" in command
    assert command[-2:] == ["--wandb-mode", "offline"]


def test_parse_affine_control_setting_and_command() -> None:
    setting = parse_setting(
        "mode=affine,groups=8,grid=2,bits=4,gain_range=1.5,bias_range=0.25,scale=0.75,quantizer=mu_law,mu=32"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.eval_mode == "condition_residual_affine"
    assert setting.control_bytes() == 32
    assert "affine_g8_s2_b4_gr1.5_br0.25_sc0.75_mu_law" in setting.label
    assert "--counted-control-mode" in command
    assert "condition_residual_affine" in command
    assert "--control-affine-gain-range" in command
    assert "1.5" in command
    assert "--control-affine-bias-range" in command
    assert "0.25" in command


def test_parse_affine_dct_control_setting_and_command() -> None:
    setting = parse_setting(
        "mode=affine_dct,groups=8,grid=4,coeffs=4,bits=4,affine_groups=8,affine_grid=1,"
        "range=0.25,gain_range=1.0,bias_range=0.25,scale=1.0,quantizer=uniform"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.eval_mode == "condition_residual_affine_dct"
    assert setting.control_bytes() == 24
    assert "affinedct_ag8_as1_dg8_ds4_k4_b4" in setting.label
    assert "--counted-control-mode" in command
    assert "condition_residual_affine_dct" in command
    assert "--control-affine-groups" in command
    assert "8" in command
    assert "--control-affine-grid-size" in command
    assert "1" in command


def test_parse_affine_grid_control_setting_and_command() -> None:
    setting = parse_setting(
        "mode=affine_grid,groups=8,grid=2,bits=4,affine_groups=8,affine_grid=1,"
        "range=0.25,gain_range=1.0,bias_range=0.25,scale=1.0,quantizer=uniform"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.eval_mode == "condition_residual_affine_grid"
    assert setting.control_bytes() == 24
    assert "affinegrid_ag8_as1_gg8_gs2_b4" in setting.label
    assert "--counted-control-mode" in command
    assert "condition_residual_affine_grid" in command
    assert "--control-affine-groups" in command
    assert "--control-affine-grid-size" in command


def test_parse_hybrid_affine_dct_grid_setting_and_command() -> None:
    setting = parse_setting(
        "mode=hybrid_affine_dct_grid,groups=8,grid=2,coeffs=4,bits=4,affine_groups=8,affine_grid=1,"
        "range=0.25,gain_range=1.0,bias_range=0.25,selector_bytes=1,objective=image_l1,rd_lambda=0.125,scale=1.0"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.eval_mode == "condition_residual_hybrid_affine_dct_grid"
    assert setting.hybrid_selection_objective == "image_l1"
    assert setting.control_bytes() == 25
    assert "hybrid_affinedctgrid_ag8_as1_g8_s2_k4_b4" in setting.label
    assert "--counted-control-mode" in command
    assert "condition_residual_hybrid_affine_dct_grid" in command
    assert "--control-hybrid-selector-bytes" in command
    assert "1" in command
    assert "--control-hybrid-rate-lambda" in command
    assert "0.125" in command
    assert "--control-hybrid-selection-objective" in command
    assert "image_l1" in command


def test_parse_hybrid_selector_bits_setting_and_command() -> None:
    setting = parse_setting(
        "mode=hybrid_affine_dct_grid,groups=1,grid=1,coeffs=1,bits=4,affine_groups=1,affine_grid=1,"
        "range=0.25,gain_range=1.0,bias_range=0.25,selector_bytes=1,selector_bits=4,"
        "objective=condition_l1,rd_lambda=0.0,scale=1.0"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.selector_bits == 4
    assert setting.control_bytes() == 3
    assert "selbits4" in setting.label
    assert "--control-hybrid-selector-bits" in command
    assert command[command.index("--control-hybrid-selector-bits") + 1] == "4"


def test_parse_hybrid_image_rdo_fidelity_guard_setting_and_command() -> None:
    setting = parse_setting(
        "mode=hybrid_affine_dct_grid,groups=8,grid=2,coeffs=4,bits=4,affine_groups=8,affine_grid=1,"
        "range=0.25,gain_range=1.0,bias_range=0.25,selector_bytes=1,selector_bits=2,"
        "objective=dists,rd_lambda=0.0,fidelity_lambda=10.0,fidelity_metric=image_mse,scale=1.0"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.hybrid_selection_objective == "dists"
    assert setting.hybrid_fidelity_lambda == 10.0
    assert setting.hybrid_fidelity_metric == "image_mse"
    assert "fimage_mse10" in setting.label
    assert command[command.index("--control-hybrid-fidelity-lambda") + 1] == "10.0"
    assert command[command.index("--control-hybrid-fidelity-metric") + 1] == "image_mse"


def test_parse_hybrid_affine_dct_grid_basis_setting_and_command() -> None:
    setting = parse_setting(
        "mode=hybrid_affine_dct_grid_basis,groups=16,grid=8,coeffs=8,bits=4,affine_groups=8,affine_grid=1,"
        "range=0.5,gain_range=1.0,bias_range=0.25,selector_bytes=1,objective=image_mse,rd_lambda=0.25,scale=1.0,"
        "codec=fixed_bits,quantizer=uniform"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="basis.pt",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.eval_mode == "condition_residual_hybrid_affine_dct_grid_basis"
    assert setting.hybrid_selection_objective == "image_mse"
    assert setting.control_bytes() == 521
    assert "hybridbasis_ag8_as1_g16_s8_k8_b4" in setting.label
    assert "--counted-control-mode" in command
    assert "condition_residual_hybrid_affine_dct_grid_basis" in command
    assert "--control-basis" in command
    assert "basis.pt" in command
    assert "--control-hybrid-selector-bytes" in command
    assert "--control-hybrid-rate-lambda" in command


def test_parse_affine_basis_control_setting_and_command() -> None:
    setting = parse_setting(
        "mode=affine_basis,groups=16,grid=8,coeffs=8,bits=4,affine_groups=8,affine_grid=1,"
        "range=0.5,gain_range=1.0,bias_range=0.25,scale=1.0,codec=fixed_bits,quantizer=uniform"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="basis.pt",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.eval_mode == "condition_residual_affine_basis"
    assert setting.control_bytes() == 12
    assert "affinebasis_ag8_as1_bg16_bs8_k8_b4" in setting.label
    assert "--counted-control-mode" in command
    assert "condition_residual_affine_basis" in command
    assert "--control-basis" in command
    assert "basis.pt" in command


def test_parse_affine_basis_component_range_setting_and_command() -> None:
    setting = parse_setting(
        "mode=affine_basis,groups=16,grid=8,coeffs=8,bits=4,affine_groups=8,affine_grid=1,"
        "range=0.5,gain_range=1.0,bias_range=0.25,scale=1.0,codec=fixed_bits,"
        "quantizer=uniform,basis_range_mode=component_p95,basis_range_floor=1e-5"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="basis.pt",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.basis_range_mode == "component_p95"
    assert setting.basis_range_floor == 1.0e-5
    assert setting.control_bytes() == 12
    assert "component_p95" in setting.label
    assert command[command.index("--control-basis-range-mode") + 1] == "component_p95"
    assert command[command.index("--control-basis-range-floor") + 1] == "1e-05"


def test_parse_affine_basis_component_codebook_setting_and_command() -> None:
    setting = parse_setting(
        "mode=affine_basis,groups=16,grid=8,coeffs=8,bits=4,affine_groups=8,affine_grid=1,"
        "range=0.5,gain_range=1.0,bias_range=0.25,scale=1.0,codec=fixed_bits,"
        "quantizer=uniform,basis_range_mode=component_codebook"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="basis.pt",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.basis_range_mode == "component_codebook"
    assert setting.control_bytes() == 12
    assert "component_codebook" in setting.label
    assert command[command.index("--control-basis-range-mode") + 1] == "component_codebook"


def test_parse_basis_setting_and_command() -> None:
    setting = parse_setting("mode=basis,groups=16,grid=8,coeffs=12,bits=4,range=0.5,scale=1,quantizer=mu_law,mu=16")
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="basis.pt",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.eval_mode == "condition_residual_basis"
    assert setting.quantizer == "mu_law"
    assert setting.control_bytes() == 6
    assert "condition_residual_basis" in command
    assert "--control-quantizer" in command
    assert "mu_law" in command
    assert "--control-basis" in command
    assert "basis.pt" in command
    assert "--control-basis-components" in command
    assert "12" in command
    assert "--control-basis-selection" in command
    assert "prefix" in command


def test_parse_vector_basis_setting_and_command() -> None:
    setting = parse_setting("mode=basis,groups=16,grid=8,coeffs=8,bits=8,range=0.5,scale=1,selection=vector")
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="basis.pt",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.selection == "vector"
    assert setting.control_bytes() == 1
    assert "_vector" in setting.label
    assert command[command.index("--control-basis-selection") + 1] == "vector"


def test_parse_sparse_topk_basis_setting_counts_index_payload() -> None:
    setting = parse_setting(
        "mode=basis,groups=16,grid=8,coeffs=8,candidates=64,bits=4,range=0.5,scale=1,selection=topk,quantizer=uniform"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="basis.pt",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.selection == "topk"
    assert setting.candidate_components == 64
    assert setting.control_bytes() == 10
    assert "--control-basis-selection" in command
    assert "topk" in command
    assert "--control-basis-candidate-components" in command
    assert "64" in command


def test_parse_prefix_topk_basis_setting_counts_compact_payload_and_command() -> None:
    setting = parse_setting(
        "mode=basis,groups=16,grid=8,coeffs=2,prefix=2,candidates=8,bits=4,range=0.5,scale=1,"
        "selection=prefix_topk,quantizer=uniform"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="basis.pt",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.selection == "prefix_topk"
    assert setting.prefix_components == 2
    assert setting.candidate_components == 8
    assert setting.control_bytes() == 3
    assert command[command.index("--control-basis-selection") + 1] == "prefix_topk"
    assert command[command.index("--control-basis-prefix-components") + 1] == "2"
    assert command[command.index("--control-basis-candidate-components") + 1] == "8"


def test_parse_sparse_topk_basis_component_range_setting_and_command() -> None:
    setting = parse_setting(
        "mode=basis,groups=16,grid=8,coeffs=8,candidates=64,bits=4,range=0.5,scale=1,"
        "selection=topk,quantizer=uniform,basis_range_mode=component_p99"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="basis.pt",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.selection == "topk"
    assert setting.basis_range_mode == "component_p99"
    assert setting.control_bytes() == 10
    assert command[command.index("--control-basis-range-mode") + 1] == "component_p99"


def test_parse_basis_huffman_setting_and_command() -> None:
    setting = parse_setting(
        "mode=basis,groups=16,grid=8,coeffs=12,bits=4,range=0.5,scale=1,codec=huffman,huffman_key=p95_b4_mulaw16,quantizer=mu_law,mu=16"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="basis.pt",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.codec == "huffman"
    assert setting.huffman_key == "p95_b4_mulaw16"
    assert setting.quantizer == "mu_law"
    assert (
        planned_control_bytes(
            setting,
            huffman_priors={"p95_b4_mulaw16": {"mean_payload_bytes": 4.5, "prefix_mean_payload_bytes": [1.0] * 12}},
            sparse_topk_priors={},
        )
        == 1.0
    )
    assert "--control-codec" in command
    assert "huffman" in command
    assert "--control-huffman-key" in command
    assert "p95_b4_mulaw16" in command


def test_parse_basis_codebook_huffman_setting_and_command() -> None:
    setting = parse_setting(
        "mode=basis,groups=16,grid=8,coeffs=8,bits=4,range=0.5,scale=1,"
        "codec=huffman,huffman_key=p95_b4_codebook,quantizer=uniform,basis_range_mode=component_codebook"
    )
    command = build_eval_command(
        python="python",
        checkpoint="ckpt.pt",
        manifest="manifest.jsonl",
        per_image_metrics="metrics.jsonl",
        control_basis="basis.pt",
        run_name="run",
        crop_size=512,
        limit=64,
        batch_size=4,
        num_workers=2,
        setting=setting,
        extra_args=[],
    )

    assert setting.codec == "huffman"
    assert setting.basis_range_mode == "component_codebook"
    assert (
        planned_control_bytes(
            setting,
            huffman_priors={"p95_b4_codebook": {"mean_payload_bytes": 3.5, "prefix_mean_payload_bytes": [1.0] * 8}},
            sparse_topk_priors={},
        )
        == 1.0
    )
    assert command[command.index("--control-basis-range-mode") + 1] == "component_codebook"
    assert command[command.index("--control-codec") + 1] == "huffman"


def test_parse_vector_basis_huffman_setting_uses_vector_prior_bytes() -> None:
    setting = parse_setting(
        "mode=basis,groups=16,grid=8,coeffs=8,bits=8,range=0.5,scale=1,"
        "selection=vector,codec=huffman,huffman_key=vq_k8_b8,quantizer=uniform"
    )

    assert (
        planned_control_bytes(
            setting,
            huffman_priors={"vq_k8_b8": {"huffman_mean_payload_bytes": 0.75}},
            sparse_topk_priors={},
        )
        == 0.75
    )


def test_parse_sparse_topk_huffman_setting_uses_sparse_prior_bytes() -> None:
    setting = parse_setting(
        "mode=basis,groups=16,grid=8,coeffs=8,candidates=64,bits=4,range=0.5,scale=1,selection=topk,codec=huffman,huffman_key=topk_c64_k8_p95_b4,quantizer=uniform"
    )

    assert (
        planned_control_bytes(
            setting,
            huffman_priors={},
            sparse_topk_priors={"topk_c64_k8_p95_b4": {"mean_payload_bytes": 7.25}},
        )
        == 7.25
    )


def test_parse_sparse_topk_codebook_huffman_setting_uses_sparse_prior_bytes() -> None:
    setting = parse_setting(
        "mode=basis,groups=16,grid=8,coeffs=8,candidates=64,bits=4,range=0.5,scale=1,"
        "selection=topk,codec=huffman,huffman_key=topk_c64_k8_p95_b4_codebook,"
        "quantizer=uniform,basis_range_mode=component_codebook"
    )

    assert setting.basis_range_mode == "component_codebook"
    assert (
        planned_control_bytes(
            setting,
            huffman_priors={},
            sparse_topk_priors={"topk_c64_k8_p95_b4_codebook": {"mean_payload_bytes": 6.5}},
        )
        == 6.5
    )


def test_parse_affine_basis_huffman_setting_adds_affine_bytes() -> None:
    setting = parse_setting(
        "mode=affine_basis,groups=16,grid=8,coeffs=12,bits=4,affine_groups=8,affine_grid=1,"
        "range=0.5,gain_range=1.0,bias_range=0.25,scale=1,codec=huffman,"
        "huffman_key=p95_b4_mulaw16,quantizer=mu_law,mu=16"
    )

    assert setting.eval_mode == "condition_residual_affine_basis"
    assert (
        planned_control_bytes(
            setting,
            huffman_priors={"p95_b4_mulaw16": {"mean_payload_bytes": 4.5, "prefix_mean_payload_bytes": [1.0] * 12}},
            sparse_topk_priors={},
        )
        == 9.0
    )


def test_parse_hybrid_basis_huffman_setting_uses_conservative_family_max() -> None:
    setting = parse_setting(
        "mode=hybrid_affine_dct_grid_basis,groups=16,grid=2,coeffs=4,bits=4,affine_groups=8,affine_grid=1,"
        "range=0.5,gain_range=1.0,bias_range=0.25,selector_bytes=1,rd_lambda=0.0,scale=1,"
        "codec=huffman,huffman_key=p95_b4_mulaw16,quantizer=mu_law,mu=16"
    )

    assert setting.eval_mode == "condition_residual_hybrid_affine_dct_grid_basis"
    assert (
        planned_control_bytes(
            setting,
            huffman_priors={"p95_b4_mulaw16": {"mean_payload_bytes": 4.5, "prefix_mean_payload_bytes": [1.0] * 4}},
            sparse_topk_priors={},
        )
        == 41.0
    )


def test_override_basis_setting_to_affine_basis() -> None:
    row = {
        "setting": "mode=basis,groups=16,grid=8,coeffs=8,bits=4,range=0.5,scale=1,codec=fixed_bits,quantizer=uniform",
        "control_bpp": 0.000122,
        "basis_retained_energy_fraction_mean": 0.8,
    }

    updated = override_setting_mode(
        row,
        mode="affine_basis",
        affine_groups=8,
        affine_grid=1,
        gain_range=1.0,
        bias_range=0.25,
    )

    setting = parse_setting(str(updated["setting"]))
    assert setting.mode == "affine_basis"
    assert setting.effective_affine_groups == 8
    assert setting.effective_affine_grid == 1
    assert "mode=affine_basis" in str(updated["setting"])
    assert updated["basis_retained_energy_fraction_mean"] == 0.8


def test_override_basis_setting_to_hybrid_basis() -> None:
    row = {
        "setting": "mode=basis,groups=16,grid=8,coeffs=8,bits=4,range=0.5,scale=1,codec=fixed_bits,quantizer=uniform",
        "basis_retained_energy_fraction_mean": 0.8,
    }

    updated = override_setting_mode(
        row,
        mode="hybrid_affine_dct_grid_basis",
        affine_groups=8,
        affine_grid=1,
        gain_range=1.0,
        bias_range=0.25,
        selector_bytes=1,
        hybrid_selection_objective="image_l1",
        hybrid_rate_lambda=0.5,
    )

    setting = parse_setting(str(updated["setting"]))
    assert setting.mode == "hybrid_affine_dct_grid_basis"
    assert setting.eval_mode == "condition_residual_hybrid_affine_dct_grid_basis"
    assert setting.hybrid_selection_objective == "image_l1"
    assert setting.hybrid_rate_lambda == 0.5
    assert setting.effective_affine_groups == 8
    assert setting.effective_affine_grid == 1
    assert "mode=hybrid_affine_dct_grid_basis" in str(updated["setting"])
    assert updated["basis_retained_energy_fraction_mean"] == 0.8


def test_parse_setting_rejects_dct_without_coeffs() -> None:
    try:
        parse_setting("mode=dct,groups=8,grid=8,bits=4")
    except ValueError as exc:
        assert "coeffs" in str(exc)
    else:
        raise AssertionError("dct setting without coeffs should fail")


def test_load_setting_texts_from_inspect_json_filters(tmp_path: Path) -> None:
    payload = {
        "settings": [
            {
                "setting": "mode=basis,groups=16,grid=8,coeffs=8,bits=4,range=0.5,scale=1,codec=fixed_bits,quantizer=uniform,mu=16",
                "control_bpp": 0.000122,
                "codec": "fixed_bits",
                "quantizer": "uniform",
                "quantile": "p95",
                "components": 8,
                "candidate_components": 8,
                "selection": "prefix",
                "basis_retained_energy_fraction_mean": 0.7,
                "basis_residual_energy_fraction_mean": 0.3,
                "quantization_rmse": 0.05,
                "clipped_fraction": 0.01,
            },
            {
                "setting": "mode=basis,groups=16,grid=8,coeffs=32,bits=4,range=0.5,scale=1,codec=huffman,quantizer=mu_law,mu=16,huffman_key=p99_b4_mulaw16",
                "control_bpp": 0.004,
                "codec": "huffman",
                "quantizer": "mu_law",
                "quantile": "p99",
                "components": 32,
                "candidate_components": 32,
                "selection": "prefix",
                "basis_retained_energy_fraction_mean": 0.9,
                "basis_residual_energy_fraction_mean": 0.1,
                "quantization_rmse": 0.20,
                "clipped_fraction": 0.20,
            },
        ]
    }
    path = tmp_path / "recommended_basis_settings.json"
    path.write_text(json.dumps(payload))

    rows = load_setting_texts_from_json(
        [path],
        max_control_bpp=0.001,
        include_codecs={"fixed_bits"},
        include_quantizers={"uniform"},
        include_quantiles={"p95"},
        include_components={8},
        include_candidate_components={8},
        include_selections={"prefix"},
        min_basis_retained_energy=0.6,
        max_basis_residual_energy=0.4,
        max_quantization_rmse=0.1,
        max_clipped_fraction=0.05,
    )

    assert rows == [payload["settings"][0]["setting"]]

    row_payloads = load_setting_rows_from_json(
        [path],
        max_control_bpp=0.001,
        include_codecs={"fixed_bits"},
        include_components={8},
        min_basis_retained_energy=0.6,
        max_basis_residual_energy=0.4,
        sort_by="retained_per_bpp",
    )

    assert row_payloads[0]["basis_retained_energy_fraction_mean"] == 0.7


def test_load_setting_rows_sort_and_filter_basis_diagnostics(tmp_path: Path) -> None:
    payload = {
        "settings": [
            {
                "setting": "mode=basis,groups=16,grid=8,coeffs=8,bits=4,range=0.5,scale=1,codec=fixed_bits,quantizer=uniform,mu=16",
                "control_bpp": 0.002,
                "codec": "fixed_bits",
                "quantizer": "uniform",
                "quantile": "p95",
                "components": 8,
                "candidate_components": 8,
                "selection": "prefix",
                "basis_retained_energy_fraction_mean": 0.8,
                "basis_residual_energy_fraction_mean": 0.2,
                "quantization_rmse": 0.03,
                "clipped_fraction": 0.01,
            },
            {
                "setting": "mode=basis,groups=16,grid=8,coeffs=16,bits=4,range=0.5,scale=1,codec=fixed_bits,quantizer=uniform,mu=16",
                "control_bpp": 0.001,
                "codec": "fixed_bits",
                "quantizer": "uniform",
                "quantile": "p95",
                "components": 16,
                "candidate_components": 16,
                "selection": "prefix",
                "basis_retained_energy_fraction_mean": 0.7,
                "basis_residual_energy_fraction_mean": 0.3,
                "quantization_rmse": 0.04,
                "clipped_fraction": 0.02,
            },
            {
                "setting": "mode=basis,groups=16,grid=8,coeffs=32,bits=4,range=0.5,scale=1,codec=fixed_bits,quantizer=uniform,mu=16",
                "control_bpp": 0.0005,
                "codec": "fixed_bits",
                "quantizer": "uniform",
                "quantile": "p95",
                "components": 32,
                "candidate_components": 32,
                "selection": "prefix",
                "basis_retained_energy_fraction_mean": 0.3,
                "basis_residual_energy_fraction_mean": 0.7,
                "quantization_rmse": 0.05,
                "clipped_fraction": 0.01,
            },
        ]
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(payload))

    rows = load_setting_rows_from_json(
        [path],
        min_basis_retained_energy=0.6,
        max_basis_residual_energy=0.4,
        max_quantization_rmse=0.05,
        max_clipped_fraction=0.05,
        sort_by="retained_per_bpp",
        max_settings=1,
    )

    assert len(rows) == 1
    assert rows[0]["components"] == 16


def test_load_setting_rows_roundtrips_sweep_plan_dict_settings(tmp_path: Path) -> None:
    payload = {
        "settings": [
            {
                "setting": {
                    "mode": "affine",
                    "groups": 8,
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
                "control_bpp": 0.0009765625,
                "source_setting_metadata": {
                    "counted_control_mode": "condition_residual_affine",
                    "control_grid_size": 2,
                    "control_groups": 8,
                },
            }
        ]
    }
    path = tmp_path / "sweep_plan.json"
    path.write_text(json.dumps(payload))

    rows = load_setting_rows_from_json(
        [path],
        include_modes={"affine"},
        include_quantizers={"mu_law"},
        sort_by="control_bpp",
    )

    assert len(rows) == 1
    assert rows[0]["setting"].startswith("mode=affine")
    assert "gain_range=1.5" in rows[0]["setting"]
    assert rows[0]["mode"] == "affine"
    assert rows[0]["quantizer"] == "mu_law"
    assert rows[0]["control_groups"] == 8


def test_cli_writes_output_shell_with_gpu_preflight_for_affine(tmp_path: Path) -> None:
    output = tmp_path / "affine_plan.json"
    shell = tmp_path / "affine_plan.sh"

    subprocess.run(
        [
            sys.executable,
            "scripts/sweep_stage5_counted_control.py",
            "--checkpoint",
            "ckpt.pt",
            "--manifest",
            "manifest.jsonl",
            "--per-image-metrics",
            "metrics.jsonl",
            "--run-prefix",
            "stage5_affine",
            "--setting",
            "mode=affine,groups=8,grid=1,bits=4,gain_range=1.0,bias_range=0.25,scale=1",
            "--dry-run",
            "--output-json",
            str(output),
            "--output-sh",
            str(shell),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    row = payload["settings"][0]
    script_text = shell.read_text()
    assert row["setting"]["mode"] == "affine"
    assert row["control_bytes"] == 8.0
    assert row["source_setting_metadata"]["counted_control_mode"] == "condition_residual_affine"
    assert "check_gpu_ready.py" in script_text
    assert "--counted-control-mode condition_residual_affine" in script_text


def test_cli_replans_settings_json_without_stale_command_metadata(tmp_path: Path) -> None:
    settings = tmp_path / "curve_settings.json"
    output = tmp_path / "replanned.json"
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
                        "command": ["old", "command"],
                        "summary": "old/summary.json",
                        "run_name": "old_run",
                        "control_bpp": 0.000244140625,
                        "curve_band": "control_bpp_0_0.0005",
                    }
                ]
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/sweep_stage5_counted_control.py",
            "--checkpoint",
            "ckpt.pt",
            "--manifest",
            "manifest.jsonl",
            "--per-image-metrics",
            "metrics.jsonl",
            "--run-prefix",
            "stage5_replan",
            "--settings-json",
            str(settings),
            "--dry-run",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    row = json.loads(output.read_text())["settings"][0]
    metadata = row["source_setting_metadata"]
    assert metadata["curve_band"] == "control_bpp_0_0.0005"
    assert "command" not in metadata
    assert "summary" not in metadata
    assert row["run_name"].startswith("stage5_replan_affine")
