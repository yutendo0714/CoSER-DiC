from __future__ import annotations

from pathlib import Path

import torch

from scripts.inspect_stage5_control_basis import build_basis_setting_rows, load_basis_summary


def test_inspect_stage5_control_basis_builds_quantile_settings(tmp_path: Path) -> None:
    basis_path = tmp_path / "control_basis.pt"
    torch.save(
        {
            "basis": torch.randn(4, 2, 3, 3),
            "mean": torch.zeros(2, 3, 3),
            "groups": 2,
            "grid_size": 3,
            "source": "unit",
            "coefficient_abs_quantiles": {"p95": 0.5, "p99": 0.75},
            "coefficient_component_abs_p95": [0.1, 0.2, 0.4, 0.8],
            "coefficient_component_abs_p99": [0.2, 0.3, 0.5, 1.0],
            "coefficient_abs_mean": 0.1,
            "coefficient_abs_std": 0.2,
            "coefficient_abs_max": 1.0,
            "control_huffman_priors": {
                "p95_b4": {
                    "mean_payload_bytes": 1.5,
                    "prefix_mean_payload_bytes": [1.0, 1.25, 1.5, 1.75],
                    "quantizer": "uniform",
                    "quantization_mae": 0.1,
                    "quantization_rmse": 0.2,
                    "clipped_fraction": 0.05,
                    "bits": 4,
                    "range": 0.5,
                    "huffman": {"codec": "static_control_huffman"},
                },
                "p95_b4_mulaw16": {
                    "mean_payload_bytes": 1.25,
                    "prefix_mean_payload_bytes": [0.75, 1.0, 1.25, 1.5],
                    "quantizer": "mu_law",
                    "mu": 16.0,
                    "quantization_mae": 0.05,
                    "quantization_rmse": 0.1,
                    "clipped_fraction": 0.05,
                    "bits": 4,
                    "range": 0.5,
                    "huffman": {"codec": "static_control_huffman"},
                },
                "p95_b4_codebook": {
                    "mean_payload_bytes": 1.1,
                    "prefix_mean_payload_bytes": [0.6, 0.9, 1.1, 1.3],
                    "quantizer": "uniform",
                    "quantizer_label": "component_codebook",
                    "basis_range_mode": "component_codebook",
                    "codebook_key": "lloyd_b4",
                    "quantization_mae": 0.03,
                    "quantization_rmse": 0.06,
                    "clipped_fraction": 0.0,
                    "bits": 4,
                    "range": 0.5,
                    "huffman": {"codec": "static_control_huffman"},
                }
            },
            "coefficient_component_codebooks": {
                "lloyd_b4": {
                    "codec": "component_codebook_control_fixed_bits",
                    "bits": 4,
                    "levels": 16,
                    "component_count": 4,
                    "fixed_bytes_per_image": 2,
                    "quantization_mae": 0.03,
                    "quantization_rmse": 0.06,
                    "codebooks": [[float(i) for i in range(16)] for _ in range(4)],
                }
            },
            "coefficient_vector_codebooks": {
                "vq_k2_b4": {
                    "codec": "vector_codebook_control_fixed_bits",
                    "selection": "vector",
                    "components": 2,
                    "bits": 4,
                    "levels": 16,
                    "fixed_bytes_per_image": 2,
                    "huffman_mean_payload_bytes": 1.25,
                    "quantization_mae": 0.04,
                    "quantization_rmse": 0.08,
                    "vectors": [[float(i), float(i + 1)] for i in range(16)],
                    "huffman": {"codec": "static_control_huffman"},
                }
            },
            "sparse_topk_control_priors": {
                "topk_c4_k2_p95_b4": {
                    "codec": "sparse_topk_static_huffman_prior",
                    "candidate_components": 4,
                    "selected_components": 2,
                    "mean_payload_bytes": 1.5,
                    "quantizer": "uniform",
                    "quantization_mae": 0.08,
                    "quantization_rmse": 0.12,
                    "clipped_fraction": 0.02,
                    "bits": 4,
                    "range": 0.5,
                    "index_huffman": {"codec": "static_control_huffman"},
                    "value_huffman": {"codec": "static_control_huffman"},
                },
                "topk_c4_k2_p95_b4_codebook": {
                    "codec": "sparse_topk_static_huffman_prior",
                    "candidate_components": 4,
                    "selected_components": 2,
                    "mean_payload_bytes": 1.2,
                    "quantizer": "uniform",
                    "quantizer_label": "component_codebook",
                    "basis_range_mode": "component_codebook",
                    "codebook_key": "lloyd_b4",
                    "quantization_mae": 0.025,
                    "quantization_rmse": 0.055,
                    "clipped_fraction": 0.0,
                    "bits": 4,
                    "range": 0.5,
                    "index_huffman": {"codec": "static_control_huffman"},
                    "value_huffman": {"codec": "static_control_huffman"},
                },
            },
            "basis_reconstruction_stats": {
                "prefix_k2": {
                    "retained_energy_fraction_mean": 0.50,
                    "retained_energy_fraction_p50": 0.55,
                    "residual_energy_fraction_mean": 0.50,
                    "residual_l2_mean": 0.75,
                },
                "topk_c4_k2": {
                    "retained_energy_fraction_mean": 0.80,
                    "retained_energy_fraction_p50": 0.82,
                    "residual_energy_fraction_mean": 0.20,
                    "residual_l2_mean": 0.25,
                },
                "prefixtopk_p2_c4_k2": {
                    "retained_energy_fraction_mean": 0.90,
                    "retained_energy_fraction_p50": 0.91,
                    "residual_energy_fraction_mean": 0.10,
                    "residual_l2_mean": 0.12,
                },
            },
        },
        basis_path,
    )

    summary = load_basis_summary(basis_path)
    rows = build_basis_setting_rows(
        summary,
        components=[2, 4],
        candidate_components=[],
        selections=["prefix"],
        quantiles=["p95"],
        bits=4,
        scale=1.0,
        height=512,
        width=512,
    )

    assert summary["basis_count"] == 4
    assert summary["groups"] == 2
    assert summary["grid_size"] == 3
    assert len(rows) == 2
    assert rows[0]["control_bytes"] == 1
    assert rows[1]["control_bytes"] == 2
    assert rows[0]["control_bpp"] == 8 / (512 * 512)
    assert rows[0]["basis_reconstruction_key"] == "prefix_k2"
    assert rows[0]["basis_retained_energy_fraction_mean"] == 0.50
    assert (
        rows[0]["setting"]
        == "mode=basis,groups=2,grid=3,coeffs=2,bits=4,range=0.5,scale=1,codec=fixed_bits,quantizer=uniform,mu=16,selection=prefix"
    )

    both_rows = build_basis_setting_rows(
        summary,
        components=[2],
        candidate_components=[],
        selections=["prefix"],
        quantiles=["p95"],
        bits=4,
        scale=1.0,
        height=512,
        width=512,
        codec="both",
        quantizers=["uniform", "mu_law"],
        mu=16.0,
    )

    assert [row["codec"] for row in both_rows] == ["fixed_bits", "huffman", "fixed_bits", "huffman"]
    assert both_rows[1]["control_bytes"] == 1.25
    assert both_rows[0]["quantization_mae"] == 0.1
    assert both_rows[1]["quantization_rmse"] == 0.2
    assert both_rows[1]["setting"].endswith("codec=huffman,quantizer=uniform,mu=16,selection=prefix,huffman_key=p95_b4")
    assert both_rows[3]["huffman_key"] == "p95_b4_mulaw16"
    assert both_rows[3]["control_bytes"] == 1.0

    component_rows = build_basis_setting_rows(
        summary,
        components=[2],
        candidate_components=[],
        selections=["prefix"],
        quantiles=["p95"],
        bits=4,
        scale=1.0,
        height=512,
        width=512,
        codec="fixed_bits",
        quantizers=["uniform"],
        range_modes=["global", "component_p95"],
    )

    assert len(component_rows) == 2
    assert component_rows[0]["basis_range_mode"] == "global"
    assert component_rows[1]["basis_range_mode"] == "component_p95"
    assert "basis_range_mode=component_p95" in component_rows[1]["setting"]
    assert component_rows[1]["control_bytes"] == component_rows[0]["control_bytes"]

    codebook_rows = build_basis_setting_rows(
        summary,
        components=[2],
        candidate_components=[],
        selections=["prefix"],
        quantiles=["p95"],
        bits=4,
        scale=1.0,
        height=512,
        width=512,
        codec="fixed_bits",
        quantizers=["uniform", "mu_law"],
        range_modes=["component_codebook"],
    )

    assert len(codebook_rows) == 1
    assert codebook_rows[0]["basis_range_mode"] == "component_codebook"
    assert codebook_rows[0]["quantizer"] == "uniform"
    assert codebook_rows[0]["quantization_mae"] == 0.03
    assert codebook_rows[0]["control_bytes"] == component_rows[0]["control_bytes"]
    assert "basis_range_mode=component_codebook" in codebook_rows[0]["setting"]

    codebook_huffman_rows = build_basis_setting_rows(
        summary,
        components=[2],
        candidate_components=[],
        selections=["prefix"],
        quantiles=["p95"],
        bits=4,
        scale=1.0,
        height=512,
        width=512,
        codec="huffman",
        quantizers=["uniform"],
        range_modes=["component_codebook"],
    )

    assert len(codebook_huffman_rows) == 1
    assert codebook_huffman_rows[0]["codec"] == "huffman"
    assert codebook_huffman_rows[0]["huffman_key"] == "p95_b4_codebook"
    assert codebook_huffman_rows[0]["control_bytes"] == 0.9
    assert "basis_range_mode=component_codebook" in codebook_huffman_rows[0]["setting"]

    vector_rows = build_basis_setting_rows(
        summary,
        components=[2],
        candidate_components=[],
        selections=["vector"],
        quantiles=["p95"],
        bits=4,
        scale=1.0,
        height=512,
        width=512,
        codec="both",
        quantizers=["uniform"],
    )

    assert len(vector_rows) == 2
    assert vector_rows[0]["selection"] == "vector"
    assert vector_rows[0]["codec"] == "fixed_bits"
    assert vector_rows[0]["control_bytes"] == 2
    assert vector_rows[0]["quantization_mae"] == 0.04
    assert "selection=vector" in vector_rows[0]["setting"]
    assert vector_rows[1]["codec"] == "huffman"
    assert vector_rows[1]["huffman_key"] == "vq_k2_b4"
    assert vector_rows[1]["control_bytes"] == 1.25

    sparse_rows = build_basis_setting_rows(
        summary,
        components=[2],
        candidate_components=[4],
        selections=["topk"],
        quantiles=["p95"],
        bits=4,
        scale=1.0,
        height=512,
        width=512,
        codec="both",
        quantizers=["uniform"],
    )

    assert len(sparse_rows) == 3
    assert sparse_rows[0]["selection"] == "topk"
    assert sparse_rows[0]["candidate_components"] == 2
    assert sparse_rows[0]["control_bytes"] == 2
    assert sparse_rows[1]["candidate_components"] == 4
    assert sparse_rows[1]["control_bytes"] == 2
    assert sparse_rows[1]["basis_reconstruction_key"] == "topk_c4_k2"
    assert sparse_rows[1]["basis_retained_energy_fraction_mean"] == 0.80
    assert "selection=topk,candidates=4" in sparse_rows[1]["setting"]
    assert sparse_rows[2]["codec"] == "huffman"
    assert sparse_rows[2]["huffman_key"] == "topk_c4_k2_p95_b4"
    assert sparse_rows[2]["control_bytes"] == 1.5

    prefix_topk_rows = build_basis_setting_rows(
        summary,
        components=[2],
        candidate_components=[4],
        selections=["prefix_topk"],
        quantiles=["p95"],
        bits=4,
        scale=1.0,
        height=512,
        width=512,
        codec="both",
        quantizers=["uniform"],
    )

    assert len(prefix_topk_rows) == 1
    assert prefix_topk_rows[0]["selection"] == "prefix_topk"
    assert prefix_topk_rows[0]["candidate_components"] == 4
    assert prefix_topk_rows[0]["components"] == 2
    assert prefix_topk_rows[0]["control_bytes"] == 3
    assert prefix_topk_rows[0]["basis_reconstruction_key"] == "prefixtopk_p2_c4_k2"
    assert prefix_topk_rows[0]["basis_retained_energy_fraction_mean"] == 0.90
    assert "selection=prefix_topk,prefix=2,candidates=4" in prefix_topk_rows[0]["setting"]

    sparse_component_rows = build_basis_setting_rows(
        summary,
        components=[2],
        candidate_components=[4],
        selections=["topk"],
        quantiles=["p95"],
        bits=4,
        scale=1.0,
        height=512,
        width=512,
        codec="fixed_bits",
        quantizers=["uniform"],
        range_modes=["component_p95"],
    )

    assert len(sparse_component_rows) == 2
    assert sparse_component_rows[0]["candidate_components"] == 2
    assert sparse_component_rows[1]["candidate_components"] == 4
    assert sparse_component_rows[1]["basis_range_mode"] == "component_p95"
    assert "basis_range_mode=component_p95" in sparse_component_rows[1]["setting"]

    sparse_codebook_huffman_rows = build_basis_setting_rows(
        summary,
        components=[2],
        candidate_components=[4],
        selections=["topk"],
        quantiles=["p95"],
        bits=4,
        scale=1.0,
        height=512,
        width=512,
        codec="huffman",
        quantizers=["uniform"],
        range_modes=["component_codebook"],
    )

    assert len(sparse_codebook_huffman_rows) == 1
    assert sparse_codebook_huffman_rows[0]["huffman_key"] == "topk_c4_k2_p95_b4_codebook"
    assert sparse_codebook_huffman_rows[0]["control_bytes"] == 1.2
    assert sparse_codebook_huffman_rows[0]["quantization_mae"] == 0.025
