from __future__ import annotations

import torch

from scripts.fit_stage5_condition_control_basis import (
    compute_basis_reconstruction_stats,
    fit_component_codebook_huffman_priors,
    fit_component_codebook_priors,
    fit_control_basis_from_grids,
    fit_control_huffman_priors,
    fit_sparse_topk_control_priors,
    fit_sparse_topk_component_codebook_priors,
    fit_vector_codebook_priors,
    grouped_condition_grid,
    normalize_quantile_label,
)


def test_normalize_quantile_label_accepts_numeric_and_named_forms() -> None:
    assert normalize_quantile_label("p99") == "p99"
    assert normalize_quantile_label("0.99") == "p99"
    assert normalize_quantile_label("99") == "p99"
    assert normalize_quantile_label("0.95") == "p95"


def test_grouped_condition_grid_shape() -> None:
    error = torch.randn(3, 8, 6, 6)
    grid = grouped_condition_grid(error, groups=4, grid_size=2)

    assert grid.shape == (3, 4, 2, 2)


def test_fit_control_basis_from_grids_recovers_low_rank_shape() -> None:
    base_a = torch.zeros(2, 2, 2)
    base_b = torch.zeros(2, 2, 2)
    base_a[0, 0, 0] = 1.0
    base_b[1, 1, 1] = 1.0
    coeff_a = torch.linspace(-1.0, 1.0, steps=8)
    coeff_b = torch.linspace(1.0, -1.0, steps=8)
    grids = torch.stack([a * base_a + b * base_b for a, b in zip(coeff_a, coeff_b)], dim=0)

    fitted = fit_control_basis_from_grids(grids, components=2, center=True)

    assert fitted["basis"].shape == (2, 2, 2, 2)
    assert fitted["mean"].shape == (2, 2, 2)
    assert len(fitted["explained_variance"]) == 2
    assert fitted["cumulative_explained_variance"][0] > 0.99
    assert fitted["coefficient_abs_mean"] > 0.0
    assert fitted["coefficient_abs_max"] >= fitted["coefficient_abs_quantiles"]["p99"]
    assert len(fitted["coefficient_component_abs_p95"]) == 2
    assert len(fitted["coefficient_component_abs_p99"]) == 2
    assert fitted["coefficients"].shape == (8, 2)


def test_fit_control_huffman_priors_records_payload_stats() -> None:
    coefficients = torch.tensor(
        [
            [0.0, 0.1, -0.1],
            [0.0, 0.2, -0.2],
            [0.0, 0.1, -0.1],
            [0.0, 0.2, -0.2],
        ],
        dtype=torch.float32,
    )

    priors = fit_control_huffman_priors(
        coefficients,
        bits=4,
        ranges={"p95": 0.5},
        quantizers=["uniform", "mu_law"],
        mu=16.0,
        smoothing_count=1,
    )

    prior = priors["p95_b4"]
    mulaw_prior = priors["p95_b4_mulaw16"]
    assert prior["fixed_bytes_per_image"] == 2
    assert prior["mean_payload_bytes"] <= 2.0
    assert len(prior["prefix_mean_payload_bytes"]) == 3
    assert prior["prefix_mean_payload_bytes"][0] <= prior["prefix_mean_payload_bytes"][-1]
    assert prior["quantization_mae"] >= 0.0
    assert prior["quantization_rmse"] >= 0.0
    assert 0.0 <= prior["clipped_fraction"] <= 1.0
    assert mulaw_prior["quantizer"] == "mu_law"
    assert mulaw_prior["mu"] == 16.0
    assert prior["huffman"]["codec"] == "static_control_huffman"


def test_fit_component_codebook_priors_records_fixed_decoder_state() -> None:
    coefficients = torch.tensor(
        [
            [0.00, -1.00],
            [0.02, -0.80],
            [0.08, 0.75],
            [0.10, 1.00],
        ],
        dtype=torch.float32,
    )

    priors = fit_component_codebook_priors(coefficients, bits_list=[2], max_iter=8)

    prior = priors["lloyd_b2"]
    assert prior["codec"] == "component_codebook_control_fixed_bits"
    assert prior["bits"] == 2
    assert prior["levels"] == 4
    assert prior["component_count"] == 2
    assert prior["fixed_bytes_per_image"] == 1
    assert len(prior["codebooks"]) == 2
    assert len(prior["codebooks"][0]) == 4
    assert prior["quantization_mae"] >= 0.0
    assert prior["quantization_rmse"] >= 0.0


def test_fit_component_codebook_huffman_priors_records_payload_stats() -> None:
    coefficients = torch.tensor(
        [
            [0.00, -1.00],
            [0.02, -0.80],
            [0.08, 0.75],
            [0.10, 1.00],
        ],
        dtype=torch.float32,
    )
    codebooks = fit_component_codebook_priors(coefficients, bits_list=[2], max_iter=8)

    priors = fit_component_codebook_huffman_priors(
        coefficients,
        codebook_priors=codebooks,
        ranges={"p95": 1.0},
        smoothing_count=1,
    )

    prior = priors["p95_b2_codebook"]
    assert prior["basis_range_mode"] == "component_codebook"
    assert prior["codebook_key"] == "lloyd_b2"
    assert prior["fixed_bytes_per_image"] == 1
    assert prior["mean_payload_bytes"] <= 1.0
    assert len(prior["prefix_mean_payload_bytes"]) == 2
    assert prior["clipped_fraction"] == 0.0
    assert prior["huffman"]["codec"] == "static_control_huffman"


def test_fit_sparse_topk_control_priors_records_index_and_value_stats() -> None:
    coefficients = torch.tensor(
        [
            [1.0, 0.1, 0.0, -0.8],
            [0.9, 0.2, 0.0, -0.7],
            [0.0, 1.1, -0.9, 0.0],
            [0.0, 1.0, -0.8, 0.1],
        ],
        dtype=torch.float32,
    )

    priors = fit_sparse_topk_control_priors(
        coefficients,
        bits=4,
        ranges={"p95": 1.0},
        candidate_components=[4],
        selected_components=[2],
        quantizers=["uniform"],
        smoothing_count=1,
    )

    prior = priors["topk_c4_k2_p95_b4"]
    assert prior["codec"] == "sparse_topk_static_huffman_prior"
    assert prior["candidate_components"] == 4
    assert prior["selected_components"] == 2
    assert prior["fixed_bytes_per_image"] == 2
    assert prior["mean_payload_bytes"] <= 2.0
    assert prior["index_mean_payload_bytes"] >= 1.0
    assert prior["value_mean_payload_bytes"] >= 1.0
    assert prior["index_huffman"]["codec"] == "static_control_huffman"
    assert prior["value_huffman"]["codec"] == "static_control_huffman"


def test_fit_sparse_topk_component_codebook_priors_records_payload_stats() -> None:
    coefficients = torch.tensor(
        [
            [1.0, 0.1, 0.0, -0.8],
            [0.9, 0.2, 0.0, -0.7],
            [0.0, 1.1, -0.9, 0.0],
            [0.0, 1.0, -0.8, 0.1],
        ],
        dtype=torch.float32,
    )
    codebooks = fit_component_codebook_priors(coefficients, bits_list=[2], max_iter=8)

    priors = fit_sparse_topk_component_codebook_priors(
        coefficients,
        codebook_priors=codebooks,
        ranges={"p95": 1.0},
        candidate_components=[4],
        selected_components=[2],
        smoothing_count=1,
    )

    prior = priors["topk_c4_k2_p95_b2_codebook"]
    assert prior["codec"] == "sparse_topk_static_huffman_prior"
    assert prior["basis_range_mode"] == "component_codebook"
    assert prior["candidate_components"] == 4
    assert prior["selected_components"] == 2
    assert prior["fixed_bytes_per_image"] == 2
    assert prior["mean_payload_bytes"] <= 2.0
    assert prior["index_huffman"]["codec"] == "static_control_huffman"
    assert prior["value_huffman"]["codec"] == "static_control_huffman"


def test_fit_vector_codebook_priors_records_payload_stats() -> None:
    coefficients = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.9, -0.1, 0.0],
        ],
        dtype=torch.float32,
    )

    priors = fit_vector_codebook_priors(
        coefficients,
        components=[2],
        bits_list=[2],
        max_iter=8,
        smoothing_count=1,
    )

    prior = priors["vq_k2_b2"]
    assert prior["codec"] == "vector_codebook_control_fixed_bits"
    assert prior["selection"] == "vector"
    assert prior["components"] == 2
    assert prior["bits"] == 2
    assert prior["fixed_bytes_per_image"] == 1
    assert prior["huffman_mean_payload_bytes"] <= 1.0
    assert prior["quantization_mae"] >= 0.0
    assert len(prior["vectors"]) == 4
    assert len(prior["vectors"][0]) == 2
    assert prior["huffman"]["codec"] == "static_control_huffman"


def test_compute_basis_reconstruction_stats_compares_prefix_and_topk() -> None:
    coefficients = torch.tensor(
        [
            [1.0, 0.1, 0.0, -0.8],
            [0.0, 1.0, -0.8, 0.1],
        ],
        dtype=torch.float32,
    )

    stats = compute_basis_reconstruction_stats(
        coefficients,
        prefix_components=[1, 2],
        topk_candidate_components=[4],
        topk_components=[1, 2],
    )

    assert stats["prefix_k1"]["selection"] == "prefix"
    assert stats["topk_c4_k1"]["selection"] == "topk"
    assert stats["topk_c4_k1"]["retained_energy_fraction_mean"] >= stats["prefix_k1"][
        "retained_energy_fraction_mean"
    ]
    assert stats["topk_c4_k2"]["residual_energy_fraction_mean"] <= stats["prefix_k2"][
        "residual_energy_fraction_mean"
    ]
