from __future__ import annotations

import torch

from coserdic.entropy import ComponentCodebookControlCode, UniformControlGridCode
from scripts.eval_stage4_cod_lite_adapter import basis_component_codebook_codec
from scripts.eval_stage4_cod_lite_adapter import basis_control_payload_bits
from scripts.eval_stage4_cod_lite_adapter import basis_vector_codebook_codec
from scripts.eval_stage4_cod_lite_adapter import codec_for_selected_basis_indices
from scripts.eval_stage4_cod_lite_adapter import grouped_condition_affine_control
from scripts.eval_stage4_cod_lite_adapter import grouped_condition_affine_basis_control
from scripts.eval_stage4_cod_lite_adapter import grouped_condition_affine_dct_control
from scripts.eval_stage4_cod_lite_adapter import grouped_condition_affine_grid_control
from scripts.eval_stage4_cod_lite_adapter import load_control_huffman_prior
from scripts.eval_stage4_cod_lite_adapter import load_sparse_topk_control_huffman_prior
from scripts.eval_stage4_cod_lite_adapter import load_vector_codebook_huffman_prior
from scripts.eval_stage4_cod_lite_adapter import payload_bytes_with_hybrid_selector
from scripts.eval_stage4_cod_lite_adapter import select_condition_control_by_image_rdo
from scripts.eval_stage4_cod_lite_adapter import select_condition_control_by_rd_proxy


def test_grouped_condition_affine_control_counts_payload_and_reduces_condition_error() -> None:
    base = torch.zeros(1, 4, 4, 4)
    residual = torch.tensor(
        [
            [
                [[0.10, 0.20, 0.10, 0.20]] * 4,
                [[0.12, 0.18, 0.12, 0.18]] * 4,
                [[-0.10, -0.20, -0.10, -0.20]] * 4,
                [[-0.12, -0.18, -0.12, -0.18]] * 4,
            ]
        ],
        dtype=torch.float32,
    )
    pred = base + residual
    target = base + 1.75 * residual + 0.05

    before = torch.mean(torch.abs(pred - target)).item()
    correction, payload_bytes, control_abs = grouped_condition_affine_control(
        base,
        pred,
        target,
        groups=2,
        grid_size=1,
        gain_codec=UniformControlGridCode(bits=12, value_range=1.0),
        bias_codec=UniformControlGridCode(bits=12, value_range=0.5),
        scale=1.0,
    )
    after = torch.mean(torch.abs((pred + correction) - target)).item()

    assert payload_bytes == [6]
    assert control_abs.shape == (1,)
    assert after < before * 0.25


def test_grouped_condition_affine_dct_control_codes_residual_after_affine() -> None:
    base = torch.zeros(1, 4, 4, 4)
    residual = torch.tensor(
        [
            [
                [[0.10, 0.20, 0.10, 0.20]] * 4,
                [[0.12, 0.18, 0.12, 0.18]] * 4,
                [[-0.10, -0.20, -0.10, -0.20]] * 4,
                [[-0.12, -0.18, -0.12, -0.18]] * 4,
            ]
        ],
        dtype=torch.float32,
    )
    spatial = torch.linspace(-0.04, 0.04, 4, dtype=torch.float32).view(4, 1).repeat(1, 4)
    spatial = spatial.unsqueeze(0).unsqueeze(0)
    pred = base + residual
    target = base + 1.75 * residual + 0.05 + spatial

    affine_only, _, _ = grouped_condition_affine_control(
        base,
        pred,
        target,
        groups=2,
        grid_size=1,
        gain_codec=UniformControlGridCode(bits=12, value_range=1.0),
        bias_codec=UniformControlGridCode(bits=12, value_range=0.5),
        scale=1.0,
    )
    before_dct = torch.mean(torch.abs((pred + affine_only) - target)).item()
    correction, payload_bytes, control_abs = grouped_condition_affine_dct_control(
        base,
        pred,
        target,
        affine_groups=2,
        affine_grid_size=1,
        dct_groups=2,
        dct_grid_size=4,
        dct_coeffs_per_group=4,
        gain_codec=UniformControlGridCode(bits=12, value_range=1.0),
        bias_codec=UniformControlGridCode(bits=12, value_range=0.5),
        dct_codec=UniformControlGridCode(bits=12, value_range=0.25),
        scale=1.0,
    )
    after = torch.mean(torch.abs((pred + correction) - target)).item()

    assert payload_bytes == [18]
    assert control_abs.shape == (1,)
    assert after < before_dct * 0.5


def test_grouped_condition_affine_grid_control_codes_residual_after_affine() -> None:
    base = torch.zeros(1, 4, 4, 4)
    residual = torch.tensor(
        [
            [
                [[0.10, 0.20, 0.10, 0.20]] * 4,
                [[0.12, 0.18, 0.12, 0.18]] * 4,
                [[-0.10, -0.20, -0.10, -0.20]] * 4,
                [[-0.12, -0.18, -0.12, -0.18]] * 4,
            ]
        ],
        dtype=torch.float32,
    )
    spatial = torch.tensor(
        [
            [
                [-0.04, -0.02, 0.02, 0.04],
                [-0.03, -0.01, 0.01, 0.03],
                [0.03, 0.01, -0.01, -0.03],
                [0.04, 0.02, -0.02, -0.04],
            ]
        ],
        dtype=torch.float32,
    ).unsqueeze(1)
    pred = base + residual
    target = base + 1.75 * residual + 0.05 + spatial

    affine_only, _, _ = grouped_condition_affine_control(
        base,
        pred,
        target,
        groups=2,
        grid_size=1,
        gain_codec=UniformControlGridCode(bits=12, value_range=1.0),
        bias_codec=UniformControlGridCode(bits=12, value_range=0.5),
        scale=1.0,
    )
    before_grid = torch.mean(torch.abs((pred + affine_only) - target)).item()
    correction, payload_bytes, control_abs = grouped_condition_affine_grid_control(
        base,
        pred,
        target,
        affine_groups=2,
        affine_grid_size=1,
        grid_groups=2,
        grid_size=4,
        gain_codec=UniformControlGridCode(bits=12, value_range=1.0),
        bias_codec=UniformControlGridCode(bits=12, value_range=0.5),
        grid_codec=UniformControlGridCode(bits=12, value_range=0.25),
        scale=1.0,
    )
    after = torch.mean(torch.abs((pred + correction) - target)).item()

    assert payload_bytes == [54]
    assert control_abs.shape == (1,)
    assert after < before_grid * 0.1


def test_grouped_condition_affine_basis_control_codes_residual_after_affine() -> None:
    base = torch.zeros(1, 4, 4, 4)
    residual = torch.tensor(
        [
            [
                [[0.10, 0.20, 0.10, 0.20]] * 4,
                [[0.12, 0.18, 0.12, 0.18]] * 4,
                [[-0.10, -0.20, -0.10, -0.20]] * 4,
                [[-0.12, -0.18, -0.12, -0.18]] * 4,
            ]
        ],
        dtype=torch.float32,
    )
    spatial2 = torch.tensor([[-0.05, 0.05], [0.05, -0.05]], dtype=torch.float32)
    spatial = spatial2.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1).unsqueeze(0).unsqueeze(0)
    pred = base + residual
    target = base + 1.75 * residual + 0.05 + spatial
    basis_pattern = torch.stack([spatial2, spatial2], dim=0)
    basis = (basis_pattern / basis_pattern.flatten().norm()).unsqueeze(0)
    basis_payload = {"basis": basis, "mean": None, "groups": 2, "grid_size": 2}

    affine_only, _, _ = grouped_condition_affine_control(
        base,
        pred,
        target,
        groups=2,
        grid_size=1,
        gain_codec=UniformControlGridCode(bits=12, value_range=1.0),
        bias_codec=UniformControlGridCode(bits=12, value_range=0.5),
        scale=1.0,
    )
    before_basis = torch.mean(torch.abs((pred + affine_only) - target)).item()
    correction, payload_bytes, control_abs = grouped_condition_affine_basis_control(
        base,
        pred,
        target,
        affine_groups=2,
        affine_grid_size=1,
        gain_codec=UniformControlGridCode(bits=12, value_range=1.0),
        bias_codec=UniformControlGridCode(bits=12, value_range=0.5),
        basis_payload=basis_payload,
        components=1,
        candidate_components=1,
        selection="prefix",
        basis_codec=UniformControlGridCode(bits=12, value_range=0.5),
        scale=1.0,
    )
    after = torch.mean(torch.abs((pred + correction) - target)).item()

    assert payload_bytes == [8]
    assert control_abs.shape == (1,)
    assert after < before_basis * 0.5


def test_grouped_condition_affine_basis_control_can_code_base_condition_residual() -> None:
    base = torch.zeros(1, 4, 4, 4)
    coarse = torch.full((1, 1, 4, 4), 0.05, dtype=torch.float32)
    spatial2 = torch.tensor([[-0.05, 0.05], [0.05, -0.05]], dtype=torch.float32)
    spatial = spatial2.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1).unsqueeze(0).unsqueeze(0)
    target = base + coarse + spatial
    basis_pattern = torch.stack([spatial2, spatial2], dim=0)
    basis = (basis_pattern / basis_pattern.flatten().norm()).unsqueeze(0)
    basis_payload = {"basis": basis, "mean": None, "groups": 2, "grid_size": 2}

    before = torch.mean(torch.abs(base - target)).item()
    correction, payload_bytes, control_abs = grouped_condition_affine_basis_control(
        base,
        base,
        target,
        affine_groups=2,
        affine_grid_size=1,
        gain_codec=UniformControlGridCode(bits=12, value_range=1.0),
        bias_codec=UniformControlGridCode(bits=12, value_range=0.5),
        basis_payload=basis_payload,
        components=1,
        candidate_components=1,
        selection="prefix",
        basis_codec=UniformControlGridCode(bits=12, value_range=0.5),
        scale=1.0,
    )
    after = torch.mean(torch.abs((base + correction) - target)).item()

    assert payload_bytes == [8]
    assert control_abs.shape == (1,)
    assert after < before * 0.5


def test_basis_component_codebook_codec_loads_fixed_decoder_state() -> None:
    payload = {
        "coefficient_component_codebooks": {
            "lloyd_b4": {
                "bits": 4,
                "codebooks": [[float(v) for v in range(16)] for _ in range(3)],
            }
        }
    }

    codec = basis_component_codebook_codec(payload, bits=4, components=2)
    selected = codec_for_selected_basis_indices(codec, torch.tensor([1]))

    assert isinstance(codec, ComponentCodebookControlCode)
    assert codec.component_count == 2
    assert isinstance(selected, ComponentCodebookControlCode)
    assert selected.component_count == 1
    assert selected.encoded_num_bytes((1,)) == 1


def test_codebook_huffman_prior_metadata_loads_basis_range_mode() -> None:
    huffman_payload = {
        "codec": "static_control_huffman",
        "version": 0,
        "codebook_size": 4,
        "symbol_shape": [2],
        "position_code_lengths": [[1, 2, 3, 3], [2, 1, 3, 3]],
    }
    payload = {
        "control_huffman_priors": {
            "p95_b2_codebook": {
                "bits": 2,
                "range": 1.0,
                "quantizer": "uniform",
                "mu": 16.0,
                "basis_range_mode": "component_codebook",
                "huffman": huffman_payload,
            }
        },
        "sparse_topk_control_priors": {
            "topk_c4_k2_p95_b2_codebook": {
                "candidate_components": 4,
                "selected_components": 2,
                "bits": 2,
                "range": 1.0,
                "quantizer": "uniform",
                "mu": 16.0,
                "basis_range_mode": "component_codebook",
                "index_huffman": {
                    "codec": "static_control_huffman",
                    "version": 0,
                    "codebook_size": 4,
                    "symbol_shape": [2],
                    "position_code_lengths": [[1, 2, 3, 3], [2, 1, 3, 3]],
                },
                "value_huffman": huffman_payload,
            }
        },
    }

    _, bits, value_range, quantizer, mu, key, basis_range_mode = load_control_huffman_prior(
        payload,
        key="p95_b2_codebook",
        components=2,
    )
    _, _, sparse_bits, _, _, _, sparse_key, sparse_basis_range_mode = load_sparse_topk_control_huffman_prior(
        payload,
        key="topk_c4_k2_p95_b2_codebook",
        candidate_components=4,
        components=2,
    )

    assert bits == 2
    assert value_range == 1.0
    assert quantizer == "uniform"
    assert mu == 16.0
    assert key == "p95_b2_codebook"
    assert basis_range_mode == "component_codebook"
    assert sparse_bits == 2
    assert sparse_key == "topk_c4_k2_p95_b2_codebook"
    assert sparse_basis_range_mode == "component_codebook"


def test_vector_codebook_codec_and_huffman_prior_load() -> None:
    payload = {
        "coefficient_vector_codebooks": {
            "vq_k2_b2": {
                "codec": "vector_codebook_control_fixed_bits",
                "components": 2,
                "bits": 2,
                "vectors": [[0.0, 0.0], [0.1, -0.1], [0.5, -0.5], [1.0, -1.0]],
                "huffman": {
                    "codec": "static_control_huffman",
                    "version": 0,
                    "codebook_size": 4,
                    "symbol_shape": [1],
                    "position_code_lengths": [[1, 2, 3, 3]],
                },
            }
        }
    }

    codec, key = basis_vector_codebook_codec(payload, components=2, bits=2)
    huffman, bits, huffman_key = load_vector_codebook_huffman_prior(
        payload,
        key="vq_k2_b2",
        components=2,
        bits=2,
    )

    assert key == "vq_k2_b2"
    assert codec.vector_dim == 2
    assert bits == 2
    assert huffman_key == "vq_k2_b2"
    assert huffman.symbol_shape == (1,)


def test_select_condition_control_by_rd_proxy_counts_selector_and_picks_best_condition() -> None:
    pred = torch.zeros(2, 2, 2, 2)
    target = torch.stack([torch.ones(2, 2, 2) * 0.1, torch.ones(2, 2, 2) * 0.4], dim=0)
    weak = torch.stack([torch.ones(2, 2, 2) * 0.05, torch.ones(2, 2, 2) * 0.1], dim=0)
    strong = torch.stack([torch.ones(2, 2, 2) * 0.1, torch.ones(2, 2, 2) * 0.4], dim=0)

    correction, payload_bytes, control_abs, labels, indices, condition_l1 = select_condition_control_by_rd_proxy(
        pred,
        target,
        [
            ("none", torch.zeros_like(pred), [0, 0], torch.zeros(2)),
            ("weak", weak, [2, 2], torch.ones(2) * 0.05),
            ("strong", strong, [6, 6], torch.ones(2) * 0.4),
        ],
        height=512,
        width=512,
        selector_bytes=1,
        rate_lambda=0.0,
    )

    assert labels == ["strong", "strong"]
    assert indices.tolist() == [2, 2]
    assert payload_bytes == [7, 7]
    assert torch.allclose(correction, strong)
    assert torch.allclose(control_abs, torch.ones(2) * 0.4)
    assert torch.allclose(condition_l1, torch.zeros(2))


def test_select_condition_control_by_rd_proxy_can_trade_condition_for_rate() -> None:
    pred = torch.zeros(1, 1, 2, 2)
    target = torch.ones_like(pred) * 0.10
    cheap = torch.ones_like(pred) * 0.09
    expensive = torch.ones_like(pred) * 0.10

    _, payload_bytes, _, labels, _, condition_l1 = select_condition_control_by_rd_proxy(
        pred,
        target,
        [
            ("cheap", cheap, [0], torch.ones(1) * 0.09),
            ("expensive", expensive, [4096], torch.ones(1) * 0.10),
        ],
        height=512,
        width=512,
        selector_bytes=1,
        rate_lambda=1.0,
    )

    assert labels == ["cheap"]
    assert payload_bytes == [1]
    assert condition_l1.item() > 0.0


def test_payload_bytes_with_hybrid_selector_packs_known_fixed_bits_conservatively() -> None:
    assert payload_bytes_with_hybrid_selector([2], selector_bytes=1, selector_bits=4, payload_bits=[12]) == [2]
    assert payload_bytes_with_hybrid_selector([2], selector_bytes=1, selector_bits=2, payload_bits=[16]) == [3]
    assert payload_bytes_with_hybrid_selector([2], selector_bytes=0, selector_bits=2, payload_bits=None) == [3]


def test_basis_control_payload_bits_supports_fixed_topk_and_vector() -> None:
    scalar_codec = UniformControlGridCode(bits=2, value_range=0.5)
    vector_codec = UniformControlGridCode(bits=8, value_range=0.5)

    assert (
        basis_control_payload_bits(
            selection="topk",
            components=1,
            candidate_components=3,
            basis_codec=scalar_codec,
        )
        == 4
    )
    assert (
        basis_control_payload_bits(
            selection="vector",
            components=8,
            candidate_components=8,
            basis_codec=vector_codec,
        )
        == 8
    )
    assert (
        basis_control_payload_bits(
            selection="prefix_topk",
            components=2,
            candidate_components=8,
            prefix_components=2,
            basis_codec=scalar_codec,
        )
        == 14
    )


def test_select_condition_control_by_rd_proxy_can_pack_selector_bits_for_known_payload() -> None:
    pred = torch.zeros(1, 1, 2, 2)
    target = torch.ones_like(pred) * 0.10
    correction = torch.ones_like(pred) * 0.10

    _, payload_bytes, _, labels, _, condition_l1 = select_condition_control_by_rd_proxy(
        pred,
        target,
        [
            ("exact", correction, [2], torch.ones(1) * 0.10, [12]),
        ],
        height=512,
        width=512,
        selector_bytes=1,
        selector_bits=4,
        rate_lambda=0.0,
    )

    assert labels == ["exact"]
    assert payload_bytes == [2]
    assert condition_l1.item() == 0.0


def test_select_condition_control_by_image_rdo_can_choose_image_better_than_condition_best() -> None:
    class EchoBackbone:
        def __call__(self, stage3: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            return condition

    reference = torch.ones(1, 1, 2, 2) * 0.20
    stage3 = torch.zeros_like(reference)
    pred = torch.zeros_like(reference)
    target_condition = torch.ones_like(reference) * 0.50
    image_best = torch.ones_like(reference) * 0.20
    condition_best = torch.ones_like(reference) * 0.50

    (
        selected_condition,
        selected_raw,
        payload_bytes,
        control_abs,
        labels,
        indices,
        condition_l1,
        score,
    ) = select_condition_control_by_image_rdo(
        reference,
        stage3,
        pred,
        target_condition,
        [
            ("image_best", image_best, [2], torch.ones(1) * 0.2),
            ("condition_best", condition_best, [2], torch.ones(1) * 0.5),
        ],
        backbone=EchoBackbone(),  # type: ignore[arg-type]
        selector_bytes=1,
        rate_lambda=0.0,
        objective="image_l1",
        blend_alpha=1.0,
    )

    assert labels == ["image_best"]
    assert indices.tolist() == [0]
    assert payload_bytes == [3]
    assert torch.allclose(selected_condition, image_best)
    assert torch.allclose(selected_raw, image_best)
    assert torch.allclose(control_abs, torch.ones(1) * 0.2)
    assert condition_l1.item() > 0.0
    assert score.item() == 0.0


def test_select_condition_control_by_image_rdo_fidelity_guard_can_change_choice() -> None:
    class EchoBackbone:
        def __call__(self, stage3: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            return condition

    reference = torch.zeros(1, 1, 1, 2)
    stage3 = torch.zeros_like(reference)
    pred = torch.zeros_like(reference)
    target_condition = torch.zeros_like(reference)
    lower_l1 = torch.tensor([[[[0.20, 0.00]]]])
    lower_mse = torch.tensor([[[[0.11, 0.11]]]])

    (
        selected_condition,
        _,
        payload_bytes,
        _,
        labels,
        indices,
        _,
        score,
    ) = select_condition_control_by_image_rdo(
        reference,
        stage3,
        pred,
        target_condition,
        [
            ("lower_l1", lower_l1, [0], torch.ones(1) * 0.10),
            ("lower_mse", lower_mse, [0], torch.ones(1) * 0.11),
        ],
        backbone=EchoBackbone(),  # type: ignore[arg-type]
        selector_bytes=1,
        rate_lambda=0.0,
        objective="image_l1",
        blend_alpha=1.0,
        fidelity_lambda=10.0,
        fidelity_metric="image_mse",
    )

    assert labels == ["lower_mse"]
    assert indices.tolist() == [1]
    assert payload_bytes == [1]
    assert torch.allclose(selected_condition, lower_mse)
    assert score.item() < 0.24
