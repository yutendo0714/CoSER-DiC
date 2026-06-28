from __future__ import annotations

import pytest
import torch

from coserdic.models import DecoderSideRefiner, DecoderSideRefinerConfig
from scripts.train_stage4_decoder_refiner import chroma_residual_l1, spatial_total_variation


def test_decoder_refiner_zero_init_is_identity() -> None:
    cfg = DecoderSideRefinerConfig(base_channels=16, semantic_channels=8, semantic_context_channels=4, num_res_blocks=1)
    model = DecoderSideRefiner(cfg)
    x_stage3 = torch.rand(2, 3, 32, 32)
    x_sem = torch.rand(2, 3, 32, 32)
    detail = torch.rand(2, 3, 8, 8) - 0.5
    semantic = torch.rand(2, 8, 8, 8)

    out = model(x_stage3, x_sem, detail, semantic)

    assert out["x_refined"].shape == x_stage3.shape
    assert out["refiner_residual"].shape == x_stage3.shape
    assert torch.allclose(out["x_refined"], x_stage3)
    assert torch.count_nonzero(out["refiner_residual"]) == 0


def test_decoder_refiner_requires_semantic_latent_when_enabled() -> None:
    model = DecoderSideRefiner(
        DecoderSideRefinerConfig(base_channels=16, semantic_channels=8, semantic_context_channels=4, num_res_blocks=1)
    )
    x = torch.rand(1, 3, 16, 16)

    with pytest.raises(ValueError, match="semantic_latent is required"):
        model(x, x, x)


def test_decoder_refiner_can_disable_semantic_latent() -> None:
    model = DecoderSideRefiner(
        DecoderSideRefinerConfig(base_channels=16, num_res_blocks=1, use_semantic_latent=False)
    )
    x = torch.rand(1, 3, 16, 16)
    detail = torch.rand(1, 3, 4, 4)

    out = model(x, x, detail)

    assert out["x_refined"].shape == x.shape


def test_stage4_refiner_guard_losses_are_zero_for_constant_gray_residual() -> None:
    residual = torch.ones(1, 3, 8, 8) * 0.1

    assert torch.isclose(spatial_total_variation(residual), torch.tensor(0.0))
    assert chroma_residual_l1(residual).item() < 1.0e-6


def test_stage4_refiner_guard_losses_detect_edges_and_chroma() -> None:
    residual = torch.zeros(1, 3, 8, 8)
    residual[:, 0, :, 4:] = 0.1
    residual[:, 2, 2:6, 2:6] = 0.1

    assert spatial_total_variation(residual).item() > 0.0
    assert chroma_residual_l1(residual).item() > 0.0
