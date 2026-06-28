from __future__ import annotations

import torch
from torch import nn

from coserdic.models import (
    CoDLiteOneStepBackbone,
    CoDLiteOneStepBackboneConfig,
    CoSERToCoDLiteConditionAdapter,
    CoSERToCoDLiteConditionAdapterConfig,
    CoSERToCoDLiteConditionPyramidAdapter,
    CoSERToCoDLiteConditionPyramidAdapterConfig,
)


class _FakeCoDLiteNet(nn.Module):
    hidden_size = 8

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.y_embedder = self

    def calculate_indices_size(self, height: int, width: int) -> tuple[int, int]:
        return height // 16, width // 16

    def inference(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        pooled = cond.mean(dim=1, keepdim=True)
        pooled = torch.nn.functional.interpolate(pooled, size=y.shape[-2:], mode="nearest")
        return pooled.repeat(1, 3, 1, 1).clamp(-1.0, 1.0)


def test_cod_lite_backbone_freezes_and_maps_output_range() -> None:
    net = _FakeCoDLiteNet()
    backbone = CoDLiteOneStepBackbone(net, CoDLiteOneStepBackboneConfig(freeze=True))
    x_aux = torch.rand(2, 3, 64, 64)
    cond = torch.zeros(2, 8, 4, 4)

    out = backbone(x_aux, cond)

    assert out.shape == x_aux.shape
    assert torch.allclose(out, torch.full_like(out, 0.5))
    assert not net.weight.requires_grad


def test_cod_lite_backbone_rejects_wrong_condition_shape() -> None:
    backbone = CoDLiteOneStepBackbone(_FakeCoDLiteNet(), CoDLiteOneStepBackboneConfig())
    x_aux = torch.rand(1, 3, 64, 64)
    cond = torch.zeros(1, 8, 8, 8)

    try:
        backbone(x_aux, cond)
    except ValueError as exc:
        assert "condition spatial shape mismatch" in str(exc)
    else:
        raise AssertionError("wrong condition shape should fail")


def test_coser_to_cod_lite_adapter_shape_and_zero_init() -> None:
    adapter = CoSERToCoDLiteConditionAdapter(
        CoSERToCoDLiteConditionAdapterConfig(
            semantic_channels=12,
            condition_channels=8,
            hidden_channels=16,
            zero_init_output=True,
        )
    )
    x_aux = torch.rand(2, 3, 64, 64)
    x_sem = torch.rand(2, 3, 64, 64)
    residual = torch.rand(2, 3, 64, 64) - 0.5
    semantic = torch.rand(2, 12, 4, 4)

    cond = adapter(x_aux, x_sem, residual, semantic, condition_size=(4, 4))

    assert cond.shape == (2, 8, 4, 4)
    assert torch.count_nonzero(cond) == 0


def test_coser_to_cod_lite_pyramid_adapter_uses_base_condition() -> None:
    adapter = CoSERToCoDLiteConditionPyramidAdapter(
        CoSERToCoDLiteConditionPyramidAdapterConfig(
            semantic_channels=3,
            condition_channels=8,
            hidden_channels=16,
            num_image_blocks=1,
            num_condition_blocks=1,
            num_fusion_blocks=1,
            zero_init_output=True,
        )
    )
    x_aux = torch.rand(2, 3, 64, 64)
    x_sem = torch.rand(2, 3, 64, 64)
    residual = torch.rand(2, 3, 64, 64) - 0.5
    base_condition = torch.rand(2, 8, 4, 4)

    cond = adapter(
        x_aux,
        x_sem,
        residual,
        x_sem,
        condition_size=(4, 4),
        base_condition=base_condition,
    )

    assert cond.shape == (2, 8, 4, 4)
    assert torch.count_nonzero(cond) == 0


def test_coser_to_cod_lite_pyramid_adapter_accepts_detail_context() -> None:
    adapter = CoSERToCoDLiteConditionPyramidAdapter(
        CoSERToCoDLiteConditionPyramidAdapterConfig(
            semantic_channels=12,
            detail_context_channels=6,
            condition_channels=8,
            hidden_channels=16,
            num_image_blocks=1,
            num_condition_blocks=1,
            num_fusion_blocks=1,
            zero_init_output=True,
        )
    )
    x_aux = torch.rand(2, 3, 64, 64)
    x_sem = torch.rand(2, 3, 64, 64)
    residual = torch.rand(2, 3, 64, 64) - 0.5
    semantic_latent = torch.rand(2, 12, 4, 4)
    detail_context = torch.rand(2, 6, 4, 4)
    base_condition = torch.rand(2, 8, 4, 4)

    cond = adapter(
        x_aux,
        x_sem,
        residual,
        semantic_latent,
        condition_size=(4, 4),
        base_condition=base_condition,
        detail_context=detail_context,
    )

    assert cond.shape == (2, 8, 4, 4)
    assert torch.count_nonzero(cond) == 0
