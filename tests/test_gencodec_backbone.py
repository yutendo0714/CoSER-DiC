from __future__ import annotations

import torch
import pytest
from torch import nn

from coserdic.models import (
    CoDLiteOneStepBackbone,
    CoDLiteOneStepBackboneConfig,
    CoSERToCoDLiteAlphaGate,
    CoSERToCoDLiteAlphaGateConfig,
    CoSERToCoDLiteConditionAdapter,
    CoSERToCoDLiteConditionAdapterConfig,
    CoSERToCoDLiteConditionGate,
    CoSERToCoDLiteConditionGateConfig,
    CoSERToCoDLiteConditionPyramidAdapter,
    CoSERToCoDLiteConditionPyramidAdapterConfig,
    LoRAConv2d,
    LoRALinear,
    apply_lora_adapters_by_name,
    apply_lora_adapters_from_config,
    condition_residual_rms_guard,
    lora_parameter_names,
    load_named_parameter_state,
    named_parameter_state,
    set_trainable_parameters_by_name,
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


class _FakeUp2xEmbedder(nn.Module):
    up2x = True

    def calculate_indices_size(self, height: int, width: int) -> tuple[int, int]:
        return height // 32, width // 32

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond = torch.zeros(x.shape[0], 8, x.shape[-2] // 16, x.shape[-1] // 16, device=x.device, dtype=x.dtype)
        return cond, {}


class _FakeUp2xCoDLiteNet(_FakeCoDLiteNet):
    def __init__(self) -> None:
        super().__init__()
        self.y_embedder = _FakeUp2xEmbedder()


class _FakeTrainableBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))


def test_cod_lite_backbone_freezes_and_maps_output_range() -> None:
    net = _FakeCoDLiteNet()
    backbone = CoDLiteOneStepBackbone(net, CoDLiteOneStepBackboneConfig(freeze=True))
    x_aux = torch.rand(2, 3, 64, 64)
    cond = torch.zeros(2, 8, 4, 4)

    out = backbone(x_aux, cond)

    assert out.shape == x_aux.shape
    assert torch.allclose(out, torch.full_like(out, 0.5))
    assert not net.weight.requires_grad


def test_cod_lite_backbone_condition_size_uses_native_condition_shape_for_up2x() -> None:
    backbone = CoDLiteOneStepBackbone(_FakeUp2xCoDLiteNet(), CoDLiteOneStepBackboneConfig())
    x_aux = torch.rand(1, 3, 64, 64)
    cond = torch.zeros(1, 8, 4, 4)

    assert backbone.condition_size(64, 64) == (4, 4)
    assert backbone(x_aux, cond).shape == x_aux.shape


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
            num_detail_blocks=2,
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


def test_coser_to_cod_lite_pyramid_adapter_detail_control_branch_directly_adds_condition_delta() -> None:
    adapter = CoSERToCoDLiteConditionPyramidAdapter(
        CoSERToCoDLiteConditionPyramidAdapterConfig(
            semantic_channels=12,
            detail_context_channels=6,
            condition_channels=8,
            hidden_channels=16,
            num_image_blocks=1,
            num_condition_blocks=1,
            num_detail_blocks=1,
            num_fusion_blocks=1,
            detail_control_branch=True,
            zero_init_output=True,
        )
    )
    assert adapter.detail_control_out is not None
    with torch.no_grad():
        adapter.detail_control_out.bias.fill_(0.125)
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
    assert torch.allclose(cond, torch.full_like(cond, 0.125))


def test_coser_to_cod_lite_pyramid_adapter_detail_film_modulation_zero_init() -> None:
    adapter = CoSERToCoDLiteConditionPyramidAdapter(
        CoSERToCoDLiteConditionPyramidAdapterConfig(
            semantic_channels=12,
            detail_context_channels=6,
            condition_channels=8,
            hidden_channels=16,
            num_image_blocks=1,
            num_condition_blocks=1,
            num_detail_blocks=1,
            num_fusion_blocks=1,
            detail_film_modulation=True,
            zero_init_output=True,
        )
    )
    assert adapter.detail_film is not None
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


def test_coser_to_cod_lite_pyramid_adapter_detail_film_requires_detail_context() -> None:
    with pytest.raises(ValueError, match="detail_film_modulation requires"):
        CoSERToCoDLiteConditionPyramidAdapter(
            CoSERToCoDLiteConditionPyramidAdapterConfig(
                detail_context_channels=0,
                detail_film_modulation=True,
            )
        )


def test_backbone_trainable_parameter_selection_and_state_roundtrip() -> None:
    module = _FakeTrainableBackbone()

    names = set_trainable_parameters_by_name(module, [r"decoder\.1"])

    assert names == ["decoder.1.weight", "decoder.1.bias"]
    assert not module.encoder.weight.requires_grad
    assert not module.decoder[0].weight.requires_grad
    assert module.decoder[1].weight.requires_grad

    with torch.no_grad():
        module.decoder[1].weight.fill_(0.25)
        module.decoder[1].bias.fill_(0.5)
    state = named_parameter_state(module, names)
    with torch.no_grad():
        module.decoder[1].weight.zero_()
        module.decoder[1].bias.zero_()
    loaded = load_named_parameter_state(module, state)

    assert loaded == names
    assert torch.allclose(module.decoder[1].weight, torch.full_like(module.decoder[1].weight, 0.25))
    assert torch.allclose(module.decoder[1].bias, torch.full_like(module.decoder[1].bias, 0.5))


def test_condition_residual_rms_guard_clips_global_residual() -> None:
    base = torch.ones(2, 4, 3, 3)
    residual = torch.full_like(base, 4.0)

    guarded, gate = condition_residual_rms_guard(base, residual, max_rms_ratio=0.5)

    assert gate.shape == (2, 1, 1, 1)
    assert torch.allclose(gate, torch.full_like(gate, 0.125))
    assert torch.allclose(guarded, torch.full_like(guarded, 0.5))


def test_condition_residual_rms_guard_supports_spatial_and_channel_granularity() -> None:
    base = torch.ones(2, 4, 3, 3)
    residual = torch.full_like(base, 2.0)

    spatial_guarded, spatial_gate = condition_residual_rms_guard(
        base,
        residual,
        max_rms_ratio=0.25,
        granularity="spatial",
    )
    channel_guarded, channel_gate = condition_residual_rms_guard(
        base,
        residual,
        max_rms_ratio=0.25,
        granularity="channel",
    )

    assert spatial_gate.shape == (2, 1, 3, 3)
    assert channel_gate.shape == (2, 4, 1, 1)
    assert torch.allclose(spatial_guarded, torch.full_like(spatial_guarded, 0.25))
    assert torch.allclose(channel_guarded, torch.full_like(channel_guarded, 0.25))


def test_condition_residual_rms_guard_validation() -> None:
    base = torch.ones(1, 2, 2, 2)
    residual = torch.ones_like(base)

    with pytest.raises(ValueError, match="share shape"):
        condition_residual_rms_guard(base, residual[:, :, :1], max_rms_ratio=0.5)
    with pytest.raises(ValueError, match="must be positive"):
        condition_residual_rms_guard(base, residual, max_rms_ratio=0.0)
    with pytest.raises(ValueError, match="unknown"):
        condition_residual_rms_guard(base, residual, max_rms_ratio=0.5, granularity="bad")


def test_lora_linear_zero_init_preserves_base_output_and_tracks_params() -> None:
    base = nn.Linear(4, 3).double()
    x = torch.randn(2, 4, dtype=torch.float64)
    expected = base(x)

    wrapped = LoRALinear(base, rank=2, alpha=2.0)
    out = wrapped(x)

    assert torch.allclose(out, expected)
    assert not wrapped.base.weight.requires_grad
    assert wrapped.lora_down.weight.requires_grad
    assert wrapped.lora_up.weight.requires_grad
    assert wrapped.lora_down.weight.dtype == base.weight.dtype
    assert wrapped.lora_up.weight.dtype == base.weight.dtype


def test_lora_conv2d_zero_init_preserves_base_output() -> None:
    base = nn.Conv2d(3, 5, kernel_size=3, padding=1).double()
    x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
    expected = base(x)

    wrapped = LoRAConv2d(base, rank=2, alpha=2.0)
    out = wrapped(x)

    assert torch.allclose(out, expected)
    assert not wrapped.base.weight.requires_grad
    assert wrapped.lora_down.weight.requires_grad
    assert wrapped.lora_up.weight.requires_grad
    assert wrapped.lora_down.weight.dtype == base.weight.dtype
    assert wrapped.lora_up.weight.dtype == base.weight.dtype


def test_apply_lora_adapters_by_name_and_state_roundtrip() -> None:
    module = _FakeTrainableBackbone()
    x = torch.randn(2, 2)
    expected = module.decoder[1](module.decoder[0](x))

    applied = apply_lora_adapters_by_name(module, [r"decoder\.1"], rank=2, alpha=2.0)
    names = lora_parameter_names(module)
    out = module.decoder[1](module.decoder[0](x))

    assert applied == ["decoder.1"]
    assert isinstance(module.decoder[1], LoRALinear)
    assert torch.allclose(out, expected)
    assert names == ["decoder.1.lora_down.weight", "decoder.1.lora_up.weight"]
    state = named_parameter_state(module, names)
    with torch.no_grad():
        module.decoder[1].lora_up.weight.fill_(0.25)
    load_named_parameter_state(module, state)
    assert torch.allclose(module.decoder[1].lora_up.weight, state["decoder.1.lora_up.weight"])


def test_apply_lora_adapters_from_config() -> None:
    module = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1), nn.Linear(4, 2))

    applied = apply_lora_adapters_from_config(
        module,
        {
            "patterns": [r"0", r"1"],
            "rank": 2,
            "alpha": 2.0,
            "target_module_types": ["linear", "conv2d"],
        },
    )

    assert applied == ["0", "1"]
    assert isinstance(module[0], LoRAConv2d)
    assert isinstance(module[1], LoRALinear)


def test_apply_lora_adapters_by_name_is_not_applied_twice() -> None:
    module = _FakeTrainableBackbone()

    first = apply_lora_adapters_by_name(module, [r"decoder\.1"], rank=2, alpha=2.0)
    second = apply_lora_adapters_by_name(module, [r"decoder\.1"], rank=2, alpha=2.0)

    assert first == ["decoder.1"]
    assert second == []
    assert isinstance(module.decoder[1], LoRALinear)
    assert isinstance(module.decoder[1].base, nn.Linear)


def test_coser_to_cod_lite_alpha_gate_shape_and_initial_alpha() -> None:
    gate = CoSERToCoDLiteAlphaGate(
        CoSERToCoDLiteAlphaGateConfig(
            semantic_channels=12,
            detail_context_channels=6,
            condition_channels=8,
            hidden_channels=16,
            num_blocks=1,
            init_alpha=0.3,
        )
    )
    x_aux = torch.rand(2, 3, 64, 64)
    x_sem = torch.rand(2, 3, 64, 64)
    residual = torch.rand(2, 3, 64, 64) - 0.5
    semantic_latent = torch.rand(2, 12, 4, 4)
    detail_context = torch.rand(2, 6, 4, 4)
    base_condition = torch.rand(2, 8, 4, 4)
    condition_residual = torch.rand(2, 8, 4, 4) - 0.5

    alpha = gate(
        x_aux,
        x_sem,
        residual,
        semantic_latent,
        condition_size=(4, 4),
        base_condition=base_condition,
        condition_residual=condition_residual,
        detail_context=detail_context,
    )

    assert alpha.shape == (2, 1, 1, 1)
    assert torch.all(alpha >= 0)
    assert torch.all(alpha <= 1)
    assert torch.allclose(alpha, torch.full_like(alpha, 0.3), atol=1.0e-6)


def test_coser_to_cod_lite_condition_gate_shape_and_initial_gate() -> None:
    gate = CoSERToCoDLiteConditionGate(
        CoSERToCoDLiteConditionGateConfig(
            semantic_channels=12,
            detail_context_channels=6,
            condition_channels=8,
            hidden_channels=16,
            num_blocks=1,
            init_gate=0.25,
            spatial_gate=True,
            channel_gate=False,
        )
    )
    x_aux = torch.rand(2, 3, 64, 64)
    x_sem = torch.rand(2, 3, 64, 64)
    residual = torch.rand(2, 3, 64, 64) - 0.5
    semantic_latent = torch.rand(2, 12, 4, 4)
    detail_context = torch.rand(2, 6, 4, 4)
    base_condition = torch.rand(2, 8, 4, 4)
    condition_residual = torch.rand(2, 8, 4, 4) - 0.5

    condition_gate = gate(
        x_aux,
        x_sem,
        residual,
        semantic_latent,
        condition_size=(4, 4),
        base_condition=base_condition,
        condition_residual=condition_residual,
        detail_context=detail_context,
    )

    assert condition_gate.shape == (2, 1, 4, 4)
    assert torch.all(condition_gate >= 0)
    assert torch.all(condition_gate <= 1)
    assert torch.allclose(condition_gate, torch.full_like(condition_gate, 0.25), atol=1.0e-6)


def test_coser_to_cod_lite_condition_gate_can_be_channel_global() -> None:
    gate = CoSERToCoDLiteConditionGate(
        CoSERToCoDLiteConditionGateConfig(
            semantic_channels=12,
            condition_channels=8,
            hidden_channels=16,
            num_blocks=1,
            init_gate=0.4,
            spatial_gate=False,
            channel_gate=True,
        )
    )
    x_aux = torch.rand(2, 3, 64, 64)
    x_sem = torch.rand(2, 3, 64, 64)
    residual = torch.rand(2, 3, 64, 64) - 0.5
    semantic_latent = torch.rand(2, 12, 4, 4)
    base_condition = torch.rand(2, 8, 4, 4)
    condition_residual = torch.rand(2, 8, 4, 4) - 0.5

    condition_gate = gate(
        x_aux,
        x_sem,
        residual,
        semantic_latent,
        condition_size=(4, 4),
        base_condition=base_condition,
        condition_residual=condition_residual,
    )

    assert condition_gate.shape == (2, 8, 1, 1)
    assert torch.allclose(condition_gate, torch.full_like(condition_gate, 0.4), atol=1.0e-6)


def test_coser_to_cod_lite_condition_gate_can_amplify_residual() -> None:
    gate = CoSERToCoDLiteConditionGate(
        CoSERToCoDLiteConditionGateConfig(
            semantic_channels=12,
            condition_channels=8,
            hidden_channels=16,
            num_blocks=1,
            gate_min=0.0,
            gate_max=1.5,
            init_gate=1.25,
            spatial_gate=False,
            channel_gate=False,
        )
    )
    x_aux = torch.rand(2, 3, 64, 64)
    x_sem = torch.rand(2, 3, 64, 64)
    residual = torch.rand(2, 3, 64, 64) - 0.5
    semantic_latent = torch.rand(2, 12, 4, 4)
    base_condition = torch.rand(2, 8, 4, 4)
    condition_residual = torch.rand(2, 8, 4, 4) - 0.5

    condition_gate = gate(
        x_aux,
        x_sem,
        residual,
        semantic_latent,
        condition_size=(4, 4),
        base_condition=base_condition,
        condition_residual=condition_residual,
    )

    assert condition_gate.shape == (2, 1, 1, 1)
    assert torch.all(condition_gate >= 0)
    assert torch.all(condition_gate <= 1.5)
    assert torch.allclose(condition_gate, torch.full_like(condition_gate, 1.25), atol=1.0e-6)
