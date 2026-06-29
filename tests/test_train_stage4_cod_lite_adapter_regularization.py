from __future__ import annotations

import torch
import pytest

from scripts.train_stage4_cod_lite_adapter import (
    condition_residual_rms_excess_loss,
    condition_residual_rms_ratio,
)


def test_condition_residual_rms_ratio_global() -> None:
    base = torch.ones(2, 4, 3, 3)
    residual = torch.full_like(base, 2.0)

    ratio = condition_residual_rms_ratio(base, residual)

    assert ratio.shape == (2, 1, 1, 1)
    assert torch.allclose(ratio, torch.full_like(ratio, 2.0))


def test_condition_residual_rms_excess_loss_penalizes_only_excess() -> None:
    base = torch.ones(2, 4, 3, 3)
    residual = torch.full_like(base, 2.0)

    loss, ratio_mean, ratio_max = condition_residual_rms_excess_loss(
        base,
        residual,
        max_rms_ratio=0.5,
    )

    assert torch.allclose(loss, torch.tensor(2.25))
    assert torch.allclose(ratio_mean, torch.tensor(2.0))
    assert torch.allclose(ratio_max, torch.tensor(2.0))


def test_condition_residual_rms_excess_loss_zero_when_below_guard() -> None:
    base = torch.ones(2, 4, 3, 3)
    residual = torch.full_like(base, 0.25)

    loss, _, _ = condition_residual_rms_excess_loss(base, residual, max_rms_ratio=0.5)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_condition_residual_rms_ratio_supports_spatial_and_channel() -> None:
    base = torch.ones(2, 4, 3, 3)
    residual = torch.full_like(base, 2.0)

    spatial = condition_residual_rms_ratio(base, residual, granularity="spatial")
    channel = condition_residual_rms_ratio(base, residual, granularity="channel")

    assert spatial.shape == (2, 1, 3, 3)
    assert channel.shape == (2, 4, 1, 1)
    assert torch.allclose(spatial, torch.full_like(spatial, 2.0))
    assert torch.allclose(channel, torch.full_like(channel, 2.0))


def test_condition_residual_rms_ratio_validation() -> None:
    base = torch.ones(1, 2, 2, 2)
    residual = torch.ones_like(base)

    with pytest.raises(ValueError, match="share shape"):
        condition_residual_rms_ratio(base, residual[:, :, :1])
    with pytest.raises(ValueError, match="unknown"):
        condition_residual_rms_ratio(base, residual, granularity="bad")
    with pytest.raises(ValueError, match="positive"):
        condition_residual_rms_excess_loss(base, residual, max_rms_ratio=0.0)
