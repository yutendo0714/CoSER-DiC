from __future__ import annotations

import pytest
import torch

from coserdic.models import apply_decoder_postprocess, gaussian_blur_3x3


def test_none_postprocess_is_identity() -> None:
    x = torch.rand(2, 3, 8, 8)

    y = apply_decoder_postprocess(x, mode="none", strength=0.0)

    assert torch.equal(y, x)


def test_gaussian_blur_preserves_shape_and_constant_image() -> None:
    x = torch.full((1, 3, 8, 8), 0.25)

    y = gaussian_blur_3x3(x)

    assert y.shape == x.shape
    assert torch.allclose(y, x)


def test_unsharp_preserves_constant_image() -> None:
    x = torch.full((1, 3, 8, 8), 0.25)

    y = apply_decoder_postprocess(x, mode="unsharp3x3", strength=0.5)

    assert torch.allclose(y, x)


def test_gaussian_strength_must_be_in_unit_interval() -> None:
    x = torch.rand(1, 3, 8, 8)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_decoder_postprocess(x, mode="gaussian3x3", strength=1.5)


def test_none_rejects_nonzero_strength() -> None:
    x = torch.rand(1, 3, 8, 8)

    with pytest.raises(ValueError, match="must be 0"):
        apply_decoder_postprocess(x, mode="none", strength=0.1)
