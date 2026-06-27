from __future__ import annotations

import torch
import torch.nn.functional as F


DECODER_POSTPROCESS_MODES = ("none", "gaussian3x3", "unsharp3x3")


def gaussian_blur_3x3(x: torch.Tensor) -> torch.Tensor:
    """Apply a fixed separable 3x3 Gaussian blur independently per channel."""

    if x.ndim != 4:
        raise ValueError("expected BCHW tensor")
    channels = int(x.shape[1])
    kernel = x.new_tensor(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ]
    )
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kernel, groups=channels)


def apply_decoder_postprocess(x: torch.Tensor, *, mode: str = "none", strength: float = 0.0) -> torch.Tensor:
    """Apply a fixed decoder-side postprocess without image-specific side info."""

    if mode not in DECODER_POSTPROCESS_MODES:
        raise ValueError(f"unknown decoder postprocess mode: {mode}")
    strength_value = float(strength)
    if strength_value < 0.0:
        raise ValueError("--decoder-postprocess-strength must be non-negative")
    if mode == "none":
        if abs(strength_value) > 1.0e-12:
            raise ValueError("--decoder-postprocess-strength must be 0 when mode is none")
        return x

    blur = gaussian_blur_3x3(x)
    if mode == "gaussian3x3":
        if strength_value > 1.0:
            raise ValueError("gaussian3x3 strength must be in [0, 1]")
        return torch.lerp(x, blur, strength_value).clamp(0.0, 1.0)
    if mode == "unsharp3x3":
        return (x + strength_value * (x - blur)).clamp(0.0, 1.0)
    raise AssertionError(f"unhandled decoder postprocess mode: {mode}")
