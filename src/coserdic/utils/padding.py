from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int = 64,
    mode: str = "reflect",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad BCHW or CHW tensor on bottom/right to a spatial multiple."""

    if multiple <= 0:
        raise ValueError("multiple must be positive")
    if x.ndim not in (3, 4):
        raise ValueError("expected CHW or BCHW tensor")

    height, width = x.shape[-2:]
    padded_height = ((height + multiple - 1) // multiple) * multiple
    padded_width = ((width + multiple - 1) // multiple) * multiple
    pad_h = padded_height - height
    pad_w = padded_width - width
    if pad_h == 0 and pad_w == 0:
        return x, (height, width)

    batched = x if x.ndim == 4 else x.unsqueeze(0)
    padded = F.pad(batched, (0, pad_w, 0, pad_h), mode=mode)
    if x.ndim == 3:
        padded = padded.squeeze(0)
    return padded, (height, width)


def crop_to_shape(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("crop shape must be positive")
    return x[..., :height, :width]

