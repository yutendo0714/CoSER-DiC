from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class CoSEROutput:
    x_hat: torch.Tensor
    x_aux: torch.Tensor
    x_sem: torch.Tensor
    rate_s: torch.Tensor | float
    rate_d: torch.Tensor | float
    rate_total: torch.Tensor | float
    semantic_tokens: torch.Tensor | None = None
    detail_latents: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class CoSERDiC(nn.Module):
    """Core MVP interface.

    Concrete modules are intentionally CoSER-DiC-owned. External repositories
    may initialize or benchmark components, but they should not become this
    class hierarchy.
    """

    def forward_train(
        self,
        x: torch.Tensor,
        target_bpp: float | None = None,
        rate_level: int | None = None,
        perception_level: int | None = None,
        stage: str | None = None,
    ) -> CoSEROutput:
        raise NotImplementedError

    def compress(
        self,
        x: torch.Tensor,
        target_bpp: float | None = None,
        rate_level: int | None = None,
        perception_level: int | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        raise NotImplementedError

    def decompress(self, bitstream: bytes) -> torch.Tensor:
        raise NotImplementedError

    def estimate_bpp(
        self,
        x: torch.Tensor,
        target_bpp: float | None = None,
        rate_level: int | None = None,
    ) -> dict[str, float]:
        raise NotImplementedError

