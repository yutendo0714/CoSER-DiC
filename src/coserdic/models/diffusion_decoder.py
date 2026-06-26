from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .conditioning_adapter import CoSERConditioningAdapter


@dataclass(frozen=True)
class DiffusionDecoderConfig:
    backbone_name: str = "cod_lite"
    pretrained_path: str = "external/pretrained/CoD_Lite/CoD_Lite_pretrain.pt"
    allow_debug_identity_without_backbone: bool = True
    train_backbone_initially: bool = False


class CoSERDiffusionDecoder(nn.Module):
    """CoSER diffusion boundary for MVP-v0.

    The production path should wrap a CoD-Lite initialized backbone. Until that
    integration is implemented, debug identity mode lets Stage 1-3 tests run
    without pretending to be the final diffusion result.
    """

    def __init__(
        self,
        adapter: CoSERConditioningAdapter,
        backbone: nn.Module | None = None,
        cfg: DiffusionDecoderConfig = DiffusionDecoderConfig(),
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone
        self.cfg = cfg
        if self.backbone is None and not cfg.allow_debug_identity_without_backbone:
            raise ValueError("diffusion backbone is required when debug identity mode is disabled")
        if self.backbone is not None and not cfg.train_backbone_initially:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    @property
    def has_pretrained_file(self) -> bool:
        return Path(self.cfg.pretrained_path).exists()

    def forward(
        self,
        x_aux: torch.Tensor,
        s_hat: torch.Tensor,
        d_hat: torch.Tensor | None,
        rate_level: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        cond = self.adapter(s_hat, d_hat, rate_level)
        if self.backbone is None:
            return x_aux
        return self.backbone(x_aux, cond)

