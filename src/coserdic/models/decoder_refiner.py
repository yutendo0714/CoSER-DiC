from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ResBlock


@dataclass(frozen=True)
class DecoderSideRefinerConfig:
    image_channels: int = 3
    semantic_channels: int = 256
    detail_channels: int = 3
    base_channels: int = 64
    semantic_context_channels: int = 32
    num_res_blocks: int = 6
    residual_scale: float = 0.05
    use_semantic_latent: bool = True
    zero_init_output: bool = True


class DecoderSideRefiner(nn.Module):
    """Fixed decoder-side refinement module for Stage 4.

    The refiner only consumes signals the decoder can reproduce from transmitted
    Stage 3 payloads and fixed model weights. It must not consume per-image
    masks, prompts, maps, or controls unless those are explicitly bit-counted.
    """

    def __init__(self, cfg: DecoderSideRefinerConfig = DecoderSideRefinerConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        context_channels = int(cfg.semantic_context_channels) if cfg.use_semantic_latent else 0
        if cfg.use_semantic_latent:
            self.semantic_proj: nn.Module = nn.Sequential(
                nn.Conv2d(cfg.semantic_channels, context_channels, kernel_size=1),
                nn.SiLU(inplace=True),
            )
        else:
            self.semantic_proj = nn.Identity()
        input_channels = cfg.image_channels * 2 + cfg.detail_channels + context_channels
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, cfg.base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        ]
        layers.extend(ResBlock(cfg.base_channels) for _ in range(cfg.num_res_blocks))
        layers.extend(
            [
                nn.Conv2d(cfg.base_channels, cfg.base_channels, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(cfg.base_channels, cfg.image_channels, kernel_size=3, padding=1),
            ]
        )
        self.net = nn.Sequential(*layers)
        if cfg.zero_init_output:
            last = self.net[-1]
            if isinstance(last, nn.Conv2d):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(
        self,
        x_stage3: torch.Tensor,
        x_sem: torch.Tensor,
        detail_residual: torch.Tensor,
        semantic_latent: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x_stage3.shape != x_sem.shape:
            raise ValueError("x_stage3 and x_sem must share BCHW shape")
        if detail_residual.shape[-2:] != x_stage3.shape[-2:]:
            detail_residual = F.interpolate(
                detail_residual,
                size=tuple(int(v) for v in x_stage3.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
        if detail_residual.shape[:2] != (x_stage3.shape[0], self.cfg.detail_channels):
            raise ValueError("detail_residual must have the configured detail channels")

        inputs = [x_stage3, x_sem, detail_residual]
        if self.cfg.use_semantic_latent:
            if semantic_latent is None:
                raise ValueError("semantic_latent is required when use_semantic_latent is true")
            semantic_context = self.semantic_proj(semantic_latent)
            semantic_context = F.interpolate(
                semantic_context,
                size=tuple(int(v) for v in x_stage3.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
            inputs.append(semantic_context)

        residual = float(self.cfg.residual_scale) * torch.tanh(self.net(torch.cat(inputs, dim=1)))
        x_refined = (x_stage3 + residual).clamp(0.0, 1.0)
        return {"x_refined": x_refined, "refiner_residual": residual}
