from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ConditioningAdapterConfig:
    semantic_channels: int = 256
    detail_channels: int = 64
    hidden_channels: int = 256
    rate_levels: int = 3
    rate_embed_dim: int = 128


class CoSERConditioningAdapter(nn.Module):
    """Small CoSER-owned adapter between semantic/detail streams and decoder.

    This is deliberately separate from any external diffusion repo so that
    CoD-Lite remains an initialization/reference, not the CoSER-DiC method.
    """

    def __init__(self, cfg: ConditioningAdapterConfig = ConditioningAdapterConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.semantic_proj = nn.Conv2d(cfg.semantic_channels, cfg.hidden_channels, kernel_size=1)
        self.detail_proj = nn.Conv2d(cfg.detail_channels, cfg.hidden_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(cfg.hidden_channels * 2, cfg.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cfg.hidden_channels, cfg.hidden_channels, kernel_size=1),
        )
        self.rate_embedding = nn.Embedding(cfg.rate_levels, cfg.rate_embed_dim)
        self.rate_to_scale_shift = nn.Linear(cfg.rate_embed_dim, cfg.hidden_channels * 2)

    def forward(
        self,
        s_hat: torch.Tensor,
        d_hat: torch.Tensor | None,
        rate_level: torch.Tensor | int | None = None,
    ) -> dict[str, torch.Tensor]:
        if d_hat is None:
            d_hat = s_hat.new_zeros(
                s_hat.shape[0], self.cfg.detail_channels, s_hat.shape[-2], s_hat.shape[-1]
            )
        if s_hat.shape[-2:] != d_hat.shape[-2:]:
            raise ValueError("semantic and detail conditions must share spatial shape")

        cond = self.fuse(torch.cat([self.semantic_proj(s_hat), self.detail_proj(d_hat)], dim=1))

        if rate_level is None:
            rate_level = torch.zeros(s_hat.shape[0], dtype=torch.long, device=s_hat.device)
        elif isinstance(rate_level, int):
            rate_level = torch.full((s_hat.shape[0],), rate_level, dtype=torch.long, device=s_hat.device)
        else:
            rate_level = rate_level.to(device=s_hat.device, dtype=torch.long).view(-1)

        scale_shift = self.rate_to_scale_shift(self.rate_embedding(rate_level))
        scale, shift = scale_shift.chunk(2, dim=1)
        return {
            "condition": cond,
            "film_scale": scale[:, :, None, None],
            "film_shift": shift[:, :, None, None],
            "rate_level": rate_level,
        }

