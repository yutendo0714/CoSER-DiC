from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .blocks import ResBlock, ResDownBlock, UpBlock


@dataclass(frozen=True)
class ResidualDetailConfig:
    input_channels: int = 12
    semantic_channels: int = 256
    base_channels: int = 64
    hidden_channels: int = 128
    detail_channels: int = 3
    num_res_blocks: int = 3
    quant_bits: int = 5
    value_range: float = 0.25
    residual_scale: float = 0.25


class UniformScalarQuantizer(nn.Module):
    """Uniform scalar quantizer with a differentiable training path."""

    def __init__(self, bits: int, value_range: float) -> None:
        super().__init__()
        if not 1 <= int(bits) <= 16:
            raise ValueError("bits must be in [1, 16]")
        if float(value_range) <= 0:
            raise ValueError("value_range must be positive")
        self.bits = int(bits)
        self.value_range = float(value_range)

    @property
    def levels(self) -> int:
        return 1 << self.bits

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        clipped = x.clamp(-self.value_range, self.value_range)
        normalized = (clipped + self.value_range) / (2.0 * self.value_range)
        return torch.round(normalized * float(self.levels - 1)).to(torch.long)

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        values = codes.to(dtype=torch.float32)
        normalized = values / float(self.levels - 1)
        out = normalized * (2.0 * self.value_range) - self.value_range
        return out.to(device=codes.device)

    def forward(self, x: torch.Tensor, training_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        clipped = x.clamp(-self.value_range, self.value_range)
        normalized = (clipped + self.value_range) / (2.0 * self.value_range)
        scaled = normalized * float(self.levels - 1)
        if self.training and training_noise:
            noisy = scaled + torch.empty_like(scaled).uniform_(-0.5, 0.5)
            quantized = noisy.clamp(0.0, float(self.levels - 1))
        else:
            rounded = torch.round(scaled)
            quantized = scaled + (rounded - scaled).detach()
        dequantized = quantized / float(self.levels - 1) * (2.0 * self.value_range) - self.value_range
        codes = torch.round(scaled).clamp(0, self.levels - 1).to(torch.long)
        return dequantized, codes


class ResidualDetailEncoder(nn.Module):
    def __init__(self, cfg: ResidualDetailConfig = ResidualDetailConfig()) -> None:
        super().__init__()
        c = int(cfg.base_channels)
        h = int(cfg.hidden_channels)
        self.value_range = float(cfg.value_range)
        self.down = nn.Sequential(
            nn.Conv2d(cfg.input_channels, c, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            ResDownBlock(c, c),
            ResDownBlock(c, h),
            ResDownBlock(h, h),
            ResDownBlock(h, h),
            *[ResBlock(h) for _ in range(cfg.num_res_blocks)],
        )
        self.semantic_film = nn.Conv2d(cfg.semantic_channels, 2 * h, kernel_size=1)
        self.out = nn.Conv2d(h, cfg.detail_channels, kernel_size=1)
        nn.init.zeros_(self.semantic_film.weight)
        nn.init.zeros_(self.semantic_film.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, residual_input: torch.Tensor, semantic_latent: torch.Tensor) -> torch.Tensor:
        h = self.down(residual_input)
        gamma, beta = self.semantic_film(semantic_latent).chunk(2, dim=1)
        h = h * (1.0 + gamma) + beta
        return self.value_range * torch.tanh(self.out(h))


class ResidualDetailDecoder(nn.Module):
    def __init__(self, cfg: ResidualDetailConfig = ResidualDetailConfig()) -> None:
        super().__init__()
        h = int(cfg.hidden_channels)
        self.residual_scale = float(cfg.residual_scale)
        self.net = nn.Sequential(
            nn.Conv2d(cfg.semantic_channels + cfg.detail_channels, h, kernel_size=1),
            *[ResBlock(h) for _ in range(cfg.num_res_blocks)],
            UpBlock(h, h),
            UpBlock(h, h),
            UpBlock(h, h),
            UpBlock(h, cfg.base_channels),
            UpBlock(cfg.base_channels, cfg.base_channels),
            nn.Conv2d(cfg.base_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, semantic_latent: torch.Tensor, detail_latent: torch.Tensor, x_sem: torch.Tensor) -> dict[str, torch.Tensor]:
        residual = self.residual_scale * torch.tanh(self.net(torch.cat([semantic_latent, detail_latent], dim=1)))
        x_aux = (x_sem + residual).clamp(0.0, 1.0)
        return {"x_aux": x_aux, "residual_pred": residual}


class ResidualDetailAutoEncoder(nn.Module):
    """Semantic-conditioned residual branch for the CoSER-DiC Core MVP."""

    def __init__(self, cfg: ResidualDetailConfig = ResidualDetailConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ResidualDetailEncoder(cfg)
        self.quantizer = UniformScalarQuantizer(cfg.quant_bits, cfg.value_range)
        self.decoder = ResidualDetailDecoder(cfg)

    def forward(
        self,
        x: torch.Tensor,
        x_sem: torch.Tensor,
        semantic_latent: torch.Tensor,
        *,
        training_noise: bool = True,
    ) -> dict[str, torch.Tensor]:
        residual = x - x_sem
        residual_input = torch.cat([x, x_sem, residual, residual.abs()], dim=1)
        h_detail = self.encoder(residual_input, semantic_latent)
        detail_latent, detail_codes = self.quantizer(h_detail, training_noise=training_noise)
        decoded = self.decoder(semantic_latent, detail_latent, x_sem)
        return {
            "x_aux": decoded["x_aux"],
            "residual_pred": decoded["residual_pred"],
            "h_detail": h_detail,
            "detail_latent": detail_latent,
            "detail_codes": detail_codes,
        }
