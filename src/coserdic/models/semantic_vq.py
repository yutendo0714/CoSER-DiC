from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ResBlock, ResDownBlock, UpBlock


@dataclass(frozen=True)
class SemanticVQConfig:
    base_channels: int = 128
    latent_channels: int = 256
    codebook_size: int = 8192
    commitment_weight: float = 0.25
    ema_update: bool = True
    ema_decay: float = 0.99
    ema_eps: float = 1.0e-5
    num_res_blocks: int = 4
    usage_temperature: float = 1.0
    codebook_init_std: float = 0.5
    soft_st: bool = True
    soft_st_temperature: float = 1.0


class SemanticEncoder(nn.Module):
    def __init__(self, cfg: SemanticVQConfig = SemanticVQConfig()) -> None:
        super().__init__()
        c = cfg.base_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, stride=2, padding=1),
            ResDownBlock(c, c),
            ResDownBlock(c, c + 64),
            ResDownBlock(c + 64, cfg.latent_channels),
            ResDownBlock(cfg.latent_channels, cfg.latent_channels),
            *[ResBlock(cfg.latent_channels) for _ in range(cfg.num_res_blocks)],
            nn.Conv2d(cfg.latent_channels, cfg.latent_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemanticAuxiliaryDecoder(nn.Module):
    def __init__(self, cfg: SemanticVQConfig = SemanticVQConfig()) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *[ResBlock(cfg.latent_channels) for _ in range(cfg.num_res_blocks)],
            UpBlock(cfg.latent_channels, cfg.latent_channels),
            UpBlock(cfg.latent_channels, 192),
            UpBlock(192, 128),
            UpBlock(128, 96),
            UpBlock(96, 64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, s_hat: torch.Tensor) -> torch.Tensor:
        return self.net(s_hat)


class DifferentiableVQ(nn.Module):
    def __init__(self, cfg: SemanticVQConfig = SemanticVQConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.codebook_size, cfg.latent_channels)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=cfg.codebook_init_std)
        self.register_buffer("ema_cluster_size", torch.ones(cfg.codebook_size))
        self.register_buffer("ema_embed", self.embedding.weight.detach().clone())

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        b, c, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, c)
        flat_fp32 = flat.float()
        embed_fp32 = self.embedding.weight.float()
        distances = (
            flat_fp32.pow(2).sum(dim=1, keepdim=True)
            + embed_fp32.pow(2).sum(dim=1)
            - 2.0 * flat_fp32 @ embed_fp32.t()
        )
        soft_embed_fp32 = embed_fp32.detach() if self.cfg.ema_update else embed_fp32
        soft_distances = (
            flat_fp32.pow(2).sum(dim=1, keepdim=True)
            + soft_embed_fp32.pow(2).sum(dim=1)
            - 2.0 * flat_fp32 @ soft_embed_fp32.t()
        )
        assignment_probs = F.softmax(
            -soft_distances / max(self.cfg.soft_st_temperature, 1.0e-6),
            dim=1,
        )
        usage_probs = F.softmax(-soft_distances / max(self.cfg.usage_temperature, 1.0e-6), dim=1)
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.cfg.codebook_size).to(flat_fp32.dtype)
        quant_flat = encodings @ embed_fp32
        quant = quant_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous().to(z.dtype)
        soft_quant_flat = assignment_probs @ soft_embed_fp32
        soft_quant = soft_quant_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous().to(z.dtype)

        if self.training and self.cfg.ema_update:
            self._ema_update(flat_fp32, encodings)

        commitment = F.mse_loss(z, quant.detach())
        if self.cfg.ema_update:
            codebook_loss = z.new_tensor(0.0)
        else:
            codebook_loss = F.mse_loss(quant, z.detach())
        loss = codebook_loss + self.cfg.commitment_weight * commitment
        if self.cfg.soft_st:
            quant_st = soft_quant + (quant - soft_quant).detach()
        else:
            quant_st = z + (quant - z).detach()

        avg_probs = encodings.float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1.0e-10)))
        avg_soft_probs = usage_probs.mean(dim=0)
        soft_entropy = -torch.sum(avg_soft_probs * torch.log(avg_soft_probs + 1.0e-10))
        usage_loss = 1.0 - soft_entropy / torch.log(z.new_tensor(float(self.cfg.codebook_size)))
        soft_perplexity = torch.exp(soft_entropy)
        used_codes = torch.count_nonzero(encodings.sum(dim=0)).to(z.dtype)
        dead_code_ratio = 1.0 - used_codes / float(self.cfg.codebook_size)

        return {
            "quantized": quant_st,
            "indices": indices.view(b, h, w),
            "loss": loss,
            "commitment_loss": commitment.detach(),
            "codebook_loss": codebook_loss.detach(),
            "perplexity": perplexity.detach(),
            "soft_perplexity": soft_perplexity.detach(),
            "usage_loss": usage_loss,
            "dead_code_ratio": dead_code_ratio.detach(),
            "used_codes": used_codes.detach(),
        }

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, encodings: torch.Tensor) -> None:
        cluster_size = encodings.sum(dim=0)
        embed_sum = encodings.t() @ flat
        self.ema_cluster_size.mul_(self.cfg.ema_decay).add_(cluster_size, alpha=1 - self.cfg.ema_decay)
        self.ema_embed.mul_(self.cfg.ema_decay).add_(embed_sum, alpha=1 - self.cfg.ema_decay)

        n = self.ema_cluster_size.sum()
        smoothed = (
            (self.ema_cluster_size + self.cfg.ema_eps)
            / (n + self.cfg.codebook_size * self.cfg.ema_eps)
            * n
        )
        normalized = self.ema_embed / smoothed.unsqueeze(1).clamp_min(self.cfg.ema_eps)
        self.embedding.weight.data.copy_(normalized)


class SemanticVQAutoEncoder(nn.Module):
    def __init__(self, cfg: SemanticVQConfig = SemanticVQConfig()) -> None:
        super().__init__()
        self.encoder = SemanticEncoder(cfg)
        self.vq = DifferentiableVQ(cfg)
        self.decoder = SemanticAuxiliaryDecoder(cfg)

    def forward(self, x: torch.Tensor, quantize_mix: float = 1.0) -> dict[str, torch.Tensor]:
        if not 0.0 <= quantize_mix <= 1.0:
            raise ValueError("quantize_mix must be in [0, 1]")
        h = self.encoder(x)
        q = self.vq(h)
        decoder_input = quantize_mix * q["quantized"] + (1.0 - quantize_mix) * h
        x_sem = self.decoder(decoder_input)
        return {"x_sem": x_sem, "h": h, **q}
