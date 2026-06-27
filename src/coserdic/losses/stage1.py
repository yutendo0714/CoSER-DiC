from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torch import nn


@dataclass(frozen=True)
class Stage1LossWeights:
    l1_sem: float = 1.0
    ms_ssim_sem: float = 0.0
    lpips_sem: float = 0.0
    vq: float = 1.0
    codebook_usage: float = 0.01


class Stage1SemanticVQLoss(nn.Module):
    def __init__(self, weights: Stage1LossWeights, use_lpips: bool = False) -> None:
        super().__init__()
        self.weights = weights
        self.lpips_model = None
        if use_lpips and weights.lpips_sem > 0:
            import lpips

            self.lpips_model = lpips.LPIPS(net="alex").eval()
            for param in self.lpips_model.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        out: dict[str, torch.Tensor],
        vq_scale: float = 1.0,
        usage_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        x_sem = out["x_sem"]
        l1 = F.l1_loss(x_sem, x)
        if self.weights.ms_ssim_sem > 0:
            ms = 1.0 - ms_ssim(x_sem.float(), x.float(), data_range=1.0, size_average=True)
        else:
            ms = x.new_tensor(0.0)
        if self.lpips_model is None:
            lp = x.new_tensor(0.0)
        else:
            self.lpips_model = self.lpips_model.to(x.device)
            lp = self.lpips_model(x_sem * 2.0 - 1.0, x * 2.0 - 1.0).mean()

        # Differentiable batch-level entropy pressure against immediate codebook
        # collapse. Hard perplexity remains a diagnostic.
        usage = out["usage_loss"]
        total = (
            self.weights.l1_sem * l1
            + self.weights.ms_ssim_sem * ms
            + self.weights.lpips_sem * lp
            + self.weights.vq * vq_scale * out["loss"]
            + self.weights.codebook_usage * usage_scale * usage
        )
        return {
            "total": total,
            "l1_sem": l1.detach(),
            "ms_ssim_sem": ms.detach(),
            "lpips_sem": lp.detach(),
            "vq": out["loss"].detach(),
            "vq_commitment": out["commitment_loss"].detach(),
            "vq_codebook": out["codebook_loss"].detach(),
            "usage": usage.detach(),
        }
