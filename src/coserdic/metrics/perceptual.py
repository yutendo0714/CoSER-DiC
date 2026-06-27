from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PerceptualMetricResult:
    lpips_alex: float
    dists: float


class PerceptualMetricBundle(nn.Module):
    """LPIPS/DISTS bundle for evaluation-time perceptual quality metrics."""

    def __init__(self) -> None:
        super().__init__()
        import lpips
        from DISTS_pytorch import DISTS

        self.lpips_alex = lpips.LPIPS(net="alex").eval()
        self.dists = DISTS().eval()
        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, reference: torch.Tensor, reconstruction: torch.Tensor) -> PerceptualMetricResult:
        ref = reference.float().clamp(0.0, 1.0)
        rec = reconstruction.float().clamp(0.0, 1.0)
        lpips_value = self.lpips_alex(rec * 2.0 - 1.0, ref * 2.0 - 1.0).mean()
        dists_value = self.dists(rec, ref).mean()
        return PerceptualMetricResult(
            lpips_alex=float(lpips_value.item()),
            dists=float(dists_value.item()),
        )
