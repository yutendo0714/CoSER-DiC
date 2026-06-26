from __future__ import annotations

import torch
from torch import nn


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            group_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            group_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.net = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ResBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

