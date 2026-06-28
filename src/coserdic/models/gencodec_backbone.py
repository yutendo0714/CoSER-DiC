from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import math
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class CoDLiteOneStepBackboneConfig:
    repo_root: str = "external/repos/GenCodec/CoD_Lite"
    checkpoint_path: str = "external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt"
    config_path: str = "external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml"
    freeze: bool = True
    output_range: str = "zero_one"
    use_zero_image_input: bool = True


class CoDLiteOneStepBackbone(nn.Module):
    """Thin CoSER-owned wrapper around the official CoD-Lite one-step decoder.

    The wrapper deliberately does not expose the native CoD-Lite bitstream as a
    CoSER stream. It only provides a frozen or low-lr decoder backbone that can
    consume a condition tensor produced from decoded CoSER information.
    """

    def __init__(self, net: nn.Module, cfg: CoDLiteOneStepBackboneConfig) -> None:
        super().__init__()
        self.net = net
        self.cfg = cfg
        if cfg.freeze:
            for parameter in self.net.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def load(cls, cfg: CoDLiteOneStepBackboneConfig, device: str | torch.device = "cuda") -> "CoDLiteOneStepBackbone":
        repo_root = Path(cfg.repo_root).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"CoD-Lite repo not found: {repo_root}")
        checkpoint_path = Path(cfg.checkpoint_path)
        config_path = Path(cfg.config_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"CoD-Lite checkpoint not found: {checkpoint_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"CoD-Lite config not found: {config_path}")
        _prepend_sys_path(repo_root)

        from finetuned_one_step_codec.inference import load_net

        net = load_net(str(checkpoint_path), str(config_path)).to(device)
        return cls(net=net, cfg=cfg)

    @property
    def condition_channels(self) -> int:
        return int(self.net.hidden_size)

    def condition_size(self, height: int, width: int) -> tuple[int, int]:
        h, w = self.net.y_embedder.calculate_indices_size(height, width)
        return int(h), int(w)

    def native_condition(self, image: torch.Tensor) -> torch.Tensor:
        """Return the official CoD-Lite condition for supervision/debug only."""

        cond, _ = self.net.y_embedder(image)
        return cond

    def forward(self, x_aux: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.shape[1] != self.condition_channels:
            raise ValueError(
                f"condition channel mismatch: got {cond.shape[1]}, expected {self.condition_channels}"
            )
        expected_hw = self.condition_size(int(x_aux.shape[-2]), int(x_aux.shape[-1]))
        if tuple(cond.shape[-2:]) != expected_hw:
            raise ValueError(f"condition spatial shape mismatch: got {tuple(cond.shape[-2:])}, expected {expected_hw}")

        y = torch.zeros_like(x_aux) if self.cfg.use_zero_image_input else x_aux
        raw = self.net.inference(y=y, cond=cond)
        if self.cfg.output_range == "minus_one_one":
            return raw
        if self.cfg.output_range != "zero_one":
            raise ValueError(f"unknown output_range: {self.cfg.output_range}")
        return ((raw + 1.0) * 0.5).clamp(0.0, 1.0)


@dataclass(frozen=True)
class CoSERToCoDLiteConditionAdapterConfig:
    semantic_channels: int = 256
    detail_context_channels: int = 0
    condition_channels: int = 384
    hidden_channels: int = 128
    image_feature_channels: int = 9
    zero_init_output: bool = True


class _ConditionResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CoSERToCoDLiteConditionAdapter(nn.Module):
    """Project decoded CoSER information into the CoD-Lite condition space."""

    def __init__(self, cfg: CoSERToCoDLiteConditionAdapterConfig = CoSERToCoDLiteConditionAdapterConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_encoder = nn.Sequential(
            nn.Conv2d(cfg.image_feature_channels, cfg.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cfg.hidden_channels, cfg.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(cfg.semantic_channels, cfg.hidden_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.detail_proj = None
        fusion_inputs = 2
        if cfg.detail_context_channels > 0:
            self.detail_proj = nn.Sequential(
                nn.Conv2d(cfg.detail_context_channels, cfg.hidden_channels, kernel_size=1),
                nn.SiLU(),
            )
            fusion_inputs += 1
        self.fuse = nn.Sequential(
            nn.Conv2d(cfg.hidden_channels * fusion_inputs, cfg.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cfg.hidden_channels, cfg.condition_channels, kernel_size=1),
        )
        if cfg.zero_init_output:
            final = self.fuse[-1]
            if isinstance(final, nn.Conv2d):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def forward(
        self,
        x_aux: torch.Tensor,
        x_sem: torch.Tensor,
        residual_hat: torch.Tensor,
        semantic_latent: torch.Tensor,
        *,
        condition_size: tuple[int, int],
        base_condition: torch.Tensor | None = None,
        detail_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_aux.shape != x_sem.shape or x_aux.shape != residual_hat.shape:
            raise ValueError("x_aux, x_sem, and residual_hat must share shape")
        image_features = torch.cat([x_aux, x_sem, residual_hat], dim=1)
        if image_features.shape[1] != self.cfg.image_feature_channels:
            raise ValueError(
                f"image feature channel mismatch: got {image_features.shape[1]}, expected {self.cfg.image_feature_channels}"
            )

        image_features = self.image_encoder(image_features)
        image_features = F.interpolate(image_features, size=condition_size, mode="bilinear", align_corners=False)

        semantic_features = self.semantic_proj(semantic_latent)
        semantic_features = F.interpolate(semantic_features, size=condition_size, mode="bilinear", align_corners=False)

        features = [image_features, semantic_features]
        if self.detail_proj is not None:
            if detail_context is None:
                raise ValueError("detail_context is required when detail_context_channels > 0")
            if detail_context.shape[1] != self.cfg.detail_context_channels:
                raise ValueError(
                    "detail_context channel mismatch: "
                    f"got {detail_context.shape[1]}, expected {self.cfg.detail_context_channels}"
                )
            detail_features = self.detail_proj(detail_context)
            detail_features = F.interpolate(detail_features, size=condition_size, mode="bilinear", align_corners=False)
            features.append(detail_features)

        return self.fuse(torch.cat(features, dim=1))


@dataclass(frozen=True)
class CoSERToCoDLiteConditionPyramidAdapterConfig:
    semantic_channels: int = 3
    detail_context_channels: int = 0
    condition_channels: int = 384
    hidden_channels: int = 192
    image_feature_channels: int = 9
    num_image_blocks: int = 4
    num_condition_blocks: int = 4
    num_fusion_blocks: int = 4
    zero_init_output: bool = True


class CoSERToCoDLiteConditionPyramidAdapter(nn.Module):
    """Stronger decoder-side condition refiner for CoD-Lite integration.

    The key difference from the lightweight adapter is that this module sees
    the decoder-available native CoD-Lite condition computed from Stage 3. It
    therefore learns a residual correction in the same condition space instead
    of inferring the correction only from image-space reconstructions.
    """

    def __init__(
        self,
        cfg: CoSERToCoDLiteConditionPyramidAdapterConfig = CoSERToCoDLiteConditionPyramidAdapterConfig(),
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_in = nn.Sequential(
            nn.Conv2d(cfg.image_feature_channels, cfg.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.image_blocks = nn.Sequential(*[_ConditionResidualBlock(cfg.hidden_channels) for _ in range(cfg.num_image_blocks)])
        self.condition_in = nn.Sequential(
            nn.Conv2d(cfg.condition_channels, cfg.hidden_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.condition_blocks = nn.Sequential(
            *[_ConditionResidualBlock(cfg.hidden_channels) for _ in range(cfg.num_condition_blocks)]
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(cfg.semantic_channels, cfg.hidden_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.detail_proj = None
        fusion_inputs = 3
        if cfg.detail_context_channels > 0:
            self.detail_proj = nn.Sequential(
                nn.Conv2d(cfg.detail_context_channels, cfg.hidden_channels, kernel_size=1),
                nn.SiLU(),
            )
            fusion_inputs += 1
        self.fuse_in = nn.Sequential(
            nn.Conv2d(cfg.hidden_channels * fusion_inputs, cfg.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.fusion_blocks = nn.Sequential(
            *[_ConditionResidualBlock(cfg.hidden_channels) for _ in range(cfg.num_fusion_blocks)]
        )
        self.out = nn.Conv2d(cfg.hidden_channels, cfg.condition_channels, kernel_size=1)
        if cfg.zero_init_output:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x_aux: torch.Tensor,
        x_sem: torch.Tensor,
        residual_hat: torch.Tensor,
        semantic_latent: torch.Tensor,
        *,
        condition_size: tuple[int, int],
        base_condition: torch.Tensor | None = None,
        detail_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base_condition is None:
            raise ValueError("base_condition is required for CoSERToCoDLiteConditionPyramidAdapter")
        if x_aux.shape != x_sem.shape or x_aux.shape != residual_hat.shape:
            raise ValueError("x_aux, x_sem, and residual_hat must share shape")
        if base_condition.shape[1] != self.cfg.condition_channels:
            raise ValueError(
                "base_condition channel mismatch: "
                f"got {base_condition.shape[1]}, expected {self.cfg.condition_channels}"
            )
        if tuple(base_condition.shape[-2:]) != condition_size:
            raise ValueError(
                f"base_condition spatial shape mismatch: got {tuple(base_condition.shape[-2:])}, expected {condition_size}"
            )
        base_condition = base_condition.to(dtype=x_aux.dtype)

        image_features = torch.cat([x_aux, x_sem, residual_hat], dim=1)
        if image_features.shape[1] != self.cfg.image_feature_channels:
            raise ValueError(
                f"image feature channel mismatch: got {image_features.shape[1]}, expected {self.cfg.image_feature_channels}"
            )
        image_features = F.interpolate(image_features, size=condition_size, mode="bilinear", align_corners=False)
        image_features = self.image_blocks(self.image_in(image_features))

        condition_features = self.condition_blocks(self.condition_in(base_condition))

        semantic_features = self.semantic_proj(semantic_latent)
        semantic_features = F.interpolate(semantic_features, size=condition_size, mode="bilinear", align_corners=False)

        features = [image_features, condition_features, semantic_features]
        if self.detail_proj is not None:
            if detail_context is None:
                raise ValueError("detail_context is required when detail_context_channels > 0")
            if detail_context.shape[1] != self.cfg.detail_context_channels:
                raise ValueError(
                    "detail_context channel mismatch: "
                    f"got {detail_context.shape[1]}, expected {self.cfg.detail_context_channels}"
                )
            detail_features = self.detail_proj(detail_context)
            detail_features = F.interpolate(detail_features, size=condition_size, mode="bilinear", align_corners=False)
            features.append(detail_features)

        fused = self.fuse_in(torch.cat(features, dim=1))
        fused = self.fusion_blocks(fused)
        return self.out(fused)


@dataclass(frozen=True)
class CoSERToCoDLiteAlphaGateConfig:
    semantic_channels: int = 256
    detail_context_channels: int = 0
    condition_channels: int = 384
    hidden_channels: int = 128
    image_feature_channels: int = 9
    num_blocks: int = 2
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    init_alpha: float = 0.3


class CoSERToCoDLiteAlphaGate(nn.Module):
    """Predict a deterministic decoder-side blend strength from decoded CoSER state."""

    def __init__(self, cfg: CoSERToCoDLiteAlphaGateConfig = CoSERToCoDLiteAlphaGateConfig()) -> None:
        super().__init__()
        if not 0.0 <= cfg.alpha_min < cfg.alpha_max <= 1.0:
            raise ValueError("alpha_min/alpha_max must satisfy 0 <= min < max <= 1")
        if not cfg.alpha_min < cfg.init_alpha < cfg.alpha_max:
            raise ValueError("init_alpha must be inside (alpha_min, alpha_max)")
        self.cfg = cfg
        self.image_proj = nn.Sequential(
            nn.Conv2d(cfg.image_feature_channels, cfg.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(cfg.semantic_channels, cfg.hidden_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.condition_proj = nn.Sequential(
            nn.Conv2d(cfg.condition_channels * 2, cfg.hidden_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.detail_proj = None
        fusion_inputs = 3
        if cfg.detail_context_channels > 0:
            self.detail_proj = nn.Sequential(
                nn.Conv2d(cfg.detail_context_channels, cfg.hidden_channels, kernel_size=1),
                nn.SiLU(),
            )
            fusion_inputs += 1
        self.fuse = nn.Sequential(
            nn.Conv2d(cfg.hidden_channels * fusion_inputs, cfg.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            *[_ConditionResidualBlock(cfg.hidden_channels) for _ in range(cfg.num_blocks)],
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cfg.hidden_channels, cfg.hidden_channels),
            nn.SiLU(),
            nn.Linear(cfg.hidden_channels, 1),
        )
        final = self.head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            normalized = (cfg.init_alpha - cfg.alpha_min) / (cfg.alpha_max - cfg.alpha_min)
            normalized = min(max(normalized, 1.0e-6), 1.0 - 1.0e-6)
            nn.init.constant_(final.bias, math.log(normalized / (1.0 - normalized)))

    def forward(
        self,
        x_aux: torch.Tensor,
        x_sem: torch.Tensor,
        residual_hat: torch.Tensor,
        semantic_latent: torch.Tensor,
        *,
        condition_size: tuple[int, int],
        base_condition: torch.Tensor,
        condition_residual: torch.Tensor,
        detail_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_aux.shape != x_sem.shape or x_aux.shape != residual_hat.shape:
            raise ValueError("x_aux, x_sem, and residual_hat must share shape")
        if tuple(base_condition.shape[-2:]) != condition_size:
            raise ValueError(
                f"base_condition spatial shape mismatch: got {tuple(base_condition.shape[-2:])}, expected {condition_size}"
            )
        if base_condition.shape != condition_residual.shape:
            raise ValueError("base_condition and condition_residual must share shape")
        if base_condition.shape[1] != self.cfg.condition_channels:
            raise ValueError(
                "base_condition channel mismatch: "
                f"got {base_condition.shape[1]}, expected {self.cfg.condition_channels}"
            )

        image_features = torch.cat([x_aux, x_sem, residual_hat], dim=1)
        if image_features.shape[1] != self.cfg.image_feature_channels:
            raise ValueError(
                f"image feature channel mismatch: got {image_features.shape[1]}, expected {self.cfg.image_feature_channels}"
            )
        image_features = F.interpolate(image_features, size=condition_size, mode="bilinear", align_corners=False)
        image_features = self.image_proj(image_features)

        semantic_features = self.semantic_proj(semantic_latent)
        semantic_features = F.interpolate(semantic_features, size=condition_size, mode="bilinear", align_corners=False)

        condition_features = self.condition_proj(torch.cat([base_condition, condition_residual], dim=1))
        features = [image_features, semantic_features, condition_features]
        if self.detail_proj is not None:
            if detail_context is None:
                raise ValueError("detail_context is required when detail_context_channels > 0")
            if detail_context.shape[1] != self.cfg.detail_context_channels:
                raise ValueError(
                    "detail_context channel mismatch: "
                    f"got {detail_context.shape[1]}, expected {self.cfg.detail_context_channels}"
                )
            detail_features = self.detail_proj(detail_context)
            detail_features = F.interpolate(detail_features, size=condition_size, mode="bilinear", align_corners=False)
            features.append(detail_features)

        logit = self.head(self.fuse(torch.cat(features, dim=1)))
        alpha = torch.sigmoid(logit).view(-1, 1, 1, 1)
        return self.cfg.alpha_min + (self.cfg.alpha_max - self.cfg.alpha_min) * alpha


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)
