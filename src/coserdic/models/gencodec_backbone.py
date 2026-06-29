from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import math
import re
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
        self._condition_size_cache: dict[tuple[int, int], tuple[int, int]] = {}
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
        fallback = (int(h), int(w))
        if not bool(getattr(self.net.y_embedder, "up2x", False)):
            return fallback
        key = (int(height), int(width))
        if key in self._condition_size_cache:
            return self._condition_size_cache[key]
        try:
            parameter = next(self.net.parameters())
            sample = torch.zeros(1, 3, int(height), int(width), device=parameter.device, dtype=parameter.dtype)
            with torch.no_grad():
                cond = self.native_condition(sample)
            value = (int(cond.shape[-2]), int(cond.shape[-1]))
        except Exception:
            value = fallback
        self._condition_size_cache[key] = value
        return value

    def native_condition(self, image: torch.Tensor) -> torch.Tensor:
        """Return the official CoD-Lite condition for supervision/debug only."""

        cond, _ = self.net.y_embedder(image)
        return cond

    def forward(self, x_aux: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.shape[1] != self.condition_channels:
            raise ValueError(
                f"condition channel mismatch: got {cond.shape[1]}, expected {self.condition_channels}"
            )
        if x_aux.shape[0] != cond.shape[0]:
            raise ValueError(f"batch size mismatch: x_aux={x_aux.shape[0]}, cond={cond.shape[0]}")
        expected_hw = self.condition_size(int(x_aux.shape[-2]), int(x_aux.shape[-1]))
        if tuple(cond.shape[-2:]) != expected_hw:
            raise ValueError(f"condition spatial shape mismatch: got {tuple(cond.shape[-2:])}, expected {expected_hw}")

        if x_aux.shape[0] > 1:
            return torch.cat(
                [self._forward_one(x_aux[i : i + 1], cond[i : i + 1]) for i in range(x_aux.shape[0])],
                dim=0,
            )
        return self._forward_one(x_aux, cond)

    def _forward_one(self, x_aux: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x_aux) if self.cfg.use_zero_image_input else x_aux
        raw = self.net.inference(y=y, cond=cond)
        if self.cfg.output_range == "minus_one_one":
            return raw
        if self.cfg.output_range != "zero_one":
            raise ValueError(f"unknown output_range: {self.cfg.output_range}")
        return ((raw + 1.0) * 0.5).clamp(0.0, 1.0)


def compile_parameter_name_patterns(patterns: list[str] | tuple[str, ...]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for item in patterns:
        for pattern in str(item).split(","):
            pattern = pattern.strip()
            if pattern:
                compiled.append(re.compile(pattern))
    return compiled


def set_trainable_parameters_by_name(
    module: nn.Module,
    patterns: list[str] | tuple[str, ...],
) -> list[str]:
    """Unfreeze parameters whose names match regex patterns, freezing all others.

    This is intentionally name-based rather than architecture-specific so it
    can target official CoD-Lite / CoD checkpoints without copying their code
    into CoSER-DiC.
    """

    compiled = compile_parameter_name_patterns(patterns)
    if not compiled:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return []
    trainable: list[str] = []
    for name, parameter in module.named_parameters():
        matched = any(pattern.search(name) for pattern in compiled)
        parameter.requires_grad_(matched)
        if matched:
            trainable.append(name)
    return trainable


def condition_residual_rms_guard(
    base_condition: torch.Tensor,
    condition_residual: torch.Tensor,
    *,
    max_rms_ratio: float,
    granularity: str = "global",
    min_gate: float = 0.0,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip condition residual RMS relative to decoder-available base condition.

    This is a deterministic decoder-side guard: it uses only the decoded CoSER
    state and fixed model tensors. It does not use the reference condition and
    does not add transmitted side information.
    """

    if base_condition.shape != condition_residual.shape:
        raise ValueError("base_condition and condition_residual must share shape")
    if max_rms_ratio <= 0:
        raise ValueError("max_rms_ratio must be positive")
    if not 0.0 <= min_gate <= 1.0:
        raise ValueError("min_gate must be in [0, 1]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if granularity == "global":
        dims = (1, 2, 3)
    elif granularity == "spatial":
        dims = (1,)
    elif granularity == "channel":
        dims = (2, 3)
    else:
        raise ValueError(f"unknown condition residual guard granularity: {granularity}")

    base = base_condition.float()
    residual = condition_residual.float()
    base_rms = torch.sqrt(torch.mean(base.pow(2), dim=dims, keepdim=True).clamp_min(eps * eps))
    residual_rms = torch.sqrt(torch.mean(residual.pow(2), dim=dims, keepdim=True).clamp_min(eps * eps))
    max_residual_rms = float(max_rms_ratio) * base_rms
    gate = torch.minimum(torch.ones_like(residual_rms), max_residual_rms / residual_rms.clamp_min(eps))
    if min_gate > 0.0:
        gate = gate.clamp_min(float(min_gate))
    guarded = residual * gate.to(dtype=residual.dtype)
    return guarded.to(dtype=condition_residual.dtype), gate.to(dtype=condition_residual.dtype)


class LoRALinear(nn.Module):
    """Low-rank residual adapter for a frozen linear layer."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.lora_down.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_up.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale


class LoRAConv2d(nn.Module):
    """Low-rank residual adapter for a frozen Conv2d layer."""

    def __init__(self, base: nn.Conv2d, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if base.groups != 1:
            raise ValueError("LoRAConv2d currently supports groups=1 only")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        self.lora_down = nn.Conv2d(
            base.in_channels,
            rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            padding_mode=base.padding_mode,
            bias=False,
        )
        self.lora_up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.lora_down.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_up.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale


def apply_lora_adapters_by_name(
    module: nn.Module,
    patterns: list[str] | tuple[str, ...],
    *,
    rank: int,
    alpha: float,
    target_module_types: list[str] | tuple[str, ...] = ("linear",),
) -> list[str]:
    """Replace matching Linear/Conv2d children with frozen-base LoRA adapters."""

    compiled = compile_parameter_name_patterns(patterns)
    if not compiled:
        return []
    targets = {str(item).lower() for item in target_module_types}
    if not targets <= {"linear", "conv2d"}:
        raise ValueError(f"unknown LoRA target module types: {sorted(targets)}")
    applied: list[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (LoRALinear, LoRAConv2d)):
                continue
            matched = any(pattern.search(full_name) for pattern in compiled)
            replaced = False
            if matched and isinstance(child, nn.Linear) and "linear" in targets:
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                applied.append(full_name)
                replaced = True
            elif matched and isinstance(child, nn.Conv2d) and "conv2d" in targets and child.groups == 1:
                setattr(parent, child_name, LoRAConv2d(child, rank=rank, alpha=alpha))
                applied.append(full_name)
                replaced = True
            if not replaced:
                visit(child, full_name)

    visit(module)
    return applied


def apply_lora_adapters_from_config(module: nn.Module, config: object) -> list[str]:
    if not isinstance(config, dict):
        return []
    patterns = config.get("patterns", [])
    if not patterns:
        return []
    rank = int(config.get("rank", 0))
    alpha = float(config.get("alpha", 0.0))
    target_module_types = config.get("target_module_types", ("linear",))
    if not isinstance(target_module_types, (list, tuple)):
        target_module_types = (str(target_module_types),)
    return apply_lora_adapters_by_name(
        module,
        [str(pattern) for pattern in patterns],
        rank=rank,
        alpha=alpha,
        target_module_types=tuple(str(item) for item in target_module_types),
    )


def lora_parameter_names(module: nn.Module) -> list[str]:
    return [
        name
        for name, parameter in module.named_parameters()
        if (".lora_down." in name or ".lora_up." in name) and parameter.requires_grad
    ]


def named_parameter_state(module: nn.Module, names: list[str] | tuple[str, ...]) -> dict[str, torch.Tensor]:
    parameters = dict(module.named_parameters())
    state: dict[str, torch.Tensor] = {}
    for name in names:
        if name not in parameters:
            raise KeyError(f"unknown parameter name: {name}")
        state[name] = parameters[name].detach().cpu().clone()
    return state


def load_named_parameter_state(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> list[str]:
    parameters = dict(module.named_parameters())
    loaded: list[str] = []
    missing: list[str] = []
    incompatible: list[str] = []
    with torch.no_grad():
        for name, tensor in state.items():
            if name not in parameters:
                missing.append(name)
                continue
            parameter = parameters[name]
            if tuple(parameter.shape) != tuple(tensor.shape):
                incompatible.append(name)
                continue
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.append(name)
    if strict and (missing or incompatible):
        raise RuntimeError(
            "failed to load named parameter state: "
            f"missing={missing}, incompatible_shape={incompatible}"
        )
    return loaded


@dataclass(frozen=True)
class CoSERToCoDLiteConditionAdapterConfig:
    semantic_channels: int = 256
    detail_context_channels: int = 0
    condition_channels: int = 384
    hidden_channels: int = 128
    image_feature_channels: int = 9
    zero_init_output: bool = True


class _ConditionResidualBlock(nn.Module):
    def __init__(self, channels: int, *, zero_init_last: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        if zero_init_last:
            final = self.net[-1]
            if isinstance(final, nn.Conv2d):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

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
    num_detail_blocks: int = 0
    num_fusion_blocks: int = 4
    detail_control_branch: bool = False
    detail_control_blocks: int = 0
    detail_control_condition_fusion: bool = False
    detail_highfreq_context_branch: bool = False
    detail_film_modulation: bool = False
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
        if cfg.detail_film_modulation and cfg.detail_context_channels <= 0:
            raise ValueError("detail_film_modulation requires detail_context_channels > 0")
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
            if cfg.detail_highfreq_context_branch:
                self.detail_highfreq_proj = nn.Conv2d(
                    cfg.detail_context_channels * 2,
                    cfg.hidden_channels,
                    kernel_size=1,
                )
                nn.init.zeros_(self.detail_highfreq_proj.weight)
                nn.init.zeros_(self.detail_highfreq_proj.bias)
            else:
                self.detail_highfreq_proj = None
            self.detail_blocks = nn.Sequential(
                *[
                    _ConditionResidualBlock(cfg.hidden_channels, zero_init_last=True)
                    for _ in range(cfg.num_detail_blocks)
                ]
            )
            if cfg.detail_control_branch:
                if cfg.detail_control_condition_fusion:
                    self.detail_control_in = nn.Sequential(
                        nn.Conv2d(cfg.hidden_channels * 2, cfg.hidden_channels, kernel_size=1),
                        nn.SiLU(),
                    )
                else:
                    self.detail_control_in = nn.Identity()
                self.detail_control_blocks = nn.Sequential(
                    *[
                        _ConditionResidualBlock(cfg.hidden_channels, zero_init_last=True)
                        for _ in range(cfg.detail_control_blocks)
                    ]
                )
                self.detail_control_out = nn.Conv2d(cfg.hidden_channels, cfg.condition_channels, kernel_size=1)
                nn.init.zeros_(self.detail_control_out.weight)
                nn.init.zeros_(self.detail_control_out.bias)
            else:
                self.detail_control_in = None
                self.detail_control_blocks = None
                self.detail_control_out = None
            if cfg.detail_film_modulation:
                self.detail_film = nn.Conv2d(cfg.hidden_channels, cfg.hidden_channels * 2, kernel_size=1)
                nn.init.zeros_(self.detail_film.weight)
                nn.init.zeros_(self.detail_film.bias)
            else:
                self.detail_film = None
            fusion_inputs += 1
        else:
            self.detail_blocks = nn.Identity()
            self.detail_highfreq_proj = None
            self.detail_control_in = None
            self.detail_control_blocks = None
            self.detail_control_out = None
            self.detail_film = None
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
        detail_condition_delta = None
        detail_film_params: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.detail_proj is not None:
            if detail_context is None:
                raise ValueError("detail_context is required when detail_context_channels > 0")
            if detail_context.shape[1] != self.cfg.detail_context_channels:
                raise ValueError(
                    "detail_context channel mismatch: "
                    f"got {detail_context.shape[1]}, expected {self.cfg.detail_context_channels}"
                )
            detail_features = self.detail_proj(detail_context)
            if self.detail_highfreq_proj is not None:
                detail_highfreq = detail_context - F.avg_pool2d(detail_context, kernel_size=3, stride=1, padding=1)
                detail_features = detail_features + self.detail_highfreq_proj(
                    torch.cat([detail_highfreq, detail_context.abs()], dim=1)
                )
            detail_features = F.interpolate(detail_features, size=condition_size, mode="bilinear", align_corners=False)
            detail_features = self.detail_blocks(detail_features)
            if self.detail_control_out is not None:
                detail_control_features = detail_features
                if self.cfg.detail_control_condition_fusion:
                    detail_control_features = torch.cat([detail_control_features, condition_features], dim=1)
                if self.detail_control_in is None or self.detail_control_blocks is None:
                    raise RuntimeError("detail control modules are not initialized")
                detail_control_features = self.detail_control_in(detail_control_features)
                detail_control_features = self.detail_control_blocks(detail_control_features)
                detail_condition_delta = self.detail_control_out(detail_control_features)
            if self.detail_film is not None:
                detail_gamma, detail_beta = self.detail_film(detail_features).chunk(2, dim=1)
                detail_film_params = (detail_gamma, detail_beta)
            features.append(detail_features)

        fused = self.fuse_in(torch.cat(features, dim=1))
        if detail_film_params is not None:
            detail_gamma, detail_beta = detail_film_params
            fused = fused * (1.0 + detail_gamma) + detail_beta
        fused = self.fusion_blocks(fused)
        out = self.out(fused)
        if detail_condition_delta is not None:
            out = out + detail_condition_delta
        return out


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


@dataclass(frozen=True)
class CoSERToCoDLiteConditionGateConfig:
    semantic_channels: int = 256
    detail_context_channels: int = 0
    condition_channels: int = 384
    hidden_channels: int = 128
    image_feature_channels: int = 9
    num_blocks: int = 2
    gate_min: float = 0.0
    gate_max: float = 1.0
    init_gate: float = 0.3
    spatial_gate: bool = True
    channel_gate: bool = False


class CoSERToCoDLiteConditionGate(nn.Module):
    """Gate CoD-Lite condition residuals from decoded CoSER state.

    Unlike ``CoSERToCoDLiteAlphaGate``, this module does not blend final RGB
    outputs. It controls how much of the CoSER-predicted condition residual is
    injected into the frozen CoD-Lite backbone.
    """

    def __init__(self, cfg: CoSERToCoDLiteConditionGateConfig = CoSERToCoDLiteConditionGateConfig()) -> None:
        super().__init__()
        if not 0.0 <= cfg.gate_min < cfg.gate_max:
            raise ValueError("gate_min/gate_max must satisfy 0 <= min < max")
        if not cfg.gate_min < cfg.init_gate < cfg.gate_max:
            raise ValueError("init_gate must be inside (gate_min, gate_max)")
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
        out_channels = cfg.condition_channels if cfg.channel_gate else 1
        if cfg.spatial_gate:
            self.head = nn.Conv2d(cfg.hidden_channels, out_channels, kernel_size=1)
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(cfg.hidden_channels, cfg.hidden_channels),
                nn.SiLU(),
                nn.Linear(cfg.hidden_channels, out_channels),
            )
        self._init_gate_head()

    def _init_gate_head(self) -> None:
        normalized = (self.cfg.init_gate - self.cfg.gate_min) / (self.cfg.gate_max - self.cfg.gate_min)
        normalized = min(max(normalized, 1.0e-6), 1.0 - 1.0e-6)
        bias = math.log(normalized / (1.0 - normalized))
        if isinstance(self.head, nn.Conv2d):
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, bias)
            return
        final = self.head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, bias)

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
        if not self.cfg.spatial_gate:
            logit = logit.view(logit.shape[0], logit.shape[1], 1, 1)
        gate = torch.sigmoid(logit)
        return self.cfg.gate_min + (self.cfg.gate_max - self.cfg.gate_min) * gate


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)
