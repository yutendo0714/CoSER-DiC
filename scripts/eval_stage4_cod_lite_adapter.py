from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from coserdic.entropy import (
    CoSERBitstream,
    ComponentCodebookControlCode,
    ComponentMuLawControlCode,
    ComponentUniformControlCode,
    MuLawControlGridCode,
    PrefixTopKControlBasisCode,
    SparseControlBasisCode,
    StaticControlHuffmanCode,
    UniformControlGridCode,
    VectorCodebookControlCode,
    build_control_grid_code,
    dct2_orthonormal,
    idct2_orthonormal,
    project_onto_control_basis,
    reconstruct_from_control_basis,
    semantic_bits_per_token,
    zigzag_indices,
)
from coserdic.metrics import PerceptualMetricBundle
from coserdic.models import (
    CoDLiteOneStepBackbone,
    CoDLiteOneStepBackboneConfig,
    CoSERToCoDLiteAlphaGate,
    CoSERToCoDLiteAlphaGateConfig,
    CoSERToCoDLiteConditionAdapter,
    CoSERToCoDLiteConditionAdapterConfig,
    CoSERToCoDLiteConditionGate,
    CoSERToCoDLiteConditionGateConfig,
    CoSERToCoDLiteConditionPyramidAdapter,
    CoSERToCoDLiteConditionPyramidAdapterConfig,
    apply_lora_adapters_from_config,
    condition_residual_rms_guard,
    load_named_parameter_state,
)
from coserdic.utils import seed_everything
from coserdic.utils.wandb_utils import init_wandb


def detail_context_channels(mode: str) -> int:
    return {"none": 0, "residual_grid": 3, "residual_grid_codes": 6}[mode]


class Stage4ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        per_image_metrics: Path | None = None,
        limit: int = 0,
        crop_size: int = 0,
        semantic_channels: int = 3,
        detail_context: str = "none",
        semantic_latent_ablation: str = "normal",
        detail_context_ablation: str = "normal",
        shuffle_seed: int = 1234,
        require_payload_bpp: bool = False,
    ) -> None:
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        metric_rows = []
        if per_image_metrics is not None:
            metric_rows = [json.loads(line) for line in per_image_metrics.read_text().splitlines() if line.strip()]
        if limit:
            rows = rows[:limit]
            metric_rows = metric_rows[:limit]
        if not rows:
            raise ValueError(f"empty manifest: {manifest}")
        self.rows = rows
        self.metric_rows = metric_rows
        self.crop_size = crop_size
        self.semantic_channels = int(semantic_channels)
        self.detail_context = detail_context
        self.require_payload_bpp = bool(require_payload_bpp)
        self.semantic_latent_ablation = semantic_latent_ablation
        self.detail_context_ablation = detail_context_ablation
        if semantic_latent_ablation not in {"normal", "zero", "shuffle"}:
            raise ValueError(f"unknown semantic_latent_ablation: {semantic_latent_ablation}")
        if detail_context_ablation not in {"normal", "zero", "shuffle"}:
            raise ValueError(f"unknown detail_context_ablation: {detail_context_ablation}")
        self.semantic_shuffle_indices = _non_self_randperm(len(rows), int(shuffle_seed))
        self.detail_shuffle_indices = _non_self_randperm(len(rows), int(shuffle_seed) + 17)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        metric_row = self.metric_rows[index] if index < len(self.metric_rows) else {}
        payload_row = metric_row if "actual_payload_bpp" in metric_row else row
        if self.require_payload_bpp and "actual_payload_bpp" not in payload_row:
            raise ValueError(
                "Stage 4 evaluation requires actual payload bpp. "
                "Pass --per-image-metrics from the Stage 3/Stage 4 anchor, "
                "or include actual_payload_bpp in the manifest rows."
            )
        actual_payload_bpp = float(payload_row.get("actual_payload_bpp", 0.0))
        paper_bpp = float(metric_row.get("paper_bpp", row.get("paper_bpp", actual_payload_bpp)))
        semantic = _load_image(Path(row["semantic_only"]), self.crop_size)
        semantic_latent = semantic
        detail_context = torch.empty(0, 1, 1)
        cache_path = str(row.get("decoder_feature_cache", ""))
        if self.semantic_channels != 3 or self.detail_context != "none":
            if not cache_path:
                raise FileNotFoundError(
                    f"manifest row {index} has no decoder_feature_cache; "
                    f"required for semantic_channels={self.semantic_channels} "
                    f"and detail_context={self.detail_context}"
                )
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            cache = {}
        if self.semantic_channels != 3:
            semantic_cache = cache
            if self.semantic_latent_ablation == "shuffle":
                semantic_cache = self._load_cache(self.semantic_shuffle_indices[index])
            semantic_latent = _load_semantic_latent(semantic_cache)
            if self.semantic_latent_ablation == "zero":
                semantic_latent = torch.zeros_like(semantic_latent)
        elif self.semantic_latent_ablation == "zero":
            semantic_latent = torch.zeros_like(semantic)
        elif self.semantic_latent_ablation == "shuffle":
            semantic_row = self.rows[self.semantic_shuffle_indices[index]]
            semantic_latent = _load_image(Path(semantic_row["semantic_only"]), self.crop_size)
        if self.detail_context != "none":
            detail_cache = cache
            if self.detail_context_ablation == "shuffle":
                detail_cache = self._load_cache(self.detail_shuffle_indices[index])
            detail_context = _load_detail_context(detail_cache, self.detail_context)
            if self.detail_context_ablation == "zero":
                detail_context = torch.zeros_like(detail_context)
        return {
            "reference": _load_image(Path(row["reference"]), self.crop_size),
            "semantic": semantic,
            "semantic_latent": semantic_latent,
            "detail_context": detail_context,
            "stage3": _load_image(Path(row["stage3"]), self.crop_size),
            "index": int(row.get("index", index)),
            "source_path": str(row.get("source_path", metric_row.get("path", ""))),
            "actual_payload_bpp": actual_payload_bpp,
            "paper_bpp": paper_bpp,
        }

    def _load_cache(self, index: int) -> dict[str, object]:
        cache_path = str(self.rows[index].get("decoder_feature_cache", ""))
        if not cache_path:
            raise FileNotFoundError(f"manifest row {index} has no decoder_feature_cache for ablation")
        return torch.load(cache_path, map_location="cpu", weights_only=False)


def _load_image(path: Path, crop_size: int) -> torch.Tensor:
    tensor = TF.to_tensor(Image.open(path).convert("RGB"))
    if crop_size > 0:
        _, h, w = tensor.shape
        if h < crop_size or w < crop_size:
            scale = crop_size / min(h, w)
            tensor = TF.resize(
                tensor,
                [max(crop_size, int(round(h * scale))), max(crop_size, int(round(w * scale)))],
                antialias=True,
            )
            _, h, w = tensor.shape
        top = max(0, (h - crop_size) // 2)
        left = max(0, (w - crop_size) // 2)
        tensor = TF.crop(tensor, top, left, crop_size, crop_size)
    return tensor


def _non_self_randperm(length: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    if length <= 1:
        return indices
    for idx, value in enumerate(indices):
        if value == idx:
            swap_idx = (idx + 1) % length
            indices[idx], indices[swap_idx] = indices[swap_idx], indices[idx]
    return indices


def _load_detail_context(cache: dict[str, object], mode: str) -> torch.Tensor:
    residual_grid = cache["residual_grid_hat"].float()
    if residual_grid.ndim == 4 and residual_grid.shape[0] == 1:
        residual_grid = residual_grid.squeeze(0)
    if mode == "residual_grid":
        return residual_grid
    if mode == "residual_grid_codes":
        detail_codes = cache["detail_codes"].float()
        if detail_codes.ndim == 4 and detail_codes.shape[0] == 1:
            detail_codes = detail_codes.squeeze(0)
        detail_codes = detail_codes / 15.0 * 2.0 - 1.0
        return torch.cat([residual_grid, detail_codes], dim=0)
    raise ValueError(f"unknown detail_context: {mode}")


def _load_semantic_latent(cache: dict[str, object]) -> torch.Tensor:
    semantic_latent = cache["semantic_latent"].float()
    if semantic_latent.ndim == 4 and semantic_latent.shape[0] == 1:
        semantic_latent = semantic_latent.squeeze(0)
    return semantic_latent


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((x - y).pow(2), dim=(1, 2, 3)).clamp_min(1.0e-12)
    return -10.0 * torch.log10(mse)


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def apply_condition_residual(
    base_condition: torch.Tensor,
    cond_delta: torch.Tensor,
    *,
    residual_scale: float,
    residual_tanh: bool,
) -> torch.Tensor:
    if residual_tanh:
        cond_delta = torch.tanh(cond_delta)
    return base_condition + residual_scale * cond_delta


def build_adapter_from_payload(
    payload: dict[str, object],
) -> CoSERToCoDLiteConditionAdapter | CoSERToCoDLiteConditionPyramidAdapter:
    adapter_kind = str(payload.get("adapter_kind", "light"))
    adapter_config = dict(payload["adapter_config"])
    if adapter_kind == "light":
        return CoSERToCoDLiteConditionAdapter(CoSERToCoDLiteConditionAdapterConfig(**adapter_config))
    if adapter_kind == "pyramid":
        return CoSERToCoDLiteConditionPyramidAdapter(CoSERToCoDLiteConditionPyramidAdapterConfig(**adapter_config))
    raise ValueError(f"unknown adapter_kind in checkpoint: {adapter_kind}")


def build_gate_from_payload(payload: dict[str, object]) -> CoSERToCoDLiteAlphaGate:
    gate_config = dict(payload["gate_config"])
    return CoSERToCoDLiteAlphaGate(CoSERToCoDLiteAlphaGateConfig(**gate_config))


def build_condition_gate_from_payload(payload: dict[str, object]) -> CoSERToCoDLiteConditionGate:
    gate_config = dict(payload["condition_gate_config"])
    return CoSERToCoDLiteConditionGate(CoSERToCoDLiteConditionGateConfig(**gate_config))


def channel_group_sizes(channels: int, groups: int) -> list[int]:
    if channels <= 0:
        raise ValueError("channels must be positive")
    if groups <= 0:
        raise ValueError("groups must be positive")
    if groups > channels:
        raise ValueError("groups must be <= channels")
    base = channels // groups
    remainder = channels % groups
    return [base + (1 if group < remainder else 0) for group in range(groups)]


def grouped_condition_residual_control(
    condition_error: torch.Tensor,
    *,
    groups: int,
    grid_size: int,
    codec: (
        UniformControlGridCode
        | MuLawControlGridCode
        | ComponentUniformControlCode
        | ComponentMuLawControlCode
        | ComponentCodebookControlCode
        | VectorCodebookControlCode
    ),
    scale: float,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Encode/decode a tiny condition-control payload, then expand it.

    The encoder computes ``condition_error`` from the reference image condition.
    The decoder only receives the returned payload bytes; this helper simulates
    that by actually encoding and decoding the control grid before expansion.
    """

    if condition_error.ndim != 4:
        raise ValueError("condition_error must be BCHW")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    batch, channels, height, width = condition_error.shape
    sizes = channel_group_sizes(int(channels), int(groups))
    grouped_maps: list[torch.Tensor] = []
    start = 0
    for size in sizes:
        end = start + size
        grouped_maps.append(condition_error[:, start:end].float().mean(dim=1, keepdim=True))
        start = end
    control_grid = F.adaptive_avg_pool2d(torch.cat(grouped_maps, dim=1), output_size=(grid_size, grid_size))

    decoded_items: list[torch.Tensor] = []
    payload_bytes: list[int] = []
    for item in range(batch):
        payload = codec.encode_values(control_grid[item].detach().cpu())
        decoded = codec.decode_values(payload, shape=(groups, grid_size, grid_size)).to(
            device=condition_error.device,
            dtype=torch.float32,
        )
        decoded_items.append(decoded)
        payload_bytes.append(len(payload))
    decoded_grid = torch.stack(decoded_items, dim=0)

    expanded_groups: list[torch.Tensor] = []
    for group, size in enumerate(sizes):
        expanded_groups.append(decoded_grid[:, group : group + 1].repeat(1, size, 1, 1))
    expanded = torch.cat(expanded_groups, dim=1)
    correction = F.interpolate(expanded, size=(height, width), mode="bilinear", align_corners=False)
    return correction * float(scale), payload_bytes, control_grid.detach().float().abs().mean(dim=(1, 2, 3))


def grouped_condition_dct_control(
    condition_error: torch.Tensor,
    *,
    groups: int,
    grid_size: int,
    coeffs_per_group: int,
    codec: UniformControlGridCode,
    scale: float,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    if condition_error.ndim != 4:
        raise ValueError("condition_error must be BCHW")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if coeffs_per_group <= 0:
        raise ValueError("coeffs_per_group must be positive")
    if coeffs_per_group > grid_size * grid_size:
        raise ValueError("coeffs_per_group must be <= grid_size * grid_size")
    batch, channels, height, width = condition_error.shape
    sizes = channel_group_sizes(int(channels), int(groups))
    grouped_maps: list[torch.Tensor] = []
    start = 0
    for size in sizes:
        end = start + size
        grouped_maps.append(condition_error[:, start:end].float().mean(dim=1, keepdim=True))
        start = end
    control_grid = F.adaptive_avg_pool2d(torch.cat(grouped_maps, dim=1), output_size=(grid_size, grid_size))
    coeff_grid = dct2_orthonormal(control_grid)
    coeff_positions = zigzag_indices(grid_size, coeffs_per_group)
    selected = torch.stack([coeff_grid[..., y, x] for y, x in coeff_positions], dim=-1)

    decoded_items: list[torch.Tensor] = []
    payload_bytes: list[int] = []
    for item in range(batch):
        payload = codec.encode_values(selected[item].detach().cpu())
        decoded = codec.decode_values(payload, shape=(groups, coeffs_per_group)).to(
            device=condition_error.device,
            dtype=torch.float32,
        )
        decoded_items.append(decoded)
        payload_bytes.append(len(payload))
    decoded_coeffs = torch.stack(decoded_items, dim=0)
    decoded_grid = torch.zeros_like(coeff_grid)
    for coeff_index, (y, x) in enumerate(coeff_positions):
        decoded_grid[..., y, x] = decoded_coeffs[..., coeff_index]
    reconstructed_grid = idct2_orthonormal(decoded_grid)

    expanded_groups: list[torch.Tensor] = []
    for group, size in enumerate(sizes):
        expanded_groups.append(reconstructed_grid[:, group : group + 1].repeat(1, size, 1, 1))
    expanded = torch.cat(expanded_groups, dim=1)
    correction = F.interpolate(expanded, size=(height, width), mode="bilinear", align_corners=False)
    return correction * float(scale), payload_bytes, selected.detach().float().abs().mean(dim=(1, 2))


def grouped_condition_affine_control(
    base_condition: torch.Tensor,
    pred_condition: torch.Tensor,
    target_condition: torch.Tensor,
    *,
    groups: int,
    grid_size: int,
    gain_codec: UniformControlGridCode,
    bias_codec: UniformControlGridCode,
    scale: float,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Encode/decode group-wise affine condition residual controls.

    The encoder fits a small local affine correction:

    ``target - base ~= affine_gain * (pred - base) + affine_bias``.

    The decoder can reproduce ``base`` and ``pred - base`` from fixed model
    state plus the CoSER semantic/detail payload. Only quantized affine gain
    deltas and biases are transmitted here and counted in actual_payload_bpp.
    """

    if base_condition.shape != pred_condition.shape or base_condition.shape != target_condition.shape:
        raise ValueError("base_condition, pred_condition, and target_condition must have the same shape")
    if base_condition.ndim != 4:
        raise ValueError("conditions must be BCHW")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    batch, channels, height, width = base_condition.shape
    sizes = channel_group_sizes(int(channels), int(groups))
    residual = (pred_condition - base_condition).float()
    target_delta = (target_condition - base_condition).float()

    gain_delta_groups: list[torch.Tensor] = []
    bias_groups: list[torch.Tensor] = []
    start = 0
    for size in sizes:
        end = start + size
        residual_group = residual[:, start:end]
        target_group = target_delta[:, start:end]
        mean_r = F.adaptive_avg_pool2d(residual_group, output_size=(grid_size, grid_size)).mean(dim=1, keepdim=True)
        mean_t = F.adaptive_avg_pool2d(target_group, output_size=(grid_size, grid_size)).mean(dim=1, keepdim=True)
        mean_rt = F.adaptive_avg_pool2d(residual_group * target_group, output_size=(grid_size, grid_size)).mean(
            dim=1,
            keepdim=True,
        )
        mean_rr = F.adaptive_avg_pool2d(residual_group * residual_group, output_size=(grid_size, grid_size)).mean(
            dim=1,
            keepdim=True,
        )
        var_r = (mean_rr - mean_r.pow(2)).clamp_min(0.0)
        cov_rt = mean_rt - mean_r * mean_t
        affine_gain = torch.where(var_r > eps, cov_rt / (var_r + eps), torch.ones_like(var_r))
        affine_bias = mean_t - affine_gain * mean_r
        gain_delta_groups.append(affine_gain - 1.0)
        bias_groups.append(affine_bias)
        start = end

    gain_delta_grid = torch.cat(gain_delta_groups, dim=1)
    bias_grid = torch.cat(bias_groups, dim=1)

    decoded_gain_items: list[torch.Tensor] = []
    decoded_bias_items: list[torch.Tensor] = []
    payload_bytes: list[int] = []
    for item in range(batch):
        gain_payload = gain_codec.encode_values(gain_delta_grid[item].detach().cpu())
        bias_payload = bias_codec.encode_values(bias_grid[item].detach().cpu())
        decoded_gain = gain_codec.decode_values(gain_payload, shape=(groups, grid_size, grid_size)).to(
            device=base_condition.device,
            dtype=torch.float32,
        )
        decoded_bias = bias_codec.decode_values(bias_payload, shape=(groups, grid_size, grid_size)).to(
            device=base_condition.device,
            dtype=torch.float32,
        )
        decoded_gain_items.append(decoded_gain)
        decoded_bias_items.append(decoded_bias)
        payload_bytes.append(len(gain_payload) + len(bias_payload))
    decoded_gain_grid = torch.stack(decoded_gain_items, dim=0)
    decoded_bias_grid = torch.stack(decoded_bias_items, dim=0)

    expanded_gain_groups: list[torch.Tensor] = []
    expanded_bias_groups: list[torch.Tensor] = []
    for group, size in enumerate(sizes):
        expanded_gain_groups.append(decoded_gain_grid[:, group : group + 1].repeat(1, size, 1, 1))
        expanded_bias_groups.append(decoded_bias_grid[:, group : group + 1].repeat(1, size, 1, 1))
    expanded_gain = torch.cat(expanded_gain_groups, dim=1)
    expanded_bias = torch.cat(expanded_bias_groups, dim=1)
    gain_map = F.interpolate(expanded_gain, size=(height, width), mode="bilinear", align_corners=False)
    bias_map = F.interpolate(expanded_bias, size=(height, width), mode="bilinear", align_corners=False)
    correction = gain_map * residual + bias_map
    control_abs = torch.cat([gain_delta_grid.abs(), bias_grid.abs()], dim=1).mean(dim=(1, 2, 3))
    return correction * float(scale), payload_bytes, control_abs.detach().float()


def grouped_condition_affine_dct_control(
    base_condition: torch.Tensor,
    pred_condition: torch.Tensor,
    target_condition: torch.Tensor,
    *,
    affine_groups: int,
    affine_grid_size: int,
    dct_groups: int,
    dct_grid_size: int,
    dct_coeffs_per_group: int,
    gain_codec: UniformControlGridCode,
    bias_codec: UniformControlGridCode,
    dct_codec: UniformControlGridCode,
    scale: float,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Encode affine correction, then DCT-code the remaining condition error."""

    affine_correction, affine_bytes, affine_abs = grouped_condition_affine_control(
        base_condition,
        pred_condition,
        target_condition,
        groups=affine_groups,
        grid_size=affine_grid_size,
        gain_codec=gain_codec,
        bias_codec=bias_codec,
        scale=scale,
    )
    after_affine = pred_condition.float() + affine_correction.float()
    dct_correction, dct_bytes, dct_abs = grouped_condition_dct_control(
        target_condition.float() - after_affine,
        groups=dct_groups,
        grid_size=dct_grid_size,
        coeffs_per_group=dct_coeffs_per_group,
        codec=dct_codec,
        scale=scale,
    )
    payload_bytes = [int(a) + int(b) for a, b in zip(affine_bytes, dct_bytes)]
    control_abs = 0.5 * (affine_abs.float() + dct_abs.float())
    return affine_correction + dct_correction, payload_bytes, control_abs.detach().float()


def grouped_condition_affine_grid_control(
    base_condition: torch.Tensor,
    pred_condition: torch.Tensor,
    target_condition: torch.Tensor,
    *,
    affine_groups: int,
    affine_grid_size: int,
    grid_groups: int,
    grid_size: int,
    gain_codec: UniformControlGridCode,
    bias_codec: UniformControlGridCode,
    grid_codec: UniformControlGridCode,
    scale: float,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Encode affine correction, then grid-code the remaining condition error."""

    affine_correction, affine_bytes, affine_abs = grouped_condition_affine_control(
        base_condition,
        pred_condition,
        target_condition,
        groups=affine_groups,
        grid_size=affine_grid_size,
        gain_codec=gain_codec,
        bias_codec=bias_codec,
        scale=scale,
    )
    after_affine = pred_condition.float() + affine_correction.float()
    grid_correction, grid_bytes, grid_abs = grouped_condition_residual_control(
        target_condition.float() - after_affine,
        groups=grid_groups,
        grid_size=grid_size,
        codec=grid_codec,
        scale=scale,
    )
    payload_bytes = [int(a) + int(b) for a, b in zip(affine_bytes, grid_bytes)]
    control_abs = 0.5 * (affine_abs.float() + grid_abs.float())
    return affine_correction + grid_correction, payload_bytes, control_abs.detach().float()


def grouped_condition_affine_basis_control(
    base_condition: torch.Tensor,
    pred_condition: torch.Tensor,
    target_condition: torch.Tensor,
    *,
    affine_groups: int,
    affine_grid_size: int,
    gain_codec: UniformControlGridCode,
    bias_codec: UniformControlGridCode,
    basis_payload: dict[str, object],
    components: int,
    candidate_components: int,
    selection: str,
    basis_codec: (
        UniformControlGridCode
        | MuLawControlGridCode
        | ComponentUniformControlCode
        | ComponentMuLawControlCode
        | ComponentCodebookControlCode
        | VectorCodebookControlCode
    ),
    prefix_components: int = 0,
    huffman: StaticControlHuffmanCode | None = None,
    sparse_huffman: tuple[StaticControlHuffmanCode, StaticControlHuffmanCode] | None = None,
    scale: float,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Encode affine correction, then basis-code the remaining condition error."""

    affine_correction, affine_bytes, affine_abs = grouped_condition_affine_control(
        base_condition,
        pred_condition,
        target_condition,
        groups=affine_groups,
        grid_size=affine_grid_size,
        gain_codec=gain_codec,
        bias_codec=bias_codec,
        scale=scale,
    )
    after_affine = pred_condition.float() + affine_correction.float()
    basis_correction, basis_bytes, basis_abs = grouped_condition_basis_control(
        target_condition.float() - after_affine,
        basis_payload=basis_payload,
        components=components,
        candidate_components=candidate_components,
        selection=selection,
        codec=basis_codec,
        prefix_components=prefix_components,
        huffman=huffman,
        sparse_huffman=sparse_huffman,
        scale=scale,
    )
    payload_bytes = [int(a) + int(b) for a, b in zip(affine_bytes, basis_bytes)]
    control_abs = 0.5 * (affine_abs.float() + basis_abs.float())
    return affine_correction + basis_correction, payload_bytes, control_abs.detach().float()


def repeated_payload_bits(batch: int, bit_count: int | None) -> list[int] | None:
    if bit_count is None:
        return None
    if batch < 0:
        raise ValueError("batch must be non-negative")
    if bit_count < 0:
        raise ValueError("bit_count must be non-negative")
    return [int(bit_count) for _ in range(int(batch))]


def affine_control_payload_bits(
    *,
    groups: int,
    grid_size: int,
    gain_codec: UniformControlGridCode,
    bias_codec: UniformControlGridCode,
) -> int:
    shape = (int(groups), int(grid_size), int(grid_size))
    return int(gain_codec.encoded_num_bits(shape)) + int(bias_codec.encoded_num_bits(shape))


def basis_control_payload_bits(
    *,
    selection: str,
    components: int,
    candidate_components: int,
    basis_codec: object,
    prefix_components: int = 0,
) -> int | None:
    if selection == "prefix":
        if not hasattr(basis_codec, "encoded_num_bits"):
            raise TypeError("basis codec must expose encoded_num_bits")
        return int(basis_codec.encoded_num_bits((int(components),)))
    if selection == "vector":
        if not hasattr(basis_codec, "encoded_num_bits"):
            raise TypeError("basis codec must expose encoded_num_bits")
        return int(basis_codec.encoded_num_bits((1,)))
    if selection == "topk":
        if not hasattr(basis_codec, "bits"):
            raise TypeError("top-k basis codec must expose bits")
        if candidate_components <= 1:
            raise ValueError("candidate_components must be > 1 for top-k basis bits")
        return int(components) * (
            int(semantic_bits_per_token(int(candidate_components))) + int(basis_codec.bits)
        )
    if selection == "prefix_topk":
        if not hasattr(basis_codec, "bits"):
            raise TypeError("prefix-top-k basis codec must expose bits")
        if prefix_components <= 0:
            raise ValueError("prefix_components must be positive for prefix-top-k basis bits")
        tail_components = int(candidate_components) - int(prefix_components)
        if tail_components <= 1:
            raise ValueError("candidate_components - prefix_components must be > 1")
        if components <= 0 or components > tail_components:
            raise ValueError("components must be in [1, candidate_components - prefix_components]")
        return int(prefix_components) * int(basis_codec.bits) + int(components) * (
            int(semantic_bits_per_token(tail_components)) + int(basis_codec.bits)
        )
    return None


def unpack_control_candidate(
    candidate: tuple,
) -> tuple[str, torch.Tensor, list[int], torch.Tensor, list[int] | None]:
    if len(candidate) == 4:
        label, correction, payload_bytes, control_abs = candidate
        payload_bits = None
    elif len(candidate) == 5:
        label, correction, payload_bytes, control_abs, payload_bits = candidate
    else:
        raise ValueError("control candidates must have 4 or 5 fields")
    if payload_bits is not None and len(payload_bits) != len(payload_bytes):
        raise ValueError(f"candidate payload bit count mismatch for {label}")
    return str(label), correction, payload_bytes, control_abs, payload_bits


def payload_bytes_with_hybrid_selector(
    payload_bytes: list[int],
    *,
    selector_bytes: int,
    selector_bits: int = 0,
    payload_bits: list[int] | None = None,
) -> list[int]:
    """Return transmitted bytes after adding the per-image hybrid mode selector.

    ``selector_bytes`` is the conservative byte-granular fallback used for
    variable-length payloads. When exact fixed-bit payload sizes are available,
    ``selector_bits`` lets the selector share the same packed control stream.
    """

    if selector_bytes < 0:
        raise ValueError("selector_bytes must be non-negative")
    if selector_bits < 0:
        raise ValueError("selector_bits must be non-negative")
    base = [int(value) for value in payload_bytes]
    if any(value < 0 for value in base):
        raise ValueError("payload byte counts must be non-negative")
    fallback_selector_bytes = int(selector_bytes)
    if selector_bits > 0:
        fallback_selector_bytes = max(fallback_selector_bytes, (int(selector_bits) + 7) // 8)
    if selector_bits <= 0 or payload_bits is None:
        return [value + fallback_selector_bytes for value in base]
    if len(payload_bits) != len(base):
        raise ValueError("payload_bits length must match payload_bytes")
    out: list[int] = []
    for byte_count, bit_count in zip(base, payload_bits):
        bit_count = int(bit_count)
        if bit_count < 0:
            raise ValueError("payload bit counts must be non-negative")
        packed_bytes = (bit_count + int(selector_bits) + 7) // 8
        out.append(max(int(byte_count), int(packed_bytes)))
    return out


def select_condition_control_by_rd_proxy(
    pred_condition: torch.Tensor,
    target_condition: torch.Tensor,
    candidates: list[tuple],
    *,
    height: int,
    width: int,
    selector_bytes: int,
    rate_lambda: float,
    selector_bits: int = 0,
) -> tuple[torch.Tensor, list[int], torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    """Select one transmitted condition-control payload per image.

    The encoder is allowed to inspect the reference-derived target condition
    and choose which control family to transmit. The decoder receives a compact
    mode selector plus the selected control payload, so selector bytes must be
    counted in the payload length.
    """

    if pred_condition.ndim != 4 or target_condition.ndim != 4:
        raise ValueError("pred_condition and target_condition must be BCHW")
    if pred_condition.shape != target_condition.shape:
        raise ValueError("pred_condition and target_condition must have the same shape")
    if not candidates:
        raise ValueError("at least one control candidate is required")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if selector_bytes < 0:
        raise ValueError("selector_bytes must be non-negative")
    if rate_lambda < 0:
        raise ValueError("rate_lambda must be non-negative")
    batch = int(pred_condition.shape[0])
    corrections: list[torch.Tensor] = []
    abs_values: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    total_payload_bytes_by_candidate: list[list[int]] = []
    for candidate in candidates:
        label, correction, payload_bytes, control_abs, payload_bits = unpack_control_candidate(candidate)
        if not label:
            raise ValueError("candidate label must be non-empty")
        if correction.shape != pred_condition.shape:
            raise ValueError(f"candidate correction shape mismatch for {label}")
        if len(payload_bytes) != batch:
            raise ValueError(f"candidate payload byte count mismatch for {label}")
        if control_abs.shape != (batch,):
            raise ValueError(f"candidate control_abs shape mismatch for {label}")
        total_payload_bytes = payload_bytes_with_hybrid_selector(
            payload_bytes,
            selector_bytes=selector_bytes,
            selector_bits=selector_bits,
            payload_bits=payload_bits,
        )
        total_payload = torch.tensor(
            [CoSERBitstream.bytes_to_bpp(int(value), height, width) for value in total_payload_bytes],
            device=pred_condition.device,
            dtype=torch.float32,
        )
        condition_l1 = torch.mean(
            torch.abs((pred_condition.float() + correction.float()) - target_condition.float()),
            dim=(1, 2, 3),
        )
        scores.append(condition_l1 + float(rate_lambda) * total_payload)
        corrections.append(correction.float())
        abs_values.append(control_abs.float())
        total_payload_bytes_by_candidate.append(total_payload_bytes)

    score_tensor = torch.stack(scores, dim=0)
    selected_index = torch.argmin(score_tensor, dim=0)
    selected_correction = torch.zeros_like(pred_condition, dtype=torch.float32)
    selected_abs = torch.zeros((batch,), device=pred_condition.device, dtype=torch.float32)
    selected_payload_bytes = [0 for _ in range(batch)]
    selected_labels: list[str] = []
    for item in range(batch):
        index = int(selected_index[item].item())
        label, _, _, _, _ = unpack_control_candidate(candidates[index])
        selected_labels.append(label)
        selected_payload_bytes[item] = int(total_payload_bytes_by_candidate[index][item])
        selected_correction[item] = corrections[index][item]
        selected_abs[item] = abs_values[index][item]
    selected_condition_l1 = torch.mean(
        torch.abs((pred_condition.float() + selected_correction) - target_condition.float()),
        dim=(1, 2, 3),
    )
    return (
        selected_correction,
        selected_payload_bytes,
        selected_abs.detach().float(),
        selected_labels,
        selected_index.detach(),
        selected_condition_l1.detach().float(),
    )


def select_condition_control_by_image_rdo(
    reference: torch.Tensor,
    stage3: torch.Tensor,
    pred_condition: torch.Tensor,
    target_condition: torch.Tensor,
    candidates: list[tuple],
    *,
    backbone: CoDLiteOneStepBackbone,
    selector_bytes: int,
    rate_lambda: float,
    objective: str,
    blend_alpha: float,
    perceptual: PerceptualMetricBundle | None = None,
    selector_bits: int = 0,
    fidelity_lambda: float = 0.0,
    fidelity_metric: str = "image_l1",
) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor, list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one control candidate by running the decoder-side image RDO proxy."""

    if objective not in {"image_l1", "image_mse", "lpips_alex", "dists"}:
        raise ValueError("objective must be image_l1, image_mse, lpips_alex, or dists")
    if fidelity_metric not in {"image_l1", "image_mse"}:
        raise ValueError("fidelity_metric must be image_l1 or image_mse")
    if reference.shape != stage3.shape:
        raise ValueError("reference and stage3 must have the same shape")
    if perceptual is None and objective in {"lpips_alex", "dists"}:
        raise ValueError("perceptual metrics are required for LPIPS/DISTS hybrid RDO")
    if selector_bytes < 0:
        raise ValueError("selector_bytes must be non-negative")
    if rate_lambda < 0:
        raise ValueError("rate_lambda must be non-negative")
    if fidelity_lambda < 0:
        raise ValueError("fidelity_lambda must be non-negative")
    batch = int(reference.shape[0])
    height = int(reference.shape[-2])
    width = int(reference.shape[-1])
    raw_outputs: list[torch.Tensor] = []
    final_outputs: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    condition_l1_values: list[torch.Tensor] = []
    abs_values: list[torch.Tensor] = []
    total_payload_bytes_by_candidate: list[list[int]] = []
    for candidate in candidates:
        label, correction, payload_bytes, control_abs, payload_bits = unpack_control_candidate(candidate)
        if not label:
            raise ValueError("candidate label must be non-empty")
        if correction.shape != pred_condition.shape:
            raise ValueError(f"candidate correction shape mismatch for {label}")
        if len(payload_bytes) != batch:
            raise ValueError(f"candidate payload byte count mismatch for {label}")
        if control_abs.shape != (batch,):
            raise ValueError(f"candidate control_abs shape mismatch for {label}")
        total_payload_bytes = payload_bytes_with_hybrid_selector(
            payload_bytes,
            selector_bytes=selector_bytes,
            selector_bits=selector_bits,
            payload_bits=payload_bits,
        )
        candidate_condition = pred_condition.float() + correction.float()
        raw = backbone(stage3, candidate_condition.to(dtype=pred_condition.dtype))
        final = ((1.0 - float(blend_alpha)) * stage3 + float(blend_alpha) * raw).clamp(0, 1)
        raw_outputs.append(raw)
        final_outputs.append(final)
        condition_l1 = torch.mean(torch.abs(candidate_condition - target_condition.float()), dim=(1, 2, 3))
        condition_l1_values.append(condition_l1)
        abs_values.append(control_abs.float())
        if objective == "image_l1":
            distortion = torch.mean(torch.abs(final.float() - reference.float()), dim=(1, 2, 3))
        elif objective == "image_mse":
            distortion = torch.mean((final.float() - reference.float()).pow(2), dim=(1, 2, 3))
        else:
            values: list[float] = []
            if perceptual is None:
                raise RuntimeError("perceptual bundle unexpectedly missing")
            with torch.autocast(device_type=reference.device.type, enabled=False):
                for item in range(batch):
                    p = perceptual(reference[item : item + 1].float(), final[item : item + 1].float())
                    values.append(float(getattr(p, objective)))
            distortion = torch.tensor(values, device=reference.device, dtype=torch.float32)
        if fidelity_metric == "image_l1":
            fidelity = torch.mean(torch.abs(final.float() - reference.float()), dim=(1, 2, 3))
        else:
            fidelity = torch.mean((final.float() - reference.float()).pow(2), dim=(1, 2, 3))
        total_payload = torch.tensor(
            [CoSERBitstream.bytes_to_bpp(int(value), height, width) for value in total_payload_bytes],
            device=reference.device,
            dtype=torch.float32,
        )
        scores.append(distortion + float(fidelity_lambda) * fidelity + float(rate_lambda) * total_payload)
        total_payload_bytes_by_candidate.append(total_payload_bytes)

    score_tensor = torch.stack(scores, dim=0)
    selected_index = torch.argmin(score_tensor, dim=0)
    selected_raw = torch.zeros_like(raw_outputs[0])
    selected_condition = torch.zeros_like(pred_condition, dtype=torch.float32)
    selected_abs = torch.zeros((batch,), device=reference.device, dtype=torch.float32)
    selected_payload_bytes = [0 for _ in range(batch)]
    selected_labels: list[str] = []
    selected_condition_l1 = torch.zeros((batch,), device=reference.device, dtype=torch.float32)
    selected_score = torch.zeros((batch,), device=reference.device, dtype=torch.float32)
    for item in range(batch):
        index = int(selected_index[item].item())
        label, correction, _, _, _ = unpack_control_candidate(candidates[index])
        selected_labels.append(label)
        selected_payload_bytes[item] = int(total_payload_bytes_by_candidate[index][item])
        selected_raw[item] = raw_outputs[index][item]
        selected_condition[item] = pred_condition.float()[item] + correction.float()[item]
        selected_abs[item] = abs_values[index][item]
        selected_condition_l1[item] = condition_l1_values[index][item]
        selected_score[item] = score_tensor[index, item]
    return (
        selected_condition,
        selected_raw,
        selected_payload_bytes,
        selected_abs.detach().float(),
        selected_labels,
        selected_index.detach(),
        selected_condition_l1.detach().float(),
        selected_score.detach().float(),
    )


def load_control_basis(path: str, *, device: torch.device) -> dict[str, object]:
    if not path:
        raise ValueError("--control-basis is required for condition_residual_basis")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    basis = payload.get("basis")
    if not isinstance(basis, torch.Tensor) or basis.ndim != 4:
        raise ValueError("control basis checkpoint must contain basis tensor with shape KGSS")
    mean = payload.get("mean")
    if mean is not None and (not isinstance(mean, torch.Tensor) or mean.ndim != 3):
        raise ValueError("control basis mean must be a GSS tensor when present")
    groups = int(payload.get("groups", basis.shape[1]))
    grid_size = int(payload.get("grid_size", basis.shape[-1]))
    if tuple(basis.shape[1:]) != (groups, grid_size, grid_size):
        raise ValueError("control basis metadata does not match basis tensor shape")
    if mean is not None and tuple(mean.shape) != (groups, grid_size, grid_size):
        raise ValueError("control basis mean shape does not match basis metadata")
    return {
        "basis": basis.to(device=device, dtype=torch.float32),
        "mean": mean.to(device=device, dtype=torch.float32) if isinstance(mean, torch.Tensor) else None,
        "groups": groups,
        "grid_size": grid_size,
        "source": str(payload.get("source", "")),
        "explained_variance": payload.get("explained_variance", []),
        "cumulative_explained_variance": payload.get("cumulative_explained_variance", []),
        "coefficient_abs_quantiles": payload.get("coefficient_abs_quantiles", {}),
        "coefficient_component_abs_p95": payload.get("coefficient_component_abs_p95", []),
        "coefficient_component_abs_p99": payload.get("coefficient_component_abs_p99", []),
        "coefficient_abs_mean": payload.get("coefficient_abs_mean", 0.0),
        "coefficient_abs_std": payload.get("coefficient_abs_std", 0.0),
        "coefficient_abs_max": payload.get("coefficient_abs_max", 0.0),
        "control_huffman_priors": payload.get("control_huffman_priors", {}),
        "coefficient_component_codebooks": payload.get("coefficient_component_codebooks", {}),
        "coefficient_vector_codebooks": payload.get("coefficient_vector_codebooks", {}),
        "sparse_topk_control_priors": payload.get("sparse_topk_control_priors", {}),
    }


def basis_component_ranges(
    basis_payload: dict[str, object],
    *,
    mode: str,
    components: int,
    floor: float,
) -> list[float] | None:
    if mode == "global":
        return None
    if mode not in {"component_p95", "component_p99"}:
        raise ValueError("basis range mode must be global, component_p95, or component_p99")
    if components <= 0:
        raise ValueError("components must be positive")
    if floor <= 0:
        raise ValueError("component range floor must be positive")
    key = "coefficient_component_abs_p95" if mode == "component_p95" else "coefficient_component_abs_p99"
    raw = basis_payload.get(key, [])
    if not isinstance(raw, list) or len(raw) < components:
        raise ValueError(f"control basis checkpoint does not contain enough {key} entries")
    return [max(float(value), float(floor)) for value in raw[:components]]


def basis_component_codebook_codec(
    basis_payload: dict[str, object],
    *,
    bits: int,
    components: int,
) -> ComponentCodebookControlCode:
    priors = basis_payload.get("coefficient_component_codebooks", {})
    if not isinstance(priors, dict) or not priors:
        raise ValueError("control basis checkpoint does not contain coefficient_component_codebooks")
    key = f"lloyd_b{int(bits)}"
    prior = priors.get(key)
    if not isinstance(prior, dict):
        matches = [
            candidate
            for candidate in priors.values()
            if isinstance(candidate, dict) and int(candidate.get("bits", -1)) == int(bits)
        ]
        if len(matches) != 1:
            raise ValueError(f"control basis checkpoint does not contain a unique component codebook for bits={bits}")
        prior = matches[0]
    raw_codebooks = prior.get("codebooks")
    if not isinstance(raw_codebooks, list) or len(raw_codebooks) < components:
        raise ValueError("component codebook prior does not contain enough component rows")
    selected = raw_codebooks[:components]
    return ComponentCodebookControlCode(
        bits=int(bits),
        codebooks=tuple(tuple(float(v) for v in row) for row in selected),
    )


def basis_vector_codebook_prior(
    basis_payload: dict[str, object],
    *,
    components: int,
    bits: int,
    key: str = "",
) -> tuple[str, dict[str, object]]:
    priors = basis_payload.get("coefficient_vector_codebooks", {})
    if not isinstance(priors, dict) or not priors:
        raise ValueError("control basis checkpoint does not contain coefficient_vector_codebooks")
    if not key:
        key = f"vq_k{int(components)}_b{int(bits)}"
    prior = priors.get(key)
    if not isinstance(prior, dict):
        raise ValueError(f"unknown vector codebook prior key: {key}")
    if int(prior.get("components", -1)) != int(components):
        raise ValueError("vector codebook components do not match requested value")
    if int(prior.get("bits", -1)) != int(bits):
        raise ValueError("vector codebook bits do not match requested value")
    return key, prior


def basis_vector_codebook_codec(
    basis_payload: dict[str, object],
    *,
    components: int,
    bits: int,
    key: str = "",
) -> tuple[VectorCodebookControlCode, str]:
    resolved_key, prior = basis_vector_codebook_prior(
        basis_payload,
        components=components,
        bits=bits,
        key=key,
    )
    raw_vectors = prior.get("vectors")
    if not isinstance(raw_vectors, list):
        raise ValueError("vector codebook prior does not contain vectors")
    return (
        VectorCodebookControlCode(
            bits=int(bits),
            vectors=tuple(tuple(float(v) for v in row) for row in raw_vectors),
        ),
        resolved_key,
    )


def codec_for_selected_basis_indices(
    codec: object,
    indices: torch.Tensor,
) -> object:
    """Return a value codec whose component ranges follow selected basis indices."""

    if isinstance(codec, ComponentCodebookControlCode):
        return codec.select(indices)
    if isinstance(codec, ComponentUniformControlCode):
        ranges = torch.tensor(codec.value_ranges, dtype=torch.float32)
        selected = ranges.index_select(0, indices.detach().cpu().to(torch.long))
        return ComponentUniformControlCode(bits=codec.bits, value_ranges=tuple(float(v) for v in selected.tolist()))
    if isinstance(codec, ComponentMuLawControlCode):
        ranges = torch.tensor(codec.value_ranges, dtype=torch.float32)
        selected = ranges.index_select(0, indices.detach().cpu().to(torch.long))
        return ComponentMuLawControlCode(
            bits=codec.bits,
            value_ranges=tuple(float(v) for v in selected.tolist()),
            mu=codec.mu,
        )
    return codec


def load_control_huffman_prior(
    basis_payload: dict[str, object],
    *,
    key: str,
    components: int,
) -> tuple[StaticControlHuffmanCode, int, float, str, float, str, str]:
    priors = basis_payload.get("control_huffman_priors", {})
    if not isinstance(priors, dict) or not priors:
        raise ValueError("control basis checkpoint does not contain control_huffman_priors")
    if not key:
        if len(priors) != 1:
            raise ValueError("--control-huffman-key is required when the basis contains multiple Huffman priors")
        key = next(iter(priors))
    prior = priors.get(key)
    if not isinstance(prior, dict):
        raise ValueError(f"unknown control Huffman prior key: {key}")
    huffman_payload = prior.get("huffman")
    if not isinstance(huffman_payload, dict):
        raise ValueError("control Huffman prior must contain a huffman payload")
    bits = int(prior.get("bits", 0))
    value_range = float(prior.get("range", 0.0))
    quantizer = str(prior.get("quantizer", "uniform"))
    mu = float(prior.get("mu", 16.0))
    basis_range_mode = str(prior.get("basis_range_mode", "global"))
    if bits <= 0 or value_range <= 0:
        raise ValueError("control Huffman prior must contain positive bits and range")
    if basis_range_mode not in {"global", "component_codebook"}:
        raise ValueError("control Huffman prior basis_range_mode must be global or component_codebook")
    huffman = StaticControlHuffmanCode.from_dict(huffman_payload).prefix(components)
    return huffman, bits, value_range, quantizer, mu, key, basis_range_mode


def load_sparse_topk_control_huffman_prior(
    basis_payload: dict[str, object],
    *,
    key: str,
    candidate_components: int,
    components: int,
) -> tuple[StaticControlHuffmanCode, StaticControlHuffmanCode, int, float, str, float, str, str]:
    priors = basis_payload.get("sparse_topk_control_priors", {})
    if not isinstance(priors, dict) or not priors:
        raise ValueError("control basis checkpoint does not contain sparse_topk_control_priors")
    if not key:
        matches = [
            name
            for name, prior in priors.items()
            if isinstance(prior, dict)
            and int(prior.get("candidate_components", -1)) == int(candidate_components)
            and int(prior.get("selected_components", -1)) == int(components)
        ]
        if len(matches) != 1:
            raise ValueError(
                "--control-huffman-key is required when sparse top-k priors are ambiguous "
                f"for candidate_components={candidate_components}, components={components}"
            )
        key = matches[0]
    prior = priors.get(key)
    if not isinstance(prior, dict):
        raise ValueError(f"unknown sparse top-k control Huffman prior key: {key}")
    if int(prior.get("candidate_components", -1)) != int(candidate_components):
        raise ValueError("sparse top-k prior candidate_components does not match requested value")
    if int(prior.get("selected_components", -1)) != int(components):
        raise ValueError("sparse top-k prior selected_components does not match requested value")
    index_payload = prior.get("index_huffman")
    value_payload = prior.get("value_huffman")
    if not isinstance(index_payload, dict) or not isinstance(value_payload, dict):
        raise ValueError("sparse top-k prior must contain index_huffman and value_huffman payloads")
    bits = int(prior.get("bits", 0))
    value_range = float(prior.get("range", 0.0))
    quantizer = str(prior.get("quantizer", "uniform"))
    mu = float(prior.get("mu", 16.0))
    basis_range_mode = str(prior.get("basis_range_mode", "global"))
    if bits <= 0 or value_range <= 0:
        raise ValueError("sparse top-k prior must contain positive bits and range")
    if basis_range_mode not in {"global", "component_codebook"}:
        raise ValueError("sparse top-k prior basis_range_mode must be global or component_codebook")
    index_huffman = StaticControlHuffmanCode.from_dict(index_payload)
    value_huffman = StaticControlHuffmanCode.from_dict(value_payload)
    return index_huffman, value_huffman, bits, value_range, quantizer, mu, key, basis_range_mode


def load_vector_codebook_huffman_prior(
    basis_payload: dict[str, object],
    *,
    key: str,
    components: int,
    bits: int,
) -> tuple[StaticControlHuffmanCode, int, str]:
    resolved_key, prior = basis_vector_codebook_prior(
        basis_payload,
        components=components,
        bits=bits,
        key=key,
    )
    huffman_payload = prior.get("huffman")
    if not isinstance(huffman_payload, dict):
        raise ValueError("vector codebook prior must contain a huffman payload")
    huffman = StaticControlHuffmanCode.from_dict(huffman_payload)
    if huffman.symbol_shape != (1,):
        raise ValueError("vector codebook Huffman symbol_shape must be (1,)")
    return huffman, int(prior["bits"]), resolved_key


def grouped_condition_basis_control(
    condition_error: torch.Tensor,
    *,
    basis_payload: dict[str, object],
    components: int,
    candidate_components: int,
    selection: str,
    codec: UniformControlGridCode,
    prefix_components: int = 0,
    huffman: StaticControlHuffmanCode | None = None,
    sparse_huffman: tuple[StaticControlHuffmanCode, StaticControlHuffmanCode] | None = None,
    scale: float,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    if condition_error.ndim != 4:
        raise ValueError("condition_error must be BCHW")
    basis = basis_payload["basis"]
    if not isinstance(basis, torch.Tensor):
        raise TypeError("basis_payload['basis'] must be a tensor")
    mean = basis_payload.get("mean")
    if mean is not None and not isinstance(mean, torch.Tensor):
        raise TypeError("basis_payload['mean'] must be a tensor or None")
    groups = int(basis_payload["groups"])
    grid_size = int(basis_payload["grid_size"])
    batch, channels, height, width = condition_error.shape
    sizes = channel_group_sizes(int(channels), groups)
    grouped_maps: list[torch.Tensor] = []
    start = 0
    for size in sizes:
        end = start + size
        grouped_maps.append(condition_error[:, start:end].float().mean(dim=1, keepdim=True))
        start = end
    control_grid = F.adaptive_avg_pool2d(torch.cat(grouped_maps, dim=1), output_size=(grid_size, grid_size))
    selection = str(selection)
    if selection == "prefix":
        coeffs = project_onto_control_basis(control_grid, basis, mean, components=components)
        quantized_coeffs = codec.quantize(coeffs)

        decoded_items: list[torch.Tensor] = []
        payload_bytes: list[int] = []
        for item in range(batch):
            if huffman is None:
                payload = codec.encode(quantized_coeffs[item].detach().cpu())
                decoded_codes = codec.decode(payload, shape=(components,))
            else:
                payload = huffman.encode(quantized_coeffs[item].detach().cpu())
                decoded_codes = huffman.decode(payload, shape=(components,))
            decoded = codec.dequantize(decoded_codes).to(device=condition_error.device, dtype=torch.float32)
            decoded_items.append(decoded)
            payload_bytes.append(len(payload))
        coeff_abs_mean = coeffs.detach().float().abs().mean(dim=1)
    elif selection == "vector":
        if not isinstance(codec, VectorCodebookControlCode):
            raise TypeError("selection=vector requires VectorCodebookControlCode")
        if codec.vector_dim != int(components):
            raise ValueError("vector codebook dimension must match components")
        coeffs = project_onto_control_basis(control_grid, basis, mean, components=components)
        quantized_indices = codec.quantize(coeffs)
        decoded_items = []
        payload_bytes = []
        for item in range(batch):
            code = quantized_indices[item].reshape(1).detach().cpu()
            if huffman is None:
                payload = codec.encode(code)
                decoded_code = codec.decode(payload, shape=(1,))
            else:
                payload = huffman.encode(code)
                decoded_code = huffman.decode(payload, shape=(1,))
            decoded = codec.dequantize(decoded_code).reshape(components).to(
                device=condition_error.device,
                dtype=torch.float32,
            )
            decoded_items.append(decoded)
            payload_bytes.append(len(payload))
        coeff_abs_mean = coeffs.detach().float().abs().mean(dim=1)
    elif selection == "topk":
        if huffman is not None:
            raise ValueError("prefix huffman must not be used with topk basis selection")
        if candidate_components <= 0:
            raise ValueError("candidate_components must be positive for topk basis selection")
        if components > candidate_components:
            raise ValueError("components must be <= candidate_components for topk basis selection")
        coeffs = project_onto_control_basis(control_grid, basis, mean, components=candidate_components)
        sparse_code = SparseControlBasisCode(
            candidate_components=int(candidate_components),
            selected_components=int(components),
            value_codec=codec,
        )
        decoded_items = []
        payload_bytes = []
        coeff_abs_values: list[torch.Tensor] = []
        for item in range(batch):
            selected_indices, selected_values = sparse_code.select(coeffs[item].detach().cpu())
            item_value_codec = codec_for_selected_basis_indices(codec, selected_indices)
            if not isinstance(
                item_value_codec,
                (
                    UniformControlGridCode,
                    MuLawControlGridCode,
                    ComponentUniformControlCode,
                    ComponentMuLawControlCode,
                    ComponentCodebookControlCode,
                ),
            ):
                raise TypeError("top-k basis value codec has an unsupported type")
            quantized_values = item_value_codec.quantize(selected_values)
            item_sparse_code = SparseControlBasisCode(
                candidate_components=int(candidate_components),
                selected_components=int(components),
                value_codec=item_value_codec,
            )
            if sparse_huffman is None:
                compact_payload = item_sparse_code.encode_compact(selected_indices, quantized_values)
                decoded_indices, decoded_codes = item_sparse_code.decode_compact(compact_payload)
                payload_len = len(compact_payload)
            else:
                index_huffman, value_huffman = sparse_huffman
                index_payload, value_payload = item_sparse_code.encode_entropy(
                    selected_indices,
                    quantized_values,
                    index_huffman=index_huffman,
                    value_huffman=value_huffman,
                )
                decoded_indices, decoded_codes = item_sparse_code.decode_entropy(
                    index_payload,
                    value_payload,
                    index_huffman=index_huffman,
                    value_huffman=value_huffman,
                )
                payload_len = len(index_payload) + len(value_payload)
            decoded_values = item_value_codec.dequantize(decoded_codes).to(
                device=condition_error.device,
                dtype=torch.float32,
            )
            decoded_full = item_sparse_code.scatter_values(decoded_indices, decoded_values).to(
                device=condition_error.device
            )
            decoded_items.append(decoded_full)
            payload_bytes.append(payload_len)
            coeff_abs_values.append(selected_values.float().abs().mean())
        coeff_abs_mean = torch.stack(coeff_abs_values).to(device=condition_error.device)
    elif selection == "prefix_topk":
        if huffman is not None or sparse_huffman is not None:
            raise ValueError("prefix_topk basis selection currently supports fixed_bits only")
        if candidate_components <= 0:
            raise ValueError("candidate_components must be positive for prefix_topk basis selection")
        if prefix_components <= 0:
            raise ValueError("prefix_components must be positive for prefix_topk basis selection")
        if prefix_components >= candidate_components:
            raise ValueError("prefix_components must be < candidate_components")
        if components > candidate_components - prefix_components:
            raise ValueError("components must be <= candidate_components - prefix_components")
        coeffs = project_onto_control_basis(control_grid, basis, mean, components=candidate_components)
        prefix_topk_code = PrefixTopKControlBasisCode(
            candidate_components=int(candidate_components),
            prefix_components=int(prefix_components),
            selected_components=int(components),
            value_codec=codec,
        )
        prefix_indices = torch.arange(int(prefix_components), dtype=torch.long)
        prefix_value_codec = codec_for_selected_basis_indices(codec, prefix_indices)
        if not isinstance(
            prefix_value_codec,
            (
                UniformControlGridCode,
                MuLawControlGridCode,
                ComponentUniformControlCode,
                ComponentMuLawControlCode,
                ComponentCodebookControlCode,
            ),
        ):
            raise TypeError("prefix-top-k prefix value codec has an unsupported type")
        decoded_items = []
        payload_bytes = []
        coeff_abs_values = []
        for item in range(batch):
            prefix_values, local_indices, global_indices, selected_values = prefix_topk_code.select(
                coeffs[item].detach().cpu()
            )
            selected_value_codec = codec_for_selected_basis_indices(codec, global_indices)
            if not isinstance(
                selected_value_codec,
                (
                    UniformControlGridCode,
                    MuLawControlGridCode,
                    ComponentUniformControlCode,
                    ComponentMuLawControlCode,
                    ComponentCodebookControlCode,
                ),
            ):
                raise TypeError("prefix-top-k selected value codec has an unsupported type")
            prefix_codes = prefix_value_codec.quantize(prefix_values)
            selected_codes = selected_value_codec.quantize(selected_values)
            compact_payload = prefix_topk_code.encode_compact(prefix_codes, local_indices, selected_codes)
            decoded_prefix_codes, decoded_local_indices, decoded_selected_codes = prefix_topk_code.decode_compact(
                compact_payload
            )
            decoded_prefix_values = prefix_value_codec.dequantize(decoded_prefix_codes).to(
                device=condition_error.device,
                dtype=torch.float32,
            )
            decoded_global_indices = decoded_local_indices + int(prefix_components)
            decoded_selected_value_codec = codec_for_selected_basis_indices(codec, decoded_global_indices)
            decoded_selected_values = decoded_selected_value_codec.dequantize(decoded_selected_codes).to(
                device=condition_error.device,
                dtype=torch.float32,
            )
            decoded_full = prefix_topk_code.scatter_values(
                decoded_prefix_values,
                decoded_local_indices,
                decoded_selected_values,
            ).to(device=condition_error.device)
            decoded_items.append(decoded_full)
            payload_bytes.append(len(compact_payload))
            coeff_abs_values.append(torch.cat([prefix_values.float().abs(), selected_values.float().abs()]).mean())
        coeff_abs_mean = torch.stack(coeff_abs_values).to(device=condition_error.device)
    else:
        raise ValueError("selection must be prefix, vector, topk, or prefix_topk")
    decoded_coeffs = torch.stack(decoded_items, dim=0)
    reconstructed_grid = reconstruct_from_control_basis(decoded_coeffs, basis, mean)

    expanded_groups: list[torch.Tensor] = []
    for group, size in enumerate(sizes):
        expanded_groups.append(reconstructed_grid[:, group : group + 1].repeat(1, size, 1, 1))
    expanded = torch.cat(expanded_groups, dim=1)
    correction = F.interpolate(expanded, size=(height, width), mode="bilinear", align_corners=False)
    return correction * float(scale), payload_bytes, coeff_abs_mean


def write_run_doc(path: Path, payload: dict[str, object]) -> None:
    lines = [
        f"# {payload['run_name']}",
        "",
        f"Date: {payload['date']}",
        "",
        "## Command",
        "",
        "```bash",
        str(payload["command"]),
        "```",
        "",
        "## Summary",
        "",
    ]
    for key, value in dict(payload["summary"]).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Artifacts", ""])
    for key, value in dict(payload["artifacts"]).items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate-checkpoint", default="")
    parser.add_argument("--condition-gate-checkpoint", default="")
    parser.add_argument("--post-control-condition-gate-checkpoint", default="")
    parser.add_argument("--condition-delta-ablation", choices=("normal", "zero"), default="normal")
    parser.add_argument("--semantic-latent-ablation", choices=("normal", "zero", "shuffle"), default="normal")
    parser.add_argument("--detail-context-ablation", choices=("normal", "zero", "shuffle"), default="normal")
    parser.add_argument("--ablation-shuffle-seed", type=int, default=1234)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--per-image-metrics", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="results/stage4_cod_lite_adapter_eval")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument("--save-reconstruction-limit", type=int, default=0)
    parser.add_argument(
        "--save-reconstruction-kinds",
        default="reference,semantic_only,stage3,stage4,quad",
        help="Comma-separated subset of reference, semantic_only, stage3, stage4, quad.",
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=1.0,
        help="Deterministic decoder-side blend: stage4 = (1-alpha) * stage3 + alpha * cod_lite_adapter.",
    )
    parser.add_argument(
        "--condition-residual-scale",
        type=float,
        default=None,
        help="Override checkpoint condition residual scale. Defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--condition-residual-tanh",
        choices=("checkpoint", "true", "false"),
        default="checkpoint",
        help="Override checkpoint condition residual tanh setting.",
    )
    parser.add_argument("--condition-residual-guard", choices=("none", "rms_clip"), default="none")
    parser.add_argument(
        "--condition-residual-guard-granularity",
        choices=("global", "spatial", "channel"),
        default="global",
    )
    parser.add_argument("--condition-residual-max-rms-ratio", type=float, default=0.5)
    parser.add_argument("--condition-residual-min-gate", type=float, default=0.0)
    parser.add_argument(
        "--counted-control-mode",
        choices=(
            "none",
            "condition_residual_grid",
            "condition_residual_dct",
            "condition_residual_basis",
            "condition_residual_affine",
            "condition_residual_affine_dct",
            "condition_residual_affine_grid",
            "condition_residual_affine_basis",
            "condition_residual_hybrid_affine_dct_grid",
            "condition_residual_hybrid_affine_dct_grid_basis",
        ),
        default="none",
    )
    parser.add_argument("--control-grid-size", type=int, default=4)
    parser.add_argument("--control-groups", type=int, default=8)
    parser.add_argument("--control-dct-coeffs-per-group", type=int, default=4)
    parser.add_argument("--control-basis", default="")
    parser.add_argument("--control-basis-components", type=int, default=8)
    parser.add_argument("--control-basis-candidate-components", type=int, default=0)
    parser.add_argument("--control-basis-prefix-components", type=int, default=0)
    parser.add_argument("--control-basis-selection", choices=("prefix", "topk", "vector", "prefix_topk"), default="prefix")
    parser.add_argument(
        "--control-basis-range-mode",
        choices=("global", "component_p95", "component_p99", "component_codebook"),
        default="global",
        help="Use scalar, fixed per-component ranges, or fixed per-component codebooks for basis controls.",
    )
    parser.add_argument("--control-basis-range-floor", type=float, default=1.0e-6)
    parser.add_argument("--control-codec", choices=("fixed_bits", "huffman"), default="fixed_bits")
    parser.add_argument("--control-huffman-key", default="")
    parser.add_argument("--control-quantizer", choices=("uniform", "mu_law"), default="uniform")
    parser.add_argument("--control-mu", type=float, default=16.0)
    parser.add_argument("--control-bits", type=int, default=4)
    parser.add_argument("--control-range", type=float, default=0.25)
    parser.add_argument(
        "--control-affine-groups",
        type=int,
        default=0,
        help="Override affine-control channel groups. Defaults to --control-groups.",
    )
    parser.add_argument(
        "--control-affine-grid-size",
        type=int,
        default=0,
        help="Override affine-control grid size. Defaults to --control-grid-size.",
    )
    parser.add_argument("--control-affine-gain-range", type=float, default=1.0)
    parser.add_argument("--control-affine-bias-range", type=float, default=0.25)
    parser.add_argument(
        "--control-hybrid-selector-bytes",
        type=int,
        default=1,
        help="Conservative per-image bytes counted for the hybrid control mode selector.",
    )
    parser.add_argument(
        "--control-hybrid-selector-bits",
        type=int,
        default=0,
        help=(
            "Optional exact selector bit count for fixed-bit hybrid payload packing. "
            "Variable-length candidates fall back to --control-hybrid-selector-bytes."
        ),
    )
    parser.add_argument(
        "--control-hybrid-rate-lambda",
        type=float,
        default=0.0,
        help="Condition-space RD proxy penalty applied to extra control bpp when choosing hybrid mode.",
    )
    parser.add_argument(
        "--control-hybrid-selection-objective",
        choices=("condition_l1", "image_l1", "image_mse", "lpips_alex", "dists"),
        default="condition_l1",
        help="Encoder-side RDO proxy used to choose the counted hybrid control mode.",
    )
    parser.add_argument(
        "--control-hybrid-fidelity-lambda",
        type=float,
        default=0.0,
        help="Optional image fidelity guard added to image-RDO hybrid selection scores.",
    )
    parser.add_argument(
        "--control-hybrid-fidelity-metric",
        choices=("image_l1", "image_mse"),
        default="image_l1",
        help="Image fidelity guard metric used when --control-hybrid-fidelity-lambda is positive.",
    )
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 4 evaluation.")
    if args.gate_checkpoint and args.condition_gate_checkpoint:
        raise ValueError("--gate-checkpoint and --condition-gate-checkpoint are mutually exclusive")
    if (args.condition_gate_checkpoint or args.post_control_condition_gate_checkpoint) and args.blend_alpha != 1.0:
        raise ValueError("condition-gate evaluation does not use RGB output blend; keep --blend-alpha 1.0")
    if args.condition_residual_max_rms_ratio <= 0:
        raise ValueError("--condition-residual-max-rms-ratio must be positive")
    if not 0.0 <= args.condition_residual_min_gate <= 1.0:
        raise ValueError("--condition-residual-min-gate must be in [0, 1]")
    if args.control_grid_size <= 0:
        raise ValueError("--control-grid-size must be positive")
    if args.control_groups <= 0:
        raise ValueError("--control-groups must be positive")
    effective_affine_groups = int(args.control_affine_groups or args.control_groups)
    effective_affine_grid_size = int(args.control_affine_grid_size or args.control_grid_size)
    if effective_affine_groups <= 0:
        raise ValueError("--control-affine-groups must be positive when provided")
    if effective_affine_grid_size <= 0:
        raise ValueError("--control-affine-grid-size must be positive when provided")
    if args.control_bits < 1:
        raise ValueError("--control-bits must be >= 1")
    if args.control_range <= 0:
        raise ValueError("--control-range must be positive")
    if args.control_basis_range_floor <= 0:
        raise ValueError("--control-basis-range-floor must be positive")
    if args.control_affine_gain_range <= 0:
        raise ValueError("--control-affine-gain-range must be positive")
    if args.control_affine_bias_range <= 0:
        raise ValueError("--control-affine-bias-range must be positive")
    if args.control_hybrid_selector_bytes < 0:
        raise ValueError("--control-hybrid-selector-bytes must be non-negative")
    if args.control_hybrid_selector_bits < 0:
        raise ValueError("--control-hybrid-selector-bits must be non-negative")
    if args.control_hybrid_rate_lambda < 0:
        raise ValueError("--control-hybrid-rate-lambda must be non-negative")
    if args.control_hybrid_fidelity_lambda < 0:
        raise ValueError("--control-hybrid-fidelity-lambda must be non-negative")
    if args.control_hybrid_selection_objective in {"image_l1", "image_mse", "lpips_alex", "dists"} and (
        args.gate_checkpoint or args.condition_gate_checkpoint
    ):
        raise ValueError("image-RDO hybrid selection currently supports fixed blend-alpha without learned gates")
    if args.control_mu <= 0:
        raise ValueError("--control-mu must be positive")
    if args.control_dct_coeffs_per_group <= 0:
        raise ValueError("--control-dct-coeffs-per-group must be positive")
    if args.control_dct_coeffs_per_group > args.control_grid_size * args.control_grid_size:
        raise ValueError("--control-dct-coeffs-per-group must be <= --control-grid-size squared")
    if args.control_basis_components <= 0:
        raise ValueError("--control-basis-components must be positive")
    if args.control_basis_candidate_components < 0:
        raise ValueError("--control-basis-candidate-components must be non-negative")
    if args.control_basis_prefix_components < 0:
        raise ValueError("--control-basis-prefix-components must be non-negative")
    if args.control_basis_selection == "prefix_topk" and args.control_codec == "huffman":
        raise ValueError("selection=prefix_topk currently supports fixed_bits only")
    if args.control_codec == "huffman" and args.counted_control_mode not in {
        "condition_residual_basis",
        "condition_residual_affine_basis",
        "condition_residual_hybrid_affine_dct_grid_basis",
    }:
        raise ValueError("--control-codec huffman is currently supported only for basis control modes")
    if args.control_basis_range_mode in {"component_p95", "component_p99"} and args.control_codec != "fixed_bits":
        raise ValueError("--control-basis-range-mode component_p* currently supports only fixed_bits basis controls")

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda")
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_stage4_cod_lite_adapter_eval"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_kinds = tuple(kind.strip() for kind in args.save_reconstruction_kinds.split(",") if kind.strip())
    valid_save_kinds = {"reference", "semantic_only", "stage3", "stage4", "quad"}
    unknown_save_kinds = sorted(set(save_kinds) - valid_save_kinds)
    if unknown_save_kinds:
        raise ValueError(f"unknown save reconstruction kinds: {unknown_save_kinds}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone_cfg = CoDLiteOneStepBackboneConfig(**payload["backbone_config"])
    backbone = CoDLiteOneStepBackbone.load(backbone_cfg, device=device)
    apply_lora_adapters_from_config(backbone.net, payload.get("backbone_lora_config", {}))
    if payload.get("backbone_trainable_state"):
        load_named_parameter_state(backbone.net, payload["backbone_trainable_state"], strict=True)
    backbone.eval()
    adapter = build_adapter_from_payload(payload).to(device)
    adapter.load_state_dict(payload["model"])
    adapter.eval()
    gate = None
    gate_payload: dict[str, object] = {}
    if args.gate_checkpoint:
        gate_payload = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=False)
        gate = build_gate_from_payload(gate_payload).to(device)
        gate.load_state_dict(gate_payload["gate_model"])
        gate.eval()
    condition_gate = None
    condition_gate_payload: dict[str, object] = {}
    if args.condition_gate_checkpoint:
        condition_gate_payload = torch.load(args.condition_gate_checkpoint, map_location="cpu", weights_only=False)
        condition_gate = build_condition_gate_from_payload(condition_gate_payload).to(device)
        condition_gate.load_state_dict(condition_gate_payload["condition_gate_model"])
        condition_gate.eval()
    post_control_condition_gate = None
    post_control_condition_gate_payload: dict[str, object] = {}
    if args.post_control_condition_gate_checkpoint:
        post_control_condition_gate_payload = torch.load(
            args.post_control_condition_gate_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        post_control_condition_gate = build_condition_gate_from_payload(post_control_condition_gate_payload).to(device)
        post_control_condition_gate.load_state_dict(post_control_condition_gate_payload["condition_gate_model"])
        post_control_condition_gate.eval()
    adapter_config = dict(payload.get("adapter_config", {}))
    semantic_channels = int(adapter_config.get("semantic_channels", 3))
    detail_context = str(payload.get("detail_context", "none"))
    detail_channels = detail_context_channels(detail_context)
    residual_scale = float(
        payload.get("condition_residual_scale", 1.0)
        if args.condition_residual_scale is None
        else args.condition_residual_scale
    )
    if args.condition_residual_tanh == "checkpoint":
        residual_tanh = bool(payload.get("condition_residual_tanh", False))
    else:
        residual_tanh = args.condition_residual_tanh == "true"
    if args.counted_control_mode != "none":
        channel_group_sizes(backbone.condition_channels, args.control_groups)
    if args.counted_control_mode in {
        "condition_residual_affine",
        "condition_residual_affine_dct",
        "condition_residual_affine_grid",
        "condition_residual_affine_basis",
        "condition_residual_hybrid_affine_dct_grid",
        "condition_residual_hybrid_affine_dct_grid_basis",
    }:
        channel_group_sizes(backbone.condition_channels, effective_affine_groups)
    control_basis_payload: dict[str, object] | None = None
    control_huffman: StaticControlHuffmanCode | None = None
    control_sparse_huffman: tuple[StaticControlHuffmanCode, StaticControlHuffmanCode] | None = None
    effective_control_bits = int(args.control_bits)
    effective_control_range = float(args.control_range)
    effective_control_quantizer = str(args.control_quantizer)
    effective_control_mu = float(args.control_mu)
    effective_control_huffman_key = str(args.control_huffman_key)
    effective_control_basis_range_mode = str(args.control_basis_range_mode)
    effective_candidate_components = int(args.control_basis_components)
    if args.counted_control_mode in {
        "condition_residual_basis",
        "condition_residual_affine_basis",
        "condition_residual_hybrid_affine_dct_grid_basis",
    }:
        control_basis_payload = load_control_basis(args.control_basis, device=device)
        if int(control_basis_payload["groups"]) != args.control_groups:
            raise ValueError("--control-groups must match control basis groups")
        if int(control_basis_payload["grid_size"]) != args.control_grid_size:
            raise ValueError("--control-grid-size must match control basis grid_size")
        basis_tensor = control_basis_payload["basis"]
        if not isinstance(basis_tensor, torch.Tensor):
            raise TypeError("loaded control basis is not a tensor")
        effective_candidate_components = int(args.control_basis_components)
        if args.control_basis_selection in {"topk", "prefix_topk"} and args.control_basis_candidate_components > 0:
            effective_candidate_components = int(args.control_basis_candidate_components)
        if args.control_basis_components > effective_candidate_components:
            raise ValueError("--control-basis-components must be <= --control-basis-candidate-components")
        if args.control_basis_selection == "vector":
            effective_candidate_components = int(args.control_basis_components)
        if args.control_basis_selection == "prefix_topk":
            if args.control_basis_prefix_components <= 0:
                raise ValueError("--control-basis-prefix-components must be positive for selection=prefix_topk")
            if args.control_basis_prefix_components >= effective_candidate_components:
                raise ValueError("--control-basis-prefix-components must be < candidate components")
            if args.control_basis_components > effective_candidate_components - args.control_basis_prefix_components:
                raise ValueError(
                    "--control-basis-components must be <= candidate-prefix components for selection=prefix_topk"
                )
        if effective_candidate_components > int(basis_tensor.shape[0]):
            raise ValueError("--control-basis-candidate-components exceeds basis tensor count")
        if args.control_codec == "huffman" and args.control_basis_selection == "prefix":
            (
                control_huffman,
                effective_control_bits,
                effective_control_range,
                effective_control_quantizer,
                effective_control_mu,
                effective_control_huffman_key,
                prior_basis_range_mode,
            ) = load_control_huffman_prior(
                control_basis_payload,
                key=args.control_huffman_key,
                components=args.control_basis_components,
            )
            if prior_basis_range_mode != "global":
                if effective_control_basis_range_mode not in {"global", prior_basis_range_mode}:
                    raise ValueError("Huffman prior basis_range_mode conflicts with --control-basis-range-mode")
                effective_control_basis_range_mode = prior_basis_range_mode
        elif args.control_codec == "huffman" and args.control_basis_selection == "topk":
            (
                index_huffman,
                value_huffman,
                effective_control_bits,
                effective_control_range,
                effective_control_quantizer,
                effective_control_mu,
                effective_control_huffman_key,
                prior_basis_range_mode,
            ) = load_sparse_topk_control_huffman_prior(
                control_basis_payload,
                key=args.control_huffman_key,
                candidate_components=effective_candidate_components,
                components=args.control_basis_components,
            )
            control_sparse_huffman = (index_huffman, value_huffman)
            if prior_basis_range_mode != "global":
                if effective_control_basis_range_mode not in {"global", prior_basis_range_mode}:
                    raise ValueError("sparse Huffman prior basis_range_mode conflicts with --control-basis-range-mode")
                effective_control_basis_range_mode = prior_basis_range_mode
        elif args.control_codec == "huffman" and args.control_basis_selection == "vector":
            (
                control_huffman,
                effective_control_bits,
                effective_control_huffman_key,
            ) = load_vector_codebook_huffman_prior(
                control_basis_payload,
                key=args.control_huffman_key,
                components=args.control_basis_components,
                bits=args.control_bits,
            )
    control_codec = build_control_grid_code(
        quantizer=effective_control_quantizer,
        bits=effective_control_bits,
        value_range=effective_control_range,
        mu=effective_control_mu,
    )
    basis_range_components = (
        effective_candidate_components
        if args.control_basis_selection in {"topk", "prefix_topk"}
        else args.control_basis_components
    )
    if control_basis_payload is not None and args.control_basis_selection == "vector":
        basis_control_codec, vector_key = basis_vector_codebook_codec(
            control_basis_payload,
            components=args.control_basis_components,
            bits=effective_control_bits,
            key=effective_control_huffman_key if args.control_codec == "huffman" else "",
        )
        if not effective_control_huffman_key:
            effective_control_huffman_key = vector_key
    elif control_basis_payload is not None and effective_control_basis_range_mode == "component_codebook":
        basis_control_codec = basis_component_codebook_codec(
            control_basis_payload,
            bits=effective_control_bits,
            components=basis_range_components,
        )
    else:
        basis_value_ranges = (
            basis_component_ranges(
                control_basis_payload,
                mode=effective_control_basis_range_mode,
                components=basis_range_components,
                floor=args.control_basis_range_floor,
            )
            if control_basis_payload is not None
            else None
        )
        basis_control_codec = build_control_grid_code(
            quantizer=effective_control_quantizer,
            bits=effective_control_bits,
            value_range=effective_control_range,
            mu=effective_control_mu,
            value_ranges=basis_value_ranges,
        )
    affine_gain_codec = build_control_grid_code(
        quantizer=effective_control_quantizer,
        bits=effective_control_bits,
        value_range=args.control_affine_gain_range,
        mu=effective_control_mu,
    )
    affine_bias_codec = build_control_grid_code(
        quantizer=effective_control_quantizer,
        bits=effective_control_bits,
        value_range=args.control_affine_bias_range,
        mu=effective_control_mu,
    )

    dataset = Stage4ManifestDataset(
        Path(args.manifest),
        Path(args.per_image_metrics) if args.per_image_metrics else None,
        limit=args.limit,
        crop_size=args.crop_size,
        semantic_channels=semantic_channels,
        detail_context=detail_context,
        semantic_latent_ablation=args.semantic_latent_ablation,
        detail_context_ablation=args.detail_context_ablation,
        shuffle_seed=args.ablation_shuffle_seed,
        require_payload_bpp=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    perceptual = PerceptualMetricBundle().to(device)

    wandb_run = init_wandb(
        {
            "stage": "stage4_cod_lite_adapter_eval",
            "checkpoint": args.checkpoint,
            "gate_checkpoint": args.gate_checkpoint,
            "condition_gate_checkpoint": args.condition_gate_checkpoint,
            "post_control_condition_gate_checkpoint": args.post_control_condition_gate_checkpoint,
            "manifest": args.manifest,
            "per_image_metrics": args.per_image_metrics,
            "crop_size": args.crop_size,
            "limit": args.limit,
            "detail_context": detail_context,
            "condition_residual_scale": residual_scale,
            "condition_residual_tanh": residual_tanh,
            "condition_residual_guard": args.condition_residual_guard,
            "condition_residual_guard_granularity": args.condition_residual_guard_granularity,
            "condition_residual_max_rms_ratio": args.condition_residual_max_rms_ratio,
            "condition_residual_min_gate": args.condition_residual_min_gate,
            "condition_delta_ablation": args.condition_delta_ablation,
            "counted_control_mode": args.counted_control_mode,
            "control_grid_size": args.control_grid_size,
            "control_groups": args.control_groups,
            "control_dct_coeffs_per_group": args.control_dct_coeffs_per_group,
            "control_affine_groups": effective_affine_groups,
            "control_affine_grid_size": effective_affine_grid_size,
            "control_basis": args.control_basis,
            "control_basis_components": args.control_basis_components,
            "control_basis_candidate_components": effective_candidate_components,
            "control_basis_prefix_components": args.control_basis_prefix_components,
            "control_basis_selection": args.control_basis_selection,
            "control_basis_range_mode": effective_control_basis_range_mode,
            "control_basis_range_floor": args.control_basis_range_floor,
            "control_codec_type": args.control_codec,
            "control_huffman_key": effective_control_huffman_key,
            "control_quantizer": effective_control_quantizer,
            "control_mu": effective_control_mu,
            "control_bits": effective_control_bits,
            "control_range": effective_control_range,
            "control_affine_gain_range": args.control_affine_gain_range,
            "control_affine_bias_range": args.control_affine_bias_range,
            "control_hybrid_selector_bytes": args.control_hybrid_selector_bytes,
            "control_hybrid_selector_bits": args.control_hybrid_selector_bits,
            "control_hybrid_rate_lambda": args.control_hybrid_rate_lambda,
            "control_hybrid_selection_objective": args.control_hybrid_selection_objective,
            "control_hybrid_fidelity_lambda": args.control_hybrid_fidelity_lambda,
            "control_hybrid_fidelity_metric": args.control_hybrid_fidelity_metric,
            "control_scale": args.control_scale,
            "semantic_latent_ablation": args.semantic_latent_ablation,
            "detail_context_ablation": args.detail_context_ablation,
            "ablation_shuffle_seed": args.ablation_shuffle_seed,
            "save_reconstruction_kinds": list(save_kinds),
            "wandb": {"mode": args.wandb_mode, "project": "coserdic"},
        },
        run_name=run_name,
    )

    recon_dir = out_dir / "reconstructions"
    if args.save_reconstructions:
        for name in save_kinds:
            (recon_dir / name).mkdir(parents=True, exist_ok=True)

    metrics: dict[str, list[float]] = {
        "actual_payload_bpp": [],
        "stage3_actual_payload_bpp": [],
        "control_payload_bpp": [],
        "control_payload_bytes": [],
        "stage3_psnr": [],
        "stage4_psnr": [],
        "stage3_ms_ssim": [],
        "stage4_ms_ssim": [],
        "stage3_lpips_alex": [],
        "stage4_lpips_alex": [],
        "stage3_dists": [],
        "stage4_dists": [],
        "stage4_l1": [],
        "stage3_l1": [],
        "condition_l1": [],
        "pre_control_condition_l1": [],
        "control_condition_l1_delta": [],
        "ungated_condition_l1": [],
        "base_condition_l1": [],
        "condition_l1_delta_vs_base": [],
        "condition_residual_l1": [],
        "condition_delta_raw_l1": [],
        "condition_residual_guard_mean": [],
        "condition_residual_guard_min": [],
        "condition_residual_guard_max": [],
        "control_grid_abs_mean": [],
        "control_hybrid_mode_index": [],
        "control_hybrid_rdo_score": [],
        "stage4_alpha": [],
        "condition_gate_mean": [],
        "post_control_condition_gate_mean": [],
    }
    per_image: list[dict[str, object]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=run_name)):
            reference = batch["reference"].to(device, non_blocking=True)
            semantic = batch["semantic"].to(device, non_blocking=True)
            semantic_latent = batch["semantic_latent"].to(device, non_blocking=True)
            detail_context_tensor = (
                batch["detail_context"].to(device, non_blocking=True) if detail_channels > 0 else None
            )
            stage3 = batch["stage3"].to(device, non_blocking=True)
            residual = stage3 - semantic
            condition_size = backbone.condition_size(int(stage3.shape[-2]), int(stage3.shape[-1]))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                base_cond = backbone.native_condition(stage3)
                target_cond = backbone.native_condition(reference)
                cond_delta = adapter(
                    stage3,
                    semantic,
                    residual,
                    semantic_latent,
                    condition_size=condition_size,
                    base_condition=base_cond,
                    detail_context=detail_context_tensor,
                )
                if args.condition_delta_ablation == "zero":
                    cond_delta = torch.zeros_like(cond_delta)
                ungated_pred_cond = apply_condition_residual(
                    base_cond,
                    cond_delta,
                    residual_scale=residual_scale,
                    residual_tanh=residual_tanh,
                )
                if args.condition_residual_guard == "rms_clip":
                    guarded_residual, residual_guard_tensor = condition_residual_rms_guard(
                        base_cond,
                        ungated_pred_cond - base_cond,
                        max_rms_ratio=args.condition_residual_max_rms_ratio,
                        granularity=args.condition_residual_guard_granularity,
                        min_gate=args.condition_residual_min_gate,
                    )
                    ungated_pred_cond = base_cond + guarded_residual
                else:
                    residual_guard_tensor = stage3.new_ones((stage3.shape[0], 1, 1, 1))
                if condition_gate is None:
                    pred_cond = ungated_pred_cond
                    condition_gate_tensor = stage3.new_zeros((stage3.shape[0], 1, 1, 1))
                else:
                    condition_gate_tensor = condition_gate(
                        stage3,
                        semantic,
                        residual,
                        semantic_latent,
                        condition_size=condition_size,
                        base_condition=base_cond,
                        condition_residual=ungated_pred_cond - base_cond,
                        detail_context=detail_context_tensor,
                    )
                    pred_cond = base_cond + condition_gate_tensor * (ungated_pred_cond - base_cond)
                pre_control_pred_cond = pred_cond
                precomputed_stage4_raw: torch.Tensor | None = None
                control_hybrid_labels = ["none" for _ in range(stage3.shape[0])]
                control_hybrid_mode_index = torch.zeros((stage3.shape[0],), device=stage3.device, dtype=torch.float32)
                control_hybrid_condition_l1 = torch.zeros(
                    (stage3.shape[0],),
                    device=stage3.device,
                    dtype=torch.float32,
                )
                control_hybrid_rdo_score = torch.zeros(
                    (stage3.shape[0],),
                    device=stage3.device,
                    dtype=torch.float32,
                )
                if args.counted_control_mode == "condition_residual_grid":
                    control_correction, control_payload_bytes, control_grid_abs_mean = grouped_condition_residual_control(
                        target_cond.float() - pred_cond.float(),
                        groups=args.control_groups,
                        grid_size=args.control_grid_size,
                        codec=basis_control_codec,
                        scale=args.control_scale,
                    )
                    pred_cond = pred_cond + control_correction.to(dtype=pred_cond.dtype)
                elif args.counted_control_mode == "condition_residual_dct":
                    control_correction, control_payload_bytes, control_grid_abs_mean = grouped_condition_dct_control(
                        target_cond.float() - pred_cond.float(),
                        groups=args.control_groups,
                        grid_size=args.control_grid_size,
                        coeffs_per_group=args.control_dct_coeffs_per_group,
                        codec=control_codec,
                        scale=args.control_scale,
                    )
                    pred_cond = pred_cond + control_correction.to(dtype=pred_cond.dtype)
                elif args.counted_control_mode == "condition_residual_basis":
                    if control_basis_payload is None:
                        raise RuntimeError("control_basis_payload was not loaded")
                    control_correction, control_payload_bytes, control_grid_abs_mean = grouped_condition_basis_control(
                        target_cond.float() - pred_cond.float(),
                        basis_payload=control_basis_payload,
                        components=args.control_basis_components,
                        candidate_components=effective_candidate_components,
                        selection=args.control_basis_selection,
                        codec=basis_control_codec,
                        prefix_components=args.control_basis_prefix_components,
                        huffman=control_huffman,
                        sparse_huffman=control_sparse_huffman,
                        scale=args.control_scale,
                    )
                    pred_cond = pred_cond + control_correction.to(dtype=pred_cond.dtype)
                elif args.counted_control_mode == "condition_residual_affine":
                    control_correction, control_payload_bytes, control_grid_abs_mean = grouped_condition_affine_control(
                        base_cond.float(),
                        pred_cond.float(),
                        target_cond.float(),
                        groups=effective_affine_groups,
                        grid_size=effective_affine_grid_size,
                        gain_codec=affine_gain_codec,
                        bias_codec=affine_bias_codec,
                        scale=args.control_scale,
                    )
                    pred_cond = pred_cond + control_correction.to(dtype=pred_cond.dtype)
                elif args.counted_control_mode == "condition_residual_affine_dct":
                    control_correction, control_payload_bytes, control_grid_abs_mean = (
                        grouped_condition_affine_dct_control(
                            base_cond.float(),
                            pred_cond.float(),
                            target_cond.float(),
                            affine_groups=effective_affine_groups,
                            affine_grid_size=effective_affine_grid_size,
                            dct_groups=args.control_groups,
                            dct_grid_size=args.control_grid_size,
                            dct_coeffs_per_group=args.control_dct_coeffs_per_group,
                            gain_codec=affine_gain_codec,
                            bias_codec=affine_bias_codec,
                            dct_codec=control_codec,
                            scale=args.control_scale,
                        )
                    )
                    pred_cond = pred_cond + control_correction.to(dtype=pred_cond.dtype)
                elif args.counted_control_mode == "condition_residual_affine_grid":
                    control_correction, control_payload_bytes, control_grid_abs_mean = (
                        grouped_condition_affine_grid_control(
                            base_cond.float(),
                            pred_cond.float(),
                            target_cond.float(),
                            affine_groups=effective_affine_groups,
                            affine_grid_size=effective_affine_grid_size,
                            grid_groups=args.control_groups,
                            grid_size=args.control_grid_size,
                            gain_codec=affine_gain_codec,
                            bias_codec=affine_bias_codec,
                            grid_codec=control_codec,
                            scale=args.control_scale,
                        )
                    )
                    pred_cond = pred_cond + control_correction.to(dtype=pred_cond.dtype)
                elif args.counted_control_mode == "condition_residual_affine_basis":
                    if control_basis_payload is None:
                        raise RuntimeError("control_basis_payload was not loaded")
                    control_correction, control_payload_bytes, control_grid_abs_mean = (
                        grouped_condition_affine_basis_control(
                            base_cond.float(),
                            pred_cond.float(),
                            target_cond.float(),
                            affine_groups=effective_affine_groups,
                            affine_grid_size=effective_affine_grid_size,
                            gain_codec=affine_gain_codec,
                            bias_codec=affine_bias_codec,
                            basis_payload=control_basis_payload,
                            components=args.control_basis_components,
                            candidate_components=effective_candidate_components,
                            selection=args.control_basis_selection,
                            basis_codec=basis_control_codec,
                            prefix_components=args.control_basis_prefix_components,
                            huffman=control_huffman,
                            sparse_huffman=control_sparse_huffman,
                            scale=args.control_scale,
                        )
                    )
                    pred_cond = pred_cond + control_correction.to(dtype=pred_cond.dtype)
                elif args.counted_control_mode == "condition_residual_hybrid_affine_dct_grid":
                    batch_items = int(stage3.shape[0])
                    zero_correction = torch.zeros_like(pred_cond, dtype=torch.float32)
                    zero_bytes = [0 for _ in range(batch_items)]
                    zero_abs = stage3.new_zeros((batch_items,), dtype=torch.float32)
                    affine_correction, affine_bytes, affine_abs = grouped_condition_affine_control(
                        base_cond.float(),
                        pred_cond.float(),
                        target_cond.float(),
                        groups=effective_affine_groups,
                        grid_size=effective_affine_grid_size,
                        gain_codec=affine_gain_codec,
                        bias_codec=affine_bias_codec,
                        scale=args.control_scale,
                    )
                    affine_dct_correction, affine_dct_bytes, affine_dct_abs = (
                        grouped_condition_affine_dct_control(
                            base_cond.float(),
                            pred_cond.float(),
                            target_cond.float(),
                            affine_groups=effective_affine_groups,
                            affine_grid_size=effective_affine_grid_size,
                            dct_groups=args.control_groups,
                            dct_grid_size=args.control_grid_size,
                            dct_coeffs_per_group=args.control_dct_coeffs_per_group,
                            gain_codec=affine_gain_codec,
                            bias_codec=affine_bias_codec,
                            dct_codec=control_codec,
                            scale=args.control_scale,
                        )
                    )
                    affine_grid_correction, affine_grid_bytes, affine_grid_abs = (
                        grouped_condition_affine_grid_control(
                            base_cond.float(),
                            pred_cond.float(),
                            target_cond.float(),
                            affine_groups=effective_affine_groups,
                            affine_grid_size=effective_affine_grid_size,
                            grid_groups=args.control_groups,
                            grid_size=args.control_grid_size,
                            gain_codec=affine_gain_codec,
                            bias_codec=affine_bias_codec,
                            grid_codec=control_codec,
                            scale=args.control_scale,
                        )
                    )
                    affine_bits = affine_control_payload_bits(
                        groups=effective_affine_groups,
                        grid_size=effective_affine_grid_size,
                        gain_codec=affine_gain_codec,
                        bias_codec=affine_bias_codec,
                    )
                    dct_bits = affine_bits + int(
                        control_codec.encoded_num_bits((args.control_groups, args.control_dct_coeffs_per_group))
                    )
                    grid_bits = affine_bits + int(
                        control_codec.encoded_num_bits((args.control_groups, args.control_grid_size, args.control_grid_size))
                    )
                    hybrid_candidates = [
                        ("none", zero_correction, zero_bytes, zero_abs, repeated_payload_bits(batch_items, 0)),
                        ("affine", affine_correction, affine_bytes, affine_abs, repeated_payload_bits(batch_items, affine_bits)),
                        (
                            "affine_dct",
                            affine_dct_correction,
                            affine_dct_bytes,
                            affine_dct_abs,
                            repeated_payload_bits(batch_items, dct_bits),
                        ),
                        (
                            "affine_grid",
                            affine_grid_correction,
                            affine_grid_bytes,
                            affine_grid_abs,
                            repeated_payload_bits(batch_items, grid_bits),
                        ),
                    ]
                    if args.control_hybrid_selection_objective == "condition_l1":
                        (
                            control_correction,
                            control_payload_bytes,
                            control_grid_abs_mean,
                            control_hybrid_labels,
                            control_hybrid_mode_index,
                            control_hybrid_condition_l1,
                        ) = select_condition_control_by_rd_proxy(
                            pred_cond.float(),
                            target_cond.float(),
                            hybrid_candidates,
                            height=int(stage3.shape[-2]),
                            width=int(stage3.shape[-1]),
                            selector_bytes=args.control_hybrid_selector_bytes,
                            rate_lambda=args.control_hybrid_rate_lambda,
                            selector_bits=args.control_hybrid_selector_bits,
                        )
                        control_hybrid_rdo_score = control_hybrid_condition_l1
                        pred_cond = pred_cond + control_correction.to(dtype=pred_cond.dtype)
                    else:
                        (
                            selected_condition,
                            precomputed_stage4_raw,
                            control_payload_bytes,
                            control_grid_abs_mean,
                            control_hybrid_labels,
                            control_hybrid_mode_index,
                            control_hybrid_condition_l1,
                            control_hybrid_rdo_score,
                        ) = select_condition_control_by_image_rdo(
                            reference.float(),
                            stage3.float(),
                            pred_cond.float(),
                            target_cond.float(),
                            hybrid_candidates,
                            backbone=backbone,
                            selector_bytes=args.control_hybrid_selector_bytes,
                            rate_lambda=args.control_hybrid_rate_lambda,
                            objective=args.control_hybrid_selection_objective,
                            blend_alpha=args.blend_alpha,
                            perceptual=perceptual,
                            selector_bits=args.control_hybrid_selector_bits,
                            fidelity_lambda=args.control_hybrid_fidelity_lambda,
                            fidelity_metric=args.control_hybrid_fidelity_metric,
                        )
                        pred_cond = selected_condition.to(dtype=pred_cond.dtype)
                elif args.counted_control_mode == "condition_residual_hybrid_affine_dct_grid_basis":
                    if control_basis_payload is None:
                        raise RuntimeError("control_basis_payload was not loaded")
                    batch_items = int(stage3.shape[0])
                    zero_correction = torch.zeros_like(pred_cond, dtype=torch.float32)
                    zero_bytes = [0 for _ in range(batch_items)]
                    zero_abs = stage3.new_zeros((batch_items,), dtype=torch.float32)
                    affine_correction, affine_bytes, affine_abs = grouped_condition_affine_control(
                        base_cond.float(),
                        pred_cond.float(),
                        target_cond.float(),
                        groups=effective_affine_groups,
                        grid_size=effective_affine_grid_size,
                        gain_codec=affine_gain_codec,
                        bias_codec=affine_bias_codec,
                        scale=args.control_scale,
                    )
                    affine_dct_correction, affine_dct_bytes, affine_dct_abs = (
                        grouped_condition_affine_dct_control(
                            base_cond.float(),
                            pred_cond.float(),
                            target_cond.float(),
                            affine_groups=effective_affine_groups,
                            affine_grid_size=effective_affine_grid_size,
                            dct_groups=args.control_groups,
                            dct_grid_size=args.control_grid_size,
                            dct_coeffs_per_group=args.control_dct_coeffs_per_group,
                            gain_codec=affine_gain_codec,
                            bias_codec=affine_bias_codec,
                            dct_codec=control_codec,
                            scale=args.control_scale,
                        )
                    )
                    affine_grid_correction, affine_grid_bytes, affine_grid_abs = (
                        grouped_condition_affine_grid_control(
                            base_cond.float(),
                            pred_cond.float(),
                            target_cond.float(),
                            affine_groups=effective_affine_groups,
                            affine_grid_size=effective_affine_grid_size,
                            grid_groups=args.control_groups,
                            grid_size=args.control_grid_size,
                            gain_codec=affine_gain_codec,
                            bias_codec=affine_bias_codec,
                            grid_codec=control_codec,
                            scale=args.control_scale,
                        )
                    )
                    affine_basis_correction, affine_basis_bytes, affine_basis_abs = (
                        grouped_condition_affine_basis_control(
                            base_cond.float(),
                            pred_cond.float(),
                            target_cond.float(),
                            affine_groups=effective_affine_groups,
                            affine_grid_size=effective_affine_grid_size,
                            gain_codec=affine_gain_codec,
                            bias_codec=affine_bias_codec,
                            basis_payload=control_basis_payload,
                            components=args.control_basis_components,
                            candidate_components=effective_candidate_components,
                            selection=args.control_basis_selection,
                            basis_codec=basis_control_codec,
                            prefix_components=args.control_basis_prefix_components,
                            huffman=control_huffman,
                            sparse_huffman=control_sparse_huffman,
                            scale=args.control_scale,
                        )
                    )
                    affine_bits = affine_control_payload_bits(
                        groups=effective_affine_groups,
                        grid_size=effective_affine_grid_size,
                        gain_codec=affine_gain_codec,
                        bias_codec=affine_bias_codec,
                    )
                    dct_bits = affine_bits + int(
                        control_codec.encoded_num_bits((args.control_groups, args.control_dct_coeffs_per_group))
                    )
                    grid_bits = affine_bits + int(
                        control_codec.encoded_num_bits((args.control_groups, args.control_grid_size, args.control_grid_size))
                    )
                    basis_bits: int | None = None
                    if args.control_codec != "huffman":
                        basis_bits = basis_control_payload_bits(
                            selection=args.control_basis_selection,
                            components=args.control_basis_components,
                            candidate_components=effective_candidate_components,
                            basis_codec=basis_control_codec,
                            prefix_components=args.control_basis_prefix_components,
                        )
                    affine_basis_bits = None if basis_bits is None else affine_bits + basis_bits
                    hybrid_candidates = [
                        ("none", zero_correction, zero_bytes, zero_abs, repeated_payload_bits(batch_items, 0)),
                        ("affine", affine_correction, affine_bytes, affine_abs, repeated_payload_bits(batch_items, affine_bits)),
                        (
                            "affine_dct",
                            affine_dct_correction,
                            affine_dct_bytes,
                            affine_dct_abs,
                            repeated_payload_bits(batch_items, dct_bits),
                        ),
                        (
                            "affine_grid",
                            affine_grid_correction,
                            affine_grid_bytes,
                            affine_grid_abs,
                            repeated_payload_bits(batch_items, grid_bits),
                        ),
                        (
                            "affine_basis",
                            affine_basis_correction,
                            affine_basis_bytes,
                            affine_basis_abs,
                            repeated_payload_bits(batch_items, affine_basis_bits),
                        ),
                    ]
                    if args.control_hybrid_selection_objective == "condition_l1":
                        (
                            control_correction,
                            control_payload_bytes,
                            control_grid_abs_mean,
                            control_hybrid_labels,
                            control_hybrid_mode_index,
                            control_hybrid_condition_l1,
                        ) = select_condition_control_by_rd_proxy(
                            pred_cond.float(),
                            target_cond.float(),
                            hybrid_candidates,
                            height=int(stage3.shape[-2]),
                            width=int(stage3.shape[-1]),
                            selector_bytes=args.control_hybrid_selector_bytes,
                            rate_lambda=args.control_hybrid_rate_lambda,
                            selector_bits=args.control_hybrid_selector_bits,
                        )
                        control_hybrid_rdo_score = control_hybrid_condition_l1
                        pred_cond = pred_cond + control_correction.to(dtype=pred_cond.dtype)
                    else:
                        (
                            selected_condition,
                            precomputed_stage4_raw,
                            control_payload_bytes,
                            control_grid_abs_mean,
                            control_hybrid_labels,
                            control_hybrid_mode_index,
                            control_hybrid_condition_l1,
                            control_hybrid_rdo_score,
                        ) = select_condition_control_by_image_rdo(
                            reference.float(),
                            stage3.float(),
                            pred_cond.float(),
                            target_cond.float(),
                            hybrid_candidates,
                            backbone=backbone,
                            selector_bytes=args.control_hybrid_selector_bytes,
                            rate_lambda=args.control_hybrid_rate_lambda,
                            objective=args.control_hybrid_selection_objective,
                            blend_alpha=args.blend_alpha,
                            perceptual=perceptual,
                            selector_bits=args.control_hybrid_selector_bits,
                            fidelity_lambda=args.control_hybrid_fidelity_lambda,
                            fidelity_metric=args.control_hybrid_fidelity_metric,
                        )
                        pred_cond = selected_condition.to(dtype=pred_cond.dtype)
                else:
                    control_payload_bytes = [0 for _ in range(stage3.shape[0])]
                    control_grid_abs_mean = stage3.new_zeros((stage3.shape[0],), dtype=torch.float32)
                if post_control_condition_gate is None:
                    post_control_condition_gate_tensor = stage3.new_zeros((stage3.shape[0], 1, 1, 1))
                else:
                    post_control_condition_gate_tensor = post_control_condition_gate(
                        stage3,
                        semantic,
                        residual,
                        semantic_latent,
                        condition_size=condition_size,
                        base_condition=base_cond,
                        condition_residual=pred_cond - base_cond,
                        detail_context=detail_context_tensor,
                    )
                    pred_cond = base_cond + post_control_condition_gate_tensor * (pred_cond - base_cond)
                    precomputed_stage4_raw = None
                stage4_raw = precomputed_stage4_raw if precomputed_stage4_raw is not None else backbone(stage3, pred_cond)
                if condition_gate is not None:
                    alpha = stage3.new_ones((stage3.shape[0], 1, 1, 1))
                    stage4 = stage4_raw.clamp(0, 1)
                elif gate is None:
                    alpha = stage3.new_full((stage3.shape[0], 1, 1, 1), float(args.blend_alpha))
                else:
                    alpha = gate(
                        stage3,
                        semantic,
                        residual,
                        semantic_latent,
                        condition_size=condition_size,
                        base_condition=base_cond,
                        condition_residual=pred_cond - base_cond,
                        detail_context=detail_context_tensor,
                    )
                if condition_gate is None:
                    stage4 = ((1.0 - alpha) * stage3 + alpha * stage4_raw).clamp(0, 1)
            stage4_alpha = alpha.float().flatten(1).mean(dim=1)
            condition_l1 = torch.mean(torch.abs((pred_cond - target_cond).float()), dim=(1, 2, 3))
            pre_control_condition_l1 = torch.mean(
                torch.abs((pre_control_pred_cond - target_cond).float()),
                dim=(1, 2, 3),
            )
            ungated_condition_l1 = torch.mean(torch.abs((ungated_pred_cond - target_cond).float()), dim=(1, 2, 3))
            base_condition_l1 = torch.mean(torch.abs((base_cond - target_cond).float()), dim=(1, 2, 3))
            condition_residual_l1 = torch.mean(torch.abs((pred_cond - base_cond).float()), dim=(1, 2, 3))
            condition_delta_raw_l1 = torch.mean(torch.abs(cond_delta.float()), dim=(1, 2, 3))
            residual_guard_flat = residual_guard_tensor.float().flatten(1)
            residual_guard_mean = residual_guard_flat.mean(dim=1)
            residual_guard_min = residual_guard_flat.min(dim=1).values
            residual_guard_max = residual_guard_flat.max(dim=1).values
            condition_gate_mean = condition_gate_tensor.float().flatten(1).mean(dim=1)
            post_control_condition_gate_mean = post_control_condition_gate_tensor.float().flatten(1).mean(dim=1)

            stage3_psnr = psnr(reference.float(), stage3.float())
            stage4_psnr = psnr(reference.float(), stage4.float())
            stage3_msssim = ms_ssim(stage3.float(), reference.float(), data_range=1.0, size_average=False)
            stage4_msssim = ms_ssim(stage4.float(), reference.float(), data_range=1.0, size_average=False)
            stage3_l1 = torch.mean(torch.abs(stage3.float() - reference.float()), dim=(1, 2, 3))
            stage4_l1 = torch.mean(torch.abs(stage4.float() - reference.float()), dim=(1, 2, 3))

            for item in range(reference.shape[0]):
                ref_i = reference[item : item + 1]
                s3_i = stage3[item : item + 1]
                s4_i = stage4[item : item + 1]
                p3 = perceptual(ref_i, s3_i)
                p4 = perceptual(ref_i, s4_i)
                stage3_payload_bpp = float(batch["actual_payload_bpp"][item])
                control_payload_bpp = CoSERBitstream.bytes_to_bpp(
                    int(control_payload_bytes[item]),
                    int(reference.shape[-2]),
                    int(reference.shape[-1]),
                )
                actual_payload_bpp = stage3_payload_bpp + control_payload_bpp
                row = {
                    "index": int(batch["index"][item]),
                    "source_path": str(batch["source_path"][item]),
                    "actual_payload_bpp": actual_payload_bpp,
                    "paper_bpp": actual_payload_bpp,
                    "stage3_actual_payload_bpp": stage3_payload_bpp,
                    "control_payload_bpp": control_payload_bpp,
                    "control_payload_bytes": float(control_payload_bytes[item]),
                    "stage3_psnr": float(stage3_psnr[item].item()),
                    "stage4_psnr": float(stage4_psnr[item].item()),
                    "stage4_psnr_delta_vs_stage3": float(stage4_psnr[item].item() - stage3_psnr[item].item()),
                    "stage3_ms_ssim": float(stage3_msssim[item].item()),
                    "stage4_ms_ssim": float(stage4_msssim[item].item()),
                    "stage4_ms_ssim_delta_vs_stage3": float(stage4_msssim[item].item() - stage3_msssim[item].item()),
                    "stage3_l1": float(stage3_l1[item].item()),
                    "stage4_l1": float(stage4_l1[item].item()),
                    "stage4_l1_delta_vs_stage3": float(stage4_l1[item].item() - stage3_l1[item].item()),
                    "stage3_lpips_alex": p3.lpips_alex,
                    "stage4_lpips_alex": p4.lpips_alex,
                    "stage4_lpips_alex_delta_vs_stage3": p4.lpips_alex - p3.lpips_alex,
                    "stage3_dists": p3.dists,
                    "stage4_dists": p4.dists,
                    "stage4_dists_delta_vs_stage3": p4.dists - p3.dists,
                    "condition_l1": float(condition_l1[item].item()),
                    "pre_control_condition_l1": float(pre_control_condition_l1[item].item()),
                    "control_condition_l1_delta": float(
                        condition_l1[item].item() - pre_control_condition_l1[item].item()
                    ),
                    "ungated_condition_l1": float(ungated_condition_l1[item].item()),
                    "base_condition_l1": float(base_condition_l1[item].item()),
                    "condition_l1_delta_vs_base": float(condition_l1[item].item() - base_condition_l1[item].item()),
                    "condition_residual_l1": float(condition_residual_l1[item].item()),
                    "condition_delta_raw_l1": float(condition_delta_raw_l1[item].item()),
                    "condition_residual_guard_mean": float(residual_guard_mean[item].item()),
                    "condition_residual_guard_min": float(residual_guard_min[item].item()),
                    "condition_residual_guard_max": float(residual_guard_max[item].item()),
                    "control_grid_abs_mean": float(control_grid_abs_mean[item].item()),
                    "control_hybrid_mode": control_hybrid_labels[item],
                    "control_hybrid_mode_index": float(control_hybrid_mode_index[item].item()),
                    "control_hybrid_condition_l1": float(control_hybrid_condition_l1[item].item()),
                    "control_hybrid_rdo_score": float(control_hybrid_rdo_score[item].item()),
                    "stage4_alpha": float(stage4_alpha[item].item()),
                    "condition_gate_mean": float(condition_gate_mean[item].item()),
                    "post_control_condition_gate_mean": float(post_control_condition_gate_mean[item].item()),
                }
                per_image.append(row)
                for key in metrics:
                    metrics[key].append(float(row[key]))
                wandb_run.log({key: value for key, value in row.items() if isinstance(value, float)})

                global_index = len(per_image) - 1
                if args.save_reconstructions and (
                    args.save_reconstruction_limit <= 0 or global_index < args.save_reconstruction_limit
                ):
                    image_name = f"image{global_index:05d}.png"
                    if "reference" in save_kinds:
                        save_image(ref_i.detach().cpu(), recon_dir / "reference" / image_name)
                    if "semantic_only" in save_kinds:
                        save_image(semantic[item : item + 1].detach().cpu(), recon_dir / "semantic_only" / image_name)
                    if "stage3" in save_kinds:
                        save_image(s3_i.detach().cpu(), recon_dir / "stage3" / image_name)
                    if "stage4" in save_kinds:
                        save_image(s4_i.detach().cpu(), recon_dir / "stage4" / image_name)
                    if "quad" in save_kinds:
                        save_image(
                            torch.cat(
                                [
                                    ref_i.detach().cpu(),
                                    semantic[item : item + 1].detach().cpu(),
                                    s3_i.detach().cpu(),
                                    s4_i.detach().cpu(),
                                ],
                                dim=0,
                            ),
                            recon_dir / "quad" / image_name,
                            nrow=4,
                        )

    summary = {f"{key}_mean": mean(values) for key, values in metrics.items()}
    hybrid_mode_counts: dict[str, int] = {}
    for row in per_image:
        mode = str(row.get("control_hybrid_mode", "none"))
        hybrid_mode_counts[mode] = hybrid_mode_counts.get(mode, 0) + 1
    summary.update(
        {
            "count": len(per_image),
            "stage4_psnr_win_rate": mean([1.0 if r["stage4_psnr_delta_vs_stage3"] > 0 else 0.0 for r in per_image]),
            "stage4_ms_ssim_win_rate": mean(
                [1.0 if r["stage4_ms_ssim_delta_vs_stage3"] > 0 else 0.0 for r in per_image]
            ),
            "stage4_lpips_win_rate": mean(
                [1.0 if r["stage4_lpips_alex_delta_vs_stage3"] < 0 else 0.0 for r in per_image]
            ),
            "stage4_dists_win_rate": mean([1.0 if r["stage4_dists_delta_vs_stage3"] < 0 else 0.0 for r in per_image]),
            "stage4_blend_alpha": args.blend_alpha,
            "stage4_alpha_min": min(metrics["stage4_alpha"]) if metrics["stage4_alpha"] else 0.0,
            "stage4_alpha_max": max(metrics["stage4_alpha"]) if metrics["stage4_alpha"] else 0.0,
            "stage4_alpha_std": float(torch.tensor(metrics["stage4_alpha"]).std(unbiased=False).item())
            if metrics["stage4_alpha"]
            else 0.0,
            "stage4_gate_checkpoint": args.gate_checkpoint,
            "stage4_condition_gate_checkpoint": args.condition_gate_checkpoint,
            "stage4_post_control_condition_gate_checkpoint": args.post_control_condition_gate_checkpoint,
            "condition_residual_scale": residual_scale,
            "condition_residual_tanh": residual_tanh,
            "condition_residual_guard": args.condition_residual_guard,
            "condition_residual_guard_granularity": args.condition_residual_guard_granularity,
            "condition_residual_max_rms_ratio": args.condition_residual_max_rms_ratio,
            "condition_residual_min_gate": args.condition_residual_min_gate,
            "condition_delta_ablation": args.condition_delta_ablation,
            "counted_control_mode": args.counted_control_mode,
            "control_grid_size": args.control_grid_size,
            "control_groups": args.control_groups,
            "control_dct_coeffs_per_group": args.control_dct_coeffs_per_group,
            "control_affine_groups": effective_affine_groups,
            "control_affine_grid_size": effective_affine_grid_size,
            "control_basis": args.control_basis,
            "control_basis_components": args.control_basis_components,
            "control_basis_candidate_components": effective_candidate_components,
            "control_basis_selection": args.control_basis_selection,
            "control_basis_range_mode": effective_control_basis_range_mode,
            "control_basis_range_floor": args.control_basis_range_floor,
            "control_basis_source": str(control_basis_payload.get("source", "")) if control_basis_payload else "",
            "control_basis_explained_variance": control_basis_payload.get("explained_variance", [])
            if control_basis_payload
            else [],
            "control_basis_cumulative_explained_variance": control_basis_payload.get(
                "cumulative_explained_variance", []
            )
            if control_basis_payload
            else [],
            "control_basis_coefficient_abs_quantiles": control_basis_payload.get("coefficient_abs_quantiles", {})
            if control_basis_payload
            else {},
            "control_basis_coefficient_abs_mean": control_basis_payload.get("coefficient_abs_mean", 0.0)
            if control_basis_payload
            else 0.0,
            "control_basis_coefficient_abs_std": control_basis_payload.get("coefficient_abs_std", 0.0)
            if control_basis_payload
            else 0.0,
            "control_basis_coefficient_abs_max": control_basis_payload.get("coefficient_abs_max", 0.0)
            if control_basis_payload
            else 0.0,
            "control_basis_component_abs_p95": control_basis_payload.get("coefficient_component_abs_p95", [])
            if control_basis_payload
            else [],
            "control_basis_component_abs_p99": control_basis_payload.get("coefficient_component_abs_p99", [])
            if control_basis_payload
            else [],
            "control_codec_type": args.control_codec,
            "control_huffman_key": effective_control_huffman_key,
            "control_quantizer": effective_control_quantizer,
            "control_mu": effective_control_mu,
            "control_bits": effective_control_bits,
            "control_range": effective_control_range,
            "control_affine_gain_range": args.control_affine_gain_range,
            "control_affine_bias_range": args.control_affine_bias_range,
            "control_hybrid_selector_bytes": args.control_hybrid_selector_bytes,
            "control_hybrid_selector_bits": args.control_hybrid_selector_bits,
            "control_hybrid_rate_lambda": args.control_hybrid_rate_lambda,
            "control_hybrid_selection_objective": args.control_hybrid_selection_objective,
            "control_hybrid_fidelity_lambda": args.control_hybrid_fidelity_lambda,
            "control_hybrid_fidelity_metric": args.control_hybrid_fidelity_metric,
            "control_hybrid_mode_counts": hybrid_mode_counts,
            "control_scale": args.control_scale,
            "control_codec": control_codec.to_dict(),
            "control_basis_codec": basis_control_codec.to_dict(),
            "control_affine_gain_codec": affine_gain_codec.to_dict(),
            "control_affine_bias_codec": affine_bias_codec.to_dict(),
            "semantic_latent_ablation": args.semantic_latent_ablation,
            "detail_context_ablation": args.detail_context_ablation,
            "ablation_shuffle_seed": args.ablation_shuffle_seed,
            "save_reconstruction_kinds": list(save_kinds),
            "stage4_payload_policy": (
                "actual_payload_bpp = Stage 3 semantic/detail payload plus any counted control stream payload; "
                "fixed CoD-Lite weights, adapter, deterministic decoder-side RGB diagnostic blend, or deterministic "
                "condition gate are not image-specific side information"
            ),
        }
    )
    summary_path = out_dir / "summary.json"
    per_image_path = out_dir / "per_image_metrics.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    per_image_path.write_text("\n".join(json.dumps(row, allow_nan=False) for row in per_image) + "\n")
    if args.save_reconstructions:
        saved_limit = len(per_image) if args.save_reconstruction_limit <= 0 else min(args.save_reconstruction_limit, len(per_image))
        manifest_rows = []
        for global_index, row in enumerate(per_image[:saved_limit]):
            image_name = f"image{global_index:05d}.png"
            manifest_row: dict[str, object] = {
                "index": int(row["index"]),
                "source_path": str(row["source_path"]),
            }
            for kind in save_kinds:
                if kind == "quad":
                    continue
                manifest_row[kind] = str(recon_dir / kind / image_name)
            manifest_rows.append(manifest_row)
        (recon_dir / "manifest.jsonl").write_text(
            "\n".join(json.dumps(row, allow_nan=False) for row in manifest_rows) + "\n"
        )

    write_run_doc(
        Path("docs/experiments") / f"{run_name}.md",
        {
            "run_name": run_name,
            "date": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "summary": summary,
            "artifacts": {
                "summary": str(summary_path),
                "per_image": str(per_image_path),
                "reconstructions": str(recon_dir) if args.save_reconstructions else "",
                "wandb": str(Path(wandb_run.dir).parent if wandb_run is not None else ""),
            },
        },
    )
    wandb_run.finish()


if __name__ == "__main__":
    main()
