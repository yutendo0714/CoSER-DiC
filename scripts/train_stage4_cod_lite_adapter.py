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

from coserdic.models import (
    CoDLiteOneStepBackbone,
    CoDLiteOneStepBackboneConfig,
    CoSERToCoDLiteConditionAdapter,
    CoSERToCoDLiteConditionAdapterConfig,
    CoSERToCoDLiteConditionPyramidAdapter,
    CoSERToCoDLiteConditionPyramidAdapterConfig,
    apply_lora_adapters_by_name,
    apply_lora_adapters_from_config,
    lora_parameter_names,
    load_named_parameter_state,
    named_parameter_state,
    set_trainable_parameters_by_name,
)
from coserdic.utils import seed_everything
from coserdic.utils.wandb_utils import init_wandb
from eval_stage4_cod_lite_adapter import (
    basis_component_codebook_codec as control_basis_component_codebook_codec,
    basis_component_ranges as control_basis_component_ranges,
    basis_vector_codebook_codec as control_basis_vector_codebook_codec,
    build_control_grid_code as control_build_control_grid_code,
    channel_group_sizes as control_channel_group_sizes,
    grouped_condition_affine_basis_control as control_grouped_condition_affine_basis_control,
    load_control_basis as control_load_control_basis,
    load_control_huffman_prior as control_load_control_huffman_prior,
    load_sparse_topk_control_huffman_prior as control_load_sparse_topk_control_huffman_prior,
    load_vector_codebook_huffman_prior as control_load_vector_codebook_huffman_prior,
)


def detail_context_channels(mode: str) -> int:
    return {"none": 0, "residual_grid": 3, "residual_grid_codes": 6}[mode]


class ReconstructionManifestDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        limit: int = 0,
        crop_size: int = 0,
        semantic_channels: int = 3,
        detail_context: str = "none",
    ) -> None:
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if limit:
            rows = rows[:limit]
        if not rows:
            raise ValueError(f"empty manifest: {manifest}")
        self.rows = rows
        self.crop_size = crop_size
        self.semantic_channels = int(semantic_channels)
        self.detail_context = detail_context

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.rows[index]
        reference = _load_image(Path(row["reference"]), self.crop_size)
        semantic = _load_image(Path(row["semantic_only"]), self.crop_size)
        stage3 = _load_image(Path(row["stage3"]), self.crop_size)
        semantic_latent = semantic
        cache_path = str(row.get("decoder_feature_cache", ""))
        has_decoder_feature_cache = bool(cache_path)
        detail_context = torch.empty(0, 1, 1)
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
            semantic_latent = cache["semantic_latent"].float()
            if semantic_latent.ndim == 4 and semantic_latent.shape[0] == 1:
                semantic_latent = semantic_latent.squeeze(0)
        if self.detail_context != "none":
            detail_context = _load_detail_context(cache, self.detail_context)
        return {
            "reference": reference,
            "semantic": semantic,
            "semantic_latent": semantic_latent,
            "detail_context": detail_context,
            "has_decoder_feature_cache": has_decoder_feature_cache,
            "stage3": stage3,
            "index": int(row.get("index", index)),
            "source_path": str(row.get("source_path", "")),
        }


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
        detail_bits = int(cache.get("detail_bits", 4))
        detail_levels = int(cache.get("detail_levels", 1 << detail_bits))
        if detail_levels <= 1:
            raise ValueError(f"invalid detail_levels in decoder feature cache: {detail_levels}")
        detail_quantizer = str(cache.get("detail_quantizer", "uniform"))
        if detail_quantizer == "zero_centered":
            zero_code = detail_levels // 2
            positive_denominator = max(detail_levels - 1 - zero_code, 1)
            signed = detail_codes - float(zero_code)
            detail_codes = torch.where(
                signed >= 0,
                signed / float(positive_denominator),
                signed / float(zero_code),
            )
        elif detail_quantizer == "uniform":
            detail_codes = detail_codes / float(detail_levels - 1) * 2.0 - 1.0
        else:
            raise ValueError(f"unknown detail_quantizer in decoder feature cache: {detail_quantizer}")
        return torch.cat([residual_grid, detail_codes], dim=0)
    raise ValueError(f"unknown detail_context: {mode}")


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((x - y).pow(2), dim=(1, 2, 3)).clamp_min(1.0e-12)
    return -10.0 * torch.log10(mse)


def condition_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(pred.float().flatten(1), target.float().flatten(1), dim=1, eps=1.0e-8)
    return 1.0 - cosine.mean()


def condition_channel_stats_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.float()
    target_f = target.float()
    pred_mean = pred_f.mean(dim=(2, 3))
    target_mean = target_f.mean(dim=(2, 3))
    pred_std = pred_f.std(dim=(2, 3), unbiased=False)
    target_std = target_f.std(dim=(2, 3), unbiased=False)
    return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


def condition_highfreq_ratio(tensor: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
    x = tensor.float()
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.rfft2(x, norm="ortho")
    power = spectrum.abs().square()
    height = int(x.shape[-2])
    width = int(x.shape[-1])
    freq_y = torch.fft.fftfreq(height, device=x.device).abs().view(height, 1)
    freq_x = torch.fft.rfftfreq(width, device=x.device).abs().view(1, width // 2 + 1)
    radius = torch.sqrt(freq_y.square() + freq_x.square())
    high_mask = radius >= threshold
    total = power.sum(dim=(-2, -1)).clamp_min(1.0e-12)
    high = power[..., high_mask].sum(dim=-1)
    return high / total


def condition_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, *, threshold: float) -> torch.Tensor:
    return F.l1_loss(
        condition_highfreq_ratio(pred, threshold=threshold),
        condition_highfreq_ratio(target, threshold=threshold),
    )


def condition_residual_rms_ratio(
    base_condition: torch.Tensor,
    condition_residual: torch.Tensor,
    *,
    granularity: str = "global",
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if base_condition.shape != condition_residual.shape:
        raise ValueError("base_condition and condition_residual must share shape")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if granularity == "global":
        dims = (1, 2, 3)
    elif granularity == "spatial":
        dims = (1,)
    elif granularity == "channel":
        dims = (2, 3)
    else:
        raise ValueError(f"unknown condition residual RMS granularity: {granularity}")
    base = base_condition.float()
    residual = condition_residual.float()
    base_rms = torch.sqrt(torch.mean(base.pow(2), dim=dims, keepdim=True).clamp_min(eps * eps))
    residual_rms = torch.sqrt(torch.mean(residual.pow(2), dim=dims, keepdim=True).clamp_min(eps * eps))
    return residual_rms / base_rms.clamp_min(eps)


def condition_residual_rms_excess_loss(
    base_condition: torch.Tensor,
    condition_residual: torch.Tensor,
    *,
    max_rms_ratio: float,
    granularity: str = "global",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_rms_ratio <= 0:
        raise ValueError("max_rms_ratio must be positive")
    ratio = condition_residual_rms_ratio(
        base_condition,
        condition_residual,
        granularity=granularity,
    )
    excess = torch.relu(ratio - float(max_rms_ratio))
    return excess.square().mean(), ratio.mean(), ratio.max()


def condition_local_highpass(tensor: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    x = tensor.float()
    low = F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )
    return x - low


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


def build_adapter(
    *,
    adapter_kind: str,
    semantic_channels: int,
    detail_context_channels: int,
    condition_channels: int,
    hidden_channels: int,
    num_image_blocks: int,
    num_condition_blocks: int,
    num_detail_blocks: int,
    num_fusion_blocks: int,
    detail_control_branch: bool,
    detail_control_blocks: int,
    detail_control_condition_fusion: bool,
    detail_highfreq_context_branch: bool,
    detail_film_modulation: bool,
) -> CoSERToCoDLiteConditionAdapter | CoSERToCoDLiteConditionPyramidAdapter:
    if adapter_kind == "light":
        return CoSERToCoDLiteConditionAdapter(
            CoSERToCoDLiteConditionAdapterConfig(
                semantic_channels=semantic_channels,
                detail_context_channels=detail_context_channels,
                condition_channels=condition_channels,
                hidden_channels=hidden_channels,
                zero_init_output=True,
            )
        )
    if adapter_kind == "pyramid":
        return CoSERToCoDLiteConditionPyramidAdapter(
            CoSERToCoDLiteConditionPyramidAdapterConfig(
                semantic_channels=semantic_channels,
                detail_context_channels=detail_context_channels,
                condition_channels=condition_channels,
                hidden_channels=hidden_channels,
                num_image_blocks=num_image_blocks,
                num_condition_blocks=num_condition_blocks,
                num_detail_blocks=num_detail_blocks,
                num_fusion_blocks=num_fusion_blocks,
                detail_control_branch=detail_control_branch,
                detail_control_blocks=detail_control_blocks,
                detail_control_condition_fusion=detail_control_condition_fusion,
                detail_highfreq_context_branch=detail_highfreq_context_branch,
                detail_film_modulation=detail_film_modulation,
                zero_init_output=True,
            )
        )
    raise ValueError(f"unknown adapter_kind: {adapter_kind}")


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


def load_lora_plan_setting(path: Path, preset: str) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"LoRA plan must be a JSON object: {path}")
    settings = payload.get("settings")
    if not isinstance(settings, list):
        raise ValueError(f"LoRA plan must contain a settings list: {path}")
    matches = [
        setting
        for setting in settings
        if isinstance(setting, dict) and str(setting.get("preset", "")) == str(preset)
    ]
    if len(matches) != 1:
        raise ValueError(f"LoRA plan preset {preset!r} matched {len(matches)} settings in {path}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="checkpoints/stage4_cod_lite_adapter")
    parser.add_argument("--results-dir", default="results/stage4_cod_lite_adapter")
    parser.add_argument("--cod-lite-checkpoint", default="external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt")
    parser.add_argument("--cod-lite-config", default="external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml")
    parser.add_argument("--cod-lite-repo", default="external/repos/GenCodec/CoD_Lite")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--condition-l1-weight", type=float, default=1.0)
    parser.add_argument(
        "--pre-control-condition-l1-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary L1 loss on the adapter condition before counted-control correction. "
            "Useful for control-aware training so the adapter stays useful on its own and "
            "the transmitted control stream complements it instead of carrying the whole correction."
        ),
    )
    parser.add_argument("--condition-cosine-weight", type=float, default=0.0)
    parser.add_argument("--condition-channel-stats-weight", type=float, default=0.0)
    parser.add_argument("--condition-highfreq-weight", type=float, default=0.0)
    parser.add_argument("--condition-highfreq-threshold", type=float, default=0.25)
    parser.add_argument("--condition-residual-rms-guard-weight", type=float, default=0.0)
    parser.add_argument("--condition-residual-rms-guard-ratio", type=float, default=0.5)
    parser.add_argument(
        "--condition-residual-rms-guard-granularity",
        choices=("global", "spatial", "channel"),
        default="global",
    )
    parser.add_argument("--detail-contrast-weight", type=float, default=0.0)
    parser.add_argument("--detail-contrast-margin", type=float, default=0.0)
    parser.add_argument("--detail-highfreq-residual-weight", type=float, default=0.0)
    parser.add_argument("--detail-highfreq-kernel-size", type=int, default=5)
    parser.add_argument("--detail-residual-target-weight", type=float, default=0.0)
    parser.add_argument("--image-l1-weight", type=float, default=0.25)
    parser.add_argument("--lpips-weight", type=float, default=0.0)
    parser.add_argument("--dists-weight", type=float, default=0.0)
    parser.add_argument("--ms-ssim-weight", type=float, default=0.0)
    parser.add_argument("--stage3-l1-guard-weight", type=float, default=0.0)
    parser.add_argument("--stage3-mse-guard-weight", type=float, default=0.0)
    parser.add_argument("--stage3-guard-margin", type=float, default=0.0)
    parser.add_argument("--condition-residual-scale", type=float, default=1.0)
    parser.add_argument("--condition-residual-tanh", action="store_true")
    parser.add_argument(
        "--train-counted-control-mode",
        choices=("none", "condition_residual_affine_basis", "condition_base_affine_basis"),
        default="none",
        help=(
            "Use an actually entropy-coded control simulation inside adapter training. "
            "condition_residual_affine_basis codes the residual after the current adapter condition. "
            "condition_base_affine_basis codes a fixed teacher/base correction, so the adapter learns "
            "the complementary condition residual. Only decoded, byte-countable control corrections "
            "are used by the image/condition losses."
        ),
    )
    parser.add_argument("--control-grid-size", type=int, default=8)
    parser.add_argument("--control-groups", type=int, default=32)
    parser.add_argument("--control-basis", default="")
    parser.add_argument("--control-basis-components", type=int, default=16)
    parser.add_argument("--control-basis-candidate-components", type=int, default=0)
    parser.add_argument("--control-basis-prefix-components", type=int, default=0)
    parser.add_argument("--control-basis-selection", choices=("prefix", "topk", "vector", "prefix_topk"), default="topk")
    parser.add_argument(
        "--control-basis-range-mode",
        choices=("global", "component_p95", "component_p99", "component_codebook"),
        default="global",
    )
    parser.add_argument("--control-basis-range-floor", type=float, default=1.0e-6)
    parser.add_argument("--control-codec", choices=("fixed_bits", "huffman"), default="huffman")
    parser.add_argument("--control-huffman-key", default="")
    parser.add_argument("--control-prefix-huffman-key", default="")
    parser.add_argument("--control-quantizer", choices=("uniform", "mu_law"), default="mu_law")
    parser.add_argument("--control-mu", type=float, default=16.0)
    parser.add_argument("--control-bits", type=int, default=4)
    parser.add_argument("--control-range", type=float, default=0.25)
    parser.add_argument("--control-affine-groups", type=int, default=16)
    parser.add_argument("--control-affine-grid-size", type=int, default=1)
    parser.add_argument("--control-affine-gain-range", type=float, default=1.0)
    parser.add_argument("--control-affine-bias-range", type=float, default=0.25)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument(
        "--backbone-train-pattern",
        action="append",
        default=[],
        help=(
            "Regex for official CoD-Lite backbone parameter names to train. "
            "Repeat or comma-separate. All non-matching backbone params stay frozen."
        ),
    )
    parser.add_argument("--backbone-lr", type=float, default=0.0)
    parser.add_argument("--backbone-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--backbone-lora-pattern",
        action="append",
        default=[],
        help="Regex for official CoD-Lite Linear/Conv2d module names to wrap with LoRA adapters.",
    )
    parser.add_argument("--backbone-lora-plan", default="", help="LoRA target plan JSON from plan_stage5_lora_targets.py.")
    parser.add_argument("--backbone-lora-preset", default="", help="Preset key inside --backbone-lora-plan.")
    parser.add_argument("--backbone-lora-rank", type=int, default=4)
    parser.add_argument("--backbone-lora-alpha", type=float, default=4.0)
    parser.add_argument("--backbone-lora-lr", type=float, default=0.0)
    parser.add_argument("--backbone-lora-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--backbone-lora-target",
        action="append",
        default=[],
        help="LoRA target module type: linear and/or conv2d. Repeat or comma-separate.",
    )
    parser.add_argument("--base-condition", choices=("native_stage3", "zero"), default="native_stage3")
    parser.add_argument("--adapter-kind", choices=("light", "pyramid"), default="light")
    parser.add_argument("--semantic-channels", type=int, default=3)
    parser.add_argument("--detail-context", choices=("none", "residual_grid", "residual_grid_codes"), default="none")
    parser.add_argument("--semantic-latent-dropout-prob", type=float, default=0.0)
    parser.add_argument("--detail-context-dropout-prob", type=float, default=0.0)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-image-blocks", type=int, default=4)
    parser.add_argument("--num-condition-blocks", type=int, default=4)
    parser.add_argument("--num-detail-blocks", type=int, default=0)
    parser.add_argument("--num-fusion-blocks", type=int, default=4)
    parser.add_argument("--detail-control-branch", action="store_true")
    parser.add_argument("--detail-control-blocks", type=int, default=0)
    parser.add_argument("--detail-control-condition-fusion", action="store_true")
    parser.add_argument("--detail-highfreq-context-branch", action="store_true")
    parser.add_argument("--detail-film-modulation", action="store_true")
    parser.add_argument("--init-nonstrict", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--save-sample-every", type=int, default=100)
    args = parser.parse_args()

    lora_plan_setting: dict[str, object] = {}
    if args.backbone_lora_plan:
        if not args.backbone_lora_preset:
            raise ValueError("--backbone-lora-plan requires --backbone-lora-preset")
        lora_plan_setting = load_lora_plan_setting(Path(args.backbone_lora_plan), args.backbone_lora_preset)
        lora_regex = str(lora_plan_setting.get("regex", ""))
        if not lora_regex:
            raise ValueError(f"LoRA plan preset {args.backbone_lora_preset!r} has no regex")
        args.backbone_lora_pattern.append(lora_regex)
        if not args.backbone_lora_target and isinstance(lora_plan_setting.get("target_types"), list):
            args.backbone_lora_target.extend(str(item) for item in lora_plan_setting["target_types"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 4 training.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if not 0.0 <= args.semantic_latent_dropout_prob <= 1.0:
        raise ValueError("--semantic-latent-dropout-prob must be in [0, 1]")
    if not 0.0 <= args.detail_context_dropout_prob <= 1.0:
        raise ValueError("--detail-context-dropout-prob must be in [0, 1]")
    if args.detail_contrast_weight < 0.0:
        raise ValueError("--detail-contrast-weight must be >= 0")
    if args.detail_contrast_margin < 0.0:
        raise ValueError("--detail-contrast-margin must be >= 0")
    if args.detail_highfreq_residual_weight < 0.0:
        raise ValueError("--detail-highfreq-residual-weight must be >= 0")
    if args.detail_highfreq_kernel_size < 1 or args.detail_highfreq_kernel_size % 2 != 1:
        raise ValueError("--detail-highfreq-kernel-size must be a positive odd integer")
    if args.detail_residual_target_weight < 0.0:
        raise ValueError("--detail-residual-target-weight must be >= 0")
    if args.detail_control_blocks < 0:
        raise ValueError("--detail-control-blocks must be >= 0")
    if args.detail_control_blocks > 0 and not args.detail_control_branch:
        raise ValueError("--detail-control-blocks requires --detail-control-branch")
    if args.detail_control_condition_fusion and not args.detail_control_branch:
        raise ValueError("--detail-control-condition-fusion requires --detail-control-branch")
    if args.detail_highfreq_context_branch and detail_context_channels(args.detail_context) <= 0:
        raise ValueError("--detail-highfreq-context-branch requires --detail-context != none")
    if args.condition_residual_rms_guard_weight < 0.0:
        raise ValueError("--condition-residual-rms-guard-weight must be >= 0")
    if args.condition_residual_rms_guard_ratio <= 0.0:
        raise ValueError("--condition-residual-rms-guard-ratio must be positive")
    if args.pre_control_condition_l1_weight < 0.0:
        raise ValueError("--pre-control-condition-l1-weight must be >= 0")
    if args.train_counted_control_mode != "none":
        if not args.control_basis:
            raise ValueError("--train-counted-control-mode requires --control-basis")
        if args.control_grid_size <= 0:
            raise ValueError("--control-grid-size must be positive")
        if args.control_groups <= 0:
            raise ValueError("--control-groups must be positive")
        if args.control_basis_components <= 0:
            raise ValueError("--control-basis-components must be positive")
        if args.control_bits <= 0:
            raise ValueError("--control-bits must be positive")
        if args.control_range <= 0.0:
            raise ValueError("--control-range must be positive")
        if args.control_basis_range_floor <= 0.0:
            raise ValueError("--control-basis-range-floor must be positive")
        if args.control_affine_groups <= 0:
            raise ValueError("--control-affine-groups must be positive")
        if args.control_affine_grid_size <= 0:
            raise ValueError("--control-affine-grid-size must be positive")
        if args.control_affine_gain_range <= 0.0:
            raise ValueError("--control-affine-gain-range must be positive")
        if args.control_affine_bias_range <= 0.0:
            raise ValueError("--control-affine-bias-range must be positive")
    if args.backbone_train_pattern and args.backbone_lr <= 0.0:
        raise ValueError("--backbone-train-pattern requires --backbone-lr > 0")
    if args.backbone_lora_pattern and args.backbone_lora_lr <= 0.0:
        raise ValueError("--backbone-lora-pattern requires --backbone-lora-lr > 0")
    if args.backbone_lora_rank <= 0:
        raise ValueError("--backbone-lora-rank must be positive")
    if args.backbone_lora_alpha <= 0:
        raise ValueError("--backbone-lora-alpha must be positive")

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda")
    detail_channels = detail_context_channels(args.detail_context)
    if args.detail_contrast_weight > 0.0 and detail_channels <= 0:
        raise ValueError("--detail-contrast-weight requires --detail-context != none")
    if args.detail_highfreq_residual_weight > 0.0 and detail_channels <= 0:
        raise ValueError("--detail-highfreq-residual-weight requires --detail-context != none")
    if args.detail_residual_target_weight > 0.0 and detail_channels <= 0:
        raise ValueError("--detail-residual-target-weight requires --detail-context != none")
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_stage4_cod_lite_adapter"
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = ReconstructionManifestDataset(
        Path(args.manifest),
        limit=args.limit,
        crop_size=args.crop_size,
        semantic_channels=args.semantic_channels,
        detail_context=args.detail_context,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    backbone = CoDLiteOneStepBackbone.load(
        CoDLiteOneStepBackboneConfig(
            repo_root=args.cod_lite_repo,
            checkpoint_path=args.cod_lite_checkpoint,
            config_path=args.cod_lite_config,
            freeze=True,
        ),
        device=device,
    )
    backbone.eval()
    backbone_trainable_names = set_trainable_parameters_by_name(backbone.net, args.backbone_train_pattern)
    if args.backbone_train_pattern and not backbone_trainable_names:
        raise ValueError(
            "No CoD-Lite backbone parameters matched --backbone-train-pattern. "
            "Check the regex against backbone.net.named_parameters() after eval() cache conversion."
        )
    lora_target_args = args.backbone_lora_target or ["linear"]
    lora_targets = tuple(
        target.strip()
        for value in lora_target_args
        for target in str(value).split(",")
        if target.strip()
    )
    backbone_lora_module_names = apply_lora_adapters_by_name(
        backbone.net,
        args.backbone_lora_pattern,
        rank=args.backbone_lora_rank,
        alpha=args.backbone_lora_alpha,
        target_module_types=lora_targets,
    )
    backbone_lora_param_names = lora_parameter_names(backbone.net)
    if args.backbone_lora_pattern and not backbone_lora_module_names:
        raise ValueError(
            "No CoD-Lite backbone modules matched --backbone-lora-pattern. "
            "Check the regex against backbone.net.named_modules()."
        )
    parameter_lookup = dict(backbone.net.named_parameters())
    backbone_trainable_names = [
        name for name in backbone_trainable_names if name in parameter_lookup and parameter_lookup[name].requires_grad
    ]
    adapter = build_adapter(
        adapter_kind=args.adapter_kind,
        semantic_channels=args.semantic_channels,
        detail_context_channels=detail_channels,
        condition_channels=backbone.condition_channels,
        hidden_channels=args.hidden_channels,
        num_image_blocks=args.num_image_blocks,
        num_condition_blocks=args.num_condition_blocks,
        num_detail_blocks=args.num_detail_blocks,
        num_fusion_blocks=args.num_fusion_blocks,
        detail_control_branch=args.detail_control_branch,
        detail_control_blocks=args.detail_control_blocks,
        detail_control_condition_fusion=args.detail_control_condition_fusion,
        detail_highfreq_context_branch=args.detail_highfreq_context_branch,
        detail_film_modulation=args.detail_film_modulation,
    ).to(device)
    if args.init_checkpoint:
        init_payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        init_kind = str(init_payload.get("adapter_kind", "light"))
        if init_kind != args.adapter_kind:
            raise ValueError(f"init checkpoint adapter_kind mismatch: {init_kind} != {args.adapter_kind}")
        incompatible = adapter.load_state_dict(init_payload["model"], strict=not args.init_nonstrict)
        if args.init_nonstrict:
            print(
                "Loaded init checkpoint non-strictly: "
                f"missing={list(incompatible.missing_keys)}, unexpected={list(incompatible.unexpected_keys)}"
            )
        if init_payload.get("backbone_trainable_state"):
            init_lora_config = init_payload.get("backbone_lora_config", {})
            if isinstance(init_lora_config, dict) and init_lora_config.get("patterns") and not backbone_lora_module_names:
                loaded_lora_modules = apply_lora_adapters_from_config(backbone.net, init_lora_config)
                backbone_lora_module_names = loaded_lora_modules
                backbone_lora_param_names = lora_parameter_names(backbone.net)
                if backbone_lora_param_names and args.backbone_lora_lr <= 0.0:
                    raise ValueError(
                        "init checkpoint contains LoRA state; pass --backbone-lora-lr > 0 "
                        "to continue training those adapters"
                    )
            if isinstance(init_lora_config, dict) and init_lora_config.get("patterns") and not backbone_lora_module_names:
                raise ValueError("init checkpoint contains LoRA state but current run did not configure LoRA modules")
            loaded_backbone_names = load_named_parameter_state(
                backbone.net,
                init_payload["backbone_trainable_state"],
                strict=False,
            )
            print(f"Loaded backbone trainable state tensors: {len(loaded_backbone_names)}")

    control_basis_payload: dict[str, object] | None = None
    control_huffman = None
    control_sparse_huffman = None
    basis_control_codec = None
    affine_gain_codec = None
    affine_bias_codec = None
    effective_control_bits = int(args.control_bits)
    effective_control_range = float(args.control_range)
    effective_control_quantizer = str(args.control_quantizer)
    effective_control_mu = float(args.control_mu)
    effective_control_huffman_key = str(args.control_huffman_key)
    effective_control_basis_range_mode = str(args.control_basis_range_mode)
    effective_candidate_components = int(args.control_basis_components)
    if args.train_counted_control_mode in {"condition_residual_affine_basis", "condition_base_affine_basis"}:
        control_channel_group_sizes(backbone.condition_channels, args.control_affine_groups)
        control_basis_payload = control_load_control_basis(args.control_basis, device=device)
        if int(control_basis_payload["groups"]) != args.control_groups:
            raise ValueError("--control-groups must match control basis groups")
        if int(control_basis_payload["grid_size"]) != args.control_grid_size:
            raise ValueError("--control-grid-size must match control basis grid_size")
        basis_tensor = control_basis_payload["basis"]
        if not isinstance(basis_tensor, torch.Tensor):
            raise TypeError("loaded control basis is not a tensor")
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
            ) = control_load_control_huffman_prior(
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
            ) = control_load_sparse_topk_control_huffman_prior(
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
        elif args.control_codec == "huffman" and args.control_basis_selection == "prefix_topk":
            (
                prefix_huffman,
                prefix_bits,
                prefix_range,
                prefix_quantizer,
                prefix_mu,
                prefix_huffman_key,
                prefix_basis_range_mode,
            ) = control_load_control_huffman_prior(
                control_basis_payload,
                key=args.control_prefix_huffman_key,
                components=args.control_basis_prefix_components,
            )
            (
                index_huffman,
                value_huffman,
                sparse_bits,
                sparse_range,
                sparse_quantizer,
                sparse_mu,
                sparse_huffman_key,
                sparse_basis_range_mode,
            ) = control_load_sparse_topk_control_huffman_prior(
                control_basis_payload,
                key=args.control_huffman_key,
                candidate_components=effective_candidate_components,
                components=args.control_basis_components,
            )
            if int(prefix_bits) != int(sparse_bits):
                raise ValueError("prefix_topk prefix and sparse Huffman priors must use the same bit depth")
            if str(prefix_quantizer) != str(sparse_quantizer):
                raise ValueError("prefix_topk prefix and sparse Huffman priors must use the same quantizer")
            if abs(float(prefix_range) - float(sparse_range)) > 1.0e-8:
                raise ValueError("prefix_topk prefix and sparse Huffman priors must use the same value range")
            if abs(float(prefix_mu) - float(sparse_mu)) > 1.0e-8:
                raise ValueError("prefix_topk prefix and sparse Huffman priors must use the same mu")
            for prior_basis_range_mode in (prefix_basis_range_mode, sparse_basis_range_mode):
                if prior_basis_range_mode != "global":
                    if effective_control_basis_range_mode not in {"global", prior_basis_range_mode}:
                        raise ValueError(
                            "prefix_topk Huffman prior basis_range_mode conflicts with --control-basis-range-mode"
                        )
                    effective_control_basis_range_mode = prior_basis_range_mode
            control_huffman = prefix_huffman
            control_sparse_huffman = (index_huffman, value_huffman)
            effective_control_bits = int(sparse_bits)
            effective_control_range = float(sparse_range)
            effective_control_quantizer = str(sparse_quantizer)
            effective_control_mu = float(sparse_mu)
            effective_control_huffman_key = f"prefix:{prefix_huffman_key}|tail:{sparse_huffman_key}"
        elif args.control_codec == "huffman" and args.control_basis_selection == "vector":
            (
                control_huffman,
                effective_control_bits,
                effective_control_huffman_key,
            ) = control_load_vector_codebook_huffman_prior(
                control_basis_payload,
                key=args.control_huffman_key,
                components=args.control_basis_components,
                bits=args.control_bits,
            )

        basis_range_components = (
            effective_candidate_components
            if args.control_basis_selection in {"topk", "prefix_topk"}
            else args.control_basis_components
        )
        if args.control_basis_selection == "vector":
            basis_control_codec, vector_key = control_basis_vector_codebook_codec(
                control_basis_payload,
                components=args.control_basis_components,
                bits=effective_control_bits,
                key=effective_control_huffman_key if args.control_codec == "huffman" else "",
            )
            if not effective_control_huffman_key:
                effective_control_huffman_key = vector_key
        elif effective_control_basis_range_mode == "component_codebook":
            basis_control_codec = control_basis_component_codebook_codec(
                control_basis_payload,
                bits=effective_control_bits,
                components=basis_range_components,
            )
        else:
            basis_value_ranges = control_basis_component_ranges(
                control_basis_payload,
                mode=effective_control_basis_range_mode,
                components=basis_range_components,
                floor=args.control_basis_range_floor,
            )
            basis_control_codec = control_build_control_grid_code(
                quantizer=effective_control_quantizer,
                bits=effective_control_bits,
                value_range=effective_control_range,
                mu=effective_control_mu,
                value_ranges=basis_value_ranges,
            )
        affine_gain_codec = control_build_control_grid_code(
            quantizer=effective_control_quantizer,
            bits=effective_control_bits,
            value_range=args.control_affine_gain_range,
            mu=effective_control_mu,
        )
        affine_bias_codec = control_build_control_grid_code(
            quantizer=effective_control_quantizer,
            bits=effective_control_bits,
            value_range=args.control_affine_bias_range,
            mu=effective_control_mu,
        )

    train_counted_control_config = {
        "mode": args.train_counted_control_mode,
        "basis": args.control_basis,
        "grid_size": args.control_grid_size,
        "groups": args.control_groups,
        "basis_components": args.control_basis_components,
        "basis_candidate_components": effective_candidate_components,
        "basis_prefix_components": args.control_basis_prefix_components,
        "basis_selection": args.control_basis_selection,
        "basis_range_mode": effective_control_basis_range_mode,
        "codec": args.control_codec,
        "huffman_key": effective_control_huffman_key,
        "quantizer": effective_control_quantizer,
        "mu": effective_control_mu,
        "bits": effective_control_bits,
        "range": effective_control_range,
        "affine_groups": args.control_affine_groups,
        "affine_grid_size": args.control_affine_grid_size,
        "affine_gain_range": args.control_affine_gain_range,
        "affine_bias_range": args.control_affine_bias_range,
        "scale": args.control_scale,
    }
    optimizer_groups: list[dict[str, object]] = [{"params": list(adapter.parameters()), "lr": args.lr}]
    if backbone_trainable_names:
        backbone_trainable_params = [
            dict(backbone.net.named_parameters())[name] for name in backbone_trainable_names
        ]
        optimizer_groups.append(
            {
                "params": backbone_trainable_params,
                "lr": args.backbone_lr,
                "weight_decay": args.backbone_weight_decay,
            }
        )
    if backbone_lora_param_names:
        parameter_lookup = dict(backbone.net.named_parameters())
        optimizer_groups.append(
            {
                "params": [parameter_lookup[name] for name in backbone_lora_param_names],
                "lr": args.backbone_lora_lr,
                "weight_decay": args.backbone_lora_weight_decay,
            }
        )
    trainable_parameters_for_clip = [
        parameter
        for group in optimizer_groups
        for parameter in group["params"]
        if isinstance(parameter, torch.nn.Parameter) and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(optimizer_groups, lr=args.lr)

    wandb_run = init_wandb(
        {
            "stage": "stage4_cod_lite_adapter",
            "manifest": args.manifest,
            "crop_size": args.crop_size,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "base_condition": args.base_condition,
            "adapter_kind": args.adapter_kind,
            "detail_context": args.detail_context,
            "semantic_latent_dropout_prob": args.semantic_latent_dropout_prob,
            "detail_context_dropout_prob": args.detail_context_dropout_prob,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "condition_residual_scale": args.condition_residual_scale,
            "condition_residual_tanh": args.condition_residual_tanh,
            "pre_control_condition_l1_weight": args.pre_control_condition_l1_weight,
            "train_counted_control": train_counted_control_config,
            "num_image_blocks": args.num_image_blocks,
            "num_condition_blocks": args.num_condition_blocks,
            "num_detail_blocks": args.num_detail_blocks,
            "num_fusion_blocks": args.num_fusion_blocks,
            "detail_control_branch": args.detail_control_branch,
            "detail_control_blocks": args.detail_control_blocks,
            "detail_control_condition_fusion": args.detail_control_condition_fusion,
            "detail_highfreq_context_branch": args.detail_highfreq_context_branch,
            "detail_film_modulation": args.detail_film_modulation,
            "init_nonstrict": args.init_nonstrict,
            "condition_cosine_weight": args.condition_cosine_weight,
            "condition_channel_stats_weight": args.condition_channel_stats_weight,
            "condition_highfreq_weight": args.condition_highfreq_weight,
            "condition_highfreq_threshold": args.condition_highfreq_threshold,
            "condition_residual_rms_guard_weight": args.condition_residual_rms_guard_weight,
            "condition_residual_rms_guard_ratio": args.condition_residual_rms_guard_ratio,
            "condition_residual_rms_guard_granularity": args.condition_residual_rms_guard_granularity,
            "detail_contrast_weight": args.detail_contrast_weight,
            "detail_contrast_margin": args.detail_contrast_margin,
            "detail_highfreq_residual_weight": args.detail_highfreq_residual_weight,
            "detail_highfreq_kernel_size": args.detail_highfreq_kernel_size,
            "detail_residual_target_weight": args.detail_residual_target_weight,
            "grad_clip_norm": args.grad_clip_norm,
            "init_checkpoint": args.init_checkpoint,
            "backbone_train_pattern": args.backbone_train_pattern,
            "backbone_lr": args.backbone_lr,
            "backbone_weight_decay": args.backbone_weight_decay,
            "backbone_lora_pattern": args.backbone_lora_pattern,
            "backbone_lora_rank": args.backbone_lora_rank,
            "backbone_lora_alpha": args.backbone_lora_alpha,
            "backbone_lora_lr": args.backbone_lora_lr,
            "backbone_lora_weight_decay": args.backbone_lora_weight_decay,
            "backbone_lora_target": lora_targets,
            "backbone_lora_plan": args.backbone_lora_plan,
            "backbone_lora_preset": args.backbone_lora_preset,
            "backbone_lora_plan_setting": lora_plan_setting,
            "backbone_lora_module_names": backbone_lora_module_names,
            "backbone_lora_param_names": backbone_lora_param_names,
            "backbone_trainable_param_count": int(
                sum(parameter.numel() for parameter in backbone.net.parameters() if parameter.requires_grad)
            ),
            "backbone_trainable_param_names": backbone_trainable_names + backbone_lora_param_names,
            "image_l1_weight": args.image_l1_weight,
            "lpips_weight": args.lpips_weight,
            "dists_weight": args.dists_weight,
            "ms_ssim_weight": args.ms_ssim_weight,
            "stage3_l1_guard_weight": args.stage3_l1_guard_weight,
            "stage3_mse_guard_weight": args.stage3_mse_guard_weight,
            "stage3_guard_margin": args.stage3_guard_margin,
            "wandb": {"mode": args.wandb_mode, "project": "coserdic"},
        },
        run_name=run_name,
    )
    lpips_model = None
    if args.lpips_weight > 0:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").eval().to(device)
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)
    dists_model = None
    if args.dists_weight > 0:
        from DISTS_pytorch import DISTS

        dists_model = DISTS().eval().to(device)
        for parameter in dists_model.parameters():
            parameter.requires_grad_(False)
    use_image_decode_loss = (
        args.image_l1_weight > 0
        or args.lpips_weight > 0
        or args.dists_weight > 0
        or args.ms_ssim_weight > 0
        or args.stage3_l1_guard_weight > 0
        or args.stage3_mse_guard_weight > 0
    )

    metrics: dict[str, list[float]] = {
        "loss": [],
        "condition_l1": [],
        "adapter_target_condition_l1": [],
        "pre_control_condition_l1": [],
        "control_condition_l1_delta": [],
        "control_payload_bytes": [],
        "control_payload_bpp_512": [],
        "control_grid_abs": [],
        "condition_cosine_loss": [],
        "condition_channel_stats_loss": [],
        "condition_highfreq_loss": [],
        "condition_residual_rms_guard_loss": [],
        "condition_residual_rms_ratio_mean": [],
        "condition_residual_rms_ratio_max": [],
        "detail_contrast_loss": [],
        "detail_condition_l1_zero": [],
        "detail_condition_l1_gap": [],
        "detail_highfreq_residual_loss": [],
        "detail_highfreq_residual_pred_l1": [],
        "detail_highfreq_residual_target_l1": [],
        "detail_residual_target_loss": [],
        "detail_residual_target_pred_l1": [],
        "detail_residual_target_target_l1": [],
        "base_condition_l1": [],
        "condition_l1_delta_vs_base": [],
        "condition_cosine": [],
        "pred_condition_std": [],
        "target_condition_std": [],
        "pred_condition_highfreq": [],
        "target_condition_highfreq": [],
        "image_l1": [],
        "lpips": [],
        "dists_loss": [],
        "ms_ssim_loss": [],
        "stage3_l1_guard": [],
        "stage3_mse_guard": [],
        "stage4_psnr": [],
        "stage3_psnr": [],
        "condition_residual_l1": [],
        "condition_delta_raw_l1": [],
        "semantic_latent_drop_fraction": [],
        "detail_context_drop_fraction": [],
    }
    step = 0
    loader_iter = iter(loader)
    pbar = tqdm(total=args.max_steps, desc=run_name)
    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_rows: list[dict[str, float]] = []
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            reference = batch["reference"].to(device, non_blocking=True)
            semantic = batch["semantic"].to(device, non_blocking=True)
            semantic_latent = batch["semantic_latent"].to(device, non_blocking=True)
            detail_context = (
                batch["detail_context"].to(device, non_blocking=True) if detail_channels > 0 else None
            )
            stage3 = batch["stage3"].to(device, non_blocking=True)
            if args.semantic_channels != int(semantic_latent.shape[1]):
                raise ValueError(
                    f"--semantic-channels={args.semantic_channels} but semantic context has "
                    f"{int(semantic_latent.shape[1])} channels. Use a decoder feature cache for 256-channel latents."
                )
            residual = stage3 - semantic
            condition_size = backbone.condition_size(int(stage3.shape[-2]), int(stage3.shape[-1]))
            semantic_latent_for_adapter = semantic_latent
            detail_context_for_adapter = detail_context
            semantic_drop_fraction = reference.new_tensor(0.0)
            detail_drop_fraction = reference.new_tensor(0.0)
            if args.semantic_latent_dropout_prob > 0:
                keep_mask = (
                    torch.rand((semantic_latent.shape[0], 1, 1, 1), device=device)
                    >= args.semantic_latent_dropout_prob
                )
                semantic_latent_for_adapter = semantic_latent * keep_mask.to(dtype=semantic_latent.dtype)
                semantic_drop_fraction = 1.0 - keep_mask.float().mean()
            if detail_context is not None and args.detail_context_dropout_prob > 0:
                keep_mask = (
                    torch.rand((detail_context.shape[0], 1, 1, 1), device=device)
                    >= args.detail_context_dropout_prob
                )
                detail_context_for_adapter = detail_context * keep_mask.to(dtype=detail_context.dtype)
                detail_drop_fraction = 1.0 - keep_mask.float().mean()

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                target_cond = backbone.native_condition(reference)
                if args.base_condition == "native_stage3":
                    base_cond = backbone.native_condition(stage3)
                else:
                    base_cond = torch.zeros_like(target_cond)

            cond_delta = adapter(
                stage3,
                semantic,
                residual,
                semantic_latent_for_adapter,
                condition_size=condition_size,
                base_condition=base_cond,
                detail_context=detail_context_for_adapter,
            )
            pred_cond = apply_condition_residual(
                base_cond,
                cond_delta,
                residual_scale=args.condition_residual_scale,
                residual_tanh=args.condition_residual_tanh,
            )
            pre_control_pred_cond = pred_cond
            pre_control_condition_l1 = F.l1_loss(pre_control_pred_cond.float(), target_cond.float())
            adapter_target_cond = target_cond.float()
            control_payload_bytes = reference.new_tensor(0.0)
            control_payload_bpp_512 = reference.new_tensor(0.0)
            control_grid_abs = reference.new_tensor(0.0)
            if args.train_counted_control_mode in {"condition_residual_affine_basis", "condition_base_affine_basis"}:
                if (
                    control_basis_payload is None
                    or basis_control_codec is None
                    or affine_gain_codec is None
                    or affine_bias_codec is None
                ):
                    raise RuntimeError("counted control codecs were not initialized")
                control_source_cond = (
                    base_cond.detach().float()
                    if args.train_counted_control_mode == "condition_base_affine_basis"
                    else pre_control_pred_cond.detach().float()
                )
                with torch.no_grad():
                    control_correction, control_payload_byte_rows, control_abs_rows = (
                        control_grouped_condition_affine_basis_control(
                            base_cond.float(),
                            control_source_cond,
                            target_cond.float(),
                            affine_groups=args.control_affine_groups,
                            affine_grid_size=args.control_affine_grid_size,
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
                    control_payload_bytes = reference.new_tensor(
                        float(sum(control_payload_byte_rows) / max(len(control_payload_byte_rows), 1))
                    )
                    control_payload_bpp_512 = control_payload_bytes * 8.0 / float(stage3.shape[-2] * stage3.shape[-1])
                    control_grid_abs = control_abs_rows.float().mean()
                    adapter_target_cond = target_cond.float() - control_correction.float()
                pred_cond = pre_control_pred_cond + control_correction.to(dtype=pre_control_pred_cond.dtype)
            condition_l1 = F.l1_loss(pred_cond.float(), target_cond.float())
            adapter_target_condition_l1 = F.l1_loss(pre_control_pred_cond.float(), adapter_target_cond.float())
            base_condition_l1 = F.l1_loss(base_cond.float(), target_cond.float())
            control_condition_l1_delta = condition_l1 - pre_control_condition_l1
            condition_cosine = F.cosine_similarity(
                pred_cond.float().flatten(1),
                target_cond.float().flatten(1),
                dim=1,
                eps=1.0e-8,
            ).mean()
            pred_condition_std = pred_cond.float().std(dim=(1, 2, 3), unbiased=False).mean()
            target_condition_std = target_cond.float().std(dim=(1, 2, 3), unbiased=False).mean()
            pred_condition_hf = condition_highfreq_ratio(pred_cond).mean()
            target_condition_hf = condition_highfreq_ratio(target_cond).mean()
            if args.condition_cosine_weight > 0:
                condition_cos = condition_cosine_loss(pred_cond, target_cond)
            else:
                condition_cos = condition_l1.new_tensor(0.0)
            if args.condition_channel_stats_weight > 0:
                condition_stats = condition_channel_stats_loss(pred_cond, target_cond)
            else:
                condition_stats = condition_l1.new_tensor(0.0)
            if args.condition_highfreq_weight > 0:
                condition_hf = condition_highfreq_loss(
                    pred_cond,
                    target_cond,
                    threshold=args.condition_highfreq_threshold,
                )
            else:
                condition_hf = condition_l1.new_tensor(0.0)
            condition_residual = pred_cond - base_cond
            condition_rms_guard, condition_rms_ratio_mean, condition_rms_ratio_max = (
                condition_residual_rms_excess_loss(
                    base_cond,
                    condition_residual,
                    max_rms_ratio=args.condition_residual_rms_guard_ratio,
                    granularity=args.condition_residual_rms_guard_granularity,
                )
            )
            detail_contrast = condition_l1.new_tensor(0.0)
            detail_condition_l1_zero = condition_l1.new_tensor(0.0)
            detail_condition_l1_gap = condition_l1.new_tensor(0.0)
            detail_hf_residual = condition_l1.new_tensor(0.0)
            detail_hf_residual_pred_l1 = condition_l1.new_tensor(0.0)
            detail_hf_residual_target_l1 = condition_l1.new_tensor(0.0)
            detail_residual_target = condition_l1.new_tensor(0.0)
            detail_residual_target_pred_l1 = condition_l1.new_tensor(0.0)
            detail_residual_target_target_l1 = condition_l1.new_tensor(0.0)
            zero_detail_pred_cond = None
            if (
                args.detail_contrast_weight > 0
                or args.detail_highfreq_residual_weight > 0
                or args.detail_residual_target_weight > 0
            ):
                if detail_context_for_adapter is None:
                    raise RuntimeError("detail residual losses require detail_context_for_adapter")
                zero_detail_context = torch.zeros_like(detail_context_for_adapter)
                zero_detail_delta = adapter(
                    stage3,
                    semantic,
                    residual,
                    semantic_latent_for_adapter,
                    condition_size=condition_size,
                    base_condition=base_cond,
                    detail_context=zero_detail_context,
                )
                zero_detail_pred_cond = apply_condition_residual(
                    base_cond,
                    zero_detail_delta,
                    residual_scale=args.condition_residual_scale,
                    residual_tanh=args.condition_residual_tanh,
                )
            if args.detail_contrast_weight > 0:
                if zero_detail_pred_cond is None:
                    raise RuntimeError("detail contrast requires zero_detail_pred_cond")
                detail_condition_l1_zero = F.l1_loss(zero_detail_pred_cond.float(), adapter_target_cond.float())
                detail_condition_l1_gap = detail_condition_l1_zero - adapter_target_condition_l1
                detail_contrast = torch.relu(
                    adapter_target_condition_l1 - detail_condition_l1_zero + args.detail_contrast_margin
                )
            if args.detail_highfreq_residual_weight > 0:
                if zero_detail_pred_cond is None:
                    raise RuntimeError("detail high-frequency residual loss requires zero_detail_pred_cond")
                pred_detail_residual = pre_control_pred_cond - zero_detail_pred_cond
                target_condition_residual = adapter_target_cond - base_cond.float()
                pred_detail_hf = condition_local_highpass(
                    pred_detail_residual,
                    kernel_size=args.detail_highfreq_kernel_size,
                )
                target_detail_hf = condition_local_highpass(
                    target_condition_residual,
                    kernel_size=args.detail_highfreq_kernel_size,
                )
                detail_hf_residual = F.l1_loss(pred_detail_hf, target_detail_hf)
                detail_hf_residual_pred_l1 = pred_detail_hf.abs().mean()
                detail_hf_residual_target_l1 = target_detail_hf.abs().mean()
            if args.detail_residual_target_weight > 0:
                if zero_detail_pred_cond is None:
                    raise RuntimeError("detail residual target loss requires zero_detail_pred_cond")
                pred_detail_residual = pre_control_pred_cond - zero_detail_pred_cond
                target_condition_residual = adapter_target_cond - base_cond.float()
                detail_residual_target = F.l1_loss(
                    pred_detail_residual.float(),
                    target_condition_residual.float(),
                )
                detail_residual_target_pred_l1 = pred_detail_residual.float().abs().mean()
                detail_residual_target_target_l1 = target_condition_residual.float().abs().mean()
            if use_image_decode_loss:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    stage4 = backbone(stage3, pred_cond)
                image_l1 = F.l1_loss(stage4.float(), reference.float())
                ms_ssim_stage4 = ms_ssim(stage4.float(), reference.float(), data_range=1.0, size_average=True)
                msssim_loss = 1.0 - ms_ssim_stage4
                stage3_l1 = torch.mean(torch.abs(stage3.float() - reference.float()), dim=(1, 2, 3))
                stage4_l1 = torch.mean(torch.abs(stage4.float() - reference.float()), dim=(1, 2, 3))
                stage3_mse = torch.mean((stage3.float() - reference.float()).pow(2), dim=(1, 2, 3))
                stage4_mse = torch.mean((stage4.float() - reference.float()).pow(2), dim=(1, 2, 3))
                l1_guard = torch.relu(stage4_l1 - stage3_l1 - args.stage3_guard_margin).mean()
                mse_guard = torch.relu(stage4_mse - stage3_mse - args.stage3_guard_margin).mean()
            else:
                stage4 = None
                image_l1 = condition_l1.new_tensor(0.0)
                msssim_loss = condition_l1.new_tensor(0.0)
                l1_guard = condition_l1.new_tensor(0.0)
                mse_guard = condition_l1.new_tensor(0.0)
            if lpips_model is None or stage4 is None:
                lpips_loss = condition_l1.new_tensor(0.0)
            else:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    lpips_loss = lpips_model(
                        stage4.float().clamp(0, 1) * 2.0 - 1.0,
                        reference.float().clamp(0, 1) * 2.0 - 1.0,
                    ).mean()
            if dists_model is None or stage4 is None:
                dists_loss = condition_l1.new_tensor(0.0)
            else:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    dists_loss = dists_model(
                        stage4.float().clamp(0, 1),
                        reference.float().clamp(0, 1),
                    ).mean()
            loss = (
                args.condition_l1_weight * condition_l1
                + args.pre_control_condition_l1_weight * pre_control_condition_l1
                + args.condition_cosine_weight * condition_cos
                + args.condition_channel_stats_weight * condition_stats
                + args.condition_highfreq_weight * condition_hf
                + args.condition_residual_rms_guard_weight * condition_rms_guard
                + args.detail_contrast_weight * detail_contrast
                + args.detail_highfreq_residual_weight * detail_hf_residual
                + args.detail_residual_target_weight * detail_residual_target
                + args.image_l1_weight * image_l1
                + args.lpips_weight * lpips_loss
                + args.dists_weight * dists_loss
                + args.ms_ssim_weight * msssim_loss
                + args.stage3_l1_guard_weight * l1_guard
                + args.stage3_mse_guard_weight * mse_guard
            )
            (loss / args.grad_accum_steps).backward()

            with torch.no_grad():
                if stage4 is None:
                    stage4_psnr = reference.new_tensor(0.0)
                else:
                    stage4_psnr = psnr(reference.float(), stage4.float()).mean()
                stage3_psnr = psnr(reference.float(), stage3.float()).mean()
                condition_residual_l1 = torch.mean(torch.abs((pred_cond - base_cond).float()))
                condition_delta_raw_l1 = torch.mean(torch.abs(cond_delta.float()))
            row = {
                "loss": float(loss.item()),
                "condition_l1": float(condition_l1.item()),
                "adapter_target_condition_l1": float(adapter_target_condition_l1.item()),
                "pre_control_condition_l1": float(pre_control_condition_l1.item()),
                "control_condition_l1_delta": float(control_condition_l1_delta.item()),
                "control_payload_bytes": float(control_payload_bytes.item()),
                "control_payload_bpp_512": float(control_payload_bpp_512.item()),
                "control_grid_abs": float(control_grid_abs.item()),
                "condition_cosine_loss": float(condition_cos.item()),
                "condition_channel_stats_loss": float(condition_stats.item()),
                "condition_highfreq_loss": float(condition_hf.item()),
                "condition_residual_rms_guard_loss": float(condition_rms_guard.item()),
                "condition_residual_rms_ratio_mean": float(condition_rms_ratio_mean.item()),
                "condition_residual_rms_ratio_max": float(condition_rms_ratio_max.item()),
                "detail_contrast_loss": float(detail_contrast.item()),
                "detail_condition_l1_zero": float(detail_condition_l1_zero.item()),
                "detail_condition_l1_gap": float(detail_condition_l1_gap.item()),
                "detail_highfreq_residual_loss": float(detail_hf_residual.item()),
                "detail_highfreq_residual_pred_l1": float(detail_hf_residual_pred_l1.item()),
                "detail_highfreq_residual_target_l1": float(detail_hf_residual_target_l1.item()),
                "detail_residual_target_loss": float(detail_residual_target.item()),
                "detail_residual_target_pred_l1": float(detail_residual_target_pred_l1.item()),
                "detail_residual_target_target_l1": float(detail_residual_target_target_l1.item()),
                "base_condition_l1": float(base_condition_l1.item()),
                "condition_l1_delta_vs_base": float(condition_l1.item() - base_condition_l1.item()),
                "condition_cosine": float(condition_cosine.item()),
                "pred_condition_std": float(pred_condition_std.item()),
                "target_condition_std": float(target_condition_std.item()),
                "pred_condition_highfreq": float(pred_condition_hf.item()),
                "target_condition_highfreq": float(target_condition_hf.item()),
                "image_l1": float(image_l1.item()),
                "lpips": float(lpips_loss.item()),
                "dists_loss": float(dists_loss.item()),
                "ms_ssim_loss": float(msssim_loss.item()),
                "stage3_l1_guard": float(l1_guard.item()),
                "stage3_mse_guard": float(mse_guard.item()),
                "stage4_psnr": float(stage4_psnr.item()),
                "stage3_psnr": float(stage3_psnr.item()),
                "condition_residual_l1": float(condition_residual_l1.item()),
                "condition_delta_raw_l1": float(condition_delta_raw_l1.item()),
                "semantic_latent_drop_fraction": float(semantic_drop_fraction.item()),
                "detail_context_drop_fraction": float(detail_drop_fraction.item()),
            }
            accum_rows.append(row)
            if stage4 is not None and args.save_sample_every > 0 and (
                step == 0 or (step + 1) % args.save_sample_every == 0
            ):
                sample = torch.cat(
                    [
                        reference[:1].detach().cpu(),
                        semantic[:1].detach().cpu(),
                        stage3[:1].detach().cpu(),
                        stage4[:1].detach().cpu(),
                    ],
                    dim=0,
                )
                save_image(sample, results_dir / f"sample_step{step + 1:06d}.png", nrow=4)
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters_for_clip, args.grad_clip_norm)
        optimizer.step()

        row = {
            key: float(sum(item[key] for item in accum_rows) / max(len(accum_rows), 1))
            for key in accum_rows[0]
        }
        for key, value in row.items():
            metrics[key].append(value)
        wandb_run.log(row, step=step)
        step += 1
        pbar.update(1)
    pbar.close()

    summary = {f"{key}_mean": float(sum(values) / max(len(values), 1)) for key, values in metrics.items()}
    checkpoint_path = output_dir / f"{run_name}.pt"
    checkpoint_backbone_names = backbone_trainable_names + backbone_lora_param_names
    backbone_trainable_state = (
        named_parameter_state(backbone.net, checkpoint_backbone_names) if checkpoint_backbone_names else {}
    )
    torch.save(
        {
            "model": adapter.state_dict(),
            "adapter_config": adapter.cfg.__dict__,
            "adapter_kind": args.adapter_kind,
            "detail_context": args.detail_context,
            "semantic_latent_dropout_prob": args.semantic_latent_dropout_prob,
            "detail_context_dropout_prob": args.detail_context_dropout_prob,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "condition_residual_scale": args.condition_residual_scale,
            "condition_residual_tanh": args.condition_residual_tanh,
            "pre_control_condition_l1_weight": args.pre_control_condition_l1_weight,
            "train_counted_control": train_counted_control_config,
            "num_image_blocks": args.num_image_blocks,
            "num_condition_blocks": args.num_condition_blocks,
            "num_detail_blocks": args.num_detail_blocks,
            "num_fusion_blocks": args.num_fusion_blocks,
            "detail_control_branch": args.detail_control_branch,
            "detail_control_blocks": args.detail_control_blocks,
            "detail_control_condition_fusion": args.detail_control_condition_fusion,
            "detail_highfreq_context_branch": args.detail_highfreq_context_branch,
            "detail_film_modulation": args.detail_film_modulation,
            "init_nonstrict": args.init_nonstrict,
            "condition_cosine_weight": args.condition_cosine_weight,
            "condition_channel_stats_weight": args.condition_channel_stats_weight,
            "condition_highfreq_weight": args.condition_highfreq_weight,
            "condition_highfreq_threshold": args.condition_highfreq_threshold,
            "condition_residual_rms_guard_weight": args.condition_residual_rms_guard_weight,
            "condition_residual_rms_guard_ratio": args.condition_residual_rms_guard_ratio,
            "condition_residual_rms_guard_granularity": args.condition_residual_rms_guard_granularity,
            "detail_contrast_weight": args.detail_contrast_weight,
            "detail_contrast_margin": args.detail_contrast_margin,
            "detail_highfreq_residual_weight": args.detail_highfreq_residual_weight,
            "detail_highfreq_kernel_size": args.detail_highfreq_kernel_size,
            "detail_residual_target_weight": args.detail_residual_target_weight,
            "image_l1_weight": args.image_l1_weight,
            "lpips_weight": args.lpips_weight,
            "dists_weight": args.dists_weight,
            "ms_ssim_weight": args.ms_ssim_weight,
            "stage3_l1_guard_weight": args.stage3_l1_guard_weight,
            "stage3_mse_guard_weight": args.stage3_mse_guard_weight,
            "stage3_guard_margin": args.stage3_guard_margin,
            "backbone_config": backbone.cfg.__dict__,
            "backbone_train_pattern": args.backbone_train_pattern,
            "backbone_lr": args.backbone_lr,
            "backbone_weight_decay": args.backbone_weight_decay,
            "backbone_lora_config": {
                "patterns": args.backbone_lora_pattern,
                "rank": args.backbone_lora_rank,
                "alpha": args.backbone_lora_alpha,
                "target_module_types": list(lora_targets),
                "module_names": backbone_lora_module_names,
                "param_names": backbone_lora_param_names,
                "plan": args.backbone_lora_plan,
                "preset": args.backbone_lora_preset,
                "plan_setting": lora_plan_setting,
            },
            "backbone_trainable_param_names": checkpoint_backbone_names,
            "backbone_trainable_state": backbone_trainable_state,
            "summary": summary,
            "run_name": run_name,
            "manifest": args.manifest,
        },
        checkpoint_path,
    )
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_run_doc(
        Path("docs/experiments") / f"{run_name}.md",
        {
            "run_name": run_name,
            "date": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "summary": summary,
            "artifacts": {
                "checkpoint": str(checkpoint_path),
                "summary": str(summary_path),
                "results_dir": str(results_dir),
                "wandb": str(Path(wandb_run.dir).parent if wandb_run is not None else ""),
            },
        },
    )
    wandb_run.finish()


if __name__ == "__main__":
    main()
