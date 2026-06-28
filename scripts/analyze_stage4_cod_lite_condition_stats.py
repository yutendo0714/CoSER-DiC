from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from coserdic.models import CoDLiteOneStepBackbone, CoDLiteOneStepBackboneConfig
from coserdic.utils import seed_everything
from coserdic.utils.wandb_utils import init_wandb
from eval_stage4_cod_lite_adapter import (
    Stage4ManifestDataset,
    apply_condition_residual,
    build_adapter_from_payload,
    detail_context_channels,
    mean,
    write_run_doc,
)


class ChannelAccumulator:
    def __init__(self) -> None:
        self.sum: torch.Tensor | None = None
        self.sumsq: torch.Tensor | None = None
        self.spatial_var_sum: torch.Tensor | None = None
        self.count = 0
        self.sample_count = 0

    def update(self, tensor: torch.Tensor) -> None:
        x = tensor.detach().float().cpu()
        if x.ndim != 4:
            raise ValueError(f"expected BCHW tensor, got shape={tuple(x.shape)}")
        sum_ = x.sum(dim=(0, 2, 3))
        sumsq = x.square().sum(dim=(0, 2, 3))
        spatial_var = x.var(dim=(2, 3), unbiased=False).sum(dim=0)
        if self.sum is None:
            self.sum = torch.zeros_like(sum_)
            self.sumsq = torch.zeros_like(sumsq)
            self.spatial_var_sum = torch.zeros_like(spatial_var)
        self.sum += sum_
        self.sumsq += sumsq
        self.spatial_var_sum += spatial_var
        self.count += int(x.shape[0] * x.shape[2] * x.shape[3])
        self.sample_count += int(x.shape[0])

    def finalize(self) -> dict[str, object]:
        if self.sum is None or self.sumsq is None or self.spatial_var_sum is None or self.count == 0:
            return {}
        mean_ = self.sum / self.count
        var = (self.sumsq / self.count - mean_.square()).clamp_min(0)
        return {
            "channel_mean": [float(v) for v in mean_],
            "channel_std": [float(v) for v in torch.sqrt(var)],
            "channel_spatial_var_mean": [float(v) for v in self.spatial_var_sum / max(self.sample_count, 1)],
        }


class HistogramAccumulator:
    def __init__(self, *, bins: int, min_value: float, max_value: float) -> None:
        self.bins = int(bins)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.counts = torch.zeros(self.bins, dtype=torch.long)
        self.below = 0
        self.above = 0
        self.total = 0

    def update(self, tensor: torch.Tensor) -> None:
        x = tensor.detach().float().cpu()
        self.below += int((x < self.min_value).sum().item())
        self.above += int((x > self.max_value).sum().item())
        self.total += int(x.numel())
        self.counts += torch.histc(x.clamp(self.min_value, self.max_value), bins=self.bins, min=self.min_value, max=self.max_value).long()

    def finalize(self) -> dict[str, object]:
        edges = torch.linspace(self.min_value, self.max_value, self.bins + 1)
        return {
            "range": [self.min_value, self.max_value],
            "bins": self.bins,
            "edges": [float(v) for v in edges],
            "counts": [int(v) for v in self.counts],
            "below_range": int(self.below),
            "above_range": int(self.above),
            "total": int(self.total),
        }


def tensor_metric_arrays(prefix: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    x = tensor.float()
    dims = (1, 2, 3)
    channel_mean = x.mean(dim=(2, 3))
    channel_std = x.std(dim=(2, 3), unbiased=False)
    flattened = x.flatten(1)
    return {
        f"{prefix}_mean": x.mean(dim=dims),
        f"{prefix}_std": x.std(dim=dims, unbiased=False),
        f"{prefix}_min": x.amin(dim=dims),
        f"{prefix}_max": x.amax(dim=dims),
        f"{prefix}_abs_mean": x.abs().mean(dim=dims),
        f"{prefix}_rms": torch.sqrt(x.square().mean(dim=dims).clamp_min(1.0e-12)),
        f"{prefix}_l2": torch.linalg.vector_norm(flattened, dim=1),
        f"{prefix}_channel_mean_std": channel_mean.std(dim=1, unbiased=False),
        f"{prefix}_channel_std_mean": channel_std.mean(dim=1),
        f"{prefix}_channel_std_std": channel_std.std(dim=1, unbiased=False),
        f"{prefix}_spatial_highfreq_ratio": spatial_high_frequency_ratio(x),
    }


def pair_metric_arrays(prefix: str, candidate: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    cand = candidate.float()
    ref = target.float()
    diff = cand - ref
    cand_flat = cand.flatten(1)
    ref_flat = ref.flatten(1)
    diff_flat = diff.flatten(1)
    ref_norm = torch.linalg.vector_norm(ref_flat, dim=1).clamp_min(1.0e-8)
    return {
        f"{prefix}_l1": diff.abs().mean(dim=(1, 2, 3)),
        f"{prefix}_mse": diff.square().mean(dim=(1, 2, 3)),
        f"{prefix}_rms": torch.sqrt(diff.square().mean(dim=(1, 2, 3)).clamp_min(1.0e-12)),
        f"{prefix}_relative_l2": torch.linalg.vector_norm(diff_flat, dim=1) / ref_norm,
        f"{prefix}_cosine": F.cosine_similarity(cand_flat, ref_flat, dim=1, eps=1.0e-8),
        f"{prefix}_mean_delta": diff.mean(dim=(1, 2, 3)),
        f"{prefix}_std_delta": cand.std(dim=(1, 2, 3), unbiased=False) - ref.std(dim=(1, 2, 3), unbiased=False),
    }


def spatial_high_frequency_ratio(tensor: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
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
    return (high / total).mean(dim=1)


def append_batch_rows(
    rows: list[dict[str, object]],
    batch: dict[str, object],
    arrays: dict[str, torch.Tensor],
) -> None:
    batch_size = int(next(iter(arrays.values())).shape[0])
    for item in range(batch_size):
        row: dict[str, object] = {
            "index": int(batch["index"][item]),
            "source_path": str(batch["source_path"][item]),
            "actual_payload_bpp": float(batch["actual_payload_bpp"][item]),
            "paper_bpp": float(batch["paper_bpp"][item]),
        }
        for key, value in arrays.items():
            row[key] = float(value[item].detach().float().cpu().item())
        rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--per-image-metrics", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="results/stage4_cod_lite_condition_stats")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
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
    parser.add_argument("--hist-bins", type=int, default=100)
    parser.add_argument("--hist-min", type=float, default=-5.0)
    parser.add_argument("--hist-max", type=float, default=5.0)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; stop Stage 4 condition-stat analysis until the container is restarted.")

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda")
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_stage4_cod_lite_condition_stats"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone_cfg = CoDLiteOneStepBackboneConfig(**payload["backbone_config"])
    backbone = CoDLiteOneStepBackbone.load(backbone_cfg, device=device)
    backbone.eval()
    adapter = build_adapter_from_payload(payload).to(device)
    adapter.load_state_dict(payload["model"])
    adapter.eval()

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

    dataset = Stage4ManifestDataset(
        Path(args.manifest),
        Path(args.per_image_metrics) if args.per_image_metrics else None,
        limit=args.limit,
        crop_size=args.crop_size,
        semantic_channels=semantic_channels,
        detail_context=detail_context,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    wandb_run = init_wandb(
        {
            "stage": "stage4_cod_lite_condition_stats",
            "checkpoint": args.checkpoint,
            "manifest": args.manifest,
            "per_image_metrics": args.per_image_metrics,
            "crop_size": args.crop_size,
            "limit": args.limit,
            "batch_size": args.batch_size,
            "detail_context": detail_context,
            "condition_residual_scale": residual_scale,
            "condition_residual_tanh": residual_tanh,
            "hist_bins": args.hist_bins,
            "hist_min": args.hist_min,
            "hist_max": args.hist_max,
            "wandb": {"mode": args.wandb_mode, "project": "coserdic"},
        },
        run_name=run_name,
    )

    per_image: list[dict[str, object]] = []
    metrics: dict[str, list[float]] = defaultdict(list)
    channel_accumulators = {name: ChannelAccumulator() for name in ("target", "base", "pred", "pred_minus_target", "base_minus_target", "pred_minus_base", "adapter_delta_raw")}
    histograms = {
        name: HistogramAccumulator(bins=args.hist_bins, min_value=args.hist_min, max_value=args.hist_max)
        for name in channel_accumulators
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc=run_name):
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
                target_cond = backbone.native_condition(reference)
                base_cond = backbone.native_condition(stage3)
                cond_delta = adapter(
                    stage3,
                    semantic,
                    residual,
                    semantic_latent,
                    condition_size=condition_size,
                    base_condition=base_cond,
                    detail_context=detail_context_tensor,
                )
                pred_cond = apply_condition_residual(
                    base_cond,
                    cond_delta,
                    residual_scale=residual_scale,
                    residual_tanh=residual_tanh,
                )

            tensors = {
                "target": target_cond,
                "base": base_cond,
                "pred": pred_cond,
                "pred_minus_target": pred_cond - target_cond,
                "base_minus_target": base_cond - target_cond,
                "pred_minus_base": pred_cond - base_cond,
                "adapter_delta_raw": cond_delta,
            }
            for name, tensor in tensors.items():
                channel_accumulators[name].update(tensor)
                histograms[name].update(tensor)

            arrays: dict[str, torch.Tensor] = {}
            arrays.update(tensor_metric_arrays("target", target_cond))
            arrays.update(tensor_metric_arrays("base", base_cond))
            arrays.update(tensor_metric_arrays("pred", pred_cond))
            arrays.update(tensor_metric_arrays("adapter_delta_raw", cond_delta))
            arrays.update(pair_metric_arrays("base_to_target", base_cond, target_cond))
            arrays.update(pair_metric_arrays("pred_to_target", pred_cond, target_cond))
            arrays.update(pair_metric_arrays("pred_to_base", pred_cond, base_cond))
            arrays["pred_l1_improvement_vs_base"] = arrays["base_to_target_l1"] - arrays["pred_to_target_l1"]
            arrays["pred_relative_l2_improvement_vs_base"] = (
                arrays["base_to_target_relative_l2"] - arrays["pred_to_target_relative_l2"]
            )

            append_batch_rows(per_image, batch, arrays)
            for key, value in arrays.items():
                values = [float(v) for v in value.detach().float().cpu()]
                metrics[key].extend(values)
            wandb_run.log({f"{key}_mean": mean([float(v) for v in value.detach().float().cpu()]) for key, value in arrays.items()})

    summary = {f"{key}_mean": mean(values) for key, values in metrics.items()}
    summary.update(
        {
            "count": len(per_image),
            "actual_payload_bpp_mean": mean([float(row["actual_payload_bpp"]) for row in per_image]),
            "pred_condition_l1_win_rate": mean(
                [1.0 if float(row["pred_l1_improvement_vs_base"]) > 0 else 0.0 for row in per_image]
            ),
            "pred_relative_l2_win_rate": mean(
                [1.0 if float(row["pred_relative_l2_improvement_vs_base"]) > 0 else 0.0 for row in per_image]
            ),
            "condition_residual_scale": residual_scale,
            "condition_residual_tanh": residual_tanh,
            "payload_policy": (
                "analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, "
                "and fixed CoD-Lite weights, so no image-specific side information is introduced"
            ),
        }
    )

    channel_stats = {name: acc.finalize() for name, acc in channel_accumulators.items()}
    histogram_stats = {name: hist.finalize() for name, hist in histograms.items()}
    summary_path = out_dir / "summary.json"
    per_image_path = out_dir / "per_image_condition_stats.jsonl"
    channel_path = out_dir / "channel_stats.json"
    histogram_path = out_dir / "activation_histograms.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    per_image_path.write_text("\n".join(json.dumps(row, allow_nan=False, sort_keys=True) for row in per_image) + "\n")
    channel_path.write_text(json.dumps(channel_stats, indent=2, sort_keys=True) + "\n")
    histogram_path.write_text(json.dumps(histogram_stats, indent=2, sort_keys=True) + "\n")

    write_run_doc(
        Path("docs/experiments") / f"{run_name}.md",
        {
            "run_name": run_name,
            "date": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "summary": summary,
            "artifacts": {
                "summary": str(summary_path),
                "per_image_condition_stats": str(per_image_path),
                "channel_stats": str(channel_path),
                "activation_histograms": str(histogram_path),
                "wandb": str(Path(wandb_run.dir).parent if wandb_run is not None else ""),
            },
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    wandb_run.finish()


if __name__ == "__main__":
    main()
