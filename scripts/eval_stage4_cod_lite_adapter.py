from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from coserdic.metrics import PerceptualMetricBundle
from coserdic.models import (
    CoDLiteOneStepBackbone,
    CoDLiteOneStepBackboneConfig,
    CoSERToCoDLiteAlphaGate,
    CoSERToCoDLiteAlphaGateConfig,
    CoSERToCoDLiteConditionAdapter,
    CoSERToCoDLiteConditionAdapterConfig,
    CoSERToCoDLiteConditionPyramidAdapter,
    CoSERToCoDLiteConditionPyramidAdapterConfig,
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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        metric_row = self.metric_rows[index] if index < len(self.metric_rows) else {}
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
            semantic_latent = cache["semantic_latent"].float()
            if semantic_latent.ndim == 4 and semantic_latent.shape[0] == 1:
                semantic_latent = semantic_latent.squeeze(0)
        if self.detail_context != "none":
            detail_context = _load_detail_context(cache, self.detail_context)
        return {
            "reference": _load_image(Path(row["reference"]), self.crop_size),
            "semantic": semantic,
            "semantic_latent": semantic_latent,
            "detail_context": detail_context,
            "stage3": _load_image(Path(row["stage3"]), self.crop_size),
            "index": int(row.get("index", index)),
            "source_path": str(row.get("source_path", metric_row.get("path", ""))),
            "actual_payload_bpp": float(metric_row.get("actual_payload_bpp", 0.0)),
            "paper_bpp": float(metric_row.get("paper_bpp", metric_row.get("actual_payload_bpp", 0.0))),
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
        detail_codes = detail_codes / 15.0 * 2.0 - 1.0
        return torch.cat([residual_grid, detail_codes], dim=0)
    raise ValueError(f"unknown detail_context: {mode}")


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
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 4 evaluation.")

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda")
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_stage4_cod_lite_adapter_eval"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone_cfg = CoDLiteOneStepBackboneConfig(**payload["backbone_config"])
    backbone = CoDLiteOneStepBackbone.load(backbone_cfg, device=device)
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
    perceptual = PerceptualMetricBundle().to(device)

    wandb_run = init_wandb(
        {
            "stage": "stage4_cod_lite_adapter_eval",
            "checkpoint": args.checkpoint,
            "gate_checkpoint": args.gate_checkpoint,
            "manifest": args.manifest,
            "per_image_metrics": args.per_image_metrics,
            "crop_size": args.crop_size,
            "limit": args.limit,
            "detail_context": detail_context,
            "condition_residual_scale": residual_scale,
            "condition_residual_tanh": residual_tanh,
            "wandb": {"mode": args.wandb_mode, "project": "coserdic"},
        },
        run_name=run_name,
    )

    recon_dir = out_dir / "reconstructions"
    if args.save_reconstructions:
        for name in ("reference", "semantic_only", "stage3", "stage4", "quad"):
            (recon_dir / name).mkdir(parents=True, exist_ok=True)

    metrics: dict[str, list[float]] = {
        "actual_payload_bpp": [],
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
        "base_condition_l1": [],
        "condition_l1_delta_vs_base": [],
        "condition_residual_l1": [],
        "condition_delta_raw_l1": [],
        "stage4_alpha": [],
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
                pred_cond = apply_condition_residual(
                    base_cond,
                    cond_delta,
                    residual_scale=residual_scale,
                    residual_tanh=residual_tanh,
                )
                stage4_raw = backbone(stage3, pred_cond)
                if gate is None:
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
                stage4 = ((1.0 - alpha) * stage3 + alpha * stage4_raw).clamp(0, 1)
            condition_l1 = torch.mean(torch.abs((pred_cond - target_cond).float()), dim=(1, 2, 3))
            base_condition_l1 = torch.mean(torch.abs((base_cond - target_cond).float()), dim=(1, 2, 3))
            condition_residual_l1 = torch.mean(torch.abs((pred_cond - base_cond).float()), dim=(1, 2, 3))
            condition_delta_raw_l1 = torch.mean(torch.abs(cond_delta.float()), dim=(1, 2, 3))

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
                row = {
                    "index": int(batch["index"][item]),
                    "source_path": str(batch["source_path"][item]),
                    "actual_payload_bpp": float(batch["actual_payload_bpp"][item]),
                    "paper_bpp": float(batch["paper_bpp"][item]),
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
                    "base_condition_l1": float(base_condition_l1[item].item()),
                    "condition_l1_delta_vs_base": float(condition_l1[item].item() - base_condition_l1[item].item()),
                    "condition_residual_l1": float(condition_residual_l1[item].item()),
                    "condition_delta_raw_l1": float(condition_delta_raw_l1[item].item()),
                    "stage4_alpha": float(alpha[item].item()),
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
                    save_image(ref_i.detach().cpu(), recon_dir / "reference" / image_name)
                    save_image(semantic[item : item + 1].detach().cpu(), recon_dir / "semantic_only" / image_name)
                    save_image(s3_i.detach().cpu(), recon_dir / "stage3" / image_name)
                    save_image(s4_i.detach().cpu(), recon_dir / "stage4" / image_name)
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
            "condition_residual_scale": residual_scale,
            "condition_residual_tanh": residual_tanh,
            "stage4_payload_policy": (
                "inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, "
                "and deterministic decoder-side blend/gate alpha are not image-specific side information"
            ),
        }
    )
    summary_path = out_dir / "summary.json"
    per_image_path = out_dir / "per_image_metrics.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    per_image_path.write_text("\n".join(json.dumps(row, allow_nan=False) for row in per_image) + "\n")

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
