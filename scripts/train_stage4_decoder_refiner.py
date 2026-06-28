from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from pytorch_msssim import ms_ssim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from coserdic.datasets.image_folder import IMAGE_SUFFIXES
from coserdic.entropy import UniformResidualGridCode
from coserdic.models import (
    DECODER_POSTPROCESS_MODES,
    DecoderSideRefiner,
    DecoderSideRefinerConfig,
    SemanticVQAutoEncoder,
    SemanticVQConfig,
    apply_decoder_postprocess,
)
from coserdic.utils import seed_everything


class RandomCropImageDataset(Dataset):
    def __init__(self, roots: list[str], crop_size: int, random_flip: bool = True, limit: int = 0) -> None:
        self.paths: list[Path] = []
        for root in roots:
            self.paths.extend(sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no images found under {roots}")
        self.crop_size = int(crop_size)
        self.random_flip = bool(random_flip)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        tensor = TF.to_tensor(image)
        _, h, w = tensor.shape
        if h < self.crop_size or w < self.crop_size:
            scale = self.crop_size / min(h, w)
            tensor = TF.resize(
                tensor,
                [max(self.crop_size, int(round(h * scale))), max(self.crop_size, int(round(w * scale)))],
                antialias=True,
            )
            _, h, w = tensor.shape
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        tensor = TF.crop(tensor, top, left, self.crop_size, self.crop_size)
        if self.random_flip and random.random() < 0.5:
            tensor = TF.hflip(tensor)
        return tensor


class PerceptualTrainingLoss(nn.Module):
    def __init__(self, *, lpips_weight: float, dists_weight: float) -> None:
        super().__init__()
        self.lpips_weight = float(lpips_weight)
        self.dists_weight = float(dists_weight)
        self.lpips_alex: nn.Module | None = None
        self.dists: nn.Module | None = None
        if self.lpips_weight > 0:
            import lpips

            self.lpips_alex = lpips.LPIPS(net="alex").eval()
        if self.dists_weight > 0:
            from DISTS_pytorch import DISTS

            self.dists = DISTS().eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, reconstruction: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        if self.lpips_alex is not None:
            losses["lpips"] = self.lpips_alex(
                reconstruction.float().clamp(0.0, 1.0) * 2.0 - 1.0,
                reference.float().clamp(0.0, 1.0) * 2.0 - 1.0,
            ).mean()
        if self.dists is not None:
            losses["dists"] = self.dists(
                reconstruction.float().clamp(0.0, 1.0),
                reference.float().clamp(0.0, 1.0),
            ).mean()
        return losses


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y).pow(2)).detach()
    if float(mse.cpu()) <= 0:
        return float("inf")
    return float((-10.0 * torch.log10(mse)).cpu())


def spatial_total_variation(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("spatial_total_variation expects BCHW tensor")
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def chroma_residual_l1(rgb_residual: torch.Tensor) -> torch.Tensor:
    if rgb_residual.ndim != 4 or rgb_residual.shape[1] != 3:
        raise ValueError("chroma_residual_l1 expects Bx3xHxW RGB residual tensor")
    r, g, b = rgb_residual[:, 0:1], rgb_residual[:, 1:2], rgb_residual[:, 2:3]
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.mean(torch.abs(cb)) + torch.mean(torch.abs(cr))


def write_run_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['run_name']}",
        "",
        f"Date: {payload['date']}",
        "",
        "## Command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Artifacts", ""])
    for key, value in payload["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def load_yaml(path: str) -> dict[str, Any]:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text())


def choose_int(cli_value: int, config_value: Any, default: int) -> int:
    return int(cli_value if cli_value else config_value if config_value is not None else default)


def choose_float(cli_value: float, config_value: Any, default: float) -> float:
    return float(cli_value if cli_value != 0.0 else config_value if config_value is not None else default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/train_stage4_decoder_refiner.yaml")
    parser.add_argument("--stage1-checkpoint", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--train-root", action="append", default=None)
    parser.add_argument("--output-dir", default="outputs/stage4_decoder_refiner")
    parser.add_argument("--checkpoint-dir", default="checkpoints/stage4_decoder_refiner")
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=0)
    parser.add_argument("--semantic-context-channels", type=int, default=0)
    parser.add_argument("--num-res-blocks", type=int, default=0)
    parser.add_argument("--residual-scale", type=float, default=0.0)
    parser.add_argument("--detail-downsample-factor", type=int, default=0)
    parser.add_argument("--detail-bits", type=int, default=0)
    parser.add_argument("--detail-range", type=float, default=0.0)
    parser.add_argument("--detail-gain", type=float, default=0.0)
    parser.add_argument("--decoder-postprocess", choices=DECODER_POSTPROCESS_MODES, default="")
    parser.add_argument("--decoder-postprocess-strength", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-l1", type=float, default=-1.0)
    parser.add_argument("--loss-ms-ssim", type=float, default=-1.0)
    parser.add_argument("--loss-lpips", type=float, default=-1.0)
    parser.add_argument("--loss-dists", type=float, default=-1.0)
    parser.add_argument("--loss-anchor-stage3", type=float, default=-1.0)
    parser.add_argument("--loss-refiner-tv", type=float, default=-1.0)
    parser.add_argument("--loss-refiner-chroma", type=float, default=-1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    seed_everything(int(raw.get("experiment", {}).get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 4 training.")

    data_cfg = raw.get("data", {})
    opt_cfg = raw.get("optimization", {})
    loss_cfg = raw.get("loss", {})
    model_cfg = raw.get("model", {})
    stage3_cfg = raw.get("stage3_decoder_input", {})

    stage1_checkpoint = args.stage1_checkpoint or model_cfg.get("stage1_checkpoint", "")
    if not stage1_checkpoint:
        raise ValueError("--stage1-checkpoint or model.stage1_checkpoint is required")
    crop_size = int(args.crop_size or data_cfg.get("crop_size", 512))
    max_steps = int(args.max_steps or opt_cfg.get("max_steps", 100))
    batch_size = int(args.batch_size or opt_cfg.get("batch_size", 2))
    checkpoint_every = int(args.checkpoint_every or opt_cfg.get("checkpoint_every", 0) or 0)
    detail_downsample_factor = choose_int(
        args.detail_downsample_factor,
        stage3_cfg.get("detail_downsample_factor"),
        32,
    )
    if crop_size % detail_downsample_factor != 0:
        raise ValueError("--crop-size must be divisible by --detail-downsample-factor")

    roots = args.train_root or data_cfg.get("train_roots") or ["/dpl/open-images-v6/train/data"]
    dataset = RandomCropImageDataset(roots, crop_size=crop_size, random_flip=True, limit=args.limit_images)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    stage1_path = Path(stage1_checkpoint)
    stage1_payload = torch.load(stage1_path, map_location="cpu", weights_only=False)
    semantic_cfg = SemanticVQConfig(**stage1_payload["semantic_vq_config"])
    semantic_model = SemanticVQAutoEncoder(semantic_cfg).to(device)
    semantic_model.load_state_dict(stage1_payload["model"])
    semantic_model.eval()
    for param in semantic_model.parameters():
        param.requires_grad_(False)

    refiner_cfg = DecoderSideRefinerConfig(
        semantic_channels=int(semantic_cfg.latent_channels),
        base_channels=choose_int(args.base_channels, model_cfg.get("base_channels"), 64),
        semantic_context_channels=choose_int(
            args.semantic_context_channels,
            model_cfg.get("semantic_context_channels"),
            32,
        ),
        num_res_blocks=choose_int(args.num_res_blocks, model_cfg.get("num_res_blocks"), 6),
        residual_scale=choose_float(args.residual_scale, model_cfg.get("residual_scale"), 0.05),
        use_semantic_latent=bool(model_cfg.get("use_semantic_latent", True)),
        zero_init_output=bool(model_cfg.get("zero_init_output", True)),
    )
    refiner = DecoderSideRefiner(refiner_cfg).to(device)
    init_summary: dict[str, object] = {}
    if args.init_checkpoint:
        init_payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        load_result = refiner.load_state_dict(init_payload["model"], strict=False)
        init_summary = {
            "init_checkpoint": args.init_checkpoint,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }

    detail_bits = choose_int(args.detail_bits, stage3_cfg.get("detail_bits"), 4)
    detail_range = choose_float(args.detail_range, stage3_cfg.get("detail_range"), 0.25)
    detail_gain = choose_float(args.detail_gain, stage3_cfg.get("detail_gain"), 0.5)
    postprocess_mode = args.decoder_postprocess or str(stage3_cfg.get("decoder_postprocess", "unsharp3x3"))
    postprocess_strength = choose_float(
        args.decoder_postprocess_strength,
        stage3_cfg.get("decoder_postprocess_strength"),
        0.15,
    )
    residual_code = UniformResidualGridCode(bits=detail_bits, value_range=detail_range, codec="fixed_bits")
    detail_hw = crop_size // detail_downsample_factor

    lr = float(args.lr or opt_cfg.get("lr", 5.0e-5))
    opt = torch.optim.AdamW(
        refiner.parameters(),
        lr=lr,
        weight_decay=choose_float(args.weight_decay, opt_cfg.get("weight_decay"), 1.0e-4),
    )
    amp_enabled = bool(opt_cfg.get("mixed_precision", True))
    if args.mixed_precision is not None:
        amp_enabled = bool(args.mixed_precision)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_l1 = float(args.loss_l1 if args.loss_l1 >= 0 else loss_cfg.get("l1", 0.30))
    loss_ms_ssim = float(args.loss_ms_ssim if args.loss_ms_ssim >= 0 else loss_cfg.get("ms_ssim", 0.20))
    loss_lpips = float(args.loss_lpips if args.loss_lpips >= 0 else loss_cfg.get("lpips", 0.30))
    loss_dists = float(args.loss_dists if args.loss_dists >= 0 else loss_cfg.get("dists", 0.0))
    loss_anchor = float(
        args.loss_anchor_stage3 if args.loss_anchor_stage3 >= 0 else loss_cfg.get("anchor_stage3_l1", 0.05)
    )
    loss_refiner_tv = float(args.loss_refiner_tv if args.loss_refiner_tv >= 0 else loss_cfg.get("refiner_tv", 0.0))
    loss_refiner_chroma = float(
        args.loss_refiner_chroma if args.loss_refiner_chroma >= 0 else loss_cfg.get("refiner_chroma", 0.0)
    )
    perceptual_loss = PerceptualTrainingLoss(lpips_weight=loss_lpips, dists_weight=loss_dists).to(device).eval()

    import wandb

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{timestamp}_stage4_decoder_refiner"
    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config={
            "config": raw,
            "stage1_checkpoint": str(stage1_path),
            "train_roots": roots,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "checkpoint_every": checkpoint_every,
            "crop_size": crop_size,
            "refiner": refiner_cfg.__dict__,
            "stage3_decoder_input": {
                "detail_downsample_factor": detail_downsample_factor,
                "detail_bits": detail_bits,
                "detail_range": detail_range,
                "detail_gain": detail_gain,
                "decoder_postprocess": postprocess_mode,
                "decoder_postprocess_strength": postprocess_strength,
            },
            "initialization": init_summary,
            "loss": {
                "l1": loss_l1,
                "ms_ssim": loss_ms_ssim,
                "lpips": loss_lpips,
                "dists": loss_dists,
                "anchor_stage3_l1": loss_anchor,
                "refiner_tv": loss_refiner_tv,
                "refiner_chroma": loss_refiner_chroma,
            },
            "mixed_precision": amp_enabled,
            "payload_policy": "fixed decoder-side weights; no additional actual_payload_bpp",
        },
    )

    out_dir = Path(args.output_dir) / run_name
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    start = time.time()
    last_metrics: dict[str, float] = {}
    intermediate_checkpoints: list[str] = []

    def checkpoint_payload(*, current_step: int, metrics: dict[str, float]) -> dict[str, Any]:
        return {
            "model": refiner.state_dict(),
            "refiner_config": refiner_cfg.__dict__,
            "stage1_checkpoint": str(stage1_path),
            "semantic_vq_config": semantic_cfg.__dict__,
            "stage3_decoder_input": {
                "detail_downsample_factor": detail_downsample_factor,
                "detail_bits": detail_bits,
                "detail_range": detail_range,
                "detail_gain": detail_gain,
                "decoder_postprocess": postprocess_mode,
                "decoder_postprocess_strength": postprocess_strength,
            },
            "config": raw,
            "step": current_step,
            "metrics": metrics,
        }

    progress = tqdm(total=max_steps, desc=run_name)
    refiner.train()
    while step < max_steps:
        for x_cpu in loader:
            step += 1
            x = x_cpu.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                semantic = semantic_model(x, quantize_mix=1.0)
                x_sem = semantic["x_sem"].detach()
                s_hat = semantic["quantized"].detach()
                residual = x - x_sem
                residual_grid = F.adaptive_avg_pool2d(residual, output_size=(detail_hw, detail_hw)).detach().cpu()
                detail_codes = residual_code.quantize(residual_grid)
                residual_grid_hat = residual_code.dequantize(detail_codes).to(device=device, dtype=x.dtype)
                residual_hat = F.interpolate(
                    residual_grid_hat,
                    size=tuple(int(v) for v in x.shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                )
                residual_hat = residual_hat * detail_gain
                x_stage3 = (x_sem + residual_hat).clamp(0.0, 1.0)
                x_stage3 = apply_decoder_postprocess(
                    x_stage3,
                    mode=postprocess_mode,
                    strength=postprocess_strength,
                )

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                refined = refiner(x_stage3, x_sem, residual_hat, s_hat)
                x_refined = refined["x_refined"]
            with torch.amp.autocast("cuda", enabled=False):
                x_refined_f = x_refined.float()
                x_f = x.float()
                x_stage3_f = x_stage3.float()
                l1 = F.l1_loss(x_refined_f, x_f)
                anchor = F.l1_loss(x_refined_f, x_stage3_f)
                if loss_ms_ssim > 0:
                    ms = 1.0 - ms_ssim(x_refined_f, x_f, data_range=1.0, size_average=True)
                else:
                    ms = x.new_tensor(0.0)
                p_losses = perceptual_loss(x_refined_f, x_f)
                lpips_loss = p_losses.get("lpips", x.new_tensor(0.0))
                dists_loss = p_losses.get("dists", x.new_tensor(0.0))
                refiner_residual_f = refined["refiner_residual"].float()
                refiner_tv = spatial_total_variation(refiner_residual_f)
                refiner_chroma = chroma_residual_l1(refiner_residual_f)
                loss = (
                    loss_l1 * l1
                    + loss_ms_ssim * ms
                    + loss_lpips * lpips_loss
                    + loss_dists * dists_loss
                    + loss_anchor * anchor
                    + loss_refiner_tv * refiner_tv
                    + loss_refiner_chroma * refiner_chroma
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage 4 refiner loss at step {step}: {loss}")
            scaler.scale(loss).backward()
            grad_norm = None
            if args.grad_clip_norm > 0:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at step {step}: {grad_norm}")
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                last_metrics = {
                    "loss_total": float(loss.detach().cpu()),
                    "loss_l1": float(l1.detach().cpu()),
                    "loss_ms_ssim": float(ms.detach().cpu()),
                    "loss_lpips": float(lpips_loss.detach().cpu()),
                    "loss_dists": float(dists_loss.detach().cpu()),
                    "loss_anchor_stage3": float(anchor.detach().cpu()),
                    "loss_refiner_tv": float(refiner_tv.detach().cpu()),
                    "loss_refiner_chroma": float(refiner_chroma.detach().cpu()),
                    "psnr_stage3": psnr(x, x_stage3),
                    "psnr_refined": psnr(x, x_refined),
                    "psnr_delta_refined_vs_stage3": psnr(x, x_refined) - psnr(x, x_stage3),
                    "l1_stage3": float(F.l1_loss(x_stage3, x).detach().cpu()),
                    "l1_refined": float(F.l1_loss(x_refined, x).detach().cpu()),
                    "refiner_residual_abs_mean": float(refined["refiner_residual"].abs().mean().detach().cpu()),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "step": float(step),
                }
                if grad_norm is not None:
                    last_metrics["grad_norm"] = float(grad_norm.detach().cpu())

            if step % args.log_every == 0 or step == 1:
                wandb.log(last_metrics, step=step)
                progress.set_postfix({k: round(v, 4) for k, v in last_metrics.items() if k != "step"})

            if step % args.sample_every == 0 or step == max_steps:
                sample = torch.cat(
                    [
                        x[:4].detach().cpu(),
                        x_sem[:4].detach().cpu(),
                        x_stage3[:4].detach().cpu(),
                        x_refined[:4].detach().cpu(),
                    ],
                    dim=0,
                )
                sample_path = out_dir / f"sample_step_{step:06d}.png"
                save_image(sample, sample_path, nrow=min(4, x.shape[0]))
                wandb.log({"samples": wandb.Image(str(sample_path))}, step=step)

            if checkpoint_every > 0 and step % checkpoint_every == 0 and step < max_steps:
                intermediate_path = ckpt_dir / f"{run_name}_step{step:06d}.pt"
                torch.save(checkpoint_payload(current_step=step, metrics=last_metrics), intermediate_path)
                intermediate_checkpoints.append(str(intermediate_path))
                wandb_run.summary[f"intermediate_checkpoint_step_{step}"] = str(intermediate_path)

            progress.update(1)
            if step >= max_steps:
                break
    progress.close()

    ckpt_path = ckpt_dir / f"{run_name}.pt"
    torch.save(checkpoint_payload(current_step=step, metrics=last_metrics), ckpt_path)
    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.time() - start,
        "checkpoint": str(ckpt_path),
        "intermediate_checkpoints": intermediate_checkpoints,
        "summary": last_metrics,
        "command": " ".join([sys.executable, *sys.argv]),
        "payload_policy": "fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    write_run_doc(
        Path("docs/experiments") / f"{run_name}.md",
        {
            "run_name": run_name,
            "date": payload["date"],
            "command": payload["command"],
            "summary": {**last_metrics, "payload_policy": payload["payload_policy"]},
            "artifacts": {
                "checkpoint": str(ckpt_path),
                "intermediate_checkpoints": intermediate_checkpoints,
                "output_dir": str(out_dir),
                "summary": str(summary_path),
            },
        },
    )
    wandb_run.summary.update(last_metrics)
    wandb_run.summary["intermediate_checkpoints"] = intermediate_checkpoints
    wandb_run.finish()
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
