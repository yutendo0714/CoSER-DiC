from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from coserdic.datasets.image_folder import IMAGE_SUFFIXES
from coserdic.models import ResidualDetailAutoEncoder, ResidualDetailConfig, SemanticVQAutoEncoder, SemanticVQConfig
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


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y).pow(2)).detach()
    if float(mse.cpu()) <= 0:
        return float("inf")
    return float((-10.0 * torch.log10(mse)).cpu())


def code_entropy_bits(codes: torch.Tensor, levels: int) -> float:
    counts = torch.bincount(codes.detach().cpu().reshape(-1), minlength=levels).float()
    probs = counts / counts.sum().clamp_min(1.0)
    nonzero = probs[probs > 0]
    return float((-(nonzero * torch.log2(nonzero))).sum().item())


def write_run_doc(path: Path, payload: dict) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/train_stage3_residual.yaml")
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--train-root", action="append", default=None)
    parser.add_argument("--output-dir", default="outputs/stage3_residual_autoencoder")
    parser.add_argument("--checkpoint-dir", default="checkpoints/stage3_residual_autoencoder")
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--detail-channels", type=int, default=3)
    parser.add_argument("--num-res-blocks", type=int, default=3)
    parser.add_argument("--quant-bits", type=int, default=0)
    parser.add_argument("--value-range", type=float, default=0.0)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--loss-l1", type=float, default=-1.0)
    parser.add_argument("--loss-ms-ssim", type=float, default=-1.0)
    parser.add_argument("--loss-rate-proxy", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    seed_everything(int(raw.get("experiment", {}).get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 3 training.")
    if args.crop_size % 32 != 0:
        raise ValueError("--crop-size must be divisible by 32")

    roots = args.train_root or raw.get("data", {}).get("train_roots") or [
        "/dpl/open-images-v6/train/data",
        "/dpl/div2k",
    ]
    dataset = RandomCropImageDataset(roots, crop_size=args.crop_size, random_flip=True, limit=args.limit_images)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    stage1_path = Path(args.stage1_checkpoint)
    stage1_checkpoint = torch.load(stage1_path, map_location="cpu", weights_only=False)
    semantic_cfg = SemanticVQConfig(**stage1_checkpoint["semantic_vq_config"])
    semantic_model = SemanticVQAutoEncoder(semantic_cfg).to(device)
    semantic_model.load_state_dict(stage1_checkpoint["model"])
    semantic_model.eval()
    for param in semantic_model.parameters():
        param.requires_grad_(False)

    bootstrap = raw.get("model", {}).get("bootstrap_residual", {})
    detail_cfg = ResidualDetailConfig(
        semantic_channels=int(semantic_cfg.latent_channels),
        base_channels=int(args.base_channels),
        hidden_channels=int(args.hidden_channels),
        detail_channels=int(args.detail_channels),
        num_res_blocks=int(args.num_res_blocks),
        quant_bits=int(args.quant_bits or bootstrap.get("bits", 5)),
        value_range=float(args.value_range or bootstrap.get("value_range", 0.25)),
        residual_scale=float(args.residual_scale),
    )
    detail_model = ResidualDetailAutoEncoder(detail_cfg).to(device)
    init_summary: dict[str, object] = {}
    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        load_result = detail_model.load_state_dict(init_checkpoint["model"], strict=False)
        init_summary = {
            "init_checkpoint": args.init_checkpoint,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }

    lr = args.lr if args.lr > 0 else float(raw.get("optimization", {}).get("lr_main", 1.0e-4))
    opt = torch.optim.AdamW(detail_model.parameters(), lr=lr, weight_decay=float(args.weight_decay))
    amp_enabled = bool(raw.get("optimization", {}).get("mixed_precision", True))
    if args.mixed_precision is not None:
        amp_enabled = bool(args.mixed_precision)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    l1_weight = args.loss_l1 if args.loss_l1 >= 0 else float(raw.get("loss", {}).get("l1_aux", 1.0))
    ms_weight = args.loss_ms_ssim if args.loss_ms_ssim >= 0 else float(raw.get("loss", {}).get("ms_ssim_aux", 0.0))

    import wandb

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{timestamp}_stage3_residual_ae_probe"
    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config={
            "config": raw,
            "stage1_checkpoint": str(stage1_path),
            "train_roots": roots,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "detail_model": detail_cfg.__dict__,
            "initialization": init_summary,
            "loss_l1": l1_weight,
            "loss_ms_ssim": ms_weight,
            "loss_rate_proxy": args.loss_rate_proxy,
            "mixed_precision": amp_enabled,
        },
    )

    out_dir = Path(args.output_dir) / run_name
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    start = time.time()
    last_metrics: dict[str, float] = {}
    progress = tqdm(total=args.max_steps, desc=run_name)
    detail_model.train()
    while step < args.max_steps:
        for x_cpu in loader:
            step += 1
            x = x_cpu.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                semantic = semantic_model(x, quantize_mix=1.0)
                x_sem = semantic["x_sem"].detach()
                s_hat = semantic["quantized"].detach()

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = detail_model(x, x_sem, s_hat, training_noise=True)
                x_aux = out["x_aux"]
                l1 = F.l1_loss(x_aux, x)
                if ms_weight > 0:
                    ms = 1.0 - ms_ssim(x_aux.float(), x.float(), data_range=1.0, size_average=True)
                else:
                    ms = x.new_tensor(0.0)
                rate_proxy = torch.mean(torch.abs(out["h_detail"]) / max(detail_cfg.value_range, 1.0e-6))
                loss = l1_weight * l1 + ms_weight * ms + float(args.loss_rate_proxy) * rate_proxy

            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage 3 residual loss at step {step}: {loss}")
            scaler.scale(loss).backward()
            grad_norm = None
            if args.grad_clip_norm > 0:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(detail_model.parameters(), args.grad_clip_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at step {step}: {grad_norm}")
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                detail_model.eval()
                eval_out = detail_model(x, x_sem, s_hat, training_noise=False)
                detail_model.train()
                h_detail = out["h_detail"].detach()
                codes = eval_out["detail_codes"].detach()
                last_metrics = {
                    "loss_total": float(loss.detach().cpu()),
                    "loss_l1_aux": float(l1.detach().cpu()),
                    "loss_ms_ssim_aux": float(ms.detach().cpu()),
                    "loss_rate_proxy": float(rate_proxy.detach().cpu()),
                    "psnr_sem": psnr(x, x_sem),
                    "psnr_aux": psnr(x, eval_out["x_aux"]),
                    "psnr_delta_aux_vs_sem": psnr(x, eval_out["x_aux"]) - psnr(x, x_sem),
                    "l1_sem": float(F.l1_loss(x_sem, x).detach().cpu()),
                    "l1_aux_eval": float(F.l1_loss(eval_out["x_aux"], x).detach().cpu()),
                    "ms_ssim_sem": float(ms_ssim(x_sem.float(), x.float(), data_range=1.0, size_average=True).detach().cpu()),
                    "ms_ssim_aux_eval": float(
                        ms_ssim(eval_out["x_aux"].float(), x.float(), data_range=1.0, size_average=True).detach().cpu()
                    ),
                    "h_detail_abs_mean": float(h_detail.abs().mean().cpu()),
                    "h_detail_std": float(h_detail.std(unbiased=False).cpu()),
                    "h_detail_clip_ratio": float((h_detail.abs() >= detail_cfg.value_range).float().mean().cpu()),
                    "detail_code_entropy_bits": code_entropy_bits(codes, detail_model.quantizer.levels),
                    "detail_code_min": float(codes.min().cpu()),
                    "detail_code_max": float(codes.max().cpu()),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "step": float(step),
                }
                if grad_norm is not None:
                    last_metrics["grad_norm"] = float(grad_norm.detach().cpu())

            if step % args.log_every == 0 or step == 1:
                wandb.log(last_metrics, step=step)
                progress.set_postfix({k: round(v, 4) for k, v in last_metrics.items() if k != "step"})

            if step % args.sample_every == 0 or step == args.max_steps:
                sample = torch.cat([x[:4].detach().cpu(), x_sem[:4].detach().cpu(), eval_out["x_aux"][:4].detach().cpu()], dim=0)
                sample_path = out_dir / f"sample_step_{step:06d}.png"
                save_image(sample, sample_path, nrow=min(4, x.shape[0]))
                wandb.log({"samples": wandb.Image(str(sample_path))}, step=step)

            progress.update(1)
            if step >= args.max_steps:
                break
    progress.close()

    ckpt_path = ckpt_dir / f"{run_name}.pt"
    torch.save(
        {
            "model": detail_model.state_dict(),
            "detail_config": detail_cfg.__dict__,
            "stage1_checkpoint": str(stage1_path),
            "semantic_vq_config": semantic_cfg.__dict__,
            "config": raw,
            "step": step,
            "metrics": last_metrics,
        },
        ckpt_path,
    )
    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.time() - start,
        "checkpoint": str(ckpt_path),
        "summary": last_metrics,
        "command": " ".join([sys.executable, *sys.argv]),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    write_run_doc(
        Path("docs/experiments") / f"{run_name}.md",
        {
            "run_name": run_name,
            "date": payload["date"],
            "command": payload["command"],
            "summary": last_metrics,
            "artifacts": {"checkpoint": str(ckpt_path), "output_dir": str(out_dir), "summary": str(summary_path)},
        },
    )
    wandb_run.summary.update(last_metrics)
    wandb_run.finish()
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
