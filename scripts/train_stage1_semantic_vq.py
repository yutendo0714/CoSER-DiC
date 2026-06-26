from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from coserdic.datasets.image_folder import IMAGE_SUFFIXES
from coserdic.losses import Stage1LossWeights, Stage1SemanticVQLoss
from coserdic.models import SemanticVQAutoEncoder, SemanticVQConfig
from coserdic.utils import seed_everything


class RandomCropImageDataset(Dataset):
    def __init__(self, roots: list[str], crop_size: int, random_flip: bool = True, limit: int = 0) -> None:
        self.paths: list[Path] = []
        for root in roots:
            path = Path(root)
            self.paths.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no images found under {roots}")
        self.crop_size = crop_size
        self.random_flip = random_flip

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        tensor = TF.to_tensor(image)
        _, h, w = tensor.shape
        if h < self.crop_size or w < self.crop_size:
            scale = self.crop_size / min(h, w)
            new_h = max(self.crop_size, int(round(h * scale)))
            new_w = max(self.crop_size, int(round(w * scale)))
            tensor = TF.resize(tensor, [new_h, new_w], antialias=True)
            _, h, w = tensor.shape

        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        tensor = TF.crop(tensor, top, left, self.crop_size, self.crop_size)
        if self.random_flip and random.random() < 0.5:
            tensor = TF.hflip(tensor)
        return tensor


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y).pow(2)).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * torch.log10(torch.tensor(mse)).item()


def linear_warmup_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, float(step) / float(warmup_steps)))


def make_config(
    raw: dict,
    debug_small: bool,
    codebook_size: int | None = None,
    ema_update: bool | None = None,
) -> SemanticVQConfig:
    model = raw["model"]
    enc = model["semantic_encoder"]
    vq = model["vq"]
    dec = model["semantic_aux_decoder"]
    if debug_small:
        return SemanticVQConfig(
            base_channels=32,
            latent_channels=64,
            codebook_size=codebook_size or 128,
            commitment_weight=float(vq["commitment_weight"]),
            ema_update=bool(vq["ema_update"]) if ema_update is None else ema_update,
            num_res_blocks=1,
            soft_st=bool(vq.get("soft_st", True)),
            soft_st_temperature=float(vq.get("soft_st_temperature", 1.0)),
        )
    return SemanticVQConfig(
        base_channels=int(enc["base_channels"]),
        latent_channels=int(enc["output_channels"]),
        codebook_size=codebook_size or int(vq["codebook_size"]),
        commitment_weight=float(vq["commitment_weight"]),
        ema_update=bool(vq["ema_update"]) if ema_update is None else ema_update,
        num_res_blocks=int(dec["num_res_blocks"]),
        soft_st=bool(vq.get("soft_st", True)),
        soft_st_temperature=float(vq.get("soft_st_temperature", 1.0)),
    )


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
    parser.add_argument("--config", default="configs/train/train_stage1_semantic_vq.yaml")
    parser.add_argument("--train-root", action="append", default=None)
    parser.add_argument("--output-dir", default="outputs/stage1_semantic_vq")
    parser.add_argument("--checkpoint-dir", default="checkpoints/stage1_semantic_vq")
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--debug-small", action="store_true")
    parser.add_argument("--codebook-size", type=int, default=0)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--quantize-warmup-steps", type=int, default=-1)
    parser.add_argument("--vq-warmup-steps", type=int, default=-1)
    parser.add_argument("--usage-warmup-steps", type=int, default=-1)
    parser.add_argument("--disable-lpips", action="store_true")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    seed_everything(int(raw["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before training.")

    train_roots = args.train_root or raw["data"].get("train_roots", ["/dpl/race_pilot_openimages_crops"])
    crop_size = int(raw["data"]["train_crop_size"])
    dataset = RandomCropImageDataset(
        train_roots,
        crop_size=crop_size,
        random_flip=bool(raw["data"]["random_flip"]),
        limit=args.limit_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    cfg = make_config(
        raw,
        args.debug_small,
        codebook_size=args.codebook_size or None,
        ema_update=False if args.no_ema else None,
    )
    model = SemanticVQAutoEncoder(cfg).to(device)
    weights = Stage1LossWeights(
        l1_sem=float(raw["loss"]["l1_sem"]),
        ms_ssim_sem=float(raw["loss"]["ms_ssim_sem"]),
        lpips_sem=0.0 if args.disable_lpips else float(raw["loss"]["lpips_sem"]),
        vq=1.0,
        codebook_usage=float(raw["loss"]["codebook_usage"]),
    )
    loss_fn = Stage1SemanticVQLoss(weights, use_lpips=not args.disable_lpips).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(raw["optimization"]["lr"]),
        weight_decay=float(raw["optimization"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(raw["optimization"]["mixed_precision"]))
    quantize_warmup_steps = (
        args.quantize_warmup_steps
        if args.quantize_warmup_steps >= 0
        else int(raw["optimization"].get("quantize_mix_warmup_steps", 0))
    )
    vq_warmup_steps = (
        args.vq_warmup_steps
        if args.vq_warmup_steps >= 0
        else int(raw["loss"].get("vq_warmup_steps", 0))
    )
    usage_warmup_steps = (
        args.usage_warmup_steps
        if args.usage_warmup_steps >= 0
        else int(raw["loss"].get("usage_warmup_steps", 0))
    )

    import wandb

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{timestamp}_stage1_semantic_vq_smoke"
    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config={
            "config": raw,
            "debug_small": args.debug_small,
            "train_roots": train_roots,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "model": cfg.__dict__,
            "quantize_warmup_steps": quantize_warmup_steps,
            "vq_warmup_steps": vq_warmup_steps,
            "usage_warmup_steps": usage_warmup_steps,
        },
    )

    out_dir = Path(args.output_dir) / run_name
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    last_metrics: dict[str, float] = {}
    start = time.time()
    progress = tqdm(total=args.max_steps, desc=run_name)
    while step < args.max_steps:
        for x in loader:
            step += 1
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            quantize_mix = linear_warmup_scale(step, quantize_warmup_steps)
            vq_scale = linear_warmup_scale(step, vq_warmup_steps)
            usage_scale = linear_warmup_scale(step, usage_warmup_steps)
            with torch.amp.autocast("cuda", enabled=bool(raw["optimization"]["mixed_precision"])):
                out = model(x, quantize_mix=quantize_mix)
                losses = loss_fn(x, out, vq_scale=vq_scale, usage_scale=usage_scale)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"non-finite Stage 1 loss at step {step}: {losses}")
            scaler.scale(losses["total"]).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                x_sem = out["x_sem"].clamp(0, 1)
                last_metrics = {
                    "loss_total": float(losses["total"].detach().cpu()),
                    "loss_l1_sem": float(losses["l1_sem"].cpu()),
                    "loss_ms_ssim_sem": float(losses["ms_ssim_sem"].cpu()),
                    "loss_vq": float(losses["vq"].cpu()),
                    "perplexity": float(out["perplexity"].cpu()),
                    "soft_perplexity": float(out["soft_perplexity"].cpu()),
                    "dead_code_ratio": float(out["dead_code_ratio"].cpu()),
                    "used_codes": float(out["used_codes"].cpu()),
                    "usage_loss": float(out["usage_loss"].detach().cpu()),
                    "quantize_mix": quantize_mix,
                    "vq_scale": vq_scale,
                    "usage_scale": usage_scale,
                    "psnr_sem": psnr(x, x_sem),
                    "lr": opt.param_groups[0]["lr"],
                    "step": step,
                }

            if step % args.log_every == 0 or step == 1:
                wandb.log(last_metrics, step=step)
                progress.set_postfix({k: round(v, 4) for k, v in last_metrics.items() if k != "step"})

            if step % args.sample_every == 0 or step == args.max_steps:
                sample = torch.cat([x[:4].detach().cpu(), x_sem[:4].detach().cpu()], dim=0)
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
            "model": model.state_dict(),
            "config": raw,
            "semantic_vq_config": cfg.__dict__,
            "step": step,
            "metrics": last_metrics,
        },
        ckpt_path,
    )
    summary_path = out_dir / "summary.json"
    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.time() - start,
        "checkpoint": str(ckpt_path),
        "summary": last_metrics,
        "command": " ".join([sys.executable, *sys.argv]),
    }
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
