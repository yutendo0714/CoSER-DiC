from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from PIL import Image
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from coserdic.datasets.image_folder import IMAGE_SUFFIXES
from coserdic.models import SemanticVQAutoEncoder, SemanticVQConfig
from coserdic.utils import seed_everything


class CenterCropImageDataset(Dataset):
    def __init__(self, roots: list[str], crop_size: int, limit: int = 0) -> None:
        self.paths: list[Path] = []
        for root in roots:
            path = Path(root)
            self.paths.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no images found under {roots}")
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image = Image.open(self.paths[index]).convert("RGB")
        tensor = TF.to_tensor(image)
        _, h, w = tensor.shape
        if h < self.crop_size or w < self.crop_size:
            scale = self.crop_size / min(h, w)
            new_h = max(self.crop_size, int(round(h * scale)))
            new_w = max(self.crop_size, int(round(w * scale)))
            tensor = TF.resize(tensor, [new_h, new_w], antialias=True)
            _, h, w = tensor.shape
        top = max(0, (h - self.crop_size) // 2)
        left = max(0, (w - self.crop_size) // 2)
        tensor = TF.crop(tensor, top, left, self.crop_size, self.crop_size)
        return {"image": tensor, "path": str(self.paths[index])}


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((x - y).pow(2), dim=(1, 2, 3)).clamp_min(1.0e-12)
    return -10.0 * torch.log10(mse)


def tensor_stats(x: torch.Tensor, prefix: str) -> dict[str, float]:
    x = x.detach().float()
    return {
        f"{prefix}_mean": float(x.mean().cpu()),
        f"{prefix}_std": float(x.std(unbiased=False).cpu()),
        f"{prefix}_min": float(x.min().cpu()),
        f"{prefix}_max": float(x.max().cpu()),
    }


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


def load_config_from_checkpoint(checkpoint: dict, fallback_config: str | None) -> dict:
    if "config" in checkpoint:
        return checkpoint["config"]
    if fallback_config is None:
        raise KeyError("checkpoint has no raw config; pass --config")
    return yaml.safe_load(Path(fallback_config).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--output-dir", default="results/analysis/stage1_semantic_vq")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--quantize-mix", type=float, default=1.0)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before analysis.")
    if not 0.0 <= args.quantize_mix <= 1.0:
        raise ValueError("--quantize-mix must be in [0, 1]")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = load_config_from_checkpoint(checkpoint, args.config or None)
    seed_everything(int(raw.get("experiment", {}).get("seed", 42)))

    cfg_payload = checkpoint.get("semantic_vq_config")
    if cfg_payload is None:
        raise KeyError("checkpoint has no semantic_vq_config")
    cfg = SemanticVQConfig(**cfg_payload)
    model = SemanticVQAutoEncoder(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    roots = args.image_root or raw.get("data", {}).get("train_roots")
    if not roots:
        raise ValueError("no analysis roots found; pass --image-root")
    dataset = CenterCropImageDataset(roots, crop_size=args.crop_size, limit=args.max_images)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    run_name = args.run_name or f"{checkpoint_path.stem}_analysis"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    import wandb

    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config={
            "checkpoint": str(checkpoint_path),
            "semantic_vq_config": cfg.__dict__,
            "roots": roots,
            "max_images": args.max_images,
            "crop_size": args.crop_size,
            "quantize_mix": args.quantize_mix,
        },
    )

    code_hist = torch.zeros(cfg.codebook_size, dtype=torch.long)
    psnr_values: list[float] = []
    l1_values: list[float] = []
    ms_ssim_values: list[float] = []
    per_image_used: list[int] = []
    soft_perplexities: list[float] = []
    assignment_sample_entropy_bits: list[float] = []
    assignment_avg_entropy_bits: list[float] = []
    soft_usage_entropy_bits: list[float] = []
    batch_used_codes: list[float] = []
    batch_dead_code_ratios: list[float] = []
    commitment_losses: list[float] = []
    codebook_losses: list[float] = []
    semantic_token_counts: list[float] = []
    latent_stats: list[dict[str, float]] = []
    quant_stats: list[dict[str, float]] = []
    vq_errors: list[float] = []
    sample_written = False

    with torch.no_grad():
        for batch in tqdm(loader, desc=run_name):
            x = batch["image"].to(device, non_blocking=True)
            out = model(x, quantize_mix=args.quantize_mix)
            x_sem = out["x_sem"].clamp(0, 1)
            indices = out["indices"].detach().cpu()
            code_hist.scatter_add_(0, indices.reshape(-1), torch.ones(indices.numel(), dtype=torch.long))

            psnr_values.extend(float(v) for v in psnr(x, x_sem).detach().cpu())
            l1_batch = torch.mean(torch.abs(x_sem - x), dim=(1, 2, 3)).detach().cpu()
            l1_values.extend(float(v) for v in l1_batch)
            ms_batch = ms_ssim(x_sem, x, data_range=1.0, size_average=False).detach().cpu()
            ms_ssim_values.extend(float(v) for v in ms_batch)
            per_image_used.extend(int(torch.unique(sample).numel()) for sample in indices)
            soft_perplexities.append(float(out["soft_perplexity"].detach().cpu()))
            assignment_sample_entropy_bits.append(float(out["assignment_sample_entropy_bits"].detach().cpu()))
            assignment_avg_entropy_bits.append(float(out["assignment_avg_entropy_bits"].detach().cpu()))
            soft_usage_entropy_bits.append(float(out["soft_usage_entropy_bits"].detach().cpu()))
            batch_used_codes.append(float(out["used_codes"].detach().cpu()))
            batch_dead_code_ratios.append(float(out["dead_code_ratio"].detach().cpu()))
            commitment_losses.append(float(out["commitment_loss"].detach().cpu()))
            codebook_losses.append(float(out["codebook_loss"].detach().cpu()))
            semantic_token_counts.extend(float(sample.numel()) for sample in indices)
            latent_stats.append(tensor_stats(out["h"], "latent"))
            quant_stats.append(tensor_stats(out["quantized"], "quantized"))
            vq_errors.append(float(torch.mean((out["h"].float() - out["quantized"].float()).pow(2)).detach().cpu()))

            if not sample_written:
                sample = torch.cat([x[:4].detach().cpu(), x_sem[:4].detach().cpu()], dim=0)
                sample_path = out_dir / "analysis_recon_grid.png"
                save_image(sample, sample_path, nrow=min(4, x.shape[0]))
                wandb.log({"analysis_recon_grid": wandb.Image(str(sample_path))})
                sample_written = True

    total_tokens = int(code_hist.sum().item())
    probs = code_hist.float() / max(total_tokens, 1)
    nonzero_probs = probs[probs > 0]
    global_entropy = float(-(nonzero_probs * torch.log2(nonzero_probs)).sum().item())
    global_perplexity = float(2.0**global_entropy)
    active_codes = int(torch.count_nonzero(code_hist).item())
    topk = min(32, cfg.codebook_size)
    top_counts, top_indices = torch.topk(code_hist, k=topk)

    def mean(values: list[float]) -> float:
        return float(sum(values) / max(len(values), 1))

    def percentile(values: list[float], pct: float) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
        return float(ordered[index])

    merged_latent_stats = {
        key: mean([stats[key] for stats in latent_stats])
        for key in latent_stats[0].keys()
    }
    merged_quant_stats = {
        key: mean([stats[key] for stats in quant_stats])
        for key in quant_stats[0].keys()
    }
    summary = {
        "checkpoint": str(checkpoint_path),
        "quantize_mix": args.quantize_mix,
        "num_images": len(dataset),
        "total_tokens": total_tokens,
        "semantic_tokens_per_image": int(round(mean(semantic_token_counts))),
        "semantic_tokens_per_image_min": int(min(semantic_token_counts)) if semantic_token_counts else 0,
        "semantic_tokens_per_image_max": int(max(semantic_token_counts)) if semantic_token_counts else 0,
        "active_codes_global": active_codes,
        "active_code_ratio_global": active_codes / float(cfg.codebook_size),
        "dead_code_ratio_global": 1.0 - active_codes / float(cfg.codebook_size),
        "global_code_entropy_bits": global_entropy,
        "global_code_perplexity": global_perplexity,
        "mean_per_image_used_codes": mean([float(v) for v in per_image_used]),
        "p50_per_image_used_codes": percentile([float(v) for v in per_image_used], 50),
        "max_per_image_used_codes": max(per_image_used) if per_image_used else 0,
        "mean_soft_perplexity_batch": mean(soft_perplexities),
        "mean_assignment_sample_entropy_bits": mean(assignment_sample_entropy_bits),
        "mean_assignment_avg_entropy_bits_batch": mean(assignment_avg_entropy_bits),
        "mean_soft_usage_entropy_bits": mean(soft_usage_entropy_bits),
        "mean_batch_used_codes": mean(batch_used_codes),
        "mean_batch_dead_code_ratio": mean(batch_dead_code_ratios),
        "mean_commitment_mse": mean(commitment_losses),
        "mean_codebook_mse": mean(codebook_losses),
        "mean_psnr_sem": mean(psnr_values),
        "mean_l1_sem": mean(l1_values),
        "mean_ms_ssim_sem": mean(ms_ssim_values),
        "mean_vq_mse": mean(vq_errors),
        **merged_latent_stats,
        **merged_quant_stats,
    }
    if not math.isfinite(summary["global_code_perplexity"]):
        raise FloatingPointError(summary)

    summary_path = out_dir / "analysis.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    top_codes_path = out_dir / "top_codes.csv"
    with top_codes_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "code", "count", "probability"])
        for rank, (code, count) in enumerate(zip(top_indices.tolist(), top_counts.tolist()), start=1):
            writer.writerow([rank, code, count, count / max(total_tokens, 1)])

    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "summary": summary,
        "artifacts": {
            "analysis": str(summary_path),
            "top_codes": str(top_codes_path),
            "output_dir": str(out_dir),
        },
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    wandb_run.summary.update(summary)
    wandb_run.finish()
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
