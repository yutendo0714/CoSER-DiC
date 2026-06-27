from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from coserdic.datasets.image_folder import IMAGE_SUFFIXES
from coserdic.entropy import StaticResidualGridHuffmanCode, StaticResidualGridPositionHuffmanCode, UniformResidualGridCode
from coserdic.models import SemanticVQAutoEncoder, SemanticVQConfig
from coserdic.utils import seed_everything


class CenterCropImageDataset(Dataset):
    def __init__(self, roots: list[str], crop_size: int, limit: int = 0) -> None:
        self.paths: list[Path] = []
        for root in roots:
            self.paths.extend(sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no images found under {roots}")
        self.crop_size = crop_size

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
        top = max(0, (h - self.crop_size) // 2)
        left = max(0, (w - self.crop_size) // 2)
        return TF.crop(tensor, top, left, self.crop_size, self.crop_size)


def load_config_from_checkpoint(checkpoint: dict, fallback_config: str | None) -> dict:
    if "config" in checkpoint:
        return checkpoint["config"]
    if fallback_config is None:
        raise KeyError("checkpoint has no raw config; pass --config")
    return yaml.safe_load(Path(fallback_config).read_text())


def decode_indices_to_image(model: SemanticVQAutoEncoder, cfg: SemanticVQConfig, indices: torch.Tensor) -> torch.Tensor:
    flat = indices.to(model.vq.embedding.weight.device, dtype=torch.long).reshape(-1)
    embedding = model.vq.embedding.weight.float()
    if cfg.normalize_latent:
        embedding = F.normalize(embedding, dim=1) * math.sqrt(float(cfg.latent_channels))
    quant = embedding.index_select(0, flat)
    b, h, w = indices.shape
    quant = quant.view(b, h, w, cfg.latent_channels).permute(0, 3, 1, 2).contiguous()
    return model.decoder(quant.to(dtype=next(model.parameters()).dtype))


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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--output-dir", default="outputs/stage3_residual_entropy")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--detail-downsample-factor", type=int, default=32)
    parser.add_argument("--detail-bits", type=int, default=4)
    parser.add_argument("--detail-range", type=float, default=0.25)
    parser.add_argument("--coding-mode", choices=["global_huffman", "position_huffman"], default="global_huffman")
    parser.add_argument("--smoothing-count", type=int, default=1)
    parser.add_argument("--wandb-mode", default="offline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 3 prior fitting.")
    if args.crop_size % args.detail_downsample_factor != 0:
        raise ValueError("--crop-size must be divisible by --detail-downsample-factor")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = load_config_from_checkpoint(checkpoint, args.config or None)
    seed_everything(int(raw.get("experiment", {}).get("seed", 42)))
    cfg = SemanticVQConfig(**checkpoint["semantic_vq_config"])
    model = SemanticVQAutoEncoder(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    roots = args.image_root or raw.get("data", {}).get("train_roots")
    if not roots:
        raise ValueError("no calibration roots found; pass --image-root")
    dataset = CenterCropImageDataset(roots, crop_size=args.crop_size, limit=args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    detail_hw = args.crop_size // args.detail_downsample_factor
    quantizer = UniformResidualGridCode(bits=args.detail_bits, value_range=args.detail_range, codec="fixed_bits")
    counts = torch.zeros(quantizer.levels, dtype=torch.long)
    position_counts = torch.zeros(3, detail_hw, detail_hw, quantizer.levels, dtype=torch.long)
    total_symbols = 0
    residual_abs_sum = 0.0
    residual_sq_sum = 0.0
    clipped_symbols = 0

    with torch.no_grad():
        for x_cpu in tqdm(loader, desc="fit residual Huffman"):
            x = x_cpu.to(device, non_blocking=True)
            out = model(x, quantize_mix=1.0)
            indices_batch = out["indices"].detach().cpu()
            x_sem = decode_indices_to_image(model, cfg, indices_batch).clamp(0, 1)
            residual = x - x_sem
            residual_grid = F.adaptive_avg_pool2d(residual, output_size=(detail_hw, detail_hw)).detach().cpu()
            codes = quantizer.quantize(residual_grid)
            counts += torch.bincount(codes.reshape(-1), minlength=quantizer.levels)
            if args.coding_mode == "position_huffman":
                for channel in range(3):
                    for y in range(detail_hw):
                        for x_pos in range(detail_hw):
                            position_counts[channel, y, x_pos] += torch.bincount(
                                codes[:, channel, y, x_pos].reshape(-1),
                                minlength=quantizer.levels,
                            )
            total_symbols += int(codes.numel())
            residual_abs_sum += float(torch.sum(torch.abs(residual_grid)).item())
            residual_sq_sum += float(torch.sum(residual_grid.float().pow(2)).item())
            clipped_symbols += int(torch.count_nonzero(torch.abs(residual_grid) >= float(args.detail_range)).item())

    if args.coding_mode == "position_huffman":
        prior = StaticResidualGridPositionHuffmanCode.from_counts(
            position_counts,
            bits=args.detail_bits,
            value_range=args.detail_range,
            smoothing_count=args.smoothing_count,
        )
    else:
        prior = StaticResidualGridHuffmanCode.from_counts(
            counts,
            bits=args.detail_bits,
            value_range=args.detail_range,
            smoothing_count=args.smoothing_count,
        )

    probs = counts.float() / counts.sum().clamp_min(1)
    nonzero = probs[probs > 0]
    entropy_bits = float((-(nonzero * torch.log2(nonzero))).sum().item())
    if isinstance(prior, StaticResidualGridPositionHuffmanCode):
        mean_code_length = 0.0
        for channel in range(3):
            for y in range(detail_hw):
                for x_pos in range(detail_hw):
                    position = channel * detail_hw * detail_hw + y * detail_hw + x_pos
                    codebook = prior.position_codes[position]
                    mean_code_length += sum(
                        int(c) * int(l)
                        for c, l in zip(position_counts[channel, y, x_pos].tolist(), codebook.code_lengths)
                    )
        mean_code_length = float(mean_code_length / max(total_symbols, 1))
    else:
        mean_code_length = float(sum(int(c) * int(l) for c, l in zip(counts.tolist(), prior.huffman.code_lengths)) / max(total_symbols, 1))
    mean_residual_abs = float(residual_abs_sum / max(total_symbols, 1))
    residual_std = float(math.sqrt(max(residual_sq_sum / max(total_symbols, 1), 0.0)))
    clip_ratio = float(clipped_symbols / max(total_symbols, 1))

    run_name = args.run_name or (
        f"{checkpoint_path.stem}_stage3_residual_{args.coding_mode}_d{args.detail_downsample_factor}_b{args.detail_bits}_r{int(round(args.detail_range * 100)):03d}"
    )
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.coding_mode == "position_huffman":
        prior_path = out_dir / "static_residual_grid_position_huffman_prior.json"
    else:
        prior_path = out_dir / "static_residual_grid_huffman_prior.json"
    summary_path = out_dir / "summary.json"

    prior_payload = prior.to_dict()
    prior_payload.update(
        {
            "stage1_checkpoint": str(checkpoint_path),
            "calibration_roots": [str(v) for v in roots],
            "detail_downsample_factor": int(args.detail_downsample_factor),
            "detail_shape": [3, int(detail_hw), int(detail_hw)],
            "coding_mode": args.coding_mode,
            "smoothing_count": int(args.smoothing_count),
            "counts": [int(v) for v in counts.tolist()],
        }
    )
    if args.coding_mode == "position_huffman":
        prior_payload["position_counts"] = [
            [int(v) for v in position_counts[channel, y, x_pos].tolist()]
            for channel in range(3)
            for y in range(detail_hw)
            for x_pos in range(detail_hw)
        ]
    prior_path.write_text(json.dumps(prior_payload, indent=2, allow_nan=False))

    summary = {
        "checkpoint": str(checkpoint_path),
        "prior": str(prior_path),
        "num_images": len(dataset),
        "crop_size": int(args.crop_size),
        "detail_downsample_factor": int(args.detail_downsample_factor),
        "detail_shape": [3, int(detail_hw), int(detail_hw)],
        "detail_bits": int(args.detail_bits),
        "detail_range": float(args.detail_range),
        "coding_mode": args.coding_mode,
        "smoothing_count": int(args.smoothing_count),
        "total_symbols": int(total_symbols),
        "counts": [int(v) for v in counts.tolist()],
        "symbol_probs": [float(v) for v in probs.tolist()],
        "symbol_entropy_bits": entropy_bits,
        "mean_huffman_bits_per_symbol": mean_code_length,
        "fixed_bits_per_symbol": int(args.detail_bits),
        "mean_residual_abs": mean_residual_abs,
        "residual_std": residual_std,
        "clip_ratio": clip_ratio,
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    import wandb

    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )
    wandb_run.summary.update(summary)
    wandb_run.finish()

    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "summary": summary,
        "artifacts": {"prior": str(prior_path), "summary": str(summary_path)},
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
