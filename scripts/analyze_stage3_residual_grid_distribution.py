from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
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
        self.crop_size = int(crop_size)

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


def entropy_bits_from_counts(counts: torch.Tensor) -> float:
    total = counts.sum().float().clamp_min(1.0)
    probs = counts.float() / total
    nonzero = probs[probs > 0]
    return float((-(nonzero * torch.log2(nonzero))).sum().item())


def normalize_image_map(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().float().cpu()
    y = y - y.min()
    y = y / y.max().clamp_min(1.0e-12)
    return y


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
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--output-dir", default="results/analysis/stage3_residual_grid_distribution")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--detail-downsample-factor", type=int, default=32)
    parser.add_argument("--detail-bits", type=int, default=5)
    parser.add_argument("--detail-range", type=float, default=0.25)
    parser.add_argument("--detail-prior", default="")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 3 analysis.")
    if args.crop_size % args.detail_downsample_factor != 0:
        raise ValueError("--crop-size must be divisible by --detail-downsample-factor")

    roots = args.image_root or ["/dpl/kodak"]
    dataset = CenterCropImageDataset(roots, crop_size=args.crop_size, limit=args.max_images)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = SemanticVQConfig(**checkpoint["semantic_vq_config"])
    model = SemanticVQAutoEncoder(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.detail_prior:
        payload = json.loads(Path(args.detail_prior).read_text())
        if payload.get("codec") == "static_residual_grid_position_huffman":
            residual_code = StaticResidualGridPositionHuffmanCode.from_dict(payload)
        else:
            residual_code = StaticResidualGridHuffmanCode.from_dict(payload)
        if int(args.detail_bits) != residual_code.bits:
            raise ValueError("--detail-bits does not match residual Huffman prior")
        if abs(float(args.detail_range) - residual_code.value_range) > 1.0e-9:
            raise ValueError("--detail-range does not match residual Huffman prior")
    else:
        residual_code = UniformResidualGridCode(bits=args.detail_bits, value_range=args.detail_range, codec="fixed_bits")

    detail_hw = args.crop_size // args.detail_downsample_factor
    levels = 1 << int(args.detail_bits)
    counts = torch.zeros(levels, dtype=torch.long)
    channel_counts = torch.zeros(3, levels, dtype=torch.long)
    position_counts = torch.zeros(3, detail_hw, detail_hw, levels, dtype=torch.long)
    abs_sum = torch.zeros(3, detail_hw, detail_hw, dtype=torch.float64)
    sq_sum = torch.zeros(3, detail_hw, detail_hw, dtype=torch.float64)
    clip_sum = torch.zeros(3, detail_hw, detail_hw, dtype=torch.float64)
    total_images = 0

    with torch.no_grad():
        for x_cpu in tqdm(loader, desc="analyze residual grid"):
            x = x_cpu.to(device, non_blocking=True)
            semantic = model(x, quantize_mix=1.0)
            residual = x - semantic["x_sem"]
            residual_grid = F.adaptive_avg_pool2d(residual, output_size=(detail_hw, detail_hw)).detach().cpu()
            codes = residual_code.quantize(residual_grid)
            total_images += int(x.shape[0])
            counts += torch.bincount(codes.reshape(-1), minlength=levels).to(torch.long)
            for channel in range(3):
                channel_codes = codes[:, channel]
                channel_counts[channel] += torch.bincount(channel_codes.reshape(-1), minlength=levels).to(torch.long)
                for y in range(detail_hw):
                    for x_pos in range(detail_hw):
                        position_counts[channel, y, x_pos] += torch.bincount(
                            channel_codes[:, y, x_pos].reshape(-1),
                            minlength=levels,
                        ).to(torch.long)
            abs_sum += residual_grid.abs().sum(dim=0).double()
            sq_sum += residual_grid.float().pow(2).sum(dim=0).double()
            clip_sum += (residual_grid.abs() >= float(args.detail_range)).float().sum(dim=0).double()

    denom = float(max(total_images, 1))
    abs_mean = abs_sum / denom
    rms = torch.sqrt((sq_sum / denom).clamp_min(0.0))
    clip_ratio = clip_sum / denom
    channel_abs_mean = abs_mean.mean(dim=(1, 2))
    channel_rms = rms.mean(dim=(1, 2))
    channel_clip = clip_ratio.mean(dim=(1, 2))
    position_entropy = torch.zeros(3, detail_hw, detail_hw, dtype=torch.float32)
    for channel in range(3):
        for y in range(detail_hw):
            for x_pos in range(detail_hw):
                position_entropy[channel, y, x_pos] = entropy_bits_from_counts(position_counts[channel, y, x_pos])

    symbol_probs = (counts.float() / counts.sum().float().clamp_min(1.0)).tolist()
    most_common = sorted(
        ((index, int(count), float(symbol_probs[index])) for index, count in enumerate(counts.tolist())),
        key=lambda item: item[1],
        reverse=True,
    )[:8]
    least_common_nonzero = sorted(
        ((index, int(count), float(symbol_probs[index])) for index, count in enumerate(counts.tolist()) if count > 0),
        key=lambda item: item[1],
    )[:8]
    channel_entropy = [entropy_bits_from_counts(channel_counts[channel]) for channel in range(3)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{timestamp}_stage3_residual_grid_distribution_d{args.detail_downsample_factor}_b{args.detail_bits}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    abs_map = normalize_image_map(abs_mean.mean(dim=0)).unsqueeze(0).repeat(3, 1, 1)
    rms_map = normalize_image_map(rms.mean(dim=0)).unsqueeze(0).repeat(3, 1, 1)
    entropy_map = normalize_image_map(position_entropy.mean(dim=0)).unsqueeze(0).repeat(3, 1, 1)
    clip_map = normalize_image_map(clip_ratio.mean(dim=0)).unsqueeze(0).repeat(3, 1, 1)
    save_image(abs_map, out_dir / "abs_mean_map.png")
    save_image(rms_map, out_dir / "rms_map.png")
    save_image(entropy_map, out_dir / "entropy_map.png")
    save_image(clip_map, out_dir / "clip_ratio_map.png")

    detail_symbols = int(counts.sum().item())
    mean_huffman_bits = None
    if isinstance(residual_code, StaticResidualGridHuffmanCode):
        mean_huffman_bits = float(
            sum(int(counts[index].item()) * residual_code.huffman.code_lengths[index] for index in range(levels))
            / max(detail_symbols, 1)
        )
    elif isinstance(residual_code, StaticResidualGridPositionHuffmanCode):
        mean_huffman_bits = 0.0
        for channel in range(3):
            for y in range(detail_hw):
                for x_pos in range(detail_hw):
                    position = channel * detail_hw * detail_hw + y * detail_hw + x_pos
                    codebook = residual_code.position_codes[position]
                    mean_huffman_bits += sum(
                        int(c) * int(l)
                        for c, l in zip(position_counts[channel, y, x_pos].tolist(), codebook.code_lengths)
                    )
        mean_huffman_bits = float(mean_huffman_bits / max(detail_symbols, 1))

    summary = {
        "checkpoint": str(checkpoint_path),
        "image_roots": roots,
        "num_images": total_images,
        "crop_size": int(args.crop_size),
        "detail_downsample_factor": int(args.detail_downsample_factor),
        "detail_shape": [3, detail_hw, detail_hw],
        "detail_bits": int(args.detail_bits),
        "detail_range": float(args.detail_range),
        "detail_prior": str(args.detail_prior),
        "total_symbols": detail_symbols,
        "symbol_entropy_bits": entropy_bits_from_counts(counts),
        "mean_huffman_bits_per_symbol": mean_huffman_bits,
        "symbol_counts": [int(x) for x in counts.tolist()],
        "symbol_probs": symbol_probs,
        "most_common_symbols": most_common,
        "least_common_nonzero_symbols": least_common_nonzero,
        "channel_entropy_bits": channel_entropy,
        "channel_abs_mean": [float(x) for x in channel_abs_mean.tolist()],
        "channel_rms": [float(x) for x in channel_rms.tolist()],
        "channel_clip_ratio": [float(x) for x in channel_clip.tolist()],
        "global_abs_mean": float(abs_mean.mean().item()),
        "global_rms": float(rms.mean().item()),
        "global_clip_ratio": float(clip_ratio.mean().item()),
        "position_entropy_mean": float(position_entropy.mean().item()),
        "position_entropy_max": float(position_entropy.max().item()),
        "position_abs_mean_max": float(abs_mean.max().item()),
        "position_rms_max": float(rms.max().item()),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    torch.save(
        {
            "counts": counts,
            "channel_counts": channel_counts,
            "position_counts": position_counts,
            "abs_mean": abs_mean.float(),
            "rms": rms.float(),
            "clip_ratio": clip_ratio.float(),
            "position_entropy": position_entropy,
        },
        out_dir / "diagnostics.pt",
    )

    import wandb

    run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )
    run.summary.update({k: v for k, v in summary.items() if k not in {"symbol_counts", "symbol_probs"}})
    wandb.log(
        {
            "abs_mean_map": wandb.Image(str(out_dir / "abs_mean_map.png")),
            "rms_map": wandb.Image(str(out_dir / "rms_map.png")),
            "entropy_map": wandb.Image(str(out_dir / "entropy_map.png")),
            "clip_ratio_map": wandb.Image(str(out_dir / "clip_ratio_map.png")),
        }
    )
    run.finish()

    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "summary": summary,
        "artifacts": {
            "summary": str(summary_path),
            "diagnostics": str(out_dir / "diagnostics.pt"),
            "abs_mean_map": str(out_dir / "abs_mean_map.png"),
            "rms_map": str(out_dir / "rms_map.png"),
            "entropy_map": str(out_dir / "entropy_map.png"),
            "clip_ratio_map": str(out_dir / "clip_ratio_map.png"),
        },
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
