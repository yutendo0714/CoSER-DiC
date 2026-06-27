from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from coserdic.datasets.image_folder import IMAGE_SUFFIXES
from coserdic.entropy import (
    StaticANSCode,
    StaticHuffmanCode,
    StaticLeftContextHuffmanCode,
    StaticPositionHuffmanCode,
    semantic_bits_per_token,
)
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


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


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
    parser.add_argument("--output-dir", default="outputs/stage2_semantic_entropy")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoothing-count", type=int, default=1)
    parser.add_argument("--prior-kind", choices=["global", "ans", "position", "left_context"], default="global")
    parser.add_argument("--position-backoff-mass", type=float, default=0.0)
    parser.add_argument("--context-topk", type=int, default=64)
    parser.add_argument("--context-backoff-mass", type=float, default=256.0)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before fitting Stage 2 prior.")

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
        raise ValueError("no calibration roots found; pass --image-root")
    dataset = CenterCropImageDataset(roots, crop_size=args.crop_size, limit=args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    code_hist = torch.zeros(cfg.codebook_size, dtype=torch.long)
    position_hist: torch.Tensor | None = None
    samples: list[torch.Tensor] = []
    with torch.no_grad():
        for x in tqdm(loader, desc="fit_static_huffman"):
            out = model(x.to(device, non_blocking=True), quantize_mix=1.0)
            indices = out["indices"].detach().cpu()
            code_hist.scatter_add_(0, indices.reshape(-1), torch.ones(indices.numel(), dtype=torch.long))
            if position_hist is None:
                token_h, token_w = (int(v) for v in indices.shape[-2:])
                position_hist = torch.zeros(token_h, token_w, cfg.codebook_size, dtype=torch.long)
            if tuple(indices.shape[-2:]) != tuple(position_hist.shape[:2]):
                raise ValueError("all calibration samples must share the same semantic token shape")
            for sample in indices:
                for y in range(sample.shape[0]):
                    for x_pos in range(sample.shape[1]):
                        position_hist[y, x_pos, int(sample[y, x_pos].item())] += 1
            samples.extend(sample.clone() for sample in indices)

    if position_hist is None or not samples:
        raise RuntimeError("no semantic token samples collected")
    if args.prior_kind == "position":
        if args.position_backoff_mass > 0:
            code = StaticPositionHuffmanCode.from_position_counts_with_global_backoff(
                position_hist,
                code_hist,
                backoff_mass=args.position_backoff_mass,
                smoothing_count=float(args.smoothing_count),
            )
        else:
            code = StaticPositionHuffmanCode.from_counts(position_hist, smoothing_count=args.smoothing_count)
        prior_filename = "static_position_huffman_prior.json"
        huffman_bits = [float(code.encoded_bits(sample)) for sample in samples]
        huffman_payload_bytes = [float(len(code.encode(sample))) for sample in samples]
        prior_payload = {
            **code.to_dict(),
            "position_counts": [
                [
                    [int(v) for v in position_hist[y, x].tolist()]
                    for x in range(position_hist.shape[1])
                ]
                for y in range(position_hist.shape[0])
            ],
        }
    elif args.prior_kind == "left_context":
        code = StaticLeftContextHuffmanCode.from_samples(
            samples,
            codebook_size=cfg.codebook_size,
            context_topk=args.context_topk,
            backoff_mass=args.context_backoff_mass,
            smoothing_count=float(args.smoothing_count),
        )
        prior_filename = "static_left_context_huffman_prior.json"
        huffman_bits = [float(code.encoded_bits(sample)) for sample in samples]
        huffman_payload_bytes = [float(len(code.encode(sample))) for sample in samples]
        prior_payload = code.to_dict()
    elif args.prior_kind == "ans":
        code = StaticANSCode.from_counts(
            code_hist,
            smoothing_count=float(args.smoothing_count),
        )
        prior_filename = "static_ans_prior.json"
        huffman_bits = [float(code.encoded_bits(sample)) for sample in samples]
        huffman_payload_bytes = [float(len(code.encode(sample))) for sample in samples]
        prior_payload = {**code.to_dict(), "counts": [int(v) for v in code_hist.tolist()]}
    else:
        code = StaticHuffmanCode.from_counts(code_hist, smoothing_count=args.smoothing_count)
        prior_filename = "static_huffman_prior.json"
        huffman_bits = [float(code.encoded_bits(sample)) for sample in samples]
        huffman_payload_bytes = [float(len(code.encode(sample))) for sample in samples]
        prior_payload = {**code.to_dict(), "counts": [int(v) for v in code_hist.tolist()]}

    token_count = int(code_hist.sum().item())
    probs = code_hist.float() / max(token_count, 1)
    nonzero = probs[probs > 0]
    entropy_bits = float(-(nonzero * torch.log2(nonzero)).sum().item())
    fixed_bits = semantic_bits_per_token(cfg.codebook_size)
    pixels = float(args.crop_size * args.crop_size)

    run_name = args.run_name or f"{checkpoint_path.stem}_stage2_static_huffman_fit"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prior_path = out_dir / prior_filename
    prior_payload = {
        **prior_payload,
        "global_counts": [int(v) for v in code_hist.tolist()],
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "roots": roots,
            "crop_size": args.crop_size,
            "semantic_token_shape": [int(v) for v in samples[0].shape],
            "num_images": len(dataset),
            "smoothing_count": args.smoothing_count,
            "prior_kind": args.prior_kind,
            "position_backoff_mass": args.position_backoff_mass,
            "context_topk": args.context_topk,
            "context_backoff_mass": args.context_backoff_mass,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    prior_path.write_text(json.dumps(prior_payload, indent=2, allow_nan=False))

    summary = {
        "checkpoint": str(checkpoint_path),
        "num_images": len(dataset),
        "crop_size": args.crop_size,
        "codebook_size": cfg.codebook_size,
        "prior_kind": args.prior_kind,
        "position_backoff_mass": args.position_backoff_mass,
        "context_topk": args.context_topk,
        "context_backoff_mass": args.context_backoff_mass,
        "semantic_token_shape": [int(v) for v in samples[0].shape],
        "total_tokens": token_count,
        "active_codes": int(torch.count_nonzero(code_hist).item()),
        "global_entropy_bits": entropy_bits,
        "fixed_bits_per_token": fixed_bits,
        "mean_huffman_bits_per_token": mean(huffman_bits) / max(float(samples[0].numel()), 1.0),
        "mean_huffman_payload_bpp": 8.0 * mean(huffman_payload_bytes) / pixels,
        "fixed_bits_payload_bpp": fixed_bits * float(samples[0].numel()) / pixels,
        "mean_huffman_payload_bytes": mean(huffman_payload_bytes),
    }
    if args.prior_kind == "left_context":
        summary["left_context_top_tokens"] = len(code.top_tokens)
    summary["payload_bpp_delta_vs_fixed"] = summary["mean_huffman_payload_bpp"] - summary["fixed_bits_payload_bpp"]

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    import wandb

    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config={
            "checkpoint": str(checkpoint_path),
            "roots": roots,
            "smoothing_count": args.smoothing_count,
            "prior_kind": args.prior_kind,
            "position_backoff_mass": args.position_backoff_mass,
            "context_topk": args.context_topk,
            "context_backoff_mass": args.context_backoff_mass,
        },
    )
    wandb_run.summary.update(summary)
    wandb_run.finish()

    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "summary": summary,
        "artifacts": {"prior": str(prior_path), "summary": str(summary_path), "output_dir": str(out_dir)},
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
