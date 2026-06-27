from __future__ import annotations

import argparse
import json
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
from coserdic.models import SemanticVQAutoEncoder, SemanticVQConfig
from coserdic.utils import seed_everything


class CenterCropPathDataset(Dataset):
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
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
        return TF.crop(tensor, top, left, self.crop_size, self.crop_size), str(path)


def load_config_from_checkpoint(checkpoint: dict, fallback_config: str | None) -> dict:
    if "config" in checkpoint:
        return checkpoint["config"]
    if fallback_config is None:
        raise KeyError("checkpoint has no raw config; pass --config")
    return yaml.safe_load(Path(fallback_config).read_text())


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
    parser.add_argument("--output-dir", default="outputs/stage2_semantic_tokens")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before exporting tokens.")

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
        raise ValueError("no roots found; pass --image-root")
    dataset = CenterCropPathDataset(roots, crop_size=args.crop_size, limit=args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    token_batches: list[torch.Tensor] = []
    paths: list[str] = []
    code_hist = torch.zeros(cfg.codebook_size, dtype=torch.long)
    position_hist: torch.Tensor | None = None

    with torch.no_grad():
        for x_cpu, batch_paths in tqdm(loader, desc="export_stage1_tokens"):
            out = model(x_cpu.to(device, non_blocking=True), quantize_mix=1.0)
            indices = out["indices"].detach().cpu().to(torch.long)
            token_batches.append(indices.to(torch.int16))
            paths.extend(str(p) for p in batch_paths)
            code_hist.scatter_add_(0, indices.reshape(-1), torch.ones(indices.numel(), dtype=torch.long))
            if position_hist is None:
                token_h, token_w = (int(v) for v in indices.shape[-2:])
                position_hist = torch.zeros(token_h, token_w, cfg.codebook_size, dtype=torch.long)
            for sample in indices:
                for y in range(sample.shape[0]):
                    for x in range(sample.shape[1]):
                        position_hist[y, x, int(sample[y, x].item())] += 1

    if position_hist is None:
        raise RuntimeError("no token batches exported")
    tokens = torch.cat(token_batches, dim=0).contiguous()
    token_count = int(code_hist.sum().item())
    probs = code_hist.float() / max(token_count, 1)
    nonzero = probs[probs > 0]
    entropy_bits = float(-(nonzero * torch.log2(nonzero)).sum().item())

    run_name = args.run_name or f"{checkpoint_path.stem}_stage1_tokens"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    token_path = out_dir / "semantic_tokens.pt"
    summary_path = out_dir / "summary.json"
    hist_path = out_dir / "histograms.pt"

    torch.save(
        {
            "tokens": tokens,
            "paths": paths,
            "checkpoint": str(checkpoint_path),
            "roots": roots,
            "crop_size": args.crop_size,
            "codebook_size": cfg.codebook_size,
            "token_shape": tuple(int(v) for v in tokens.shape[-2:]),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        token_path,
    )
    torch.save({"code_hist": code_hist, "position_hist": position_hist}, hist_path)

    summary = {
        "checkpoint": str(checkpoint_path),
        "roots": roots,
        "num_images": int(tokens.shape[0]),
        "crop_size": args.crop_size,
        "codebook_size": cfg.codebook_size,
        "token_shape": [int(v) for v in tokens.shape[-2:]],
        "total_tokens": token_count,
        "active_codes": int(torch.count_nonzero(code_hist).item()),
        "global_entropy_bits": entropy_bits,
        "tokens_dtype": str(tokens.dtype),
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    import wandb

    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config={"checkpoint": str(checkpoint_path), "roots": roots},
    )
    wandb_run.summary.update(summary)
    wandb_run.finish()

    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "summary": summary,
        "artifacts": {"tokens": str(token_path), "histograms": str(hist_path), "summary": str(summary_path)},
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
