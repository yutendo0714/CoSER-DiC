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
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from coserdic.datasets.image_folder import IMAGE_SUFFIXES
from coserdic.entropy import (
    CoSERBitstream,
    CoSERHeader,
    StaticHuffmanCode,
    decode_semantic_tokens,
    encode_semantic_tokens,
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


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((x - y).pow(2), dim=(1, 2, 3)).clamp_min(1.0e-12)
    return -10.0 * torch.log10(mse)


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


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
    parser.add_argument("--prior", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--output-dir", default="results/bitstreams/stage2_static_huffman")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-bitstreams", action="store_true")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 2 evaluation.")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = load_config_from_checkpoint(checkpoint, args.config or None)
    seed_everything(int(raw.get("experiment", {}).get("seed", 42)))
    cfg = SemanticVQConfig(**checkpoint["semantic_vq_config"])
    model = SemanticVQAutoEncoder(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    code = StaticHuffmanCode.from_dict(json.loads(Path(args.prior).read_text()))
    if code.codebook_size != cfg.codebook_size:
        raise ValueError("prior codebook_size does not match checkpoint")

    roots = args.image_root or raw.get("data", {}).get("train_roots")
    if not roots:
        raise ValueError("no evaluation roots found; pass --image-root")
    dataset = CenterCropImageDataset(roots, crop_size=args.crop_size, limit=args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    run_name = args.run_name or f"{checkpoint_path.stem}_stage2_static_huffman_eval"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = out_dir / "streams"
    if args.save_bitstreams:
        stream_dir.mkdir(parents=True, exist_ok=True)

    buckets = {
        "static_huffman": {"payload_bpp": [], "actual_bpp": [], "stream_bytes": [], "payload_bytes": [], "psnr": [], "l1": [], "ms_ssim": [], "roundtrip": []},
        "fixed_bits": {"payload_bpp": [], "actual_bpp": [], "stream_bytes": [], "payload_bytes": [], "psnr": [], "l1": [], "ms_ssim": [], "roundtrip": []},
    }
    coder = CoSERBitstream()
    sample_written = False

    with torch.no_grad():
        for batch_index, x_cpu in enumerate(tqdm(loader, desc=run_name)):
            x = x_cpu.to(device, non_blocking=True)
            out = model(x, quantize_mix=1.0)
            indices_batch = out["indices"].detach().cpu()
            for image_index in range(x.shape[0]):
                x_i = x[image_index : image_index + 1]
                indices = indices_batch[image_index]
                payloads = {
                    "static_huffman": code.encode(indices),
                    "fixed_bits": encode_semantic_tokens(indices, codebook_size=cfg.codebook_size, codec="fixed_bits"),
                }
                for codec_name, token_payload in payloads.items():
                    header = CoSERHeader(
                        codec_version="stage2-static-huffman-mvp",
                        image_height=int(x_i.shape[-2]),
                        image_width=int(x_i.shape[-1]),
                        padded_height=int(x_i.shape[-2]),
                        padded_width=int(x_i.shape[-1]),
                        color_space="RGB",
                        rate_level_id=0,
                        perception_level_id=0,
                        semantic_shape=tuple(int(v) for v in indices.shape),
                        detail_shape=(0,),
                        entropy_model_version=f"semantic-token-{codec_name}-v0",
                        cdf_precision=0,
                    )
                    stream = coder.pack(header, semantic_tokens=token_payload)
                    unpacked = coder.unpack(stream)
                    if codec_name == "static_huffman":
                        decoded = code.decode(unpacked.semantic_tokens, shape=tuple(unpacked.header.semantic_shape))
                    else:
                        decoded = decode_semantic_tokens(
                            unpacked.semantic_tokens,
                            shape=tuple(unpacked.header.semantic_shape),
                            codebook_size=cfg.codebook_size,
                            codec="fixed_bits",
                        )
                    x_hat = decode_indices_to_image(model, cfg, decoded.unsqueeze(0)).clamp(0, 1)
                    bucket = buckets[codec_name]
                    bucket["payload_bpp"].append(coder.actual_bpp(token_payload, int(x_i.shape[-2]), int(x_i.shape[-1])))
                    bucket["actual_bpp"].append(coder.actual_bpp(stream, int(x_i.shape[-2]), int(x_i.shape[-1])))
                    bucket["stream_bytes"].append(float(len(stream)))
                    bucket["payload_bytes"].append(float(len(token_payload)))
                    bucket["psnr"].append(float(psnr(x_i, x_hat).item()))
                    bucket["l1"].append(float(torch.mean(torch.abs(x_hat - x_i)).item()))
                    bucket["ms_ssim"].append(float(ms_ssim(x_hat, x_i, data_range=1.0, size_average=True).item()))
                    bucket["roundtrip"].append(float(torch.equal(decoded, indices)))
                    if args.save_bitstreams:
                        (stream_dir / f"batch{batch_index:04d}_img{image_index:02d}_{codec_name}.coser").write_bytes(stream)
                    if not sample_written and codec_name == "static_huffman":
                        sample_path = out_dir / "stage2_static_huffman_recon_grid.png"
                        save_image(torch.cat([x_i.detach().cpu(), x_hat.detach().cpu()], dim=0), sample_path, nrow=1)
                        sample_written = True

    summary: dict[str, float | int | str | bool] = {
        "checkpoint": str(checkpoint_path),
        "prior": str(args.prior),
        "num_images": len(dataset),
        "crop_size": args.crop_size,
        "codebook_size": cfg.codebook_size,
    }
    for codec_name, metrics in buckets.items():
        for metric_name, values in metrics.items():
            summary[f"{codec_name}_{metric_name}_mean"] = mean(values)
        summary[f"{codec_name}_all_tokens_roundtrip"] = bool(all(v == 1.0 for v in metrics["roundtrip"]))
    summary["static_huffman_payload_bpp_delta_vs_fixed_bits"] = (
        float(summary["static_huffman_payload_bpp_mean"]) - float(summary["fixed_bits_payload_bpp_mean"])
    )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    import wandb

    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config={"checkpoint": str(checkpoint_path), "prior": str(args.prior), "roots": roots},
    )
    wandb_run.summary.update(summary)
    wandb_run.finish()

    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "summary": summary,
        "artifacts": {
            "summary": str(summary_path),
            "output_dir": str(out_dir),
            "streams": str(stream_dir) if args.save_bitstreams else "",
        },
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
