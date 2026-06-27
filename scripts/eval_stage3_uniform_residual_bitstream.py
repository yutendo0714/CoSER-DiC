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
    StaticResidualGridHuffmanCode,
    StaticResidualGridPositionHuffmanCode,
    TopKEscapeHuffmanCode,
    UniformResidualGridCode,
)
from coserdic.models import (
    CausalTokenPrior,
    CausalTokenPriorConfig,
    SemanticVQAutoEncoder,
    SemanticVQConfig,
    decoder_schedule_topk_indices,
    topk_from_prefix,
)
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


def load_token_prior(path: Path, device: torch.device) -> CausalTokenPrior:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = CausalTokenPriorConfig(**checkpoint["config"])
    model = CausalTokenPrior(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


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


@torch.no_grad()
def encode_semantic_payload(
    code: TopKEscapeHuffmanCode,
    token_prior: CausalTokenPrior,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[bytes, dict[str, float]]:
    topk_indices = decoder_schedule_topk_indices(token_prior, indices, topk=code.topk, device=device)
    payload = code.encode(indices.detach().cpu(), topk_indices)
    escape_count = code.escape_count(indices.detach().cpu(), topk_indices)
    return payload, {
        "escape_count": float(escape_count),
        "topk_hit_rate": float(1.0 - escape_count / max(indices.numel(), 1)),
        "unpadded_bits": float(code.encoded_bits(indices.detach().cpu(), topk_indices)),
    }


@torch.no_grad()
def decode_semantic_payload(
    code: TopKEscapeHuffmanCode,
    token_prior: CausalTokenPrior,
    payload: bytes,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    def provider(_index: int, prefix: tuple[int, ...]) -> list[int]:
        return topk_from_prefix(token_prior, prefix, topk=code.topk, device=device)

    return code.decode_with_provider(payload, shape=shape, topk_provider=provider)


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


def summarize_residual_code(
    residual_code: UniformResidualGridCode | StaticResidualGridHuffmanCode | StaticResidualGridPositionHuffmanCode,
) -> dict[str, object]:
    payload = residual_code.to_dict()
    if payload.get("codec") != "static_residual_grid_position_huffman":
        return payload

    lengths = [
        int(length)
        for position_lengths in payload.get("position_code_lengths", [])
        for length in position_lengths
    ]
    summary: dict[str, object] = {
        "codec": payload["codec"],
        "version": payload["version"],
        "bits": payload["bits"],
        "value_range": payload["value_range"],
        "payload_codec": payload["payload_codec"],
        "detail_shape": payload["detail_shape"],
        "num_position_codes": len(payload.get("position_code_lengths", [])),
    }
    if lengths:
        summary.update(
            {
                "min_code_length": min(lengths),
                "max_code_length": max(lengths),
                "mean_code_length_unweighted": float(sum(lengths) / len(lengths)),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--semantic-prior", required=True)
    parser.add_argument("--token-prior-checkpoint", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--output-dir", default="results/bitstreams/stage3_uniform_residual")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--detail-downsample-factor", type=int, default=32)
    parser.add_argument("--detail-bits", type=int, default=6)
    parser.add_argument("--detail-range", type=float, default=0.5)
    parser.add_argument(
        "--detail-codec",
        choices=["fixed_bits", "zlib_fixed_bits", "huffman", "position_huffman"],
        default="zlib_fixed_bits",
    )
    parser.add_argument("--detail-prior", default="")
    parser.add_argument("--stream-header-codec", choices=["json", "compact"], default="json")
    parser.add_argument("--stream-checksum-codec", choices=["sha256", "crc32"], default="sha256")
    parser.add_argument("--save-bitstreams", action="store_true")
    parser.add_argument("--allow-roundtrip-failures", action="store_true")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 3 evaluation.")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = load_config_from_checkpoint(checkpoint, args.config or None)
    seed_everything(int(raw.get("experiment", {}).get("seed", 42)))
    cfg = SemanticVQConfig(**checkpoint["semantic_vq_config"])
    model = SemanticVQAutoEncoder(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    semantic_prior_payload = json.loads(Path(args.semantic_prior).read_text())
    semantic_code = TopKEscapeHuffmanCode.from_dict(semantic_prior_payload)
    token_prior_value = args.token_prior_checkpoint or str(semantic_prior_payload.get("token_prior_checkpoint", ""))
    if not token_prior_value:
        raise ValueError("pass --token-prior-checkpoint or store token_prior_checkpoint in semantic prior")
    token_prior = load_token_prior(Path(token_prior_value), device)
    if args.detail_codec in {"huffman", "position_huffman"}:
        if not args.detail_prior:
            raise ValueError("--detail-prior is required when --detail-codec uses static Huffman")
        residual_prior_payload = json.loads(Path(args.detail_prior).read_text())
        if args.detail_codec == "position_huffman":
            residual_code = StaticResidualGridPositionHuffmanCode.from_dict(residual_prior_payload)
        else:
            residual_code = StaticResidualGridHuffmanCode.from_dict(residual_prior_payload)
        if int(args.detail_bits) != residual_code.bits:
            raise ValueError("--detail-bits does not match residual Huffman prior")
        if abs(float(args.detail_range) - residual_code.value_range) > 1.0e-9:
            raise ValueError("--detail-range does not match residual Huffman prior")
    else:
        residual_code = UniformResidualGridCode(bits=args.detail_bits, value_range=args.detail_range, codec=args.detail_codec)

    if args.crop_size % args.detail_downsample_factor != 0:
        raise ValueError("--crop-size must be divisible by --detail-downsample-factor")
    detail_hw = args.crop_size // args.detail_downsample_factor

    roots = args.image_root or raw.get("data", {}).get("train_roots")
    if not roots:
        raise ValueError("no evaluation roots found; pass --image-root")
    dataset = CenterCropImageDataset(roots, crop_size=args.crop_size, limit=args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    run_name = args.run_name or (
        f"{checkpoint_path.stem}_stage3_uniform_residual_d{args.detail_downsample_factor}_b{args.detail_bits}"
    )
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = out_dir / "streams"
    if args.save_bitstreams:
        stream_dir.mkdir(parents=True, exist_ok=True)

    coder = CoSERBitstream(header_codec=args.stream_header_codec, checksum_codec=args.stream_checksum_codec)
    sample_written = False
    roundtrip_failures: list[dict[str, object]] = []
    metrics: dict[str, list[float]] = {
        "semantic_payload_bpp": [],
        "detail_payload_bpp": [],
        "total_payload_bpp": [],
        "semantic_only_full_stream_bpp": [],
        "stage3_full_stream_bpp": [],
        "semantic_payload_bytes": [],
        "detail_payload_bytes": [],
        "stage3_stream_bytes": [],
        "semantic_only_psnr": [],
        "semantic_only_l1": [],
        "semantic_only_ms_ssim": [],
        "stage3_psnr": [],
        "stage3_l1": [],
        "stage3_ms_ssim": [],
        "semantic_topk_hit_rate": [],
        "semantic_token_roundtrip": [],
        "detail_code_roundtrip": [],
        "residual_grid_abs_mean": [],
        "residual_grid_std": [],
        "residual_grid_clip_ratio": [],
        "detail_code_entropy_bits": [],
    }

    entropy_model_version = (
        f"s3urg-d{args.detail_downsample_factor}-b{args.detail_bits}-r{int(round(args.detail_range * 100)):03d}"
    )

    with torch.no_grad():
        for batch_index, x_cpu in enumerate(tqdm(loader, desc=run_name)):
            x = x_cpu.to(device, non_blocking=True)
            out = model(x, quantize_mix=1.0)
            indices_batch = out["indices"].detach().cpu()
            for image_index in range(x.shape[0]):
                global_index = batch_index * args.batch_size + image_index
                x_i = x[image_index : image_index + 1]
                indices = indices_batch[image_index]

                semantic_payload, semantic_stats = encode_semantic_payload(semantic_code, token_prior, indices, device=device)
                decoded_indices = decode_semantic_payload(
                    semantic_code,
                    token_prior,
                    semantic_payload,
                    shape=tuple(indices.shape),
                    device=device,
                )
                x_sem = decode_indices_to_image(model, cfg, decoded_indices.unsqueeze(0)).clamp(0, 1)
                token_roundtrip = bool(torch.equal(decoded_indices, indices))

                residual = x_i - x_sem
                residual_grid = F.adaptive_avg_pool2d(residual, output_size=(detail_hw, detail_hw)).squeeze(0)
                detail_codes = residual_code.quantize(residual_grid)
                detail_hist = torch.bincount(detail_codes.reshape(-1), minlength=residual_code.levels).float()
                detail_probs = detail_hist / detail_hist.sum().clamp_min(1.0)
                detail_nonzero = detail_probs[detail_probs > 0]
                detail_entropy_bits = float((-(detail_nonzero * torch.log2(detail_nonzero))).sum().item())
                detail_payload = residual_code.encode(detail_codes)
                decoded_detail_codes = residual_code.decode(detail_payload, shape=tuple(detail_codes.shape))
                detail_roundtrip = bool(torch.equal(decoded_detail_codes, detail_codes))
                residual_grid_hat = residual_code.dequantize(decoded_detail_codes).unsqueeze(0).to(device=device, dtype=x_sem.dtype)
                residual_hat = F.interpolate(
                    residual_grid_hat,
                    size=tuple(int(v) for v in x_i.shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                )
                x_hat = (x_sem + residual_hat).clamp(0, 1)

                common_header = {
                    "codec_version": "s3urg0",
                    "image_height": int(x_i.shape[-2]),
                    "image_width": int(x_i.shape[-1]),
                    "padded_height": int(x_i.shape[-2]),
                    "padded_width": int(x_i.shape[-1]),
                    "color_space": "RGB",
                    "rate_level_id": 0,
                    "perception_level_id": 0,
                    "semantic_shape": tuple(int(v) for v in indices.shape),
                    "cdf_precision": 0,
                }
                semantic_only_header = CoSERHeader(
                    **common_header,
                    detail_shape=(0,),
                    entropy_model_version="s2sem-lteh0",
                )
                stage3_header = CoSERHeader(
                    **common_header,
                    detail_shape=tuple(int(v) for v in detail_codes.shape),
                    entropy_model_version=entropy_model_version,
                )
                semantic_only_stream = coder.pack(semantic_only_header, semantic_tokens=semantic_payload)
                stage3_stream = coder.pack(
                    stage3_header,
                    semantic_tokens=semantic_payload,
                    detail_latents=detail_payload,
                )
                unpacked = coder.unpack(stage3_stream)
                decoded_detail_from_stream = residual_code.decode(
                    unpacked.detail_latents,
                    shape=tuple(unpacked.header.detail_shape),
                )
                stream_detail_roundtrip = bool(torch.equal(decoded_detail_from_stream, detail_codes))

                if not (token_roundtrip and detail_roundtrip and stream_detail_roundtrip):
                    roundtrip_failures.append(
                        {
                            "index": int(global_index),
                            "path": str(dataset.paths[global_index]) if global_index < len(dataset.paths) else "",
                            "semantic_token_roundtrip": token_roundtrip,
                            "detail_code_roundtrip": detail_roundtrip,
                            "stream_detail_roundtrip": stream_detail_roundtrip,
                        }
                    )

                h = int(x_i.shape[-2])
                w = int(x_i.shape[-1])
                metrics["semantic_payload_bpp"].append(coder.actual_bpp(semantic_payload, h, w))
                metrics["detail_payload_bpp"].append(coder.actual_bpp(detail_payload, h, w))
                metrics["total_payload_bpp"].append(coder.actual_bpp(semantic_payload + detail_payload, h, w))
                metrics["semantic_only_full_stream_bpp"].append(coder.actual_bpp(semantic_only_stream, h, w))
                metrics["stage3_full_stream_bpp"].append(coder.actual_bpp(stage3_stream, h, w))
                metrics["semantic_payload_bytes"].append(float(len(semantic_payload)))
                metrics["detail_payload_bytes"].append(float(len(detail_payload)))
                metrics["stage3_stream_bytes"].append(float(len(stage3_stream)))
                metrics["semantic_only_psnr"].append(float(psnr(x_i, x_sem).item()))
                metrics["semantic_only_l1"].append(float(torch.mean(torch.abs(x_sem - x_i)).item()))
                metrics["semantic_only_ms_ssim"].append(float(ms_ssim(x_sem, x_i, data_range=1.0, size_average=True).item()))
                metrics["stage3_psnr"].append(float(psnr(x_i, x_hat).item()))
                metrics["stage3_l1"].append(float(torch.mean(torch.abs(x_hat - x_i)).item()))
                metrics["stage3_ms_ssim"].append(float(ms_ssim(x_hat, x_i, data_range=1.0, size_average=True).item()))
                metrics["semantic_topk_hit_rate"].append(float(semantic_stats["topk_hit_rate"]))
                metrics["semantic_token_roundtrip"].append(float(token_roundtrip))
                metrics["detail_code_roundtrip"].append(float(detail_roundtrip and stream_detail_roundtrip))
                metrics["residual_grid_abs_mean"].append(float(torch.mean(torch.abs(residual_grid)).item()))
                metrics["residual_grid_std"].append(float(torch.std(residual_grid.float(), unbiased=False).item()))
                metrics["residual_grid_clip_ratio"].append(
                    float(torch.mean((torch.abs(residual_grid) >= float(args.detail_range)).float()).item())
                )
                metrics["detail_code_entropy_bits"].append(detail_entropy_bits)

                if args.save_bitstreams:
                    (stream_dir / f"image{global_index:05d}.coser").write_bytes(stage3_stream)
                if not sample_written:
                    sample_path = out_dir / "stage3_uniform_residual_grid.png"
                    save_image(
                        torch.cat([x_i.detach().cpu(), x_sem.detach().cpu(), x_hat.detach().cpu()], dim=0),
                        sample_path,
                        nrow=1,
                    )
                    sample_written = True

    summary: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "semantic_prior": str(args.semantic_prior),
        "detail_prior": str(args.detail_prior),
        "token_prior_checkpoint": str(token_prior_value),
        "num_images": len(dataset),
        "crop_size": int(args.crop_size),
        "semantic_topk": int(semantic_code.topk),
        "detail_downsample_factor": int(args.detail_downsample_factor),
        "detail_shape": [3, int(detail_hw), int(detail_hw)],
        "detail_bits": int(args.detail_bits),
        "detail_range": float(args.detail_range),
        "detail_codec": args.detail_codec,
        "stream_header_codec": args.stream_header_codec,
        "stream_checksum_codec": args.stream_checksum_codec,
        "residual_code": summarize_residual_code(residual_code),
    }
    for metric_name, values in metrics.items():
        summary[f"{metric_name}_mean"] = mean(values)
    summary["stage3_psnr_delta_vs_semantic_only"] = (
        float(summary["stage3_psnr_mean"]) - float(summary["semantic_only_psnr_mean"])
    )
    summary["stage3_l1_delta_vs_semantic_only"] = (
        float(summary["stage3_l1_mean"]) - float(summary["semantic_only_l1_mean"])
    )
    summary["stage3_ms_ssim_delta_vs_semantic_only"] = (
        float(summary["stage3_ms_ssim_mean"]) - float(summary["semantic_only_ms_ssim_mean"])
    )
    summary["all_semantic_tokens_roundtrip"] = bool(all(v == 1.0 for v in metrics["semantic_token_roundtrip"]))
    summary["all_detail_codes_roundtrip"] = bool(all(v == 1.0 for v in metrics["detail_code_roundtrip"]))
    summary["roundtrip_failure_count"] = len(roundtrip_failures)
    summary["roundtrip_failures"] = roundtrip_failures[:20]

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    import wandb

    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config={**vars(args), **residual_code.to_dict()},
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
            "sample": str(out_dir / "stage3_uniform_residual_grid.png"),
        },
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))
    if roundtrip_failures and not args.allow_roundtrip_failures:
        raise RuntimeError(f"roundtrip failed for {len(roundtrip_failures)} images; see {summary_path}")


if __name__ == "__main__":
    main()
