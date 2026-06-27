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
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from coserdic.datasets.image_folder import IMAGE_SUFFIXES
from coserdic.entropy import CoSERBitstream, CoSERHeader, StaticResidualGridHuffmanCode, TopKEscapeHuffmanCode
from coserdic.models import (
    CausalTokenPrior,
    CausalTokenPriorConfig,
    ResidualDetailAutoEncoder,
    ResidualDetailConfig,
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


def decode_indices_to_latent_and_image(
    model: SemanticVQAutoEncoder,
    cfg: SemanticVQConfig,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = indices.to(model.vq.embedding.weight.device, dtype=torch.long).reshape(-1)
    embedding = model.vq.embedding.weight.float()
    if cfg.normalize_latent:
        embedding = F.normalize(embedding, dim=1) * math.sqrt(float(cfg.latent_channels))
    quant = embedding.index_select(0, flat)
    b, h, w = indices.shape
    quant = quant.view(b, h, w, cfg.latent_channels).permute(0, 3, 1, 2).contiguous()
    quant = quant.to(dtype=next(model.parameters()).dtype)
    return quant, model.decoder(quant)


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


def code_entropy_bits(codes: torch.Tensor, levels: int) -> float:
    hist = torch.bincount(codes.detach().cpu().reshape(-1), minlength=levels).float()
    probs = hist / hist.sum().clamp_min(1.0)
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


@torch.no_grad()
def fit_detail_prior(
    semantic_model: SemanticVQAutoEncoder,
    semantic_cfg: SemanticVQConfig,
    detail_model: ResidualDetailAutoEncoder,
    roots: list[str],
    *,
    crop_size: int,
    max_images: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    smoothing_count: int,
) -> tuple[StaticResidualGridHuffmanCode, dict[str, object]]:
    dataset = CenterCropImageDataset(roots, crop_size=crop_size, limit=max_images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    counts = torch.zeros(detail_model.quantizer.levels, dtype=torch.long)
    total = 0
    abs_sum = 0.0
    sq_sum = 0.0
    for x_cpu in tqdm(loader, desc="fit learned residual detail prior"):
        x = x_cpu.to(device, non_blocking=True)
        semantic = semantic_model(x, quantize_mix=1.0)
        x_sem = semantic["x_sem"].detach()
        s_hat = semantic["quantized"].detach()
        out = detail_model(x, x_sem, s_hat, training_noise=False)
        codes = out["detail_codes"].detach().cpu()
        h = out["h_detail"].detach().cpu().float()
        counts += torch.bincount(codes.reshape(-1), minlength=detail_model.quantizer.levels)
        total += int(codes.numel())
        abs_sum += float(h.abs().sum().item())
        sq_sum += float(h.pow(2).sum().item())
    prior = StaticResidualGridHuffmanCode.from_counts(
        counts,
        bits=detail_model.quantizer.bits,
        value_range=detail_model.quantizer.value_range,
        smoothing_count=smoothing_count,
    )
    probs = counts.float() / counts.sum().clamp_min(1)
    nonzero = probs[probs > 0]
    entropy = float((-(nonzero * torch.log2(nonzero))).sum().item())
    mean_huffman = float(sum(int(c) * int(l) for c, l in zip(counts.tolist(), prior.huffman.code_lengths)) / max(total, 1))
    summary = {
        "fit_num_images": len(dataset),
        "fit_total_symbols": int(total),
        "fit_counts": [int(v) for v in counts.tolist()],
        "fit_symbol_entropy_bits": entropy,
        "fit_mean_huffman_bits_per_symbol": mean_huffman,
        "fit_h_detail_abs_mean": float(abs_sum / max(total, 1)),
        "fit_h_detail_std": float(math.sqrt(max(sq_sum / max(total, 1), 0.0))),
    }
    return prior, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--semantic-prior", required=True)
    parser.add_argument("--token-prior-checkpoint", default="")
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--fit-prior-root", action="append", default=None)
    parser.add_argument("--detail-prior", default="")
    parser.add_argument("--output-dir", default="results/bitstreams/stage3_residual_autoencoder")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--fit-prior-max-images", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoothing-count", type=int, default=1)
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
    seed_everything(int(checkpoint.get("config", {}).get("experiment", {}).get("seed", 42)))

    stage1_path = Path(checkpoint["stage1_checkpoint"])
    stage1_checkpoint = torch.load(stage1_path, map_location="cpu", weights_only=False)
    semantic_cfg = SemanticVQConfig(**stage1_checkpoint["semantic_vq_config"])
    semantic_model = SemanticVQAutoEncoder(semantic_cfg).to(device)
    semantic_model.load_state_dict(stage1_checkpoint["model"])
    semantic_model.eval()

    detail_cfg = ResidualDetailConfig(**checkpoint["detail_config"])
    detail_model = ResidualDetailAutoEncoder(detail_cfg).to(device)
    detail_model.load_state_dict(checkpoint["model"])
    detail_model.eval()

    semantic_prior_payload = json.loads(Path(args.semantic_prior).read_text())
    semantic_code = TopKEscapeHuffmanCode.from_dict(semantic_prior_payload)
    token_prior_value = args.token_prior_checkpoint or str(semantic_prior_payload.get("token_prior_checkpoint", ""))
    if not token_prior_value:
        raise ValueError("pass --token-prior-checkpoint or store token_prior_checkpoint in semantic prior")
    token_prior = load_token_prior(Path(token_prior_value), device)

    run_name = args.run_name or f"{checkpoint_path.stem}_bitstream_eval"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = out_dir / "streams"
    if args.save_bitstreams:
        stream_dir.mkdir(parents=True, exist_ok=True)

    prior_fit_summary: dict[str, object] = {}
    if args.detail_prior:
        residual_code = StaticResidualGridHuffmanCode.from_dict(json.loads(Path(args.detail_prior).read_text()))
    else:
        roots = args.fit_prior_root
        if not roots:
            raise ValueError("pass --detail-prior or --fit-prior-root")
        residual_code, prior_fit_summary = fit_detail_prior(
            semantic_model,
            semantic_cfg,
            detail_model,
            roots,
            crop_size=args.crop_size,
            max_images=args.fit_prior_max_images,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            smoothing_count=args.smoothing_count,
        )
        prior_payload = residual_code.to_dict()
        prior_payload.update(
            {
                "stage3_checkpoint": str(checkpoint_path),
                "stage1_checkpoint": str(stage1_path),
                "fit_roots": [str(v) for v in roots],
                **prior_fit_summary,
            }
        )
        prior_path = out_dir / "learned_residual_huffman_prior.json"
        prior_path.write_text(json.dumps(prior_payload, indent=2, allow_nan=False))
        args.detail_prior = str(prior_path)

    if residual_code.bits != detail_model.quantizer.bits:
        raise ValueError("detail prior bits do not match model quantizer")
    if abs(residual_code.value_range - detail_model.quantizer.value_range) > 1.0e-9:
        raise ValueError("detail prior value_range does not match model quantizer")

    roots = args.image_root
    if not roots:
        raise ValueError("no evaluation roots found; pass --image-root")
    dataset = CenterCropImageDataset(roots, crop_size=args.crop_size, limit=args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    coder = CoSERBitstream(header_codec=args.stream_header_codec, checksum_codec=args.stream_checksum_codec)
    sample_written = False
    roundtrip_failures: list[dict[str, object]] = []
    metrics: dict[str, list[float]] = {
        "semantic_payload_bpp": [],
        "detail_payload_bpp": [],
        "actual_payload_bpp": [],
        "paper_bpp": [],
        "total_payload_bpp": [],
        "debug_semantic_only_full_stream_bpp": [],
        "debug_full_stream_bpp": [],
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
        "h_detail_abs_mean": [],
        "h_detail_std": [],
        "detail_code_entropy_bits": [],
    }

    with torch.no_grad():
        for batch_index, x_cpu in enumerate(tqdm(loader, desc=run_name)):
            x = x_cpu.to(device, non_blocking=True)
            semantic = semantic_model(x, quantize_mix=1.0)
            indices_batch = semantic["indices"].detach().cpu()
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
                s_hat, x_sem = decode_indices_to_latent_and_image(semantic_model, semantic_cfg, decoded_indices.unsqueeze(0))
                x_sem = x_sem.clamp(0, 1)
                token_roundtrip = bool(torch.equal(decoded_indices, indices))

                encoded = detail_model(x_i, x_sem, s_hat, training_noise=False)
                detail_codes = encoded["detail_codes"].squeeze(0).detach().cpu()
                detail_payload = residual_code.encode(detail_codes)
                decoded_detail_codes = residual_code.decode(detail_payload, shape=tuple(detail_codes.shape))
                detail_roundtrip = bool(torch.equal(decoded_detail_codes, detail_codes))
                detail_latent = detail_model.quantizer.dequantize(decoded_detail_codes.to(device)).unsqueeze(0).to(dtype=x_sem.dtype)
                decoded = detail_model.decoder(s_hat, detail_latent, x_sem)
                x_hat = decoded["x_aux"].clamp(0, 1)

                common_header = {
                    "codec_version": "s3rae0",
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
                semantic_only_stream = coder.pack(
                    CoSERHeader(**common_header, detail_shape=(0,), entropy_model_version="s2sem-lteh0"),
                    semantic_tokens=semantic_payload,
                )
                stage3_stream = coder.pack(
                    CoSERHeader(
                        **common_header,
                        detail_shape=tuple(int(v) for v in detail_codes.shape),
                        entropy_model_version=f"s3rae-c{detail_cfg.detail_channels}-b{detail_cfg.quant_bits}",
                    ),
                    semantic_tokens=semantic_payload,
                    detail_latents=detail_payload,
                )
                unpacked = coder.unpack(stage3_stream)
                stream_detail = residual_code.decode(unpacked.detail_latents, shape=tuple(unpacked.header.detail_shape))
                stream_detail_roundtrip = bool(torch.equal(stream_detail, detail_codes))
                if not (token_roundtrip and detail_roundtrip and stream_detail_roundtrip):
                    roundtrip_failures.append(
                        {
                            "index": int(global_index),
                            "semantic_token_roundtrip": token_roundtrip,
                            "detail_code_roundtrip": detail_roundtrip,
                            "stream_detail_roundtrip": stream_detail_roundtrip,
                        }
                    )

                h = int(x_i.shape[-2])
                w = int(x_i.shape[-1])
                semantic_payload_bpp = coder.actual_payload_bpp(semantic_payload, h, w)
                detail_payload_bpp = coder.actual_payload_bpp(detail_payload, h, w)
                actual_payload_bpp = coder.actual_payload_bpp((semantic_payload, detail_payload), h, w)
                semantic_only_debug_bpp = coder.debug_full_stream_bpp(semantic_only_stream, h, w)
                stage3_debug_bpp = coder.debug_full_stream_bpp(stage3_stream, h, w)
                metrics["semantic_payload_bpp"].append(semantic_payload_bpp)
                metrics["detail_payload_bpp"].append(detail_payload_bpp)
                metrics["actual_payload_bpp"].append(actual_payload_bpp)
                metrics["paper_bpp"].append(actual_payload_bpp)
                metrics["total_payload_bpp"].append(actual_payload_bpp)
                metrics["debug_semantic_only_full_stream_bpp"].append(semantic_only_debug_bpp)
                metrics["debug_full_stream_bpp"].append(stage3_debug_bpp)
                metrics["semantic_only_full_stream_bpp"].append(semantic_only_debug_bpp)
                metrics["stage3_full_stream_bpp"].append(stage3_debug_bpp)
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
                metrics["h_detail_abs_mean"].append(float(encoded["h_detail"].abs().mean().item()))
                metrics["h_detail_std"].append(float(encoded["h_detail"].float().std(unbiased=False).item()))
                metrics["detail_code_entropy_bits"].append(code_entropy_bits(detail_codes, residual_code.levels))

                if args.save_bitstreams:
                    (stream_dir / f"image{global_index:05d}.coser").write_bytes(stage3_stream)
                if not sample_written:
                    save_image(
                        torch.cat([x_i.detach().cpu(), x_sem.detach().cpu(), x_hat.detach().cpu()], dim=0),
                        out_dir / "stage3_residual_autoencoder.png",
                        nrow=1,
                    )
                    sample_written = True

    summary: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "stage1_checkpoint": str(stage1_path),
        "semantic_prior": str(args.semantic_prior),
        "detail_prior": str(args.detail_prior),
        "token_prior_checkpoint": str(token_prior_value),
        "num_images": len(dataset),
        "crop_size": int(args.crop_size),
        "semantic_topk": int(semantic_code.topk),
        "detail_shape": [int(detail_cfg.detail_channels), args.crop_size // 32, args.crop_size // 32],
        "detail_bits": int(detail_cfg.quant_bits),
        "detail_range": float(detail_cfg.value_range),
        "detail_codec": "learned_residual_ae_static_huffman",
        "stream_header_codec": args.stream_header_codec,
        "stream_checksum_codec": args.stream_checksum_codec,
        "main_bpp_metric": "actual_payload_bpp_mean",
        "paper_bpp_metric": "paper_bpp_mean",
        "debug_bpp_metric": "debug_full_stream_bpp_mean",
        "residual_code": residual_code.to_dict(),
        **prior_fit_summary,
    }
    for metric_name, values in metrics.items():
        summary[f"{metric_name}_mean"] = mean(values)
    summary["stage3_psnr_delta_vs_semantic_only"] = float(summary["stage3_psnr_mean"]) - float(summary["semantic_only_psnr_mean"])
    summary["stage3_l1_delta_vs_semantic_only"] = float(summary["stage3_l1_mean"]) - float(summary["semantic_only_l1_mean"])
    summary["stage3_ms_ssim_delta_vs_semantic_only"] = float(summary["stage3_ms_ssim_mean"]) - float(summary["semantic_only_ms_ssim_mean"])
    summary["all_semantic_tokens_roundtrip"] = bool(all(v == 1.0 for v in metrics["semantic_token_roundtrip"]))
    summary["all_detail_codes_roundtrip"] = bool(all(v == 1.0 for v in metrics["detail_code_roundtrip"]))
    summary["roundtrip_failure_count"] = len(roundtrip_failures)
    summary["roundtrip_failures"] = roundtrip_failures[:20]

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    import wandb

    wandb_run = wandb.init(project="coserdic", name=run_name, mode=args.wandb_mode, config=vars(args))
    wandb_run.summary.update(summary)
    wandb_run.finish()

    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "summary": summary,
        "artifacts": {
            "summary": str(summary_path),
            "sample": str(out_dir / "stage3_residual_autoencoder.png"),
            "detail_prior": str(args.detail_prior),
        },
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))
    if roundtrip_failures and not args.allow_roundtrip_failures:
        raise RuntimeError(f"roundtrip failed for {len(roundtrip_failures)} images; see {summary_path}")


if __name__ == "__main__":
    main()
