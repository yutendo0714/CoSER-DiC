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

from coserdic.datasets.eval_protocols import (
    EVAL_PROTOCOLS,
    flatten_selection_paths,
    protocol_summary,
    resolve_eval_protocol,
)
from coserdic.datasets.image_folder import IMAGE_SUFFIXES
from coserdic.entropy import (
    CoSERBitstream,
    CoSERHeader,
    StaticResidualGridHuffmanCode,
    StaticResidualGridHybridHuffmanCode,
    StaticResidualGridPositionHuffmanCode,
    StaticResidualGridSemanticPositionLeftContextHuffmanCode,
    StaticResidualGridSemanticPositionHuffmanCode,
    TopKEscapeHuffmanCode,
    UniformResidualGridCode,
)
from coserdic.metrics import PerceptualMetricBundle
from coserdic.models import (
    CausalTokenPrior,
    CausalTokenPriorConfig,
    DECODER_POSTPROCESS_MODES,
    SemanticVQAutoEncoder,
    SemanticVQConfig,
    apply_decoder_postprocess,
    decoder_prefix_topk_indices,
    topk_from_prefix,
)
from coserdic.utils import seed_everything


class CenterCropImageDataset(Dataset):
    def __init__(
        self,
        roots: list[str],
        crop_size: int,
        limit: int = 0,
        image_paths: list[str | Path] | None = None,
    ) -> None:
        if image_paths is None:
            self.paths = []
            for root in roots:
                self.paths.extend(sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
        else:
            self.paths = [Path(path) for path in image_paths]
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no images found under {roots or image_paths}")
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
    topk_indices = decoder_prefix_topk_indices(token_prior, indices, topk=code.topk, device=device)
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


def safe_image_name(index: int, path: str) -> str:
    stem = Path(path).stem if path else "image"
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
    return f"image{index:05d}_{safe_stem}.png"


def summarize_residual_code(
    residual_code: (
        UniformResidualGridCode
        | StaticResidualGridHuffmanCode
        | StaticResidualGridHybridHuffmanCode
        | StaticResidualGridPositionHuffmanCode
        | StaticResidualGridSemanticPositionLeftContextHuffmanCode
        | StaticResidualGridSemanticPositionHuffmanCode
    ),
) -> dict[str, object]:
    if isinstance(residual_code, StaticResidualGridHybridHuffmanCode):
        return {
            "codec": "static_residual_grid_hybrid_huffman",
            "version": 0,
            "bits": int(residual_code.bits),
            "value_range": float(residual_code.value_range),
            "payload_codec": "hybrid_huffman",
            "position_code": summarize_residual_code(residual_code.position_code),
            "semantic_position_code": summarize_residual_code(residual_code.semantic_position_code),
        }
    payload = residual_code.to_dict()
    if payload.get("codec") == "static_residual_grid_semantic_position_leftctx_huffman":
        lengths = [
            int(length)
            for position_lengths in payload.get("position_group_context_code_lengths", [])
            for length in position_lengths
        ]
        summary: dict[str, object] = {
            "codec": payload["codec"],
            "version": payload["version"],
            "bits": payload["bits"],
            "value_range": payload["value_range"],
            "payload_codec": payload["payload_codec"],
            "detail_shape": payload["detail_shape"],
            "semantic_shape": payload["semantic_shape"],
            "group_count": payload["group_count"],
            "left_context_mode": payload["left_context_mode"],
            "left_context_count": payload["left_context_count"],
            "num_position_group_context_codes": len(payload.get("position_group_context_code_lengths", [])),
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
    if payload.get("codec") == "static_residual_grid_semantic_position_huffman":
        lengths = [
            int(length)
            for position_lengths in payload.get("position_group_code_lengths", [])
            for length in position_lengths
        ]
        summary: dict[str, object] = {
            "codec": payload["codec"],
            "version": payload["version"],
            "bits": payload["bits"],
            "value_range": payload["value_range"],
            "payload_codec": payload["payload_codec"],
            "detail_shape": payload["detail_shape"],
            "semantic_shape": payload["semantic_shape"],
            "group_count": payload["group_count"],
            "num_position_group_codes": len(payload.get("position_group_code_lengths", [])),
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
    parser.add_argument("--eval-protocol", choices=sorted(EVAL_PROTOCOLS), default="")
    parser.add_argument("--eval-dataset", action="append", default=None)
    parser.add_argument("--dpl-root", default="/dpl")
    parser.add_argument("--allow-protocol-count-mismatch", action="store_true")
    parser.add_argument("--allow-nondeterministic-eval", action="store_true")
    parser.add_argument("--output-dir", default="results/bitstreams/stage3_uniform_residual")
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--detail-downsample-factor", type=int, default=32)
    parser.add_argument("--detail-bits", type=int, default=6)
    parser.add_argument("--detail-range", type=float, default=0.5)
    parser.add_argument("--detail-gain", type=float, default=1.0)
    parser.add_argument("--decoder-postprocess", choices=DECODER_POSTPROCESS_MODES, default="none")
    parser.add_argument("--decoder-postprocess-strength", type=float, default=0.0)
    parser.add_argument(
        "--detail-codec",
        choices=[
            "fixed_bits",
            "zlib_fixed_bits",
            "huffman",
            "position_huffman",
            "semantic_position_huffman",
            "semantic_position_leftctx_huffman",
            "hybrid_huffman",
        ],
        default="zlib_fixed_bits",
    )
    parser.add_argument("--detail-prior", default="")
    parser.add_argument("--stream-header-codec", choices=["json", "compact"], default="json")
    parser.add_argument("--stream-checksum-codec", choices=["sha256", "crc32"], default="sha256")
    parser.add_argument("--save-bitstreams", action="store_true")
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument("--save-reconstruction-limit", type=int, default=0)
    parser.add_argument("--save-reconstruction-triptychs", action="store_true")
    parser.add_argument("--allow-roundtrip-failures", action="store_true")
    parser.add_argument("--compute-perceptual", action="store_true")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()
    protocol_default_crop_size = (
        EVAL_PROTOCOLS[args.eval_protocol].default_crop_size if args.eval_protocol else None
    )
    if args.crop_size is None:
        args.crop_size = int(protocol_default_crop_size or 256)
    if args.max_images is None:
        args.max_images = 0 if args.eval_protocol else 24

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 3 evaluation.")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = load_config_from_checkpoint(checkpoint, args.config or None)
    seed_everything(
        int(raw.get("experiment", {}).get("seed", 42)),
        deterministic=not args.allow_nondeterministic_eval,
    )
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
    if args.detail_codec in {
        "huffman",
        "position_huffman",
        "semantic_position_huffman",
        "semantic_position_leftctx_huffman",
        "hybrid_huffman",
    }:
        if not args.detail_prior:
            raise ValueError("--detail-prior is required when --detail-codec uses static Huffman")
        residual_prior_payload = json.loads(Path(args.detail_prior).read_text())
        if args.detail_codec == "hybrid_huffman":
            residual_code = StaticResidualGridHybridHuffmanCode.from_dict(residual_prior_payload)
        elif args.detail_codec == "semantic_position_leftctx_huffman":
            residual_code = StaticResidualGridSemanticPositionLeftContextHuffmanCode.from_dict(residual_prior_payload)
        elif args.detail_codec == "semantic_position_huffman":
            residual_code = StaticResidualGridSemanticPositionHuffmanCode.from_dict(residual_prior_payload)
        elif args.detail_codec == "position_huffman":
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
    if args.detail_gain < 0:
        raise ValueError("--detail-gain must be non-negative")
    if args.decoder_postprocess == "none" and abs(float(args.decoder_postprocess_strength)) > 1.0e-12:
        raise ValueError("--decoder-postprocess-strength must be 0 when --decoder-postprocess is none")
    if args.decoder_postprocess != "none" and float(args.decoder_postprocess_strength) <= 0.0:
        raise ValueError("--decoder-postprocess-strength must be positive when postprocess is enabled")
    detail_hw = args.crop_size // args.detail_downsample_factor

    eval_protocol_summary: dict[str, object] | None = None
    eval_image_paths: list[Path] | None = None
    roots = args.image_root or raw.get("data", {}).get("train_roots")
    if args.eval_protocol:
        selections = resolve_eval_protocol(
            args.eval_protocol,
            dpl_root=args.dpl_root,
            dataset_keys=args.eval_dataset,
            strict_expected_counts=not args.allow_protocol_count_mismatch,
        )
        eval_protocol_summary = protocol_summary(args.eval_protocol, selections)
        eval_image_paths = flatten_selection_paths(selections)
        roots = [str(root) for selection in selections for root in selection.source_roots]
    if not roots and eval_image_paths is None:
        raise ValueError("no evaluation roots found; pass --image-root")
    dataset = CenterCropImageDataset(roots, crop_size=args.crop_size, limit=args.max_images, image_paths=eval_image_paths)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    perceptual = PerceptualMetricBundle().to(device).eval() if args.compute_perceptual else None

    run_name = args.run_name or (
        f"{checkpoint_path.stem}_stage3_uniform_residual_d{args.detail_downsample_factor}_b{args.detail_bits}"
    )
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = out_dir / "streams"
    if args.save_bitstreams:
        stream_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dirs: dict[str, Path] = {}
    if args.save_reconstructions:
        reconstruction_root = out_dir / "reconstructions"
        reconstruction_dirs = {
            "reference": reconstruction_root / "reference",
            "semantic_only": reconstruction_root / "semantic_only",
            "stage3": reconstruction_root / "stage3",
        }
        if args.save_reconstruction_triptychs:
            reconstruction_dirs["triptych"] = reconstruction_root / "triptych"
        for directory in reconstruction_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    coder = CoSERBitstream(header_codec=args.stream_header_codec, checksum_codec=args.stream_checksum_codec)
    sample_written = False
    roundtrip_failures: list[dict[str, object]] = []
    per_image_records: list[dict[str, object]] = []
    reconstruction_manifest: list[dict[str, object]] = []
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
        "residual_grid_abs_mean": [],
        "residual_grid_std": [],
        "residual_grid_clip_ratio": [],
        "detail_code_entropy_bits": [],
    }
    if args.compute_perceptual:
        metrics.update(
            {
                "semantic_only_lpips_alex": [],
                "semantic_only_dists": [],
                "stage3_lpips_alex": [],
                "stage3_dists": [],
            }
        )
    if isinstance(residual_code, StaticResidualGridHybridHuffmanCode):
        metrics["detail_hybrid_semantic_mode"] = []

    entropy_model_version = (
        f"s3urg-d{args.detail_downsample_factor}-b{args.detail_bits}-r{int(round(args.detail_range * 100)):03d}"
    )
    if abs(float(args.detail_gain) - 1.0) > 1.0e-9:
        entropy_model_version += f"-g{int(round(args.detail_gain * 1000)):04d}"
    if args.decoder_postprocess != "none":
        strength_tag = int(round(float(args.decoder_postprocess_strength) * 1000))
        entropy_model_version += f"-pp{args.decoder_postprocess}{strength_tag:04d}"

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
                detail_hybrid_mode: float | None = None
                if isinstance(residual_code, StaticResidualGridHybridHuffmanCode):
                    detail_hybrid_mode = float(residual_code.select_mode(detail_codes, semantic_indices=decoded_indices))
                    metrics["detail_hybrid_semantic_mode"].append(detail_hybrid_mode)
                if isinstance(
                    residual_code,
                    (
                        StaticResidualGridSemanticPositionHuffmanCode,
                        StaticResidualGridSemanticPositionLeftContextHuffmanCode,
                        StaticResidualGridHybridHuffmanCode,
                    ),
                ):
                    detail_payload = residual_code.encode(detail_codes, semantic_indices=decoded_indices)
                    decoded_detail_codes = residual_code.decode(
                        detail_payload,
                        shape=tuple(detail_codes.shape),
                        semantic_indices=decoded_indices,
                    )
                else:
                    detail_payload = residual_code.encode(detail_codes)
                    decoded_detail_codes = residual_code.decode(detail_payload, shape=tuple(detail_codes.shape))
                detail_roundtrip = bool(torch.equal(decoded_detail_codes, detail_codes))
                residual_grid_hat = residual_code.dequantize(decoded_detail_codes).unsqueeze(0).to(device=device, dtype=x_sem.dtype)
                residual_grid_hat = residual_grid_hat * float(args.detail_gain)
                residual_hat = F.interpolate(
                    residual_grid_hat,
                    size=tuple(int(v) for v in x_i.shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                )
                x_hat = (x_sem + residual_hat).clamp(0, 1)
                x_hat = apply_decoder_postprocess(
                    x_hat,
                    mode=args.decoder_postprocess,
                    strength=float(args.decoder_postprocess_strength),
                )

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
                decoded_semantic_from_stream = decode_semantic_payload(
                    semantic_code,
                    token_prior,
                    unpacked.semantic_tokens,
                    shape=tuple(unpacked.header.semantic_shape),
                    device=device,
                )
                if isinstance(
                    residual_code,
                    (
                        StaticResidualGridSemanticPositionHuffmanCode,
                        StaticResidualGridSemanticPositionLeftContextHuffmanCode,
                        StaticResidualGridHybridHuffmanCode,
                    ),
                ):
                    decoded_detail_from_stream = residual_code.decode(
                        unpacked.detail_latents,
                        shape=tuple(unpacked.header.detail_shape),
                        semantic_indices=decoded_semantic_from_stream,
                    )
                else:
                    decoded_detail_from_stream = residual_code.decode(
                        unpacked.detail_latents,
                        shape=tuple(unpacked.header.detail_shape),
                    )
                stream_detail_roundtrip = bool(torch.equal(decoded_detail_from_stream, detail_codes))
                stream_semantic_roundtrip = bool(torch.equal(decoded_semantic_from_stream, indices))

                if not (token_roundtrip and detail_roundtrip and stream_detail_roundtrip and stream_semantic_roundtrip):
                    roundtrip_failures.append(
                        {
                            "index": int(global_index),
                            "path": str(dataset.paths[global_index]) if global_index < len(dataset.paths) else "",
                            "semantic_token_roundtrip": token_roundtrip,
                            "stream_semantic_roundtrip": stream_semantic_roundtrip,
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
                semantic_only_psnr = float(psnr(x_i, x_sem).item())
                semantic_only_l1 = float(torch.mean(torch.abs(x_sem - x_i)).item())
                semantic_only_ms_ssim = float(ms_ssim(x_sem, x_i, data_range=1.0, size_average=True).item())
                stage3_psnr = float(psnr(x_i, x_hat).item())
                stage3_l1 = float(torch.mean(torch.abs(x_hat - x_i)).item())
                stage3_ms_ssim = float(ms_ssim(x_hat, x_i, data_range=1.0, size_average=True).item())
                semantic_only_lpips_alex: float | None = None
                semantic_only_dists: float | None = None
                stage3_lpips_alex: float | None = None
                stage3_dists: float | None = None
                if perceptual is not None:
                    semantic_perceptual = perceptual(x_i, x_sem)
                    stage3_perceptual = perceptual(x_i, x_hat)
                    semantic_only_lpips_alex = semantic_perceptual.lpips_alex
                    semantic_only_dists = semantic_perceptual.dists
                    stage3_lpips_alex = stage3_perceptual.lpips_alex
                    stage3_dists = stage3_perceptual.dists
                semantic_topk_hit_rate = float(semantic_stats["topk_hit_rate"])
                residual_grid_abs_mean = float(torch.mean(torch.abs(residual_grid)).item())
                residual_grid_std = float(torch.std(residual_grid.float(), unbiased=False).item())
                residual_grid_clip_ratio = float(
                    torch.mean((torch.abs(residual_grid) >= float(args.detail_range)).float()).item()
                )
                semantic_roundtrip_ok = float(token_roundtrip and stream_semantic_roundtrip)
                detail_roundtrip_ok = float(detail_roundtrip and stream_detail_roundtrip)

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
                metrics["semantic_only_psnr"].append(semantic_only_psnr)
                metrics["semantic_only_l1"].append(semantic_only_l1)
                metrics["semantic_only_ms_ssim"].append(semantic_only_ms_ssim)
                metrics["stage3_psnr"].append(stage3_psnr)
                metrics["stage3_l1"].append(stage3_l1)
                metrics["stage3_ms_ssim"].append(stage3_ms_ssim)
                if args.compute_perceptual:
                    metrics["semantic_only_lpips_alex"].append(float(semantic_only_lpips_alex))
                    metrics["semantic_only_dists"].append(float(semantic_only_dists))
                    metrics["stage3_lpips_alex"].append(float(stage3_lpips_alex))
                    metrics["stage3_dists"].append(float(stage3_dists))
                metrics["semantic_topk_hit_rate"].append(semantic_topk_hit_rate)
                metrics["semantic_token_roundtrip"].append(semantic_roundtrip_ok)
                metrics["detail_code_roundtrip"].append(detail_roundtrip_ok)
                metrics["residual_grid_abs_mean"].append(residual_grid_abs_mean)
                metrics["residual_grid_std"].append(residual_grid_std)
                metrics["residual_grid_clip_ratio"].append(residual_grid_clip_ratio)
                metrics["detail_code_entropy_bits"].append(detail_entropy_bits)
                per_image_record: dict[str, object] = {
                    "index": int(global_index),
                    "path": str(dataset.paths[global_index]) if global_index < len(dataset.paths) else "",
                    "height": h,
                    "width": w,
                    "semantic_payload_bytes": int(len(semantic_payload)),
                    "detail_payload_bytes": int(len(detail_payload)),
                    "stage3_stream_bytes": int(len(stage3_stream)),
                    "semantic_payload_bpp": semantic_payload_bpp,
                    "detail_payload_bpp": detail_payload_bpp,
                    "actual_payload_bpp": actual_payload_bpp,
                    "paper_bpp": actual_payload_bpp,
                    "debug_semantic_only_full_stream_bpp": semantic_only_debug_bpp,
                    "debug_full_stream_bpp": stage3_debug_bpp,
                    "semantic_only_psnr": semantic_only_psnr,
                    "stage3_psnr": stage3_psnr,
                    "stage3_psnr_delta_vs_semantic_only": stage3_psnr - semantic_only_psnr,
                    "semantic_only_l1": semantic_only_l1,
                    "stage3_l1": stage3_l1,
                    "stage3_l1_delta_vs_semantic_only": stage3_l1 - semantic_only_l1,
                    "semantic_only_ms_ssim": semantic_only_ms_ssim,
                    "stage3_ms_ssim": stage3_ms_ssim,
                    "stage3_ms_ssim_delta_vs_semantic_only": stage3_ms_ssim - semantic_only_ms_ssim,
                    "semantic_topk_hit_rate": semantic_topk_hit_rate,
                    "semantic_token_roundtrip": bool(token_roundtrip and stream_semantic_roundtrip),
                    "detail_code_roundtrip": bool(detail_roundtrip and stream_detail_roundtrip),
                    "residual_grid_abs_mean": residual_grid_abs_mean,
                    "residual_grid_std": residual_grid_std,
                    "residual_grid_clip_ratio": residual_grid_clip_ratio,
                    "detail_code_entropy_bits": detail_entropy_bits,
                }
                if detail_hybrid_mode is not None:
                    per_image_record["detail_hybrid_semantic_mode"] = detail_hybrid_mode
                if args.compute_perceptual:
                    per_image_record.update(
                        {
                            "semantic_only_lpips_alex": semantic_only_lpips_alex,
                            "stage3_lpips_alex": stage3_lpips_alex,
                            "stage3_lpips_alex_delta_vs_semantic_only": float(stage3_lpips_alex)
                            - float(semantic_only_lpips_alex),
                            "semantic_only_dists": semantic_only_dists,
                            "stage3_dists": stage3_dists,
                            "stage3_dists_delta_vs_semantic_only": float(stage3_dists) - float(semantic_only_dists),
                        }
                    )
                if args.save_reconstructions and (
                    args.save_reconstruction_limit <= 0 or global_index < args.save_reconstruction_limit
                ):
                    original_path = str(dataset.paths[global_index]) if global_index < len(dataset.paths) else ""
                    image_name = safe_image_name(global_index, original_path)
                    reference_path = reconstruction_dirs["reference"] / image_name
                    semantic_path = reconstruction_dirs["semantic_only"] / image_name
                    stage3_path = reconstruction_dirs["stage3"] / image_name
                    save_image(x_i.detach().cpu(), reference_path)
                    save_image(x_sem.detach().cpu(), semantic_path)
                    save_image(x_hat.detach().cpu(), stage3_path)
                    manifest_record: dict[str, object] = {
                        "index": int(global_index),
                        "source_path": original_path,
                        "reference": str(reference_path),
                        "semantic_only": str(semantic_path),
                        "stage3": str(stage3_path),
                    }
                    if args.save_reconstruction_triptychs:
                        triptych_path = reconstruction_dirs["triptych"] / image_name
                        save_image(
                            torch.cat([x_i.detach().cpu(), x_sem.detach().cpu(), x_hat.detach().cpu()], dim=0),
                            triptych_path,
                            nrow=3,
                        )
                        manifest_record["triptych"] = str(triptych_path)
                    per_image_record.update(
                        {
                            "reference_image": str(reference_path),
                            "semantic_only_image": str(semantic_path),
                            "stage3_image": str(stage3_path),
                        }
                    )
                    if args.save_reconstruction_triptychs:
                        per_image_record["triptych_image"] = str(manifest_record["triptych"])
                    reconstruction_manifest.append(manifest_record)
                per_image_records.append(per_image_record)

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
        "protocol_default_crop_size": protocol_default_crop_size,
        "crop_size_matches_protocol_default": (
            None if protocol_default_crop_size is None else int(args.crop_size) == int(protocol_default_crop_size)
        ),
        "semantic_topk": int(semantic_code.topk),
        "semantic_topk_schedule": "prefix_replay_decoder_safe",
        "detail_downsample_factor": int(args.detail_downsample_factor),
        "detail_shape": [3, int(detail_hw), int(detail_hw)],
        "detail_bits": int(args.detail_bits),
        "detail_range": float(args.detail_range),
        "detail_gain": float(args.detail_gain),
        "decoder_postprocess": args.decoder_postprocess,
        "decoder_postprocess_strength": float(args.decoder_postprocess_strength),
        "detail_codec": args.detail_codec,
        "stream_header_codec": args.stream_header_codec,
        "stream_checksum_codec": args.stream_checksum_codec,
        "main_bpp_metric": "actual_payload_bpp_mean",
        "paper_bpp_metric": "paper_bpp_mean",
        "debug_bpp_metric": "debug_full_stream_bpp_mean",
        "eval_protocol": args.eval_protocol or "manual_roots",
        "eval_datasets": args.eval_dataset or [],
        "eval_image_roots": roots,
        "eval_protocol_summary": eval_protocol_summary or {},
        "deterministic_eval": not args.allow_nondeterministic_eval,
        "compute_perceptual": bool(args.compute_perceptual),
        "save_reconstructions": bool(args.save_reconstructions),
        "save_reconstruction_limit": int(args.save_reconstruction_limit),
        "save_reconstruction_triptychs": bool(args.save_reconstruction_triptychs),
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
    if args.compute_perceptual:
        summary["stage3_lpips_alex_delta_vs_semantic_only"] = (
            float(summary["stage3_lpips_alex_mean"]) - float(summary["semantic_only_lpips_alex_mean"])
        )
        summary["stage3_dists_delta_vs_semantic_only"] = (
            float(summary["stage3_dists_mean"]) - float(summary["semantic_only_dists_mean"])
        )
    summary["all_semantic_tokens_roundtrip"] = bool(all(v == 1.0 for v in metrics["semantic_token_roundtrip"]))
    summary["all_detail_codes_roundtrip"] = bool(all(v == 1.0 for v in metrics["detail_code_roundtrip"]))
    summary["roundtrip_failure_count"] = len(roundtrip_failures)
    summary["roundtrip_failures"] = roundtrip_failures[:20]

    per_image_path = out_dir / "per_image_metrics.jsonl"
    per_image_text = "\n".join(json.dumps(record, allow_nan=False) for record in per_image_records)
    per_image_path.write_text(per_image_text + ("\n" if per_image_text else ""))
    summary["per_image_metrics"] = str(per_image_path)
    reconstruction_manifest_path: Path | None = None
    if args.save_reconstructions:
        reconstruction_manifest_path = out_dir / "reconstructions" / "manifest.jsonl"
        manifest_text = "\n".join(json.dumps(record, allow_nan=False) for record in reconstruction_manifest)
        reconstruction_manifest_path.write_text(manifest_text + ("\n" if manifest_text else ""))
        summary["reconstruction_manifest"] = str(reconstruction_manifest_path)
        summary["reconstruction_count"] = len(reconstruction_manifest)
        summary["reconstruction_dirs"] = {key: str(value) for key, value in reconstruction_dirs.items()}

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
            "per_image_metrics": str(per_image_path),
            "sample": str(out_dir / "stage3_uniform_residual_grid.png"),
        },
    }
    if reconstruction_manifest_path is not None:
        payload["artifacts"]["reconstruction_manifest"] = str(reconstruction_manifest_path)
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))
    if roundtrip_failures and not args.allow_roundtrip_failures:
        raise RuntimeError(f"roundtrip failed for {len(roundtrip_failures)} images; see {summary_path}")


if __name__ == "__main__":
    main()
