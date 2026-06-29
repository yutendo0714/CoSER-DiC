from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from coserdic.entropy import (
    StaticResidualGridHuffmanCode,
    StaticResidualGridHybridHuffmanCode,
    StaticResidualGridPositionHuffmanCode,
    StaticResidualGridSemanticPositionLeftContextHuffmanCode,
    StaticResidualGridSemanticPositionHuffmanCode,
    UniformResidualGridCode,
)
from coserdic.models import (
    DECODER_POSTPROCESS_MODES,
    SemanticVQAutoEncoder,
    SemanticVQConfig,
    apply_decoder_postprocess,
)
from coserdic.utils import seed_everything

from scripts.eval_stage3_uniform_residual_bitstream import (
    CenterCropImageDataset,
    decode_indices_to_latent,
    load_config_from_checkpoint,
    mean,
    psnr,
    safe_image_name,
    write_run_doc,
)


def residual_quantizer_name(residual_code) -> str:
    if isinstance(residual_code, StaticResidualGridHybridHuffmanCode):
        return str(residual_code.position_code.quantizer.quantizer)
    quantizer = getattr(residual_code, "quantizer", "uniform")
    if isinstance(quantizer, str):
        return quantizer
    return str(getattr(quantizer, "quantizer", "uniform"))


def load_residual_code(args: argparse.Namespace):
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
        if str(args.detail_quantizer) != residual_quantizer_name(residual_code):
            raise ValueError("--detail-quantizer does not match residual Huffman prior")
        return residual_code
    return UniformResidualGridCode(
        bits=args.detail_bits,
        value_range=args.detail_range,
        codec=args.detail_codec,
        quantizer=args.detail_quantizer,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--output-dir", default="results/bitstreams/stage3_training_cache_fast")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=2048)
    parser.add_argument("--shuffle-images", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--detail-downsample-factor", type=int, default=32)
    parser.add_argument("--detail-bits", type=int, default=4)
    parser.add_argument("--detail-range", type=float, default=0.25)
    parser.add_argument("--detail-quantizer", choices=["uniform", "zero_centered"], default="uniform")
    parser.add_argument("--detail-gain", type=float, default=1.0)
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
        default="semantic_position_leftctx_huffman",
    )
    parser.add_argument("--detail-prior", default="")
    parser.add_argument("--decoder-postprocess", choices=DECODER_POSTPROCESS_MODES, default="none")
    parser.add_argument("--decoder-postprocess-strength", type=float, default=0.0)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    if args.crop_size % args.detail_downsample_factor != 0:
        raise ValueError("--crop-size must be divisible by --detail-downsample-factor")
    if args.detail_gain < 0:
        raise ValueError("--detail-gain must be non-negative")
    if args.decoder_postprocess == "none" and abs(float(args.decoder_postprocess_strength)) > 1.0e-12:
        raise ValueError("--decoder-postprocess-strength must be 0 when postprocess is none")
    if args.decoder_postprocess != "none" and float(args.decoder_postprocess_strength) <= 0.0:
        raise ValueError("--decoder-postprocess-strength must be positive when postprocess is enabled")
    if not args.image_root:
        raise ValueError("pass at least one --image-root")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 3 cache export.")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = load_config_from_checkpoint(checkpoint, args.config or None)
    seed_everything(int(raw.get("experiment", {}).get("seed", 42)), deterministic=True)
    cfg = SemanticVQConfig(**checkpoint["semantic_vq_config"])
    model = SemanticVQAutoEncoder(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    residual_code = load_residual_code(args)
    detail_hw = args.crop_size // args.detail_downsample_factor
    dataset = CenterCropImageDataset(
        args.image_root,
        crop_size=args.crop_size,
        limit=args.max_images,
        shuffle_images=args.shuffle_images,
        shuffle_seed=args.shuffle_seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    run_name = args.run_name or f"{checkpoint_path.stem}_stage3_traincache_fast_{len(dataset)}"
    out_dir = Path(args.output_dir) / run_name
    reconstruction_root = out_dir / "reconstructions"
    reconstruction_dirs = {
        "reference": reconstruction_root / "reference",
        "semantic_only": reconstruction_root / "semantic_only",
        "stage3": reconstruction_root / "stage3",
    }
    feature_cache_dir = out_dir / "decoder_feature_cache"
    for directory in [*reconstruction_dirs.values(), feature_cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = reconstruction_root / "manifest.jsonl"
    per_image_path = out_dir / "per_image_metrics.jsonl"
    metrics: dict[str, list[float]] = {
        "semantic_only_psnr": [],
        "semantic_only_l1": [],
        "semantic_only_ms_ssim": [],
        "stage3_psnr": [],
        "stage3_l1": [],
        "stage3_ms_ssim": [],
        "residual_grid_abs_mean": [],
        "residual_grid_std": [],
        "residual_grid_clip_ratio": [],
    }
    sample_written = False
    count = 0

    with manifest_path.open("w") as manifest_file, per_image_path.open("w") as per_image_file:
        with torch.no_grad():
            for batch_index, x_cpu in enumerate(tqdm(loader, desc=run_name)):
                x = x_cpu.to(device, non_blocking=True)
                out = model(x, quantize_mix=1.0)
                indices_batch = out["indices"].detach().cpu()
                semantic_latent_batch = decode_indices_to_latent(model, cfg, indices_batch).to(
                    device=device,
                    dtype=next(model.parameters()).dtype,
                )
                x_sem_batch = model.decoder(semantic_latent_batch).clamp(0, 1)
                for image_index in range(x.shape[0]):
                    global_index = batch_index * args.batch_size + image_index
                    original_path = str(dataset.paths[global_index]) if global_index < len(dataset.paths) else ""
                    image_name = safe_image_name(global_index, original_path)
                    x_i = x[image_index : image_index + 1]
                    x_sem = x_sem_batch[image_index : image_index + 1]
                    semantic_latent = semantic_latent_batch[image_index : image_index + 1]
                    decoded_indices = indices_batch[image_index]

                    residual = x_i - x_sem
                    residual_grid = F.adaptive_avg_pool2d(residual, output_size=(detail_hw, detail_hw)).squeeze(0)
                    detail_codes = residual_code.quantize(residual_grid)
                    residual_grid_hat = residual_code.dequantize(detail_codes).unsqueeze(0).to(
                        device=device,
                        dtype=x_sem.dtype,
                    )
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

                    reference_path = reconstruction_dirs["reference"] / image_name
                    semantic_path = reconstruction_dirs["semantic_only"] / image_name
                    stage3_path = reconstruction_dirs["stage3"] / image_name
                    cache_path = feature_cache_dir / f"{Path(image_name).stem}.pt"
                    save_image(x_i.detach().cpu(), reference_path)
                    save_image(x_sem.detach().cpu(), semantic_path)
                    save_image(x_hat.detach().cpu(), stage3_path)
                    torch.save(
                        {
                            "index": int(global_index),
                            "source_path": original_path,
                            "semantic_indices": decoded_indices.detach().cpu().to(torch.int16),
                            "semantic_latent": semantic_latent.detach().cpu().to(torch.bfloat16),
                            "residual_grid_hat": residual_grid_hat.detach().cpu().to(torch.bfloat16),
                            "detail_codes": detail_codes.detach().cpu().to(torch.int16),
                            "detail_bits": int(args.detail_bits),
                            "detail_levels": int(residual_code.levels),
                            "detail_range": float(args.detail_range),
                            "detail_quantizer": str(args.detail_quantizer),
                            "detail_downsample_factor": int(args.detail_downsample_factor),
                            "detail_codec": str(args.detail_codec),
                            "semantic_shape": tuple(int(v) for v in decoded_indices.shape),
                            "semantic_latent_shape": tuple(int(v) for v in semantic_latent.shape),
                            "detail_shape": tuple(int(v) for v in detail_codes.shape),
                            "cache_mode": "fast_decoded_equivalent_no_entropy_roundtrip",
                        },
                        cache_path,
                    )

                    semantic_only_psnr = float(psnr(x_i, x_sem).item())
                    semantic_only_l1 = float(torch.mean(torch.abs(x_sem - x_i)).item())
                    semantic_only_ms_ssim = float(ms_ssim(x_sem, x_i, data_range=1.0, size_average=True).item())
                    stage3_psnr = float(psnr(x_i, x_hat).item())
                    stage3_l1 = float(torch.mean(torch.abs(x_hat - x_i)).item())
                    stage3_ms_ssim = float(ms_ssim(x_hat, x_i, data_range=1.0, size_average=True).item())
                    residual_grid_abs_mean = float(torch.mean(torch.abs(residual_grid)).item())
                    residual_grid_std = float(torch.std(residual_grid.float(), unbiased=False).item())
                    residual_grid_clip_ratio = float(
                        torch.mean((torch.abs(residual_grid) >= float(args.detail_range)).float()).item()
                    )
                    metrics["semantic_only_psnr"].append(semantic_only_psnr)
                    metrics["semantic_only_l1"].append(semantic_only_l1)
                    metrics["semantic_only_ms_ssim"].append(semantic_only_ms_ssim)
                    metrics["stage3_psnr"].append(stage3_psnr)
                    metrics["stage3_l1"].append(stage3_l1)
                    metrics["stage3_ms_ssim"].append(stage3_ms_ssim)
                    metrics["residual_grid_abs_mean"].append(residual_grid_abs_mean)
                    metrics["residual_grid_std"].append(residual_grid_std)
                    metrics["residual_grid_clip_ratio"].append(residual_grid_clip_ratio)

                    manifest_record = {
                        "index": int(global_index),
                        "source_path": original_path,
                        "reference": str(reference_path),
                        "semantic_only": str(semantic_path),
                        "stage3": str(stage3_path),
                        "decoder_feature_cache": str(cache_path),
                        "cache_mode": "fast_decoded_equivalent_no_entropy_roundtrip",
                    }
                    per_image_record = {
                        "index": int(global_index),
                        "path": original_path,
                        "semantic_only_psnr": semantic_only_psnr,
                        "semantic_only_l1": semantic_only_l1,
                        "semantic_only_ms_ssim": semantic_only_ms_ssim,
                        "stage3_psnr": stage3_psnr,
                        "stage3_l1": stage3_l1,
                        "stage3_ms_ssim": stage3_ms_ssim,
                        "residual_grid_abs_mean": residual_grid_abs_mean,
                        "residual_grid_std": residual_grid_std,
                        "residual_grid_clip_ratio": residual_grid_clip_ratio,
                        "cache_mode": "fast_decoded_equivalent_no_entropy_roundtrip",
                    }
                    manifest_file.write(json.dumps(manifest_record, allow_nan=False) + "\n")
                    per_image_file.write(json.dumps(per_image_record, allow_nan=False) + "\n")
                    count += 1

                    if not sample_written:
                        sample_path = out_dir / "stage3_training_cache_fast_grid.png"
                        save_image(torch.cat([x_i.detach().cpu(), x_sem.detach().cpu(), x_hat.detach().cpu()], dim=0), sample_path, nrow=3)
                        sample_written = True

    summary = {f"{key}_mean": mean(value) for key, value in metrics.items()}
    summary.update(
        {
            "cache_mode": "fast_decoded_equivalent_no_entropy_roundtrip",
            "count": int(count),
            "crop_size": int(args.crop_size),
            "detail_downsample_factor": int(args.detail_downsample_factor),
            "detail_bits": int(args.detail_bits),
            "detail_range": float(args.detail_range),
            "detail_quantizer": str(args.detail_quantizer),
            "detail_gain": float(args.detail_gain),
            "actual_payload_bpp": "",
            "paper_bpp": "",
            "bpp_note": "not computed; this cache is for Stage4 training only. Use eval_stage3_uniform_residual_bitstream.py for actual payload bpp.",
        }
    )
    summary_path = out_dir / "summary.json"
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
        "artifacts": {
            "summary": str(summary_path),
            "per_image_metrics": str(per_image_path),
            "reconstruction_manifest": str(manifest_path),
            "decoder_feature_cache": str(feature_cache_dir),
            "sample": str(out_dir / "stage3_training_cache_fast_grid.png"),
        },
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
