from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision.transforms import functional as TF
from tqdm import tqdm

import compressai.zoo as zoo
from coserdic.datasets.eval_protocols import (
    EVAL_PROTOCOLS,
    flatten_selection_paths,
    protocol_summary,
    resolve_eval_protocol,
)
from coserdic.datasets.image_folder import list_images
from coserdic.metrics import PerceptualMetricBundle
from coserdic.metrics import bytes_to_bpp
from coserdic.utils import crop_to_shape, pad_to_multiple


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y).pow(2)).item()
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def num_stream_bytes(strings) -> int:
    total = 0
    for stream_group in strings:
        if isinstance(stream_group, (bytes, bytearray)):
            total += len(stream_group)
        else:
            total += sum(len(s) for s in stream_group)
    return total


def load_model(name: str, quality: int, device: torch.device) -> torch.nn.Module:
    if not hasattr(zoo, name):
        raise ValueError(f"unknown CompressAI model: {name}")
    model_fn = getattr(zoo, name)
    model = model_fn(quality=quality, pretrained=True).to(device).eval()
    model.update(force=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/dpl/kodak")
    parser.add_argument("--eval-protocol", choices=sorted(EVAL_PROTOCOLS), default="")
    parser.add_argument("--eval-dataset", action="append", default=None)
    parser.add_argument("--dpl-root", default="/dpl")
    parser.add_argument("--allow-protocol-count-mismatch", action="store_true")
    parser.add_argument("--model", default="bmshj2018_hyperprior")
    parser.add_argument("--quality", type=int, default=1)
    parser.add_argument("--pad-multiple", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="results/baselines/compressai_kodak_q1.json")
    parser.add_argument("--save-recon-dir", default="")
    parser.add_argument("--compute-perceptual", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model, args.quality, device)
    perceptual = PerceptualMetricBundle().to(device).eval() if args.compute_perceptual else None

    eval_protocol_summary: dict[str, object] | None = None
    if args.eval_protocol:
        selections = resolve_eval_protocol(
            args.eval_protocol,
            dpl_root=args.dpl_root,
            dataset_keys=args.eval_dataset,
            strict_expected_counts=not args.allow_protocol_count_mismatch,
        )
        eval_protocol_summary = protocol_summary(args.eval_protocol, selections)
        image_paths = flatten_selection_paths(selections)
    else:
        image_paths = list_images(args.dataset)
    if args.limit:
        image_paths = image_paths[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recon_dir = Path(args.save_recon_dir) if args.save_recon_dir else None
    if recon_dir:
        recon_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in tqdm(image_paths, desc=f"{args.model}-q{args.quality}"):
        image = Image.open(path).convert("RGB")
        width, height = image.size
        x = TF.to_tensor(image).unsqueeze(0).to(device)
        padded, original_shape = pad_to_multiple(x, args.pad_multiple)

        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t0 = time.perf_counter()
        compressed = model.compress(padded)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        encode_time = time.perf_counter() - t0

        bitstream_bytes = num_stream_bytes(compressed["strings"])
        actual_bpp = bytes_to_bpp(bitstream_bytes, height, width)

        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t1 = time.perf_counter()
        decompressed = model.decompress(compressed["strings"], compressed["shape"])
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        decode_time = time.perf_counter() - t1

        x_hat = crop_to_shape(decompressed["x_hat"].clamp(0, 1), original_shape)
        x_ref = crop_to_shape(x, original_shape)

        with torch.no_grad():
            row = {
                "image": path.name,
                "height": height,
                "width": width,
                "bitstream_bytes": bitstream_bytes,
                "actual_payload_bpp": actual_bpp,
                "psnr_rgb": psnr(x_ref, x_hat),
                "ms_ssim_rgb": float(ms_ssim(x_ref, x_hat, data_range=1.0).item()),
                "encode_time_sec": encode_time,
                "decode_time_sec": decode_time,
                "model": args.model,
                "quality": args.quality,
            }
            if perceptual is not None:
                perceptual_result = perceptual(x_ref, x_hat)
                row["lpips_alex"] = perceptual_result.lpips_alex
                row["dists"] = perceptual_result.dists
        rows.append(row)

        if recon_dir:
            TF.to_pil_image(x_hat.squeeze(0).cpu()).save(recon_dir / path.name)

    summary = {
        "model": args.model,
        "quality": args.quality,
        "dataset": str(Path(args.dataset)) if not args.eval_protocol else args.eval_protocol,
        "eval_protocol": args.eval_protocol or "manual_dataset",
        "eval_datasets": args.eval_dataset or [],
        "eval_protocol_summary": eval_protocol_summary or {},
        "compute_perceptual": bool(args.compute_perceptual),
        "num_images": len(rows),
        "mean_actual_payload_bpp": sum(r["actual_payload_bpp"] for r in rows) / max(1, len(rows)),
        "paper_bpp_metric": "actual_payload_bpp",
        "mean_psnr_rgb": sum(r["psnr_rgb"] for r in rows) / max(1, len(rows)),
        "mean_ms_ssim_rgb": sum(r["ms_ssim_rgb"] for r in rows) / max(1, len(rows)),
        "rows": rows,
    }
    if args.compute_perceptual:
        summary["mean_lpips_alex"] = sum(r["lpips_alex"] for r in rows) / max(1, len(rows))
        summary["mean_dists"] = sum(r["dists"] for r in rows) / max(1, len(rows))
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
