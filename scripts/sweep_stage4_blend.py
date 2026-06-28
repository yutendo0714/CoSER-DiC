from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from coserdic.metrics import PerceptualMetricBundle


def list_images(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y).pow(2)).clamp_min(1.0e-12)
    return float((-10.0 * torch.log10(mse)).item())


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def image_to_uint8(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x.clamp(0, 1) * 255.0).to(torch.uint8)


def update_patch_fid(
    fid: FrechetInceptionDistance,
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    patch_size: int,
    patch_num: int,
) -> int:
    if patch_size <= 0:
        return 0
    count = 0
    offsets = [0]
    unit = patch_size // patch_num
    for unit_i in range(1, patch_num):
        limit_size = (2.0 - unit_i / patch_num) * patch_size
        if reference.shape[-2] >= limit_size and reference.shape[-1] >= limit_size:
            offsets.append(unit * unit_i)
    for offset in offsets:
        ref_crop = reference[:, :, offset:, offset:]
        rec_crop = reconstruction[:, :, offset:, offset:]
        ref_patches = (
            F.unfold(ref_crop, kernel_size=patch_size, stride=patch_size)
            .permute(0, 2, 1)
            .reshape(-1, 3, patch_size, patch_size)
        )
        rec_patches = (
            F.unfold(rec_crop, kernel_size=patch_size, stride=patch_size)
            .permute(0, 2, 1)
            .reshape(-1, 3, patch_size, patch_size)
        )
        fid.update(image_to_uint8(ref_patches), real=True)
        fid.update(image_to_uint8(rec_patches), real=False)
        count += int(ref_patches.shape[0])
    return count


@torch.no_grad()
def evaluate_alpha(
    *,
    reference_dir: Path,
    stage3_dir: Path,
    stage4_dir: Path,
    output_dir: Path,
    alpha: float,
    device: torch.device,
    fid_patch_size: int,
    fid_patch_num: int,
    save_images: bool,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    perceptual = PerceptualMetricBundle().to(device)
    fid = FrechetInceptionDistance().to(device)
    fid.set_dtype(torch.float32)
    if save_images:
        (output_dir / "reconstructions" / f"alpha_{alpha:.3f}").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for ref_path in tqdm(list_images(reference_dir), desc=f"alpha={alpha:.3f}"):
        stage3_path = stage3_dir / ref_path.name
        stage4_path = stage4_dir / ref_path.name
        reference = load_image(ref_path, device)
        stage3 = load_image(stage3_path, device)
        stage4 = load_image(stage4_path, device)
        blend = ((1.0 - alpha) * stage3 + alpha * stage4).clamp(0, 1)
        perceptual_result = perceptual(reference, blend)
        patch_count = update_patch_fid(
            fid,
            reference,
            blend,
            patch_size=fid_patch_size,
            patch_num=fid_patch_num,
        )
        row = {
            "name": ref_path.name,
            "alpha": alpha,
            "psnr": psnr(reference, blend),
            "ms_ssim": float(ms_ssim(blend, reference, data_range=1.0, size_average=True).item()),
            "lpips_alex": perceptual_result.lpips_alex,
            "dists": perceptual_result.dists,
            "fid_patch_count": patch_count,
        }
        rows.append(row)
        if save_images:
            save_image(blend.detach().cpu(), output_dir / "reconstructions" / f"alpha_{alpha:.3f}" / ref_path.name)
    summary: dict[str, float | int] = {
        "alpha": alpha,
        "count": len(rows),
        "psnr_mean": mean([float(row["psnr"]) for row in rows]),
        "ms_ssim_mean": mean([float(row["ms_ssim"]) for row in rows]),
        "lpips_alex_mean": mean([float(row["lpips_alex"]) for row in rows]),
        "dists_mean": mean([float(row["dists"]) for row in rows]),
        "fid": float(fid.compute().item()),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--stage3-dir", required=True)
    parser.add_argument("--stage4-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="results/stage4_blend_sweep")
    parser.add_argument("--alphas", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--actual-payload-bpp", type=float, default=0.0)
    parser.add_argument("--fid-patch-size", type=int, default=64)
    parser.add_argument("--fid-patch-num", type=int, default=2)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; stop blend sweep until the container is restarted.")

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_rows = []
    for alpha in [float(item) for item in args.alphas.split(",") if item.strip()]:
        summary, rows = evaluate_alpha(
            reference_dir=Path(args.reference_dir),
            stage3_dir=Path(args.stage3_dir),
            stage4_dir=Path(args.stage4_dir),
            output_dir=out_dir,
            alpha=alpha,
            device=torch.device("cuda"),
            fid_patch_size=args.fid_patch_size,
            fid_patch_num=args.fid_patch_num,
            save_images=args.save_images,
        )
        summary["actual_payload_bpp_mean"] = args.actual_payload_bpp
        summary["payload_policy"] = "deterministic decoder-side blend; no transmitted side information"
        summaries.append(summary)
        all_rows.extend(rows)
    (out_dir / "summary.json").write_text(json.dumps({"runs": summaries}, indent=2, sort_keys=True) + "\n")
    (out_dir / "per_image_metrics.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in all_rows) + "\n"
    )
    doc_lines = [
        f"# {args.run_name}",
        "",
        f"Date: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        "| alpha | bpp | PSNR | MS-SSIM | LPIPS-Alex | DISTS | patch-FID |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        doc_lines.append(
            "| {alpha:.3f} | {actual_payload_bpp_mean:.6f} | {psnr_mean:.4f} | {ms_ssim_mean:.4f} | "
            "{lpips_alex_mean:.4f} | {dists_mean:.4f} | {fid:.4f} |".format(**summary)
        )
    doc_lines.extend(
        [
            "",
            "Payload policy: deterministic decoder-side blend; no extra actual_payload_bpp.",
            "",
            "Artifacts:",
            "",
            f"- summary: `{out_dir / 'summary.json'}`",
            f"- per-image: `{out_dir / 'per_image_metrics.jsonl'}`",
        ]
    )
    Path("docs/experiments").mkdir(parents=True, exist_ok=True)
    (Path("docs/experiments") / f"{args.run_name}.md").write_text("\n".join(doc_lines) + "\n")
    print(json.dumps({"runs": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
