from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as TF
from tqdm import tqdm

from coserdic.metrics import PerceptualMetricBundle


def prepare_input_from_manifest(manifest: Path, input_dir: Path, limit: int) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    for row in rows:
        reference = Path(row["reference"])
        shutil.copy2(reference, input_dir / reference.name)


def run_official_cli(
    *,
    repo: Path,
    module: str,
    checkpoint: Path,
    config: Path,
    mode: str,
    input_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-m",
        module,
        mode,
        "--ckpt",
        str(checkpoint.resolve()),
        "--config",
        str(config.resolve()),
        "--input",
        str(input_dir.resolve()),
        "--output",
        str(output_dir.resolve()),
    ]
    subprocess.run(cmd, cwd=repo, env=env, check=True)


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
    patches = 0
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
        patches += int(ref_patches.shape[0])
    return patches


@torch.no_grad()
def evaluate_reconstructions(
    *,
    input_dir: Path,
    reconstruction_dir: Path,
    bitstream_dir: Path,
    device: torch.device,
    fid_patch_size: int,
    fid_patch_num: int,
    codec_name: str,
    header_bytes: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    perceptual = PerceptualMetricBundle().to(device)
    fid = FrechetInceptionDistance().to(device)
    fid.set_dtype(torch.float32)
    rows: list[dict[str, float | int | str]] = []
    image_paths = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg"))
    for image_path in tqdm(image_paths, desc="metrics"):
        reconstruction_path = reconstruction_dir / image_path.name
        cod_path = bitstream_dir / f"{image_path.name}.cod"
        reference = TF.to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        reconstruction = TF.to_tensor(Image.open(reconstruction_path).convert("RGB")).unsqueeze(0).to(device)
        _, _, h, w = reference.shape
        cod_file_bytes = cod_path.stat().st_size
        cod_payload_bytes = max(0, cod_file_bytes - header_bytes)
        perceptual_result = perceptual(reference, reconstruction)
        patch_count = update_patch_fid(
            fid,
            reference,
            reconstruction,
            patch_size=fid_patch_size,
            patch_num=fid_patch_num,
        )
        rows.append(
            {
                "name": image_path.name,
                "width": w,
                "height": h,
                "actual_payload_bpp": 8.0 * cod_payload_bytes / (h * w),
                "cod_payload_bpp": 8.0 * cod_payload_bytes / (h * w),
                "codec_file_bpp": 8.0 * cod_file_bytes / (h * w),
                "cod_payload_bytes": cod_payload_bytes,
                "cod_file_bytes": cod_file_bytes,
                "psnr": psnr(reference, reconstruction),
                "ms_ssim": float(ms_ssim(reconstruction, reference, data_range=1.0, size_average=True).item()),
                "lpips_alex": perceptual_result.lpips_alex,
                "dists": perceptual_result.dists,
                "fid_patch_count": patch_count,
            }
        )
    summary: dict[str, float | int | str] = {
        "count": len(rows),
        "actual_payload_bpp_mean": mean([float(row["actual_payload_bpp"]) for row in rows]),
        "cod_payload_bpp_mean": mean([float(row["cod_payload_bpp"]) for row in rows]),
        "codec_file_bpp_mean": mean([float(row["codec_file_bpp"]) for row in rows]),
        "psnr_mean": mean([float(row["psnr"]) for row in rows]),
        "ms_ssim_mean": mean([float(row["ms_ssim"]) for row in rows]),
        "lpips_alex_mean": mean([float(row["lpips_alex"]) for row in rows]),
        "dists_mean": mean([float(row["dists"]) for row in rows]),
        "fid": float(fid.compute().item()) if rows and fid_patch_size > 0 else float("nan"),
        "fid_patch_size": fid_patch_size,
        "fid_patch_num": fid_patch_num,
        "payload_policy": (
            f"{codec_name} .cod size minus {header_bytes}-byte width/height header; "
            "fixed model/codebook weights excluded."
        ),
    }
    return summary, rows


def write_run_doc(path: Path, payload: dict[str, object]) -> None:
    lines = [
        f"# {payload['run_name']}",
        "",
        f"Date: {payload['date']}",
        "",
        "## Command",
        "",
        "```bash",
        str(payload["command"]),
        "```",
        "",
        "## Summary",
        "",
    ]
    for key, value in dict(payload["summary"]).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Artifacts", ""])
    for key, value in dict(payload["artifacts"]).items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cod-lite-repo", default="external/repos/GenCodec/CoD_Lite")
    parser.add_argument("--official-repo", default="", help="Overrides --cod-lite-repo for non-CoD-Lite GenCodec CLIs.")
    parser.add_argument("--official-module", default="finetuned_one_step_codec.inference")
    parser.add_argument("--codec-name", default="CoD-Lite")
    parser.add_argument("--header-bytes", type=int, default=4)
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="results/baselines/cod_lite_official")
    parser.add_argument("--fid-patch-size", type=int, default=64)
    parser.add_argument("--fid-patch-num", type=int, default=2)
    parser.add_argument("--skip-codec", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; stop baseline evaluation until the container is restarted.")

    out_dir = Path(args.output_dir) / args.run_name
    official_repo = Path(args.official_repo or args.cod_lite_repo)
    input_dir = Path(args.input_dir) if args.input_dir else out_dir / "input"
    if args.manifest:
        if input_dir.exists() and not any(input_dir.iterdir()):
            input_dir.rmdir()
        if not input_dir.exists():
            prepare_input_from_manifest(Path(args.manifest), input_dir, args.limit)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    bitstream_dir = out_dir / "bitstreams"
    reconstruction_dir = out_dir / "reconstructions"
    if not args.skip_codec:
        run_official_cli(
            repo=official_repo,
            module=args.official_module,
            checkpoint=Path(args.checkpoint),
            config=Path(args.config),
            mode="compress",
            input_dir=input_dir,
            output_dir=bitstream_dir,
        )
        run_official_cli(
            repo=official_repo,
            module=args.official_module,
            checkpoint=Path(args.checkpoint),
            config=Path(args.config),
            mode="decompress",
            input_dir=bitstream_dir,
            output_dir=reconstruction_dir,
        )

    summary, rows = evaluate_reconstructions(
        input_dir=input_dir,
        reconstruction_dir=reconstruction_dir,
        bitstream_dir=bitstream_dir,
        device=torch.device("cuda"),
        fid_patch_size=args.fid_patch_size,
        fid_patch_num=args.fid_patch_num,
        codec_name=args.codec_name,
        header_bytes=args.header_bytes,
    )
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "input_dir": str(input_dir),
            "reconstruction_dir": str(reconstruction_dir),
            "bitstream_dir": str(bitstream_dir),
            "official_repo": str(official_repo),
            "official_module": args.official_module,
            "codec_name": args.codec_name,
            "header_bytes": args.header_bytes,
            "main_bpp_metric": "actual_payload_bpp_mean",
        }
    )
    summary_path = out_dir / "summary.json"
    per_image_path = out_dir / "per_image_metrics.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    per_image_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    write_run_doc(
        Path("docs/experiments") / f"{args.run_name}.md",
        {
            "run_name": args.run_name,
            "date": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "summary": summary,
            "artifacts": {
                "summary": str(summary_path),
                "per_image": str(per_image_path),
                "bitstreams": str(bitstream_dir),
                "reconstructions": str(reconstruction_dir),
            },
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
