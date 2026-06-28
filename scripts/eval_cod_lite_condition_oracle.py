from __future__ import annotations

import argparse
import json
import sys
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
from coserdic.models import CoDLiteOneStepBackbone, CoDLiteOneStepBackboneConfig
from coserdic.utils import seed_everything
from coserdic.utils.wandb_utils import init_wandb


def load_manifest(path: Path, limit: int) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def load_metrics(path: Path | None) -> dict[int, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {int(row.get("index", idx)): row for idx, row in enumerate(rows)}


def load_image(path: Path, crop_size: int, device: torch.device) -> torch.Tensor:
    tensor = TF.to_tensor(Image.open(path).convert("RGB"))
    if crop_size > 0:
        _, h, w = tensor.shape
        if h < crop_size or w < crop_size:
            scale = crop_size / min(h, w)
            tensor = TF.resize(
                tensor,
                [max(crop_size, int(round(h * scale))), max(crop_size, int(round(w * scale)))],
                antialias=True,
            )
            _, h, w = tensor.shape
        top = max(0, (h - crop_size) // 2)
        left = max(0, (w - crop_size) // 2)
        tensor = TF.crop(tensor, top, left, crop_size, crop_size)
    return tensor.unsqueeze(0).to(device)


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


def candidate_specs(modes: str, lerp_betas: str) -> list[tuple[str, float | None]]:
    specs: list[tuple[str, float | None]] = []
    for mode in [item.strip() for item in modes.split(",") if item.strip()]:
        if mode == "cond_lerp":
            for beta in [float(item) for item in lerp_betas.split(",") if item.strip()]:
                specs.append((f"cond_lerp_{beta:.3f}", beta))
        else:
            specs.append((mode, None))
    return specs


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--per-image-metrics", default="")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="results/stage4_cod_lite_condition_oracle")
    parser.add_argument("--cod-lite-checkpoint", default="external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt")
    parser.add_argument("--cod-lite-config", default="external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml")
    parser.add_argument("--cod-lite-repo", default="external/repos/GenCodec/CoD_Lite")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--modes",
        default="stage3_native,semantic_native,reference_native,cond_lerp",
        help="Comma-separated modes. cond_lerp expands using --lerp-betas.",
    )
    parser.add_argument("--lerp-betas", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--fid-patch-size", type=int, default=64)
    parser.add_argument("--fid-patch-num", type=int, default=2)
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--wandb-mode", default="offline")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; stop CoD-Lite oracle evaluation until the container is restarted.")

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda")
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(Path(args.manifest), args.limit)
    metric_rows = load_metrics(Path(args.per_image_metrics) if args.per_image_metrics else None)
    specs = candidate_specs(args.modes, args.lerp_betas)

    backbone = CoDLiteOneStepBackbone.load(
        CoDLiteOneStepBackboneConfig(
            repo_root=args.cod_lite_repo,
            checkpoint_path=args.cod_lite_checkpoint,
            config_path=args.cod_lite_config,
            freeze=True,
        ),
        device=device,
    )
    backbone.eval()
    perceptual = PerceptualMetricBundle().to(device)
    fids = {name: FrechetInceptionDistance().to(device) for name, _ in specs}
    for fid in fids.values():
        fid.set_dtype(torch.float32)

    wandb_run = init_wandb(
        {
            "stage": "stage4_cod_lite_condition_oracle",
            "manifest": args.manifest,
            "per_image_metrics": args.per_image_metrics,
            "cod_lite_checkpoint": args.cod_lite_checkpoint,
            "cod_lite_config": args.cod_lite_config,
            "crop_size": args.crop_size,
            "limit": args.limit,
            "modes": args.modes,
            "lerp_betas": args.lerp_betas,
            "wandb": {"mode": args.wandb_mode, "project": "coserdic"},
        },
        run_name=args.run_name,
    )

    per_image: list[dict[str, object]] = []
    metrics: dict[str, dict[str, list[float]]] = {
        name: {
            "actual_payload_bpp": [],
            "psnr": [],
            "ms_ssim": [],
            "lpips_alex": [],
            "dists": [],
            "l1": [],
            "condition_l1_to_reference": [],
        }
        for name, _ in specs
    }
    if args.save_reconstructions:
        for name, _ in specs:
            (out_dir / "reconstructions" / name).mkdir(parents=True, exist_ok=True)

    for item_index, row in enumerate(tqdm(rows, desc=args.run_name)):
        index = int(row.get("index", item_index))
        reference = load_image(Path(str(row["reference"])), args.crop_size, device)
        semantic = load_image(Path(str(row["semantic_only"])), args.crop_size, device)
        stage3 = load_image(Path(str(row["stage3"])), args.crop_size, device)
        metric_row = metric_rows.get(index, {})
        payload_bpp = float(metric_row.get("actual_payload_bpp", 0.0))

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cond_reference = backbone.native_condition(reference)
            cond_stage3 = backbone.native_condition(stage3)
            cond_semantic = backbone.native_condition(semantic)

        for name, beta in specs:
            if name == "stage3_native" or (beta is not None and beta == 0.0):
                cond = cond_stage3
                oracle_policy = "decoder-available: CoD-Lite condition computed from Stage 3 reconstruction"
            elif name == "semantic_native":
                cond = cond_semantic
                oracle_policy = "decoder-available: CoD-Lite condition computed from semantic-only reconstruction"
            elif name == "reference_native" or (beta is not None and beta == 1.0):
                cond = cond_reference
                oracle_policy = "oracle: uses reference-derived condition; not valid as paper metric"
            elif beta is not None:
                cond = cond_stage3 + beta * (cond_reference - cond_stage3)
                oracle_policy = "oracle: interpolates toward reference-derived condition; not valid as paper metric"
            else:
                raise ValueError(f"unknown candidate: {name}")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                reconstruction = backbone(stage3, cond)
            p = perceptual(reference, reconstruction)
            patch_count = update_patch_fid(
                fids[name],
                reference,
                reconstruction,
                patch_size=args.fid_patch_size,
                patch_num=args.fid_patch_num,
            )
            row_out = {
                "name": Path(str(row["reference"])).name,
                "index": index,
                "source_path": str(row.get("source_path", "")),
                "candidate": name,
                "actual_payload_bpp": payload_bpp,
                "psnr": psnr(reference.float(), reconstruction.float()),
                "ms_ssim": float(ms_ssim(reconstruction.float(), reference.float(), data_range=1.0, size_average=True).item()),
                "lpips_alex": p.lpips_alex,
                "dists": p.dists,
                "l1": float(torch.mean(torch.abs(reconstruction.float() - reference.float())).item()),
                "condition_l1_to_reference": float(F.l1_loss(cond.float(), cond_reference.float()).item()),
                "fid_patch_count": patch_count,
                "oracle_policy": oracle_policy,
            }
            per_image.append(row_out)
            for key in metrics[name]:
                metrics[name][key].append(float(row_out[key]))
            wandb_run.log({f"{name}/{key}": value for key, value in row_out.items() if isinstance(value, float)})
            if args.save_reconstructions:
                save_image(reconstruction.detach().cpu(), out_dir / "reconstructions" / name / f"image{item_index:05d}.png")

    summaries: list[dict[str, object]] = []
    for name, _ in specs:
        summary = {
            "candidate": name,
            "count": len(metrics[name]["psnr"]),
            "actual_payload_bpp_mean": mean(metrics[name]["actual_payload_bpp"]),
            "psnr_mean": mean(metrics[name]["psnr"]),
            "ms_ssim_mean": mean(metrics[name]["ms_ssim"]),
            "lpips_alex_mean": mean(metrics[name]["lpips_alex"]),
            "dists_mean": mean(metrics[name]["dists"]),
            "l1_mean": mean(metrics[name]["l1"]),
            "condition_l1_to_reference_mean": mean(metrics[name]["condition_l1_to_reference"]),
            "fid": float(fids[name].compute().item()),
            "payload_policy": (
                "actual_payload_bpp inherits Stage 3 only for decoder-available candidates; "
                "reference-derived candidates are oracle diagnostics and are not valid paper points"
            ),
        }
        summaries.append(summary)
    summary_payload = {
        "run_name": args.run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "summaries": summaries,
    }
    summary_path = out_dir / "summary.json"
    per_image_path = out_dir / "per_image_metrics.jsonl"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    per_image_path.write_text("\n".join(json.dumps(row, allow_nan=False, sort_keys=True) for row in per_image) + "\n")

    doc_lines = [
        f"# {args.run_name}",
        "",
        f"Date: {summary_payload['date']}",
        "",
        "## Summary",
        "",
        "| candidate | bpp | cond L1 to ref | PSNR | MS-SSIM | LPIPS-Alex | DISTS | patch-FID |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        doc_lines.append(
            "| {candidate} | {actual_payload_bpp_mean:.6f} | {condition_l1_to_reference_mean:.4f} | "
            "{psnr_mean:.4f} | {ms_ssim_mean:.4f} | {lpips_alex_mean:.4f} | {dists_mean:.4f} | {fid:.4f} |".format(
                **summary
            )
        )
    doc_lines.extend(
        [
            "",
            "Reference-derived condition candidates are upper-bound diagnostics only.",
            "They are not valid CoSER-DiC paper points because the reference condition is not decoder-available.",
            "",
            "## Artifacts",
            "",
            f"- summary: `{summary_path}`",
            f"- per-image: `{per_image_path}`",
            f"- wandb: `{Path(wandb_run.dir).parent if wandb_run is not None else ''}`",
        ]
    )
    Path("docs/experiments").mkdir(parents=True, exist_ok=True)
    (Path("docs/experiments") / f"{args.run_name}.md").write_text("\n".join(doc_lines) + "\n")
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    wandb_run.finish()


if __name__ == "__main__":
    main()
