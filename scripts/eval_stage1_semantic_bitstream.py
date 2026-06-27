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
    SUPPORTED_SEMANTIC_TOKEN_CODECS,
    CoSERBitstream,
    CoSERHeader,
    decode_semantic_tokens,
    encode_semantic_tokens,
)
from coserdic.metrics import PerceptualMetricBundle
from coserdic.models import SemanticVQAutoEncoder, SemanticVQConfig
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
                path = Path(root)
                self.paths.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
        else:
            self.paths = [Path(path) for path in image_paths]
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no images found under {roots or image_paths}")
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image = Image.open(self.paths[index]).convert("RGB")
        tensor = TF.to_tensor(image)
        _, h, w = tensor.shape
        if h < self.crop_size or w < self.crop_size:
            scale = self.crop_size / min(h, w)
            new_h = max(self.crop_size, int(round(h * scale)))
            new_w = max(self.crop_size, int(round(w * scale)))
            tensor = TF.resize(tensor, [new_h, new_w], antialias=True)
            _, h, w = tensor.shape
        top = max(0, (h - self.crop_size) // 2)
        left = max(0, (w - self.crop_size) // 2)
        tensor = TF.crop(tensor, top, left, self.crop_size, self.crop_size)
        return {"image": tensor, "path": str(self.paths[index])}


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((x - y).pow(2), dim=(1, 2, 3)).clamp_min(1.0e-12)
    return -10.0 * torch.log10(mse)


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def load_config_from_checkpoint(checkpoint: dict, fallback_config: str | None) -> dict:
    if "config" in checkpoint:
        return checkpoint["config"]
    if fallback_config is None:
        raise KeyError("checkpoint has no raw config; pass --config")
    return yaml.safe_load(Path(fallback_config).read_text())


def decode_indices_to_image(
    model: SemanticVQAutoEncoder,
    cfg: SemanticVQConfig,
    indices: torch.Tensor,
) -> torch.Tensor:
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
    parser.add_argument("--config", default="")
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--eval-protocol", choices=sorted(EVAL_PROTOCOLS), default="")
    parser.add_argument("--eval-dataset", action="append", default=None)
    parser.add_argument("--dpl-root", default="/dpl")
    parser.add_argument("--allow-protocol-count-mismatch", action="store_true")
    parser.add_argument("--allow-nondeterministic-eval", action="store_true")
    parser.add_argument("--output-dir", default="results/bitstreams/stage1_semantic_vq")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--codec", action="append", default=None, choices=SUPPORTED_SEMANTIC_TOKEN_CODECS)
    parser.add_argument("--save-bitstreams", action="store_true")
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument("--write-per-image-metrics", action="store_true")
    parser.add_argument("--compute-perceptual", action="store_true")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()
    if args.max_images is None:
        args.max_images = 0 if args.eval_protocol else 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before bitstream evaluation.")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = load_config_from_checkpoint(checkpoint, args.config or None)
    seed_everything(
        int(raw.get("experiment", {}).get("seed", 42)),
        deterministic=not args.allow_nondeterministic_eval,
    )

    cfg_payload = checkpoint.get("semantic_vq_config")
    if cfg_payload is None:
        raise KeyError("checkpoint has no semantic_vq_config")
    cfg = SemanticVQConfig(**cfg_payload)
    model = SemanticVQAutoEncoder(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    codecs = args.codec or ["raw_uint16_be", "fixed_bits", "zlib_uint16_be", "zlib_fixed_bits"]
    run_name = args.run_name or f"{checkpoint_path.stem}_semantic_bitstream"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = out_dir / "streams"
    if args.save_bitstreams:
        stream_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = out_dir / "reconstructions"
    if args.save_reconstructions:
        recon_dir.mkdir(parents=True, exist_ok=True)

    import wandb

    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config={
            "checkpoint": str(checkpoint_path),
            "semantic_vq_config": cfg.__dict__,
            "roots": roots,
            "max_images": args.max_images,
            "eval_protocol": args.eval_protocol or "manual_roots",
            "eval_datasets": args.eval_dataset or [],
            "eval_protocol_summary": eval_protocol_summary or {},
            "crop_size": args.crop_size,
            "codecs": codecs,
            "save_bitstreams": args.save_bitstreams,
            "compute_perceptual": args.compute_perceptual,
        },
    )

    coder = CoSERBitstream()
    perceptual = PerceptualMetricBundle().to(device).eval() if args.compute_perceptual else None
    per_codec: dict[str, dict[str, list[float]]] = {
        codec: {
            "actual_bpp": [],
            "actual_payload_bpp": [],
            "paper_bpp": [],
            "debug_full_stream_bpp": [],
            "token_payload_bpp": [],
            "container_overhead_bpp": [],
            "stream_bytes": [],
            "token_payload_bytes": [],
            "container_overhead_bytes": [],
            "psnr": [],
            "l1": [],
            "ms_ssim": [],
            "token_roundtrip": [],
        }
        for codec in codecs
    }
    if args.compute_perceptual:
        for metrics in per_codec.values():
            metrics.update({"lpips_alex": [], "dists": []})
    sample_written = False
    semantic_token_counts: list[float] = []
    per_image_rows: list[dict[str, float | int | str | bool]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=run_name)):
            x = batch["image"].to(device, non_blocking=True)
            paths = list(batch["path"])
            out = model(x, quantize_mix=1.0)
            indices_batch = out["indices"].detach().cpu()

            for image_index in range(x.shape[0]):
                global_index = batch_index * args.batch_size + image_index
                x_i = x[image_index : image_index + 1]
                indices = indices_batch[image_index]
                semantic_token_counts.append(float(indices.numel()))
                for codec in codecs:
                    token_payload = encode_semantic_tokens(
                        indices,
                        codebook_size=cfg.codebook_size,
                        codec=codec,
                    )
                    header = CoSERHeader(
                        codec_version="stage1-semantic-vq-mvp",
                        image_height=int(x_i.shape[-2]),
                        image_width=int(x_i.shape[-1]),
                        padded_height=int(x_i.shape[-2]),
                        padded_width=int(x_i.shape[-1]),
                        color_space="RGB",
                        rate_level_id=0,
                        perception_level_id=0,
                        semantic_shape=tuple(int(v) for v in indices.shape),
                        detail_shape=(0,),
                        entropy_model_version=f"semantic-token-{codec}-v0",
                        cdf_precision=0,
                    )
                    stream = coder.pack(header, semantic_tokens=token_payload)
                    unpacked = coder.unpack(stream)
                    decoded_indices = decode_semantic_tokens(
                        unpacked.semantic_tokens,
                        shape=tuple(int(v) for v in unpacked.header.semantic_shape),
                        codebook_size=cfg.codebook_size,
                        codec=codec,
                    )
                    tokens_match = torch.equal(decoded_indices, indices)
                    x_hat = decode_indices_to_image(model, cfg, decoded_indices.unsqueeze(0)).clamp(0, 1)

                    bucket = per_codec[codec]
                    h = int(x_i.shape[-2])
                    w = int(x_i.shape[-1])
                    token_payload_bpp = coder.actual_payload_bpp(token_payload, h, w)
                    debug_full_stream_bpp = coder.debug_full_stream_bpp(stream, h, w)
                    bucket["actual_bpp"].append(debug_full_stream_bpp)
                    bucket["actual_payload_bpp"].append(token_payload_bpp)
                    bucket["paper_bpp"].append(token_payload_bpp)
                    bucket["debug_full_stream_bpp"].append(debug_full_stream_bpp)
                    bucket["token_payload_bpp"].append(token_payload_bpp)
                    bucket["container_overhead_bpp"].append(
                        coder.debug_full_stream_bpp(
                            stream[: len(stream) - len(token_payload)],
                            h,
                            w,
                        )
                    )
                    bucket["stream_bytes"].append(float(len(stream)))
                    bucket["token_payload_bytes"].append(float(len(token_payload)))
                    bucket["container_overhead_bytes"].append(float(len(stream) - len(token_payload)))
                    psnr_value = float(psnr(x_i, x_hat).item())
                    l1_value = float(torch.mean(torch.abs(x_hat - x_i)).item())
                    ms_ssim_value = float(ms_ssim(x_hat, x_i, data_range=1.0, size_average=True).item())
                    bucket["psnr"].append(psnr_value)
                    bucket["l1"].append(l1_value)
                    bucket["ms_ssim"].append(ms_ssim_value)
                    lpips_value = None
                    dists_value = None
                    if perceptual is not None:
                        perceptual_result = perceptual(x_i, x_hat)
                        lpips_value = perceptual_result.lpips_alex
                        dists_value = perceptual_result.dists
                        bucket["lpips_alex"].append(lpips_value)
                        bucket["dists"].append(dists_value)
                    bucket["token_roundtrip"].append(float(tokens_match))

                    if args.save_bitstreams:
                        stream_path = stream_dir / f"batch{batch_index:04d}_img{image_index:02d}_{codec}.coser"
                        stream_path.write_bytes(stream)
                    recon_path = None
                    triptych_path = None
                    if args.save_reconstructions and codec == codecs[0]:
                        stem = Path(paths[image_index]).stem
                        safe_stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
                        image_prefix = f"{global_index:04d}_{safe_stem}"
                        recon_path = recon_dir / f"{image_prefix}_recon.png"
                        triptych_path = recon_dir / f"{image_prefix}_triptych.png"
                        error = (x_i - x_hat).abs().mul(4.0).clamp(0, 1)
                        save_image(x_hat.detach().cpu(), recon_path)
                        save_image(
                            torch.cat([x_i.detach().cpu(), x_hat.detach().cpu(), error.detach().cpu()], dim=0),
                            triptych_path,
                            nrow=3,
                        )
                    if args.write_per_image_metrics:
                        row: dict[str, float | int | str | bool] = {
                            "image_index": global_index,
                            "path": paths[image_index],
                            "codec": codec,
                            "actual_payload_bpp": token_payload_bpp,
                            "paper_bpp": token_payload_bpp,
                            "debug_full_stream_bpp": debug_full_stream_bpp,
                            "token_payload_bytes": len(token_payload),
                            "stream_bytes": len(stream),
                            "psnr": psnr_value,
                            "l1": l1_value,
                            "ms_ssim": ms_ssim_value,
                            "token_roundtrip": tokens_match,
                        }
                        if lpips_value is not None:
                            row["lpips_alex"] = lpips_value
                        if dists_value is not None:
                            row["dists"] = dists_value
                        if recon_path is not None:
                            row["stage1_recon_image"] = str(recon_path)
                        if triptych_path is not None:
                            row["triptych_image"] = str(triptych_path)
                        per_image_rows.append(row)

                    if not sample_written and codec == codecs[0]:
                        sample_path = out_dir / "bitstream_recon_grid.png"
                        save_image(torch.cat([x_i.detach().cpu(), x_hat.detach().cpu()], dim=0), sample_path, nrow=1)
                        wandb.log({"bitstream_recon_grid": wandb.Image(str(sample_path))})
                        sample_written = True

    summary: dict[str, float | int | str] = {
        "checkpoint": str(checkpoint_path),
        "num_images": len(dataset),
        "crop_size": args.crop_size,
        "semantic_tokens_per_image": int(round(mean(semantic_token_counts))),
        "semantic_tokens_per_image_min": int(min(semantic_token_counts)) if semantic_token_counts else 0,
        "semantic_tokens_per_image_max": int(max(semantic_token_counts)) if semantic_token_counts else 0,
        "codebook_size": cfg.codebook_size,
        "main_bpp_metric": "actual_payload_bpp_mean",
        "paper_bpp_metric": "paper_bpp_mean",
        "debug_bpp_metric": "debug_full_stream_bpp_mean",
        "eval_protocol": args.eval_protocol or "manual_roots",
        "eval_datasets": args.eval_dataset or [],
        "eval_image_roots": roots,
        "eval_protocol_summary": eval_protocol_summary or {},
        "deterministic_eval": not args.allow_nondeterministic_eval,
        "compute_perceptual": bool(args.compute_perceptual),
    }
    for codec, metrics in per_codec.items():
        for metric_name, values in metrics.items():
            summary[f"{codec}_{metric_name}_mean"] = mean(values)
        summary[f"{codec}_all_tokens_roundtrip"] = bool(all(v == 1.0 for v in metrics["token_roundtrip"]))

    per_image_path = out_dir / "per_image_metrics.jsonl"
    if args.write_per_image_metrics:
        per_image_path.write_text(
            "\n".join(json.dumps(row, allow_nan=False) for row in per_image_rows) + "\n"
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "summary": summary,
        "artifacts": {
            "summary": str(summary_path),
            "output_dir": str(out_dir),
            "streams": str(stream_dir) if args.save_bitstreams else "",
            "reconstructions": str(recon_dir) if args.save_reconstructions else "",
            "per_image_metrics": str(per_image_path) if args.write_per_image_metrics else "",
        },
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    wandb_run.summary.update(summary)
    wandb_run.finish()
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
