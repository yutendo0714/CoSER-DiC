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
    TopKEscapeHuffmanCode,
    decode_semantic_tokens,
    encode_semantic_tokens,
)
from coserdic.metrics import PerceptualMetricBundle
from coserdic.models import (
    CausalTokenPrior,
    CausalTokenPriorConfig,
    SemanticVQAutoEncoder,
    SemanticVQConfig,
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
def encode_learned_payload(
    code: TopKEscapeHuffmanCode,
    token_prior: CausalTokenPrior,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[bytes, dict[str, float]]:
    # Use the exact same prefix schedule that the decoder can rebuild. Full
    # teacher-forced Transformer logits can differ by tiny amounts from prefix
    # logits near a top-k boundary, which is enough to break token roundtrip.
    topk_indices = decoder_prefix_topk_indices(token_prior, indices, topk=code.topk, device=device)
    payload = code.encode(indices.detach().cpu(), topk_indices)
    escape_count = code.escape_count(indices.detach().cpu(), topk_indices)
    return payload, {
        "escape_count": float(escape_count),
        "topk_hit_rate": float(1.0 - escape_count / max(indices.numel(), 1)),
        "unpadded_bits": float(code.encoded_bits(indices.detach().cpu(), topk_indices)),
    }


@torch.no_grad()
def decode_learned_payload(
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--token-prior-checkpoint", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--eval-protocol", choices=sorted(EVAL_PROTOCOLS), default="")
    parser.add_argument("--eval-dataset", action="append", default=None)
    parser.add_argument("--dpl-root", default="/dpl")
    parser.add_argument("--allow-protocol-count-mismatch", action="store_true")
    parser.add_argument("--allow-nondeterministic-eval", action="store_true")
    parser.add_argument("--output-dir", default="results/bitstreams/stage2_learned_topk_escape")
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stream-header-codec", choices=["json", "compact"], default="json")
    parser.add_argument("--stream-checksum-codec", choices=["sha256", "crc32"], default="sha256")
    parser.add_argument("--save-bitstreams", action="store_true")
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
        args.max_images = 0 if args.eval_protocol else 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 2 learned-prior evaluation.")

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

    prior_payload = json.loads(Path(args.prior).read_text())
    code = TopKEscapeHuffmanCode.from_dict(prior_payload)
    if code.codebook_size != cfg.codebook_size:
        raise ValueError("prior codebook_size does not match checkpoint")
    token_prior_value = args.token_prior_checkpoint or str(prior_payload.get("token_prior_checkpoint", ""))
    if not token_prior_value:
        raise ValueError("pass --token-prior-checkpoint or store token_prior_checkpoint in prior")
    token_prior_path = Path(token_prior_value)
    token_prior = load_token_prior(token_prior_path, device)
    if token_prior.cfg.vocab_size != cfg.codebook_size:
        raise ValueError("token prior vocab_size does not match Stage 1 checkpoint")

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

    run_name = args.run_name or f"{checkpoint_path.stem}_stage2_learned_topk_escape_eval"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = out_dir / "streams"
    if args.save_bitstreams:
        stream_dir.mkdir(parents=True, exist_ok=True)

    learned_name = "learned_topk_escape_huffman"
    buckets = {
        learned_name: {"payload_bpp": [], "actual_payload_bpp": [], "paper_bpp": [], "actual_bpp": [], "debug_full_stream_bpp": [], "stream_bytes": [], "payload_bytes": [], "psnr": [], "l1": [], "ms_ssim": [], "roundtrip": [], "topk_hit_rate": [], "unpadded_bits_per_token": []},
        "fixed_bits": {"payload_bpp": [], "actual_payload_bpp": [], "paper_bpp": [], "actual_bpp": [], "debug_full_stream_bpp": [], "stream_bytes": [], "payload_bytes": [], "psnr": [], "l1": [], "ms_ssim": [], "roundtrip": []},
    }
    if args.compute_perceptual:
        for metrics in buckets.values():
            metrics.update({"lpips_alex": [], "dists": []})
    coder = CoSERBitstream(header_codec=args.stream_header_codec, checksum_codec=args.stream_checksum_codec)
    perceptual = PerceptualMetricBundle().to(device).eval() if args.compute_perceptual else None
    sample_written = False
    roundtrip_failures: list[dict[str, object]] = []

    with torch.no_grad():
        for batch_index, x_cpu in enumerate(tqdm(loader, desc=run_name)):
            x = x_cpu.to(device, non_blocking=True)
            out = model(x, quantize_mix=1.0)
            indices_batch = out["indices"].detach().cpu()
            for image_index in range(x.shape[0]):
                global_index = batch_index * args.batch_size + image_index
                x_i = x[image_index : image_index + 1]
                indices = indices_batch[image_index]
                learned_payload, learned_stats = encode_learned_payload(code, token_prior, indices, device=device)
                payloads = {
                    learned_name: learned_payload,
                    "fixed_bits": encode_semantic_tokens(indices, codebook_size=cfg.codebook_size, codec="fixed_bits"),
                }
                for codec_name, token_payload in payloads.items():
                    header = CoSERHeader(
                        codec_version="s2lte0",
                        image_height=int(x_i.shape[-2]),
                        image_width=int(x_i.shape[-1]),
                        padded_height=int(x_i.shape[-2]),
                        padded_width=int(x_i.shape[-1]),
                        color_space="RGB",
                        rate_level_id=0,
                        perception_level_id=0,
                        semantic_shape=tuple(int(v) for v in indices.shape),
                        detail_shape=(0,),
                        entropy_model_version="s2sem-lteh0" if codec_name == learned_name else "s2sem-fxb0",
                        cdf_precision=0,
                    )
                    stream = coder.pack(header, semantic_tokens=token_payload)
                    unpacked = coder.unpack(stream)
                    if codec_name == learned_name:
                        decoded = decode_learned_payload(
                            code,
                            token_prior,
                            unpacked.semantic_tokens,
                            shape=tuple(unpacked.header.semantic_shape),
                            device=device,
                        )
                    else:
                        decoded = decode_semantic_tokens(
                            unpacked.semantic_tokens,
                            shape=tuple(unpacked.header.semantic_shape),
                            codebook_size=cfg.codebook_size,
                            codec="fixed_bits",
                        )
                    x_hat = decode_indices_to_image(model, cfg, decoded.unsqueeze(0)).clamp(0, 1)
                    bucket = buckets[codec_name]
                    h = int(x_i.shape[-2])
                    w = int(x_i.shape[-1])
                    payload_bpp = coder.actual_payload_bpp(token_payload, h, w)
                    debug_full_stream_bpp = coder.debug_full_stream_bpp(stream, h, w)
                    bucket["payload_bpp"].append(payload_bpp)
                    bucket["actual_payload_bpp"].append(payload_bpp)
                    bucket["paper_bpp"].append(payload_bpp)
                    bucket["actual_bpp"].append(debug_full_stream_bpp)
                    bucket["debug_full_stream_bpp"].append(debug_full_stream_bpp)
                    bucket["stream_bytes"].append(float(len(stream)))
                    bucket["payload_bytes"].append(float(len(token_payload)))
                    bucket["psnr"].append(float(psnr(x_i, x_hat).item()))
                    bucket["l1"].append(float(torch.mean(torch.abs(x_hat - x_i)).item()))
                    bucket["ms_ssim"].append(float(ms_ssim(x_hat, x_i, data_range=1.0, size_average=True).item()))
                    if perceptual is not None:
                        perceptual_result = perceptual(x_i, x_hat)
                        bucket["lpips_alex"].append(perceptual_result.lpips_alex)
                        bucket["dists"].append(perceptual_result.dists)
                    roundtrip_ok = bool(torch.equal(decoded, indices))
                    bucket["roundtrip"].append(float(roundtrip_ok))
                    if not roundtrip_ok:
                        roundtrip_failures.append(
                            {
                                "codec": codec_name,
                                "index": int(global_index),
                                "path": str(dataset.paths[global_index]) if global_index < len(dataset.paths) else "",
                            }
                        )
                    if codec_name == learned_name:
                        bucket["topk_hit_rate"].append(float(learned_stats["topk_hit_rate"]))
                        bucket["unpadded_bits_per_token"].append(float(learned_stats["unpadded_bits"] / max(indices.numel(), 1)))
                    if args.save_bitstreams:
                        (stream_dir / f"batch{batch_index:04d}_img{image_index:02d}_{codec_name}.coser").write_bytes(stream)
                    if not sample_written and codec_name == learned_name:
                        sample_path = out_dir / "stage2_learned_topk_escape_recon_grid.png"
                        save_image(torch.cat([x_i.detach().cpu(), x_hat.detach().cpu()], dim=0), sample_path, nrow=1)
                        sample_written = True

    summary: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "prior": str(args.prior),
        "token_prior_checkpoint": str(token_prior_path),
        "num_images": len(dataset),
        "crop_size": args.crop_size,
        "protocol_default_crop_size": protocol_default_crop_size,
        "crop_size_matches_protocol_default": (
            None if protocol_default_crop_size is None else int(args.crop_size) == int(protocol_default_crop_size)
        ),
        "codebook_size": cfg.codebook_size,
        "topk": code.topk,
        "learned_topk_schedule": "prefix_replay_decoder_safe",
        "stream_header_codec": args.stream_header_codec,
        "stream_checksum_codec": args.stream_checksum_codec,
        "main_bpp_metric": f"{learned_name}_actual_payload_bpp_mean",
        "paper_bpp_metric": f"{learned_name}_paper_bpp_mean",
        "debug_bpp_metric": f"{learned_name}_debug_full_stream_bpp_mean",
        "eval_protocol": args.eval_protocol or "manual_roots",
        "eval_datasets": args.eval_dataset or [],
        "eval_image_roots": roots,
        "eval_protocol_summary": eval_protocol_summary or {},
        "deterministic_eval": not args.allow_nondeterministic_eval,
        "compute_perceptual": bool(args.compute_perceptual),
    }
    for codec_name, metrics in buckets.items():
        for metric_name, values in metrics.items():
            summary[f"{codec_name}_{metric_name}_mean"] = mean(values)
        summary[f"{codec_name}_all_tokens_roundtrip"] = bool(all(v == 1.0 for v in metrics["roundtrip"]))
        summary[f"{codec_name}_roundtrip_failure_count"] = int(sum(1 for v in metrics["roundtrip"] if v != 1.0))
    summary[f"{learned_name}_payload_bpp_delta_vs_fixed_bits"] = (
        float(summary[f"{learned_name}_payload_bpp_mean"]) - float(summary["fixed_bits_payload_bpp_mean"])
    )
    summary[f"{learned_name}_payload_bpp_delta_vs_leftctx_static_huffman"] = (
        float(summary[f"{learned_name}_payload_bpp_mean"]) - 0.011728922526041666
    )
    summary["roundtrip_failures"] = roundtrip_failures[:20]

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    import wandb

    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config={**vars(args), "eval_protocol_summary": eval_protocol_summary or {}},
    )
    wandb_run.summary.update(summary)
    wandb_run.finish()

    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "summary": summary,
        "artifacts": {"summary": str(summary_path), "sample": str(out_dir / "stage2_learned_topk_escape_recon_grid.png")},
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))
    if roundtrip_failures and not args.allow_roundtrip_failures:
        raise RuntimeError(f"token roundtrip failed for {len(roundtrip_failures)} images; see {summary_path}")


if __name__ == "__main__":
    main()
