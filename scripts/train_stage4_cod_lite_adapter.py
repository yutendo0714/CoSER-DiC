from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from coserdic.models import (
    CoDLiteOneStepBackbone,
    CoDLiteOneStepBackboneConfig,
    CoSERToCoDLiteConditionAdapter,
    CoSERToCoDLiteConditionAdapterConfig,
    CoSERToCoDLiteConditionPyramidAdapter,
    CoSERToCoDLiteConditionPyramidAdapterConfig,
)
from coserdic.utils import seed_everything
from coserdic.utils.wandb_utils import init_wandb


def detail_context_channels(mode: str) -> int:
    return {"none": 0, "residual_grid": 3, "residual_grid_codes": 6}[mode]


class ReconstructionManifestDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        limit: int = 0,
        crop_size: int = 0,
        semantic_channels: int = 3,
        detail_context: str = "none",
    ) -> None:
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if limit:
            rows = rows[:limit]
        if not rows:
            raise ValueError(f"empty manifest: {manifest}")
        self.rows = rows
        self.crop_size = crop_size
        self.semantic_channels = int(semantic_channels)
        self.detail_context = detail_context

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.rows[index]
        reference = _load_image(Path(row["reference"]), self.crop_size)
        semantic = _load_image(Path(row["semantic_only"]), self.crop_size)
        stage3 = _load_image(Path(row["stage3"]), self.crop_size)
        semantic_latent = semantic
        cache_path = str(row.get("decoder_feature_cache", ""))
        has_decoder_feature_cache = bool(cache_path)
        detail_context = torch.empty(0, 1, 1)
        if self.semantic_channels != 3 or self.detail_context != "none":
            if not cache_path:
                raise FileNotFoundError(
                    f"manifest row {index} has no decoder_feature_cache; "
                    f"required for semantic_channels={self.semantic_channels} "
                    f"and detail_context={self.detail_context}"
                )
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            cache = {}
        if self.semantic_channels != 3:
            semantic_latent = cache["semantic_latent"].float()
            if semantic_latent.ndim == 4 and semantic_latent.shape[0] == 1:
                semantic_latent = semantic_latent.squeeze(0)
        if self.detail_context != "none":
            detail_context = _load_detail_context(cache, self.detail_context)
        return {
            "reference": reference,
            "semantic": semantic,
            "semantic_latent": semantic_latent,
            "detail_context": detail_context,
            "has_decoder_feature_cache": has_decoder_feature_cache,
            "stage3": stage3,
            "index": int(row.get("index", index)),
            "source_path": str(row.get("source_path", "")),
        }


def _load_image(path: Path, crop_size: int) -> torch.Tensor:
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
    return tensor


def _load_detail_context(cache: dict[str, object], mode: str) -> torch.Tensor:
    residual_grid = cache["residual_grid_hat"].float()
    if residual_grid.ndim == 4 and residual_grid.shape[0] == 1:
        residual_grid = residual_grid.squeeze(0)
    if mode == "residual_grid":
        return residual_grid
    if mode == "residual_grid_codes":
        detail_codes = cache["detail_codes"].float()
        if detail_codes.ndim == 4 and detail_codes.shape[0] == 1:
            detail_codes = detail_codes.squeeze(0)
        detail_codes = detail_codes / 15.0 * 2.0 - 1.0
        return torch.cat([residual_grid, detail_codes], dim=0)
    raise ValueError(f"unknown detail_context: {mode}")


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((x - y).pow(2), dim=(1, 2, 3)).clamp_min(1.0e-12)
    return -10.0 * torch.log10(mse)


def condition_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(pred.float().flatten(1), target.float().flatten(1), dim=1, eps=1.0e-8)
    return 1.0 - cosine.mean()


def condition_channel_stats_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.float()
    target_f = target.float()
    pred_mean = pred_f.mean(dim=(2, 3))
    target_mean = target_f.mean(dim=(2, 3))
    pred_std = pred_f.std(dim=(2, 3), unbiased=False)
    target_std = target_f.std(dim=(2, 3), unbiased=False)
    return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


def condition_highfreq_ratio(tensor: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
    x = tensor.float()
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.rfft2(x, norm="ortho")
    power = spectrum.abs().square()
    height = int(x.shape[-2])
    width = int(x.shape[-1])
    freq_y = torch.fft.fftfreq(height, device=x.device).abs().view(height, 1)
    freq_x = torch.fft.rfftfreq(width, device=x.device).abs().view(1, width // 2 + 1)
    radius = torch.sqrt(freq_y.square() + freq_x.square())
    high_mask = radius >= threshold
    total = power.sum(dim=(-2, -1)).clamp_min(1.0e-12)
    high = power[..., high_mask].sum(dim=-1)
    return high / total


def condition_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, *, threshold: float) -> torch.Tensor:
    return F.l1_loss(
        condition_highfreq_ratio(pred, threshold=threshold),
        condition_highfreq_ratio(target, threshold=threshold),
    )


def apply_condition_residual(
    base_condition: torch.Tensor,
    cond_delta: torch.Tensor,
    *,
    residual_scale: float,
    residual_tanh: bool,
) -> torch.Tensor:
    if residual_tanh:
        cond_delta = torch.tanh(cond_delta)
    return base_condition + residual_scale * cond_delta


def build_adapter(
    *,
    adapter_kind: str,
    semantic_channels: int,
    detail_context_channels: int,
    condition_channels: int,
    hidden_channels: int,
) -> CoSERToCoDLiteConditionAdapter | CoSERToCoDLiteConditionPyramidAdapter:
    if adapter_kind == "light":
        return CoSERToCoDLiteConditionAdapter(
            CoSERToCoDLiteConditionAdapterConfig(
                semantic_channels=semantic_channels,
                detail_context_channels=detail_context_channels,
                condition_channels=condition_channels,
                hidden_channels=hidden_channels,
                zero_init_output=True,
            )
        )
    if adapter_kind == "pyramid":
        return CoSERToCoDLiteConditionPyramidAdapter(
            CoSERToCoDLiteConditionPyramidAdapterConfig(
                semantic_channels=semantic_channels,
                detail_context_channels=detail_context_channels,
                condition_channels=condition_channels,
                hidden_channels=hidden_channels,
                zero_init_output=True,
            )
        )
    raise ValueError(f"unknown adapter_kind: {adapter_kind}")


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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="checkpoints/stage4_cod_lite_adapter")
    parser.add_argument("--results-dir", default="results/stage4_cod_lite_adapter")
    parser.add_argument("--cod-lite-checkpoint", default="external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt")
    parser.add_argument("--cod-lite-config", default="external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml")
    parser.add_argument("--cod-lite-repo", default="external/repos/GenCodec/CoD_Lite")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--condition-l1-weight", type=float, default=1.0)
    parser.add_argument("--condition-cosine-weight", type=float, default=0.0)
    parser.add_argument("--condition-channel-stats-weight", type=float, default=0.0)
    parser.add_argument("--condition-highfreq-weight", type=float, default=0.0)
    parser.add_argument("--condition-highfreq-threshold", type=float, default=0.25)
    parser.add_argument("--image-l1-weight", type=float, default=0.25)
    parser.add_argument("--lpips-weight", type=float, default=0.0)
    parser.add_argument("--condition-residual-scale", type=float, default=1.0)
    parser.add_argument("--condition-residual-tanh", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--base-condition", choices=("native_stage3", "zero"), default="native_stage3")
    parser.add_argument("--adapter-kind", choices=("light", "pyramid"), default="light")
    parser.add_argument("--semantic-channels", type=int, default=3)
    parser.add_argument("--detail-context", choices=("none", "residual_grid", "residual_grid_codes"), default="none")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--save-sample-every", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 4 training.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda")
    detail_channels = detail_context_channels(args.detail_context)
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_stage4_cod_lite_adapter"
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = ReconstructionManifestDataset(
        Path(args.manifest),
        limit=args.limit,
        crop_size=args.crop_size,
        semantic_channels=args.semantic_channels,
        detail_context=args.detail_context,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

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
    adapter = build_adapter(
        adapter_kind=args.adapter_kind,
        semantic_channels=args.semantic_channels,
        detail_context_channels=detail_channels,
        condition_channels=backbone.condition_channels,
        hidden_channels=args.hidden_channels,
    ).to(device)
    if args.init_checkpoint:
        init_payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        init_kind = str(init_payload.get("adapter_kind", "light"))
        if init_kind != args.adapter_kind:
            raise ValueError(f"init checkpoint adapter_kind mismatch: {init_kind} != {args.adapter_kind}")
        adapter.load_state_dict(init_payload["model"])
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)

    wandb_run = init_wandb(
        {
            "stage": "stage4_cod_lite_adapter",
            "manifest": args.manifest,
            "crop_size": args.crop_size,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "base_condition": args.base_condition,
            "adapter_kind": args.adapter_kind,
            "detail_context": args.detail_context,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "condition_residual_scale": args.condition_residual_scale,
            "condition_residual_tanh": args.condition_residual_tanh,
            "condition_cosine_weight": args.condition_cosine_weight,
            "condition_channel_stats_weight": args.condition_channel_stats_weight,
            "condition_highfreq_weight": args.condition_highfreq_weight,
            "condition_highfreq_threshold": args.condition_highfreq_threshold,
            "grad_clip_norm": args.grad_clip_norm,
            "init_checkpoint": args.init_checkpoint,
            "lpips_weight": args.lpips_weight,
            "wandb": {"mode": args.wandb_mode, "project": "coserdic"},
        },
        run_name=run_name,
    )
    lpips_model = None
    if args.lpips_weight > 0:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").eval().to(device)
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)
    use_image_decode_loss = args.image_l1_weight > 0 or args.lpips_weight > 0

    metrics: dict[str, list[float]] = {
        "loss": [],
        "condition_l1": [],
        "condition_cosine_loss": [],
        "condition_channel_stats_loss": [],
        "condition_highfreq_loss": [],
        "image_l1": [],
        "lpips": [],
        "stage4_psnr": [],
        "stage3_psnr": [],
        "condition_residual_l1": [],
        "condition_delta_raw_l1": [],
    }
    step = 0
    loader_iter = iter(loader)
    pbar = tqdm(total=args.max_steps, desc=run_name)
    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_rows: list[dict[str, float]] = []
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            reference = batch["reference"].to(device, non_blocking=True)
            semantic = batch["semantic"].to(device, non_blocking=True)
            semantic_latent = batch["semantic_latent"].to(device, non_blocking=True)
            detail_context = (
                batch["detail_context"].to(device, non_blocking=True) if detail_channels > 0 else None
            )
            stage3 = batch["stage3"].to(device, non_blocking=True)
            if args.semantic_channels != int(semantic_latent.shape[1]):
                raise ValueError(
                    f"--semantic-channels={args.semantic_channels} but semantic context has "
                    f"{int(semantic_latent.shape[1])} channels. Use a decoder feature cache for 256-channel latents."
                )
            residual = stage3 - semantic
            condition_size = backbone.condition_size(int(stage3.shape[-2]), int(stage3.shape[-1]))

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                target_cond = backbone.native_condition(reference)
                if args.base_condition == "native_stage3":
                    base_cond = backbone.native_condition(stage3)
                else:
                    base_cond = torch.zeros_like(target_cond)

            cond_delta = adapter(
                stage3,
                semantic,
                residual,
                semantic_latent,
                condition_size=condition_size,
                base_condition=base_cond,
                detail_context=detail_context,
            )
            pred_cond = apply_condition_residual(
                base_cond,
                cond_delta,
                residual_scale=args.condition_residual_scale,
                residual_tanh=args.condition_residual_tanh,
            )
            condition_l1 = F.l1_loss(pred_cond.float(), target_cond.float())
            if args.condition_cosine_weight > 0:
                condition_cos = condition_cosine_loss(pred_cond, target_cond)
            else:
                condition_cos = condition_l1.new_tensor(0.0)
            if args.condition_channel_stats_weight > 0:
                condition_stats = condition_channel_stats_loss(pred_cond, target_cond)
            else:
                condition_stats = condition_l1.new_tensor(0.0)
            if args.condition_highfreq_weight > 0:
                condition_hf = condition_highfreq_loss(
                    pred_cond,
                    target_cond,
                    threshold=args.condition_highfreq_threshold,
                )
            else:
                condition_hf = condition_l1.new_tensor(0.0)
            if use_image_decode_loss:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    stage4 = backbone(stage3, pred_cond)
                image_l1 = F.l1_loss(stage4.float(), reference.float())
            else:
                stage4 = None
                image_l1 = condition_l1.new_tensor(0.0)
            if lpips_model is None or stage4 is None:
                lpips_loss = condition_l1.new_tensor(0.0)
            else:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    lpips_loss = lpips_model(
                        stage4.float().clamp(0, 1) * 2.0 - 1.0,
                        reference.float().clamp(0, 1) * 2.0 - 1.0,
                    ).mean()
            loss = (
                args.condition_l1_weight * condition_l1
                + args.condition_cosine_weight * condition_cos
                + args.condition_channel_stats_weight * condition_stats
                + args.condition_highfreq_weight * condition_hf
                + args.image_l1_weight * image_l1
                + args.lpips_weight * lpips_loss
            )
            (loss / args.grad_accum_steps).backward()

            with torch.no_grad():
                if stage4 is None:
                    stage4_psnr = reference.new_tensor(0.0)
                else:
                    stage4_psnr = psnr(reference.float(), stage4.float()).mean()
                stage3_psnr = psnr(reference.float(), stage3.float()).mean()
                condition_residual_l1 = torch.mean(torch.abs((pred_cond - base_cond).float()))
                condition_delta_raw_l1 = torch.mean(torch.abs(cond_delta.float()))
            row = {
                "loss": float(loss.item()),
                "condition_l1": float(condition_l1.item()),
                "condition_cosine_loss": float(condition_cos.item()),
                "condition_channel_stats_loss": float(condition_stats.item()),
                "condition_highfreq_loss": float(condition_hf.item()),
                "image_l1": float(image_l1.item()),
                "lpips": float(lpips_loss.item()),
                "stage4_psnr": float(stage4_psnr.item()),
                "stage3_psnr": float(stage3_psnr.item()),
                "condition_residual_l1": float(condition_residual_l1.item()),
                "condition_delta_raw_l1": float(condition_delta_raw_l1.item()),
            }
            accum_rows.append(row)
            if stage4 is not None and args.save_sample_every > 0 and (
                step == 0 or (step + 1) % args.save_sample_every == 0
            ):
                sample = torch.cat(
                    [
                        reference[:1].detach().cpu(),
                        semantic[:1].detach().cpu(),
                        stage3[:1].detach().cpu(),
                        stage4[:1].detach().cpu(),
                    ],
                    dim=0,
                )
                save_image(sample, results_dir / f"sample_step{step + 1:06d}.png", nrow=4)
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip_norm)
        optimizer.step()

        row = {
            key: float(sum(item[key] for item in accum_rows) / max(len(accum_rows), 1))
            for key in accum_rows[0]
        }
        for key, value in row.items():
            metrics[key].append(value)
        wandb_run.log(row, step=step)
        step += 1
        pbar.update(1)
    pbar.close()

    summary = {f"{key}_mean": float(sum(values) / max(len(values), 1)) for key, values in metrics.items()}
    checkpoint_path = output_dir / f"{run_name}.pt"
    torch.save(
        {
            "model": adapter.state_dict(),
            "adapter_config": adapter.cfg.__dict__,
            "adapter_kind": args.adapter_kind,
            "detail_context": args.detail_context,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "condition_residual_scale": args.condition_residual_scale,
            "condition_residual_tanh": args.condition_residual_tanh,
            "condition_cosine_weight": args.condition_cosine_weight,
            "condition_channel_stats_weight": args.condition_channel_stats_weight,
            "condition_highfreq_weight": args.condition_highfreq_weight,
            "condition_highfreq_threshold": args.condition_highfreq_threshold,
            "backbone_config": backbone.cfg.__dict__,
            "summary": summary,
            "run_name": run_name,
            "manifest": args.manifest,
        },
        checkpoint_path,
    )
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_run_doc(
        Path("docs/experiments") / f"{run_name}.md",
        {
            "run_name": run_name,
            "date": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "summary": summary,
            "artifacts": {
                "checkpoint": str(checkpoint_path),
                "summary": str(summary_path),
                "results_dir": str(results_dir),
                "wandb": str(Path(wandb_run.dir).parent if wandb_run is not None else ""),
            },
        },
    )
    wandb_run.finish()


if __name__ == "__main__":
    main()
