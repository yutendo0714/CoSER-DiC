from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from coserdic.models import (
    CoDLiteOneStepBackbone,
    CoDLiteOneStepBackboneConfig,
    CoSERToCoDLiteAlphaGate,
    CoSERToCoDLiteAlphaGateConfig,
    apply_lora_adapters_from_config,
    load_named_parameter_state,
)
from coserdic.utils import seed_everything
from coserdic.utils.wandb_utils import init_wandb
from eval_stage4_cod_lite_adapter import (
    Stage4ManifestDataset,
    apply_condition_residual,
    build_adapter_from_payload,
    detail_context_channels,
    psnr,
    write_run_doc,
)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values)
    return float(tensor.std(unbiased=False).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="checkpoints/stage4_cod_lite_gate")
    parser.add_argument("--results-dir", default="results/stage4_cod_lite_gate")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--init-alpha", type=float, default=0.3)
    parser.add_argument("--image-l1-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=0.05)
    parser.add_argument("--ms-ssim-weight", type=float, default=0.2)
    parser.add_argument("--stage3-l1-guard-weight", type=float, default=2.0)
    parser.add_argument("--stage3-mse-guard-weight", type=float, default=4.0)
    parser.add_argument("--stage3-guard-margin", type=float, default=0.0)
    parser.add_argument("--alpha-mean-target", type=float, default=0.0)
    parser.add_argument("--alpha-mean-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--wandb-mode", default="offline")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 4 gate training.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda")
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_stage4_cod_lite_gate"
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    adapter_payload = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    backbone_cfg = CoDLiteOneStepBackboneConfig(**adapter_payload["backbone_config"])
    backbone = CoDLiteOneStepBackbone.load(backbone_cfg, device=device)
    apply_lora_adapters_from_config(backbone.net, adapter_payload.get("backbone_lora_config", {}))
    if adapter_payload.get("backbone_trainable_state"):
        load_named_parameter_state(backbone.net, adapter_payload["backbone_trainable_state"], strict=True)
    backbone.eval()
    adapter = build_adapter_from_payload(adapter_payload).to(device)
    adapter.load_state_dict(adapter_payload["model"])
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)

    adapter_config = dict(adapter_payload.get("adapter_config", {}))
    semantic_channels = int(adapter_config.get("semantic_channels", 3))
    detail_context = str(adapter_payload.get("detail_context", "none"))
    detail_channels = detail_context_channels(detail_context)
    residual_scale = float(adapter_payload.get("condition_residual_scale", 1.0))
    residual_tanh = bool(adapter_payload.get("condition_residual_tanh", False))

    dataset = Stage4ManifestDataset(
        Path(args.manifest),
        None,
        limit=args.limit,
        crop_size=args.crop_size,
        semantic_channels=semantic_channels,
        detail_context=detail_context,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    gate = CoSERToCoDLiteAlphaGate(
        CoSERToCoDLiteAlphaGateConfig(
            semantic_channels=semantic_channels,
            detail_context_channels=detail_channels,
            condition_channels=backbone.condition_channels,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            init_alpha=args.init_alpha,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr)

    lpips_model = None
    if args.lpips_weight > 0:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").eval().to(device)
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)

    wandb_run = init_wandb(
        {
            "stage": "stage4_cod_lite_gate",
            "adapter_checkpoint": args.adapter_checkpoint,
            "manifest": args.manifest,
            "crop_size": args.crop_size,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "detail_context": detail_context,
            "semantic_channels": semantic_channels,
            "condition_residual_scale": residual_scale,
            "condition_residual_tanh": residual_tanh,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "init_alpha": args.init_alpha,
            "image_l1_weight": args.image_l1_weight,
            "lpips_weight": args.lpips_weight,
            "ms_ssim_weight": args.ms_ssim_weight,
            "stage3_l1_guard_weight": args.stage3_l1_guard_weight,
            "stage3_mse_guard_weight": args.stage3_mse_guard_weight,
            "stage3_guard_margin": args.stage3_guard_margin,
            "alpha_mean_target": args.alpha_mean_target,
            "alpha_mean_weight": args.alpha_mean_weight,
            "wandb": {"mode": args.wandb_mode, "project": "coserdic"},
        },
        run_name=run_name,
    )

    metrics: dict[str, list[float]] = {
        "loss": [],
        "image_l1": [],
        "lpips": [],
        "ms_ssim_loss": [],
        "stage3_l1_guard": [],
        "stage3_mse_guard": [],
        "alpha_mean_loss": [],
        "stage4_psnr": [],
        "stage3_psnr": [],
        "stage4_ms_ssim": [],
        "stage3_ms_ssim": [],
        "alpha_mean": [],
        "alpha_std": [],
        "alpha_min": [],
        "alpha_max": [],
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
            detail_context_tensor = (
                batch["detail_context"].to(device, non_blocking=True) if detail_channels > 0 else None
            )
            stage3 = batch["stage3"].to(device, non_blocking=True)
            residual = stage3 - semantic
            condition_size = backbone.condition_size(int(stage3.shape[-2]), int(stage3.shape[-1]))

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                base_cond = backbone.native_condition(stage3)
                cond_delta = adapter(
                    stage3,
                    semantic,
                    residual,
                    semantic_latent,
                    condition_size=condition_size,
                    base_condition=base_cond,
                    detail_context=detail_context_tensor,
                )
                pred_cond = apply_condition_residual(
                    base_cond,
                    cond_delta,
                    residual_scale=residual_scale,
                    residual_tanh=residual_tanh,
                )
                stage4_raw = backbone(stage3, pred_cond)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                alpha = gate(
                    stage3,
                    semantic,
                    residual,
                    semantic_latent,
                    condition_size=condition_size,
                    base_condition=base_cond,
                    condition_residual=pred_cond - base_cond,
                    detail_context=detail_context_tensor,
                )
                stage4 = ((1.0 - alpha) * stage3 + alpha * stage4_raw).clamp(0, 1)

            image_l1 = F.l1_loss(stage4.float(), reference.float())
            stage3_l1 = torch.mean(torch.abs(stage3.float() - reference.float()), dim=(1, 2, 3))
            stage4_l1 = torch.mean(torch.abs(stage4.float() - reference.float()), dim=(1, 2, 3))
            stage3_mse = torch.mean((stage3.float() - reference.float()).pow(2), dim=(1, 2, 3))
            stage4_mse = torch.mean((stage4.float() - reference.float()).pow(2), dim=(1, 2, 3))
            l1_guard = torch.relu(stage4_l1 - stage3_l1 - args.stage3_guard_margin).mean()
            mse_guard = torch.relu(stage4_mse - stage3_mse - args.stage3_guard_margin).mean()
            ms_ssim_stage4 = ms_ssim(stage4.float(), reference.float(), data_range=1.0, size_average=True)
            msssim_loss = 1.0 - ms_ssim_stage4
            if lpips_model is None:
                lpips_loss = image_l1.new_tensor(0.0)
            else:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    lpips_loss = lpips_model(
                        stage4.float().clamp(0, 1) * 2.0 - 1.0,
                        reference.float().clamp(0, 1) * 2.0 - 1.0,
                    ).mean()
            if args.alpha_mean_weight > 0:
                alpha_mean_loss = (alpha.float().mean() - args.alpha_mean_target) ** 2
            else:
                alpha_mean_loss = image_l1.new_tensor(0.0)
            loss = (
                args.image_l1_weight * image_l1
                + args.lpips_weight * lpips_loss
                + args.ms_ssim_weight * msssim_loss
                + args.stage3_l1_guard_weight * l1_guard
                + args.stage3_mse_guard_weight * mse_guard
                + args.alpha_mean_weight * alpha_mean_loss
            )
            (loss / args.grad_accum_steps).backward()

            with torch.no_grad():
                stage4_psnr = psnr(reference.float(), stage4.float()).mean()
                stage3_psnr = psnr(reference.float(), stage3.float()).mean()
                stage3_msssim = ms_ssim(stage3.float(), reference.float(), data_range=1.0, size_average=True)
                alpha_flat = alpha.detach().float().flatten()
                row = {
                    "loss": float(loss.item()),
                    "image_l1": float(image_l1.item()),
                    "lpips": float(lpips_loss.item()),
                    "ms_ssim_loss": float(msssim_loss.item()),
                    "stage3_l1_guard": float(l1_guard.item()),
                    "stage3_mse_guard": float(mse_guard.item()),
                    "alpha_mean_loss": float(alpha_mean_loss.item()),
                    "stage4_psnr": float(stage4_psnr.item()),
                    "stage3_psnr": float(stage3_psnr.item()),
                    "stage4_ms_ssim": float(ms_ssim_stage4.item()),
                    "stage3_ms_ssim": float(stage3_msssim.item()),
                    "alpha_mean": float(alpha_flat.mean().item()),
                    "alpha_std": float(alpha_flat.std(unbiased=False).item()),
                    "alpha_min": float(alpha_flat.min().item()),
                    "alpha_max": float(alpha_flat.max().item()),
                }
            accum_rows.append(row)

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(gate.parameters(), args.grad_clip_norm)
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

    summary = {
        f"{key}_mean": _mean(values)
        for key, values in metrics.items()
    }
    summary.update(
        {
            "alpha_mean_over_steps": _mean(metrics["alpha_mean"]),
            "alpha_mean_std_over_steps": _std(metrics["alpha_mean"]),
            "payload_policy": (
                "deterministic decoder-side gate from decoded CoSER tensors and fixed model weights; "
                "no transmitted side information"
            ),
        }
    )
    checkpoint_path = output_dir / f"{run_name}.pt"
    torch.save(
        {
            "gate_model": gate.state_dict(),
            "gate_config": gate.cfg.__dict__,
            "adapter_checkpoint": args.adapter_checkpoint,
            "adapter_run_name": adapter_payload.get("run_name", ""),
            "backbone_config": backbone.cfg.__dict__,
            "summary": summary,
            "run_name": run_name,
            "manifest": args.manifest,
            "payload_policy": summary["payload_policy"],
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
