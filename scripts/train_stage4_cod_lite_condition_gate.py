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
    CoSERToCoDLiteConditionGate,
    CoSERToCoDLiteConditionGateConfig,
    apply_lora_adapters_from_config,
    load_named_parameter_state,
)
from coserdic.utils import seed_everything
from coserdic.utils.wandb_utils import init_wandb
from eval_stage4_cod_lite_adapter import (
    Stage4ManifestDataset,
    apply_condition_residual,
    basis_component_codebook_codec,
    basis_component_ranges,
    basis_control_payload_bits,
    basis_vector_codebook_codec,
    build_adapter_from_payload,
    build_control_grid_code,
    channel_group_sizes,
    detail_context_channels,
    grouped_condition_affine_basis_control,
    load_control_basis,
    load_control_huffman_prior,
    load_sparse_topk_control_huffman_prior,
    load_vector_codebook_huffman_prior,
    psnr,
    write_run_doc,
)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(torch.tensor(values).std(unbiased=False).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="checkpoints/stage4_cod_lite_condition_gate")
    parser.add_argument("--results-dir", default="results/stage4_cod_lite_condition_gate")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--gate-min", type=float, default=0.0)
    parser.add_argument("--gate-max", type=float, default=1.0)
    parser.add_argument("--init-gate", type=float, default=0.3)
    parser.add_argument("--global-gate", action="store_true")
    parser.add_argument("--channel-gate", action="store_true")
    parser.add_argument("--condition-l1-weight", type=float, default=0.1)
    parser.add_argument("--condition-regression-guard-weight", type=float, default=0.0)
    parser.add_argument("--image-l1-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=0.05)
    parser.add_argument("--dists-weight", type=float, default=0.0)
    parser.add_argument("--ms-ssim-weight", type=float, default=0.2)
    parser.add_argument("--stage3-l1-guard-weight", type=float, default=2.0)
    parser.add_argument("--stage3-mse-guard-weight", type=float, default=4.0)
    parser.add_argument("--stage3-guard-margin", type=float, default=0.0)
    parser.add_argument("--gate-mean-target", type=float, default=0.0)
    parser.add_argument("--gate-mean-weight", type=float, default=0.0)
    parser.add_argument("--gate-deviation-weight", type=float, default=0.0)
    parser.add_argument("--gate-tv-weight", type=float, default=0.0)
    parser.add_argument(
        "--counted-control-mode",
        choices=("none", "condition_residual_affine_basis"),
        default="none",
        help="Optionally train the gate after a simulated counted condition-control stream.",
    )
    parser.add_argument("--control-grid-size", type=int, default=4)
    parser.add_argument("--control-groups", type=int, default=8)
    parser.add_argument("--control-basis", default="")
    parser.add_argument("--control-basis-components", type=int, default=8)
    parser.add_argument("--control-basis-candidate-components", type=int, default=0)
    parser.add_argument("--control-basis-prefix-components", type=int, default=0)
    parser.add_argument("--control-basis-selection", choices=("prefix", "topk", "vector", "prefix_topk"), default="prefix")
    parser.add_argument(
        "--control-basis-range-mode",
        choices=("global", "component_p95", "component_p99", "component_codebook"),
        default="global",
    )
    parser.add_argument("--control-basis-range-floor", type=float, default=1.0e-6)
    parser.add_argument("--control-codec", choices=("fixed_bits", "huffman"), default="fixed_bits")
    parser.add_argument("--control-huffman-key", default="")
    parser.add_argument("--control-quantizer", choices=("uniform", "mu_law"), default="uniform")
    parser.add_argument("--control-mu", type=float, default=16.0)
    parser.add_argument("--control-bits", type=int, default=4)
    parser.add_argument("--control-range", type=float, default=0.25)
    parser.add_argument("--control-affine-groups", type=int, default=0)
    parser.add_argument("--control-affine-grid-size", type=int, default=0)
    parser.add_argument("--control-affine-gain-range", type=float, default=1.0)
    parser.add_argument("--control-affine-bias-range", type=float, default=0.25)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--wandb-mode", default="offline")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; stop and restart the container before Stage 4 condition-gate training.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.dists_weight < 0.0:
        raise ValueError("--dists-weight must be >= 0")
    if args.counted_control_mode != "none" and not args.control_basis:
        raise ValueError("--control-basis is required when --counted-control-mode is enabled")
    if args.control_grid_size <= 0:
        raise ValueError("--control-grid-size must be positive")
    if args.control_groups <= 0:
        raise ValueError("--control-groups must be positive")
    if args.control_basis_components <= 0:
        raise ValueError("--control-basis-components must be positive")
    if args.control_basis_candidate_components < 0:
        raise ValueError("--control-basis-candidate-components must be non-negative")
    if args.control_basis_prefix_components < 0:
        raise ValueError("--control-basis-prefix-components must be non-negative")
    if args.control_basis_range_floor <= 0:
        raise ValueError("--control-basis-range-floor must be positive")
    if args.control_codec == "huffman" and args.control_basis_selection == "prefix_topk":
        raise ValueError("selection=prefix_topk currently supports fixed_bits only")
    if args.control_basis_range_mode in {"component_p95", "component_p99"} and args.control_codec != "fixed_bits":
        raise ValueError("--control-basis-range-mode component_p* currently supports only fixed_bits basis controls")
    if args.control_bits < 1:
        raise ValueError("--control-bits must be >= 1")
    if args.control_range <= 0:
        raise ValueError("--control-range must be positive")
    if args.control_mu <= 0:
        raise ValueError("--control-mu must be positive")
    if args.gate_deviation_weight < 0.0:
        raise ValueError("--gate-deviation-weight must be >= 0")
    if args.gate_tv_weight < 0.0:
        raise ValueError("--gate-tv-weight must be >= 0")
    if args.control_affine_gain_range <= 0:
        raise ValueError("--control-affine-gain-range must be positive")
    if args.control_affine_bias_range <= 0:
        raise ValueError("--control-affine-bias-range must be positive")
    effective_affine_groups = int(args.control_affine_groups or args.control_groups)
    effective_affine_grid_size = int(args.control_affine_grid_size or args.control_grid_size)
    if effective_affine_groups <= 0:
        raise ValueError("--control-affine-groups must be positive when provided")
    if effective_affine_grid_size <= 0:
        raise ValueError("--control-affine-grid-size must be positive when provided")

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda")
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_stage4_cod_lite_condition_gate"
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
    if args.counted_control_mode != "none":
        channel_group_sizes(backbone.condition_channels, args.control_groups)
        channel_group_sizes(backbone.condition_channels, effective_affine_groups)

    control_basis_payload: dict[str, object] | None = None
    control_huffman = None
    control_sparse_huffman = None
    effective_control_bits = int(args.control_bits)
    effective_control_range = float(args.control_range)
    effective_control_quantizer = str(args.control_quantizer)
    effective_control_mu = float(args.control_mu)
    effective_control_huffman_key = str(args.control_huffman_key)
    effective_control_basis_range_mode = str(args.control_basis_range_mode)
    effective_candidate_components = int(args.control_basis_components)
    if args.counted_control_mode == "condition_residual_affine_basis":
        control_basis_payload = load_control_basis(args.control_basis, device=device)
        if int(control_basis_payload["groups"]) != args.control_groups:
            raise ValueError("--control-groups must match control basis groups")
        if int(control_basis_payload["grid_size"]) != args.control_grid_size:
            raise ValueError("--control-grid-size must match control basis grid_size")
        basis_tensor = control_basis_payload["basis"]
        if not isinstance(basis_tensor, torch.Tensor):
            raise TypeError("loaded control basis is not a tensor")
        if args.control_basis_selection in {"topk", "prefix_topk"} and args.control_basis_candidate_components > 0:
            effective_candidate_components = int(args.control_basis_candidate_components)
        if args.control_basis_components > effective_candidate_components:
            raise ValueError("--control-basis-components must be <= --control-basis-candidate-components")
        if args.control_basis_selection == "vector":
            effective_candidate_components = int(args.control_basis_components)
        if args.control_basis_selection == "prefix_topk":
            if args.control_basis_prefix_components <= 0:
                raise ValueError("--control-basis-prefix-components must be positive for selection=prefix_topk")
            if args.control_basis_prefix_components >= effective_candidate_components:
                raise ValueError("--control-basis-prefix-components must be < candidate components")
            if args.control_basis_components > effective_candidate_components - args.control_basis_prefix_components:
                raise ValueError(
                    "--control-basis-components must be <= candidate-prefix components for selection=prefix_topk"
                )
        if effective_candidate_components > int(basis_tensor.shape[0]):
            raise ValueError("--control-basis-candidate-components exceeds basis tensor count")
        if args.control_codec == "huffman" and args.control_basis_selection == "prefix":
            (
                control_huffman,
                effective_control_bits,
                effective_control_range,
                effective_control_quantizer,
                effective_control_mu,
                effective_control_huffman_key,
                prior_basis_range_mode,
            ) = load_control_huffman_prior(
                control_basis_payload,
                key=args.control_huffman_key,
                components=args.control_basis_components,
            )
            if prior_basis_range_mode != "global":
                if effective_control_basis_range_mode not in {"global", prior_basis_range_mode}:
                    raise ValueError("Huffman prior basis_range_mode conflicts with --control-basis-range-mode")
                effective_control_basis_range_mode = prior_basis_range_mode
        elif args.control_codec == "huffman" and args.control_basis_selection == "topk":
            (
                index_huffman,
                value_huffman,
                effective_control_bits,
                effective_control_range,
                effective_control_quantizer,
                effective_control_mu,
                effective_control_huffman_key,
                prior_basis_range_mode,
            ) = load_sparse_topk_control_huffman_prior(
                control_basis_payload,
                key=args.control_huffman_key,
                candidate_components=effective_candidate_components,
                components=args.control_basis_components,
            )
            control_sparse_huffman = (index_huffman, value_huffman)
            if prior_basis_range_mode != "global":
                if effective_control_basis_range_mode not in {"global", prior_basis_range_mode}:
                    raise ValueError("sparse Huffman prior basis_range_mode conflicts with --control-basis-range-mode")
                effective_control_basis_range_mode = prior_basis_range_mode
        elif args.control_codec == "huffman" and args.control_basis_selection == "vector":
            (
                control_huffman,
                effective_control_bits,
                effective_control_huffman_key,
            ) = load_vector_codebook_huffman_prior(
                control_basis_payload,
                key=args.control_huffman_key,
                components=args.control_basis_components,
                bits=args.control_bits,
            )

    basis_range_components = (
        effective_candidate_components
        if args.control_basis_selection in {"topk", "prefix_topk"}
        else args.control_basis_components
    )
    if control_basis_payload is not None and args.control_basis_selection == "vector":
        basis_control_codec, vector_key = basis_vector_codebook_codec(
            control_basis_payload,
            components=args.control_basis_components,
            bits=effective_control_bits,
            key=effective_control_huffman_key if args.control_codec == "huffman" else "",
        )
        if not effective_control_huffman_key:
            effective_control_huffman_key = vector_key
    elif control_basis_payload is not None and effective_control_basis_range_mode == "component_codebook":
        basis_control_codec = basis_component_codebook_codec(
            control_basis_payload,
            bits=effective_control_bits,
            components=basis_range_components,
        )
    else:
        basis_value_ranges = (
            basis_component_ranges(
                control_basis_payload,
                mode=effective_control_basis_range_mode,
                components=basis_range_components,
                floor=args.control_basis_range_floor,
            )
            if control_basis_payload is not None
            else None
        )
        basis_control_codec = build_control_grid_code(
            quantizer=effective_control_quantizer,
            bits=effective_control_bits,
            value_range=effective_control_range,
            mu=effective_control_mu,
            value_ranges=basis_value_ranges,
        )
    affine_gain_codec = build_control_grid_code(
        quantizer=effective_control_quantizer,
        bits=effective_control_bits,
        value_range=args.control_affine_gain_range,
        mu=effective_control_mu,
    )
    affine_bias_codec = build_control_grid_code(
        quantizer=effective_control_quantizer,
        bits=effective_control_bits,
        value_range=args.control_affine_bias_range,
        mu=effective_control_mu,
    )

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
    condition_gate = CoSERToCoDLiteConditionGate(
        CoSERToCoDLiteConditionGateConfig(
            semantic_channels=semantic_channels,
            detail_context_channels=detail_channels,
            condition_channels=backbone.condition_channels,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            gate_min=args.gate_min,
            gate_max=args.gate_max,
            init_gate=args.init_gate,
            spatial_gate=not args.global_gate,
            channel_gate=args.channel_gate,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(condition_gate.parameters(), lr=args.lr)

    lpips_model = None
    if args.lpips_weight > 0:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").eval().to(device)
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)
    dists_model = None
    if args.dists_weight > 0:
        from DISTS_pytorch import DISTS

        dists_model = DISTS().eval().to(device)
        for parameter in dists_model.parameters():
            parameter.requires_grad_(False)

    wandb_run = init_wandb(
        {
            "stage": "stage4_cod_lite_condition_gate",
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
            "gate_min": args.gate_min,
            "gate_max": args.gate_max,
            "init_gate": args.init_gate,
            "global_gate": args.global_gate,
            "channel_gate": args.channel_gate,
            "condition_l1_weight": args.condition_l1_weight,
            "condition_regression_guard_weight": args.condition_regression_guard_weight,
            "image_l1_weight": args.image_l1_weight,
            "lpips_weight": args.lpips_weight,
            "dists_weight": args.dists_weight,
            "ms_ssim_weight": args.ms_ssim_weight,
            "stage3_l1_guard_weight": args.stage3_l1_guard_weight,
            "stage3_mse_guard_weight": args.stage3_mse_guard_weight,
            "stage3_guard_margin": args.stage3_guard_margin,
            "gate_mean_target": args.gate_mean_target,
            "gate_mean_weight": args.gate_mean_weight,
            "gate_deviation_weight": args.gate_deviation_weight,
            "gate_tv_weight": args.gate_tv_weight,
            "counted_control_mode": args.counted_control_mode,
            "control_basis": args.control_basis,
            "control_basis_components": args.control_basis_components,
            "control_basis_candidate_components": args.control_basis_candidate_components,
            "effective_control_basis_candidate_components": effective_candidate_components,
            "control_basis_selection": args.control_basis_selection,
            "control_basis_range_mode": effective_control_basis_range_mode,
            "control_codec": args.control_codec,
            "control_huffman_key": effective_control_huffman_key,
            "control_quantizer": effective_control_quantizer,
            "control_mu": effective_control_mu,
            "control_bits": effective_control_bits,
            "control_range": effective_control_range,
            "control_groups": args.control_groups,
            "control_grid_size": args.control_grid_size,
            "control_affine_groups": effective_affine_groups,
            "control_affine_grid_size": effective_affine_grid_size,
            "control_affine_gain_range": args.control_affine_gain_range,
            "control_affine_bias_range": args.control_affine_bias_range,
            "control_scale": args.control_scale,
            "wandb": {"mode": args.wandb_mode, "project": "coserdic"},
        },
        run_name=run_name,
    )

    metrics: dict[str, list[float]] = {
        "loss": [],
        "image_l1": [],
        "lpips": [],
        "dists_loss": [],
        "ms_ssim_loss": [],
        "condition_l1": [],
        "ungated_condition_l1": [],
        "pre_control_condition_l1": [],
        "condition_regression_guard": [],
        "stage3_l1_guard": [],
        "stage3_mse_guard": [],
        "gate_mean_loss": [],
        "gate_deviation_loss": [],
        "gate_tv_loss": [],
        "control_payload_bytes": [],
        "control_grid_abs_mean": [],
        "stage4_psnr": [],
        "stage3_psnr": [],
        "stage4_ms_ssim": [],
        "stage3_ms_ssim": [],
        "gate_mean": [],
        "gate_std": [],
        "gate_min": [],
        "gate_max": [],
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
                target_cond = backbone.native_condition(reference)
                cond_delta = adapter(
                    stage3,
                    semantic,
                    residual,
                    semantic_latent,
                    condition_size=condition_size,
                    base_condition=base_cond,
                    detail_context=detail_context_tensor,
                )
                ungated_pred_cond = apply_condition_residual(
                    base_cond,
                    cond_delta,
                    residual_scale=residual_scale,
                    residual_tanh=residual_tanh,
                )
                pre_control_condition_l1 = torch.mean(torch.abs((ungated_pred_cond - target_cond).float()))
                if args.counted_control_mode == "condition_residual_affine_basis":
                    if control_basis_payload is None:
                        raise RuntimeError("control_basis_payload was not loaded")
                    control_correction, control_payload_bytes, control_grid_abs = grouped_condition_affine_basis_control(
                        base_cond.float(),
                        ungated_pred_cond.float(),
                        target_cond.float(),
                        affine_groups=effective_affine_groups,
                        affine_grid_size=effective_affine_grid_size,
                        gain_codec=affine_gain_codec,
                        bias_codec=affine_bias_codec,
                        basis_payload=control_basis_payload,
                        components=args.control_basis_components,
                        candidate_components=effective_candidate_components,
                        selection=args.control_basis_selection,
                        basis_codec=basis_control_codec,
                        prefix_components=args.control_basis_prefix_components,
                        huffman=control_huffman,
                        sparse_huffman=control_sparse_huffman,
                        scale=args.control_scale,
                    )
                    pre_gate_pred_cond = ungated_pred_cond + control_correction.to(dtype=ungated_pred_cond.dtype)
                    control_payload_bytes_mean = float(sum(control_payload_bytes) / max(len(control_payload_bytes), 1))
                    control_grid_abs_mean = float(control_grid_abs.float().mean().item())
                else:
                    pre_gate_pred_cond = ungated_pred_cond
                    control_payload_bytes_mean = 0.0
                    control_grid_abs_mean = 0.0
                condition_residual = pre_gate_pred_cond - base_cond

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gate = condition_gate(
                    stage3,
                    semantic,
                    residual,
                    semantic_latent,
                    condition_size=condition_size,
                    base_condition=base_cond,
                    condition_residual=condition_residual,
                    detail_context=detail_context_tensor,
                )
                gated_pred_cond = base_cond + gate * condition_residual
                stage4 = backbone(stage3, gated_pred_cond)

            image_l1 = F.l1_loss(stage4.float(), reference.float())
            stage3_l1 = torch.mean(torch.abs(stage3.float() - reference.float()), dim=(1, 2, 3))
            stage4_l1 = torch.mean(torch.abs(stage4.float() - reference.float()), dim=(1, 2, 3))
            stage3_mse = torch.mean((stage3.float() - reference.float()).pow(2), dim=(1, 2, 3))
            stage4_mse = torch.mean((stage4.float() - reference.float()).pow(2), dim=(1, 2, 3))
            l1_guard = torch.relu(stage4_l1 - stage3_l1 - args.stage3_guard_margin).mean()
            mse_guard = torch.relu(stage4_mse - stage3_mse - args.stage3_guard_margin).mean()
            ms_ssim_stage4 = ms_ssim(stage4.float(), reference.float(), data_range=1.0, size_average=True)
            msssim_loss = 1.0 - ms_ssim_stage4
            condition_l1 = F.l1_loss(gated_pred_cond.float(), target_cond.float())
            ungated_condition_l1 = F.l1_loss(pre_gate_pred_cond.float(), target_cond.float())
            condition_guard = torch.relu(condition_l1 - ungated_condition_l1)
            if lpips_model is None:
                lpips_loss = image_l1.new_tensor(0.0)
            else:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    lpips_loss = lpips_model(
                        stage4.float().clamp(0, 1) * 2.0 - 1.0,
                        reference.float().clamp(0, 1) * 2.0 - 1.0,
                    ).mean()
            if dists_model is None:
                dists_loss = image_l1.new_tensor(0.0)
            else:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    dists_loss = dists_model(
                        stage4.float().clamp(0, 1),
                        reference.float().clamp(0, 1),
                    ).mean()
            if args.gate_mean_weight > 0:
                gate_mean_loss = (gate.float().mean() - args.gate_mean_target) ** 2
            else:
                gate_mean_loss = image_l1.new_tensor(0.0)
            if args.gate_deviation_weight > 0:
                gate_deviation_loss = (gate.float() - args.gate_mean_target).pow(2).mean()
            else:
                gate_deviation_loss = image_l1.new_tensor(0.0)
            if args.gate_tv_weight > 0:
                gate_float = gate.float()
                if gate_float.shape[-2] > 1:
                    gate_tv_h = torch.mean(torch.abs(gate_float[..., 1:, :] - gate_float[..., :-1, :]))
                else:
                    gate_tv_h = image_l1.new_tensor(0.0)
                if gate_float.shape[-1] > 1:
                    gate_tv_w = torch.mean(torch.abs(gate_float[..., :, 1:] - gate_float[..., :, :-1]))
                else:
                    gate_tv_w = image_l1.new_tensor(0.0)
                gate_tv_loss = gate_tv_h + gate_tv_w
            else:
                gate_tv_loss = image_l1.new_tensor(0.0)
            loss = (
                args.image_l1_weight * image_l1
                + args.lpips_weight * lpips_loss
                + args.dists_weight * dists_loss
                + args.ms_ssim_weight * msssim_loss
                + args.condition_l1_weight * condition_l1
                + args.condition_regression_guard_weight * condition_guard
                + args.stage3_l1_guard_weight * l1_guard
                + args.stage3_mse_guard_weight * mse_guard
                + args.gate_mean_weight * gate_mean_loss
                + args.gate_deviation_weight * gate_deviation_loss
                + args.gate_tv_weight * gate_tv_loss
            )
            (loss / args.grad_accum_steps).backward()

            with torch.no_grad():
                stage4_psnr = psnr(reference.float(), stage4.float()).mean()
                stage3_psnr = psnr(reference.float(), stage3.float()).mean()
                stage3_msssim = ms_ssim(stage3.float(), reference.float(), data_range=1.0, size_average=True)
                gate_flat = gate.detach().float().flatten()
                row = {
                    "loss": float(loss.item()),
                    "image_l1": float(image_l1.item()),
                    "lpips": float(lpips_loss.item()),
                    "dists_loss": float(dists_loss.item()),
                    "ms_ssim_loss": float(msssim_loss.item()),
                    "condition_l1": float(condition_l1.item()),
                    "ungated_condition_l1": float(ungated_condition_l1.item()),
                    "pre_control_condition_l1": float(pre_control_condition_l1.item()),
                    "condition_regression_guard": float(condition_guard.item()),
                    "stage3_l1_guard": float(l1_guard.item()),
                    "stage3_mse_guard": float(mse_guard.item()),
                    "gate_mean_loss": float(gate_mean_loss.item()),
                    "gate_deviation_loss": float(gate_deviation_loss.item()),
                    "gate_tv_loss": float(gate_tv_loss.item()),
                    "control_payload_bytes": control_payload_bytes_mean,
                    "control_grid_abs_mean": control_grid_abs_mean,
                    "stage4_psnr": float(stage4_psnr.item()),
                    "stage3_psnr": float(stage3_psnr.item()),
                    "stage4_ms_ssim": float(ms_ssim_stage4.item()),
                    "stage3_ms_ssim": float(stage3_msssim.item()),
                    "gate_mean": float(gate_flat.mean().item()),
                    "gate_std": float(gate_flat.std(unbiased=False).item()),
                    "gate_min": float(gate_flat.min().item()),
                    "gate_max": float(gate_flat.max().item()),
                }
            accum_rows.append(row)

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(condition_gate.parameters(), args.grad_clip_norm)
        optimizer.step()

        row = {key: float(sum(item[key] for item in accum_rows) / max(len(accum_rows), 1)) for key in accum_rows[0]}
        for key, value in row.items():
            metrics[key].append(value)
        wandb_run.log(row, step=step)
        step += 1
        pbar.update(1)
    pbar.close()

    summary = {f"{key}_mean": _mean(values) for key, values in metrics.items()}
    summary.update(
        {
            "gate_mean_over_steps": _mean(metrics["gate_mean"]),
            "gate_mean_std_over_steps": _std(metrics["gate_mean"]),
            "payload_policy": (
                "deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; "
                "no transmitted gate side information and no RGB output blending"
            ),
            "gate_mean_target": args.gate_mean_target,
            "gate_mean_weight": args.gate_mean_weight,
            "gate_deviation_weight": args.gate_deviation_weight,
            "gate_tv_weight": args.gate_tv_weight,
            "counted_control_mode": args.counted_control_mode,
            "control_basis": args.control_basis,
            "control_basis_components": args.control_basis_components,
            "control_basis_candidate_components": args.control_basis_candidate_components,
            "effective_control_basis_candidate_components": effective_candidate_components,
            "control_basis_selection": args.control_basis_selection,
            "control_basis_range_mode": effective_control_basis_range_mode,
            "control_codec": args.control_codec,
            "control_huffman_key": effective_control_huffman_key,
            "control_quantizer": effective_control_quantizer,
            "control_mu": effective_control_mu,
            "control_bits": effective_control_bits,
            "control_range": effective_control_range,
            "control_groups": args.control_groups,
            "control_grid_size": args.control_grid_size,
            "control_affine_groups": effective_affine_groups,
            "control_affine_grid_size": effective_affine_grid_size,
            "control_affine_gain_range": args.control_affine_gain_range,
            "control_affine_bias_range": args.control_affine_bias_range,
            "control_scale": args.control_scale,
        }
    )
    checkpoint_path = output_dir / f"{run_name}.pt"
    torch.save(
        {
            "condition_gate_model": condition_gate.state_dict(),
            "condition_gate_config": condition_gate.cfg.__dict__,
            "adapter_checkpoint": args.adapter_checkpoint,
            "adapter_run_name": adapter_payload.get("run_name", ""),
            "backbone_config": backbone.cfg.__dict__,
            "summary": summary,
            "run_name": run_name,
            "manifest": args.manifest,
            "counted_control_mode": args.counted_control_mode,
            "control_basis": args.control_basis,
            "control_basis_components": args.control_basis_components,
            "control_basis_candidate_components": args.control_basis_candidate_components,
            "effective_control_basis_candidate_components": effective_candidate_components,
            "control_basis_selection": args.control_basis_selection,
            "control_basis_range_mode": effective_control_basis_range_mode,
            "control_codec": args.control_codec,
            "control_huffman_key": effective_control_huffman_key,
            "control_quantizer": effective_control_quantizer,
            "control_mu": effective_control_mu,
            "control_bits": effective_control_bits,
            "control_range": effective_control_range,
            "control_groups": args.control_groups,
            "control_grid_size": args.control_grid_size,
            "control_affine_groups": effective_affine_groups,
            "control_affine_grid_size": effective_affine_grid_size,
            "control_affine_gain_range": args.control_affine_gain_range,
            "control_affine_bias_range": args.control_affine_bias_range,
            "control_scale": args.control_scale,
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
