from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


PYTHON = ".venv/bin/python"


def gpu_preflight_command(python: str = PYTHON) -> str:
    return shlex.join([python, "scripts/check_gpu_ready.py", "--min-devices", "1"])


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


def plan_settings(
    payload: dict[str, object],
    *,
    presets: list[str],
    max_lora_params: int,
    include_y_embedder: bool,
) -> list[dict[str, object]]:
    settings = payload.get("settings")
    if not isinstance(settings, list):
        raise ValueError("LoRA target plan must contain settings")
    rows: list[dict[str, object]] = []
    wanted = set(presets)
    for item in settings:
        if not isinstance(item, dict):
            continue
        preset = str(item.get("preset", ""))
        if wanted and preset not in wanted:
            continue
        if not include_y_embedder and preset.startswith("y_"):
            continue
        if int(item.get("module_count", 0)) <= 0:
            continue
        if max_lora_params > 0 and int(item.get("lora_param_count", 0)) > max_lora_params:
            continue
        rows.append(item)
    return rows


def lr_tag(value: float) -> str:
    text = f"{value:.0e}" if value < 0.001 else f"{value:g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def run_name_for(*, prefix: str, preset: str, rank: int, lora_lr: float, max_steps: int) -> str:
    return f"{prefix}_{preset}_r{rank}_lr{lr_tag(lora_lr)}_ft{max_steps}"


def build_train_command(args: argparse.Namespace, *, preset: str, run_name: str) -> list[str]:
    command = [
        PYTHON,
        "scripts/train_stage4_cod_lite_adapter.py",
        "--manifest",
        args.train_manifest,
        "--init-checkpoint",
        args.init_checkpoint,
        "--run-name",
        run_name,
        "--output-dir",
        args.checkpoint_dir,
        "--results-dir",
        args.train_results_dir,
        "--cod-lite-checkpoint",
        args.cod_lite_checkpoint,
        "--cod-lite-config",
        args.cod_lite_config,
        "--crop-size",
        str(args.crop_size),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--num-workers",
        str(args.num_workers),
        "--max-steps",
        str(args.max_steps),
        "--lr",
        str(args.lr),
        "--adapter-kind",
        args.adapter_kind,
        "--semantic-channels",
        str(args.semantic_channels),
        "--detail-context",
        args.detail_context,
        "--hidden-channels",
        str(args.hidden_channels),
        "--num-image-blocks",
        str(args.num_image_blocks),
        "--num-condition-blocks",
        str(args.num_condition_blocks),
        "--num-detail-blocks",
        str(args.num_detail_blocks),
        "--num-fusion-blocks",
        str(args.num_fusion_blocks),
        "--condition-l1-weight",
        str(args.condition_l1_weight),
        "--condition-cosine-weight",
        str(args.condition_cosine_weight),
        "--condition-channel-stats-weight",
        str(args.condition_channel_stats_weight),
        "--image-l1-weight",
        str(args.image_l1_weight),
        "--lpips-weight",
        str(args.lpips_weight),
        "--dists-weight",
        str(args.dists_weight),
        "--ms-ssim-weight",
        str(args.ms_ssim_weight),
        "--stage3-l1-guard-weight",
        str(args.stage3_l1_guard_weight),
        "--stage3-mse-guard-weight",
        str(args.stage3_mse_guard_weight),
        "--condition-residual-scale",
        str(args.condition_residual_scale),
        "--condition-residual-rms-guard-weight",
        str(args.condition_residual_rms_guard_weight),
        "--condition-residual-rms-guard-ratio",
        str(args.condition_residual_rms_guard_ratio),
        "--condition-residual-rms-guard-granularity",
        args.condition_residual_rms_guard_granularity,
        "--backbone-lora-plan",
        args.lora_plan,
        "--backbone-lora-preset",
        preset,
        "--backbone-lora-rank",
        str(args.lora_rank),
        "--backbone-lora-alpha",
        str(args.lora_alpha),
        "--backbone-lora-lr",
        str(args.lora_lr),
        "--backbone-lora-weight-decay",
        str(args.lora_weight_decay),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--wandb-mode",
        args.wandb_mode,
    ]
    if args.init_nonstrict:
        command.append("--init-nonstrict")
    if args.detail_control_branch:
        command.append("--detail-control-branch")
    if args.detail_control_blocks:
        command.extend(["--detail-control-blocks", str(args.detail_control_blocks)])
    if args.detail_control_condition_fusion:
        command.append("--detail-control-condition-fusion")
    if args.detail_highfreq_context_branch:
        command.append("--detail-highfreq-context-branch")
    if args.detail_film_modulation:
        command.append("--detail-film-modulation")
    if args.condition_residual_tanh:
        command.append("--condition-residual-tanh")
    command.extend(args.train_extra_arg)
    return command


def build_eval_command(args: argparse.Namespace, *, checkpoint: Path, run_name: str) -> list[str]:
    command = [
        PYTHON,
        "scripts/eval_stage4_cod_lite_adapter.py",
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        args.eval_manifest,
        "--per-image-metrics",
        args.eval_per_image_metrics,
        "--run-name",
        f"{run_name}_limit{args.eval_limit}_eval",
        "--output-dir",
        args.eval_output_dir,
        "--crop-size",
        str(args.crop_size),
        "--limit",
        str(args.eval_limit),
        "--batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.eval_num_workers),
        "--wandb-mode",
        args.wandb_mode,
    ]
    command.extend(args.eval_extra_arg)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Stage 5 LoRA backbone train/eval commands from a LoRA target plan."
    )
    parser.add_argument("--lora-plan", required=True)
    parser.add_argument("--preset", action="append", default=[])
    parser.add_argument("--include-y-embedder", action="store_true")
    parser.add_argument("--max-lora-params", type=int, default=0)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--eval-per-image-metrics", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints/stage4_cod_lite_adapter")
    parser.add_argument("--train-results-dir", default="results/stage4_cod_lite_adapter")
    parser.add_argument("--eval-output-dir", default="results/stage4_cod_lite_adapter_eval")
    parser.add_argument("--cod-lite-checkpoint", default="external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.pt")
    parser.add_argument("--cod-lite-config", default="external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.yaml")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--adapter-kind", default="pyramid")
    parser.add_argument("--semantic-channels", type=int, default=256)
    parser.add_argument("--detail-context", default="residual_grid_codes")
    parser.add_argument("--hidden-channels", type=int, default=192)
    parser.add_argument("--num-image-blocks", type=int, default=4)
    parser.add_argument("--num-condition-blocks", type=int, default=4)
    parser.add_argument("--num-detail-blocks", type=int, default=2)
    parser.add_argument("--num-fusion-blocks", type=int, default=4)
    parser.add_argument("--detail-control-branch", action="store_true", default=True)
    parser.add_argument("--no-detail-control-branch", action="store_false", dest="detail_control_branch")
    parser.add_argument("--detail-control-blocks", type=int, default=0)
    parser.add_argument("--detail-control-condition-fusion", action="store_true")
    parser.add_argument("--detail-highfreq-context-branch", action="store_true")
    parser.add_argument("--detail-film-modulation", action="store_true", default=True)
    parser.add_argument("--no-detail-film-modulation", action="store_false", dest="detail_film_modulation")
    parser.add_argument("--condition-l1-weight", type=float, default=0.35)
    parser.add_argument("--condition-cosine-weight", type=float, default=0.05)
    parser.add_argument("--condition-channel-stats-weight", type=float, default=0.05)
    parser.add_argument("--image-l1-weight", type=float, default=0.20)
    parser.add_argument("--lpips-weight", type=float, default=0.03)
    parser.add_argument("--dists-weight", type=float, default=0.04)
    parser.add_argument("--ms-ssim-weight", type=float, default=0.10)
    parser.add_argument("--stage3-l1-guard-weight", type=float, default=2.0)
    parser.add_argument("--stage3-mse-guard-weight", type=float, default=4.0)
    parser.add_argument("--condition-residual-scale", type=float, default=0.85)
    parser.add_argument("--condition-residual-tanh", action="store_true", default=True)
    parser.add_argument("--no-condition-residual-tanh", action="store_false", dest="condition_residual_tanh")
    parser.add_argument("--condition-residual-rms-guard-weight", type=float, default=0.0)
    parser.add_argument("--condition-residual-rms-guard-ratio", type=float, default=0.5)
    parser.add_argument(
        "--condition-residual-rms-guard-granularity",
        choices=("global", "spatial", "channel"),
        default="global",
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    parser.add_argument("--lora-lr", type=float, default=2.0e-5)
    parser.add_argument("--lora-weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--init-nonstrict", action="store_true", default=True)
    parser.add_argument("--strict-init", action="store_false", dest="init_nonstrict")
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--train-extra-arg", action="append", default=[])
    parser.add_argument("--eval-extra-arg", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-sh", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = read_json(Path(args.lora_plan))
    settings = plan_settings(
        plan,
        presets=args.preset,
        max_lora_params=args.max_lora_params,
        include_y_embedder=args.include_y_embedder,
    )
    if not settings:
        raise ValueError("no LoRA settings selected")
    rows: list[dict[str, object]] = []
    for setting in settings:
        preset = str(setting["preset"])
        run_name = run_name_for(
            prefix=args.run_prefix,
            preset=preset,
            rank=args.lora_rank,
            lora_lr=args.lora_lr,
            max_steps=args.max_steps,
        )
        checkpoint = Path(args.checkpoint_dir) / f"{run_name}.pt"
        train_command = build_train_command(args, preset=preset, run_name=run_name)
        eval_command = build_eval_command(args, checkpoint=checkpoint, run_name=run_name)
        summary_path = Path(args.eval_output_dir) / f"{run_name}_limit{args.eval_limit}_eval" / "summary.json"
        rows.append(
            {
                "preset": preset,
                "run_name": run_name,
                "checkpoint": str(checkpoint),
                "summary": str(summary_path),
                "module_count": setting.get("module_count", 0),
                "lora_param_count": setting.get("lora_param_count", 0),
                "base_param_count": setting.get("base_param_count", 0),
                "rationale": setting.get("rationale", ""),
                "source_setting_metadata": {
                    "lora_preset": preset,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_lr": args.lora_lr,
                    "lora_param_count": setting.get("lora_param_count", 0),
                    "lora_module_count": setting.get("module_count", 0),
                    "lora_rationale": setting.get("rationale", ""),
                },
                "train_command": train_command,
                "train_command_text": shlex.join(train_command),
                "eval_command": eval_command,
                "eval_command_text": shlex.join(eval_command),
                "command": eval_command,
                "command_text": shlex.join(eval_command),
            }
        )
    payload = {
        "lora_plan": args.lora_plan,
        "run_prefix": args.run_prefix,
        "init_checkpoint": args.init_checkpoint,
        "train_manifest": args.train_manifest,
        "eval_manifest": args.eval_manifest,
        "eval_per_image_metrics": args.eval_per_image_metrics,
        "policy": "Run train_command first, then eval_command. Promote only guarded limit64 wins to full552.",
        "settings": rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_sh:
        sh_path = Path(args.output_sh)
        sh_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "cd /workspace/CoSER-DiC",
            "",
            gpu_preflight_command(),
            "",
        ]
        for row in rows:
            lines.append(str(row["train_command_text"]))
            lines.append(str(row["eval_command_text"]))
            lines.append("")
        sh_path.write_text("\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
