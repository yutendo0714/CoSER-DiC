from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_RATIOS = (0.15, 0.25, 0.35, 0.5, 0.75, 1.0)
DEFAULT_GRANULARITIES = ("global", "channel", "spatial")


def float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def gpu_preflight_command(python: str) -> str:
    return shlex.join([python, "scripts/check_gpu_ready.py", "--min-devices", "1"])


def run_gpu_preflight(python: str) -> None:
    result = subprocess.run(
        [python, "scripts/check_gpu_ready.py", "--min-devices", "1"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)


def run_name_for(*, prefix: str, guard: str, granularity: str, ratio: float, min_gate: float) -> str:
    if guard == "none":
        return f"{prefix}_guard_none"
    return f"{prefix}_guard_{granularity}_r{float_tag(ratio)}_ming{float_tag(min_gate)}"


def build_eval_command(
    *,
    python: str,
    checkpoint: str,
    manifest: str,
    per_image_metrics: str,
    run_name: str,
    output_dir: str,
    crop_size: int,
    limit: int,
    batch_size: int,
    num_workers: int,
    guard: str,
    granularity: str,
    ratio: float,
    min_gate: float,
    extra_args: list[str],
) -> list[str]:
    command = [
        python,
        "scripts/eval_stage4_cod_lite_adapter.py",
        "--checkpoint",
        checkpoint,
        "--manifest",
        manifest,
        "--per-image-metrics",
        per_image_metrics,
        "--run-name",
        run_name,
        "--output-dir",
        output_dir,
        "--crop-size",
        str(crop_size),
        "--limit",
        str(limit),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--blend-alpha",
        "1.0",
        "--condition-residual-guard",
        guard,
    ]
    if guard == "rms_clip":
        command.extend(
            [
                "--condition-residual-guard-granularity",
                granularity,
                "--condition-residual-max-rms-ratio",
                str(ratio),
                "--condition-residual-min-gate",
                str(min_gate),
            ]
        )
    command.extend(extra_args)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Stage 5 decoder-side condition residual guard eval commands. "
            "This sweeps a deterministic condition-space RMS guard without adding transmitted bits."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--per-image-metrics", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--output-dir", default="results/stage4_cod_lite_adapter_eval")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ratio", action="append", type=float, default=[])
    parser.add_argument("--granularity", action="append", choices=DEFAULT_GRANULARITIES, default=[])
    parser.add_argument("--min-gate", type=float, default=0.0)
    parser.add_argument("--include-none", action="store_true", default=True)
    parser.add_argument("--no-include-none", action="store_false", dest="include_none")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-gpu-preflight", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-sh", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = args.ratio or list(DEFAULT_RATIOS)
    granularities = args.granularity or list(DEFAULT_GRANULARITIES)
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("--ratio values must be positive")
    if not 0.0 <= args.min_gate <= 1.0:
        raise ValueError("--min-gate must be in [0, 1]")

    rows: list[dict[str, object]] = []
    settings: list[tuple[str, str, float]] = []
    if args.include_none:
        settings.append(("none", "global", 0.0))
    for granularity in granularities:
        for ratio in ratios:
            settings.append(("rms_clip", str(granularity), float(ratio)))

    if not args.dry_run and not args.skip_gpu_preflight:
        run_gpu_preflight(args.python)

    for guard, granularity, ratio in settings:
        run_name = run_name_for(
            prefix=args.run_prefix,
            guard=guard,
            granularity=granularity,
            ratio=ratio,
            min_gate=args.min_gate,
        )
        command = build_eval_command(
            python=args.python,
            checkpoint=args.checkpoint,
            manifest=args.manifest,
            per_image_metrics=args.per_image_metrics,
            run_name=run_name,
            output_dir=args.output_dir,
            crop_size=args.crop_size,
            limit=args.limit,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            guard=guard,
            granularity=granularity,
            ratio=ratio,
            min_gate=args.min_gate,
            extra_args=args.extra_arg,
        )
        row = {
            "run_name": run_name,
            "checkpoint": args.checkpoint,
            "summary": str(Path(args.output_dir) / run_name / "summary.json"),
            "per_image": str(Path(args.output_dir) / run_name / "per_image_metrics.jsonl"),
            "condition_residual_guard": guard,
            "condition_residual_guard_granularity": granularity,
            "condition_residual_max_rms_ratio": ratio,
            "condition_residual_min_gate": args.min_gate,
            "actual_payload_note": "No extra payload; guard is deterministic from decoded condition tensors.",
            "command": command,
            "command_text": shlex.join(command),
        }
        rows.append(row)
        print(row["command_text"])
        if not args.dry_run:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)

    payload = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "per_image_metrics": args.per_image_metrics,
        "run_prefix": args.run_prefix,
        "crop_size": args.crop_size,
        "limit": args.limit,
        "policy": (
            "Decoder-side condition residual RMS guard. Use as a Stage 5 stability probe; "
            "promote only if full552 metrics improve without changing actual_payload_bpp."
        ),
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
            gpu_preflight_command(args.python),
            "",
        ]
        lines.extend(str(row["command_text"]) for row in rows)
        lines.append("")
        sh_path.write_text("\n".join(lines))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
