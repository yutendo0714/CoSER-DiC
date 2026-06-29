from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from compute_bd_rate import DEFAULT_RATE_KEYS, first_present_float


METRIC_KEY_BY_STAGE = {
    "stage3": {
        "psnr": ("stage3_psnr_mean",),
        "ms_ssim": ("stage3_ms_ssim_mean",),
        "lpips": ("stage3_lpips_alex_mean",),
        "dists": ("stage3_dists_mean",),
        "fid": ("stage3_fid_mean", "stage3_patch_fid_mean", "patch_fid_mean", "fid_mean", "patch_fid", "fid"),
        "kid": ("stage3_kid_mean", "kid_mean", "kid"),
    },
    "stage4": {
        "psnr": ("stage4_psnr_mean",),
        "ms_ssim": ("stage4_ms_ssim_mean",),
        "lpips": ("stage4_lpips_alex_mean",),
        "dists": ("stage4_dists_mean",),
        "fid": ("stage4_fid_mean", "stage4_patch_fid_mean", "patch_fid_mean", "fid_mean", "patch_fid", "fid"),
        "kid": ("stage4_kid_mean", "kid_mean", "kid"),
    },
    "direct": {
        "psnr": ("psnr", "psnr_mean"),
        "ms_ssim": ("ms_ssim", "ms_ssim_mean"),
        "lpips": ("lpips", "lpips_alex", "lpips_mean", "lpips_alex_mean"),
        "dists": ("dists", "dists_mean"),
        "fid": ("fid", "fid_mean", "patch_fid", "patch_fid_mean"),
        "kid": ("kid", "kid_mean"),
    },
}


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.stem, path


def command_arg(command: list[str], flag: str) -> str:
    prefix = f"{flag}="
    for index, item in enumerate(command):
        if item == flag:
            if index + 1 >= len(command):
                return ""
            return command[index + 1]
        if item.startswith(prefix):
            return item[len(prefix) :]
    return ""


def iter_promotion_curve_inputs(
    path: Path,
    *,
    default_output_dir: str,
) -> list[tuple[str, Path]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"promotion JSON must be an object: {path}")
    commands = payload.get("commands")
    if not isinstance(commands, list):
        raise ValueError(f"promotion JSON must contain a commands list: {path}")

    rows: list[tuple[str, Path]] = []
    for index, row in enumerate(commands):
        if not isinstance(row, dict):
            raise ValueError(f"promotion command row {index} must be an object: {path}")

        command = row.get("command")
        if isinstance(command, list) and all(isinstance(item, str) for item in command):
            run_name = command_arg(command, "--run-name") or str(row.get("promoted_run_name", ""))
            output_dir = command_arg(command, "--output-dir") or default_output_dir
        else:
            run_name = str(row.get("promoted_run_name", ""))
            output_dir = default_output_dir

        summary_value = row.get("promoted_summary_path", row.get("promoted_summary"))
        if isinstance(summary_value, str) and summary_value:
            summary_path = Path(summary_value)
        else:
            if not run_name:
                raise ValueError(f"promotion command row {index} has no --run-name or promoted_run_name: {path}")
            summary_path = Path(output_dir) / run_name / "summary.json"

        label = str(row.get("promoted_run_name") or row.get("source_label") or run_name or summary_path.parent.name)
        rows.append((label, summary_path))
    return rows


def load_summary(path: Path, *, dataset_key: str = "") -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    if not dataset_key:
        return payload
    by_dataset = payload.get("by_dataset")
    if not isinstance(by_dataset, dict) or dataset_key not in by_dataset:
        raise ValueError(f"dataset key {dataset_key!r} not found in {path}")
    dataset_payload = by_dataset[dataset_key]
    if not isinstance(dataset_payload, dict):
        raise ValueError(f"dataset summary must be an object for {dataset_key!r} in {path}")
    flattened: dict[str, object] = {}
    for key, value in dataset_payload.items():
        if isinstance(value, dict) and "mean" in value:
            flattened[f"{key}_mean"] = value["mean"]
        else:
            flattened[key] = value
    return flattened


def load_extra_metric_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"extra metric JSON must be an object: {path}")
    flattened: dict[str, object] = {}
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        metric_sources = metrics
    else:
        metric_sources = payload
    key_map = {
        "frechet_inception_distance": "fid",
        "kernel_inception_distance_mean": "kid",
        "kernel_inception_distance_std": "kid_std",
        "fid": "fid",
        "fid_mean": "fid",
        "patch_fid": "fid",
        "patch_fid_mean": "fid",
        "kid": "kid",
        "kid_mean": "kid",
        "kid_std": "kid_std",
    }
    for source_key, target_key in key_map.items():
        if source_key in metric_sources:
            flattened[target_key] = metric_sources[source_key]
    return flattened


def collect_point(
    label: str,
    path: Path,
    *,
    stage: str,
    metrics: list[str],
    rate_key: str | None,
    dataset_key: str,
    allow_missing: bool,
    extra_metrics: dict[str, object] | None = None,
    extra_metric_sources: list[str] | None = None,
) -> dict[str, object]:
    payload = load_summary(path, dataset_key=dataset_key)
    if extra_metrics:
        payload.update(extra_metrics)
    rate_keys = (rate_key,) if rate_key else DEFAULT_RATE_KEYS
    rate = first_present_float(payload, rate_keys, kind=str(path))
    if rate is None:
        raise ValueError(f"no rate key found in {path}; tried {rate_keys}")

    point: dict[str, object] = {
        "label": label,
        "source": str(path),
        "bpp": rate,
        "actual_payload_bpp": rate,
    }
    if dataset_key:
        point["dataset_key"] = dataset_key
    if extra_metric_sources:
        point["extra_metric_sources"] = extra_metric_sources
    key_map = METRIC_KEY_BY_STAGE[stage]
    missing: list[str] = []
    for metric in metrics:
        if metric not in key_map:
            raise ValueError(f"metric {metric!r} is not supported for stage={stage}")
        value = first_present_float(payload, key_map[metric], kind=f"{path}.{metric}")
        if value is None:
            missing.append(metric)
        else:
            point[metric] = value
    if missing and not allow_missing:
        raise ValueError(f"{path} is missing metrics for stage={stage}: {', '.join(missing)}")
    return point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect CoSER-DiC eval summary JSON files into a BD-rate curve JSON. "
            "Use stage4 for adapter outputs and stage3 for the auxiliary anchor."
        )
    )
    parser.add_argument("--input", action="append", default=[], help="summary.json or label=summary.json")
    parser.add_argument(
        "--promotion-json",
        action="append",
        default=[],
        help=(
            "promotion JSON emitted by promote_stage5_control_candidates.py; "
            "summary paths are inferred from --output-dir/--run-name in each command"
        ),
    )
    parser.add_argument(
        "--promotion-output-dir-default",
        default="results/stage4_cod_lite_adapter_eval",
        help="Fallback output dir when a promoted command has no --output-dir.",
    )
    parser.add_argument(
        "--extra-metric-json",
        action="append",
        default=[],
        help=(
            "Extra metric JSON to merge into a point, usually label=patch_fid_kid.json. "
            "The label must match the point label."
        ),
    )
    parser.add_argument("--name", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--stage", choices=("stage3", "stage4", "direct"), default="stage4")
    parser.add_argument("--dataset-key", default="", help="Read a split from summarize_per_image_metrics output.")
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--rate-key", default="")
    parser.add_argument(
        "--bpp-policy",
        default="actual_payload_bpp / paper_bpp from entropy-coded CoSER payload streams",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = args.metric or ["psnr", "ms_ssim", "lpips", "dists"]
    inputs = [parse_labeled_path(value) for value in args.input]
    for value in args.promotion_json:
        inputs.extend(
            iter_promotion_curve_inputs(
                Path(value),
                default_output_dir=args.promotion_output_dir_default,
            )
        )
    if not inputs:
        raise ValueError("at least one --input or --promotion-json is required")

    extra_by_label: dict[str, dict[str, object]] = {}
    extra_sources_by_label: dict[str, list[str]] = {}
    for value in args.extra_metric_json:
        label, path = parse_labeled_path(value)
        extra_by_label.setdefault(label, {}).update(load_extra_metric_json(path))
        extra_sources_by_label.setdefault(label, []).append(str(path))

    points = []
    for label, path in inputs:
        points.append(
            collect_point(
                label,
                path,
                stage=args.stage,
                metrics=metrics,
                rate_key=args.rate_key or None,
                dataset_key=args.dataset_key,
                allow_missing=args.allow_missing,
                extra_metrics=extra_by_label.get(label),
                extra_metric_sources=extra_sources_by_label.get(label),
            )
        )

    payload = {
        "name": args.name or f"coserdic_{args.stage}_curve",
        "dataset": args.dataset,
        "stage": args.stage,
        "dataset_key": args.dataset_key,
        "metrics": metrics,
        "inputs": [str(path) for _, path in inputs],
        "promotion_jsons": [str(path) for path in args.promotion_json],
        "extra_metric_jsons": [str(parse_labeled_path(value)[1]) for value in args.extra_metric_json],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bpp_policy": args.bpp_policy,
        "points": points,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
