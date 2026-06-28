from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from scripts.summarize_per_image_metrics import grouped_dataset_key, read_jsonl
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from summarize_per_image_metrics import grouped_dataset_key, read_jsonl


QUALITY_METRICS: dict[str, str] = {
    "stage3_lpips_alex": "lower",
    "stage3_dists": "lower",
    "stage3_psnr": "higher",
    "stage3_ms_ssim": "higher",
}

DRIVER_METRICS = (
    "actual_payload_bpp",
    "semantic_payload_bpp",
    "detail_payload_bpp",
    "semantic_topk_hit_rate",
    "residual_grid_abs_mean",
    "residual_grid_std",
    "residual_grid_clip_ratio",
    "detail_code_entropy_bits",
)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    x_norm = math.sqrt(sum(x * x for x in x_centered))
    y_norm = math.sqrt(sum(y * y for y in y_centered))
    if x_norm <= 0.0 or y_norm <= 0.0:
        return None
    return float(sum(x * y for x, y in zip(x_centered, y_centered)) / (x_norm * y_norm))


def metric_pairs(rows: list[dict[str, Any]], metric_a: str, metric_b: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = finite_float(row.get(metric_a))
        y = finite_float(row.get(metric_b))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def compact_row(row: dict[str, Any], metrics: tuple[str, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "index": row.get("index"),
        "path": row.get("path", ""),
        "dataset": grouped_dataset_key(str(row.get("path", "")), "cod_reproduction_512"),
    }
    for metric in metrics:
        if metric in row:
            value = finite_float(row[metric])
            output[metric] = value if value is not None else row[metric]
    for key in ("reference_image", "semantic_only_image", "stage3_image", "triptych_image"):
        if key in row:
            output[key] = row[key]
    return output


def worst_rows(rows: list[dict[str, Any]], metric: str, direction: str, top_k: int) -> list[dict[str, Any]]:
    metric_rows = [(finite_float(row.get(metric)), row) for row in rows]
    valid_rows = [(value, row) for value, row in metric_rows if value is not None]
    if direction not in {"high", "low"}:
        raise ValueError(f"unknown direction: {direction}")
    reverse = direction == "high"
    sorted_valid = sorted(valid_rows, key=lambda item: item[0], reverse=reverse)
    context_metrics = (
        metric,
        "actual_payload_bpp",
        "semantic_payload_bpp",
        "detail_payload_bpp",
        "semantic_topk_hit_rate",
        "residual_grid_abs_mean",
        "residual_grid_std",
        "detail_code_entropy_bits",
    )
    return [compact_row(row, context_metrics) for _, row in sorted_valid[:top_k]]


def semantic_delta_summary(rows: list[dict[str, Any]], metric: str, direction: str) -> dict[str, Any]:
    delta_key = f"{metric}_delta_vs_semantic_only"
    values = [finite_float(row.get(delta_key)) for row in rows]
    deltas = [value for value in values if value is not None]
    if direction == "lower":
        improvement_count = sum(1 for value in deltas if value < 0.0)
        regression_count = sum(1 for value in deltas if value > 0.0)
    elif direction == "higher":
        improvement_count = sum(1 for value in deltas if value > 0.0)
        regression_count = sum(1 for value in deltas if value < 0.0)
    else:
        raise ValueError(f"unknown direction: {direction}")
    return {
        "delta_metric": delta_key,
        "count": len(deltas),
        "mean_delta": mean(deltas) if deltas else None,
        "improvement_count": improvement_count,
        "regression_count": regression_count,
    }


def analyze_rows(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "count": len(rows),
        "quality": {},
        "correlations": {},
    }
    for quality_metric, direction in QUALITY_METRICS.items():
        if not any(quality_metric in row for row in rows):
            continue
        sort_direction = "low" if direction == "higher" else "high"
        payload["quality"][quality_metric] = {
            "direction": direction,
            "semantic_delta": semantic_delta_summary(rows, quality_metric, direction),
            "worst_rows": worst_rows(rows, quality_metric, sort_direction, top_k),
        }
        metric_correlations: dict[str, Any] = {}
        for driver in DRIVER_METRICS:
            xs, ys = metric_pairs(rows, driver, quality_metric)
            corr = pearson(xs, ys)
            if corr is not None:
                metric_correlations[driver] = {
                    "pearson": corr,
                    "count": len(xs),
                }
        payload["correlations"][quality_metric] = metric_correlations
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true", help="Write the JSON artifact without printing it.")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(grouped_dataset_key(str(row["path"]), "cod_reproduction_512"), []).append(row)
    payload = {
        "label": args.label or args.input.parent.name,
        "input": str(args.input),
        "top_k": int(args.top_k),
        "overall": analyze_rows(rows, top_k=args.top_k),
        "by_dataset": {key: analyze_rows(group_rows, top_k=args.top_k) for key, group_rows in sorted(groups.items())},
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if not args.quiet:
        print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
