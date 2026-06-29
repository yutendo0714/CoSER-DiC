from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_METRICS = (
    "semantic_payload_bpp",
    "detail_payload_bpp",
    "actual_payload_bpp",
    "paper_bpp",
    "debug_full_stream_bpp",
    "semantic_only_psnr",
    "stage3_psnr",
    "stage3_psnr_delta_vs_semantic_only",
    "semantic_only_ms_ssim",
    "stage3_ms_ssim",
    "stage3_ms_ssim_delta_vs_semantic_only",
    "semantic_only_lpips_alex",
    "stage3_lpips_alex",
    "stage3_lpips_alex_delta_vs_semantic_only",
    "semantic_only_dists",
    "stage3_dists",
    "stage3_dists_delta_vs_semantic_only",
    "stage4_psnr",
    "stage4_psnr_delta_vs_stage3",
    "stage4_ms_ssim",
    "stage4_ms_ssim_delta_vs_stage3",
    "stage4_lpips_alex",
    "stage4_lpips_alex_delta_vs_stage3",
    "stage4_dists",
    "stage4_dists_delta_vs_stage3",
    "stage4_l1",
    "stage4_l1_delta_vs_stage3",
    "condition_l1",
    "base_condition_l1",
    "condition_l1_delta_vs_base",
    "condition_gate_mean",
    "semantic_topk_hit_rate",
    "residual_grid_abs_mean",
    "residual_grid_std",
    "residual_grid_clip_ratio",
    "detail_code_entropy_bits",
    "semantic_token_roundtrip",
    "detail_code_roundtrip",
)

GROUPINGS = ("path", "cod_reproduction_512", "gencodec_reproduction")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "path" not in row:
            if "source_path" not in row:
                raise ValueError(f"{path}:{line_number} has no path or source_path")
            row["path"] = row["source_path"]
        rows.append(row)
    return rows


def dataset_key(path_value: str) -> str:
    path = Path(path_value)
    parts = path.parts
    joined = path.as_posix()
    if "/dpl/kodak/" in joined:
        return "kodak"
    if "/dpl/clic/professional/valid/" in joined:
        return "clic_professional_valid"
    if "/dpl/clic/mobile/valid/" in joined:
        return "clic_mobile_valid"
    if "/dpl/clic/professional/test/" in joined:
        return "clic2020_test_professional"
    if "/dpl/clic/mobile/test/" in joined:
        return "clic2020_test_mobile"
    if "/dpl/div2k/" in joined:
        try:
            index = int(path.stem[:4])
        except ValueError:
            return "div2k_unknown"
        if 801 <= index <= 900:
            return "div2k_val"
        if 1 <= index <= 800:
            return "div2k_train"
        return "div2k_unknown"
    if len(parts) > 2:
        return parts[-2]
    return "unknown"


def grouped_dataset_key(path_value: str, grouping: str) -> str:
    key = dataset_key(path_value)
    if grouping == "path":
        return key
    if grouping in {"cod_reproduction_512", "gencodec_reproduction"}:
        if key in {"clic2020_test_professional", "clic2020_test_mobile"}:
            return "clic2020_test"
        return key
    raise ValueError(f"unknown grouping: {grouping}")


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return float(math.sqrt(sum((value - avg) ** 2 for value in values) / len(values)))


def summarize_rows(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(rows)}
    for metric in metrics:
        metric_values = [row[metric] for row in rows if metric in row]
        values = [float(value) for value in metric_values]
        if not values:
            continue
        metric_summary: dict[str, Any] = {
            "mean": mean(values),
            "std": std(values),
            "min": min(values),
            "max": max(values),
        }
        if all(isinstance(value, bool) for value in metric_values):
            true_count = sum(1 for value in metric_values if bool(value))
            metric_summary.update(
                {
                    "all_true": true_count == len(metric_values),
                    "true_count": true_count,
                    "false_count": len(metric_values) - true_count,
                }
            )
        summary[metric] = metric_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--metric", action="append", default=None)
    parser.add_argument("--grouping", choices=GROUPINGS, default="path")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    metrics = tuple(args.metric) if args.metric else DEFAULT_METRICS
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(grouped_dataset_key(str(row["path"]), args.grouping), []).append(row)

    payload = {
        "label": args.label or args.input.parent.name,
        "input": str(args.input),
        "grouping": args.grouping,
        "overall": summarize_rows(rows, metrics),
        "by_dataset": {key: summarize_rows(group_rows, metrics) for key, group_rows in sorted(groups.items())},
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
