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
    "semantic_topk_hit_rate",
    "residual_grid_abs_mean",
    "residual_grid_std",
    "residual_grid_clip_ratio",
    "detail_code_entropy_bits",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "path" not in row:
            raise ValueError(f"{path}:{line_number} has no path")
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
        values = [float(row[metric]) for row in rows if metric in row]
        if not values:
            continue
        summary[metric] = {
            "mean": mean(values),
            "std": std(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--metric", action="append", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    metrics = tuple(args.metric) if args.metric else DEFAULT_METRICS
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(dataset_key(str(row["path"])), []).append(row)

    payload = {
        "label": args.label or args.input.parent.name,
        "input": str(args.input),
        "overall": summarize_rows(rows, metrics),
        "by_dataset": {key: summarize_rows(group_rows, metrics) for key, group_rows in sorted(groups.items())},
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
