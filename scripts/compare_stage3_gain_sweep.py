from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUMMARY_METRICS = (
    "actual_payload_bpp_mean",
    "semantic_payload_bpp_mean",
    "detail_payload_bpp_mean",
    "stage3_psnr_mean",
    "stage3_psnr_delta_vs_semantic_only",
    "stage3_ms_ssim_mean",
    "stage3_ms_ssim_delta_vs_semantic_only",
    "stage3_lpips_alex_mean",
    "stage3_lpips_alex_delta_vs_semantic_only",
    "stage3_dists_mean",
    "stage3_dists_delta_vs_semantic_only",
)

QUALITY_DIRECTIONS = {
    "stage3_psnr": "higher",
    "stage3_ms_ssim": "higher",
    "stage3_lpips_alex": "lower",
    "stage3_dists": "lower",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("path") or row.get("index"))
        if not key:
            raise ValueError(f"{path}:{line_number} has no path or index")
        rows[key] = row
    return rows


def metric_win_count(
    baseline_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
    metric: str,
    direction: str,
) -> int:
    keys = sorted(set(baseline_rows) & set(candidate_rows))
    wins = 0
    for key in keys:
        base_value = float(baseline_rows[key][metric])
        candidate_value = float(candidate_rows[key][metric])
        if direction == "higher":
            wins += int(candidate_value > base_value)
        elif direction == "lower":
            wins += int(candidate_value < base_value)
        else:
            raise ValueError(f"unknown direction: {direction}")
    return wins


def semantic_improvement_count(rows: dict[str, dict[str, Any]], metric: str, direction: str) -> int:
    delta_key = f"{metric}_delta_vs_semantic_only"
    count = 0
    for row in rows.values():
        delta = float(row[delta_key])
        if direction == "higher":
            count += int(delta > 0)
        elif direction == "lower":
            count += int(delta < 0)
        else:
            raise ValueError(f"unknown direction: {direction}")
    return count


def build_row(summary_path: Path, baseline_rows: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    summary = read_json(summary_path)
    row: dict[str, Any] = {
        "run_name": summary_path.parent.name,
        "summary": str(summary_path),
        "detail_gain": float(summary.get("detail_gain", 1.0)),
        "num_images": int(summary["num_images"]),
        "per_image_metrics": summary.get("per_image_metrics", ""),
    }
    for metric in SUMMARY_METRICS:
        if metric in summary:
            row[metric] = float(summary[metric])

    per_image_path = Path(str(summary.get("per_image_metrics", "")))
    if per_image_path.exists():
        rows = read_jsonl(per_image_path)
        for metric, direction in QUALITY_DIRECTIONS.items():
            row[f"{metric}_improves_vs_semantic_only_count"] = semantic_improvement_count(rows, metric, direction)
        if baseline_rows is not None:
            for metric, direction in QUALITY_DIRECTIONS.items():
                row[f"{metric}_wins_vs_baseline_count"] = metric_win_count(baseline_rows, rows, metric, direction)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--candidate-summary", action="append", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    baseline = read_json(args.baseline_summary)
    baseline_per_image = Path(str(baseline.get("per_image_metrics", "")))
    baseline_rows = read_jsonl(baseline_per_image) if baseline_per_image.exists() else None
    rows = [build_row(args.baseline_summary, None)]
    rows.extend(build_row(path, baseline_rows) for path in args.candidate_summary)
    rows = sorted(rows, key=lambda row: row["detail_gain"])
    payload = {
        "baseline_summary": str(args.baseline_summary),
        "baseline_detail_gain": float(baseline.get("detail_gain", 1.0)),
        "rows": rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
