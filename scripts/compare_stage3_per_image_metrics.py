from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("path") or row.get("index"))
        if not key:
            raise ValueError(f"{path}:{line_number} has no path or index")
        if key in rows:
            raise ValueError(f"{path}:{line_number} duplicates key {key}")
        rows[key] = row
    return rows


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def compare(
    reference_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
    *,
    reference_label: str,
    candidate_label: str,
    rate_metric: str,
    quality_metric: str,
    quality_direction: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    keys = sorted(set(reference_rows) & set(candidate_rows))
    if not keys:
        raise ValueError("no overlapping image keys")

    per_image: list[dict[str, Any]] = []
    rate_deltas: list[float] = []
    quality_deltas: list[float] = []
    candidate_rate_wins = 0
    candidate_quality_wins = 0
    for key in keys:
        ref = reference_rows[key]
        cand = candidate_rows[key]
        ref_rate = float(ref[rate_metric])
        cand_rate = float(cand[rate_metric])
        ref_quality = float(ref[quality_metric])
        cand_quality = float(cand[quality_metric])
        rate_delta = cand_rate - ref_rate
        quality_delta = cand_quality - ref_quality
        rate_deltas.append(rate_delta)
        quality_deltas.append(quality_delta)
        candidate_rate_wins += int(rate_delta < 0)
        if quality_direction == "higher":
            candidate_quality_wins += int(quality_delta > 0)
        elif quality_direction == "lower":
            candidate_quality_wins += int(quality_delta < 0)
        else:
            raise ValueError(f"unknown quality direction: {quality_direction}")
        per_image.append(
            {
                "key": key,
                f"{reference_label}_{rate_metric}": ref_rate,
                f"{candidate_label}_{rate_metric}": cand_rate,
                f"{candidate_label}_minus_{reference_label}_{rate_metric}": rate_delta,
                f"{reference_label}_{quality_metric}": ref_quality,
                f"{candidate_label}_{quality_metric}": cand_quality,
                f"{candidate_label}_minus_{reference_label}_{quality_metric}": quality_delta,
            }
        )

    summary = {
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "num_images": len(keys),
        "rate_metric": rate_metric,
        "quality_metric": quality_metric,
        "quality_direction": quality_direction,
        "reference_rate_mean": mean([float(reference_rows[key][rate_metric]) for key in keys]),
        "candidate_rate_mean": mean([float(candidate_rows[key][rate_metric]) for key in keys]),
        "candidate_minus_reference_rate_mean": mean(rate_deltas),
        "reference_quality_mean": mean([float(reference_rows[key][quality_metric]) for key in keys]),
        "candidate_quality_mean": mean([float(candidate_rows[key][quality_metric]) for key in keys]),
        "candidate_minus_reference_quality_mean": mean(quality_deltas),
        "candidate_rate_win_count": candidate_rate_wins,
        "candidate_quality_win_count": candidate_quality_wins,
    }
    return summary, per_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--rate-metric", default="actual_payload_bpp")
    parser.add_argument("--quality-metric", default="stage3_psnr")
    parser.add_argument("--quality-direction", choices=["higher", "lower"], default="higher")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    summary, per_image = compare(
        read_jsonl(args.reference),
        read_jsonl(args.candidate),
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
        rate_metric=args.rate_metric,
        quality_metric=args.quality_metric,
        quality_direction=args.quality_direction,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"summary": summary, "per_image": per_image}, indent=2, allow_nan=False))
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
            writer.writeheader()
            writer.writerows(per_image)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
