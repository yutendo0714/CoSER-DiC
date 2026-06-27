from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_STRICT_FIELDS = (
    "semantic_payload_bytes",
    "detail_payload_bytes",
    "actual_payload_bpp",
    "paper_bpp",
    "semantic_payload_bpp",
    "detail_payload_bpp",
    "semantic_topk_hit_rate",
    "semantic_token_roundtrip",
    "detail_code_roundtrip",
)

DEFAULT_TOLERANCE_FIELDS = (
    "semantic_only_psnr",
    "stage3_psnr",
    "semantic_only_l1",
    "stage3_l1",
    "semantic_only_ms_ssim",
    "stage3_ms_ssim",
    "residual_grid_abs_mean",
    "residual_grid_std",
    "detail_code_entropy_bits",
)


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


def audit_rows(
    reference_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
    *,
    strict_fields: tuple[str, ...],
    tolerance_fields: tuple[str, ...],
    atol: float,
) -> dict[str, Any]:
    reference_keys = set(reference_rows)
    candidate_keys = set(candidate_rows)
    common_keys = sorted(reference_keys & candidate_keys)
    strict_mismatches: list[dict[str, Any]] = []
    tolerance_mismatches: list[dict[str, Any]] = []
    tolerance_max_abs_delta: dict[str, float] = {field: 0.0 for field in tolerance_fields}

    for key in common_keys:
        reference = reference_rows[key]
        candidate = candidate_rows[key]
        for field in strict_fields:
            if field in reference or field in candidate:
                if reference.get(field) != candidate.get(field):
                    strict_mismatches.append(
                        {
                            "key": key,
                            "field": field,
                            "reference": reference.get(field),
                            "candidate": candidate.get(field),
                        }
                    )
        for field in tolerance_fields:
            if field in reference or field in candidate:
                ref_value = float(reference.get(field))
                cand_value = float(candidate.get(field))
                delta = abs(cand_value - ref_value)
                tolerance_max_abs_delta[field] = max(tolerance_max_abs_delta[field], delta)
                if delta > atol:
                    tolerance_mismatches.append(
                        {
                            "key": key,
                            "field": field,
                            "reference": ref_value,
                            "candidate": cand_value,
                            "abs_delta": delta,
                        }
                    )

    return {
        "num_reference_rows": len(reference_rows),
        "num_candidate_rows": len(candidate_rows),
        "num_common_rows": len(common_keys),
        "missing_in_candidate": sorted(reference_keys - candidate_keys),
        "extra_in_candidate": sorted(candidate_keys - reference_keys),
        "strict_fields": list(strict_fields),
        "tolerance_fields": list(tolerance_fields),
        "atol": atol,
        "strict_mismatch_count": len(strict_mismatches),
        "tolerance_mismatch_count": len(tolerance_mismatches),
        "strict_mismatches": strict_mismatches[:50],
        "tolerance_mismatches": tolerance_mismatches[:50],
        "tolerance_max_abs_delta": tolerance_max_abs_delta,
        "pass": not strict_mismatches
        and not tolerance_mismatches
        and not (reference_keys - candidate_keys)
        and not (candidate_keys - reference_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--strict-field", action="append", default=None)
    parser.add_argument("--tolerance-field", action="append", default=None)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    payload = audit_rows(
        read_jsonl(args.reference),
        read_jsonl(args.candidate),
        strict_fields=tuple(args.strict_field or DEFAULT_STRICT_FIELDS),
        tolerance_fields=tuple(args.tolerance_field or DEFAULT_TOLERANCE_FIELDS),
        atol=args.atol,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(payload, indent=2, allow_nan=False))
    if args.fail_on_mismatch and not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
