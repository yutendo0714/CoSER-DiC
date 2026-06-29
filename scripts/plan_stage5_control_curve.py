from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.sweep_stage5_counted_control import load_setting_rows_from_json, optional_row_float, sort_setting_rows
except ModuleNotFoundError:  # pragma: no cover - used when invoked as a script from scripts/
    from sweep_stage5_counted_control import load_setting_rows_from_json, optional_row_float, sort_setting_rows


def parse_band(text: str) -> dict[str, object]:
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("band must be low:high or low:high:count")
    low = float(parts[0])
    high = float(parts[1])
    count = int(parts[2]) if len(parts) == 3 else 1
    if low < 0 or high <= low:
        raise ValueError("band must satisfy 0 <= low < high")
    if count <= 0:
        raise ValueError("band count must be positive")
    return {
        "label": f"control_bpp_{low:g}_{high:g}",
        "low": low,
        "high": high,
        "count": count,
    }


def row_control_bpp(row: dict[str, object]) -> float:
    value = optional_row_float(row, "control_bpp")
    if value is None:
        raise ValueError("every planned curve setting must contain control_bpp")
    return value


def select_band_rows(
    rows: list[dict[str, object]],
    *,
    band: dict[str, object],
    rank_by: str,
    allow_duplicates: bool,
    used_settings: set[str],
) -> list[dict[str, object]]:
    low = float(band["low"])
    high = float(band["high"])
    count = int(band["count"])
    candidates = []
    for row in rows:
        setting = str(row.get("setting", ""))
        if not setting:
            continue
        if not allow_duplicates and setting in used_settings:
            continue
        bpp = row_control_bpp(row)
        if low <= bpp < high:
            candidates.append(dict(row))
    ranked = sort_setting_rows(candidates, sort_by=rank_by)
    selected = []
    for index, row in enumerate(ranked[:count]):
        setting = str(row["setting"])
        used_settings.add(setting)
        row["curve_band"] = band["label"]
        row["curve_band_low"] = low
        row["curve_band_high"] = high
        row["curve_rank_in_band"] = index
        row["curve_rank_by"] = rank_by
        retained = optional_row_float(row, "basis_retained_energy_fraction_mean") or 0.0
        control_bpp = max(row_control_bpp(row), 1.0e-12)
        row["curve_retained_per_control_bpp"] = retained / control_bpp
        selected.append(row)
    return selected


def plan_curve_settings(
    rows: list[dict[str, object]],
    *,
    bands: list[dict[str, object]],
    rank_by: str,
    allow_duplicates: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected: list[dict[str, object]] = []
    band_summaries: list[dict[str, object]] = []
    used_settings: set[str] = set()
    for band in bands:
        rows_for_band = select_band_rows(
            rows,
            band=band,
            rank_by=rank_by,
            allow_duplicates=allow_duplicates,
            used_settings=used_settings,
        )
        selected.extend(rows_for_band)
        band_summaries.append(
            {
                "label": band["label"],
                "low": band["low"],
                "high": band["high"],
                "requested_count": band["count"],
                "selected_count": len(rows_for_band),
            }
        )
    return selected, band_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings-json", action="append", required=True)
    parser.add_argument("--band", action="append", required=True, help="control-bpp band as low:high or low:high:count")
    parser.add_argument(
        "--rank-by",
        choices=("input", "control_bpp", "retained_energy", "residual_energy", "retained_per_bpp", "rmse"),
        default="retained_per_bpp",
    )
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("--max-control-bpp", type=float, default=0.0)
    parser.add_argument("--min-basis-retained-energy", type=float, default=0.0)
    parser.add_argument("--max-basis-residual-energy", type=float, default=1.0)
    parser.add_argument("--max-quantization-rmse", type=float, default=0.0)
    parser.add_argument("--max-clipped-fraction", type=float, default=1.0)
    parser.add_argument("--include-codec", action="append", default=[])
    parser.add_argument("--include-quantizer", action="append", default=[])
    parser.add_argument("--include-quantile", action="append", default=[])
    parser.add_argument("--include-components", nargs="+", type=int, default=[])
    parser.add_argument("--include-candidate-components", nargs="+", type=int, default=[])
    parser.add_argument("--include-selection", action="append", default=[])
    parser.add_argument("--include-mode", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_setting_rows_from_json(
        [Path(path) for path in args.settings_json],
        max_control_bpp=args.max_control_bpp,
        min_basis_retained_energy=args.min_basis_retained_energy,
        max_basis_residual_energy=args.max_basis_residual_energy,
        max_quantization_rmse=args.max_quantization_rmse,
        max_clipped_fraction=args.max_clipped_fraction,
        include_codecs=set(args.include_codec) if args.include_codec else None,
        include_quantizers=set(args.include_quantizer) if args.include_quantizer else None,
        include_quantiles=set(args.include_quantile) if args.include_quantile else None,
        include_components=set(args.include_components) if args.include_components else None,
        include_candidate_components=set(args.include_candidate_components) if args.include_candidate_components else None,
        include_selections=set(args.include_selection) if args.include_selection else None,
        include_modes=set(args.include_mode) if args.include_mode else None,
        sort_by="input",
    )
    bands = [parse_band(text) for text in args.band]
    selected, band_summaries = plan_curve_settings(
        rows,
        bands=bands,
        rank_by=args.rank_by,
        allow_duplicates=args.allow_duplicates,
    )
    payload = {
        "source_settings_json": args.settings_json,
        "rank_by": args.rank_by,
        "filters": {
            "max_control_bpp": args.max_control_bpp,
            "min_basis_retained_energy": args.min_basis_retained_energy,
            "max_basis_residual_energy": args.max_basis_residual_energy,
            "max_quantization_rmse": args.max_quantization_rmse,
            "max_clipped_fraction": args.max_clipped_fraction,
            "include_codec": args.include_codec,
            "include_quantizer": args.include_quantizer,
            "include_quantile": args.include_quantile,
            "include_components": args.include_components,
            "include_candidate_components": args.include_candidate_components,
            "include_selection": args.include_selection,
            "include_mode": args.include_mode,
        },
        "bands": band_summaries,
        "settings": selected,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
