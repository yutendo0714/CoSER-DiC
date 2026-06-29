from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LOWER_IS_BETTER = {"lpips", "dists", "fid", "kid", "condition_l1"}
HIGHER_IS_BETTER = {"psnr", "ms_ssim"}


def _as_float(value: object, *, key: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite, got {number!r}")
    return number


def first_float(row: dict[str, object], keys: tuple[str, ...], *, default: float | None = None) -> float | None:
    for key in keys:
        if key in row and row[key] is not None:
            return _as_float(row[key], key=key)
    return default


def load_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    payload.setdefault("summary_path", str(path))
    return payload


def iter_summary_paths_from_sweep_plan(path: Path) -> list[Path]:
    return [item[0] for item in iter_summary_entries_from_sweep_plan(path)]


def iter_summary_entries_from_sweep_plan(path: Path) -> list[tuple[Path, dict[str, object]]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("sweep plan must be a JSON object")
    settings = payload.get("settings", payload.get("commands"))
    if not isinstance(settings, list):
        raise ValueError("sweep/promotion plan must contain a settings or commands list")
    entries: list[tuple[Path, dict[str, object]]] = []
    for item in settings:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary")
        if not isinstance(summary, str):
            summary = _summary_path_from_plan_command(item)
        if isinstance(summary, str):
            metadata = {
                key: item[key]
                for key in (
                    "run_name",
                    "promoted_run_name",
                    "source_label",
                    "source_summary",
                    "preset",
                    "module_count",
                    "lora_param_count",
                    "base_param_count",
                    "control_bytes",
                    "control_bpp",
                    "control_bytes_note",
                    "control_basis",
                    "condition_residual_guard",
                    "condition_residual_guard_granularity",
                    "condition_residual_max_rms_ratio",
                    "condition_residual_min_gate",
                    "basis_reconstruction_key",
                    "basis_retained_energy_fraction_mean",
                    "basis_retained_energy_fraction_p50",
                    "basis_residual_energy_fraction_mean",
                    "basis_residual_l2_mean",
                    "quantization_mae",
                    "quantization_rmse",
                    "clipped_fraction",
                )
                if key in item
            }
            source_metadata = item.get("source_setting_metadata")
            if isinstance(source_metadata, dict):
                metadata["source_setting_metadata"] = dict(source_metadata)
            entries.append((Path(summary), metadata))
    return entries


def _command_value(command: list[str], flag: str) -> str:
    if flag not in command:
        return ""
    index = command.index(flag)
    if index + 1 >= len(command):
        return ""
    return command[index + 1]


def _summary_path_from_plan_command(row: dict[str, object]) -> str:
    command = row.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return ""
    run_name = _command_value(command, "--run-name") or str(row.get("promoted_run_name", ""))
    if not run_name:
        return ""
    output_dir = _command_value(command, "--output-dir") or "results/stage4_cod_lite_adapter_eval"
    return str(Path(output_dir) / run_name / "summary.json")


def load_reference_curve(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"reference curve must be a JSON object: {path}")
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"reference curve must contain non-empty points: {path}")
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError(f"reference point {index} must be an object")
        first_float(point, ("bpp", "actual_payload_bpp", "paper_bpp"), default=None)
    payload.setdefault("source_path", str(path))
    return payload


def _sorted_curve_points(curve: dict[str, object], metric: str) -> list[tuple[float, float]]:
    points = curve.get("points")
    if not isinstance(points, list):
        raise ValueError("reference curve points must be a list")
    rows: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, dict) or metric not in point:
            continue
        bpp = first_float(point, ("bpp", "actual_payload_bpp", "paper_bpp"))
        value = first_float(point, (metric,))
        if bpp is None or value is None:
            continue
        rows.append((bpp, value))
    return sorted(rows, key=lambda item: item[0])


def interpolate_metric_at_bpp(curve: dict[str, object], metric: str, bpp: float) -> float | None:
    rows = _sorted_curve_points(curve, metric)
    if len(rows) < 2:
        return None
    if bpp < rows[0][0] or bpp > rows[-1][0]:
        return None
    for (b0, m0), (b1, m1) in zip(rows, rows[1:]):
        if b0 <= bpp <= b1:
            if b1 == b0:
                return m0
            t = (bpp - b0) / (b1 - b0)
            return m0 * (1.0 - t) + m1 * t
    return None


def interpolate_bpp_for_metric(curve: dict[str, object], metric: str, value: float) -> float | None:
    rows = _sorted_curve_points(curve, metric)
    if len(rows) < 2:
        return None
    metric_rows = sorted(rows, key=lambda item: item[1])
    if value < metric_rows[0][1] or value > metric_rows[-1][1]:
        return None
    for (b0, m0), (b1, m1) in zip(metric_rows, metric_rows[1:]):
        if m0 <= value <= m1:
            if m1 == m0:
                return b0
            t = (value - m0) / (m1 - m0)
            return b0 * (1.0 - t) + b1 * t
    return None


def add_reference_curve_gaps(rows: list[dict[str, object]], curve: dict[str, object], *, metrics: list[str]) -> None:
    curve_name = str(curve.get("name", curve.get("source_path", "reference_curve")))
    for row in rows:
        bpp = float(row["actual_payload_bpp"])
        for metric in metrics:
            if metric not in row:
                continue
            reference_metric = interpolate_metric_at_bpp(curve, metric, bpp)
            if reference_metric is not None:
                row[f"reference_{metric}_at_bpp"] = reference_metric
                row[f"{metric}_gap_vs_reference_at_bpp"] = float(row[metric]) - reference_metric
                if metric in LOWER_IS_BETTER:
                    row[f"{metric}_beats_reference_at_bpp"] = float(row[metric]) <= reference_metric
                elif metric in HIGHER_IS_BETTER:
                    row[f"{metric}_beats_reference_at_bpp"] = float(row[metric]) >= reference_metric
            reference_bpp = interpolate_bpp_for_metric(curve, metric, float(row[metric]))
            if reference_bpp is not None:
                row[f"reference_bpp_for_{metric}"] = reference_bpp
                row[f"{metric}_single_point_rate_proxy_percent"] = (bpp / reference_bpp - 1.0) * 100.0
        row["reference_curve_name"] = curve_name


def normalize_candidate(summary: dict[str, object], *, missing_ok: bool = False) -> dict[str, object]:
    label = str(summary.get("run_name", summary.get("label", Path(str(summary.get("summary_path", ""))).parent.name)))
    actual_bpp = first_float(summary, ("actual_payload_bpp_mean", "paper_bpp_mean", "actual_payload_bpp", "paper_bpp"))
    if actual_bpp is None:
        raise ValueError(f"{label}: missing actual payload bpp")
    row: dict[str, object] = {
        "label": label,
        "summary_path": str(summary.get("summary_path", "")),
        "count": first_float(summary, ("count",), default=None),
        "actual_payload_bpp": actual_bpp,
        "stage3_actual_payload_bpp": first_float(summary, ("stage3_actual_payload_bpp_mean",), default=actual_bpp),
        "control_payload_bpp": first_float(summary, ("control_payload_bpp_mean",), default=0.0),
        "control_payload_bytes": first_float(summary, ("control_payload_bytes_mean",), default=0.0),
        "counted_control_mode": summary.get("counted_control_mode", ""),
        "control_codec_type": summary.get("control_codec_type", ""),
        "control_quantizer": summary.get("control_quantizer", ""),
        "control_huffman_key": summary.get("control_huffman_key", ""),
        "control_basis_components": summary.get("control_basis_components", ""),
        "control_grid_size": summary.get("control_grid_size", ""),
        "control_groups": summary.get("control_groups", ""),
        "control_affine_groups": summary.get("control_affine_groups", ""),
        "control_affine_grid_size": summary.get("control_affine_grid_size", ""),
        "control_range": summary.get("control_range", ""),
        "control_affine_gain_range": summary.get("control_affine_gain_range", ""),
        "control_affine_bias_range": summary.get("control_affine_bias_range", ""),
        "control_scale": summary.get("control_scale", ""),
        "basis_reconstruction_key": summary.get("basis_reconstruction_key", ""),
        "lora_preset": summary.get("lora_preset", summary.get("preset", "")),
        "lora_param_count": summary.get("lora_param_count", ""),
        "lora_module_count": summary.get("lora_module_count", summary.get("module_count", "")),
        "lora_base_param_count": summary.get("lora_base_param_count", summary.get("base_param_count", "")),
        "lora_rationale": summary.get("lora_rationale", ""),
        "condition_residual_guard": summary.get("condition_residual_guard", ""),
        "condition_residual_guard_granularity": summary.get("condition_residual_guard_granularity", ""),
        "condition_residual_max_rms_ratio": summary.get("condition_residual_max_rms_ratio", ""),
        "condition_residual_min_gate": summary.get("condition_residual_min_gate", ""),
    }
    for key in (
        "basis_retained_energy_fraction_mean",
        "basis_retained_energy_fraction_p50",
        "basis_residual_energy_fraction_mean",
        "basis_residual_l2_mean",
        "quantization_mae",
        "quantization_rmse",
        "clipped_fraction",
    ):
        value = first_float(summary, (key,))
        if value is not None:
            row[key] = value
    source_setting_metadata = summary.get("source_setting_metadata")
    if isinstance(source_setting_metadata, dict):
        row["source_setting_metadata"] = dict(source_setting_metadata)
        for key in (
            "lora_preset",
            "lora_rank",
            "lora_alpha",
            "lora_lr",
            "lora_param_count",
            "lora_module_count",
            "lora_rationale",
            "counted_control_mode",
            "control_grid_size",
            "control_groups",
            "control_affine_groups",
            "control_affine_grid_size",
            "control_range",
            "control_affine_gain_range",
            "control_affine_bias_range",
            "control_scale",
            "control_codec_type",
            "control_quantizer",
        ):
            if key in source_setting_metadata:
                row[key] = source_setting_metadata[key]
    metric_map = {
        "psnr": ("stage4_psnr_mean", "psnr_mean", "psnr"),
        "ms_ssim": ("stage4_ms_ssim_mean", "ms_ssim_mean", "ms_ssim"),
        "lpips": ("stage4_lpips_alex_mean", "lpips_alex_mean", "lpips"),
        "dists": ("stage4_dists_mean", "dists_mean", "dists"),
    }
    missing: list[str] = []
    for metric, keys in metric_map.items():
        value = first_float(summary, keys)
        if value is None:
            missing.append(metric)
        else:
            row[metric] = value
    if missing and not missing_ok:
        raise ValueError(f"{label}: missing metrics: {', '.join(missing)}")
    condition_l1 = first_float(summary, ("condition_l1_mean",))
    if condition_l1 is not None:
        row["condition_l1"] = condition_l1
    stage3_map = {
        "stage3_psnr": ("stage3_psnr_mean",),
        "stage3_ms_ssim": ("stage3_ms_ssim_mean",),
        "stage3_lpips": ("stage3_lpips_alex_mean",),
        "stage3_dists": ("stage3_dists_mean",),
    }
    for key, keys in stage3_map.items():
        value = first_float(summary, keys)
        if value is not None:
            row[key] = value
    add_stage3_deltas(row)
    return row


def add_stage3_deltas(row: dict[str, object]) -> None:
    pairs = (
        ("psnr", "stage3_psnr", "psnr_delta_vs_stage3"),
        ("ms_ssim", "stage3_ms_ssim", "ms_ssim_delta_vs_stage3"),
        ("lpips", "stage3_lpips", "lpips_delta_vs_stage3"),
        ("dists", "stage3_dists", "dists_delta_vs_stage3"),
    )
    for metric, anchor, out_key in pairs:
        if metric in row and anchor in row:
            row[out_key] = float(row[metric]) - float(row[anchor])


def add_anchor_deltas(row: dict[str, object], anchor: dict[str, object]) -> None:
    row["anchor_label"] = anchor.get("label", "")
    if row.get("count") is not None and anchor.get("count") is not None:
        row["count_delta_vs_anchor"] = int(float(row["count"])) - int(float(anchor["count"]))
        row["anchor_count_match"] = int(float(row["count"])) == int(float(anchor["count"]))
    row["actual_payload_bpp_delta_vs_anchor"] = float(row["actual_payload_bpp"]) - float(anchor["actual_payload_bpp"])
    for metric in ("psnr", "ms_ssim", "lpips", "dists", "condition_l1"):
        if metric not in row or metric not in anchor:
            continue
        delta = float(row[metric]) - float(anchor[metric])
        row[f"{metric}_delta_vs_anchor"] = delta
        if metric in LOWER_IS_BETTER:
            row[f"{metric}_improves_vs_anchor"] = delta < 0.0
        elif metric in HIGHER_IS_BETTER:
            row[f"{metric}_improves_vs_anchor"] = delta > 0.0


def find_anchor_row(rows: list[dict[str, object]], label: str) -> dict[str, object]:
    matches = [row for row in rows if str(row.get("label", "")) == label]
    if len(matches) != 1:
        raise ValueError(f"anchor label {label!r} matched {len(matches)} rows")
    return matches[0]


def anchor_count_mismatch_warnings(rows: list[dict[str, object]], anchor: dict[str, object] | None) -> list[dict[str, object]]:
    if anchor is None or anchor.get("count") is None:
        return []
    anchor_count = int(float(anchor["count"]))
    warnings: list[dict[str, object]] = []
    for row in rows:
        if row.get("count") is None:
            continue
        row_count = int(float(row["count"]))
        if row_count != anchor_count:
            warnings.append(
                {
                    "label": row.get("label", ""),
                    "summary_path": row.get("summary_path", ""),
                    "count": row_count,
                    "anchor_label": anchor.get("label", ""),
                    "anchor_count": anchor_count,
                    "warning": (
                        "candidate/anchor counts differ; compare limit64 candidates against a limit64 anchor "
                        "and full552 candidates against a full552 anchor"
                    ),
                }
            )
    return warnings


def metric_improves(candidate: dict[str, object], other: dict[str, object], metric: str, *, eps: float = 0.0) -> bool:
    cand = float(candidate[metric])
    ref = float(other[metric])
    if metric in LOWER_IS_BETTER:
        return cand < ref - eps
    if metric in HIGHER_IS_BETTER:
        return cand > ref + eps
    raise ValueError(f"unknown metric direction: {metric}")


def metric_not_worse(candidate: dict[str, object], other: dict[str, object], metric: str, *, eps: float = 0.0) -> bool:
    cand = float(candidate[metric])
    ref = float(other[metric])
    if metric in LOWER_IS_BETTER:
        return cand <= ref + eps
    if metric in HIGHER_IS_BETTER:
        return cand >= ref - eps
    raise ValueError(f"unknown metric direction: {metric}")


def dominates(candidate: dict[str, object], other: dict[str, object], *, metrics: list[str], eps: float) -> bool:
    if float(candidate["actual_payload_bpp"]) > float(other["actual_payload_bpp"]) + eps:
        return False
    comparable_metrics = [metric for metric in metrics if metric in candidate and metric in other]
    if not comparable_metrics:
        return False
    not_worse = all(metric_not_worse(candidate, other, metric, eps=eps) for metric in comparable_metrics)
    strictly_better_rate = float(candidate["actual_payload_bpp"]) < float(other["actual_payload_bpp"]) - eps
    strictly_better_metric = any(metric_improves(candidate, other, metric, eps=eps) for metric in comparable_metrics)
    return not_worse and (strictly_better_rate or strictly_better_metric)


def mark_pareto(rows: list[dict[str, object]], *, metrics: list[str], eps: float) -> None:
    for index, row in enumerate(rows):
        row["pareto"] = not any(
            other_index != index and dominates(other, row, metrics=metrics, eps=eps)
            for other_index, other in enumerate(rows)
        )


def stage3_guard_pass(
    row: dict[str, object],
    *,
    max_psnr_drop: float,
    max_ms_ssim_drop: float,
    max_lpips_increase: float,
    max_dists_increase: float,
) -> bool:
    checks = []
    if "psnr_delta_vs_stage3" in row:
        checks.append(float(row["psnr_delta_vs_stage3"]) >= -max_psnr_drop)
    if "ms_ssim_delta_vs_stage3" in row:
        checks.append(float(row["ms_ssim_delta_vs_stage3"]) >= -max_ms_ssim_drop)
    if "lpips_delta_vs_stage3" in row:
        checks.append(float(row["lpips_delta_vs_stage3"]) <= max_lpips_increase)
    if "dists_delta_vs_stage3" in row:
        checks.append(float(row["dists_delta_vs_stage3"]) <= max_dists_increase)
    return all(checks) if checks else True


def anchor_guard_pass(
    row: dict[str, object],
    *,
    max_psnr_drop: float,
    max_ms_ssim_drop: float,
    max_lpips_increase: float,
    max_dists_increase: float,
) -> bool:
    checks = []
    if "psnr_delta_vs_anchor" in row:
        checks.append(float(row["psnr_delta_vs_anchor"]) >= -max_psnr_drop)
    if "ms_ssim_delta_vs_anchor" in row:
        checks.append(float(row["ms_ssim_delta_vs_anchor"]) >= -max_ms_ssim_drop)
    if "lpips_delta_vs_anchor" in row:
        checks.append(float(row["lpips_delta_vs_anchor"]) <= max_lpips_increase)
    if "dists_delta_vs_anchor" in row:
        checks.append(float(row["dists_delta_vs_anchor"]) <= max_dists_increase)
    return all(checks) if checks else True


def basis_guard_pass(
    row: dict[str, object],
    *,
    min_retained_energy: float,
    max_residual_energy: float,
) -> bool:
    checks = []
    retained = first_float(row, ("basis_retained_energy_fraction_mean",), default=None)
    residual = first_float(row, ("basis_residual_energy_fraction_mean",), default=None)
    if retained is not None and min_retained_energy > 0.0:
        checks.append(retained >= min_retained_energy)
    if residual is not None and max_residual_energy < 1.0:
        checks.append(residual <= max_residual_energy)
    return all(checks) if checks else True


def add_scores(
    rows: list[dict[str, object]],
    *,
    lpips_weight: float,
    dists_weight: float,
    psnr_guard_drop: float,
    ms_ssim_guard_drop: float,
    lpips_guard_increase: float,
    dists_guard_increase: float,
    min_basis_retained_energy: float,
    max_basis_residual_energy: float,
    max_anchor_psnr_drop: float = 0.25,
    max_anchor_ms_ssim_drop: float = 0.01,
    max_anchor_lpips_increase: float = 0.0,
    max_anchor_dists_increase: float = 0.0,
) -> None:
    for row in rows:
        perceptual_gain = 0.0
        if "lpips_delta_vs_stage3" in row:
            perceptual_gain += lpips_weight * (-float(row["lpips_delta_vs_stage3"]))
        if "dists_delta_vs_stage3" in row:
            perceptual_gain += dists_weight * (-float(row["dists_delta_vs_stage3"]))
        control_bpp = max(float(row.get("control_payload_bpp", 0.0)), 1.0e-9)
        row["perceptual_gain_score"] = perceptual_gain
        row["perceptual_gain_per_control_bpp"] = perceptual_gain / control_bpp
        retained = first_float(row, ("basis_retained_energy_fraction_mean",), default=0.0) or 0.0
        residual = first_float(row, ("basis_residual_energy_fraction_mean",), default=1.0) or 1.0
        row["basis_retained_energy_per_control_bpp"] = retained / control_bpp
        row["basis_residual_energy_fraction_for_sort"] = residual
        row["stage3_guard_pass"] = stage3_guard_pass(
            row,
            max_psnr_drop=psnr_guard_drop,
            max_ms_ssim_drop=ms_ssim_guard_drop,
            max_lpips_increase=lpips_guard_increase,
            max_dists_increase=dists_guard_increase,
        )
        row["basis_guard_pass"] = basis_guard_pass(
            row,
            min_retained_energy=min_basis_retained_energy,
            max_residual_energy=max_basis_residual_energy,
        )
        row["anchor_guard_pass"] = anchor_guard_pass(
            row,
            max_psnr_drop=max_anchor_psnr_drop,
            max_ms_ssim_drop=max_anchor_ms_ssim_drop,
            max_lpips_increase=max_anchor_lpips_increase,
            max_dists_increase=max_anchor_dists_increase,
        )
        row["promotion_guard_pass"] = (
            bool(row["stage3_guard_pass"])
            and bool(row["basis_guard_pass"])
            and bool(row["anchor_guard_pass"])
        )


def sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def external_gap(row: dict[str, object]) -> float:
        gap = 0.0
        count = 0
        for key in ("lpips_gap_vs_reference_at_bpp", "dists_gap_vs_reference_at_bpp"):
            if key in row:
                gap += float(row[key])
                count += 1
        return gap / max(count, 1)

    return sorted(
        rows,
        key=lambda row: (
            not bool(row.get("promotion_guard_pass", row.get("stage3_guard_pass", False))),
            not bool(row.get("pareto", False)),
            external_gap(row),
            float(row.get("lpips_delta_vs_anchor", 0.0)) if "lpips_delta_vs_anchor" in row else 0.0,
            float(row.get("dists_delta_vs_anchor", 0.0)) if "dists_delta_vs_anchor" in row else 0.0,
            -float(row.get("perceptual_gain_score", 0.0)),
            -float(row.get("basis_retained_energy_fraction_mean", 0.0)),
            float(row.get("basis_residual_energy_fraction_mean", 1.0)),
            float(row.get("actual_payload_bpp", 0.0)),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", default=[], help="summary.json path")
    parser.add_argument("--sweep-plan", action="append", default=[], help="sweep plan JSON from sweep_stage5_counted_control.py")
    parser.add_argument("--anchor-summary", default="", help="summary.json path for current anchor/baseline.")
    parser.add_argument("--anchor-label", default="", help="Use a loaded candidate label as the anchor/baseline.")
    parser.add_argument("--fail-on-anchor-count-mismatch", action="store_true")
    parser.add_argument("--reference-curve", action="append", default=[], help="External baseline curve JSON with points.")
    parser.add_argument("--reference-metric", action="append", default=[], help="Metric to compare to reference curves.")
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1.0e-12)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--max-psnr-drop", type=float, default=0.75)
    parser.add_argument("--max-ms-ssim-drop", type=float, default=0.025)
    parser.add_argument("--max-lpips-increase", type=float, default=0.0)
    parser.add_argument("--max-dists-increase", type=float, default=0.0)
    parser.add_argument("--max-anchor-psnr-drop", type=float, default=0.25)
    parser.add_argument("--max-anchor-ms-ssim-drop", type=float, default=0.01)
    parser.add_argument("--max-anchor-lpips-increase", type=float, default=0.0)
    parser.add_argument("--max-anchor-dists-increase", type=float, default=0.0)
    parser.add_argument("--min-basis-retained-energy", type=float, default=0.0)
    parser.add_argument("--max-basis-residual-energy", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=1.0)
    parser.add_argument("--dists-weight", type=float, default=1.0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries: list[tuple[Path, dict[str, object]]] = [(Path(value), {}) for value in args.summary]
    for sweep_plan in args.sweep_plan:
        entries.extend(iter_summary_entries_from_sweep_plan(Path(sweep_plan)))
    if not entries:
        raise ValueError("provide at least one --summary or --sweep-plan")
    existing_entries = [(path, metadata) for path, metadata in entries if path.exists()]
    missing_paths = [str(path) for path, _ in entries if not path.exists()]
    rows = []
    for path, metadata in existing_entries:
        summary = load_summary(path)
        for key, value in metadata.items():
            summary.setdefault(key, value)
        rows.append(normalize_candidate(summary, missing_ok=args.allow_missing))
    anchor: dict[str, object] | None = None
    if args.anchor_summary:
        anchor = normalize_candidate(load_summary(Path(args.anchor_summary)), missing_ok=args.allow_missing)
    if args.anchor_label:
        label_anchor = find_anchor_row(rows, args.anchor_label)
        if anchor is not None and str(anchor.get("label", "")) != str(label_anchor.get("label", "")):
            raise ValueError("--anchor-summary and --anchor-label refer to different anchors")
        anchor = label_anchor
    if anchor is not None:
        for row in rows:
            add_anchor_deltas(row, anchor)
    anchor_count_warnings = anchor_count_mismatch_warnings(rows, anchor)
    if anchor_count_warnings and args.fail_on_anchor_count_mismatch:
        raise ValueError(
            "anchor/candidate count mismatch; use matching split anchors. "
            f"First mismatch: {anchor_count_warnings[0]}"
        )
    metrics = args.metric or ["lpips", "dists", "psnr", "ms_ssim"]
    mark_pareto(rows, metrics=metrics, eps=args.eps)
    add_scores(
        rows,
        lpips_weight=args.lpips_weight,
        dists_weight=args.dists_weight,
        psnr_guard_drop=args.max_psnr_drop,
        ms_ssim_guard_drop=args.max_ms_ssim_drop,
        lpips_guard_increase=args.max_lpips_increase,
        dists_guard_increase=args.max_dists_increase,
        min_basis_retained_energy=args.min_basis_retained_energy,
        max_basis_residual_energy=args.max_basis_residual_energy,
        max_anchor_psnr_drop=args.max_anchor_psnr_drop,
        max_anchor_ms_ssim_drop=args.max_anchor_ms_ssim_drop,
        max_anchor_lpips_increase=args.max_anchor_lpips_increase,
        max_anchor_dists_increase=args.max_anchor_dists_increase,
    )
    reference_curves = [load_reference_curve(Path(path)) for path in args.reference_curve]
    reference_metrics = args.reference_metric or ["lpips", "dists"]
    for curve in reference_curves:
        add_reference_curve_gaps(rows, curve, metrics=reference_metrics)
    ranked = sort_rows(rows)
    recommended_rows = [
        row
        for row in ranked
        if bool(row.get("promotion_guard_pass", row.get("stage3_guard_pass", True)))
    ]
    payload = {
        "metrics": metrics,
        "input_count": len(entries),
        "loaded_count": len(existing_entries),
        "missing_summaries": missing_paths,
        "reference_curves": [
            {
                "name": curve.get("name", ""),
                "dataset": curve.get("dataset", ""),
                "bpp_policy": curve.get("bpp_policy", ""),
                "source_path": curve.get("source_path", ""),
            }
            for curve in reference_curves
        ],
        "anchor": anchor
        if anchor is None
        else {
            "label": anchor.get("label", ""),
            "summary_path": anchor.get("summary_path", ""),
            "actual_payload_bpp": anchor.get("actual_payload_bpp", None),
            "count": anchor.get("count", None),
            "psnr": anchor.get("psnr", None),
            "ms_ssim": anchor.get("ms_ssim", None),
            "lpips": anchor.get("lpips", None),
            "dists": anchor.get("dists", None),
        },
        "anchor_count_warnings": anchor_count_warnings,
        "selection_policy": {
            "rate": "actual_payload_bpp / paper_bpp only",
            "pareto_metrics": metrics,
            "external_reference_note": (
                "reference gaps are same-bpp interpolation / single-point proxy only; "
                "they are not BD-rate and must not be reported as BD-rate"
            ),
            "stage3_guard": {
                "max_psnr_drop": args.max_psnr_drop,
                "max_ms_ssim_drop": args.max_ms_ssim_drop,
                "max_lpips_increase": args.max_lpips_increase,
                "max_dists_increase": args.max_dists_increase,
            },
            "basis_guard": {
                "min_basis_retained_energy": args.min_basis_retained_energy,
                "max_basis_residual_energy": args.max_basis_residual_energy,
                "note": "applies only when basis_reconstruction_stats are present in the sweep plan",
            },
            "anchor_guard": {
                "max_psnr_drop": args.max_anchor_psnr_drop,
                "max_ms_ssim_drop": args.max_anchor_ms_ssim_drop,
                "max_lpips_increase": args.max_anchor_lpips_increase,
                "max_dists_increase": args.max_anchor_dists_increase,
                "note": "applies only when --anchor-summary or --anchor-label is provided",
            },
        },
        "candidates": ranked,
        "recommended": recommended_rows[: max(args.top_k, 0)],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
