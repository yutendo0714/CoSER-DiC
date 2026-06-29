from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_RATE_KEYS = (
    "actual_payload_bpp_mean",
    "paper_bpp_mean",
    "actual_payload_bpp",
    "paper_bpp",
    "bpp",
    "rate",
)

SUMMARY_METRIC_KEYS = {
    "psnr": ("stage4_psnr_mean", "psnr_mean", "psnr"),
    "ms_ssim": ("stage4_ms_ssim_mean", "ms_ssim_mean", "ms_ssim"),
    "lpips": ("stage4_lpips_alex_mean", "lpips_alex_mean", "lpips"),
    "lpips_alex": ("stage4_lpips_alex_mean", "lpips_alex_mean", "lpips"),
    "dists": ("stage4_dists_mean", "dists_mean", "dists"),
    "fid": ("fid", "patch_fid", "patch_fid_mean", "fid_mean"),
    "kid": ("kid", "kid_mean"),
}


@dataclass(frozen=True)
class RateMetricPoint:
    rate: float
    metric: float
    label: str = ""


@dataclass(frozen=True)
class BdRateResult:
    metric: str
    reference_name: str
    candidate_name: str
    reference_points: int
    candidate_points: int
    overlap_min: float | None
    overlap_max: float | None
    bd_rate_percent: float | None
    status: str

    def to_json(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "reference_name": self.reference_name,
            "candidate_name": self.candidate_name,
            "reference_points": self.reference_points,
            "candidate_points": self.candidate_points,
            "overlap_min": self.overlap_min,
            "overlap_max": self.overlap_max,
            "bd_rate_percent": self.bd_rate_percent,
            "status": self.status,
        }


def _as_float(value: object, *, key: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite, got {number!r}")
    return number


def first_present_float(row: dict[str, object], keys: Iterable[str], *, kind: str) -> float | None:
    for key in keys:
        if key in row and row[key] is not None:
            return _as_float(row[key], key=f"{kind}.{key}")
    return None


def metric_keys(metric: str, explicit_key: str | None = None) -> tuple[str, ...]:
    if explicit_key:
        return (explicit_key,)
    normalized = metric.lower().replace("-", "_")
    if normalized in SUMMARY_METRIC_KEYS:
        return SUMMARY_METRIC_KEYS[normalized]
    return (f"stage4_{normalized}_mean", f"{normalized}_mean", normalized)


def normalize_points(points: Iterable[RateMetricPoint]) -> list[RateMetricPoint]:
    grouped: dict[float, list[RateMetricPoint]] = {}
    for point in points:
        if point.rate <= 0:
            raise ValueError(f"rate must be positive, got {point.rate}")
        grouped.setdefault(point.metric, []).append(point)
    merged = [
        RateMetricPoint(
            rate=sum(item.rate for item in items) / len(items),
            metric=metric,
            label=";".join(item.label for item in items if item.label),
        )
        for metric, items in grouped.items()
    ]
    return sorted(merged, key=lambda item: item.metric)


def read_json_or_jsonl(path: Path) -> object:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def curve_metadata(path: Path) -> dict[str, object]:
    payload = read_json_or_jsonl(path)
    if not isinstance(payload, dict):
        return {}
    keys = ("name", "dataset", "stage", "dataset_key", "bpp_policy", "source")
    return {key: payload[key] for key in keys if key in payload}


def _normalized_text(value: object) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def protocol_warnings(
    reference_metadata: dict[str, object],
    candidate_metadata: dict[str, object],
) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    reference_dataset = reference_metadata.get("dataset")
    candidate_dataset = candidate_metadata.get("dataset")
    if reference_dataset and candidate_dataset and _normalized_text(reference_dataset) != _normalized_text(candidate_dataset):
        warnings.append(
            {
                "kind": "dataset_mismatch",
                "reference": reference_dataset,
                "candidate": candidate_dataset,
                "message": "Reference and candidate datasets differ; treat BD-rate as diagnostic only.",
            }
        )
    reference_dataset_key = reference_metadata.get("dataset_key")
    candidate_dataset_key = candidate_metadata.get("dataset_key")
    if (
        reference_dataset_key
        and candidate_dataset_key
        and _normalized_text(reference_dataset_key) != _normalized_text(candidate_dataset_key)
    ):
        warnings.append(
            {
                "kind": "dataset_key_mismatch",
                "reference": reference_dataset_key,
                "candidate": candidate_dataset_key,
                "message": "Reference and candidate dataset split keys differ.",
            }
        )
    reference_bpp_policy = reference_metadata.get("bpp_policy")
    candidate_bpp_policy = candidate_metadata.get("bpp_policy")
    if (
        reference_bpp_policy
        and candidate_bpp_policy
        and _normalized_text(reference_bpp_policy) != _normalized_text(candidate_bpp_policy)
    ):
        warnings.append(
            {
                "kind": "bpp_policy_mismatch",
                "reference": reference_bpp_policy,
                "candidate": candidate_bpp_policy,
                "message": "Reference and candidate bpp policies differ; ensure rates are comparable.",
            }
        )
    return warnings


def iter_rows(payload: object) -> Iterable[dict[str, object]]:
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError("point lists must contain JSON objects")
            yield row
        return
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object, an object with points, or a list")
    if "points" in payload:
        points = payload["points"]
        if not isinstance(points, list):
            raise ValueError("points must be a list")
        for row in points:
            if not isinstance(row, dict):
                raise ValueError("points must contain JSON objects")
            yield row
        return
    yield payload


def load_points(
    path: Path,
    *,
    metric: str,
    metric_key: str | None = None,
    rate_key: str | None = None,
) -> list[RateMetricPoint]:
    payload = read_json_or_jsonl(path)
    rate_keys = (rate_key,) if rate_key else DEFAULT_RATE_KEYS
    keys = metric_keys(metric, metric_key)
    points: list[RateMetricPoint] = []
    for index, row in enumerate(iter_rows(payload)):
        rate = first_present_float(row, rate_keys, kind=f"{path}:{index}")
        value = first_present_float(row, keys, kind=f"{path}:{index}")
        if rate is None or value is None:
            continue
        label = str(row.get("label", row.get("run_name", path.name)))
        points.append(RateMetricPoint(rate=rate, metric=value, label=label))
    return normalize_points(points)


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for index in range(len(xs) - 1):
        x0 = xs[index]
        x1 = xs[index + 1]
        if x0 <= x <= x1:
            if x1 == x0:
                return ys[index]
            t = (x - x0) / (x1 - x0)
            return ys[index] * (1.0 - t) + ys[index + 1] * t
    return ys[-1]


def _integrate_piecewise_log_rate(points: list[RateMetricPoint], low: float, high: float) -> float:
    xs = [point.metric for point in points]
    ys = [math.log(point.rate) for point in points]
    knots = [low, high]
    knots.extend(x for x in xs if low < x < high)
    knots = sorted(set(knots))
    area = 0.0
    for x0, x1 in zip(knots, knots[1:]):
        y0 = _interp(xs, ys, x0)
        y1 = _interp(xs, ys, x1)
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return area


def compute_bd_rate(
    reference: list[RateMetricPoint],
    candidate: list[RateMetricPoint],
    *,
    metric: str,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> BdRateResult:
    reference = normalize_points(reference)
    candidate = normalize_points(candidate)
    if len(reference) < 2 or len(candidate) < 2:
        return BdRateResult(
            metric=metric,
            reference_name=reference_name,
            candidate_name=candidate_name,
            reference_points=len(reference),
            candidate_points=len(candidate),
            overlap_min=None,
            overlap_max=None,
            bd_rate_percent=None,
            status="insufficient_points",
        )
    low = max(reference[0].metric, candidate[0].metric)
    high = min(reference[-1].metric, candidate[-1].metric)
    if not low < high:
        return BdRateResult(
            metric=metric,
            reference_name=reference_name,
            candidate_name=candidate_name,
            reference_points=len(reference),
            candidate_points=len(candidate),
            overlap_min=low,
            overlap_max=high,
            bd_rate_percent=None,
            status="no_metric_overlap",
        )
    reference_area = _integrate_piecewise_log_rate(reference, low, high)
    candidate_area = _integrate_piecewise_log_rate(candidate, low, high)
    avg_delta = (candidate_area - reference_area) / (high - low)
    bd_rate_percent = (math.exp(avg_delta) - 1.0) * 100.0
    return BdRateResult(
        metric=metric,
        reference_name=reference_name,
        candidate_name=candidate_name,
        reference_points=len(reference),
        candidate_points=len(candidate),
        overlap_min=low,
        overlap_max=high,
        bd_rate_percent=bd_rate_percent,
        status="ok",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Bjontegaard delta-rate from actual bpp/metric points. "
            "The implementation integrates log(rate) over the overlapping metric range "
            "using piecewise-linear interpolation and refuses one-point curves."
        )
    )
    parser.add_argument("--reference", required=True, help="Reference curve JSON/JSONL.")
    parser.add_argument("--candidate", required=True, help="Candidate curve JSON/JSONL.")
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--metric", action="append", required=True, help="Metric to compare. Repeatable.")
    parser.add_argument("--reference-rate-key", default="")
    parser.add_argument("--candidate-rate-key", default="")
    parser.add_argument("--reference-metric-key", default="")
    parser.add_argument("--candidate-metric-key", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--fail-on-unavailable", action="store_true")
    parser.add_argument(
        "--fail-on-dataset-mismatch",
        action="store_true",
        help="Exit non-zero when reference/candidate curve metadata have different dataset names.",
    )
    parser.add_argument(
        "--max-bd-rate-percent",
        type=float,
        default=None,
        help=(
            "Optional promotion threshold. For example, pass -10 to require every requested "
            "metric to achieve <= -10 percent BD-rate. Unavailable metrics fail when this is set."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference)
    candidate_path = Path(args.candidate)
    reference_metadata = curve_metadata(reference_path)
    candidate_metadata = curve_metadata(candidate_path)
    warnings = protocol_warnings(reference_metadata, candidate_metadata)
    results = []
    for metric in args.metric:
        reference_points = load_points(
            reference_path,
            metric=metric,
            rate_key=args.reference_rate_key or None,
            metric_key=args.reference_metric_key or None,
        )
        candidate_points = load_points(
            candidate_path,
            metric=metric,
            rate_key=args.candidate_rate_key or None,
            metric_key=args.candidate_metric_key or None,
        )
        result = compute_bd_rate(
            reference_points,
            candidate_points,
            metric=metric,
            reference_name=args.reference_name,
            candidate_name=args.candidate_name,
        )
        results.append(result)

    threshold = args.max_bd_rate_percent
    promotion_passed = None
    if threshold is not None:
        promotion_passed = all(
            result.status == "ok"
            and result.bd_rate_percent is not None
            and result.bd_rate_percent <= threshold
            for result in results
        )

    payload = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "curve_metadata": {
            "reference": reference_metadata,
            "candidate": candidate_metadata,
        },
        "protocol": {
            "warnings": warnings,
        },
        "results": [result.to_json() for result in results],
        "rate_policy": "Use actual_payload_bpp / paper_bpp for CoSER-DiC paper comparisons.",
        "integration": "piecewise_linear_log_rate_over_metric_overlap",
        "promotion": {
            "max_bd_rate_percent": threshold,
            "passed": promotion_passed,
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text)
    print(text)

    if (args.fail_on_unavailable or threshold is not None) and any(
        result.status != "ok" for result in results
    ):
        sys.exit(2)
    if threshold is not None and not promotion_passed:
        sys.exit(3)
    if args.fail_on_dataset_mismatch and any(warning["kind"] == "dataset_mismatch" for warning in warnings):
        sys.exit(4)


if __name__ == "__main__":
    main()
