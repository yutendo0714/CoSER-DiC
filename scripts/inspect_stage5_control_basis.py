from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coserdic.entropy import (
    CoSERBitstream,
    PrefixTopKControlBasisCode,
    SparseControlBasisCode,
    build_control_grid_code,
    control_quantizer_label,
)


def load_basis_summary(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    basis = payload.get("basis")
    if not isinstance(basis, torch.Tensor) or basis.ndim != 4:
        raise ValueError("basis checkpoint must contain basis tensor with shape KGSS")
    groups = int(payload.get("groups", basis.shape[1]))
    grid_size = int(payload.get("grid_size", basis.shape[-1]))
    if tuple(basis.shape[1:]) != (groups, grid_size, grid_size):
        raise ValueError("basis metadata does not match basis tensor shape")
    quantiles = payload.get("coefficient_abs_quantiles", {})
    if not isinstance(quantiles, dict):
        raise ValueError("coefficient_abs_quantiles must be a dictionary when present")
    return {
        "path": str(path),
        "basis_count": int(basis.shape[0]),
        "groups": groups,
        "grid_size": grid_size,
        "source": str(payload.get("source", "")),
        "explained_variance": payload.get("explained_variance", []),
        "cumulative_explained_variance": payload.get("cumulative_explained_variance", []),
        "coefficient_abs_quantiles": quantiles,
        "coefficient_component_abs_p95": payload.get("coefficient_component_abs_p95", []),
        "coefficient_component_abs_p99": payload.get("coefficient_component_abs_p99", []),
        "coefficient_abs_mean": float(payload.get("coefficient_abs_mean", 0.0)),
        "coefficient_abs_std": float(payload.get("coefficient_abs_std", 0.0)),
        "coefficient_abs_max": float(payload.get("coefficient_abs_max", 0.0)),
        "control_huffman_priors": payload.get("control_huffman_priors", {}),
        "coefficient_component_codebooks": payload.get("coefficient_component_codebooks", {}),
        "coefficient_vector_codebooks": payload.get("coefficient_vector_codebooks", {}),
        "sparse_topk_control_priors": payload.get("sparse_topk_control_priors", {}),
        "basis_reconstruction_stats": payload.get("basis_reconstruction_stats", {}),
    }


def component_codebook_prior(summary: dict[str, object], *, bits: int) -> dict[str, object]:
    priors = summary.get("coefficient_component_codebooks", {})
    if not isinstance(priors, dict) or not priors:
        raise ValueError("basis summary does not contain coefficient_component_codebooks")
    key = f"lloyd_b{int(bits)}"
    prior = priors.get(key)
    if isinstance(prior, dict):
        return prior
    matches = [
        candidate
        for candidate in priors.values()
        if isinstance(candidate, dict) and int(candidate.get("bits", -1)) == int(bits)
    ]
    if len(matches) != 1:
        raise ValueError(f"basis summary does not contain a unique component codebook for bits={bits}")
    return matches[0]


def vector_codebook_prior(summary: dict[str, object], *, components: int, bits: int) -> tuple[str, dict[str, object]]:
    priors = summary.get("coefficient_vector_codebooks", {})
    if not isinstance(priors, dict) or not priors:
        raise ValueError("basis summary does not contain coefficient_vector_codebooks")
    key = f"vq_k{int(components)}_b{int(bits)}"
    prior = priors.get(key)
    if not isinstance(prior, dict):
        raise ValueError(f"basis summary does not contain vector codebook {key}")
    return key, prior


def build_basis_setting_rows(
    summary: dict[str, object],
    *,
    components: list[int],
    candidate_components: list[int],
    selections: list[str],
    quantiles: list[str],
    bits: int,
    scale: float,
    height: int,
    width: int,
    codec: str = "fixed_bits",
    quantizers: list[str] | tuple[str, ...] = ("uniform",),
    mu: float = 16.0,
    range_modes: list[str] | tuple[str, ...] = ("global",),
) -> list[dict[str, object]]:
    groups = int(summary["groups"])
    grid_size = int(summary["grid_size"])
    basis_count = int(summary["basis_count"])
    coefficient_abs_quantiles = summary.get("coefficient_abs_quantiles", {})
    if not isinstance(coefficient_abs_quantiles, dict):
        raise ValueError("coefficient_abs_quantiles must be a dictionary")
    rows: list[dict[str, object]] = []
    huffman_priors = summary.get("control_huffman_priors", {})
    if not isinstance(huffman_priors, dict):
        raise ValueError("control_huffman_priors must be a dictionary when present")
    sparse_topk_priors = summary.get("sparse_topk_control_priors", {})
    if not isinstance(sparse_topk_priors, dict):
        raise ValueError("sparse_topk_control_priors must be a dictionary when present")
    basis_reconstruction_stats = summary.get("basis_reconstruction_stats", {})
    if not isinstance(basis_reconstruction_stats, dict):
        raise ValueError("basis_reconstruction_stats must be a dictionary when present")
    codecs = ("fixed_bits", "huffman") if codec == "both" else (codec,)
    for codec_name in codecs:
        if codec_name not in {"fixed_bits", "huffman"}:
            raise ValueError("codec must be fixed_bits, huffman, or both")
    if mu <= 0:
        raise ValueError("mu must be positive")
    for quantizer in quantizers:
        if quantizer not in {"uniform", "mu_law"}:
            raise ValueError("quantizers must contain only uniform or mu_law")
    for range_mode in range_modes:
        if range_mode not in {"global", "component_p95", "component_p99", "component_codebook"}:
            raise ValueError("range_modes must contain only global, component_p95, component_p99, or component_codebook")
    for selection in selections:
        if selection not in {"prefix", "topk", "vector", "prefix_topk"}:
            raise ValueError("selections must contain only prefix, topk, vector, or prefix_topk")
    for quantile in quantiles:
        if quantile not in coefficient_abs_quantiles:
            raise ValueError(f"basis summary does not contain coefficient quantile {quantile!r}")
        value_range = float(coefficient_abs_quantiles[quantile])
        if value_range <= 0:
            raise ValueError(f"coefficient quantile {quantile!r} must be positive")
        for component_count in components:
            if component_count <= 0 or component_count > basis_count:
                raise ValueError("component count must be in [1, basis_count]")
            candidate_counts = [component_count]
            if "topk" in selections or "prefix_topk" in selections:
                candidate_counts = sorted(
                    {
                        component_count,
                        *[
                            value
                            for value in candidate_components
                            if value >= component_count and value <= basis_count
                        ],
                    }
                )
            for quantizer in quantizers:
                quantizer_label = control_quantizer_label(quantizer, mu)
                prior_key = f"{quantile}_b{bits}" if quantizer == "uniform" else f"{quantile}_b{bits}_{quantizer_label}"
                quantizer_prior = huffman_priors.get(prior_key, {})
                if isinstance(quantizer_prior, dict):
                    quantization_mae = float(quantizer_prior.get("quantization_mae", 0.0))
                    quantization_rmse = float(quantizer_prior.get("quantization_rmse", 0.0))
                    clipped_fraction = float(quantizer_prior.get("clipped_fraction", 0.0))
                else:
                    quantization_mae = 0.0
                    quantization_rmse = 0.0
                    clipped_fraction = 0.0
                for selection in selections:
                    for candidate_count in candidate_counts:
                        if selection == "prefix" and candidate_count != component_count:
                            continue
                        if selection == "vector" and candidate_count != component_count:
                            continue
                        if selection == "prefix_topk" and candidate_count <= component_count:
                            continue
                        for codec_name in codecs:
                            mode_candidates = tuple(range_modes)
                            if codec_name != "fixed_bits":
                                mode_candidates = tuple(
                                    mode for mode in range_modes if mode in {"global", "component_codebook"}
                                )
                            for range_mode in mode_candidates:
                                if selection == "vector" and (quantizer != "uniform" or range_mode != "global"):
                                    continue
                                if selection == "prefix_topk" and codec_name != "fixed_bits":
                                    continue
                                if range_mode == "component_codebook" and quantizer != "uniform":
                                    continue
                                huffman_key = ""
                                row_quantization_mae = quantization_mae
                                row_quantization_rmse = quantization_rmse
                                row_clipped_fraction = clipped_fraction
                                vector_prior: dict[str, object] | None = None
                                if selection == "vector":
                                    try:
                                        huffman_key, vector_prior = vector_codebook_prior(
                                            summary,
                                            components=component_count,
                                            bits=bits,
                                        )
                                    except ValueError:
                                        continue
                                    row_quantization_mae = float(vector_prior.get("quantization_mae", 0.0))
                                    row_quantization_rmse = float(vector_prior.get("quantization_rmse", 0.0))
                                    row_clipped_fraction = 0.0
                                if range_mode == "component_codebook":
                                    codebook_prior = component_codebook_prior(summary, bits=bits)
                                    row_quantization_mae = float(codebook_prior.get("quantization_mae", 0.0))
                                    row_quantization_rmse = float(codebook_prior.get("quantization_rmse", 0.0))
                                    row_clipped_fraction = 0.0
                                if codec_name == "fixed_bits":
                                    if selection == "vector":
                                        if vector_prior is None:
                                            raise RuntimeError("vector prior was not loaded")
                                        control_bytes = float(vector_prior["fixed_bytes_per_image"])
                                    else:
                                        control_codec = build_control_grid_code(
                                            quantizer=quantizer,
                                            bits=bits,
                                            value_range=value_range,
                                            mu=mu,
                                        )
                                    if selection == "topk":
                                        control_bytes = float(
                                            SparseControlBasisCode(
                                                candidate_components=candidate_count,
                                                selected_components=component_count,
                                                value_codec=control_codec,
                                            ).encoded_compact_num_bytes()
                                        )
                                    elif selection == "prefix_topk":
                                        control_bytes = float(
                                            PrefixTopKControlBasisCode(
                                                candidate_components=candidate_count,
                                                prefix_components=component_count,
                                                selected_components=component_count,
                                                value_codec=control_codec,
                                            ).encoded_compact_num_bytes()
                                        )
                                    elif selection == "prefix":
                                        value_bytes = control_codec.encoded_num_bytes((component_count,))
                                        control_bytes = float(value_bytes)
                                else:
                                    if selection == "vector":
                                        if vector_prior is None:
                                            raise RuntimeError("vector prior was not loaded")
                                        prior = vector_prior
                                        control_bytes = float(prior["huffman_mean_payload_bytes"])
                                    elif selection == "topk":
                                        huffman_key_base = f"topk_c{candidate_count}_k{component_count}_{quantile}_b{bits}"
                                        if range_mode == "component_codebook":
                                            huffman_key_base = f"{huffman_key_base}_codebook"
                                        huffman_key = (
                                            huffman_key_base
                                            if quantizer == "uniform"
                                            else f"{huffman_key_base}_{quantizer_label}"
                                        )
                                        prior = sparse_topk_priors.get(huffman_key)
                                    else:
                                        huffman_key = (
                                            f"{quantile}_b{bits}_codebook"
                                            if range_mode == "component_codebook"
                                            else prior_key
                                        )
                                        prior = huffman_priors.get(huffman_key)
                                    if selection != "vector":
                                        if not isinstance(prior, dict):
                                            continue
                                        row_quantization_mae = float(prior.get("quantization_mae", row_quantization_mae))
                                        row_quantization_rmse = float(prior.get("quantization_rmse", row_quantization_rmse))
                                        row_clipped_fraction = float(prior.get("clipped_fraction", row_clipped_fraction))
                                        if selection == "prefix":
                                            prefix_means = prior.get("prefix_mean_payload_bytes", [])
                                        else:
                                            prefix_means = []
                                        if (
                                            selection == "prefix"
                                            and isinstance(prefix_means, list)
                                            and len(prefix_means) >= component_count
                                        ):
                                            control_bytes = float(prefix_means[component_count - 1])
                                        else:
                                            control_bytes = float(prior["mean_payload_bytes"])
                                setting = (
                                    f"mode=basis,groups={groups},grid={grid_size},coeffs={component_count},"
                                    f"bits={bits},range={value_range:.6g},scale={scale:g},codec={codec_name},"
                                    f"quantizer={quantizer},mu={mu:g},selection={selection}"
                                )
                                if selection == "topk":
                                    setting += f",candidates={candidate_count}"
                                if selection == "prefix_topk":
                                    setting += f",prefix={component_count},candidates={candidate_count}"
                                if huffman_key:
                                    setting += f",huffman_key={huffman_key}"
                                if range_mode != "global":
                                    setting += f",basis_range_mode={range_mode}"
                                stats_key = (
                                    f"prefix_k{component_count}"
                                    if selection == "prefix"
                                    else f"topk_c{candidate_count}_k{component_count}"
                                    if selection == "topk"
                                    else f"prefixtopk_p{component_count}_c{candidate_count}_k{component_count}"
                                    if selection == "prefix_topk"
                                    else f"prefix_k{component_count}"
                                )
                                reconstruction_stats = basis_reconstruction_stats.get(stats_key, {})
                                if not isinstance(reconstruction_stats, dict):
                                    reconstruction_stats = {}
                                rows.append(
                                    {
                                        "quantile": quantile,
                                        "components": component_count,
                                        "candidate_components": candidate_count,
                                        "selection": selection,
                                        "codec": codec_name,
                                        "huffman_key": huffman_key,
                                        "quantizer": quantizer,
                                        "quantizer_label": quantizer_label,
                                        "basis_range_mode": range_mode,
                                        "mu": mu,
                                        "bits": bits,
                                        "range": value_range,
                                        "scale": scale,
                                        "control_bytes": control_bytes,
                                        "control_bpp": CoSERBitstream.bytes_to_bpp(control_bytes, height, width),
                                        "quantization_mae": row_quantization_mae,
                                        "quantization_rmse": row_quantization_rmse,
                                        "clipped_fraction": row_clipped_fraction,
                                        "basis_reconstruction_key": stats_key,
                                        "basis_retained_energy_fraction_mean": float(
                                            reconstruction_stats.get("retained_energy_fraction_mean", 0.0)
                                        ),
                                        "basis_retained_energy_fraction_p50": float(
                                            reconstruction_stats.get("retained_energy_fraction_p50", 0.0)
                                        ),
                                        "basis_residual_energy_fraction_mean": float(
                                            reconstruction_stats.get("residual_energy_fraction_mean", 0.0)
                                        ),
                                        "basis_residual_l2_mean": float(
                                            reconstruction_stats.get("residual_l2_mean", 0.0)
                                        ),
                                        "control_bytes_note": "mean train payload bytes"
                                        if codec_name == "huffman"
                                        else "exact fixed payload bytes",
                                        "setting": setting,
                                    }
                                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", required=True)
    parser.add_argument("--components", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--candidate-components", nargs="+", type=int, default=[])
    parser.add_argument("--selection", action="append", choices=("prefix", "topk", "vector", "prefix_topk"), default=[])
    parser.add_argument("--quantile", action="append", default=[])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--codec", choices=("fixed_bits", "huffman", "both"), default="both")
    parser.add_argument("--quantizer", action="append", default=[])
    parser.add_argument("--mu", type=float, default=16.0)
    parser.add_argument(
        "--range-mode",
        action="append",
        choices=("global", "component_p95", "component_p99", "component_codebook"),
        default=[],
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    summary = load_basis_summary(Path(args.basis))
    rows = build_basis_setting_rows(
        summary,
        components=args.components,
        candidate_components=args.candidate_components,
        selections=args.selection or ["prefix"],
        quantiles=args.quantile or ["p95", "p99"],
        bits=args.bits,
        scale=args.scale,
        height=args.height,
        width=args.width,
        codec=args.codec,
        quantizers=args.quantizer or ["uniform", "mu_law"],
        mu=args.mu,
        range_modes=args.range_mode or ["global"],
    )
    payload = {"basis": summary, "settings": rows}
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
