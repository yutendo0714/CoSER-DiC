from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coserdic.entropy import (
    ComponentCodebookControlCode,
    StaticControlHuffmanCode,
    VectorCodebookControlCode,
    build_control_grid_code,
    control_quantizer_label,
)
from coserdic.entropy import semantic_bits_per_token
from coserdic.models import (
    CoDLiteOneStepBackbone,
    CoDLiteOneStepBackboneConfig,
    apply_lora_adapters_from_config,
    load_named_parameter_state,
)
from coserdic.utils import seed_everything
from scripts.eval_stage4_cod_lite_adapter import (
    Stage4ManifestDataset,
    apply_condition_residual,
    build_adapter_from_payload,
    channel_group_sizes,
    detail_context_channels,
    grouped_condition_affine_control,
    write_run_doc,
)


def normalize_quantile_label(value: str) -> str:
    value = str(value).strip()
    if value.startswith("p"):
        return value
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid quantile label {value!r}; use p99 or 0.99 style") from exc
    if 0.0 < number <= 1.0:
        return f"p{int(round(number * 100)):02d}"
    if 1.0 < number <= 100.0:
        return f"p{int(round(number)):02d}"
    raise ValueError(f"invalid quantile label {value!r}; expected 0 < q <= 1 or 0 < q <= 100")


def grouped_condition_grid(condition_error: torch.Tensor, *, groups: int, grid_size: int) -> torch.Tensor:
    if condition_error.ndim != 4:
        raise ValueError("condition_error must be BCHW")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    _, channels, _, _ = condition_error.shape
    sizes = channel_group_sizes(int(channels), int(groups))
    grouped_maps: list[torch.Tensor] = []
    start = 0
    for size in sizes:
        end = start + size
        grouped_maps.append(condition_error[:, start:end].float().mean(dim=1, keepdim=True))
        start = end
    return F.adaptive_avg_pool2d(torch.cat(grouped_maps, dim=1), output_size=(grid_size, grid_size))


def fit_control_basis_from_grids(
    grids: torch.Tensor,
    *,
    components: int,
    center: bool = True,
) -> dict[str, torch.Tensor | list[float]]:
    if grids.ndim != 4:
        raise ValueError("grids must be NGSS")
    if components <= 0:
        raise ValueError("components must be positive")
    sample_count = int(grids.shape[0])
    feature_count = int(grids[0].numel())
    if sample_count < 2:
        raise ValueError("need at least two samples to fit a basis")
    if components > min(sample_count, feature_count):
        raise ValueError("components exceeds rank upper bound")
    x = grids.float().flatten(1)
    mean_flat = x.mean(dim=0) if center else torch.zeros(feature_count, dtype=x.dtype)
    centered = x - mean_flat.unsqueeze(0)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:components].reshape(components, *grids.shape[1:]).contiguous()
    variance = singular_values.square() / max(sample_count - 1, 1)
    total_variance = float(variance.sum().item())
    if total_variance <= 0:
        explained = [0.0 for _ in range(components)]
        cumulative = [0.0 for _ in range(components)]
    else:
        explained_tensor = variance[:components] / total_variance
        cumulative_tensor = torch.cumsum(explained_tensor, dim=0)
        explained = [float(v) for v in explained_tensor]
        cumulative = [float(v) for v in cumulative_tensor]
    coefficients = torch.matmul(centered, vh[:components].t())
    coeff_abs = coefficients.abs()
    quantile_points = (0.50, 0.75, 0.90, 0.95, 0.99)
    coefficient_abs_quantiles = {
        f"p{int(q * 100):02d}": float(torch.quantile(coeff_abs.flatten(), q).item()) for q in quantile_points
    }
    coefficient_component_abs_p95 = [
        float(value)
        for value in torch.quantile(coeff_abs, 0.95, dim=0)
    ]
    coefficient_component_abs_p99 = [
        float(value)
        for value in torch.quantile(coeff_abs, 0.99, dim=0)
    ]
    return {
        "basis": basis.cpu(),
        "mean": mean_flat.reshape(*grids.shape[1:]).cpu(),
        "singular_values": singular_values[:components].cpu(),
        "explained_variance": explained,
        "cumulative_explained_variance": cumulative,
        "coefficient_abs_mean": float(coeff_abs.mean().item()),
        "coefficient_abs_std": float(coeff_abs.std(unbiased=False).item()),
        "coefficient_abs_max": float(coeff_abs.max().item()),
        "coefficient_abs_quantiles": coefficient_abs_quantiles,
        "coefficient_component_abs_p95": coefficient_component_abs_p95,
        "coefficient_component_abs_p99": coefficient_component_abs_p99,
        "coefficients": coefficients.cpu(),
    }


def fit_control_huffman_priors(
    coefficients: torch.Tensor,
    *,
    bits: int,
    ranges: dict[str, float],
    quantizers: list[str] | tuple[str, ...] = ("uniform",),
    mu: float = 16.0,
    smoothing_count: int = 1,
) -> dict[str, dict[str, object]]:
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [N, K]")
    if bits <= 0:
        raise ValueError("bits must be positive")
    if mu <= 0:
        raise ValueError("mu must be positive")
    if smoothing_count < 0:
        raise ValueError("smoothing_count must be non-negative")
    coeff = coefficients.detach().cpu().float()
    priors: dict[str, dict[str, object]] = {}
    for quantizer in quantizers:
        if quantizer not in {"uniform", "mu_law"}:
            raise ValueError("quantizers must contain only uniform or mu_law")
        quantizer_label = control_quantizer_label(quantizer, mu)
        for range_name, value_range in ranges.items():
            value_range_f = float(value_range)
            if value_range_f <= 0:
                raise ValueError(f"range {range_name} must be positive")
            codec = build_control_grid_code(quantizer=quantizer, bits=bits, value_range=value_range_f, mu=mu)
            codes = codec.quantize(coeff)
            decoded = codec.dequantize(codes)
            quantization_error = decoded - coeff
            clipped = coeff.abs() > value_range_f
            counts = torch.zeros(codes.shape[1], codec.levels, dtype=torch.long)
            ones = torch.ones(codes.shape[0], dtype=torch.long)
            for index in range(codes.shape[1]):
                counts[index].scatter_add_(0, codes[:, index].to(torch.long), ones)
            huffman = StaticControlHuffmanCode.from_counts(counts, smoothing_count=smoothing_count)
            length_columns = []
            for index, codebook in enumerate(huffman.position_codes):
                lengths = torch.tensor(codebook.code_lengths, dtype=torch.long)
                length_columns.append(lengths[codes[:, index].to(torch.long)])
            symbol_bits = torch.stack(length_columns, dim=1)
            cumulative_bits = torch.cumsum(symbol_bits, dim=1)
            prefix_payload_bytes = torch.div(cumulative_bits + 7, 8, rounding_mode="floor")
            payload_lengths = prefix_payload_bytes[:, -1]
            fixed_bytes = codec.encoded_num_bytes((codes.shape[1],))
            key = f"{range_name}_b{bits}" if quantizer == "uniform" else f"{range_name}_b{bits}_{quantizer_label}"
            priors[key] = {
                "codec": "static_control_huffman_prior",
                "version": 0,
                "quantile": range_name,
                "quantizer": quantizer,
                "quantizer_label": quantizer_label,
                "mu": float(mu),
                "bits": bits,
                "range": value_range_f,
                "smoothing_count": smoothing_count,
                "fixed_bytes_per_image": fixed_bytes,
                "quantization_mae": float(quantization_error.abs().mean().item()),
                "quantization_rmse": float(torch.sqrt(quantization_error.pow(2).mean()).item()),
                "clipped_fraction": float(clipped.float().mean().item()),
                "mean_payload_bytes": float(payload_lengths.float().mean().item()),
                "min_payload_bytes": int(payload_lengths.min().item()),
                "max_payload_bytes": int(payload_lengths.max().item()),
                "mean_bits_per_symbol": float(symbol_bits.float().mean().item()),
                "prefix_mean_payload_bytes": [
                    float(value) for value in prefix_payload_bytes.float().mean(dim=0).tolist()
                ],
                "prefix_min_payload_bytes": [int(value) for value in prefix_payload_bytes.min(dim=0).values.tolist()],
                "prefix_max_payload_bytes": [int(value) for value in prefix_payload_bytes.max(dim=0).values.tolist()],
                "huffman": huffman.to_dict(),
            }
    return priors


def _fit_scalar_lloyd_codebook(values: torch.Tensor, *, levels: int, max_iter: int) -> torch.Tensor:
    if levels <= 1:
        raise ValueError("levels must be > 1")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    flat = values.detach().cpu().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return torch.zeros(levels, dtype=torch.float32)
    if flat.numel() == 1:
        return flat.repeat(levels).to(torch.float32)
    quantiles = torch.linspace(0.0, 1.0, steps=levels)
    centroids = torch.quantile(flat, quantiles).to(torch.float32)
    for _ in range(max_iter):
        distances = (flat.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignment = torch.argmin(distances, dim=1)
        updated = centroids.clone()
        for level in range(levels):
            mask = assignment == level
            if bool(mask.any()):
                updated[level] = flat[mask].mean()
        updated = torch.sort(updated).values
        if torch.allclose(updated, centroids, atol=1.0e-6, rtol=0.0):
            centroids = updated
            break
        centroids = updated
    return centroids.to(torch.float32)


def fit_component_codebook_priors(
    coefficients: torch.Tensor,
    *,
    bits_list: list[int] | tuple[int, ...],
    max_iter: int = 50,
) -> dict[str, dict[str, object]]:
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [N, K]")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    coeff = coefficients.detach().cpu().float()
    priors: dict[str, dict[str, object]] = {}
    for bits_raw in bits_list:
        bits = int(bits_raw)
        if bits <= 0:
            raise ValueError("bits entries must be positive")
        if bits > 16:
            raise ValueError("bits entries must be <= 16")
        levels = 1 << bits
        codebooks = torch.stack(
            [
                _fit_scalar_lloyd_codebook(coeff[:, index], levels=levels, max_iter=max_iter)
                for index in range(int(coeff.shape[1]))
            ],
            dim=0,
        )
        codec = ComponentCodebookControlCode(
            bits=bits,
            codebooks=tuple(tuple(float(v) for v in row.tolist()) for row in codebooks),
        )
        codes = codec.quantize(coeff)
        decoded = codec.dequantize(codes)
        quantization_error = decoded - coeff
        key = f"lloyd_b{bits}"
        priors[key] = {
            "codec": "component_codebook_control_fixed_bits",
            "version": 0,
            "method": "lloyd_1d_per_component",
            "bits": bits,
            "levels": levels,
            "max_iter": int(max_iter),
            "component_count": int(coeff.shape[1]),
            "fixed_bytes_per_image": codec.encoded_num_bytes((int(coeff.shape[1]),)),
            "quantization_mae": float(quantization_error.abs().mean().item()),
            "quantization_rmse": float(torch.sqrt(quantization_error.pow(2).mean()).item()),
            "quantization_max_abs_error": float(quantization_error.abs().max().item()),
            "codebooks": [[float(v) for v in row] for row in codec.codebooks],
        }
    return priors


def _huffman_payload_stats(codes: torch.Tensor, huffman: StaticControlHuffmanCode) -> tuple[torch.Tensor, torch.Tensor]:
    if codes.ndim != 2:
        raise ValueError("codes must have shape [N, K]")
    length_columns = []
    for index, codebook in enumerate(huffman.position_codes):
        lengths = torch.tensor(codebook.code_lengths, dtype=torch.long)
        length_columns.append(lengths[codes[:, index].to(torch.long)])
    symbol_bits = torch.stack(length_columns, dim=1)
    payload_bytes = torch.div(symbol_bits.sum(dim=1) + 7, 8, rounding_mode="floor")
    return symbol_bits, payload_bytes


def fit_component_codebook_huffman_priors(
    coefficients: torch.Tensor,
    *,
    codebook_priors: dict[str, dict[str, object]],
    ranges: dict[str, float],
    smoothing_count: int = 1,
) -> dict[str, dict[str, object]]:
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [N, K]")
    if smoothing_count < 0:
        raise ValueError("smoothing_count must be non-negative")
    coeff = coefficients.detach().cpu().float()
    priors: dict[str, dict[str, object]] = {}
    for codebook_key, codebook_prior in codebook_priors.items():
        if not isinstance(codebook_prior, dict):
            continue
        bits = int(codebook_prior.get("bits", 0))
        raw_codebooks = codebook_prior.get("codebooks")
        if bits <= 0 or not isinstance(raw_codebooks, list):
            continue
        codec = ComponentCodebookControlCode(
            bits=bits,
            codebooks=tuple(tuple(float(v) for v in row) for row in raw_codebooks),
        )
        if codec.component_count != int(coeff.shape[1]):
            raise ValueError("component codebook count must match coefficient count")
        codes = codec.quantize(coeff)
        decoded = codec.dequantize(codes)
        quantization_error = decoded - coeff
        counts = torch.zeros(codes.shape[1], codec.levels, dtype=torch.long)
        ones = torch.ones(codes.shape[0], dtype=torch.long)
        for index in range(codes.shape[1]):
            counts[index].scatter_add_(0, codes[:, index].to(torch.long), ones)
        huffman = StaticControlHuffmanCode.from_counts(counts, smoothing_count=smoothing_count)
        symbol_bits, payload_lengths = _huffman_payload_stats(codes, huffman)
        cumulative_bits = torch.cumsum(symbol_bits, dim=1)
        prefix_payload_bytes = torch.div(cumulative_bits + 7, 8, rounding_mode="floor")
        fixed_bytes = codec.encoded_num_bytes((codes.shape[1],))
        for range_name, value_range in ranges.items():
            value_range_f = float(value_range)
            if value_range_f <= 0:
                raise ValueError(f"range {range_name} must be positive")
            key = f"{range_name}_b{bits}_codebook"
            priors[key] = {
                "codec": "static_control_huffman_prior",
                "version": 0,
                "quantile": range_name,
                "quantizer": "uniform",
                "quantizer_label": "component_codebook",
                "basis_range_mode": "component_codebook",
                "codebook_key": codebook_key,
                "mu": 16.0,
                "bits": bits,
                "range": value_range_f,
                "smoothing_count": smoothing_count,
                "fixed_bytes_per_image": fixed_bytes,
                "quantization_mae": float(quantization_error.abs().mean().item()),
                "quantization_rmse": float(torch.sqrt(quantization_error.pow(2).mean()).item()),
                "clipped_fraction": 0.0,
                "mean_payload_bytes": float(payload_lengths.float().mean().item()),
                "min_payload_bytes": int(payload_lengths.min().item()),
                "max_payload_bytes": int(payload_lengths.max().item()),
                "mean_bits_per_symbol": float(symbol_bits.float().mean().item()),
                "prefix_mean_payload_bytes": [
                    float(value) for value in prefix_payload_bytes.float().mean(dim=0).tolist()
                ],
                "prefix_min_payload_bytes": [int(value) for value in prefix_payload_bytes.min(dim=0).values.tolist()],
                "prefix_max_payload_bytes": [int(value) for value in prefix_payload_bytes.max(dim=0).values.tolist()],
                "huffman": huffman.to_dict(),
            }
    return priors


def fit_sparse_topk_control_priors(
    coefficients: torch.Tensor,
    *,
    bits: int,
    ranges: dict[str, float],
    candidate_components: list[int] | tuple[int, ...],
    selected_components: list[int] | tuple[int, ...],
    quantizers: list[str] | tuple[str, ...] = ("uniform",),
    mu: float = 16.0,
    smoothing_count: int = 1,
) -> dict[str, dict[str, object]]:
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [N, K]")
    if bits <= 0:
        raise ValueError("bits must be positive")
    if mu <= 0:
        raise ValueError("mu must be positive")
    if smoothing_count < 0:
        raise ValueError("smoothing_count must be non-negative")
    coeff = coefficients.detach().cpu().float()
    basis_count = int(coeff.shape[1])
    priors: dict[str, dict[str, object]] = {}
    for candidate_count_raw in candidate_components:
        candidate_count = int(candidate_count_raw)
        if candidate_count <= 1 or candidate_count > basis_count:
            raise ValueError("candidate_components entries must be in [2, basis_count]")
        candidate_coeff = coeff[:, :candidate_count]
        for selected_count_raw in selected_components:
            selected_count = int(selected_count_raw)
            if selected_count <= 0 or selected_count > candidate_count:
                continue
            topk_indices = torch.topk(candidate_coeff.abs(), k=selected_count, dim=1, largest=True, sorted=False).indices
            sorted_indices = torch.sort(topk_indices.to(torch.long), dim=1).values
            selected_values = torch.gather(candidate_coeff, dim=1, index=sorted_indices)
            index_counts = torch.zeros(selected_count, candidate_count, dtype=torch.long)
            ones = torch.ones(sorted_indices.shape[0], dtype=torch.long)
            for position in range(selected_count):
                index_counts[position].scatter_add_(0, sorted_indices[:, position], ones)
            index_huffman = StaticControlHuffmanCode.from_counts(index_counts, smoothing_count=smoothing_count)
            index_length_columns = []
            for position, codebook in enumerate(index_huffman.position_codes):
                lengths = torch.tensor(codebook.code_lengths, dtype=torch.long)
                index_length_columns.append(lengths[sorted_indices[:, position]])
            index_bits = torch.stack(index_length_columns, dim=1).sum(dim=1)
            index_payload_bytes = torch.div(index_bits + 7, 8, rounding_mode="floor")
            fixed_index_bytes = (selected_count * semantic_bits_per_token(candidate_count) + 7) // 8
            for quantizer in quantizers:
                if quantizer not in {"uniform", "mu_law"}:
                    raise ValueError("quantizers must contain only uniform or mu_law")
                quantizer_label = control_quantizer_label(quantizer, mu)
                for range_name, value_range in ranges.items():
                    value_range_f = float(value_range)
                    if value_range_f <= 0:
                        raise ValueError(f"range {range_name} must be positive")
                    codec = build_control_grid_code(quantizer=quantizer, bits=bits, value_range=value_range_f, mu=mu)
                    value_codes = codec.quantize(selected_values)
                    decoded = codec.dequantize(value_codes)
                    quantization_error = decoded - selected_values
                    clipped = selected_values.abs() > value_range_f
                    value_counts = torch.zeros(selected_count, codec.levels, dtype=torch.long)
                    for position in range(selected_count):
                        value_counts[position].scatter_add_(0, value_codes[:, position].to(torch.long), ones)
                    value_huffman = StaticControlHuffmanCode.from_counts(
                        value_counts,
                        smoothing_count=smoothing_count,
                    )
                    value_length_columns = []
                    for position, codebook in enumerate(value_huffman.position_codes):
                        lengths = torch.tensor(codebook.code_lengths, dtype=torch.long)
                        value_length_columns.append(lengths[value_codes[:, position].to(torch.long)])
                    value_bits = torch.stack(value_length_columns, dim=1).sum(dim=1)
                    value_payload_bytes = torch.div(value_bits + 7, 8, rounding_mode="floor")
                    payload_lengths = index_payload_bytes + value_payload_bytes
                    fixed_value_bytes = codec.encoded_num_bytes((selected_count,))
                    key_base = f"topk_c{candidate_count}_k{selected_count}_{range_name}_b{bits}"
                    key = key_base if quantizer == "uniform" else f"{key_base}_{quantizer_label}"
                    priors[key] = {
                        "codec": "sparse_topk_static_huffman_prior",
                        "version": 0,
                        "candidate_components": candidate_count,
                        "selected_components": selected_count,
                        "quantile": range_name,
                        "quantizer": quantizer,
                        "quantizer_label": quantizer_label,
                        "mu": float(mu),
                        "bits": bits,
                        "range": value_range_f,
                        "smoothing_count": smoothing_count,
                        "fixed_index_bytes_per_image": fixed_index_bytes,
                        "fixed_value_bytes_per_image": fixed_value_bytes,
                        "fixed_bytes_per_image": fixed_index_bytes + fixed_value_bytes,
                        "quantization_mae": float(quantization_error.abs().mean().item()),
                        "quantization_rmse": float(torch.sqrt(quantization_error.pow(2).mean()).item()),
                        "clipped_fraction": float(clipped.float().mean().item()),
                        "mean_payload_bytes": float(payload_lengths.float().mean().item()),
                        "min_payload_bytes": int(payload_lengths.min().item()),
                        "max_payload_bytes": int(payload_lengths.max().item()),
                        "index_mean_payload_bytes": float(index_payload_bytes.float().mean().item()),
                        "value_mean_payload_bytes": float(value_payload_bytes.float().mean().item()),
                        "index_mean_bits_per_symbol": float(index_bits.float().mean().item()) / float(selected_count),
                        "value_mean_bits_per_symbol": float(value_bits.float().mean().item()) / float(selected_count),
                        "index_huffman": index_huffman.to_dict(),
                        "value_huffman": value_huffman.to_dict(),
                    }
    return priors


def fit_sparse_topk_component_codebook_priors(
    coefficients: torch.Tensor,
    *,
    codebook_priors: dict[str, dict[str, object]],
    ranges: dict[str, float],
    candidate_components: list[int] | tuple[int, ...],
    selected_components: list[int] | tuple[int, ...],
    smoothing_count: int = 1,
) -> dict[str, dict[str, object]]:
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [N, K]")
    if smoothing_count < 0:
        raise ValueError("smoothing_count must be non-negative")
    coeff = coefficients.detach().cpu().float()
    basis_count = int(coeff.shape[1])
    priors: dict[str, dict[str, object]] = {}
    for codebook_key, codebook_prior in codebook_priors.items():
        if not isinstance(codebook_prior, dict):
            continue
        bits = int(codebook_prior.get("bits", 0))
        raw_codebooks = codebook_prior.get("codebooks")
        if bits <= 0 or not isinstance(raw_codebooks, list):
            continue
        codebooks = torch.tensor(raw_codebooks, dtype=torch.float32)
        if int(codebooks.shape[0]) != basis_count:
            raise ValueError("component codebook count must match coefficient count")
        levels = 1 << bits
        if int(codebooks.shape[1]) != levels:
            raise ValueError("component codebook levels must match bits")
        for candidate_count_raw in candidate_components:
            candidate_count = int(candidate_count_raw)
            if candidate_count <= 1 or candidate_count > basis_count:
                raise ValueError("candidate_components entries must be in [2, basis_count]")
            candidate_coeff = coeff[:, :candidate_count]
            candidate_codebooks = codebooks[:candidate_count]
            for selected_count_raw in selected_components:
                selected_count = int(selected_count_raw)
                if selected_count <= 0 or selected_count > candidate_count:
                    continue
                topk_indices = torch.topk(
                    candidate_coeff.abs(),
                    k=selected_count,
                    dim=1,
                    largest=True,
                    sorted=False,
                ).indices
                sorted_indices = torch.sort(topk_indices.to(torch.long), dim=1).values
                selected_values = torch.gather(candidate_coeff, dim=1, index=sorted_indices)
                selected_codebooks = candidate_codebooks.index_select(0, sorted_indices.reshape(-1)).reshape(
                    sorted_indices.shape[0],
                    selected_count,
                    levels,
                )
                value_codes = torch.argmin(
                    (selected_values.unsqueeze(-1) - selected_codebooks).abs(),
                    dim=-1,
                ).to(torch.long)
                decoded = torch.gather(selected_codebooks, dim=2, index=value_codes.unsqueeze(-1)).squeeze(-1)
                quantization_error = decoded - selected_values
                index_counts = torch.zeros(selected_count, candidate_count, dtype=torch.long)
                ones = torch.ones(sorted_indices.shape[0], dtype=torch.long)
                for position in range(selected_count):
                    index_counts[position].scatter_add_(0, sorted_indices[:, position], ones)
                index_huffman = StaticControlHuffmanCode.from_counts(index_counts, smoothing_count=smoothing_count)
                index_bits, index_payload_bytes = _huffman_payload_stats(sorted_indices, index_huffman)
                fixed_index_bytes = (selected_count * semantic_bits_per_token(candidate_count) + 7) // 8
                value_counts = torch.zeros(selected_count, levels, dtype=torch.long)
                for position in range(selected_count):
                    value_counts[position].scatter_add_(0, value_codes[:, position].to(torch.long), ones)
                value_huffman = StaticControlHuffmanCode.from_counts(value_counts, smoothing_count=smoothing_count)
                value_bits, value_payload_bytes = _huffman_payload_stats(value_codes, value_huffman)
                payload_lengths = index_payload_bytes + value_payload_bytes
                fixed_value_bytes = (selected_count * bits + 7) // 8
                for range_name, value_range in ranges.items():
                    value_range_f = float(value_range)
                    if value_range_f <= 0:
                        raise ValueError(f"range {range_name} must be positive")
                    key = f"topk_c{candidate_count}_k{selected_count}_{range_name}_b{bits}_codebook"
                    priors[key] = {
                        "codec": "sparse_topk_static_huffman_prior",
                        "version": 0,
                        "candidate_components": candidate_count,
                        "selected_components": selected_count,
                        "quantile": range_name,
                        "quantizer": "uniform",
                        "quantizer_label": "component_codebook",
                        "basis_range_mode": "component_codebook",
                        "codebook_key": codebook_key,
                        "mu": 16.0,
                        "bits": bits,
                        "range": value_range_f,
                        "smoothing_count": smoothing_count,
                        "fixed_index_bytes_per_image": fixed_index_bytes,
                        "fixed_value_bytes_per_image": fixed_value_bytes,
                        "fixed_bytes_per_image": fixed_index_bytes + fixed_value_bytes,
                        "quantization_mae": float(quantization_error.abs().mean().item()),
                        "quantization_rmse": float(torch.sqrt(quantization_error.pow(2).mean()).item()),
                        "clipped_fraction": 0.0,
                        "mean_payload_bytes": float(payload_lengths.float().mean().item()),
                        "min_payload_bytes": int(payload_lengths.min().item()),
                        "max_payload_bytes": int(payload_lengths.max().item()),
                        "index_mean_payload_bytes": float(index_payload_bytes.float().mean().item()),
                        "value_mean_payload_bytes": float(value_payload_bytes.float().mean().item()),
                        "index_mean_bits_per_symbol": float(index_bits.float().mean().item()) / float(selected_count),
                        "value_mean_bits_per_symbol": float(value_bits.float().mean().item()) / float(selected_count),
                        "index_huffman": index_huffman.to_dict(),
                        "value_huffman": value_huffman.to_dict(),
                    }
    return priors


def _fit_vector_lloyd_codebook(vectors: torch.Tensor, *, levels: int, max_iter: int) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [N, K]")
    if levels <= 1:
        raise ValueError("levels must be > 1")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    values = vectors.detach().cpu().float()
    sample_count = int(values.shape[0])
    if sample_count == 0:
        raise ValueError("vectors must contain at least one sample")
    order = torch.argsort(values.square().sum(dim=1))
    if sample_count >= levels:
        init_positions = torch.linspace(0, sample_count - 1, steps=levels).round().to(torch.long)
        centroids = values.index_select(0, order.index_select(0, init_positions)).clone()
    else:
        repeats = math.ceil(levels / sample_count)
        centroids = values.index_select(0, order).repeat((repeats, 1))[:levels].clone()
    for _ in range(max_iter):
        distances = torch.cdist(values, centroids, p=2.0)
        assignment = torch.argmin(distances, dim=1)
        updated = centroids.clone()
        for level in range(levels):
            mask = assignment == level
            if bool(mask.any()):
                updated[level] = values[mask].mean(dim=0)
        if torch.allclose(updated, centroids, atol=1.0e-5, rtol=0.0):
            centroids = updated
            break
        centroids = updated
    return centroids.to(torch.float32)


def fit_vector_codebook_priors(
    coefficients: torch.Tensor,
    *,
    components: list[int] | tuple[int, ...],
    bits_list: list[int] | tuple[int, ...],
    max_iter: int = 50,
    smoothing_count: int = 1,
) -> dict[str, dict[str, object]]:
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [N, K]")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if smoothing_count < 0:
        raise ValueError("smoothing_count must be non-negative")
    coeff = coefficients.detach().cpu().float()
    basis_count = int(coeff.shape[1])
    priors: dict[str, dict[str, object]] = {}
    for component_raw in components:
        component_count = int(component_raw)
        if component_count <= 0 or component_count > basis_count:
            raise ValueError("vector codebook components entries must be in [1, basis_count]")
        vector_values = coeff[:, :component_count]
        for bits_raw in bits_list:
            bits = int(bits_raw)
            if bits <= 0:
                raise ValueError("vector codebook bits entries must be positive")
            if bits > 16:
                raise ValueError("vector codebook bits entries must be <= 16")
            levels = 1 << bits
            vectors = _fit_vector_lloyd_codebook(vector_values, levels=levels, max_iter=max_iter)
            codec = VectorCodebookControlCode(
                bits=bits,
                vectors=tuple(tuple(float(v) for v in row.tolist()) for row in vectors),
            )
            codes = codec.quantize(vector_values)
            decoded = codec.dequantize(codes).reshape(vector_values.shape)
            quantization_error = decoded - vector_values
            counts = torch.zeros(1, codec.levels, dtype=torch.long)
            counts[0].scatter_add_(0, codes.to(torch.long), torch.ones(codes.numel(), dtype=torch.long))
            huffman = StaticControlHuffmanCode.from_counts(counts, smoothing_count=smoothing_count)
            symbol_bits, payload_lengths = _huffman_payload_stats(codes.reshape(-1, 1), huffman)
            key = f"vq_k{component_count}_b{bits}"
            priors[key] = {
                "codec": "vector_codebook_control_fixed_bits",
                "version": 0,
                "method": "lloyd_vector",
                "selection": "vector",
                "components": component_count,
                "bits": bits,
                "levels": levels,
                "max_iter": int(max_iter),
                "smoothing_count": smoothing_count,
                "fixed_bytes_per_image": codec.encoded_num_bytes((1,)),
                "huffman_mean_payload_bytes": float(payload_lengths.float().mean().item()),
                "huffman_min_payload_bytes": int(payload_lengths.min().item()),
                "huffman_max_payload_bytes": int(payload_lengths.max().item()),
                "huffman_mean_bits_per_symbol": float(symbol_bits.float().mean().item()),
                "quantization_mae": float(quantization_error.abs().mean().item()),
                "quantization_rmse": float(torch.sqrt(quantization_error.pow(2).mean()).item()),
                "quantization_max_abs_error": float(quantization_error.abs().max().item()),
                "huffman": huffman.to_dict(),
                "vectors": [[float(v) for v in row] for row in codec.vectors],
            }
    return priors


def compute_basis_reconstruction_stats(
    coefficients: torch.Tensor,
    *,
    prefix_components: list[int] | tuple[int, ...],
    topk_candidate_components: list[int] | tuple[int, ...] = (),
    topk_components: list[int] | tuple[int, ...] = (),
) -> dict[str, dict[str, object]]:
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [N, K]")
    coeff = coefficients.detach().cpu().float()
    basis_count = int(coeff.shape[1])
    total_energy = coeff.square().sum(dim=1).clamp_min(1.0e-12)
    rows: dict[str, dict[str, object]] = {}

    def add_row(
        *,
        key: str,
        selection: str,
        components: int,
        candidate_components: int,
        retained_energy: torch.Tensor,
    ) -> None:
        retained_fraction = (retained_energy / total_energy).clamp(0.0, 1.0)
        residual_fraction = (1.0 - retained_fraction).clamp(0.0, 1.0)
        rows[key] = {
            "selection": selection,
            "components": int(components),
            "candidate_components": int(candidate_components),
            "retained_energy_fraction_mean": float(retained_fraction.mean().item()),
            "retained_energy_fraction_p10": float(torch.quantile(retained_fraction, 0.10).item()),
            "retained_energy_fraction_p50": float(torch.quantile(retained_fraction, 0.50).item()),
            "retained_energy_fraction_p90": float(torch.quantile(retained_fraction, 0.90).item()),
            "residual_energy_fraction_mean": float(residual_fraction.mean().item()),
            "residual_energy_fraction_p50": float(torch.quantile(residual_fraction, 0.50).item()),
            "retained_l2_mean": float(torch.sqrt(retained_energy.clamp_min(0.0)).mean().item()),
            "residual_l2_mean": float(torch.sqrt((total_energy - retained_energy).clamp_min(0.0)).mean().item()),
        }

    for count_raw in prefix_components:
        count = int(count_raw)
        if count <= 0 or count > basis_count:
            raise ValueError("prefix_components entries must be in [1, basis_count]")
        retained = coeff[:, :count].square().sum(dim=1)
        add_row(
            key=f"prefix_k{count}",
            selection="prefix",
            components=count,
            candidate_components=count,
            retained_energy=retained,
        )

    for candidate_raw in topk_candidate_components:
        candidate_count = int(candidate_raw)
        if candidate_count <= 1 or candidate_count > basis_count:
            raise ValueError("topk_candidate_components entries must be in [2, basis_count]")
        candidate_coeff = coeff[:, :candidate_count]
        for count_raw in topk_components:
            count = int(count_raw)
            if count <= 0 or count > candidate_count:
                continue
            topk_values = torch.topk(candidate_coeff.abs(), k=count, dim=1, largest=True, sorted=False).values
            retained = topk_values.square().sum(dim=1)
            add_row(
                key=f"topk_c{candidate_count}_k{count}",
                selection="topk",
                components=count,
                candidate_components=candidate_count,
                retained_energy=retained,
            )
            if count < candidate_count and count <= candidate_count - count:
                tail_coeff = candidate_coeff[:, count:candidate_count]
                tail_topk = torch.topk(tail_coeff.abs(), k=count, dim=1, largest=True, sorted=False).values
                retained_prefix_topk = coeff[:, :count].square().sum(dim=1) + tail_topk.square().sum(dim=1)
                add_row(
                    key=f"prefixtopk_p{count}_c{candidate_count}_k{count}",
                    selection="prefix_topk",
                    components=count,
                    candidate_components=candidate_count,
                    retained_energy=retained_prefix_topk,
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/stage5_control_basis")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--huffman-bits", type=int, default=4)
    parser.add_argument("--huffman-quantile", action="append", default=[])
    parser.add_argument("--huffman-quantizer", action="append", default=[])
    parser.add_argument("--huffman-mu", type=float, default=16.0)
    parser.add_argument("--huffman-smoothing-count", type=int, default=1)
    parser.add_argument("--component-codebook-bits", nargs="+", type=int, default=[])
    parser.add_argument("--component-codebook-max-iter", type=int, default=50)
    parser.add_argument("--vector-codebook-components", nargs="+", type=int, default=[])
    parser.add_argument("--vector-codebook-bits", nargs="+", type=int, default=[])
    parser.add_argument("--vector-codebook-max-iter", type=int, default=50)
    parser.add_argument("--sparse-topk-candidate-components", nargs="+", type=int, default=[])
    parser.add_argument("--sparse-topk-components", nargs="+", type=int, default=[])
    parser.add_argument("--pre-basis-affine", choices=("true", "false"), default="false")
    parser.add_argument("--pre-basis-affine-groups", type=int, default=0)
    parser.add_argument("--pre-basis-affine-grid-size", type=int, default=1)
    parser.add_argument("--pre-basis-affine-bits", type=int, default=4)
    parser.add_argument("--pre-basis-affine-gain-range", type=float, default=1.0)
    parser.add_argument("--pre-basis-affine-bias-range", type=float, default=0.25)
    parser.add_argument("--pre-basis-affine-quantizer", choices=("uniform", "mu_law"), default="uniform")
    parser.add_argument("--pre-basis-affine-mu", type=float, default=16.0)
    parser.add_argument("--pre-basis-affine-scale", type=float, default=1.0)
    parser.add_argument("--condition-residual-scale", type=float, default=None)
    parser.add_argument(
        "--condition-residual-tanh",
        choices=("checkpoint", "true", "false"),
        default="checkpoint",
    )
    parser.add_argument("--center", choices=("true", "false"), default="true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; restart the container before fitting a Stage 5 control basis.")
    if args.groups <= 0:
        raise ValueError("--groups must be positive")
    if args.grid_size <= 0:
        raise ValueError("--grid-size must be positive")
    if args.components <= 0:
        raise ValueError("--components must be positive")
    if args.huffman_bits <= 0:
        raise ValueError("--huffman-bits must be positive")
    if args.huffman_mu <= 0:
        raise ValueError("--huffman-mu must be positive")
    if args.huffman_smoothing_count < 0:
        raise ValueError("--huffman-smoothing-count must be non-negative")
    if args.component_codebook_max_iter <= 0:
        raise ValueError("--component-codebook-max-iter must be positive")
    component_codebook_bits = args.component_codebook_bits or [args.huffman_bits]
    if any(bits <= 0 for bits in component_codebook_bits):
        raise ValueError("--component-codebook-bits entries must be positive")
    if args.vector_codebook_max_iter <= 0:
        raise ValueError("--vector-codebook-max-iter must be positive")
    vector_codebook_components = args.vector_codebook_components or []
    vector_codebook_bits = args.vector_codebook_bits or []
    if any(value <= 0 for value in vector_codebook_components):
        raise ValueError("--vector-codebook-components entries must be positive")
    if any(bits <= 0 for bits in vector_codebook_bits):
        raise ValueError("--vector-codebook-bits entries must be positive")
    pre_basis_affine = args.pre_basis_affine == "true"
    effective_pre_basis_affine_groups = int(args.pre_basis_affine_groups or args.groups)
    if pre_basis_affine:
        if effective_pre_basis_affine_groups <= 0:
            raise ValueError("--pre-basis-affine-groups must be positive")
        if args.pre_basis_affine_grid_size <= 0:
            raise ValueError("--pre-basis-affine-grid-size must be positive")
        if args.pre_basis_affine_bits <= 0:
            raise ValueError("--pre-basis-affine-bits must be positive")
        if args.pre_basis_affine_gain_range <= 0:
            raise ValueError("--pre-basis-affine-gain-range must be positive")
        if args.pre_basis_affine_bias_range <= 0:
            raise ValueError("--pre-basis-affine-bias-range must be positive")
        if args.pre_basis_affine_mu <= 0:
            raise ValueError("--pre-basis-affine-mu must be positive")
        if args.pre_basis_affine_scale <= 0:
            raise ValueError("--pre-basis-affine-scale must be positive")
    huffman_quantizers = args.huffman_quantizer or ["uniform", "mu_law"]
    if any(item not in {"uniform", "mu_law"} for item in huffman_quantizers):
        raise ValueError("--huffman-quantizer must be uniform or mu_law")

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda")
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_stage5_condition_control_basis"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone_cfg = CoDLiteOneStepBackboneConfig(**payload["backbone_config"])
    backbone = CoDLiteOneStepBackbone.load(backbone_cfg, device=device)
    apply_lora_adapters_from_config(backbone.net, payload.get("backbone_lora_config", {}))
    if payload.get("backbone_trainable_state"):
        load_named_parameter_state(backbone.net, payload["backbone_trainable_state"], strict=True)
    backbone.eval()
    channel_group_sizes(backbone.condition_channels, args.groups)
    if pre_basis_affine:
        channel_group_sizes(backbone.condition_channels, effective_pre_basis_affine_groups)

    adapter = build_adapter_from_payload(payload).to(device)
    adapter.load_state_dict(payload["model"])
    adapter.eval()

    adapter_config = dict(payload.get("adapter_config", {}))
    semantic_channels = int(adapter_config.get("semantic_channels", 3))
    detail_context = str(payload.get("detail_context", "none"))
    detail_channels = detail_context_channels(detail_context)
    residual_scale = float(
        payload.get("condition_residual_scale", 1.0)
        if args.condition_residual_scale is None
        else args.condition_residual_scale
    )
    if args.condition_residual_tanh == "checkpoint":
        residual_tanh = bool(payload.get("condition_residual_tanh", False))
    else:
        residual_tanh = args.condition_residual_tanh == "true"
    pre_basis_affine_gain_codec = build_control_grid_code(
        quantizer=args.pre_basis_affine_quantizer,
        bits=args.pre_basis_affine_bits,
        value_range=args.pre_basis_affine_gain_range,
        mu=args.pre_basis_affine_mu,
    )
    pre_basis_affine_bias_codec = build_control_grid_code(
        quantizer=args.pre_basis_affine_quantizer,
        bits=args.pre_basis_affine_bits,
        value_range=args.pre_basis_affine_bias_range,
        mu=args.pre_basis_affine_mu,
    )

    dataset = Stage4ManifestDataset(
        Path(args.manifest),
        None,
        limit=args.limit,
        crop_size=args.crop_size,
        semantic_channels=semantic_channels,
        detail_context=detail_context,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    grids: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=run_name):
            reference = batch["reference"].to(device, non_blocking=True)
            semantic = batch["semantic"].to(device, non_blocking=True)
            semantic_latent = batch["semantic_latent"].to(device, non_blocking=True)
            detail_context_tensor = (
                batch["detail_context"].to(device, non_blocking=True) if detail_channels > 0 else None
            )
            stage3 = batch["stage3"].to(device, non_blocking=True)
            residual = stage3 - semantic
            condition_size = backbone.condition_size(int(stage3.shape[-2]), int(stage3.shape[-1]))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                base_cond = backbone.native_condition(stage3)
                target_cond = backbone.native_condition(reference)
                cond_delta = adapter(
                    stage3,
                    semantic,
                    residual,
                    semantic_latent,
                    condition_size=condition_size,
                    base_condition=base_cond,
                    detail_context=detail_context_tensor,
                )
                pred_cond = apply_condition_residual(
                    base_cond,
                    cond_delta,
                    residual_scale=residual_scale,
                    residual_tanh=residual_tanh,
                )
                if pre_basis_affine:
                    affine_correction, _, _ = grouped_condition_affine_control(
                        base_cond.float(),
                        pred_cond.float(),
                        target_cond.float(),
                        groups=effective_pre_basis_affine_groups,
                        grid_size=args.pre_basis_affine_grid_size,
                        gain_codec=pre_basis_affine_gain_codec,
                        bias_codec=pre_basis_affine_bias_codec,
                        scale=args.pre_basis_affine_scale,
                    )
                    pred_cond = pred_cond + affine_correction.to(dtype=pred_cond.dtype)
            error_grid = grouped_condition_grid(
                target_cond.float() - pred_cond.float(),
                groups=args.groups,
                grid_size=args.grid_size,
            )
            grids.append(error_grid.detach().cpu())

    all_grids = torch.cat(grids, dim=0)
    fitted = fit_control_basis_from_grids(
        all_grids,
        components=args.components,
        center=args.center == "true",
    )
    coefficient_abs_quantiles = fitted["coefficient_abs_quantiles"]
    if not isinstance(coefficient_abs_quantiles, dict):
        raise TypeError("coefficient_abs_quantiles must be a dict")
    huffman_quantiles = [normalize_quantile_label(q) for q in (args.huffman_quantile or ["p95", "p99"])]
    huffman_ranges = {
        name: float(coefficient_abs_quantiles[name])
        for name in huffman_quantiles
        if name in coefficient_abs_quantiles
    }
    missing_quantiles = sorted(set(huffman_quantiles) - set(huffman_ranges))
    if missing_quantiles:
        raise ValueError(f"missing coefficient quantiles for Huffman fitting: {missing_quantiles}")
    coefficients = fitted["coefficients"]
    if not isinstance(coefficients, torch.Tensor):
        raise TypeError("fit_control_basis_from_grids must return tensor coefficients")
    control_huffman_priors = fit_control_huffman_priors(
        coefficients,
        bits=args.huffman_bits,
        ranges=huffman_ranges,
        quantizers=huffman_quantizers,
        mu=args.huffman_mu,
        smoothing_count=args.huffman_smoothing_count,
    )
    coefficient_component_codebooks = fit_component_codebook_priors(
        coefficients,
        bits_list=component_codebook_bits,
        max_iter=args.component_codebook_max_iter,
    )
    control_huffman_priors.update(
        fit_component_codebook_huffman_priors(
            coefficients,
            codebook_priors=coefficient_component_codebooks,
            ranges=huffman_ranges,
            smoothing_count=args.huffman_smoothing_count,
        )
    )
    coefficient_vector_codebooks: dict[str, dict[str, object]] = {}
    if vector_codebook_components and vector_codebook_bits:
        coefficient_vector_codebooks = fit_vector_codebook_priors(
            coefficients,
            components=vector_codebook_components,
            bits_list=vector_codebook_bits,
            max_iter=args.vector_codebook_max_iter,
            smoothing_count=args.huffman_smoothing_count,
        )
    prefix_stat_components = sorted({8, 16, 32, args.components} & set(range(1, args.components + 1)))
    sparse_topk_control_priors: dict[str, dict[str, object]] = {}
    topk_stat_candidates: list[int] = []
    topk_stat_components: list[int] = []
    if args.sparse_topk_candidate_components or args.sparse_topk_components:
        sparse_candidates = args.sparse_topk_candidate_components or [args.components]
        sparse_components = args.sparse_topk_components or [8, 16, 32]
        topk_stat_candidates = sparse_candidates
        topk_stat_components = sparse_components
        sparse_topk_control_priors = fit_sparse_topk_control_priors(
            coefficients,
            bits=args.huffman_bits,
            ranges=huffman_ranges,
            candidate_components=sparse_candidates,
            selected_components=sparse_components,
            quantizers=huffman_quantizers,
            mu=args.huffman_mu,
            smoothing_count=args.huffman_smoothing_count,
        )
        sparse_topk_control_priors.update(
            fit_sparse_topk_component_codebook_priors(
                coefficients,
                codebook_priors=coefficient_component_codebooks,
                ranges=huffman_ranges,
                candidate_components=sparse_candidates,
                selected_components=sparse_components,
                smoothing_count=args.huffman_smoothing_count,
            )
        )
    basis_reconstruction_stats = compute_basis_reconstruction_stats(
        coefficients,
        prefix_components=prefix_stat_components,
        topk_candidate_components=topk_stat_candidates,
        topk_components=topk_stat_components,
    )

    basis_path = output_dir / "control_basis.pt"
    summary = {
        "run_name": run_name,
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "sample_count": int(all_grids.shape[0]),
        "groups": args.groups,
        "grid_size": args.grid_size,
        "components": args.components,
        "center": args.center == "true",
        "condition_residual_scale": residual_scale,
        "condition_residual_tanh": residual_tanh,
        "pre_basis_affine": pre_basis_affine,
        "pre_basis_affine_groups": effective_pre_basis_affine_groups,
        "pre_basis_affine_grid_size": args.pre_basis_affine_grid_size,
        "pre_basis_affine_bits": args.pre_basis_affine_bits,
        "pre_basis_affine_gain_range": args.pre_basis_affine_gain_range,
        "pre_basis_affine_bias_range": args.pre_basis_affine_bias_range,
        "pre_basis_affine_quantizer": args.pre_basis_affine_quantizer,
        "pre_basis_affine_mu": args.pre_basis_affine_mu,
        "pre_basis_affine_scale": args.pre_basis_affine_scale,
        "explained_variance": fitted["explained_variance"],
        "cumulative_explained_variance": fitted["cumulative_explained_variance"],
        "coefficient_abs_mean": fitted["coefficient_abs_mean"],
        "coefficient_abs_std": fitted["coefficient_abs_std"],
        "coefficient_abs_max": fitted["coefficient_abs_max"],
        "coefficient_abs_quantiles": fitted["coefficient_abs_quantiles"],
        "coefficient_component_abs_p95": fitted["coefficient_component_abs_p95"],
        "coefficient_component_abs_p99": fitted["coefficient_component_abs_p99"],
        "control_huffman_priors": control_huffman_priors,
        "coefficient_component_codebooks": coefficient_component_codebooks,
        "coefficient_vector_codebooks": coefficient_vector_codebooks,
        "sparse_topk_control_priors": sparse_topk_control_priors,
        "basis_reconstruction_stats": basis_reconstruction_stats,
        "huffman_quantizers": huffman_quantizers,
        "huffman_mu": args.huffman_mu,
        "component_codebook_bits": component_codebook_bits,
        "component_codebook_max_iter": args.component_codebook_max_iter,
        "vector_codebook_components": vector_codebook_components,
        "vector_codebook_bits": vector_codebook_bits,
        "vector_codebook_max_iter": args.vector_codebook_max_iter,
        "basis": str(basis_path),
    }
    torch.save(
        {
            "basis": fitted["basis"],
            "mean": fitted["mean"],
            "singular_values": fitted["singular_values"],
            "explained_variance": fitted["explained_variance"],
            "cumulative_explained_variance": fitted["cumulative_explained_variance"],
            "coefficient_abs_mean": fitted["coefficient_abs_mean"],
            "coefficient_abs_std": fitted["coefficient_abs_std"],
            "coefficient_abs_max": fitted["coefficient_abs_max"],
            "coefficient_abs_quantiles": fitted["coefficient_abs_quantiles"],
            "coefficient_component_abs_p95": fitted["coefficient_component_abs_p95"],
            "coefficient_component_abs_p99": fitted["coefficient_component_abs_p99"],
            "control_huffman_priors": control_huffman_priors,
            "coefficient_component_codebooks": coefficient_component_codebooks,
            "coefficient_vector_codebooks": coefficient_vector_codebooks,
            "sparse_topk_control_priors": sparse_topk_control_priors,
            "basis_reconstruction_stats": basis_reconstruction_stats,
            "huffman_quantizers": huffman_quantizers,
            "huffman_mu": args.huffman_mu,
            "component_codebook_bits": component_codebook_bits,
            "component_codebook_max_iter": args.component_codebook_max_iter,
            "vector_codebook_components": vector_codebook_components,
            "vector_codebook_bits": vector_codebook_bits,
            "vector_codebook_max_iter": args.vector_codebook_max_iter,
            "groups": args.groups,
            "grid_size": args.grid_size,
            "components": args.components,
            "pre_basis_affine": pre_basis_affine,
            "pre_basis_affine_groups": effective_pre_basis_affine_groups,
            "pre_basis_affine_grid_size": args.pre_basis_affine_grid_size,
            "pre_basis_affine_bits": args.pre_basis_affine_bits,
            "pre_basis_affine_gain_range": args.pre_basis_affine_gain_range,
            "pre_basis_affine_bias_range": args.pre_basis_affine_bias_range,
            "pre_basis_affine_quantizer": args.pre_basis_affine_quantizer,
            "pre_basis_affine_mu": args.pre_basis_affine_mu,
            "pre_basis_affine_scale": args.pre_basis_affine_scale,
            "source": run_name,
            "summary": summary,
        },
        basis_path,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_run_doc(
        Path("docs/experiments") / f"{run_name}.md",
        {
            "run_name": run_name,
            "date": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "summary": summary,
            "artifacts": {
                "basis": str(basis_path),
                "summary": str(summary_path),
            },
        },
    )
    stdout_summary = {
        "run_name": run_name,
        "basis": str(basis_path),
        "summary": str(summary_path),
        "sample_count": int(all_grids.shape[0]),
        "groups": args.groups,
        "grid_size": args.grid_size,
        "components": args.components,
        "coefficient_abs_quantiles": fitted["coefficient_abs_quantiles"],
        "basis_reconstruction_stats": basis_reconstruction_stats,
        "control_huffman_prior_keys": sorted(control_huffman_priors),
        "sparse_topk_control_prior_keys": sorted(sparse_topk_control_priors),
    }
    print(json.dumps(stdout_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
