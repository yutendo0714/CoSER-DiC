from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from coserdic.entropy import (
    CoSERBitstream,
    PrefixTopKControlBasisCode,
    SparseControlBasisCode,
    UniformControlGridCode,
    build_control_grid_code,
)


@dataclass(frozen=True)
class ControlSetting:
    mode: str
    groups: int
    grid: int
    bits: int
    value_range: float
    scale: float
    gain_range: float = 1.0
    bias_range: float = 0.25
    affine_groups: int = 0
    affine_grid: int = 0
    coeffs: int = 0
    codec: str = "fixed_bits"
    huffman_key: str = ""
    quantizer: str = "uniform"
    mu: float = 16.0
    selection: str = "prefix"
    candidate_components: int = 0
    prefix_components: int = 0
    basis_range_mode: str = "global"
    basis_range_floor: float = 1.0e-6
    selector_bytes: int = 1
    selector_bits: int = 0
    hybrid_rate_lambda: float = 0.0
    hybrid_selection_objective: str = "condition_l1"
    hybrid_fidelity_lambda: float = 0.0
    hybrid_fidelity_metric: str = "image_l1"

    @property
    def eval_mode(self) -> str:
        if self.mode == "grid":
            return "condition_residual_grid"
        if self.mode == "dct":
            return "condition_residual_dct"
        if self.mode == "basis":
            return "condition_residual_basis"
        if self.mode == "affine":
            return "condition_residual_affine"
        if self.mode == "affine_dct":
            return "condition_residual_affine_dct"
        if self.mode == "affine_grid":
            return "condition_residual_affine_grid"
        if self.mode == "affine_basis":
            return "condition_residual_affine_basis"
        if self.mode == "hybrid_affine_dct_grid":
            return "condition_residual_hybrid_affine_dct_grid"
        if self.mode == "hybrid_affine_dct_grid_basis":
            return "condition_residual_hybrid_affine_dct_grid_basis"
        raise ValueError(f"unknown mode: {self.mode}")

    @property
    def label(self) -> str:
        if self.mode == "grid":
            return (
                f"grid_g{self.groups}_s{self.grid}_b{self.bits}_r{self.value_range:g}_sc{self.scale:g}"
                f"_{self.quantizer}"
            )
        if self.mode == "basis":
            selection_text = (
                f"_{self.selection}c{self.candidate_components}"
                if self.selection == "topk"
                else f"_prefix{self.prefix_components}topk{self.candidate_components}"
                if self.selection == "prefix_topk"
                else "_vector"
                if self.selection == "vector"
                else ""
            )
            return (
                f"basis_g{self.groups}_s{self.grid}_k{self.coeffs}_b{self.bits}"
                f"{selection_text}_r{self.value_range:g}_sc{self.scale:g}_{self.codec}"
                f"_{self.quantizer}{self.basis_range_label}"
                f"{f'_mu{self.mu:g}' if self.quantizer == 'mu_law' else ''}"
                f"{'_' + self.huffman_key if self.huffman_key else ''}"
            )
        if self.mode == "affine":
            ag = self.effective_affine_groups
            asz = self.effective_affine_grid
            return (
                f"affine_g{ag}_s{asz}_b{self.bits}"
                f"_gr{self.gain_range:g}_br{self.bias_range:g}_sc{self.scale:g}_{self.quantizer}"
            )
        if self.mode == "affine_dct":
            ag = self.effective_affine_groups
            asz = self.effective_affine_grid
            return (
                f"affinedct_ag{ag}_as{asz}_dg{self.groups}_ds{self.grid}_k{self.coeffs}_b{self.bits}"
                f"_gr{self.gain_range:g}_br{self.bias_range:g}_r{self.value_range:g}"
                f"_sc{self.scale:g}_{self.quantizer}"
            )
        if self.mode == "affine_grid":
            ag = self.effective_affine_groups
            asz = self.effective_affine_grid
            return (
                f"affinegrid_ag{ag}_as{asz}_gg{self.groups}_gs{self.grid}_b{self.bits}"
                f"_gr{self.gain_range:g}_br{self.bias_range:g}_r{self.value_range:g}"
                f"_sc{self.scale:g}_{self.quantizer}"
            )
        if self.mode == "affine_basis":
            ag = self.effective_affine_groups
            asz = self.effective_affine_grid
            selection_text = (
                f"_{self.selection}c{self.candidate_components}"
                if self.selection == "topk"
                else f"_prefix{self.prefix_components}topk{self.candidate_components}"
                if self.selection == "prefix_topk"
                else "_vector"
                if self.selection == "vector"
                else ""
            )
            return (
                f"affinebasis_ag{ag}_as{asz}_bg{self.groups}_bs{self.grid}_k{self.coeffs}_b{self.bits}"
                f"{selection_text}_gr{self.gain_range:g}_br{self.bias_range:g}_r{self.value_range:g}"
                f"_sc{self.scale:g}_{self.codec}_{self.quantizer}{self.basis_range_label}"
                f"{f'_mu{self.mu:g}' if self.quantizer == 'mu_law' else ''}"
                f"{'_' + self.huffman_key if self.huffman_key else ''}"
            )
        if self.mode == "hybrid_affine_dct_grid":
            ag = self.effective_affine_groups
            asz = self.effective_affine_grid
            return (
                f"hybrid_affinedctgrid_ag{ag}_as{asz}_g{self.groups}_s{self.grid}_k{self.coeffs}"
                f"_b{self.bits}_gr{self.gain_range:g}_br{self.bias_range:g}_r{self.value_range:g}"
                f"_sel{self.selector_bytes}{self.selector_bits_label}_obj{self.hybrid_selection_objective}"
                f"_rl{self.hybrid_rate_lambda:g}{self.fidelity_label}_sc{self.scale:g}_{self.quantizer}"
            )
        if self.mode == "hybrid_affine_dct_grid_basis":
            ag = self.effective_affine_groups
            asz = self.effective_affine_grid
            selection_text = (
                f"_{self.selection}c{self.candidate_components}"
                if self.selection == "topk"
                else f"_prefix{self.prefix_components}topk{self.candidate_components}"
                if self.selection == "prefix_topk"
                else "_vector"
                if self.selection == "vector"
                else ""
            )
            return (
                f"hybridbasis_ag{ag}_as{asz}_g{self.groups}_s{self.grid}_k{self.coeffs}"
                f"{selection_text}_b{self.bits}_gr{self.gain_range:g}_br{self.bias_range:g}"
                f"_r{self.value_range:g}_sel{self.selector_bytes}{self.selector_bits_label}"
                f"_obj{self.hybrid_selection_objective}"
                f"_rl{self.hybrid_rate_lambda:g}{self.fidelity_label}"
                f"_sc{self.scale:g}_{self.codec}_{self.quantizer}{self.basis_range_label}"
                f"{f'_mu{self.mu:g}' if self.quantizer == 'mu_law' else ''}"
                f"{'_' + self.huffman_key if self.huffman_key else ''}"
            )
        return (
            f"dct_g{self.groups}_s{self.grid}_k{self.coeffs}_b{self.bits}"
            f"_r{self.value_range:g}_sc{self.scale:g}_{self.quantizer}"
        )

    def control_bytes(self) -> int:
        if self.codec == "huffman":
            raise ValueError("Huffman control bytes are image-dependent; use basis prior mean payload bytes")
        codec = UniformControlGridCode(bits=self.bits, value_range=self.value_range)
        if self.mode == "grid":
            return codec.encoded_num_bytes((self.groups, self.grid, self.grid))
        if self.mode == "dct":
            return codec.encoded_num_bytes((self.groups, self.coeffs))
        if self.mode == "basis":
            if self.selection == "vector":
                return codec.encoded_num_bytes((1,))
            if self.selection == "topk":
                value_codec = build_control_grid_code(
                    quantizer=self.quantizer,
                    bits=self.bits,
                    value_range=self.value_range,
                    mu=self.mu,
                )
                return SparseControlBasisCode(
                    candidate_components=int(self.candidate_components),
                    selected_components=int(self.coeffs),
                    value_codec=value_codec,
                ).encoded_compact_num_bytes()
            if self.selection == "prefix_topk":
                value_codec = build_control_grid_code(
                    quantizer=self.quantizer,
                    bits=self.bits,
                    value_range=self.value_range,
                    mu=self.mu,
                )
                return PrefixTopKControlBasisCode(
                    candidate_components=int(self.candidate_components),
                    prefix_components=int(self.prefix_components),
                    selected_components=int(self.coeffs),
                    value_codec=value_codec,
                ).encoded_compact_num_bytes()
            return codec.encoded_num_bytes((self.coeffs,))
        if self.mode == "affine":
            return self.affine_control_bytes()
        if self.mode == "affine_dct":
            dct_bytes = codec.encoded_num_bytes((self.groups, self.coeffs))
            return self.affine_control_bytes() + dct_bytes
        if self.mode == "affine_grid":
            grid_bytes = codec.encoded_num_bytes((self.groups, self.grid, self.grid))
            return self.affine_control_bytes() + grid_bytes
        if self.mode == "affine_basis":
            if self.selection == "vector":
                basis_bytes = codec.encoded_num_bytes((1,))
                return self.affine_control_bytes() + basis_bytes
            if self.selection == "topk":
                value_codec = build_control_grid_code(
                    quantizer=self.quantizer,
                    bits=self.bits,
                    value_range=self.value_range,
                    mu=self.mu,
                )
                basis_bytes = SparseControlBasisCode(
                    candidate_components=int(self.candidate_components),
                    selected_components=int(self.coeffs),
                    value_codec=value_codec,
                ).encoded_compact_num_bytes()
            elif self.selection == "prefix_topk":
                value_codec = build_control_grid_code(
                    quantizer=self.quantizer,
                    bits=self.bits,
                    value_range=self.value_range,
                    mu=self.mu,
                )
                basis_bytes = PrefixTopKControlBasisCode(
                    candidate_components=int(self.candidate_components),
                    prefix_components=int(self.prefix_components),
                    selected_components=int(self.coeffs),
                    value_codec=value_codec,
                ).encoded_compact_num_bytes()
            else:
                basis_bytes = codec.encoded_num_bytes((self.coeffs,))
            return self.affine_control_bytes() + basis_bytes
        if self.mode == "hybrid_affine_dct_grid":
            dct_bytes = codec.encoded_num_bytes((self.groups, self.coeffs))
            grid_bytes = codec.encoded_num_bytes((self.groups, self.grid, self.grid))
            affine_bytes = self.affine_control_bytes()
            affine_bits = self.affine_control_bits()
            return max(
                self.hybrid_payload_bytes(0, payload_bits=0),
                self.hybrid_payload_bytes(affine_bytes, payload_bits=affine_bits),
                self.hybrid_payload_bytes(
                    affine_bytes + dct_bytes,
                    payload_bits=affine_bits + codec.encoded_num_bits((self.groups, self.coeffs)),
                ),
                self.hybrid_payload_bytes(
                    affine_bytes + grid_bytes,
                    payload_bits=affine_bits + codec.encoded_num_bits((self.groups, self.grid, self.grid)),
                ),
            )
        if self.mode == "hybrid_affine_dct_grid_basis":
            dct_bytes = codec.encoded_num_bytes((self.groups, self.coeffs))
            grid_bytes = codec.encoded_num_bytes((self.groups, self.grid, self.grid))
            basis_bits: int | None = None
            if self.selection == "vector":
                basis_bytes = codec.encoded_num_bytes((1,))
                basis_bits = codec.encoded_num_bits((1,))
            elif self.selection == "topk":
                value_codec = build_control_grid_code(
                    quantizer=self.quantizer,
                    bits=self.bits,
                    value_range=self.value_range,
                    mu=self.mu,
                )
                sparse_code = SparseControlBasisCode(
                    candidate_components=int(self.candidate_components),
                    selected_components=int(self.coeffs),
                    value_codec=value_codec,
                )
                basis_bytes = sparse_code.encoded_compact_num_bytes()
                basis_bits = sparse_code.encoded_compact_num_bits()
            elif self.selection == "prefix_topk":
                value_codec = build_control_grid_code(
                    quantizer=self.quantizer,
                    bits=self.bits,
                    value_range=self.value_range,
                    mu=self.mu,
                )
                prefix_topk_code = PrefixTopKControlBasisCode(
                    candidate_components=int(self.candidate_components),
                    prefix_components=int(self.prefix_components),
                    selected_components=int(self.coeffs),
                    value_codec=value_codec,
                )
                basis_bytes = prefix_topk_code.encoded_compact_num_bytes()
                basis_bits = prefix_topk_code.encoded_compact_num_bits()
            else:
                basis_bytes = codec.encoded_num_bytes((self.coeffs,))
                basis_bits = codec.encoded_num_bits((self.coeffs,))
            affine_bytes = self.affine_control_bytes()
            affine_bits = self.affine_control_bits()
            return max(
                self.hybrid_payload_bytes(0, payload_bits=0),
                self.hybrid_payload_bytes(affine_bytes, payload_bits=affine_bits),
                self.hybrid_payload_bytes(
                    affine_bytes + dct_bytes,
                    payload_bits=affine_bits + codec.encoded_num_bits((self.groups, self.coeffs)),
                ),
                self.hybrid_payload_bytes(
                    affine_bytes + grid_bytes,
                    payload_bits=affine_bits + codec.encoded_num_bits((self.groups, self.grid, self.grid)),
                ),
                self.hybrid_payload_bytes(
                    affine_bytes + basis_bytes,
                    payload_bits=None if basis_bits is None else affine_bits + basis_bits,
                ),
            )
        raise ValueError(f"unknown mode: {self.mode}")

    def control_bpp(self, height: int, width: int) -> float:
        return CoSERBitstream.bytes_to_bpp(self.control_bytes(), height, width)

    @property
    def effective_affine_groups(self) -> int:
        return int(self.affine_groups or self.groups)

    @property
    def effective_affine_grid(self) -> int:
        return int(self.affine_grid or self.grid)

    def affine_control_bytes(self) -> int:
        gain_codec = UniformControlGridCode(bits=self.bits, value_range=self.gain_range)
        bias_codec = UniformControlGridCode(bits=self.bits, value_range=self.bias_range)
        shape = (self.effective_affine_groups, self.effective_affine_grid, self.effective_affine_grid)
        return gain_codec.encoded_num_bytes(shape) + bias_codec.encoded_num_bytes(shape)

    @property
    def selector_bits_label(self) -> str:
        return f"_selbits{self.selector_bits}" if self.selector_bits else ""

    @property
    def fidelity_label(self) -> str:
        if self.hybrid_fidelity_lambda <= 0:
            return ""
        return f"_f{self.hybrid_fidelity_metric}{self.hybrid_fidelity_lambda:g}"

    @property
    def basis_range_label(self) -> str:
        if self.basis_range_mode == "global":
            return ""
        return f"_{self.basis_range_mode}"

    def affine_control_bits(self) -> int:
        shape = (self.effective_affine_groups, self.effective_affine_grid, self.effective_affine_grid)
        return 2 * int(self.bits) * int(self.effective_affine_groups) * int(self.effective_affine_grid) ** 2

    def hybrid_payload_bytes(self, payload_bytes: float | int, *, payload_bits: int | None = None) -> int:
        payload_bytes_int = int(math.ceil(float(payload_bytes)))
        if payload_bytes_int < 0:
            raise ValueError("payload_bytes must be non-negative")
        if self.selector_bits <= 0 or payload_bits is None:
            fallback_selector_bytes = int(self.selector_bytes)
            if self.selector_bits > 0:
                fallback_selector_bytes = max(fallback_selector_bytes, (int(self.selector_bits) + 7) // 8)
            return payload_bytes_int + fallback_selector_bytes
        if payload_bits < 0:
            raise ValueError("payload_bits must be non-negative")
        return max(payload_bytes_int, (int(payload_bits) + int(self.selector_bits) + 7) // 8)


def setting_to_text(setting: ControlSetting) -> str:
    parts = [
        f"mode={setting.mode}",
        f"groups={setting.groups}",
        f"grid={setting.grid}",
        f"bits={setting.bits}",
    ]
    if setting.mode in {
        "dct",
        "basis",
        "affine_dct",
        "affine_basis",
        "hybrid_affine_dct_grid",
        "hybrid_affine_dct_grid_basis",
    }:
        parts.append(f"coeffs={setting.coeffs}")
    if setting.mode in {
        "affine",
        "affine_dct",
        "affine_grid",
        "affine_basis",
        "hybrid_affine_dct_grid",
        "hybrid_affine_dct_grid_basis",
    }:
        if setting.affine_groups:
            parts.append(f"affine_groups={setting.affine_groups}")
        if setting.affine_grid:
            parts.append(f"affine_grid={setting.affine_grid}")
        parts.extend(
            [
                f"gain_range={setting.gain_range:g}",
                f"bias_range={setting.bias_range:g}",
            ]
        )
    if setting.mode in {"hybrid_affine_dct_grid", "hybrid_affine_dct_grid_basis"}:
        parts.append(f"selector_bytes={setting.selector_bytes}")
        if setting.selector_bits:
            parts.append(f"selector_bits={setting.selector_bits}")
        parts.append(f"objective={setting.hybrid_selection_objective}")
        parts.append(f"rd_lambda={setting.hybrid_rate_lambda:g}")
        if setting.hybrid_fidelity_lambda > 0:
            parts.append(f"fidelity_lambda={setting.hybrid_fidelity_lambda:g}")
            parts.append(f"fidelity_metric={setting.hybrid_fidelity_metric}")
    if setting.mode not in {"affine"}:
        parts.append(f"range={setting.value_range:g}")
    parts.append(f"scale={setting.scale:g}")
    if setting.mode in {"basis", "affine_basis", "hybrid_affine_dct_grid_basis"}:
        parts.append(f"codec={setting.codec}")
        if setting.huffman_key:
            parts.append(f"huffman_key={setting.huffman_key}")
        if setting.basis_range_mode != "global":
            parts.append(f"basis_range_mode={setting.basis_range_mode}")
            parts.append(f"basis_range_floor={setting.basis_range_floor:g}")
        parts.append(f"selection={setting.selection}")
        if setting.selection == "topk":
            parts.append(f"candidates={setting.candidate_components}")
        if setting.selection == "prefix_topk":
            parts.append(f"prefix={setting.prefix_components}")
            parts.append(f"candidates={setting.candidate_components}")
    parts.append(f"quantizer={setting.quantizer}")
    if setting.quantizer == "mu_law":
        parts.append(f"mu={setting.mu:g}")
    return ",".join(parts)


def setting_text_from_row(row: dict[str, object]) -> str:
    explicit = row.get("setting_text")
    if isinstance(explicit, str) and explicit:
        return explicit
    setting = row.get("setting")
    if isinstance(setting, str) and setting:
        return setting
    if isinstance(setting, dict):
        return setting_to_text(
            ControlSetting(
                mode=str(setting.get("mode", "dct")),
                groups=int(setting["groups"]),
                grid=int(setting["grid"]),
                bits=int(setting.get("bits", 4)),
                value_range=float(setting.get("value_range", setting.get("range", 0.25))),
                scale=float(setting.get("scale", 1.0)),
                gain_range=float(setting.get("gain_range", setting.get("affine_gain_range", 1.0))),
                bias_range=float(setting.get("bias_range", setting.get("affine_bias_range", 0.25))),
                affine_groups=int(setting.get("affine_groups", setting.get("control_affine_groups", 0))),
                affine_grid=int(setting.get("affine_grid", setting.get("control_affine_grid_size", 0))),
                coeffs=int(setting.get("coeffs", 0)),
                codec=str(setting.get("codec", "fixed_bits")),
                huffman_key=str(setting.get("huffman_key", "")),
                quantizer=str(setting.get("quantizer", "uniform")),
                mu=float(setting.get("mu", 16.0)),
                selection=str(setting.get("selection", "prefix")),
                candidate_components=int(setting.get("candidate_components", setting.get("candidates", 0))),
                prefix_components=int(setting.get("prefix_components", setting.get("prefix", 0))),
                basis_range_mode=str(setting.get("basis_range_mode", "global")),
                basis_range_floor=float(setting.get("basis_range_floor", 1.0e-6)),
                selector_bytes=int(setting.get("selector_bytes", 1)),
                selector_bits=int(setting.get("selector_bits", 0)),
                hybrid_rate_lambda=float(setting.get("hybrid_rate_lambda", setting.get("rd_lambda", 0.0))),
                hybrid_selection_objective=str(
                    setting.get("hybrid_selection_objective", setting.get("objective", "condition_l1"))
                ),
                hybrid_fidelity_lambda=float(
                    setting.get("hybrid_fidelity_lambda", setting.get("fidelity_lambda", 0.0))
                ),
                hybrid_fidelity_metric=str(
                    setting.get("hybrid_fidelity_metric", setting.get("fidelity_metric", "image_l1"))
                ),
            )
        )
    return ""


def normalize_setting_row(row: dict[str, object]) -> dict[str, object]:
    normalized = dict(row)
    setting_text = setting_text_from_row(normalized)
    if not setting_text:
        return normalized
    setting = parse_setting(setting_text)
    normalized["setting"] = setting_text
    normalized.setdefault("mode", setting.mode)
    normalized.setdefault("codec", setting.codec)
    normalized.setdefault("quantizer", setting.quantizer)
    normalized.setdefault("selection", setting.selection)
    normalized.setdefault("components", setting.coeffs)
    normalized.setdefault(
        "candidate_components",
        setting.candidate_components if setting.candidate_components > 0 else setting.coeffs,
    )
    normalized.setdefault("prefix_components", setting.prefix_components)
    normalized.setdefault("control_groups", setting.groups)
    normalized.setdefault("control_grid_size", setting.grid)
    normalized.setdefault("control_scale", setting.scale)
    if setting.mode in {
        "affine",
        "affine_dct",
        "affine_grid",
        "affine_basis",
        "hybrid_affine_dct_grid",
        "hybrid_affine_dct_grid_basis",
    }:
        normalized.setdefault("control_affine_groups", setting.effective_affine_groups)
        normalized.setdefault("control_affine_grid_size", setting.effective_affine_grid)
        normalized.setdefault("control_affine_gain_range", setting.gain_range)
        normalized.setdefault("control_affine_bias_range", setting.bias_range)
    if setting.mode != "affine":
        normalized.setdefault("control_range", setting.value_range)
    source_metadata = normalized.get("source_setting_metadata")
    if isinstance(source_metadata, dict):
        for key, value in source_metadata.items():
            normalized.setdefault(key, value)
    return normalized


def override_setting_mode(
    row: dict[str, object],
    *,
    mode: str,
    affine_groups: int = 0,
    affine_grid: int = 0,
    gain_range: float = 1.0,
    bias_range: float = 0.25,
    selector_bytes: int = 1,
    selector_bits: int = 0,
    hybrid_selection_objective: str = "condition_l1",
    hybrid_rate_lambda: float = 0.0,
    hybrid_fidelity_lambda: float = 0.0,
    hybrid_fidelity_metric: str = "image_l1",
) -> dict[str, object]:
    if not mode:
        return row
    if mode not in {"affine_basis", "hybrid_affine_dct_grid_basis"}:
        raise ValueError("override mode currently supports only affine_basis or hybrid_affine_dct_grid_basis")
    setting = parse_setting(str(row["setting"]))
    if setting.mode != "basis":
        raise ValueError("--override-mode expects source settings with mode=basis")
    updated = replace(
        setting,
        mode=mode,
        affine_groups=int(affine_groups),
        affine_grid=int(affine_grid),
        gain_range=float(gain_range),
        bias_range=float(bias_range),
        selector_bytes=int(selector_bytes),
        selector_bits=int(selector_bits),
        hybrid_selection_objective=str(hybrid_selection_objective),
        hybrid_rate_lambda=float(hybrid_rate_lambda),
        hybrid_fidelity_lambda=float(hybrid_fidelity_lambda),
        hybrid_fidelity_metric=str(hybrid_fidelity_metric),
    )
    out = dict(row)
    out["setting"] = setting_to_text(updated)
    return normalize_setting_row(out)


DEFAULT_SETTINGS = (
    "mode=dct,groups=8,grid=8,coeffs=2,bits=4,range=0.25,scale=1.0",
    "mode=dct,groups=8,grid=8,coeffs=4,bits=4,range=0.25,scale=1.0",
    "mode=dct,groups=16,grid=8,coeffs=4,bits=4,range=0.25,scale=1.0",
    "mode=grid,groups=4,grid=4,bits=4,range=0.25,scale=1.0",
    "mode=grid,groups=8,grid=4,bits=4,range=0.25,scale=1.0",
    "mode=grid,groups=16,grid=4,bits=4,range=0.25,scale=1.0",
    "mode=affine,groups=8,grid=1,bits=4,gain_range=1.0,bias_range=0.25,scale=1.0",
    "mode=affine,groups=16,grid=1,bits=4,gain_range=1.0,bias_range=0.25,scale=1.0",
    "mode=affine,groups=8,grid=2,bits=4,gain_range=1.0,bias_range=0.25,scale=1.0",
    "mode=affine_dct,groups=8,grid=4,coeffs=4,bits=4,affine_groups=8,affine_grid=1,range=0.25,gain_range=1.0,bias_range=0.25,scale=1.0",
    "mode=affine_grid,groups=8,grid=2,bits=4,affine_groups=8,affine_grid=1,range=0.25,gain_range=1.0,bias_range=0.25,scale=1.0",
    "mode=hybrid_affine_dct_grid,groups=8,grid=2,coeffs=4,bits=4,affine_groups=8,affine_grid=1,range=0.25,gain_range=1.0,bias_range=0.25,selector_bytes=1,objective=condition_l1,rd_lambda=0.0,scale=1.0",
    "mode=hybrid_affine_dct_grid_basis,groups=16,grid=8,coeffs=8,bits=4,affine_groups=8,affine_grid=1,range=0.5,gain_range=1.0,bias_range=0.25,selector_bytes=1,objective=condition_l1,rd_lambda=0.0,scale=1.0,codec=fixed_bits,quantizer=uniform",
    "mode=affine_basis,groups=16,grid=8,coeffs=8,bits=4,affine_groups=8,affine_grid=1,range=0.5,gain_range=1.0,bias_range=0.25,scale=1.0,codec=fixed_bits,quantizer=uniform",
)


SETTING_ROW_METADATA_EXCLUDE = {
    "setting",
    "setting_text",
    "command",
    "summary",
    "run_name",
    "control_basis",
    "control_bytes",
    "control_bpp",
    "control_bytes_note",
    "source_setting_metadata",
}


def run_gpu_preflight(python: str) -> None:
    result = subprocess.run(
        [python, "scripts/check_gpu_ready.py", "--min-devices", "1"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)


def gpu_preflight_command(python: str) -> str:
    return shlex.join([python, "scripts/check_gpu_ready.py", "--min-devices", "1"])


def parse_setting(text: str) -> ControlSetting:
    raw: dict[str, str] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"setting item must be key=value: {item}")
        key, value = item.split("=", 1)
        raw[key.strip()] = value.strip()
    mode = raw.get("mode", "dct")
    if mode not in {
        "grid",
        "dct",
        "basis",
        "affine",
        "affine_dct",
        "affine_grid",
        "affine_basis",
        "hybrid_affine_dct_grid",
        "hybrid_affine_dct_grid_basis",
    }:
        raise ValueError(
            "mode must be grid, dct, basis, affine, affine_dct, affine_grid, affine_basis, "
            "hybrid_affine_dct_grid, or hybrid_affine_dct_grid_basis"
        )
    groups = int(raw["groups"])
    grid = int(raw["grid"])
    bits = int(raw.get("bits", "4"))
    value_range = float(raw.get("range", raw.get("value_range", "0.25")))
    gain_range = float(raw.get("gain_range", raw.get("affine_gain_range", "1.0")))
    bias_range = float(raw.get("bias_range", raw.get("affine_bias_range", str(value_range))))
    affine_groups = int(raw.get("affine_groups", raw.get("control_affine_groups", "0")))
    affine_grid = int(raw.get("affine_grid", raw.get("control_affine_grid_size", "0")))
    scale = float(raw.get("scale", "1.0"))
    coeffs = int(raw.get("coeffs", "0"))
    codec = raw.get("codec", "fixed_bits")
    if codec not in {"fixed_bits", "huffman"}:
        raise ValueError("codec must be fixed_bits or huffman")
    huffman_key = raw.get("huffman_key", raw.get("hkey", ""))
    quantizer = raw.get("quantizer", "uniform")
    if quantizer not in {"uniform", "mu_law"}:
        raise ValueError("quantizer must be uniform or mu_law")
    mu = float(raw.get("mu", "16.0"))
    selection = raw.get("selection", "prefix")
    if selection not in {"prefix", "topk", "vector", "prefix_topk"}:
        raise ValueError("selection must be prefix, topk, vector, or prefix_topk")
    candidate_components = int(raw.get("candidates", raw.get("candidate_components", "0")))
    prefix_components = int(raw.get("prefix", raw.get("prefix_components", "0")))
    basis_range_mode = raw.get("basis_range_mode", raw.get("control_basis_range_mode", "global"))
    if basis_range_mode not in {"global", "component_p95", "component_p99", "component_codebook"}:
        raise ValueError("basis_range_mode must be global, component_p95, component_p99, or component_codebook")
    basis_range_floor = float(raw.get("basis_range_floor", raw.get("control_basis_range_floor", "1e-6")))
    selector_bytes = int(raw.get("selector_bytes", "1"))
    selector_bits = int(raw.get("selector_bits", raw.get("control_hybrid_selector_bits", "0")))
    hybrid_rate_lambda = float(raw.get("rd_lambda", raw.get("hybrid_rate_lambda", "0.0")))
    hybrid_selection_objective = raw.get(
        "objective",
        raw.get("hybrid_selection_objective", "condition_l1"),
    )
    hybrid_fidelity_lambda = float(raw.get("fidelity_lambda", raw.get("hybrid_fidelity_lambda", "0.0")))
    hybrid_fidelity_metric = raw.get("fidelity_metric", raw.get("hybrid_fidelity_metric", "image_l1"))
    if hybrid_selection_objective not in {"condition_l1", "image_l1", "image_mse", "lpips_alex", "dists"}:
        raise ValueError("objective must be condition_l1, image_l1, image_mse, lpips_alex, or dists")
    if hybrid_fidelity_metric not in {"image_l1", "image_mse"}:
        raise ValueError("fidelity_metric must be image_l1 or image_mse")
    if mu <= 0:
        raise ValueError("mu must be positive")
    if basis_range_floor <= 0:
        raise ValueError("basis_range_floor must be positive")
    if gain_range <= 0:
        raise ValueError("gain_range must be positive")
    if bias_range <= 0:
        raise ValueError("bias_range must be positive")
    if selector_bytes < 0:
        raise ValueError("selector_bytes must be non-negative")
    if selector_bits < 0:
        raise ValueError("selector_bits must be non-negative")
    if prefix_components < 0:
        raise ValueError("prefix must be non-negative")
    if hybrid_rate_lambda < 0:
        raise ValueError("rd_lambda must be non-negative")
    if hybrid_fidelity_lambda < 0:
        raise ValueError("fidelity_lambda must be non-negative")
    if codec == "huffman" and mode not in {"basis", "affine_basis", "hybrid_affine_dct_grid_basis"}:
        raise ValueError("codec=huffman is currently supported only for basis modes")
    if basis_range_mode != "global" and mode not in {"basis", "affine_basis", "hybrid_affine_dct_grid_basis"}:
        raise ValueError("component basis ranges are supported only for basis modes")
    if basis_range_mode in {"component_p95", "component_p99"} and codec != "fixed_bits":
        raise ValueError("component p95/p99 basis ranges currently require codec=fixed_bits")
    if mode in {
        "dct",
        "basis",
        "affine_dct",
        "affine_basis",
        "hybrid_affine_dct_grid",
        "hybrid_affine_dct_grid_basis",
    } and coeffs <= 0:
        raise ValueError(f"{mode} mode requires coeffs > 0")
    if affine_groups < 0:
        raise ValueError("affine_groups must be non-negative")
    if affine_grid < 0:
        raise ValueError("affine_grid must be non-negative")
    if mode in {"basis", "affine_basis", "hybrid_affine_dct_grid_basis"}:
        if selection == "prefix":
            candidate_components = 0
            prefix_components = 0
        elif selection == "vector":
            candidate_components = 0
            prefix_components = 0
        elif candidate_components <= 0:
            raise ValueError("selection=topk/prefix_topk requires candidates > 0")
        elif selection == "topk":
            prefix_components = 0
            if coeffs > candidate_components:
                raise ValueError("coeffs must be <= candidates for selection=topk")
        elif selection == "prefix_topk":
            if codec == "huffman":
                raise ValueError("selection=prefix_topk currently supports codec=fixed_bits only")
            if prefix_components <= 0:
                raise ValueError("selection=prefix_topk requires prefix > 0")
            if prefix_components >= candidate_components:
                raise ValueError("prefix must be < candidates for selection=prefix_topk")
            if coeffs > candidate_components - prefix_components:
                raise ValueError("coeffs must be <= candidates-prefix for selection=prefix_topk")
    if mode in {"grid", "affine", "affine_grid"}:
        coeffs = 0
        selection = "prefix"
        candidate_components = 0
        prefix_components = 0
    return ControlSetting(
        mode=mode,
        groups=groups,
        grid=grid,
        bits=bits,
        value_range=value_range,
        scale=scale,
        gain_range=gain_range,
        bias_range=bias_range,
        affine_groups=affine_groups,
        affine_grid=affine_grid,
        coeffs=coeffs,
        codec=codec,
        huffman_key=huffman_key,
        quantizer=quantizer,
        mu=mu,
        selection=selection,
        candidate_components=candidate_components,
        prefix_components=prefix_components,
        basis_range_mode=basis_range_mode,
        basis_range_floor=basis_range_floor,
        selector_bytes=selector_bytes,
        selector_bits=selector_bits,
        hybrid_rate_lambda=hybrid_rate_lambda,
        hybrid_selection_objective=hybrid_selection_objective,
        hybrid_fidelity_lambda=hybrid_fidelity_lambda,
        hybrid_fidelity_metric=hybrid_fidelity_metric,
    )


def load_control_priors(path: str) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    if not path:
        raise ValueError("--control-basis is required for codec=huffman")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    priors = payload.get("control_huffman_priors", {})
    vector_priors = payload.get("coefficient_vector_codebooks", {})
    sparse_priors = payload.get("sparse_topk_control_priors", {})
    if not isinstance(priors, dict):
        raise ValueError("control basis checkpoint contains invalid control_huffman_priors")
    if not isinstance(vector_priors, dict):
        raise ValueError("control basis checkpoint contains invalid coefficient_vector_codebooks")
    if not isinstance(sparse_priors, dict):
        raise ValueError("control basis checkpoint contains invalid sparse_topk_control_priors")
    merged_priors = dict(priors)
    merged_priors.update(vector_priors)
    return merged_priors, sparse_priors


def planned_control_bytes(
    setting: ControlSetting,
    *,
    huffman_priors: dict[str, dict[str, object]],
    sparse_topk_priors: dict[str, dict[str, object]],
) -> float:
    if setting.codec != "huffman":
        return float(setting.control_bytes())
    affine_bytes = float(setting.affine_control_bytes()) if setting.mode in {
        "affine_basis",
        "hybrid_affine_dct_grid_basis",
    } else 0.0
    if setting.selection == "topk":
        if not setting.huffman_key:
            matches = [
                key
                for key, prior in sparse_topk_priors.items()
                if isinstance(prior, dict)
                and int(prior.get("candidate_components", -1)) == int(setting.candidate_components)
                and int(prior.get("selected_components", -1)) == int(setting.coeffs)
            ]
            if len(matches) != 1:
                raise ValueError("huffman_key is required when sparse top-k priors are ambiguous")
            key = matches[0]
        else:
            key = setting.huffman_key
        prior = sparse_topk_priors.get(key)
        if not isinstance(prior, dict):
            raise ValueError(f"unknown sparse top-k Huffman prior key: {key}")
        basis_payload_bytes = float(prior["mean_payload_bytes"])
    elif setting.selection == "vector":
        key = setting.huffman_key or f"vq_k{setting.coeffs}_b{setting.bits}"
        prior = huffman_priors.get(key)
        if not isinstance(prior, dict):
            raise ValueError(f"unknown vector codebook prior key: {key}")
        basis_payload_bytes = float(prior.get("huffman_mean_payload_bytes", prior.get("mean_payload_bytes", 0.0)))
    else:
        if not setting.huffman_key:
            if len(huffman_priors) != 1:
                raise ValueError("huffman_key is required when the basis contains multiple Huffman priors")
            key = next(iter(huffman_priors))
        else:
            key = setting.huffman_key
        prior = huffman_priors.get(key)
        if not isinstance(prior, dict):
            raise ValueError(f"unknown Huffman prior key: {key}")
        prefix_means = prior.get("prefix_mean_payload_bytes", [])
        if isinstance(prefix_means, list) and len(prefix_means) >= setting.coeffs:
            basis_payload_bytes = float(prefix_means[setting.coeffs - 1])
        else:
            basis_payload_bytes = float(prior["mean_payload_bytes"])
    if setting.mode == "hybrid_affine_dct_grid_basis":
        codec = UniformControlGridCode(bits=setting.bits, value_range=setting.value_range)
        dct_bytes = float(codec.encoded_num_bytes((setting.groups, setting.coeffs)))
        grid_bytes = float(codec.encoded_num_bytes((setting.groups, setting.grid, setting.grid)))
        affine_bits = setting.affine_control_bits()
        return float(
            max(
                setting.hybrid_payload_bytes(0, payload_bits=0),
                setting.hybrid_payload_bytes(affine_bytes, payload_bits=affine_bits),
                setting.hybrid_payload_bytes(
                    affine_bytes + dct_bytes,
                    payload_bits=affine_bits + codec.encoded_num_bits((setting.groups, setting.coeffs)),
                ),
                setting.hybrid_payload_bytes(
                    affine_bytes + grid_bytes,
                    payload_bits=affine_bits + codec.encoded_num_bits((setting.groups, setting.grid, setting.grid)),
                ),
                setting.hybrid_payload_bytes(affine_bytes + basis_payload_bytes, payload_bits=None),
            )
        )
    return affine_bytes + basis_payload_bytes


def load_setting_texts_from_json(
    paths: list[Path],
    *,
    max_control_bpp: float = 0.0,
    min_basis_retained_energy: float = 0.0,
    max_basis_residual_energy: float = 1.0,
    max_quantization_rmse: float = 0.0,
    max_clipped_fraction: float = 1.0,
    include_codecs: set[str] | None = None,
    include_quantizers: set[str] | None = None,
    include_quantiles: set[str] | None = None,
    include_components: set[int] | None = None,
    include_candidate_components: set[int] | None = None,
    include_selections: set[str] | None = None,
    include_modes: set[str] | None = None,
    sort_by: str = "input",
    max_settings: int = 0,
) -> list[str]:
    return [
        str(row["setting"])
        for row in load_setting_rows_from_json(
            paths,
            max_control_bpp=max_control_bpp,
            min_basis_retained_energy=min_basis_retained_energy,
            max_basis_residual_energy=max_basis_residual_energy,
            max_quantization_rmse=max_quantization_rmse,
            max_clipped_fraction=max_clipped_fraction,
            include_codecs=include_codecs,
            include_quantizers=include_quantizers,
            include_quantiles=include_quantiles,
            include_components=include_components,
            include_candidate_components=include_candidate_components,
            include_selections=include_selections,
            include_modes=include_modes,
            sort_by=sort_by,
            max_settings=max_settings,
        )
    ]


def optional_row_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric when present") from exc


def sort_setting_rows(rows: list[dict[str, object]], *, sort_by: str) -> list[dict[str, object]]:
    if sort_by == "input":
        return rows
    if sort_by == "control_bpp":
        return sorted(rows, key=lambda row: optional_row_float(row, "control_bpp") or 0.0)
    if sort_by == "retained_energy":
        return sorted(rows, key=lambda row: -(optional_row_float(row, "basis_retained_energy_fraction_mean") or 0.0))
    if sort_by == "residual_energy":
        return sorted(rows, key=lambda row: optional_row_float(row, "basis_residual_energy_fraction_mean") or 1.0)
    if sort_by == "retained_per_bpp":
        return sorted(
            rows,
            key=lambda row: -(
                (optional_row_float(row, "basis_retained_energy_fraction_mean") or 0.0)
                / max(optional_row_float(row, "control_bpp") or 0.0, 1.0e-12)
            ),
        )
    if sort_by == "rmse":
        return sorted(rows, key=lambda row: optional_row_float(row, "quantization_rmse") or 0.0)
    raise ValueError("sort_by must be input, control_bpp, retained_energy, residual_energy, retained_per_bpp, or rmse")


def load_setting_rows_from_json(
    paths: list[Path],
    *,
    max_control_bpp: float = 0.0,
    min_basis_retained_energy: float = 0.0,
    max_basis_residual_energy: float = 1.0,
    max_quantization_rmse: float = 0.0,
    max_clipped_fraction: float = 1.0,
    include_codecs: set[str] | None = None,
    include_quantizers: set[str] | None = None,
    include_quantiles: set[str] | None = None,
    include_components: set[int] | None = None,
    include_candidate_components: set[int] | None = None,
    include_selections: set[str] | None = None,
    include_modes: set[str] | None = None,
    sort_by: str = "input",
    max_settings: int = 0,
) -> list[dict[str, object]]:
    setting_rows: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"settings JSON must be an object: {path}")
        rows = payload.get("settings")
        if not isinstance(rows, list):
            raise ValueError(f"settings JSON must contain a settings list: {path}")
        for row in rows:
            if not isinstance(row, dict):
                continue
            row = normalize_setting_row(row)
            if max_control_bpp > 0 and float(row.get("control_bpp", 0.0)) > max_control_bpp:
                continue
            retained = optional_row_float(row, "basis_retained_energy_fraction_mean")
            if min_basis_retained_energy > 0.0 and (retained is None or retained < min_basis_retained_energy):
                continue
            residual = optional_row_float(row, "basis_residual_energy_fraction_mean")
            if max_basis_residual_energy < 1.0 and (residual is None or residual > max_basis_residual_energy):
                continue
            rmse = optional_row_float(row, "quantization_rmse")
            if max_quantization_rmse > 0.0 and (rmse is None or rmse > max_quantization_rmse):
                continue
            clipped = optional_row_float(row, "clipped_fraction")
            if max_clipped_fraction < 1.0 and (clipped is None or clipped > max_clipped_fraction):
                continue
            if include_codecs and str(row.get("codec", "")) not in include_codecs:
                continue
            if include_quantizers and str(row.get("quantizer", "")) not in include_quantizers:
                continue
            if include_quantiles and str(row.get("quantile", "")) not in include_quantiles:
                continue
            if include_modes and str(row.get("mode", "")) not in include_modes:
                continue
            if include_components and int(row.get("components", -1)) not in include_components:
                continue
            if include_candidate_components and int(row.get("candidate_components", -1)) not in include_candidate_components:
                continue
            if include_selections and str(row.get("selection", "")) not in include_selections:
                continue
            if not isinstance(row.get("setting"), str) or not row["setting"]:
                continue
            setting_rows.append(dict(row))
    setting_rows = sort_setting_rows(setting_rows, sort_by=sort_by)
    if max_settings > 0:
        return setting_rows[:max_settings]
    return setting_rows


def build_eval_command(
    *,
    python: str,
    checkpoint: str,
    manifest: str,
    per_image_metrics: str,
    control_basis: str,
    run_name: str,
    crop_size: int,
    limit: int,
    batch_size: int,
    num_workers: int,
    setting: ControlSetting,
    extra_args: list[str],
) -> list[str]:
    command = [
        python,
        "scripts/eval_stage4_cod_lite_adapter.py",
        "--checkpoint",
        checkpoint,
        "--manifest",
        manifest,
        "--per-image-metrics",
        per_image_metrics,
        "--run-name",
        run_name,
        "--crop-size",
        str(crop_size),
        "--limit",
        str(limit),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--blend-alpha",
        "1.0",
        "--counted-control-mode",
        setting.eval_mode,
        "--control-groups",
        str(setting.groups),
        "--control-grid-size",
        str(setting.grid),
        "--control-dct-coeffs-per-group",
        str(max(setting.coeffs, 1)),
        "--control-codec",
        setting.codec,
        "--control-quantizer",
        setting.quantizer,
        "--control-mu",
        str(setting.mu),
        "--control-bits",
        str(setting.bits),
        "--control-range",
        str(setting.value_range),
        "--control-affine-groups",
        str(setting.effective_affine_groups),
        "--control-affine-grid-size",
        str(setting.effective_affine_grid),
        "--control-affine-gain-range",
        str(setting.gain_range),
        "--control-affine-bias-range",
        str(setting.bias_range),
        "--control-hybrid-selector-bytes",
        str(setting.selector_bytes),
        "--control-hybrid-selector-bits",
        str(setting.selector_bits),
        "--control-hybrid-rate-lambda",
        str(setting.hybrid_rate_lambda),
        "--control-hybrid-selection-objective",
        setting.hybrid_selection_objective,
        "--control-hybrid-fidelity-lambda",
        str(setting.hybrid_fidelity_lambda),
        "--control-hybrid-fidelity-metric",
        setting.hybrid_fidelity_metric,
        "--control-scale",
        str(setting.scale),
    ]
    if setting.mode in {"basis", "affine_basis", "hybrid_affine_dct_grid_basis"}:
        command.extend(
            [
                "--control-basis",
                control_basis,
                "--control-basis-components",
                str(setting.coeffs),
                "--control-basis-selection",
                setting.selection,
                "--control-basis-range-mode",
                setting.basis_range_mode,
                "--control-basis-range-floor",
                str(setting.basis_range_floor),
            ]
        )
        if setting.selection in {"topk", "prefix_topk"}:
            command.extend(["--control-basis-candidate-components", str(setting.candidate_components)])
        if setting.selection == "prefix_topk":
            command.extend(["--control-basis-prefix-components", str(setting.prefix_components)])
    if setting.codec == "huffman":
        command.extend(["--control-huffman-key", setting.huffman_key])
    command.extend(extra_args)
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--per-image-metrics", required=True)
    parser.add_argument("--control-basis", default="")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="Evaluation limit. Use 0 for the full manifest.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--setting", action="append", default=[])
    parser.add_argument("--settings-json", action="append", default=[])
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
    parser.add_argument("--override-mode", choices=("affine_basis", "hybrid_affine_dct_grid_basis"), default="")
    parser.add_argument("--override-affine-groups", type=int, default=0)
    parser.add_argument("--override-affine-grid-size", type=int, default=1)
    parser.add_argument("--override-affine-gain-range", type=float, default=1.0)
    parser.add_argument("--override-affine-bias-range", type=float, default=0.25)
    parser.add_argument("--override-hybrid-selector-bytes", type=int, default=1)
    parser.add_argument("--override-hybrid-selector-bits", type=int, default=0)
    parser.add_argument(
        "--override-hybrid-selection-objective",
        choices=("condition_l1", "image_l1", "image_mse", "lpips_alex", "dists"),
        default="condition_l1",
    )
    parser.add_argument("--override-hybrid-rate-lambda", type=float, default=0.0)
    parser.add_argument("--override-hybrid-fidelity-lambda", type=float, default=0.0)
    parser.add_argument(
        "--override-hybrid-fidelity-metric",
        choices=("image_l1", "image_mse"),
        default="image_l1",
    )
    parser.add_argument(
        "--sort-settings-by",
        choices=("input", "control_bpp", "retained_energy", "residual_energy", "retained_per_bpp", "rmse"),
        default="input",
    )
    parser.add_argument("--max-settings", type=int, default=0)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-gpu-preflight", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-sh", default="")
    args = parser.parse_args()
    if args.override_mode:
        if args.override_affine_groups <= 0:
            raise ValueError("--override-affine-groups must be positive with --override-mode")
        if args.override_affine_grid_size <= 0:
            raise ValueError("--override-affine-grid-size must be positive with --override-mode")
        if args.override_affine_gain_range <= 0:
            raise ValueError("--override-affine-gain-range must be positive with --override-mode")
        if args.override_affine_bias_range <= 0:
            raise ValueError("--override-affine-bias-range must be positive with --override-mode")
        if args.override_hybrid_selector_bytes < 0:
            raise ValueError("--override-hybrid-selector-bytes must be non-negative with --override-mode")
        if args.override_hybrid_selector_bits < 0:
            raise ValueError("--override-hybrid-selector-bits must be non-negative with --override-mode")
        if args.override_hybrid_rate_lambda < 0:
            raise ValueError("--override-hybrid-rate-lambda must be non-negative with --override-mode")
        if args.override_hybrid_fidelity_lambda < 0:
            raise ValueError("--override-hybrid-fidelity-lambda must be non-negative with --override-mode")

    setting_rows: list[dict[str, object]] = []
    if args.settings_json:
        setting_rows.extend(
            load_setting_rows_from_json(
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
                include_candidate_components=set(args.include_candidate_components)
                if args.include_candidate_components
                else None,
                include_selections=set(args.include_selection) if args.include_selection else None,
                include_modes=set(args.include_mode) if args.include_mode else None,
                sort_by=args.sort_settings_by,
                max_settings=args.max_settings,
            )
        )
    if args.override_mode:
        setting_rows = [
            override_setting_mode(
                row,
                mode=args.override_mode,
                affine_groups=args.override_affine_groups,
                affine_grid=args.override_affine_grid_size,
                gain_range=args.override_affine_gain_range,
                bias_range=args.override_affine_bias_range,
                selector_bytes=args.override_hybrid_selector_bytes,
                selector_bits=args.override_hybrid_selector_bits,
                hybrid_selection_objective=args.override_hybrid_selection_objective,
                hybrid_rate_lambda=args.override_hybrid_rate_lambda,
                hybrid_fidelity_lambda=args.override_hybrid_fidelity_lambda,
                hybrid_fidelity_metric=args.override_hybrid_fidelity_metric,
            )
            for row in setting_rows
        ]
    setting_rows.extend({"setting": item} for item in args.setting)
    if not setting_rows:
        setting_rows = [{"setting": item} for item in DEFAULT_SETTINGS]
    settings_with_metadata = [
        (
            parse_setting(str(row["setting"])),
            {key: value for key, value in row.items() if key not in SETTING_ROW_METADATA_EXCLUDE},
        )
        for row in setting_rows
    ]
    settings = [item[0] for item in settings_with_metadata]
    if any(setting.mode in {"basis", "affine_basis", "hybrid_affine_dct_grid_basis"} for setting in settings) and not args.control_basis:
        raise ValueError("--control-basis is required when any setting uses mode=basis")
    huffman_priors: dict[str, dict[str, object]] = {}
    sparse_topk_priors: dict[str, dict[str, object]] = {}
    if any(setting.codec == "huffman" for setting in settings):
        huffman_priors, sparse_topk_priors = load_control_priors(args.control_basis)
    if not args.dry_run and not args.skip_gpu_preflight:
        run_gpu_preflight(args.python)
    rows: list[dict[str, object]] = []
    for setting, setting_metadata in settings_with_metadata:
        run_name = f"{args.run_prefix}_{setting.label}"
        command = build_eval_command(
            python=args.python,
            checkpoint=args.checkpoint,
            manifest=args.manifest,
            per_image_metrics=args.per_image_metrics,
            control_basis=args.control_basis,
            run_name=run_name,
            crop_size=args.crop_size,
            limit=args.limit,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            setting=setting,
            extra_args=args.extra_arg,
        )
        control_bytes = planned_control_bytes(
            setting,
            huffman_priors=huffman_priors,
            sparse_topk_priors=sparse_topk_priors,
        )
        row = {
            "run_name": run_name,
            "setting": asdict(setting),
            "source_setting_metadata": {
                **setting_metadata,
                "counted_control_mode": setting.eval_mode,
                "control_grid_size": setting.grid,
                "control_groups": setting.groups,
                "control_range": setting.value_range,
                "control_affine_groups": setting.effective_affine_groups,
                "control_affine_grid_size": setting.effective_affine_grid,
                "control_affine_gain_range": setting.gain_range,
                "control_affine_bias_range": setting.bias_range,
                "control_hybrid_selector_bytes": setting.selector_bytes,
                "control_hybrid_selector_bits": setting.selector_bits,
                "control_hybrid_rate_lambda": setting.hybrid_rate_lambda,
                "control_hybrid_selection_objective": setting.hybrid_selection_objective,
                "control_hybrid_fidelity_lambda": setting.hybrid_fidelity_lambda,
                "control_hybrid_fidelity_metric": setting.hybrid_fidelity_metric,
                "control_basis_range_mode": setting.basis_range_mode,
                "control_basis_range_floor": setting.basis_range_floor,
                "control_basis_selection": setting.selection,
                "control_basis_components": setting.coeffs,
                "control_basis_candidate_components": setting.candidate_components,
                "control_basis_prefix_components": setting.prefix_components,
                "control_scale": setting.scale,
                "control_codec_type": setting.codec,
                "control_quantizer": setting.quantizer,
            },
            "control_bytes": control_bytes,
            "control_bpp": CoSERBitstream.bytes_to_bpp(control_bytes, args.crop_size, args.crop_size),
            "control_bytes_note": "mean train payload bytes" if setting.codec == "huffman" else "exact fixed payload bytes",
            "control_basis": args.control_basis
            if setting.mode in {"basis", "affine_basis", "hybrid_affine_dct_grid_basis"}
            else "",
            "summary": f"results/stage4_cod_lite_adapter_eval/{run_name}/summary.json",
            "command": command,
        }
        for key in (
            "basis_reconstruction_key",
            "basis_retained_energy_fraction_mean",
            "basis_retained_energy_fraction_p50",
            "basis_residual_energy_fraction_mean",
            "basis_residual_l2_mean",
            "quantization_mae",
            "quantization_rmse",
            "clipped_fraction",
        ):
            if key in setting_metadata:
                row[key] = setting_metadata[key]
        rows.append(row)
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)

    payload = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "per_image_metrics": args.per_image_metrics,
        "control_basis": args.control_basis,
        "run_prefix": args.run_prefix,
        "crop_size": args.crop_size,
        "limit": args.limit,
        "settings_json": args.settings_json,
        "settings_json_filters": {
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
            "override_mode": args.override_mode,
            "override_affine_groups": args.override_affine_groups,
            "override_affine_grid_size": args.override_affine_grid_size,
            "override_affine_gain_range": args.override_affine_gain_range,
            "override_affine_bias_range": args.override_affine_bias_range,
            "override_hybrid_selector_bytes": args.override_hybrid_selector_bytes,
            "override_hybrid_selector_bits": args.override_hybrid_selector_bits,
            "override_hybrid_selection_objective": args.override_hybrid_selection_objective,
            "override_hybrid_rate_lambda": args.override_hybrid_rate_lambda,
            "override_hybrid_fidelity_lambda": args.override_hybrid_fidelity_lambda,
            "override_hybrid_fidelity_metric": args.override_hybrid_fidelity_metric,
            "sort_settings_by": args.sort_settings_by,
            "max_settings": args.max_settings,
        },
        "settings": rows,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_sh:
        sh_path = Path(args.output_sh)
        sh_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "cd /workspace/CoSER-DiC",
            "",
            gpu_preflight_command(args.python),
            "",
        ]
        lines.extend(shlex.join(row["command"]) for row in rows)
        lines.append("")
        sh_path.write_text("\n".join(lines))
    if not args.output_json:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
