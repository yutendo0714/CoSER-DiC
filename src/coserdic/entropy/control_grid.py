from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .semantic_tokens import decode_semantic_tokens, encode_semantic_tokens
from .semantic_tokens import semantic_bits_per_token
from .static_huffman import StaticHuffmanCode


def _validate_range_tensor(value_ranges: torch.Tensor | list[float] | tuple[float, ...]) -> torch.Tensor:
    ranges = torch.as_tensor(value_ranges, dtype=torch.float32).detach().cpu()
    if ranges.numel() == 0:
        raise ValueError("value_ranges must be non-empty")
    if torch.any(ranges <= 0):
        raise ValueError("all value_ranges entries must be positive")
    return ranges.contiguous()


def _ranges_like_values(
    value_ranges: torch.Tensor | list[float] | tuple[float, ...],
    values: torch.Tensor,
) -> torch.Tensor:
    ranges = _validate_range_tensor(value_ranges).to(device=values.device, dtype=values.float().dtype)
    if tuple(ranges.shape) == tuple(values.shape):
        return ranges
    if ranges.ndim == 1 and values.ndim >= 1 and int(values.shape[-1]) == int(ranges.numel()):
        return ranges.reshape(*([1] * (values.ndim - 1)), int(ranges.numel())).expand_as(values.float())
    if ranges.numel() == values.numel():
        return ranges.reshape(values.shape)
    raise ValueError("values shape is incompatible with value_ranges")


def _validate_codebook_tensor(codebooks: torch.Tensor | list[list[float]] | tuple[tuple[float, ...], ...]) -> torch.Tensor:
    tensor = torch.as_tensor(codebooks, dtype=torch.float32).detach().cpu()
    if tensor.ndim != 2:
        raise ValueError("codebooks must have shape [components, levels]")
    if int(tensor.shape[0]) <= 0:
        raise ValueError("codebooks must contain at least one component")
    if int(tensor.shape[1]) <= 1:
        raise ValueError("codebooks must contain at least two levels")
    if not torch.isfinite(tensor).all():
        raise ValueError("codebooks must contain only finite values")
    return tensor.contiguous()


def _pack_variable_width_values(values: list[int], widths: list[int]) -> bytes:
    if len(values) != len(widths):
        raise ValueError("values and widths length mismatch")
    out = bytearray()
    acc = 0
    bit_count = 0
    for value, width in zip(values, widths):
        width_i = int(width)
        value_i = int(value)
        if width_i <= 0:
            raise ValueError("packed widths must be positive")
        if value_i < 0 or value_i >= (1 << width_i):
            raise ValueError("packed value exceeds width")
        acc = (acc << width_i) | value_i
        bit_count += width_i
        while bit_count >= 8:
            bit_count -= 8
            out.append((acc >> bit_count) & 0xFF)
            acc &= (1 << bit_count) - 1 if bit_count else 0
    if bit_count:
        out.append((acc << (8 - bit_count)) & 0xFF)
    return bytes(out)


def _unpack_variable_width_values(payload: bytes, widths: list[int]) -> list[int]:
    values: list[int] = []
    acc = 0
    bit_count = 0
    byte_index = 0
    for width in widths:
        width_i = int(width)
        if width_i <= 0:
            raise ValueError("packed widths must be positive")
        while bit_count < width_i and byte_index < len(payload):
            acc = (acc << 8) | int(payload[byte_index])
            bit_count += 8
            byte_index += 1
        if bit_count < width_i:
            raise ValueError("truncated variable-width payload")
        bit_count -= width_i
        values.append((acc >> bit_count) & ((1 << width_i) - 1))
        acc &= (1 << bit_count) - 1 if bit_count else 0
    if byte_index != len(payload):
        raise ValueError("trailing bytes after variable-width payload")
    if bit_count and acc != 0:
        raise ValueError("non-zero padding bits in variable-width payload")
    return values


@dataclass(frozen=True)
class UniformControlGridCode:
    """Uniform fixed-bit codec for tiny Stage 5 condition-control grids.

    This is intentionally simple and auditable: every quantized control symbol
    is packed into a real payload byte stream, and the payload length is counted
    in ``actual_payload_bpp`` when transmitted.
    """

    bits: int = 4
    value_range: float = 0.25

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError("bits must be >= 1")
        if self.bits > 16:
            raise ValueError("bits must be <= 16")
        if self.value_range <= 0:
            raise ValueError("value_range must be positive")

    @property
    def levels(self) -> int:
        return 1 << int(self.bits)

    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        clipped = values.float().clamp(-self.value_range, self.value_range)
        normalized = (clipped + self.value_range) / (2.0 * self.value_range)
        codes = torch.round(normalized * (self.levels - 1)).to(torch.long)
        return codes.clamp_(0, self.levels - 1)

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        codes_f = codes.float().clamp(0, self.levels - 1)
        normalized = codes_f / float(self.levels - 1)
        return normalized * (2.0 * self.value_range) - self.value_range

    def encode(self, codes: torch.Tensor) -> bytes:
        return encode_semantic_tokens(codes.to(torch.long), codebook_size=self.levels, codec="fixed_bits")

    def decode(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        return decode_semantic_tokens(payload, shape=shape, codebook_size=self.levels, codec="fixed_bits")

    def encode_values(self, values: torch.Tensor) -> bytes:
        return self.encode(self.quantize(values))

    def decode_values(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        return self.dequantize(self.decode(payload, shape=shape))

    def encoded_num_bytes(self, shape: tuple[int, ...]) -> int:
        return (self.encoded_num_bits(shape) + 7) // 8

    def encoded_num_bits(self, shape: tuple[int, ...]) -> int:
        count = 1
        for dim in shape:
            if int(dim) <= 0:
                raise ValueError("shape dimensions must be positive")
            count *= int(dim)
        return count * int(self.bits)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "uniform_control_grid_fixed_bits",
            "bits": int(self.bits),
            "value_range": float(self.value_range),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "UniformControlGridCode":
        if payload.get("codec") != "uniform_control_grid_fixed_bits":
            raise ValueError("payload codec must be uniform_control_grid_fixed_bits")
        return cls(bits=int(payload["bits"]), value_range=float(payload["value_range"]))


@dataclass(frozen=True)
class ComponentUniformControlCode:
    """Uniform fixed-bit codec with fixed per-symbol ranges.

    The ranges are decoder-side model state fitted from non-eval data. Per
    image, only the quantized symbols are transmitted.
    """

    bits: int
    value_ranges: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError("bits must be >= 1")
        if self.bits > 16:
            raise ValueError("bits must be <= 16")
        _validate_range_tensor(self.value_ranges)

    @property
    def levels(self) -> int:
        return 1 << int(self.bits)

    @property
    def ranges(self) -> torch.Tensor:
        return _validate_range_tensor(self.value_ranges)

    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        ranges = _ranges_like_values(self.value_ranges, values)
        clipped = values.float().clamp(-ranges, ranges)
        normalized = (clipped + ranges) / (2.0 * ranges)
        codes = torch.round(normalized * (self.levels - 1)).to(torch.long)
        return codes.clamp_(0, self.levels - 1)

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        ranges = _ranges_like_values(self.value_ranges, codes.float())
        codes_f = codes.float().clamp(0, self.levels - 1)
        normalized = codes_f / float(self.levels - 1)
        return normalized * (2.0 * ranges) - ranges

    def encode(self, codes: torch.Tensor) -> bytes:
        return encode_semantic_tokens(codes.to(torch.long), codebook_size=self.levels, codec="fixed_bits")

    def decode(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        if math.prod(tuple(int(v) for v in shape)) != len(self.value_ranges):
            raise ValueError("shape must contain exactly one symbol per value_ranges entry")
        return decode_semantic_tokens(payload, shape=shape, codebook_size=self.levels, codec="fixed_bits")

    def encode_values(self, values: torch.Tensor) -> bytes:
        if values.numel() != len(self.value_ranges):
            raise ValueError("values must contain exactly one symbol per value_ranges entry")
        return self.encode(self.quantize(values))

    def decode_values(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        return self.dequantize(self.decode(payload, shape=shape))

    def encoded_num_bytes(self, shape: tuple[int, ...]) -> int:
        return (self.encoded_num_bits(shape) + 7) // 8

    def encoded_num_bits(self, shape: tuple[int, ...]) -> int:
        count = 1
        for dim in shape:
            if int(dim) <= 0:
                raise ValueError("shape dimensions must be positive")
            count *= int(dim)
        if count != len(self.value_ranges):
            raise ValueError("shape must contain exactly one symbol per value_ranges entry")
        return count * int(self.bits)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "component_uniform_control_fixed_bits",
            "bits": int(self.bits),
            "value_ranges": [float(v) for v in self.value_ranges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ComponentUniformControlCode":
        if payload.get("codec") != "component_uniform_control_fixed_bits":
            raise ValueError("payload codec must be component_uniform_control_fixed_bits")
        ranges = payload.get("value_ranges")
        if not isinstance(ranges, list):
            raise TypeError("payload value_ranges must be a list")
        return cls(bits=int(payload["bits"]), value_ranges=tuple(float(v) for v in ranges))


@dataclass(frozen=True)
class MuLawControlGridCode:
    """Mu-law companded fixed-bit codec for tiny condition-control values.

    Basis/DCT condition residual coefficients are usually concentrated near
    zero. Mu-law companding spends more quantization levels near zero while
    keeping a deterministic fixed decoder-side transform.
    """

    bits: int = 4
    value_range: float = 0.25
    mu: float = 16.0

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError("bits must be >= 1")
        if self.bits > 16:
            raise ValueError("bits must be <= 16")
        if self.value_range <= 0:
            raise ValueError("value_range must be positive")
        if self.mu <= 0:
            raise ValueError("mu must be positive")

    @property
    def levels(self) -> int:
        return 1 << int(self.bits)

    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        clipped = values.float().clamp(-self.value_range, self.value_range)
        normalized_abs = clipped.abs() / float(self.value_range)
        companded_abs = torch.log1p(float(self.mu) * normalized_abs) / math.log1p(float(self.mu))
        signed_companded = torch.sign(clipped) * companded_abs
        normalized = (signed_companded + 1.0) * 0.5
        codes = torch.round(normalized * (self.levels - 1)).to(torch.long)
        return codes.clamp_(0, self.levels - 1)

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        codes_f = codes.float().clamp(0, self.levels - 1)
        signed_companded = (codes_f / float(self.levels - 1)) * 2.0 - 1.0
        magnitude = torch.expm1(signed_companded.abs() * math.log1p(float(self.mu))) / float(self.mu)
        return torch.sign(signed_companded) * magnitude * float(self.value_range)

    def encode(self, codes: torch.Tensor) -> bytes:
        return encode_semantic_tokens(codes.to(torch.long), codebook_size=self.levels, codec="fixed_bits")

    def decode(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        return decode_semantic_tokens(payload, shape=shape, codebook_size=self.levels, codec="fixed_bits")

    def encode_values(self, values: torch.Tensor) -> bytes:
        return self.encode(self.quantize(values))

    def decode_values(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        return self.dequantize(self.decode(payload, shape=shape))

    def encoded_num_bytes(self, shape: tuple[int, ...]) -> int:
        return (self.encoded_num_bits(shape) + 7) // 8

    def encoded_num_bits(self, shape: tuple[int, ...]) -> int:
        count = 1
        for dim in shape:
            if int(dim) <= 0:
                raise ValueError("shape dimensions must be positive")
            count *= int(dim)
        return count * int(self.bits)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "mulaw_control_grid_fixed_bits",
            "bits": int(self.bits),
            "value_range": float(self.value_range),
            "mu": float(self.mu),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MuLawControlGridCode":
        if payload.get("codec") != "mulaw_control_grid_fixed_bits":
            raise ValueError("payload codec must be mulaw_control_grid_fixed_bits")
        return cls(bits=int(payload["bits"]), value_range=float(payload["value_range"]), mu=float(payload["mu"]))


@dataclass(frozen=True)
class ComponentMuLawControlCode:
    """Mu-law fixed-bit codec with fixed per-symbol ranges."""

    bits: int
    value_ranges: tuple[float, ...]
    mu: float = 16.0

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError("bits must be >= 1")
        if self.bits > 16:
            raise ValueError("bits must be <= 16")
        if self.mu <= 0:
            raise ValueError("mu must be positive")
        _validate_range_tensor(self.value_ranges)

    @property
    def levels(self) -> int:
        return 1 << int(self.bits)

    @property
    def ranges(self) -> torch.Tensor:
        return _validate_range_tensor(self.value_ranges)

    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        ranges = _ranges_like_values(self.value_ranges, values)
        clipped = values.float().clamp(-ranges, ranges)
        normalized_abs = clipped.abs() / ranges
        companded_abs = torch.log1p(float(self.mu) * normalized_abs) / math.log1p(float(self.mu))
        signed_companded = torch.sign(clipped) * companded_abs
        normalized = (signed_companded + 1.0) * 0.5
        codes = torch.round(normalized * (self.levels - 1)).to(torch.long)
        return codes.clamp_(0, self.levels - 1)

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        ranges = _ranges_like_values(self.value_ranges, codes.float())
        codes_f = codes.float().clamp(0, self.levels - 1)
        signed_companded = (codes_f / float(self.levels - 1)) * 2.0 - 1.0
        magnitude = torch.expm1(signed_companded.abs() * math.log1p(float(self.mu))) / float(self.mu)
        return torch.sign(signed_companded) * magnitude * ranges

    def encode(self, codes: torch.Tensor) -> bytes:
        return encode_semantic_tokens(codes.to(torch.long), codebook_size=self.levels, codec="fixed_bits")

    def decode(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        if math.prod(tuple(int(v) for v in shape)) != len(self.value_ranges):
            raise ValueError("shape must contain exactly one symbol per value_ranges entry")
        return decode_semantic_tokens(payload, shape=shape, codebook_size=self.levels, codec="fixed_bits")

    def encode_values(self, values: torch.Tensor) -> bytes:
        if values.numel() != len(self.value_ranges):
            raise ValueError("values must contain exactly one symbol per value_ranges entry")
        return self.encode(self.quantize(values))

    def decode_values(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        return self.dequantize(self.decode(payload, shape=shape))

    def encoded_num_bytes(self, shape: tuple[int, ...]) -> int:
        return (self.encoded_num_bits(shape) + 7) // 8

    def encoded_num_bits(self, shape: tuple[int, ...]) -> int:
        count = 1
        for dim in shape:
            if int(dim) <= 0:
                raise ValueError("shape dimensions must be positive")
            count *= int(dim)
        if count != len(self.value_ranges):
            raise ValueError("shape must contain exactly one symbol per value_ranges entry")
        return count * int(self.bits)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "component_mulaw_control_fixed_bits",
            "bits": int(self.bits),
            "value_ranges": [float(v) for v in self.value_ranges],
            "mu": float(self.mu),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ComponentMuLawControlCode":
        if payload.get("codec") != "component_mulaw_control_fixed_bits":
            raise ValueError("payload codec must be component_mulaw_control_fixed_bits")
        ranges = payload.get("value_ranges")
        if not isinstance(ranges, list):
            raise TypeError("payload value_ranges must be a list")
        return cls(bits=int(payload["bits"]), value_ranges=tuple(float(v) for v in ranges), mu=float(payload["mu"]))


@dataclass(frozen=True)
class ComponentCodebookControlCode:
    """Fixed-bit codec with fixed per-component scalar codebooks.

    The codebooks are decoder-side state fitted on non-eval data. Per image,
    this codec still transmits only fixed-bit level indices.
    """

    bits: int
    codebooks: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError("bits must be >= 1")
        if self.bits > 16:
            raise ValueError("bits must be <= 16")
        books = _validate_codebook_tensor(self.codebooks)
        if int(books.shape[1]) != self.levels:
            raise ValueError("codebook level count must equal 2 ** bits")

    @property
    def levels(self) -> int:
        return 1 << int(self.bits)

    @property
    def component_count(self) -> int:
        return len(self.codebooks)

    @property
    def codebook_tensor(self) -> torch.Tensor:
        return _validate_codebook_tensor(self.codebooks)

    def select(self, indices: torch.Tensor) -> "ComponentCodebookControlCode":
        index = indices.detach().cpu().to(torch.long).reshape(-1)
        if index.numel() == 0:
            raise ValueError("indices must be non-empty")
        if int(index.min().item()) < 0 or int(index.max().item()) >= self.component_count:
            raise ValueError("component codebook index out of range")
        books = self.codebook_tensor.index_select(0, index)
        return ComponentCodebookControlCode(
            bits=int(self.bits),
            codebooks=tuple(tuple(float(v) for v in row.tolist()) for row in books),
        )

    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        values_f = values.float()
        books = self.codebook_tensor.to(device=values_f.device, dtype=values_f.dtype)
        component_count = int(books.shape[0])
        if values_f.ndim >= 1 and int(values_f.shape[-1]) == component_count:
            view_shape = (*([1] * (values_f.ndim - 1)), component_count, self.levels)
            distances = (values_f.unsqueeze(-1) - books.reshape(view_shape)).abs()
            return torch.argmin(distances, dim=-1).to(torch.long)
        if int(values_f.numel()) == component_count:
            flat = values_f.reshape(component_count)
            distances = (flat.unsqueeze(-1) - books).abs()
            return torch.argmin(distances, dim=-1).to(torch.long).reshape(values.shape)
        raise ValueError("values shape is incompatible with component codebooks")

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        codes_long = codes.to(torch.long).clamp(0, self.levels - 1)
        books = self.codebook_tensor.to(device=codes_long.device)
        component_count = int(books.shape[0])
        if codes_long.ndim >= 1 and int(codes_long.shape[-1]) == component_count:
            view_shape = (*([1] * (codes_long.ndim - 1)), component_count, self.levels)
            expanded = books.reshape(view_shape).expand(*codes_long.shape, self.levels)
            return torch.gather(expanded, dim=-1, index=codes_long.unsqueeze(-1)).squeeze(-1)
        if int(codes_long.numel()) == component_count:
            flat = codes_long.reshape(component_count)
            return books.gather(dim=1, index=flat.unsqueeze(1)).squeeze(1).reshape(codes.shape)
        raise ValueError("codes shape is incompatible with component codebooks")

    def encode(self, codes: torch.Tensor) -> bytes:
        return encode_semantic_tokens(codes.to(torch.long), codebook_size=self.levels, codec="fixed_bits")

    def decode(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        if math.prod(tuple(int(v) for v in shape)) != self.component_count:
            raise ValueError("shape must contain exactly one symbol per component codebook")
        return decode_semantic_tokens(payload, shape=shape, codebook_size=self.levels, codec="fixed_bits")

    def encode_values(self, values: torch.Tensor) -> bytes:
        if values.numel() != self.component_count:
            raise ValueError("values must contain exactly one symbol per component codebook")
        return self.encode(self.quantize(values))

    def decode_values(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        return self.dequantize(self.decode(payload, shape=shape))

    def encoded_num_bytes(self, shape: tuple[int, ...]) -> int:
        return (self.encoded_num_bits(shape) + 7) // 8

    def encoded_num_bits(self, shape: tuple[int, ...]) -> int:
        count = 1
        for dim in shape:
            if int(dim) <= 0:
                raise ValueError("shape dimensions must be positive")
            count *= int(dim)
        if count != self.component_count:
            raise ValueError("shape must contain exactly one symbol per component codebook")
        return count * int(self.bits)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "component_codebook_control_fixed_bits",
            "bits": int(self.bits),
            "codebooks": [[float(v) for v in row] for row in self.codebooks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ComponentCodebookControlCode":
        if payload.get("codec") != "component_codebook_control_fixed_bits":
            raise ValueError("payload codec must be component_codebook_control_fixed_bits")
        codebooks = payload.get("codebooks")
        if not isinstance(codebooks, list):
            raise TypeError("payload codebooks must be a list")
        return cls(
            bits=int(payload["bits"]),
            codebooks=tuple(tuple(float(v) for v in row) for row in codebooks),
        )


@dataclass(frozen=True)
class VectorCodebookControlCode:
    """Fixed-bit vector-quantized code for a whole coefficient vector.

    The vector codebook is fixed decoder-side state. Per image, only one
    codebook index is transmitted for the selected coefficient prefix.
    """

    bits: int
    vectors: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError("bits must be >= 1")
        if self.bits > 16:
            raise ValueError("bits must be <= 16")
        tensor = _validate_codebook_tensor(self.vectors)
        if int(tensor.shape[0]) != self.levels:
            raise ValueError("vector count must equal 2 ** bits")

    @property
    def levels(self) -> int:
        return 1 << int(self.bits)

    @property
    def vector_dim(self) -> int:
        return len(self.vectors[0])

    @property
    def vector_tensor(self) -> torch.Tensor:
        return _validate_codebook_tensor(self.vectors)

    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        values_f = values.float()
        vectors = self.vector_tensor.to(device=values_f.device, dtype=values_f.dtype)
        if values_f.ndim == 1:
            if int(values_f.numel()) != self.vector_dim:
                raise ValueError("values length must match vector_dim")
            distances = torch.sum((vectors - values_f.unsqueeze(0)).square(), dim=1)
            return torch.argmin(distances).reshape(1).to(torch.long)
        if values_f.ndim == 2:
            if int(values_f.shape[1]) != self.vector_dim:
                raise ValueError("values shape must be [N, vector_dim]")
            distances = torch.cdist(values_f, vectors, p=2.0)
            return torch.argmin(distances, dim=1).to(torch.long)
        raise ValueError("values must have shape [vector_dim] or [N, vector_dim]")

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        codes_long = codes.to(torch.long).clamp(0, self.levels - 1)
        vectors = self.vector_tensor.to(device=codes_long.device)
        flat = codes_long.reshape(-1)
        decoded = vectors.index_select(0, flat)
        return decoded.reshape(*codes_long.shape, self.vector_dim)

    def encode(self, codes: torch.Tensor) -> bytes:
        return encode_semantic_tokens(codes.to(torch.long), codebook_size=self.levels, codec="fixed_bits")

    def decode(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        return decode_semantic_tokens(payload, shape=shape, codebook_size=self.levels, codec="fixed_bits")

    def encode_values(self, values: torch.Tensor) -> bytes:
        return self.encode(self.quantize(values))

    def decode_values(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        return self.dequantize(self.decode(payload, shape=shape))

    def encoded_num_bytes(self, shape: tuple[int, ...]) -> int:
        return (self.encoded_num_bits(shape) + 7) // 8

    def encoded_num_bits(self, shape: tuple[int, ...]) -> int:
        count = 1
        for dim in shape:
            if int(dim) <= 0:
                raise ValueError("shape dimensions must be positive")
            count *= int(dim)
        return count * int(self.bits)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "vector_codebook_control_fixed_bits",
            "bits": int(self.bits),
            "vector_dim": int(self.vector_dim),
            "vectors": [[float(v) for v in row] for row in self.vectors],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "VectorCodebookControlCode":
        if payload.get("codec") != "vector_codebook_control_fixed_bits":
            raise ValueError("payload codec must be vector_codebook_control_fixed_bits")
        vectors = payload.get("vectors")
        if not isinstance(vectors, list):
            raise TypeError("payload vectors must be a list")
        return cls(
            bits=int(payload["bits"]),
            vectors=tuple(tuple(float(v) for v in row) for row in vectors),
        )


def build_control_grid_code(
    *,
    quantizer: str,
    bits: int,
    value_range: float,
    mu: float = 16.0,
    value_ranges: list[float] | tuple[float, ...] | torch.Tensor | None = None,
) -> UniformControlGridCode | MuLawControlGridCode | ComponentUniformControlCode | ComponentMuLawControlCode:
    if value_ranges is not None:
        ranges = tuple(float(v) for v in _validate_range_tensor(value_ranges).flatten().tolist())
        if quantizer == "uniform":
            return ComponentUniformControlCode(bits=bits, value_ranges=ranges)
        if quantizer == "mu_law":
            return ComponentMuLawControlCode(bits=bits, value_ranges=ranges, mu=mu)
        raise ValueError("quantizer must be uniform or mu_law")
    if quantizer == "uniform":
        return UniformControlGridCode(bits=bits, value_range=value_range)
    if quantizer == "mu_law":
        return MuLawControlGridCode(bits=bits, value_range=value_range, mu=mu)
    raise ValueError("quantizer must be uniform or mu_law")


def control_quantizer_label(quantizer: str, mu: float = 16.0) -> str:
    if quantizer == "uniform":
        return "uniform"
    if quantizer == "mu_law":
        mu_text = f"{float(mu):g}".replace(".", "p")
        return f"mulaw{mu_text}"
    raise ValueError("quantizer must be uniform or mu_law")


@dataclass(frozen=True)
class StaticControlHuffmanCode:
    """Position-conditioned static Huffman code for tiny control symbols.

    The code lengths are fixed decoder-side state. Per image, only the
    entropy-coded quantized control symbols are transmitted and counted.
    """

    codebook_size: int
    symbol_shape: tuple[int, ...]
    position_codes: tuple[StaticHuffmanCode, ...]

    @classmethod
    def from_counts(
        cls,
        counts: torch.Tensor,
        *,
        smoothing_count: int = 1,
    ) -> "StaticControlHuffmanCode":
        if counts.ndim < 2:
            raise ValueError("counts must have shape [*symbol_shape, codebook_size]")
        symbol_shape = tuple(int(v) for v in counts.shape[:-1])
        codebook_size = int(counts.shape[-1])
        if any(dim <= 0 for dim in symbol_shape):
            raise ValueError("symbol_shape dimensions must be positive")
        if codebook_size <= 1:
            raise ValueError("codebook_size must be > 1")
        flat_counts = counts.detach().cpu().reshape(-1, codebook_size)
        codes = tuple(
            StaticHuffmanCode.from_counts(row, smoothing_count=smoothing_count)
            for row in flat_counts
        )
        return cls(codebook_size=codebook_size, symbol_shape=symbol_shape, position_codes=codes)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "StaticControlHuffmanCode":
        if payload.get("codec") != "static_control_huffman":
            raise ValueError("payload codec must be static_control_huffman")
        symbol_shape_raw = payload.get("symbol_shape")
        code_lengths_raw = payload.get("position_code_lengths")
        if not isinstance(symbol_shape_raw, list) or not symbol_shape_raw:
            raise TypeError("payload must contain non-empty list symbol_shape")
        if not isinstance(code_lengths_raw, list):
            raise TypeError("payload must contain list position_code_lengths")
        symbol_shape = tuple(int(v) for v in symbol_shape_raw)
        expected_positions = math.prod(symbol_shape)
        if len(code_lengths_raw) != expected_positions:
            raise ValueError("position_code_lengths length does not match symbol_shape")
        position_codes = tuple(
            StaticHuffmanCode.from_code_lengths([int(v) for v in lengths])
            for lengths in code_lengths_raw
        )
        codebook_size = int(payload.get("codebook_size", position_codes[0].codebook_size))
        if any(code.codebook_size != codebook_size for code in position_codes):
            raise ValueError("all position codes must use the same codebook_size")
        return cls(codebook_size=codebook_size, symbol_shape=symbol_shape, position_codes=position_codes)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "static_control_huffman",
            "version": 0,
            "codebook_size": int(self.codebook_size),
            "symbol_shape": list(self.symbol_shape),
            "position_code_lengths": [list(code.code_lengths) for code in self.position_codes],
        }

    def prefix(self, count: int) -> "StaticControlHuffmanCode":
        if len(self.symbol_shape) != 1:
            raise ValueError("prefix is only defined for 1D control vectors")
        count_i = int(count)
        if count_i <= 0 or count_i > self.symbol_shape[0]:
            raise ValueError("prefix count must be in [1, symbol_shape[0]]")
        return StaticControlHuffmanCode(
            codebook_size=self.codebook_size,
            symbol_shape=(count_i,),
            position_codes=self.position_codes[:count_i],
        )

    def encode(self, indices: torch.Tensor) -> bytes:
        values = self._flatten_indices(indices)
        out = bytearray()
        acc = 0
        bit_count = 0
        for value, codebook in zip(values, self.position_codes):
            code = codebook.codes[value]
            length = codebook.code_lengths[value]
            acc = (acc << length) | code
            bit_count += length
            while bit_count >= 8:
                bit_count -= 8
                out.append((acc >> bit_count) & 0xFF)
                acc &= (1 << bit_count) - 1 if bit_count else 0
        if bit_count:
            out.append((acc << (8 - bit_count)) & 0xFF)
        return bytes(out)

    def decode(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        if tuple(int(v) for v in shape) != self.symbol_shape:
            raise ValueError(f"shape {shape} does not match symbol_shape {self.symbol_shape}")
        values: list[int] = []
        acc = 0
        bit_count = 0
        byte_index = 0
        for codebook in self.position_codes:
            table = codebook._decode_table()
            max_len = max(codebook.code_lengths)
            while True:
                if bit_count < max_len and byte_index < len(payload):
                    acc = (acc << 8) | int(payload[byte_index])
                    bit_count += 8
                    byte_index += 1
                matched = False
                for length in range(1, min(max_len, bit_count) + 1):
                    prefix = (acc >> (bit_count - length)) & ((1 << length) - 1)
                    symbol = table.get((length, prefix))
                    if symbol is None:
                        continue
                    values.append(symbol)
                    bit_count -= length
                    acc &= (1 << bit_count) - 1 if bit_count else 0
                    matched = True
                    break
                if matched:
                    break
                if byte_index >= len(payload):
                    raise ValueError("truncated static control Huffman payload")
        if byte_index != len(payload):
            raise ValueError("trailing bytes after static control Huffman payload")
        if bit_count and acc != 0:
            raise ValueError("non-zero padding bits in static control Huffman payload")
        return torch.tensor(values, dtype=torch.long).reshape(self.symbol_shape)

    def encoded_bits(self, indices: torch.Tensor) -> int:
        values = self._flatten_indices(indices)
        return int(sum(codebook.code_lengths[value] for value, codebook in zip(values, self.position_codes)))

    def mean_code_length_bits(self, indices: torch.Tensor) -> float:
        return float(self.encoded_bits(indices)) / float(math.prod(self.symbol_shape))

    def _flatten_indices(self, indices: torch.Tensor) -> list[int]:
        if tuple(int(v) for v in indices.shape) != self.symbol_shape:
            raise ValueError(f"indices shape {tuple(indices.shape)} does not match symbol_shape {self.symbol_shape}")
        if torch.is_floating_point(indices) or torch.is_complex(indices):
            raise TypeError("indices must be an integer tensor")
        flat = indices.detach().cpu().to(torch.long).reshape(-1)
        min_value = int(flat.min().item())
        max_value = int(flat.max().item())
        if min_value < 0 or max_value >= self.codebook_size:
            raise ValueError(
                f"control symbol out of range for codebook_size={self.codebook_size}: "
                f"min={min_value}, max={max_value}"
            )
        return [int(v) for v in flat.tolist()]


@dataclass(frozen=True)
class SparseControlBasisCode:
    """Fixed-bit sparse index/value code for image-specific basis coefficients.

    The basis itself is fixed decoder-side state. Per image, this code transmits
    the selected coefficient indices and their quantized values, so both streams
    must be counted in actual payload bpp.
    """

    candidate_components: int
    selected_components: int
    value_codec: (
        UniformControlGridCode
        | MuLawControlGridCode
        | ComponentUniformControlCode
        | ComponentMuLawControlCode
        | ComponentCodebookControlCode
        | VectorCodebookControlCode
    )

    def __post_init__(self) -> None:
        if self.candidate_components <= 1:
            raise ValueError("candidate_components must be > 1")
        if self.selected_components <= 0:
            raise ValueError("selected_components must be positive")
        if self.selected_components > self.candidate_components:
            raise ValueError("selected_components must be <= candidate_components")

    @property
    def index_bits(self) -> int:
        return semantic_bits_per_token(int(self.candidate_components))

    def select(self, coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if coefficients.ndim != 1:
            raise ValueError("coefficients must be a 1D tensor")
        if int(coefficients.numel()) != int(self.candidate_components):
            raise ValueError("coefficients length must match candidate_components")
        topk = torch.topk(coefficients.float().abs(), k=int(self.selected_components), largest=True, sorted=False)
        indices = torch.sort(topk.indices.to(torch.long)).values
        values = coefficients.index_select(0, indices.to(device=coefficients.device))
        return indices, values

    def encode(self, indices: torch.Tensor, value_codes: torch.Tensor) -> tuple[bytes, bytes]:
        indices_long = self._validate_indices(indices)
        values_long = self._validate_value_codes(value_codes)
        index_payload = encode_semantic_tokens(
            indices_long,
            codebook_size=int(self.candidate_components),
            codec="fixed_bits",
        )
        value_payload = self.value_codec.encode(values_long)
        return index_payload, value_payload

    def encode_compact(self, indices: torch.Tensor, value_codes: torch.Tensor) -> bytes:
        """Pack fixed-bit index and value streams into one compact payload."""

        indices_long = self._validate_indices(indices)
        values_long = self._validate_value_codes(value_codes)
        values = [int(v) for v in indices_long.reshape(-1).tolist()]
        values.extend(int(v) for v in values_long.reshape(-1).tolist())
        widths = [int(self.index_bits)] * int(self.selected_components)
        widths.extend([int(self.value_codec.bits)] * int(self.selected_components))
        return self._pack_variable_width(values, widths)

    def decode_compact(self, payload: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        widths = [int(self.index_bits)] * int(self.selected_components)
        widths.extend([int(self.value_codec.bits)] * int(self.selected_components))
        values = self._unpack_variable_width(payload, widths)
        split = int(self.selected_components)
        indices = torch.tensor(values[:split], dtype=torch.long)
        value_codes = torch.tensor(values[split:], dtype=torch.long)
        self._validate_indices(indices)
        self._validate_value_codes(value_codes)
        return indices, value_codes

    def decode(self, index_payload: bytes, value_payload: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        indices = decode_semantic_tokens(
            index_payload,
            shape=(int(self.selected_components),),
            codebook_size=int(self.candidate_components),
            codec="fixed_bits",
        )
        values = self.value_codec.decode(value_payload, shape=(int(self.selected_components),))
        self._validate_indices(indices)
        self._validate_value_codes(values)
        return indices, values

    def encode_entropy(
        self,
        indices: torch.Tensor,
        value_codes: torch.Tensor,
        *,
        index_huffman: StaticControlHuffmanCode,
        value_huffman: StaticControlHuffmanCode,
    ) -> tuple[bytes, bytes]:
        indices_long = self._validate_indices(indices)
        values_long = self._validate_value_codes(value_codes)
        self._validate_huffman(index_huffman, codebook_size=int(self.candidate_components), name="index_huffman")
        self._validate_huffman(value_huffman, codebook_size=int(self.value_codec.levels), name="value_huffman")
        return index_huffman.encode(indices_long), value_huffman.encode(values_long)

    def decode_entropy(
        self,
        index_payload: bytes,
        value_payload: bytes,
        *,
        index_huffman: StaticControlHuffmanCode,
        value_huffman: StaticControlHuffmanCode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_huffman(index_huffman, codebook_size=int(self.candidate_components), name="index_huffman")
        self._validate_huffman(value_huffman, codebook_size=int(self.value_codec.levels), name="value_huffman")
        indices = index_huffman.decode(index_payload, shape=(int(self.selected_components),))
        values = value_huffman.decode(value_payload, shape=(int(self.selected_components),))
        self._validate_indices(indices)
        self._validate_value_codes(values)
        return indices, values

    def encoded_num_bytes(self) -> int:
        index_bytes = (int(self.selected_components) * int(self.index_bits) + 7) // 8
        value_bytes = self.value_codec.encoded_num_bytes((int(self.selected_components),))
        return index_bytes + value_bytes

    def encoded_compact_num_bytes(self) -> int:
        return (self.encoded_compact_num_bits() + 7) // 8

    def encoded_num_bits(self) -> int:
        index_bits = int(self.selected_components) * int(self.index_bits)
        value_bits = self.value_codec.encoded_num_bits((int(self.selected_components),))
        return index_bits + value_bits

    def encoded_compact_num_bits(self) -> int:
        return int(self.selected_components) * (int(self.index_bits) + int(self.value_codec.bits))

    def scatter_values(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        indices_long = self._validate_indices(indices)
        decoded_values = values.float()
        if decoded_values.shape != (int(self.selected_components),):
            raise ValueError("values shape must match selected_components")
        out = torch.zeros(int(self.candidate_components), dtype=decoded_values.dtype, device=decoded_values.device)
        out.scatter_(0, indices_long.to(device=decoded_values.device), decoded_values)
        return out

    def _validate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if tuple(indices.shape) != (int(self.selected_components),):
            raise ValueError("indices shape must match selected_components")
        if torch.is_floating_point(indices) or torch.is_complex(indices):
            raise TypeError("indices must be an integer tensor")
        values = indices.detach().cpu().to(torch.long)
        if int(values.min().item()) < 0 or int(values.max().item()) >= int(self.candidate_components):
            raise ValueError("sparse control index out of range")
        if int(torch.unique(values).numel()) != int(self.selected_components):
            raise ValueError("sparse control indices must be unique")
        return values

    def _validate_value_codes(self, value_codes: torch.Tensor) -> torch.Tensor:
        if tuple(value_codes.shape) != (int(self.selected_components),):
            raise ValueError("value_codes shape must match selected_components")
        if torch.is_floating_point(value_codes) or torch.is_complex(value_codes):
            raise TypeError("value_codes must be an integer tensor")
        values = value_codes.detach().cpu().to(torch.long)
        if int(values.min().item()) < 0 or int(values.max().item()) >= int(self.value_codec.levels):
            raise ValueError("sparse control value code out of range")
        return values

    def _validate_huffman(self, huffman: StaticControlHuffmanCode, *, codebook_size: int, name: str) -> None:
        if huffman.symbol_shape != (int(self.selected_components),):
            raise ValueError(f"{name} symbol_shape must match selected_components")
        if int(huffman.codebook_size) != int(codebook_size):
            raise ValueError(f"{name} codebook_size mismatch")

    def _pack_variable_width(self, values: list[int], widths: list[int]) -> bytes:
        return _pack_variable_width_values(values, widths)

    def _unpack_variable_width(self, payload: bytes, widths: list[int]) -> list[int]:
        return _unpack_variable_width_values(payload, widths)


@dataclass(frozen=True)
class PrefixTopKControlBasisCode:
    """Fixed-bit prefix plus sparse-tail basis coefficient codec.

    A stable prefix of basis coefficients is always transmitted. Additional
    image-adaptive coefficients are selected from the remaining candidate tail.
    The fixed-bit prefix values, tail indices, and tail values are packed into
    one payload stream for exact low-bpp accounting.
    """

    candidate_components: int
    prefix_components: int
    selected_components: int
    value_codec: (
        UniformControlGridCode
        | MuLawControlGridCode
        | ComponentUniformControlCode
        | ComponentMuLawControlCode
        | ComponentCodebookControlCode
    )

    def __post_init__(self) -> None:
        if self.candidate_components <= 1:
            raise ValueError("candidate_components must be > 1")
        if self.prefix_components <= 0:
            raise ValueError("prefix_components must be positive")
        if self.selected_components <= 0:
            raise ValueError("selected_components must be positive")
        if self.prefix_components >= self.candidate_components:
            raise ValueError("prefix_components must be < candidate_components")
        if self.selected_components > self.tail_components:
            raise ValueError("selected_components must be <= candidate_components - prefix_components")

    @property
    def tail_components(self) -> int:
        return int(self.candidate_components) - int(self.prefix_components)

    @property
    def tail_index_bits(self) -> int:
        return semantic_bits_per_token(int(self.tail_components))

    def select(self, coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if coefficients.ndim != 1:
            raise ValueError("coefficients must be a 1D tensor")
        if int(coefficients.numel()) != int(self.candidate_components):
            raise ValueError("coefficients length must match candidate_components")
        prefix_values = coefficients[: int(self.prefix_components)]
        tail = coefficients[int(self.prefix_components) : int(self.candidate_components)]
        topk = torch.topk(tail.float().abs(), k=int(self.selected_components), largest=True, sorted=False)
        local_indices = torch.sort(topk.indices.to(torch.long)).values
        global_indices = local_indices + int(self.prefix_components)
        selected_values = coefficients.index_select(0, global_indices.to(device=coefficients.device))
        return prefix_values, local_indices, global_indices, selected_values

    def encode_compact(
        self,
        prefix_value_codes: torch.Tensor,
        local_indices: torch.Tensor,
        selected_value_codes: torch.Tensor,
    ) -> bytes:
        prefix_codes = self._validate_prefix_codes(prefix_value_codes)
        indices_long = self._validate_local_indices(local_indices)
        selected_codes = self._validate_selected_codes(selected_value_codes)
        values = [int(v) for v in prefix_codes.reshape(-1).tolist()]
        values.extend(int(v) for v in indices_long.reshape(-1).tolist())
        values.extend(int(v) for v in selected_codes.reshape(-1).tolist())
        widths = [int(self.value_codec.bits)] * int(self.prefix_components)
        widths.extend([int(self.tail_index_bits)] * int(self.selected_components))
        widths.extend([int(self.value_codec.bits)] * int(self.selected_components))
        return _pack_variable_width_values(values, widths)

    def decode_compact(self, payload: bytes) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        widths = [int(self.value_codec.bits)] * int(self.prefix_components)
        widths.extend([int(self.tail_index_bits)] * int(self.selected_components))
        widths.extend([int(self.value_codec.bits)] * int(self.selected_components))
        values = _unpack_variable_width_values(payload, widths)
        prefix_end = int(self.prefix_components)
        index_end = prefix_end + int(self.selected_components)
        prefix_codes = torch.tensor(values[:prefix_end], dtype=torch.long)
        local_indices = torch.tensor(values[prefix_end:index_end], dtype=torch.long)
        selected_codes = torch.tensor(values[index_end:], dtype=torch.long)
        self._validate_prefix_codes(prefix_codes)
        self._validate_local_indices(local_indices)
        self._validate_selected_codes(selected_codes)
        return prefix_codes, local_indices, selected_codes

    def encoded_compact_num_bytes(self) -> int:
        return (self.encoded_compact_num_bits() + 7) // 8

    def encoded_compact_num_bits(self) -> int:
        return int(self.prefix_components) * int(self.value_codec.bits) + int(self.selected_components) * (
            int(self.tail_index_bits) + int(self.value_codec.bits)
        )

    def scatter_values(
        self,
        prefix_values: torch.Tensor,
        local_indices: torch.Tensor,
        selected_values: torch.Tensor,
    ) -> torch.Tensor:
        prefix = prefix_values.float()
        if prefix.shape != (int(self.prefix_components),):
            raise ValueError("prefix_values shape must match prefix_components")
        indices = self._validate_local_indices(local_indices)
        values = selected_values.float()
        if values.shape != (int(self.selected_components),):
            raise ValueError("selected_values shape must match selected_components")
        out = torch.zeros(int(self.candidate_components), dtype=values.dtype, device=values.device)
        out[: int(self.prefix_components)] = prefix.to(device=values.device)
        out.scatter_(0, (indices + int(self.prefix_components)).to(device=values.device), values)
        return out

    def _validate_prefix_codes(self, value_codes: torch.Tensor) -> torch.Tensor:
        if tuple(value_codes.shape) != (int(self.prefix_components),):
            raise ValueError("prefix value_codes shape must match prefix_components")
        if torch.is_floating_point(value_codes) or torch.is_complex(value_codes):
            raise TypeError("prefix value_codes must be an integer tensor")
        values = value_codes.detach().cpu().to(torch.long)
        if int(values.min().item()) < 0 or int(values.max().item()) >= int(self.value_codec.levels):
            raise ValueError("prefix value code out of range")
        return values

    def _validate_selected_codes(self, value_codes: torch.Tensor) -> torch.Tensor:
        if tuple(value_codes.shape) != (int(self.selected_components),):
            raise ValueError("selected value_codes shape must match selected_components")
        if torch.is_floating_point(value_codes) or torch.is_complex(value_codes):
            raise TypeError("selected value_codes must be an integer tensor")
        values = value_codes.detach().cpu().to(torch.long)
        if int(values.min().item()) < 0 or int(values.max().item()) >= int(self.value_codec.levels):
            raise ValueError("selected value code out of range")
        return values

    def _validate_local_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if tuple(indices.shape) != (int(self.selected_components),):
            raise ValueError("local_indices shape must match selected_components")
        if torch.is_floating_point(indices) or torch.is_complex(indices):
            raise TypeError("local_indices must be an integer tensor")
        values = indices.detach().cpu().to(torch.long)
        if int(values.min().item()) < 0 or int(values.max().item()) >= int(self.tail_components):
            raise ValueError("prefix-topk local index out of range")
        if int(torch.unique(values).numel()) != int(self.selected_components):
            raise ValueError("prefix-topk local indices must be unique")
        return values


def zigzag_indices(size: int, count: int) -> list[tuple[int, int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    max_count = size * size
    if count > max_count:
        raise ValueError("count must be <= size * size")
    pairs = [(y, x) for y in range(size) for x in range(size)]
    pairs.sort(key=lambda item: (item[0] + item[1], item[0]))
    return pairs[:count]


def dct_matrix(size: int, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if size <= 0:
        raise ValueError("size must be positive")
    n = torch.arange(size, device=device, dtype=dtype)
    k = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
    mat = torch.cos(torch.pi * (n + 0.5) * k / float(size))
    mat[0] *= (1.0 / float(size)) ** 0.5
    if size > 1:
        mat[1:] *= (2.0 / float(size)) ** 0.5
    return mat


def dct2_orthonormal(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != values.shape[-2]:
        raise ValueError("dct2_orthonormal expects square spatial grids")
    size = int(values.shape[-1])
    mat = dct_matrix(size, device=values.device, dtype=values.float().dtype)
    x = values.float()
    return torch.matmul(torch.matmul(mat, x), mat.t())


def idct2_orthonormal(coefficients: torch.Tensor) -> torch.Tensor:
    if coefficients.shape[-1] != coefficients.shape[-2]:
        raise ValueError("idct2_orthonormal expects square coefficient grids")
    size = int(coefficients.shape[-1])
    mat = dct_matrix(size, device=coefficients.device, dtype=coefficients.float().dtype)
    x = coefficients.float()
    return torch.matmul(torch.matmul(mat.t(), x), mat)


def project_onto_control_basis(
    control_grid: torch.Tensor,
    basis: torch.Tensor,
    mean: torch.Tensor | None = None,
    *,
    components: int | None = None,
) -> torch.Tensor:
    if control_grid.ndim != 4:
        raise ValueError("control_grid must be BGSS")
    if basis.ndim != 4:
        raise ValueError("basis must be KGSS")
    if tuple(control_grid.shape[1:]) != tuple(basis.shape[1:]):
        raise ValueError(
            f"control_grid shape {tuple(control_grid.shape[1:])} does not match basis shape {tuple(basis.shape[1:])}"
        )
    count = int(basis.shape[0]) if components is None else int(components)
    if count <= 0 or count > int(basis.shape[0]):
        raise ValueError("components must be in [1, basis_count]")
    centered = control_grid.float()
    if mean is not None:
        if tuple(mean.shape) != tuple(basis.shape[1:]):
            raise ValueError(f"mean shape {tuple(mean.shape)} does not match basis spatial shape {tuple(basis.shape[1:])}")
        centered = centered - mean.to(device=centered.device, dtype=centered.dtype).unsqueeze(0)
    flat = centered.flatten(1)
    basis_flat = basis[:count].to(device=flat.device, dtype=flat.dtype).flatten(1)
    return torch.matmul(flat, basis_flat.t())


def reconstruct_from_control_basis(
    coefficients: torch.Tensor,
    basis: torch.Tensor,
    mean: torch.Tensor | None = None,
) -> torch.Tensor:
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be BK")
    if basis.ndim != 4:
        raise ValueError("basis must be KGSS")
    count = int(coefficients.shape[1])
    if count <= 0 or count > int(basis.shape[0]):
        raise ValueError("coefficient count must be in [1, basis_count]")
    coeff_f = coefficients.float()
    basis_f = basis[:count].to(device=coeff_f.device, dtype=coeff_f.dtype).flatten(1)
    recon = torch.matmul(coeff_f, basis_f).reshape(coeff_f.shape[0], *basis.shape[1:])
    if mean is not None:
        if tuple(mean.shape) != tuple(basis.shape[1:]):
            raise ValueError(f"mean shape {tuple(mean.shape)} does not match basis spatial shape {tuple(basis.shape[1:])}")
        recon = recon + mean.to(device=recon.device, dtype=recon.dtype).unsqueeze(0)
    return recon
