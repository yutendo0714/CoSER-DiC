from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .semantic_tokens import decode_semantic_tokens, encode_semantic_tokens
from .static_huffman import StaticHuffmanCode


ResidualGridCodec = Literal["fixed_bits", "zlib_fixed_bits", "huffman"]


def _pack_huffman_values(
    values: list[int],
    codebooks: list[StaticHuffmanCode] | tuple[StaticHuffmanCode, ...],
    *,
    prefix_bits: tuple[int, ...] = (),
) -> bytes:
    if len(values) != len(codebooks):
        raise ValueError("values and codebooks must have the same length")
    out = bytearray()
    acc = 0
    bit_count = 0
    for bit in prefix_bits:
        bit_value = int(bit)
        if bit_value not in {0, 1}:
            raise ValueError("prefix bits must be 0 or 1")
        acc = (acc << 1) | bit_value
        bit_count += 1
        if bit_count >= 8:
            bit_count -= 8
            out.append((acc >> bit_count) & 0xFF)
            acc &= (1 << bit_count) - 1 if bit_count else 0
    for value, codebook in zip(values, codebooks):
        code = codebook.codes[int(value)]
        length = codebook.code_lengths[int(value)]
        acc = (acc << length) | code
        bit_count += length
        while bit_count >= 8:
            bit_count -= 8
            out.append((acc >> bit_count) & 0xFF)
            acc &= (1 << bit_count) - 1 if bit_count else 0
    if bit_count:
        out.append((acc << (8 - bit_count)) & 0xFF)
    return bytes(out)


def _payload_bits(payload: bytes) -> list[int]:
    bit_values: list[int] = []
    for byte in payload:
        value = int(byte)
        bit_values.extend((value >> shift) & 1 for shift in range(7, -1, -1))
    return bit_values


def _decode_huffman_values_from_bits(
    bit_values: list[int],
    *,
    offset: int,
    codebooks: list[StaticHuffmanCode] | tuple[StaticHuffmanCode, ...],
    padding_label: str,
) -> list[int]:
    decoded: list[int] = []
    for codebook in codebooks:
        table = codebook._decode_table()
        max_len = max(codebook.code_lengths)
        acc = 0
        matched = False
        for length in range(1, max_len + 1):
            if offset + length > len(bit_values):
                break
            acc = (acc << 1) | bit_values[offset + length - 1]
            symbol = table.get((length, acc))
            if symbol is None:
                continue
            decoded.append(symbol)
            offset += length
            matched = True
            break
        if not matched:
            raise ValueError(f"could not decode residual symbol {len(decoded)}")
    if any(bit_values[offset:]):
        raise ValueError(f"non-zero padding bits in {padding_label} payload")
    return decoded


@dataclass(frozen=True)
class UniformResidualGridCode:
    """Uniform quantized residual-grid payload for Stage 3 bootstrapping.

    This codec is intentionally simple and decoder-known. It gives Stage 3 an
    actual transmitted detail stream before a learned residual entropy model is
    trained.
    """

    bits: int = 6
    value_range: float = 0.5
    codec: ResidualGridCodec = "zlib_fixed_bits"

    def __post_init__(self) -> None:
        if not 1 <= int(self.bits) <= 16:
            raise ValueError("bits must be in [1, 16]")
        if float(self.value_range) <= 0:
            raise ValueError("value_range must be positive")
        if self.codec not in {"fixed_bits", "zlib_fixed_bits", "huffman"}:
            raise ValueError(f"unsupported residual grid codec: {self.codec}")

    @property
    def levels(self) -> int:
        return 1 << int(self.bits)

    def quantize(self, residual_grid: torch.Tensor) -> torch.Tensor:
        if residual_grid.numel() == 0:
            raise ValueError("residual_grid must be non-empty")
        clipped = residual_grid.detach().cpu().float().clamp(-float(self.value_range), float(self.value_range))
        normalized = (clipped + float(self.value_range)) / (2.0 * float(self.value_range))
        return torch.round(normalized * float(self.levels - 1)).to(torch.long)

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        values = codes.detach().cpu().to(torch.long)
        if values.numel() == 0:
            raise ValueError("codes must be non-empty")
        min_value = int(values.min().item())
        max_value = int(values.max().item())
        if min_value < 0 or max_value >= self.levels:
            raise ValueError(f"residual code out of range: min={min_value}, max={max_value}, levels={self.levels}")
        normalized = values.float() / float(self.levels - 1)
        return normalized * (2.0 * float(self.value_range)) - float(self.value_range)

    def encode(self, codes: torch.Tensor) -> bytes:
        if self.codec == "huffman":
            raise ValueError("use StaticResidualGridHuffmanCode for huffman residual payloads")
        return encode_semantic_tokens(codes, codebook_size=self.levels, codec=self.codec)

    def decode(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        if self.codec == "huffman":
            raise ValueError("use StaticResidualGridHuffmanCode for huffman residual payloads")
        return decode_semantic_tokens(payload, shape=shape, codebook_size=self.levels, codec=self.codec)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "uniform_residual_grid",
            "version": 0,
            "bits": int(self.bits),
            "value_range": float(self.value_range),
            "payload_codec": self.codec,
        }


@dataclass(frozen=True)
class StaticResidualGridHuffmanCode:
    """Static Huffman code for uniformly quantized residual-grid symbols."""

    quantizer: UniformResidualGridCode
    huffman: StaticHuffmanCode

    @classmethod
    def from_counts(
        cls,
        counts: torch.Tensor,
        *,
        bits: int,
        value_range: float,
        smoothing_count: int = 1,
    ) -> StaticResidualGridHuffmanCode:
        quantizer = UniformResidualGridCode(bits=bits, value_range=value_range, codec="huffman")
        values = torch.as_tensor(counts)
        if values.ndim != 1 or int(values.numel()) != quantizer.levels:
            raise ValueError("counts must have shape [2 ** bits]")
        huffman = StaticHuffmanCode.from_counts(values, smoothing_count=smoothing_count)
        return cls(quantizer=quantizer, huffman=huffman)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StaticResidualGridHuffmanCode:
        if payload.get("codec") != "static_residual_grid_huffman":
            raise ValueError("payload codec must be static_residual_grid_huffman")
        quantizer = UniformResidualGridCode(
            bits=int(payload["bits"]),
            value_range=float(payload["value_range"]),
            codec="huffman",
        )
        huffman_payload = payload.get("huffman")
        if not isinstance(huffman_payload, dict):
            raise TypeError("payload must contain huffman dict")
        huffman = StaticHuffmanCode.from_dict(huffman_payload)
        if huffman.codebook_size != quantizer.levels:
            raise ValueError("huffman codebook_size does not match residual levels")
        return cls(quantizer=quantizer, huffman=huffman)

    @property
    def bits(self) -> int:
        return int(self.quantizer.bits)

    @property
    def value_range(self) -> float:
        return float(self.quantizer.value_range)

    @property
    def levels(self) -> int:
        return int(self.quantizer.levels)

    def quantize(self, residual_grid: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(residual_grid)

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        return self.quantizer.dequantize(codes)

    def encode(self, codes: torch.Tensor) -> bytes:
        return self.huffman.encode(codes)

    def decode(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        return self.huffman.decode(payload, shape=shape)

    def encoded_bits(self, codes: torch.Tensor) -> int:
        return self.huffman.encoded_bits(codes)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "static_residual_grid_huffman",
            "version": 0,
            "bits": int(self.bits),
            "value_range": float(self.value_range),
            "payload_codec": "huffman",
            "huffman": self.huffman.to_dict(),
        }


@dataclass(frozen=True)
class StaticResidualGridPositionHuffmanCode:
    """Channel/position-conditioned static Huffman code for residual grids."""

    quantizer: UniformResidualGridCode
    detail_shape: tuple[int, int, int]
    position_codes: tuple[StaticHuffmanCode, ...]

    @classmethod
    def from_counts(
        cls,
        counts: torch.Tensor,
        *,
        bits: int,
        value_range: float,
        smoothing_count: int = 1,
    ) -> StaticResidualGridPositionHuffmanCode:
        quantizer = UniformResidualGridCode(bits=bits, value_range=value_range, codec="huffman")
        values = torch.as_tensor(counts)
        if values.ndim != 4:
            raise ValueError("counts must have shape [C, H, W, 2 ** bits]")
        c, h, w, k = (int(v) for v in values.shape)
        if k != quantizer.levels:
            raise ValueError("counts last dimension must be 2 ** bits")
        codes = tuple(
            StaticHuffmanCode.from_counts(values[channel, y, x], smoothing_count=smoothing_count)
            for channel in range(c)
            for y in range(h)
            for x in range(w)
        )
        return cls(quantizer=quantizer, detail_shape=(c, h, w), position_codes=codes)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StaticResidualGridPositionHuffmanCode:
        if payload.get("codec") != "static_residual_grid_position_huffman":
            raise ValueError("payload codec must be static_residual_grid_position_huffman")
        quantizer = UniformResidualGridCode(
            bits=int(payload["bits"]),
            value_range=float(payload["value_range"]),
            codec="huffman",
        )
        shape_raw = payload.get("detail_shape")
        lengths_raw = payload.get("position_code_lengths")
        if not isinstance(shape_raw, list) or len(shape_raw) != 3:
            raise TypeError("payload must contain detail_shape=[C, H, W]")
        if not isinstance(lengths_raw, list):
            raise TypeError("payload must contain list position_code_lengths")
        detail_shape = tuple(int(v) for v in shape_raw)
        expected_positions = detail_shape[0] * detail_shape[1] * detail_shape[2]
        if len(lengths_raw) != expected_positions:
            raise ValueError("position_code_lengths length does not match detail_shape")
        position_codes = tuple(
            StaticHuffmanCode.from_code_lengths([int(v) for v in lengths])
            for lengths in lengths_raw
        )
        if any(code.codebook_size != quantizer.levels for code in position_codes):
            raise ValueError("all position codes must match residual levels")
        return cls(quantizer=quantizer, detail_shape=detail_shape, position_codes=position_codes)

    @property
    def bits(self) -> int:
        return int(self.quantizer.bits)

    @property
    def value_range(self) -> float:
        return float(self.quantizer.value_range)

    @property
    def levels(self) -> int:
        return int(self.quantizer.levels)

    def quantize(self, residual_grid: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(residual_grid)

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        return self.quantizer.dequantize(codes)

    def _flatten_codes(self, codes: torch.Tensor) -> list[int]:
        values = codes.detach().cpu().to(torch.long)
        if tuple(values.shape) != self.detail_shape:
            raise ValueError(f"codes shape {tuple(values.shape)} does not match detail_shape {self.detail_shape}")
        if values.numel() == 0:
            raise ValueError("codes must be non-empty")
        min_value = int(values.min().item())
        max_value = int(values.max().item())
        if min_value < 0 or max_value >= self.levels:
            raise ValueError(f"residual code out of range: min={min_value}, max={max_value}, levels={self.levels}")
        return [int(v) for v in values.reshape(-1).tolist()]

    def encode(self, codes: torch.Tensor) -> bytes:
        values = self._flatten_codes(codes)
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
        if tuple(shape) != self.detail_shape:
            raise ValueError(f"shape {shape} does not match detail_shape {self.detail_shape}")
        bit_values: list[int] = []
        for byte in payload:
            value = int(byte)
            bit_values.extend((value >> shift) & 1 for shift in range(7, -1, -1))
        offset = 0
        decoded: list[int] = []
        for codebook in self.position_codes:
            table = codebook._decode_table()
            max_len = max(codebook.code_lengths)
            acc = 0
            matched = False
            for length in range(1, max_len + 1):
                if offset + length > len(bit_values):
                    break
                acc = (acc << 1) | bit_values[offset + length - 1]
                symbol = table.get((length, acc))
                if symbol is None:
                    continue
                decoded.append(symbol)
                offset += length
                matched = True
                break
            if not matched:
                raise ValueError(f"could not decode residual symbol {len(decoded)}")
        if any(bit_values[offset:]):
            raise ValueError("non-zero padding bits in residual position Huffman payload")
        return torch.tensor(decoded, dtype=torch.long).reshape(self.detail_shape)

    def encoded_bits(self, codes: torch.Tensor) -> int:
        values = self._flatten_codes(codes)
        return sum(codebook.code_lengths[value] for value, codebook in zip(values, self.position_codes))

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "static_residual_grid_position_huffman",
            "version": 0,
            "bits": int(self.bits),
            "value_range": float(self.value_range),
            "payload_codec": "position_huffman",
            "detail_shape": list(self.detail_shape),
            "position_code_lengths": [list(code.code_lengths) for code in self.position_codes],
        }


@dataclass(frozen=True)
class StaticResidualGridSemanticPositionHuffmanCode:
    """Semantic-token-group and position-conditioned residual-grid Huffman code.

    The semantic token stream is already transmitted by Stage 2. This codec
    uses the decoded semantic token grid as decoder-known context for the detail
    residual stream, so no per-image side information is added.
    """

    quantizer: UniformResidualGridCode
    detail_shape: tuple[int, int, int]
    semantic_shape: tuple[int, int]
    token_to_group: tuple[int, ...]
    group_count: int
    position_group_codes: tuple[StaticHuffmanCode, ...]

    @classmethod
    def from_counts(
        cls,
        counts: torch.Tensor,
        *,
        bits: int,
        value_range: float,
        semantic_shape: tuple[int, int],
        token_to_group: list[int] | tuple[int, ...],
        smoothing_count: int = 1,
    ) -> StaticResidualGridSemanticPositionHuffmanCode:
        quantizer = UniformResidualGridCode(bits=bits, value_range=value_range, codec="huffman")
        values = torch.as_tensor(counts)
        if values.ndim != 5:
            raise ValueError("counts must have shape [C, H, W, G, 2 ** bits]")
        c, h, w, g, k = (int(v) for v in values.shape)
        if k != quantizer.levels:
            raise ValueError("counts last dimension must be 2 ** bits")
        if g <= 0:
            raise ValueError("semantic group count must be positive")
        mapping = tuple(int(v) for v in token_to_group)
        if not mapping:
            raise ValueError("token_to_group must be non-empty")
        if min(mapping) < 0 or max(mapping) >= g:
            raise ValueError("token_to_group values must be in [0, group_count)")
        codes = tuple(
            StaticHuffmanCode.from_counts(values[channel, y, x, group], smoothing_count=smoothing_count)
            for channel in range(c)
            for y in range(h)
            for x in range(w)
            for group in range(g)
        )
        return cls(
            quantizer=quantizer,
            detail_shape=(c, h, w),
            semantic_shape=(int(semantic_shape[0]), int(semantic_shape[1])),
            token_to_group=mapping,
            group_count=g,
            position_group_codes=codes,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StaticResidualGridSemanticPositionHuffmanCode:
        if payload.get("codec") != "static_residual_grid_semantic_position_huffman":
            raise ValueError("payload codec must be static_residual_grid_semantic_position_huffman")
        quantizer = UniformResidualGridCode(
            bits=int(payload["bits"]),
            value_range=float(payload["value_range"]),
            codec="huffman",
        )
        detail_shape_raw = payload.get("detail_shape")
        semantic_shape_raw = payload.get("semantic_shape")
        token_to_group_raw = payload.get("token_to_group")
        code_lengths_raw = payload.get("position_group_code_lengths")
        if not isinstance(detail_shape_raw, list) or len(detail_shape_raw) != 3:
            raise TypeError("payload must contain detail_shape=[C, H, W]")
        if not isinstance(semantic_shape_raw, list) or len(semantic_shape_raw) != 2:
            raise TypeError("payload must contain semantic_shape=[H, W]")
        if not isinstance(token_to_group_raw, list):
            raise TypeError("payload must contain token_to_group list")
        if not isinstance(code_lengths_raw, list):
            raise TypeError("payload must contain position_group_code_lengths list")
        detail_shape = tuple(int(v) for v in detail_shape_raw)
        semantic_shape = (int(semantic_shape_raw[0]), int(semantic_shape_raw[1]))
        group_count = int(payload["group_count"])
        expected_codes = detail_shape[0] * detail_shape[1] * detail_shape[2] * group_count
        if len(code_lengths_raw) != expected_codes:
            raise ValueError("position_group_code_lengths length does not match detail_shape/group_count")
        mapping = tuple(int(v) for v in token_to_group_raw)
        if not mapping:
            raise ValueError("token_to_group must be non-empty")
        if min(mapping) < 0 or max(mapping) >= group_count:
            raise ValueError("token_to_group values must be in [0, group_count)")
        position_group_codes = tuple(
            StaticHuffmanCode.from_code_lengths([int(v) for v in lengths])
            for lengths in code_lengths_raw
        )
        if any(code.codebook_size != quantizer.levels for code in position_group_codes):
            raise ValueError("all position/group codes must match residual levels")
        return cls(
            quantizer=quantizer,
            detail_shape=detail_shape,
            semantic_shape=semantic_shape,
            token_to_group=mapping,
            group_count=group_count,
            position_group_codes=position_group_codes,
        )

    @property
    def bits(self) -> int:
        return int(self.quantizer.bits)

    @property
    def value_range(self) -> float:
        return float(self.quantizer.value_range)

    @property
    def levels(self) -> int:
        return int(self.quantizer.levels)

    def quantize(self, residual_grid: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(residual_grid)

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        return self.quantizer.dequantize(codes)

    def _semantic_groups_for_detail(self, semantic_indices: torch.Tensor) -> torch.Tensor:
        values = semantic_indices.detach().cpu().to(torch.long)
        if tuple(values.shape) != self.semantic_shape:
            raise ValueError(f"semantic_indices shape {tuple(values.shape)} does not match {self.semantic_shape}")
        min_token = int(values.min().item())
        max_token = int(values.max().item())
        if min_token < 0 or max_token >= len(self.token_to_group):
            raise ValueError("semantic token out of token_to_group range")
        group_lookup = torch.tensor(self.token_to_group, dtype=torch.long)
        semantic_groups = group_lookup[values]
        _, detail_h, detail_w = self.detail_shape
        if (detail_h, detail_w) == self.semantic_shape:
            return semantic_groups
        y_index = torch.div(torch.arange(detail_h) * self.semantic_shape[0], detail_h, rounding_mode="floor")
        x_index = torch.div(torch.arange(detail_w) * self.semantic_shape[1], detail_w, rounding_mode="floor")
        return semantic_groups[y_index][:, x_index]

    def _codebook_index(self, channel: int, y: int, x: int, group: int) -> int:
        _, h, w = self.detail_shape
        return (((int(channel) * h + int(y)) * w + int(x)) * self.group_count) + int(group)

    def _flatten_codes_and_groups(self, codes: torch.Tensor, semantic_indices: torch.Tensor) -> tuple[list[int], list[int]]:
        values = codes.detach().cpu().to(torch.long)
        if tuple(values.shape) != self.detail_shape:
            raise ValueError(f"codes shape {tuple(values.shape)} does not match detail_shape {self.detail_shape}")
        if values.numel() == 0:
            raise ValueError("codes must be non-empty")
        min_value = int(values.min().item())
        max_value = int(values.max().item())
        if min_value < 0 or max_value >= self.levels:
            raise ValueError(f"residual code out of range: min={min_value}, max={max_value}, levels={self.levels}")
        semantic_groups = self._semantic_groups_for_detail(semantic_indices)
        flattened_values: list[int] = []
        codebook_indices: list[int] = []
        c, h, w = self.detail_shape
        for channel in range(c):
            for y in range(h):
                for x in range(w):
                    group = int(semantic_groups[y, x].item())
                    flattened_values.append(int(values[channel, y, x].item()))
                    codebook_indices.append(self._codebook_index(channel, y, x, group))
        return flattened_values, codebook_indices

    def encode(self, codes: torch.Tensor, *, semantic_indices: torch.Tensor) -> bytes:
        values, codebook_indices = self._flatten_codes_and_groups(codes, semantic_indices)
        out = bytearray()
        acc = 0
        bit_count = 0
        for value, codebook_index in zip(values, codebook_indices):
            codebook = self.position_group_codes[codebook_index]
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

    def decode(self, payload: bytes, *, shape: tuple[int, ...], semantic_indices: torch.Tensor) -> torch.Tensor:
        if tuple(shape) != self.detail_shape:
            raise ValueError(f"shape {shape} does not match detail_shape {self.detail_shape}")
        semantic_groups = self._semantic_groups_for_detail(semantic_indices)
        bit_values: list[int] = []
        for byte in payload:
            value = int(byte)
            bit_values.extend((value >> shift) & 1 for shift in range(7, -1, -1))
        offset = 0
        decoded: list[int] = []
        c, h, w = self.detail_shape
        for channel in range(c):
            for y in range(h):
                for x in range(w):
                    group = int(semantic_groups[y, x].item())
                    codebook = self.position_group_codes[self._codebook_index(channel, y, x, group)]
                    table = codebook._decode_table()
                    max_len = max(codebook.code_lengths)
                    acc = 0
                    matched = False
                    for length in range(1, max_len + 1):
                        if offset + length > len(bit_values):
                            break
                        acc = (acc << 1) | bit_values[offset + length - 1]
                        symbol = table.get((length, acc))
                        if symbol is None:
                            continue
                        decoded.append(symbol)
                        offset += length
                        matched = True
                        break
                    if not matched:
                        raise ValueError(f"could not decode semantic residual symbol {len(decoded)}")
        if any(bit_values[offset:]):
            raise ValueError("non-zero padding bits in semantic residual Huffman payload")
        return torch.tensor(decoded, dtype=torch.long).reshape(self.detail_shape)

    def encoded_bits(self, codes: torch.Tensor, *, semantic_indices: torch.Tensor) -> int:
        values, codebook_indices = self._flatten_codes_and_groups(codes, semantic_indices)
        return sum(self.position_group_codes[index].code_lengths[value] for value, index in zip(values, codebook_indices))

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "static_residual_grid_semantic_position_huffman",
            "version": 0,
            "bits": int(self.bits),
            "value_range": float(self.value_range),
            "payload_codec": "semantic_position_huffman",
            "detail_shape": list(self.detail_shape),
            "semantic_shape": list(self.semantic_shape),
            "group_count": int(self.group_count),
            "token_to_group": [int(v) for v in self.token_to_group],
            "position_group_code_lengths": [list(code.code_lengths) for code in self.position_group_codes],
        }


@dataclass(frozen=True)
class StaticResidualGridHybridHuffmanCode:
    """Per-image selector between position and semantic-position Huffman.

    The selected mode is explicitly transmitted as the first payload bit:
    0 = position Huffman, 1 = semantic-position Huffman. The encoder chooses by
    actual payload byte length after adding that mode bit.
    """

    position_code: StaticResidualGridPositionHuffmanCode
    semantic_position_code: StaticResidualGridSemanticPositionHuffmanCode

    def __post_init__(self) -> None:
        if self.position_code.bits != self.semantic_position_code.bits:
            raise ValueError("hybrid codes must use the same quantizer bits")
        if abs(self.position_code.value_range - self.semantic_position_code.value_range) > 1.0e-9:
            raise ValueError("hybrid codes must use the same value_range")
        if self.position_code.detail_shape != self.semantic_position_code.detail_shape:
            raise ValueError("hybrid codes must use the same detail_shape")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StaticResidualGridHybridHuffmanCode:
        if payload.get("codec") != "static_residual_grid_hybrid_huffman":
            raise ValueError("payload codec must be static_residual_grid_hybrid_huffman")
        position_payload = payload.get("position_code")
        semantic_payload = payload.get("semantic_position_code")
        if not isinstance(position_payload, dict):
            raise TypeError("payload must contain position_code dict")
        if not isinstance(semantic_payload, dict):
            raise TypeError("payload must contain semantic_position_code dict")
        return cls(
            position_code=StaticResidualGridPositionHuffmanCode.from_dict(position_payload),
            semantic_position_code=StaticResidualGridSemanticPositionHuffmanCode.from_dict(semantic_payload),
        )

    @property
    def bits(self) -> int:
        return int(self.position_code.bits)

    @property
    def value_range(self) -> float:
        return float(self.position_code.value_range)

    @property
    def levels(self) -> int:
        return int(self.position_code.levels)

    @property
    def detail_shape(self) -> tuple[int, int, int]:
        return self.position_code.detail_shape

    def quantize(self, residual_grid: torch.Tensor) -> torch.Tensor:
        return self.position_code.quantize(residual_grid)

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        return self.position_code.dequantize(codes)

    def _semantic_codebooks(self, semantic_indices: torch.Tensor) -> list[StaticHuffmanCode]:
        semantic_groups = self.semantic_position_code._semantic_groups_for_detail(semantic_indices)
        codebooks: list[StaticHuffmanCode] = []
        c, h, w = self.detail_shape
        for channel in range(c):
            for y in range(h):
                for x in range(w):
                    group = int(semantic_groups[y, x].item())
                    codebooks.append(
                        self.semantic_position_code.position_group_codes[
                            self.semantic_position_code._codebook_index(channel, y, x, group)
                        ]
                    )
        return codebooks

    @staticmethod
    def _payload_bytes_for_bits(bit_count: int) -> int:
        return (int(bit_count) + 7) // 8

    def _select_mode(self, codes: torch.Tensor, semantic_indices: torch.Tensor) -> tuple[int, list[int], list[StaticHuffmanCode]]:
        position_values = self.position_code._flatten_codes(codes)
        position_codebooks = list(self.position_code.position_codes)
        semantic_values, semantic_codebook_indices = self.semantic_position_code._flatten_codes_and_groups(
            codes,
            semantic_indices,
        )
        semantic_codebooks = [
            self.semantic_position_code.position_group_codes[index] for index in semantic_codebook_indices
        ]
        if position_values != semantic_values:
            raise ValueError("position and semantic-position flattening disagree")
        position_bits = sum(
            codebook.code_lengths[value] for value, codebook in zip(position_values, position_codebooks)
        )
        semantic_bits = sum(
            codebook.code_lengths[value] for value, codebook in zip(semantic_values, semantic_codebooks)
        )
        position_bytes = self._payload_bytes_for_bits(position_bits + 1)
        semantic_bytes = self._payload_bytes_for_bits(semantic_bits + 1)
        if semantic_bytes < position_bytes or (semantic_bytes == position_bytes and semantic_bits < position_bits):
            return 1, semantic_values, semantic_codebooks
        return 0, position_values, position_codebooks

    def encode(self, codes: torch.Tensor, *, semantic_indices: torch.Tensor) -> bytes:
        mode, values, codebooks = self._select_mode(codes, semantic_indices)
        return _pack_huffman_values(values, codebooks, prefix_bits=(mode,))

    def select_mode(self, codes: torch.Tensor, *, semantic_indices: torch.Tensor) -> int:
        mode, _, _ = self._select_mode(codes, semantic_indices)
        return int(mode)

    def decode(self, payload: bytes, *, shape: tuple[int, ...], semantic_indices: torch.Tensor) -> torch.Tensor:
        if tuple(shape) != self.detail_shape:
            raise ValueError(f"shape {shape} does not match detail_shape {self.detail_shape}")
        bit_values = _payload_bits(payload)
        if not bit_values:
            raise ValueError("hybrid residual payload is empty")
        mode = int(bit_values[0])
        if mode == 0:
            codebooks: list[StaticHuffmanCode] | tuple[StaticHuffmanCode, ...] = self.position_code.position_codes
        elif mode == 1:
            codebooks = self._semantic_codebooks(semantic_indices)
        else:
            raise ValueError(f"unknown hybrid residual mode: {mode}")
        decoded = _decode_huffman_values_from_bits(
            bit_values,
            offset=1,
            codebooks=codebooks,
            padding_label="hybrid residual Huffman",
        )
        return torch.tensor(decoded, dtype=torch.long).reshape(self.detail_shape)

    def encoded_bits(self, codes: torch.Tensor, *, semantic_indices: torch.Tensor) -> int:
        _, values, codebooks = self._select_mode(codes, semantic_indices)
        return 1 + sum(codebook.code_lengths[value] for value, codebook in zip(values, codebooks))

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "static_residual_grid_hybrid_huffman",
            "version": 0,
            "bits": int(self.bits),
            "value_range": float(self.value_range),
            "payload_codec": "hybrid_huffman",
            "position_code": self.position_code.to_dict(),
            "semantic_position_code": self.semantic_position_code.to_dict(),
        }
