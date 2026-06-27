from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .semantic_tokens import decode_semantic_tokens, encode_semantic_tokens
from .static_huffman import StaticHuffmanCode


ResidualGridCodec = Literal["fixed_bits", "zlib_fixed_bits", "huffman"]


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
