from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import count

import torch


@dataclass(frozen=True)
class StaticHuffmanCode:
    """Canonical static Huffman code for semantic VQ tokens.

    This is a CoSER-owned Stage 2 bridge between fixed-width Stage 1 token
    packing and learned/prior-conditioned entropy coding. The code description
    is treated as decoder-known model state, not per-image side information.
    """

    codebook_size: int
    code_lengths: tuple[int, ...]
    codes: tuple[int, ...]

    @classmethod
    def from_counts(
        cls,
        counts: torch.Tensor,
        *,
        smoothing_count: int = 1,
    ) -> StaticHuffmanCode:
        values = _validate_counts(counts)
        if smoothing_count < 0:
            raise ValueError("smoothing_count must be non-negative")
        smoothed = [max(0, int(v)) + int(smoothing_count) for v in values]
        if sum(smoothed) <= 0:
            raise ValueError("at least one positive count is required")
        lengths = _huffman_code_lengths(smoothed)
        codes = _canonical_codes(lengths)
        return cls(codebook_size=len(smoothed), code_lengths=tuple(lengths), codes=tuple(codes))

    @classmethod
    def from_code_lengths(cls, code_lengths: list[int] | tuple[int, ...]) -> StaticHuffmanCode:
        if len(code_lengths) <= 1:
            raise ValueError("code_lengths must contain at least two symbols")
        lengths = [int(v) for v in code_lengths]
        if min(lengths) <= 0:
            raise ValueError("code lengths must be positive")
        return cls(
            codebook_size=len(lengths),
            code_lengths=tuple(lengths),
            codes=tuple(_canonical_codes(lengths)),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StaticHuffmanCode:
        code_lengths = payload.get("code_lengths")
        if not isinstance(code_lengths, list):
            raise TypeError("payload must contain list code_lengths")
        code = cls.from_code_lengths([int(v) for v in code_lengths])
        codebook_size = payload.get("codebook_size")
        if codebook_size is not None and int(codebook_size) != code.codebook_size:
            raise ValueError("codebook_size does not match code_lengths")
        return code

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "static_huffman",
            "version": 0,
            "codebook_size": self.codebook_size,
            "code_lengths": list(self.code_lengths),
        }

    def encode(self, indices: torch.Tensor) -> bytes:
        flat = _flatten_indices(indices, self.codebook_size)
        out = bytearray()
        acc = 0
        bit_count = 0
        for value in flat:
            code = self.codes[value]
            length = self.code_lengths[value]
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
        if any(dim <= 0 for dim in shape):
            raise ValueError("shape dimensions must be positive")
        expected_tokens = math.prod(shape)
        decode_table = self._decode_table()
        values: list[int] = []
        acc = 0
        bit_count = 0
        max_len = max(self.code_lengths)
        for byte in payload:
            acc = (acc << 8) | int(byte)
            bit_count += 8
            while bit_count > 0 and len(values) < expected_tokens:
                matched = False
                upper = min(max_len, bit_count)
                for length in range(1, upper + 1):
                    prefix = (acc >> (bit_count - length)) & ((1 << length) - 1)
                    symbol = decode_table.get((length, prefix))
                    if symbol is None:
                        continue
                    values.append(symbol)
                    bit_count -= length
                    acc &= (1 << bit_count) - 1 if bit_count else 0
                    matched = True
                    break
                if not matched:
                    break
        if len(values) != expected_tokens:
            raise ValueError(f"decoded {len(values)} tokens, expected {expected_tokens}")
        if bit_count and acc != 0:
            raise ValueError("non-zero padding bits in static Huffman payload")
        return torch.tensor(values, dtype=torch.long).reshape(shape)

    def encoded_bits(self, indices: torch.Tensor) -> int:
        flat = _flatten_indices(indices, self.codebook_size)
        return sum(self.code_lengths[value] for value in flat)

    def mean_code_length_bits(self, indices: torch.Tensor) -> float:
        flat = _flatten_indices(indices, self.codebook_size)
        return float(self.encoded_bits(indices)) / float(len(flat))

    def _decode_table(self) -> dict[tuple[int, int], int]:
        return {
            (length, code): symbol
            for symbol, (length, code) in enumerate(zip(self.code_lengths, self.codes))
        }


def _validate_counts(counts: torch.Tensor) -> list[int]:
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D tensor")
    if counts.numel() <= 1:
        raise ValueError("counts must contain at least two symbols")
    if torch.is_floating_point(counts) or torch.is_complex(counts):
        raise TypeError("counts must be an integer tensor")
    values = counts.detach().cpu().to(torch.long)
    if int(values.min().item()) < 0:
        raise ValueError("counts must be non-negative")
    return [int(v) for v in values.tolist()]


def _flatten_indices(indices: torch.Tensor, codebook_size: int) -> list[int]:
    if indices.numel() == 0:
        raise ValueError("indices must be non-empty")
    if torch.is_floating_point(indices) or torch.is_complex(indices):
        raise TypeError("indices must be an integer tensor")
    values = indices.detach().cpu().to(torch.long).reshape(-1)
    min_value = int(values.min().item())
    max_value = int(values.max().item())
    if min_value < 0 or max_value >= codebook_size:
        raise ValueError(
            f"semantic token out of range for codebook_size={codebook_size}: "
            f"min={min_value}, max={max_value}"
        )
    return [int(v) for v in values.tolist()]


def _huffman_code_lengths(counts_: list[int]) -> list[int]:
    tie = count()
    heap: list[tuple[int, int, tuple[int, ...]]] = [
        (weight, next(tie), (symbol,)) for symbol, weight in enumerate(counts_)
    ]
    heapq.heapify(heap)
    lengths = [0 for _ in counts_]
    while len(heap) > 1:
        weight_a, _, symbols_a = heapq.heappop(heap)
        weight_b, _, symbols_b = heapq.heappop(heap)
        for symbol in symbols_a:
            lengths[symbol] += 1
        for symbol in symbols_b:
            lengths[symbol] += 1
        heapq.heappush(heap, (weight_a + weight_b, next(tie), symbols_a + symbols_b))
    return [max(1, length) for length in lengths]


def _canonical_codes(lengths: list[int]) -> list[int]:
    pairs = sorted((length, symbol) for symbol, length in enumerate(lengths))
    codes = [0 for _ in lengths]
    code = 0
    prev_len = 0
    for length, symbol in pairs:
        code <<= length - prev_len
        codes[symbol] = code
        code += 1
        prev_len = length
    return codes
