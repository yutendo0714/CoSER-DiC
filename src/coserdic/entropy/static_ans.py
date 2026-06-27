from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models.entropy_models import pmf_to_quantized_cdf

from .static_huffman import _flatten_indices


@dataclass(frozen=True)
class StaticANSCode:
    """Static categorical ANS code for semantic VQ tokens.

    The categorical CDF is decoder-known model state. The transmitted payload is
    the byte stream returned by CompressAI's ANS backend.
    """

    codebook_size: int
    quantized_cdf: tuple[int, ...]
    cdf_length: int
    offset: int = 0
    precision: int = 16

    @classmethod
    def from_counts(
        cls,
        counts: torch.Tensor,
        *,
        smoothing_count: float = 1.0,
        precision: int = 16,
    ) -> StaticANSCode:
        if smoothing_count < 0:
            raise ValueError("smoothing_count must be non-negative")
        values = torch.as_tensor(counts, dtype=torch.float64).detach().cpu()
        if values.ndim != 1:
            raise ValueError("counts must be a 1D tensor")
        if values.numel() <= 1:
            raise ValueError("counts must contain at least two symbols")
        if float(values.min().item()) < 0:
            raise ValueError("counts must be non-negative")
        weights = values + float(smoothing_count)
        if float(weights.sum().item()) <= 0:
            raise ValueError("at least one positive count is required")
        pmf = (weights / weights.sum()).to(torch.float32)
        cdf = pmf_to_quantized_cdf(pmf, int(precision)).to(torch.int32)
        return cls(
            codebook_size=int(values.numel()),
            quantized_cdf=tuple(int(v) for v in cdf.tolist()),
            cdf_length=int(cdf.numel()),
            offset=0,
            precision=int(precision),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StaticANSCode:
        cdf_raw = payload.get("quantized_cdf")
        if not isinstance(cdf_raw, list):
            raise TypeError("payload must contain list quantized_cdf")
        codebook_size = int(payload["codebook_size"])
        cdf = tuple(int(v) for v in cdf_raw)
        cdf_length = int(payload.get("cdf_length", len(cdf)))
        if cdf_length != len(cdf):
            raise ValueError("cdf_length must match quantized_cdf length")
        if len(cdf) != codebook_size + 1:
            raise ValueError("quantized_cdf must have codebook_size + 1 entries")
        return cls(
            codebook_size=codebook_size,
            quantized_cdf=cdf,
            cdf_length=cdf_length,
            offset=int(payload.get("offset", 0)),
            precision=int(payload.get("precision", 16)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "static_ans",
            "version": 0,
            "codebook_size": self.codebook_size,
            "quantized_cdf": list(self.quantized_cdf),
            "cdf_length": self.cdf_length,
            "offset": self.offset,
            "precision": self.precision,
        }

    def encode(self, indices: torch.Tensor) -> bytes:
        flat = _flatten_indices(indices, self.codebook_size)
        encoder = BufferedRansEncoder()
        encoder.encode_with_indexes(
            flat,
            [0] * len(flat),
            [list(self.quantized_cdf)],
            [self.cdf_length],
            [self.offset],
        )
        return encoder.flush()

    def decode(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        if any(dim <= 0 for dim in shape):
            raise ValueError("shape dimensions must be positive")
        expected_tokens = math.prod(shape)
        decoder = RansDecoder()
        values = decoder.decode_with_indexes(
            payload,
            [0] * expected_tokens,
            [list(self.quantized_cdf)],
            [self.cdf_length],
            [self.offset],
        )
        return torch.tensor(values, dtype=torch.long).reshape(shape)

    def encoded_bits(self, indices: torch.Tensor) -> int:
        return len(self.encode(indices)) * 8

    def mean_code_length_bits(self, indices: torch.Tensor) -> float:
        if indices.numel() == 0:
            raise ValueError("indices must be non-empty")
        return float(self.encoded_bits(indices)) / float(indices.numel())
