from __future__ import annotations

import math
import zlib
from typing import Literal

import torch


SemanticTokenCodec = Literal[
    "raw_uint16_be",
    "fixed_bits",
    "zlib_uint16_be",
    "zlib_fixed_bits",
]


SUPPORTED_SEMANTIC_TOKEN_CODECS: tuple[str, ...] = (
    "raw_uint16_be",
    "fixed_bits",
    "zlib_uint16_be",
    "zlib_fixed_bits",
)


def semantic_bits_per_token(codebook_size: int) -> int:
    if codebook_size <= 1:
        raise ValueError("codebook_size must be greater than 1")
    return int(codebook_size - 1).bit_length()


def encode_semantic_tokens(
    indices: torch.Tensor,
    *,
    codebook_size: int,
    codec: SemanticTokenCodec = "fixed_bits",
) -> bytes:
    """Encode integer semantic VQ tokens into transmitted bytes.

    This is deliberately simple: it is a deterministic Stage-1 bitstream
    baseline, not the learned entropy model used by later CoSER-DiC stages.
    """
    _validate_codec(codec)
    flat = _flatten_validated_indices(indices, codebook_size)
    if codec in {"raw_uint16_be", "zlib_uint16_be"}:
        payload = _pack_uint16_be(flat)
    else:
        payload = _pack_fixed_bits(flat, semantic_bits_per_token(codebook_size))
    if codec.startswith("zlib_"):
        return zlib.compress(payload, level=9)
    return payload


def decode_semantic_tokens(
    payload: bytes,
    *,
    shape: tuple[int, ...],
    codebook_size: int,
    codec: SemanticTokenCodec = "fixed_bits",
) -> torch.Tensor:
    _validate_codec(codec)
    if any(dim <= 0 for dim in shape):
        raise ValueError("shape dimensions must be positive")
    expected_tokens = math.prod(shape)
    raw = zlib.decompress(payload) if codec.startswith("zlib_") else payload
    if codec in {"raw_uint16_be", "zlib_uint16_be"}:
        values = _unpack_uint16_be(raw, expected_tokens)
    else:
        values = _unpack_fixed_bits(
            raw,
            expected_tokens,
            semantic_bits_per_token(codebook_size),
        )
    tokens = torch.tensor(values, dtype=torch.long).reshape(shape)
    _flatten_validated_indices(tokens, codebook_size)
    return tokens


def _validate_codec(codec: str) -> None:
    if codec not in SUPPORTED_SEMANTIC_TOKEN_CODECS:
        raise ValueError(f"unsupported semantic token codec: {codec}")


def _flatten_validated_indices(indices: torch.Tensor, codebook_size: int) -> list[int]:
    if codebook_size > 65536:
        raise ValueError("Stage-1 token packer supports codebook_size <= 65536")
    if indices.numel() == 0:
        raise ValueError("indices must be non-empty")
    if not torch.is_floating_point(indices) and not torch.is_complex(indices):
        values = indices.detach().cpu().to(torch.long).reshape(-1)
    else:
        raise TypeError("indices must be an integer tensor")
    min_value = int(values.min().item())
    max_value = int(values.max().item())
    if min_value < 0 or max_value >= codebook_size:
        raise ValueError(
            f"semantic token out of range for codebook_size={codebook_size}: "
            f"min={min_value}, max={max_value}"
        )
    return [int(v) for v in values.tolist()]


def _pack_uint16_be(values: list[int]) -> bytes:
    return b"".join(value.to_bytes(2, byteorder="big", signed=False) for value in values)


def _unpack_uint16_be(payload: bytes, expected_tokens: int) -> list[int]:
    expected_len = expected_tokens * 2
    if len(payload) != expected_len:
        raise ValueError(f"uint16 payload length mismatch: got {len(payload)}, expected {expected_len}")
    return [
        int.from_bytes(payload[offset : offset + 2], byteorder="big", signed=False)
        for offset in range(0, expected_len, 2)
    ]


def _pack_fixed_bits(values: list[int], bits_per_token: int) -> bytes:
    out = bytearray()
    acc = 0
    bit_count = 0
    for value in values:
        acc = (acc << bits_per_token) | value
        bit_count += bits_per_token
        while bit_count >= 8:
            bit_count -= 8
            out.append((acc >> bit_count) & 0xFF)
            acc &= (1 << bit_count) - 1 if bit_count else 0
    if bit_count:
        out.append((acc << (8 - bit_count)) & 0xFF)
    return bytes(out)


def _unpack_fixed_bits(payload: bytes, expected_tokens: int, bits_per_token: int) -> list[int]:
    expected_len = (expected_tokens * bits_per_token + 7) // 8
    if len(payload) != expected_len:
        raise ValueError(f"fixed-bit payload length mismatch: got {len(payload)}, expected {expected_len}")
    values: list[int] = []
    acc = 0
    bit_count = 0
    mask = (1 << bits_per_token) - 1
    for byte in payload:
        acc = (acc << 8) | byte
        bit_count += 8
        while bit_count >= bits_per_token and len(values) < expected_tokens:
            bit_count -= bits_per_token
            values.append((acc >> bit_count) & mask)
            acc &= (1 << bit_count) - 1 if bit_count else 0
    if len(values) != expected_tokens:
        raise ValueError(f"decoded {len(values)} tokens, expected {expected_tokens}")
    if bit_count and acc != 0:
        raise ValueError("non-zero padding bits in fixed-bit semantic payload")
    return values
