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
        values = _validate_weights(counts)
        if smoothing_count < 0:
            raise ValueError("smoothing_count must be non-negative")
        smoothed = [max(0.0, float(v)) + float(smoothing_count) for v in values]
        if sum(smoothed) <= 0:
            raise ValueError("at least one positive count is required")
        lengths = _huffman_code_lengths(smoothed)
        codes = _canonical_codes(lengths)
        return cls(codebook_size=len(smoothed), code_lengths=tuple(lengths), codes=tuple(codes))

    @classmethod
    def from_weights(cls, weights: torch.Tensor) -> StaticHuffmanCode:
        values = _validate_weights(weights)
        if sum(values) <= 0:
            raise ValueError("at least one positive weight is required")
        lengths = _huffman_code_lengths(values)
        codes = _canonical_codes(lengths)
        return cls(codebook_size=len(values), code_lengths=tuple(lengths), codes=tuple(codes))

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


@dataclass(frozen=True)
class StaticPositionHuffmanCode:
    """Position-conditioned static Huffman code for a fixed semantic token grid."""

    codebook_size: int
    token_shape: tuple[int, int]
    position_codes: tuple[StaticHuffmanCode, ...]

    @classmethod
    def from_counts(
        cls,
        counts: torch.Tensor,
        *,
        smoothing_count: int = 1,
    ) -> StaticPositionHuffmanCode:
        if counts.ndim != 3:
            raise ValueError("position counts must have shape [H, W, K]")
        h, w, k = (int(v) for v in counts.shape)
        codes = tuple(
            StaticHuffmanCode.from_counts(counts[y, x], smoothing_count=smoothing_count)
            for y in range(h)
            for x in range(w)
        )
        return cls(codebook_size=k, token_shape=(h, w), position_codes=codes)

    @classmethod
    def from_position_counts_with_global_backoff(
        cls,
        position_counts: torch.Tensor,
        global_counts: torch.Tensor,
        *,
        backoff_mass: float,
        smoothing_count: float = 0.0,
    ) -> StaticPositionHuffmanCode:
        if position_counts.ndim != 3:
            raise ValueError("position counts must have shape [H, W, K]")
        h, w, k = (int(v) for v in position_counts.shape)
        global_values = torch.as_tensor(global_counts, dtype=torch.float64)
        if global_values.ndim != 1 or global_values.numel() != k:
            raise ValueError("global_counts must have shape [K] matching position_counts")
        if backoff_mass < 0:
            raise ValueError("backoff_mass must be non-negative")
        global_probs = global_values / global_values.sum().clamp_min(1.0)
        codes = []
        for y in range(h):
            for x in range(w):
                weights = (
                    position_counts[y, x].to(torch.float64)
                    + global_probs * float(backoff_mass)
                    + float(smoothing_count)
                )
                codes.append(StaticHuffmanCode.from_weights(weights))
        return cls(codebook_size=k, token_shape=(h, w), position_codes=tuple(codes))

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StaticPositionHuffmanCode:
        token_shape_raw = payload.get("token_shape")
        code_lengths_raw = payload.get("position_code_lengths")
        if not isinstance(token_shape_raw, list) or len(token_shape_raw) != 2:
            raise TypeError("payload must contain token_shape=[H, W]")
        if not isinstance(code_lengths_raw, list):
            raise TypeError("payload must contain list position_code_lengths")
        token_shape = (int(token_shape_raw[0]), int(token_shape_raw[1]))
        expected_positions = token_shape[0] * token_shape[1]
        if len(code_lengths_raw) != expected_positions:
            raise ValueError("position_code_lengths length does not match token_shape")
        position_codes = tuple(
            StaticHuffmanCode.from_code_lengths([int(v) for v in lengths])
            for lengths in code_lengths_raw
        )
        codebook_size = int(payload.get("codebook_size", position_codes[0].codebook_size))
        if any(code.codebook_size != codebook_size for code in position_codes):
            raise ValueError("all position codes must use the same codebook_size")
        return cls(codebook_size=codebook_size, token_shape=token_shape, position_codes=position_codes)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "static_position_huffman",
            "version": 0,
            "codebook_size": self.codebook_size,
            "token_shape": list(self.token_shape),
            "position_code_lengths": [list(code.code_lengths) for code in self.position_codes],
        }

    def encode(self, indices: torch.Tensor) -> bytes:
        values = _flatten_position_indices(indices, self.codebook_size, self.token_shape)
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
        if tuple(int(v) for v in shape) != self.token_shape:
            raise ValueError(f"shape {shape} does not match token_shape {self.token_shape}")
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
                    raise ValueError("truncated static position Huffman payload")
        if byte_index != len(payload):
            raise ValueError("trailing bytes after static position Huffman payload")
        if bit_count and acc != 0:
            raise ValueError("non-zero padding bits in static position Huffman payload")
        return torch.tensor(values, dtype=torch.long).reshape(self.token_shape)

    def encoded_bits(self, indices: torch.Tensor) -> int:
        values = _flatten_position_indices(indices, self.codebook_size, self.token_shape)
        return sum(codebook.code_lengths[value] for value, codebook in zip(values, self.position_codes))

    def mean_code_length_bits(self, indices: torch.Tensor) -> float:
        return float(self.encoded_bits(indices)) / float(math.prod(self.token_shape))


@dataclass(frozen=True)
class StaticLeftContextHuffmanCode:
    """Left-context static Huffman code for a fixed semantic token grid.

    This is a tiny autoregressive entropy-model probe. The context for token
    (y, x) is BOS at x=0, otherwise whether the already decoded left token is
    one of the globally frequent tokens. Code lengths are decoder-known model
    state; the image payload contains only entropy-coded token symbols.
    """

    codebook_size: int
    top_tokens: tuple[int, ...]
    context_codes: tuple[StaticHuffmanCode, ...]

    @classmethod
    def from_samples(
        cls,
        samples: list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        codebook_size: int,
        context_topk: int = 64,
        backoff_mass: float = 256.0,
        smoothing_count: float = 0.0,
    ) -> StaticLeftContextHuffmanCode:
        if context_topk < 0:
            raise ValueError("context_topk must be non-negative")
        if not samples:
            raise ValueError("samples must be non-empty")
        global_counts = torch.zeros(codebook_size, dtype=torch.long)
        validated: list[torch.Tensor] = []
        token_shape: tuple[int, int] | None = None
        for sample in samples:
            if sample.ndim != 2:
                raise ValueError("left-context samples must have shape [H, W]")
            if token_shape is None:
                token_shape = tuple(int(v) for v in sample.shape)
            elif tuple(int(v) for v in sample.shape) != token_shape:
                raise ValueError("all left-context samples must share the same shape")
            _flatten_indices(sample, codebook_size)
            sample_long = sample.detach().cpu().to(torch.long)
            global_counts.scatter_add_(
                0,
                sample_long.reshape(-1),
                torch.ones(sample_long.numel(), dtype=torch.long),
            )
            validated.append(sample_long)
        active = torch.nonzero(global_counts > 0, as_tuple=False).reshape(-1)
        top_count = min(int(context_topk), int(active.numel()))
        if top_count > 0:
            active_counts = global_counts[active]
            order = torch.argsort(active_counts, descending=True, stable=True)[:top_count]
            top_tokens = tuple(int(v) for v in active[order].tolist())
        else:
            top_tokens = ()
        context_counts = torch.zeros(len(top_tokens) + 2, codebook_size, dtype=torch.long)
        lookup = _left_context_lookup(top_tokens)
        for sample in validated:
            for y in range(sample.shape[0]):
                left_token: int | None = None
                for x in range(sample.shape[1]):
                    token = int(sample[y, x].item())
                    context_counts[_left_context_id(left_token, lookup, len(top_tokens)), token] += 1
                    left_token = token
        return cls.from_context_counts(
            context_counts,
            top_tokens=top_tokens,
            global_counts=global_counts,
            backoff_mass=backoff_mass,
            smoothing_count=smoothing_count,
        )

    @classmethod
    def from_context_counts(
        cls,
        context_counts: torch.Tensor,
        *,
        top_tokens: tuple[int, ...] | list[int],
        global_counts: torch.Tensor,
        backoff_mass: float,
        smoothing_count: float = 0.0,
    ) -> StaticLeftContextHuffmanCode:
        if context_counts.ndim != 2:
            raise ValueError("context_counts must have shape [C, K]")
        if backoff_mass < 0:
            raise ValueError("backoff_mass must be non-negative")
        if smoothing_count < 0:
            raise ValueError("smoothing_count must be non-negative")
        top_tuple = tuple(int(v) for v in top_tokens)
        if len(set(top_tuple)) != len(top_tuple):
            raise ValueError("top_tokens must be unique")
        expected_contexts = len(top_tuple) + 2
        if int(context_counts.shape[0]) != expected_contexts:
            raise ValueError("context_counts first dimension must be len(top_tokens) + 2")
        k = int(context_counts.shape[1])
        global_values = torch.as_tensor(global_counts, dtype=torch.float64)
        if global_values.ndim != 1 or int(global_values.numel()) != k:
            raise ValueError("global_counts must have shape [K] matching context_counts")
        if any(token < 0 or token >= k for token in top_tuple):
            raise ValueError("top_tokens contain values outside the codebook")
        global_probs = global_values / global_values.sum().clamp_min(1.0)
        codes = []
        for context_index in range(expected_contexts):
            weights = (
                context_counts[context_index].to(torch.float64)
                + global_probs * float(backoff_mass)
                + float(smoothing_count)
            )
            codes.append(StaticHuffmanCode.from_weights(weights))
        return cls(codebook_size=k, top_tokens=top_tuple, context_codes=tuple(codes))

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StaticLeftContextHuffmanCode:
        top_tokens_raw = payload.get("top_tokens")
        code_lengths_raw = payload.get("context_code_lengths")
        if not isinstance(top_tokens_raw, list):
            raise TypeError("payload must contain list top_tokens")
        if not isinstance(code_lengths_raw, list):
            raise TypeError("payload must contain list context_code_lengths")
        top_tokens = tuple(int(v) for v in top_tokens_raw)
        expected_contexts = len(top_tokens) + 2
        if len(code_lengths_raw) != expected_contexts:
            raise ValueError("context_code_lengths length must be len(top_tokens) + 2")
        context_codes = tuple(
            StaticHuffmanCode.from_code_lengths([int(v) for v in lengths])
            for lengths in code_lengths_raw
        )
        codebook_size = int(payload.get("codebook_size", context_codes[0].codebook_size))
        if any(code.codebook_size != codebook_size for code in context_codes):
            raise ValueError("all context codes must use the same codebook_size")
        if any(token < 0 or token >= codebook_size for token in top_tokens):
            raise ValueError("top_tokens contain values outside the codebook")
        return cls(codebook_size=codebook_size, top_tokens=top_tokens, context_codes=context_codes)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "static_left_context_huffman",
            "version": 0,
            "codebook_size": self.codebook_size,
            "top_tokens": list(self.top_tokens),
            "context_code_lengths": [list(code.code_lengths) for code in self.context_codes],
        }

    def encode(self, indices: torch.Tensor) -> bytes:
        values = _validate_left_context_indices(indices, self.codebook_size)
        lookup = _left_context_lookup(self.top_tokens)
        out = bytearray()
        acc = 0
        bit_count = 0
        for row in values:
            left_token: int | None = None
            for value in row:
                codebook = self.context_codes[_left_context_id(left_token, lookup, len(self.top_tokens))]
                code = codebook.codes[value]
                length = codebook.code_lengths[value]
                acc = (acc << length) | code
                bit_count += length
                while bit_count >= 8:
                    bit_count -= 8
                    out.append((acc >> bit_count) & 0xFF)
                    acc &= (1 << bit_count) - 1 if bit_count else 0
                left_token = value
        if bit_count:
            out.append((acc << (8 - bit_count)) & 0xFF)
        return bytes(out)

    def decode(self, payload: bytes, *, shape: tuple[int, ...]) -> torch.Tensor:
        if len(shape) != 2 or any(dim <= 0 for dim in shape):
            raise ValueError("shape must be a positive [H, W] tuple")
        h, w = (int(v) for v in shape)
        lookup = _left_context_lookup(self.top_tokens)
        values: list[int] = []
        acc = 0
        bit_count = 0
        byte_index = 0
        for _y in range(h):
            left_token: int | None = None
            for _x in range(w):
                codebook = self.context_codes[_left_context_id(left_token, lookup, len(self.top_tokens))]
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
                        left_token = symbol
                        matched = True
                        break
                    if matched:
                        break
                    if byte_index >= len(payload):
                        raise ValueError("truncated static left-context Huffman payload")
        if byte_index != len(payload):
            raise ValueError("trailing bytes after static left-context Huffman payload")
        if bit_count and acc != 0:
            raise ValueError("non-zero padding bits in static left-context Huffman payload")
        return torch.tensor(values, dtype=torch.long).reshape(h, w)

    def encoded_bits(self, indices: torch.Tensor) -> int:
        values = _validate_left_context_indices(indices, self.codebook_size)
        lookup = _left_context_lookup(self.top_tokens)
        bits = 0
        for row in values:
            left_token: int | None = None
            for value in row:
                codebook = self.context_codes[_left_context_id(left_token, lookup, len(self.top_tokens))]
                bits += codebook.code_lengths[value]
                left_token = value
        return int(bits)

    def mean_code_length_bits(self, indices: torch.Tensor) -> float:
        if indices.numel() == 0:
            raise ValueError("indices must be non-empty")
        return float(self.encoded_bits(indices)) / float(indices.numel())


def _validate_weights(weights: torch.Tensor) -> list[float]:
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D tensor")
    if weights.numel() <= 1:
        raise ValueError("weights must contain at least two symbols")
    if torch.is_complex(weights):
        raise TypeError("weights must be real-valued")
    values = weights.detach().cpu().to(torch.float64)
    if float(values.min().item()) < 0:
        raise ValueError("weights must be non-negative")
    return [float(v) for v in values.tolist()]


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


def _flatten_position_indices(
    indices: torch.Tensor,
    codebook_size: int,
    token_shape: tuple[int, int],
) -> list[int]:
    if tuple(int(v) for v in indices.shape) != token_shape:
        raise ValueError(f"indices shape {tuple(indices.shape)} does not match token_shape {token_shape}")
    return _flatten_indices(indices, codebook_size)


def _validate_left_context_indices(indices: torch.Tensor, codebook_size: int) -> list[list[int]]:
    if indices.ndim != 2:
        raise ValueError("left-context Huffman indices must have shape [H, W]")
    values = _flatten_indices(indices, codebook_size)
    h, w = (int(v) for v in indices.shape)
    return [values[offset : offset + w] for offset in range(0, h * w, w)]


def _left_context_lookup(top_tokens: tuple[int, ...]) -> dict[int, int]:
    return {int(token): index + 1 for index, token in enumerate(top_tokens)}


def _left_context_id(
    left_token: int | None,
    top_token_to_context: dict[int, int],
    top_token_count: int,
) -> int:
    if left_token is None:
        return 0
    return top_token_to_context.get(int(left_token), top_token_count + 1)


def _huffman_code_lengths(counts_: list[float]) -> list[int]:
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
