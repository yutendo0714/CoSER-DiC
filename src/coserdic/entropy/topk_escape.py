from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .static_huffman import StaticHuffmanCode, _flatten_indices


@dataclass(frozen=True)
class TopKEscapeHuffmanCode:
    """Top-k rank / escape Huffman bridge for learned token priors.

    The neural prior is decoder-known model state. For each token, the decoder
    rebuilds the same top-k candidate list from already decoded tokens. The
    payload stores a Huffman-coded event:

    - rank 0..topk-1 if the target token is in the rebuilt top-k list
    - ESCAPE otherwise, followed by a fixed-width raw token id

    This is intentionally simpler than RDVQ's top-k tensor-rANS backend, but it
    exercises the same causal decoder boundary with real transmitted bytes.
    """

    codebook_size: int
    topk: int
    event_code: StaticHuffmanCode
    token_bits: int

    @classmethod
    def from_event_counts(
        cls,
        counts: torch.Tensor,
        *,
        codebook_size: int,
        topk: int,
        smoothing_count: int = 1,
    ) -> TopKEscapeHuffmanCode:
        if topk <= 0:
            raise ValueError("topk must be positive")
        if codebook_size <= 1:
            raise ValueError("codebook_size must be greater than 1")
        if topk >= codebook_size:
            raise ValueError("topk must be smaller than codebook_size so ESCAPE remains meaningful")
        values = torch.as_tensor(counts)
        if values.ndim != 1 or int(values.numel()) != int(topk) + 1:
            raise ValueError("counts must have shape [topk + 1]")
        event_code = StaticHuffmanCode.from_counts(values, smoothing_count=smoothing_count)
        return cls(
            codebook_size=int(codebook_size),
            topk=int(topk),
            event_code=event_code,
            token_bits=int(math.ceil(math.log2(codebook_size))),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TopKEscapeHuffmanCode:
        topk = int(payload["topk"])
        codebook_size = int(payload["codebook_size"])
        event_code_lengths = payload.get("event_code_lengths")
        if not isinstance(event_code_lengths, list):
            raise TypeError("payload must contain list event_code_lengths")
        if len(event_code_lengths) != topk + 1:
            raise ValueError("event_code_lengths must have length topk + 1")
        event_code = StaticHuffmanCode.from_code_lengths([int(v) for v in event_code_lengths])
        token_bits = int(payload.get("token_bits", math.ceil(math.log2(codebook_size))))
        return cls(
            codebook_size=codebook_size,
            topk=topk,
            event_code=event_code,
            token_bits=token_bits,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": "learned_topk_escape_huffman",
            "version": 0,
            "codebook_size": self.codebook_size,
            "topk": self.topk,
            "token_bits": self.token_bits,
            "event_code_lengths": list(self.event_code.code_lengths),
        }

    def events_from_targets(self, targets: torch.Tensor, topk_indices: torch.Tensor) -> list[int]:
        target_values, topk_rows = _validate_targets_and_topk(
            targets,
            topk_indices,
            codebook_size=self.codebook_size,
            topk=self.topk,
        )
        return [_event_for_target(target, row, self.topk) for target, row in zip(target_values, topk_rows)]

    def encode(self, targets: torch.Tensor, topk_indices: torch.Tensor) -> bytes:
        target_values, topk_rows = _validate_targets_and_topk(
            targets,
            topk_indices,
            codebook_size=self.codebook_size,
            topk=self.topk,
        )
        writer = _BitWriter()
        for target, row in zip(target_values, topk_rows):
            event = _event_for_target(target, row, self.topk)
            writer.write_bits(self.event_code.codes[event], self.event_code.code_lengths[event])
            if event == self.topk:
                writer.write_bits(target, self.token_bits)
        return writer.finish()

    def decode(self, payload: bytes, *, topk_indices: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        if any(dim <= 0 for dim in shape):
            raise ValueError("shape dimensions must be positive")
        expected_tokens = math.prod(shape)
        topk_rows = _validate_topk_rows(topk_indices, expected_tokens=expected_tokens, codebook_size=self.codebook_size, topk=self.topk)
        reader = _BitReader(payload)
        values: list[int] = []
        for row in topk_rows:
            values.append(self._decode_next_token(reader, row))
        reader.assert_only_zero_padding()
        return torch.tensor(values, dtype=torch.long).reshape(shape)

    def decode_with_provider(
        self,
        payload: bytes,
        *,
        shape: tuple[int, ...],
        topk_provider: Callable[[int, tuple[int, ...]], Sequence[int]],
    ) -> torch.Tensor:
        if len(shape) <= 0 or any(dim <= 0 for dim in shape):
            raise ValueError("shape dimensions must be positive")
        expected_tokens = math.prod(shape)
        reader = _BitReader(payload)
        values: list[int] = []
        for index in range(expected_tokens):
            row = [int(v) for v in topk_provider(index, tuple(values))]
            if len(row) != self.topk:
                raise ValueError("topk_provider must return exactly topk entries")
            _validate_topk_row(row, codebook_size=self.codebook_size)
            values.append(self._decode_next_token(reader, row))
        reader.assert_only_zero_padding()
        return torch.tensor(values, dtype=torch.long).reshape(shape)

    def encoded_bits(self, targets: torch.Tensor, topk_indices: torch.Tensor) -> int:
        events = self.events_from_targets(targets, topk_indices)
        bits = 0
        for event in events:
            bits += self.event_code.code_lengths[event]
            if event == self.topk:
                bits += self.token_bits
        return int(bits)

    def escape_count(self, targets: torch.Tensor, topk_indices: torch.Tensor) -> int:
        return sum(1 for event in self.events_from_targets(targets, topk_indices) if event == self.topk)

    def _decode_next_token(self, reader: _BitReader, topk_row: Sequence[int]) -> int:
        event = reader.read_huffman(self.event_code)
        if event < self.topk:
            return int(topk_row[event])
        token = reader.read_bits(self.token_bits)
        if token >= self.codebook_size:
            raise ValueError(f"escaped token {token} exceeds codebook_size={self.codebook_size}")
        return int(token)


def count_topk_escape_events(
    targets: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    codebook_size: int,
    topk: int,
) -> torch.Tensor:
    target_values, topk_rows = _validate_targets_and_topk(
        targets,
        topk_indices,
        codebook_size=codebook_size,
        topk=topk,
    )
    counts = torch.zeros(int(topk) + 1, dtype=torch.long)
    for target, row in zip(target_values, topk_rows):
        counts[_event_for_target(target, row, topk)] += 1
    return counts


def _event_for_target(target: int, topk_row: Sequence[int], topk: int) -> int:
    for rank, candidate in enumerate(topk_row):
        if int(candidate) == int(target):
            return int(rank)
    return int(topk)


def _validate_targets_and_topk(
    targets: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    codebook_size: int,
    topk: int,
) -> tuple[list[int], list[list[int]]]:
    target_values = _flatten_indices(targets, codebook_size)
    topk_rows = _validate_topk_rows(topk_indices, expected_tokens=len(target_values), codebook_size=codebook_size, topk=topk)
    return target_values, topk_rows


def _validate_topk_rows(
    topk_indices: torch.Tensor,
    *,
    expected_tokens: int,
    codebook_size: int,
    topk: int,
) -> list[list[int]]:
    values = torch.as_tensor(topk_indices)
    if values.ndim < 2 or int(values.shape[-1]) != int(topk):
        raise ValueError("topk_indices must have shape [..., topk]")
    if math.prod(tuple(int(v) for v in values.shape[:-1])) != int(expected_tokens):
        raise ValueError("topk_indices leading shape must match target token count")
    flat = values.detach().cpu().to(torch.long).reshape(expected_tokens, topk)
    rows = [[int(v) for v in row.tolist()] for row in flat]
    for row in rows:
        _validate_topk_row(row, codebook_size=codebook_size)
    return rows


def _validate_topk_row(row: Sequence[int], *, codebook_size: int) -> None:
    for value in row:
        if int(value) < 0 or int(value) >= int(codebook_size):
            raise ValueError("topk row contains token outside codebook range")


class _BitWriter:
    def __init__(self) -> None:
        self._out = bytearray()
        self._acc = 0
        self._bit_count = 0

    def write_bits(self, value: int, length: int) -> None:
        if length < 0:
            raise ValueError("bit length must be non-negative")
        if length == 0:
            return
        if value < 0 or value >= (1 << length):
            raise ValueError("value does not fit in requested bit length")
        self._acc = (self._acc << length) | int(value)
        self._bit_count += int(length)
        while self._bit_count >= 8:
            self._bit_count -= 8
            self._out.append((self._acc >> self._bit_count) & 0xFF)
            self._acc &= (1 << self._bit_count) - 1 if self._bit_count else 0

    def finish(self) -> bytes:
        if self._bit_count:
            self._out.append((self._acc << (8 - self._bit_count)) & 0xFF)
            self._acc = 0
            self._bit_count = 0
        return bytes(self._out)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._bit_pos = 0

    @property
    def remaining_bits(self) -> int:
        return len(self._payload) * 8 - self._bit_pos

    def read_bits(self, length: int) -> int:
        if length < 0:
            raise ValueError("bit length must be non-negative")
        if self.remaining_bits < length:
            raise ValueError("truncated top-k escape payload")
        value = 0
        for _ in range(length):
            byte = self._payload[self._bit_pos // 8]
            shift = 7 - (self._bit_pos % 8)
            value = (value << 1) | ((byte >> shift) & 1)
            self._bit_pos += 1
        return int(value)

    def read_huffman(self, code: StaticHuffmanCode) -> int:
        table = code._decode_table()
        max_len = max(code.code_lengths)
        value = 0
        for length in range(1, max_len + 1):
            value = (value << 1) | self.read_bits(1)
            symbol = table.get((length, value))
            if symbol is not None:
                return int(symbol)
        raise ValueError("invalid top-k escape Huffman payload")

    def assert_only_zero_padding(self) -> None:
        while self.remaining_bits > 0:
            if self.read_bits(1) != 0:
                raise ValueError("non-zero padding bits in top-k escape payload")
