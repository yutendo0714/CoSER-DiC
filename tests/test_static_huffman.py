import torch

from coserdic.entropy import (
    StaticANSCode,
    StaticHuffmanCode,
    StaticLeftContextHuffmanCode,
    StaticPositionHuffmanCode,
    encode_semantic_tokens,
)


def test_static_huffman_roundtrip() -> None:
    counts = torch.tensor([100, 20, 5, 1], dtype=torch.long)
    code = StaticHuffmanCode.from_counts(counts)
    tokens = torch.tensor([[0, 0, 1, 2], [3, 0, 1, 0]], dtype=torch.long)

    payload = code.encode(tokens)
    decoded = code.decode(payload, shape=tuple(tokens.shape))

    assert torch.equal(decoded, tokens)


def test_static_huffman_can_beat_fixed_bits_for_biased_tokens() -> None:
    counts = torch.tensor([1000, 5, 5, 5], dtype=torch.long)
    code = StaticHuffmanCode.from_counts(counts)
    tokens = torch.zeros(128, dtype=torch.long)

    huffman_payload = code.encode(tokens)
    fixed_payload = encode_semantic_tokens(tokens, codebook_size=4, codec="fixed_bits")

    assert torch.equal(code.decode(huffman_payload, shape=tuple(tokens.shape)), tokens)
    assert len(huffman_payload) < len(fixed_payload)
    assert code.mean_code_length_bits(tokens) <= 1.0


def test_static_huffman_smoothing_keeps_unseen_symbols_decodable() -> None:
    counts = torch.tensor([10, 0, 0, 0], dtype=torch.long)
    code = StaticHuffmanCode.from_counts(counts, smoothing_count=1)
    tokens = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    assert torch.equal(code.decode(code.encode(tokens), shape=tuple(tokens.shape)), tokens)


def test_static_huffman_serialization_roundtrip() -> None:
    code = StaticHuffmanCode.from_counts(torch.tensor([30, 10, 2, 1], dtype=torch.long))
    restored = StaticHuffmanCode.from_dict(code.to_dict())
    tokens = torch.tensor([0, 1, 0, 2, 3, 0], dtype=torch.long)

    assert restored.code_lengths == code.code_lengths
    assert restored.codes == code.codes
    assert torch.equal(restored.decode(restored.encode(tokens), shape=tuple(tokens.shape)), tokens)


def test_static_position_huffman_roundtrip() -> None:
    counts = torch.ones(2, 3, 4, dtype=torch.long)
    counts[0, 0, 0] = 100
    counts[1, 2, 3] = 100
    code = StaticPositionHuffmanCode.from_counts(counts, smoothing_count=1)
    tokens = torch.tensor([[0, 1, 2], [3, 2, 3]], dtype=torch.long)

    payload = code.encode(tokens)
    decoded = code.decode(payload, shape=tuple(tokens.shape))

    assert torch.equal(decoded, tokens)


def test_static_position_huffman_serialization_roundtrip() -> None:
    counts = torch.ones(2, 2, 4, dtype=torch.long)
    counts[0, 0, 0] = 30
    counts[0, 1, 1] = 30
    counts[1, 0, 2] = 30
    counts[1, 1, 3] = 30
    code = StaticPositionHuffmanCode.from_counts(counts, smoothing_count=1)
    restored = StaticPositionHuffmanCode.from_dict(code.to_dict())
    tokens = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    assert restored.token_shape == code.token_shape
    assert torch.equal(restored.decode(restored.encode(tokens), shape=tuple(tokens.shape)), tokens)


def test_static_position_huffman_global_backoff_roundtrip() -> None:
    position_counts = torch.ones(2, 2, 4, dtype=torch.long)
    global_counts = torch.tensor([100, 20, 5, 1], dtype=torch.long)
    code = StaticPositionHuffmanCode.from_position_counts_with_global_backoff(
        position_counts,
        global_counts,
        backoff_mass=16.0,
    )
    tokens = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    assert torch.equal(code.decode(code.encode(tokens), shape=tuple(tokens.shape)), tokens)


def test_static_left_context_huffman_roundtrip() -> None:
    samples = [
        torch.tensor([[0, 1, 1, 2], [0, 1, 2, 3]], dtype=torch.long),
        torch.tensor([[0, 1, 1, 1], [2, 3, 2, 3]], dtype=torch.long),
    ]
    code = StaticLeftContextHuffmanCode.from_samples(
        samples,
        codebook_size=4,
        context_topk=2,
        backoff_mass=4.0,
    )
    tokens = torch.tensor([[0, 1, 2, 3], [0, 1, 1, 2]], dtype=torch.long)

    payload = code.encode(tokens)
    decoded = code.decode(payload, shape=tuple(tokens.shape))

    assert torch.equal(decoded, tokens)


def test_static_left_context_huffman_serialization_roundtrip() -> None:
    samples = [
        torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.long),
        torch.tensor([[0, 1, 2], [3, 2, 3]], dtype=torch.long),
    ]
    code = StaticLeftContextHuffmanCode.from_samples(
        samples,
        codebook_size=4,
        context_topk=3,
        backoff_mass=8.0,
    )
    restored = StaticLeftContextHuffmanCode.from_dict(code.to_dict())
    tokens = torch.tensor([[0, 1, 1], [3, 2, 3]], dtype=torch.long)

    assert restored.top_tokens == code.top_tokens
    assert torch.equal(restored.decode(restored.encode(tokens), shape=tuple(tokens.shape)), tokens)


def test_static_left_context_huffman_can_use_context() -> None:
    samples = [torch.tensor([[0, 1] * 32], dtype=torch.long)]
    code = StaticLeftContextHuffmanCode.from_samples(
        samples,
        codebook_size=4,
        context_topk=2,
        backoff_mass=1.0,
    )
    contextual_tokens = torch.tensor([[0, 1] * 32], dtype=torch.long)
    shuffled_tokens = torch.tensor([[0, 2, 0, 3] * 16], dtype=torch.long)

    assert code.mean_code_length_bits(contextual_tokens) < code.mean_code_length_bits(shuffled_tokens)


def test_static_ans_roundtrip() -> None:
    counts = torch.tensor([100, 20, 5, 1], dtype=torch.long)
    code = StaticANSCode.from_counts(counts, smoothing_count=1)
    tokens = torch.tensor([[0, 0, 1, 2], [3, 0, 1, 0]], dtype=torch.long)

    payload = code.encode(tokens)
    decoded = code.decode(payload, shape=tuple(tokens.shape))

    assert torch.equal(decoded, tokens)


def test_static_ans_serialization_roundtrip() -> None:
    code = StaticANSCode.from_counts(torch.tensor([30, 10, 2, 1], dtype=torch.long))
    restored = StaticANSCode.from_dict(code.to_dict())
    tokens = torch.tensor([0, 1, 0, 2, 3, 0], dtype=torch.long)

    assert restored.quantized_cdf == code.quantized_cdf
    assert torch.equal(restored.decode(restored.encode(tokens), shape=tuple(tokens.shape)), tokens)
