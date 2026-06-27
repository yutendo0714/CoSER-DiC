import torch

from coserdic.entropy import StaticHuffmanCode, encode_semantic_tokens


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
