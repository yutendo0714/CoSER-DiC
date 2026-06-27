import torch

from coserdic.entropy import TopKEscapeHuffmanCode, count_topk_escape_events, encode_semantic_tokens


def test_topk_escape_huffman_roundtrip_with_escapes() -> None:
    targets = torch.tensor([[0, 3, 7, 2]], dtype=torch.long)
    topk = torch.tensor([[[0, 1], [1, 2], [3, 4], [2, 0]]], dtype=torch.long)
    counts = count_topk_escape_events(targets, topk, codebook_size=8, topk=2)
    code = TopKEscapeHuffmanCode.from_event_counts(counts, codebook_size=8, topk=2)

    payload = code.encode(targets, topk)
    decoded = code.decode(payload, topk_indices=topk, shape=tuple(targets.shape))

    assert torch.equal(decoded, targets)
    assert code.escape_count(targets, topk) == 2


def test_topk_escape_huffman_provider_decode() -> None:
    targets = torch.tensor([0, 3, 7, 2], dtype=torch.long)
    rows = [[0, 1], [1, 2], [3, 4], [2, 0]]
    topk = torch.tensor(rows, dtype=torch.long)
    counts = count_topk_escape_events(targets, topk, codebook_size=8, topk=2)
    code = TopKEscapeHuffmanCode.from_event_counts(counts, codebook_size=8, topk=2)

    payload = code.encode(targets, topk)

    def provider(index: int, prefix: tuple[int, ...]) -> list[int]:
        assert len(prefix) == index
        return rows[index]

    decoded = code.decode_with_provider(payload, shape=tuple(targets.shape), topk_provider=provider)

    assert torch.equal(decoded, targets)


def test_topk_escape_huffman_serialization_roundtrip() -> None:
    targets = torch.tensor([0, 1, 0, 1, 2, 3], dtype=torch.long)
    topk = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 2], [0, 1], [1, 2]], dtype=torch.long)
    counts = count_topk_escape_events(targets, topk, codebook_size=4, topk=2)
    code = TopKEscapeHuffmanCode.from_event_counts(counts, codebook_size=4, topk=2)
    restored = TopKEscapeHuffmanCode.from_dict(code.to_dict())

    assert restored.event_code.code_lengths == code.event_code.code_lengths
    assert torch.equal(restored.decode(restored.encode(targets, topk), topk_indices=topk, shape=tuple(targets.shape)), targets)


def test_topk_escape_can_beat_fixed_bits_when_hits_are_frequent() -> None:
    targets = torch.zeros(128, dtype=torch.long)
    topk = torch.stack([torch.tensor([0, 1], dtype=torch.long) for _ in range(128)], dim=0)
    counts = count_topk_escape_events(targets, topk, codebook_size=4, topk=2)
    code = TopKEscapeHuffmanCode.from_event_counts(counts, codebook_size=4, topk=2)

    payload = code.encode(targets, topk)
    fixed_payload = encode_semantic_tokens(targets, codebook_size=4, codec="fixed_bits")

    assert len(payload) < len(fixed_payload)
