import pytest
import torch

from coserdic.entropy import (
    StaticResidualGridHuffmanCode,
    StaticResidualGridHybridHuffmanCode,
    StaticResidualGridPositionHuffmanCode,
    StaticResidualGridSemanticPositionLeftContextHuffmanCode,
    StaticResidualGridSemanticPositionHuffmanCode,
    UniformResidualGridCode,
)


def test_uniform_residual_grid_roundtrip_fixed_and_zlib() -> None:
    residual = torch.linspace(-0.4, 0.4, steps=3 * 4 * 4).reshape(3, 4, 4)
    for payload_codec in ("fixed_bits", "zlib_fixed_bits"):
        code = UniformResidualGridCode(bits=5, value_range=0.5, codec=payload_codec)
        codes = code.quantize(residual)
        payload = code.encode(codes)
        decoded = code.decode(payload, shape=tuple(codes.shape))
        dequantized = code.dequantize(decoded)

        assert torch.equal(decoded, codes)
        assert decoded.shape == codes.shape
        assert torch.max(torch.abs(dequantized - residual)).item() <= (1.0 / 31.0)


def test_uniform_residual_grid_rejects_invalid_codes() -> None:
    code = UniformResidualGridCode(bits=3, value_range=0.25)
    bad_codes = torch.tensor([8], dtype=torch.long)

    try:
        code.dequantize(bad_codes)
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("invalid residual code should fail")


def test_zero_centered_residual_grid_has_exact_zero_code() -> None:
    code = UniformResidualGridCode(bits=2, value_range=0.25, codec="fixed_bits", quantizer="zero_centered")
    residual = torch.tensor([-0.25, -0.01, 0.0, 0.01, 0.25], dtype=torch.float32)

    codes = code.quantize(residual)
    payload = code.encode(codes)
    decoded = code.decode(payload, shape=tuple(codes.shape))
    dequantized = code.dequantize(decoded)

    assert torch.equal(decoded, codes)
    assert int(code.quantize(torch.zeros(1))[0].item()) == code.zero_code
    assert float(code.dequantize(torch.tensor([code.zero_code], dtype=torch.long))[0].item()) == 0.0
    assert torch.max(torch.abs(dequantized - residual)).item() <= 0.125


def test_static_residual_grid_huffman_roundtrip() -> None:
    residual = torch.tensor(
        [
            [[-0.1, 0.0, 0.1, 0.2], [0.0, 0.0, 0.05, -0.05]],
            [[0.02, 0.01, -0.02, -0.03], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    quantizer = UniformResidualGridCode(bits=4, value_range=0.25)
    codes = quantizer.quantize(residual)
    counts = torch.bincount(codes.reshape(-1), minlength=quantizer.levels)
    huffman = StaticResidualGridHuffmanCode.from_counts(counts, bits=4, value_range=0.25, smoothing_count=1)

    payload = huffman.encode(codes)
    decoded = huffman.decode(payload, shape=tuple(codes.shape))

    assert torch.equal(decoded, codes)
    assert huffman.encoded_bits(codes) <= codes.numel() * 4
    assert StaticResidualGridHuffmanCode.from_dict(huffman.to_dict()).bits == 4


def test_static_residual_grid_huffman_preserves_zero_centered_quantizer() -> None:
    residual = torch.tensor([[-0.1, 0.0, 0.1, 0.2]], dtype=torch.float32)
    quantizer = UniformResidualGridCode(bits=3, value_range=0.25, quantizer="zero_centered")
    codes = quantizer.quantize(residual)
    counts = torch.bincount(codes.reshape(-1), minlength=quantizer.levels)
    huffman = StaticResidualGridHuffmanCode.from_counts(
        counts,
        bits=3,
        value_range=0.25,
        quantizer="zero_centered",
        smoothing_count=1,
    )

    restored = StaticResidualGridHuffmanCode.from_dict(huffman.to_dict())

    assert huffman.to_dict()["quantizer"] == "zero_centered"
    assert restored.quantizer.quantizer == "zero_centered"
    assert torch.equal(restored.decode(restored.encode(codes), shape=tuple(codes.shape)), codes)


def test_static_residual_grid_position_huffman_roundtrip() -> None:
    residual = torch.tensor(
        [
            [[-0.1, 0.0, 0.1, 0.2], [0.0, 0.0, 0.05, -0.05]],
            [[0.02, 0.01, -0.02, -0.03], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    quantizer = UniformResidualGridCode(bits=4, value_range=0.25)
    codes = quantizer.quantize(residual)
    counts = torch.zeros(*codes.shape, quantizer.levels, dtype=torch.long)
    for flat_index, value in enumerate(codes.reshape(-1).tolist()):
        counts.reshape(-1, quantizer.levels)[flat_index, int(value)] += 4
    position_huffman = StaticResidualGridPositionHuffmanCode.from_counts(
        counts,
        bits=4,
        value_range=0.25,
        smoothing_count=1,
    )

    payload = position_huffman.encode(codes)
    decoded = position_huffman.decode(payload, shape=tuple(codes.shape))
    restored = StaticResidualGridPositionHuffmanCode.from_dict(position_huffman.to_dict())

    assert torch.equal(decoded, codes)
    assert torch.equal(restored.decode(payload, shape=tuple(codes.shape)), codes)
    assert position_huffman.encoded_bits(codes) <= codes.numel() * 4
    with pytest.raises(ValueError, match="does not match"):
        position_huffman.decode(payload, shape=(1, 2, 4))


def test_uniform_residual_grid_huffman_codec_requires_static_huffman_wrapper() -> None:
    code = UniformResidualGridCode(bits=4, value_range=0.25, codec="huffman")
    with pytest.raises(ValueError, match="StaticResidualGridHuffmanCode"):
        code.encode(torch.zeros(1, dtype=torch.long))


def test_static_residual_grid_semantic_position_huffman_roundtrip() -> None:
    residual = torch.tensor(
        [
            [[-0.1, 0.0], [0.1, 0.2]],
            [[0.02, -0.03], [0.0, 0.05]],
        ],
        dtype=torch.float32,
    )
    semantic_indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    token_to_group = [0, 1, 0, 1]
    quantizer = UniformResidualGridCode(bits=4, value_range=0.25)
    codes = quantizer.quantize(residual)
    counts = torch.ones(2, 2, 2, 2, quantizer.levels, dtype=torch.long)
    for channel in range(2):
        for y in range(2):
            for x in range(2):
                group = token_to_group[int(semantic_indices[y, x])]
                counts[channel, y, x, group, int(codes[channel, y, x])] += 16
    semantic_huffman = StaticResidualGridSemanticPositionHuffmanCode.from_counts(
        counts,
        bits=4,
        value_range=0.25,
        semantic_shape=(2, 2),
        token_to_group=token_to_group,
        smoothing_count=1,
    )

    payload = semantic_huffman.encode(codes, semantic_indices=semantic_indices)
    decoded = semantic_huffman.decode(payload, shape=tuple(codes.shape), semantic_indices=semantic_indices)
    restored = StaticResidualGridSemanticPositionHuffmanCode.from_dict(semantic_huffman.to_dict())

    assert torch.equal(decoded, codes)
    assert torch.equal(restored.decode(payload, shape=tuple(codes.shape), semantic_indices=semantic_indices), codes)
    assert semantic_huffman.encoded_bits(codes, semantic_indices=semantic_indices) <= codes.numel() * 4
    with pytest.raises(ValueError, match="semantic_indices shape"):
        semantic_huffman.decode(payload, shape=tuple(codes.shape), semantic_indices=torch.zeros(1, 2, dtype=torch.long))


def test_static_residual_grid_hybrid_huffman_roundtrip() -> None:
    residual = torch.tensor(
        [
            [[-0.1, 0.0], [0.1, 0.2]],
            [[0.02, -0.03], [0.0, 0.05]],
        ],
        dtype=torch.float32,
    )
    semantic_indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    token_to_group = [0, 1, 0, 1]
    quantizer = UniformResidualGridCode(bits=4, value_range=0.25)
    codes = quantizer.quantize(residual)

    position_counts = torch.ones(*codes.shape, quantizer.levels, dtype=torch.long)
    semantic_counts = torch.ones(2, 2, 2, 2, quantizer.levels, dtype=torch.long)
    for channel in range(2):
        for y in range(2):
            for x in range(2):
                value = int(codes[channel, y, x])
                group = token_to_group[int(semantic_indices[y, x])]
                position_counts[channel, y, x, value] += 8
                semantic_counts[channel, y, x, group, value] += 16

    position_huffman = StaticResidualGridPositionHuffmanCode.from_counts(
        position_counts,
        bits=4,
        value_range=0.25,
        smoothing_count=1,
    )
    semantic_huffman = StaticResidualGridSemanticPositionHuffmanCode.from_counts(
        semantic_counts,
        bits=4,
        value_range=0.25,
        semantic_shape=(2, 2),
        token_to_group=token_to_group,
        smoothing_count=1,
    )
    hybrid = StaticResidualGridHybridHuffmanCode(
        position_code=position_huffman,
        semantic_position_code=semantic_huffman,
    )

    payload = hybrid.encode(codes, semantic_indices=semantic_indices)
    decoded = hybrid.decode(payload, shape=tuple(codes.shape), semantic_indices=semantic_indices)
    restored = StaticResidualGridHybridHuffmanCode.from_dict(hybrid.to_dict())

    assert torch.equal(decoded, codes)
    assert torch.equal(restored.decode(payload, shape=tuple(codes.shape), semantic_indices=semantic_indices), codes)
    assert hybrid.encoded_bits(codes, semantic_indices=semantic_indices) <= 1 + codes.numel() * 4
    with pytest.raises(ValueError, match="semantic_indices shape"):
        hybrid.decode(payload, shape=tuple(codes.shape), semantic_indices=torch.zeros(1, 2, dtype=torch.long))


def test_static_residual_grid_semantic_position_leftctx_huffman_roundtrip() -> None:
    residual = torch.tensor(
        [
            [[-0.1, 0.0, 0.1], [0.2, 0.1, -0.1]],
            [[0.02, -0.03, 0.0], [0.05, 0.02, -0.02]],
        ],
        dtype=torch.float32,
    )
    semantic_indices = torch.tensor([[0, 1, 2], [3, 0, 1]], dtype=torch.long)
    token_to_group = [0, 1, 0, 1]
    quantizer = UniformResidualGridCode(bits=4, value_range=0.25)
    codes = quantizer.quantize(residual)
    counts = torch.ones(2, 2, 3, 2, 4, quantizer.levels, dtype=torch.long)
    zero_code = int(round(float(quantizer.levels - 1) / 2.0))
    for channel in range(2):
        for y in range(2):
            for x in range(3):
                group = token_to_group[int(semantic_indices[y, x])]
                if x == 0:
                    left_context = 0
                elif int(codes[channel, y, x - 1]) < zero_code:
                    left_context = 1
                elif int(codes[channel, y, x - 1]) == zero_code:
                    left_context = 2
                else:
                    left_context = 3
                counts[channel, y, x, group, left_context, int(codes[channel, y, x])] += 16
    semantic_leftctx = StaticResidualGridSemanticPositionLeftContextHuffmanCode.from_counts(
        counts,
        bits=4,
        value_range=0.25,
        semantic_shape=(2, 3),
        token_to_group=token_to_group,
        smoothing_count=1,
    )

    payload = semantic_leftctx.encode(codes, semantic_indices=semantic_indices)
    decoded = semantic_leftctx.decode(payload, shape=tuple(codes.shape), semantic_indices=semantic_indices)
    restored = StaticResidualGridSemanticPositionLeftContextHuffmanCode.from_dict(semantic_leftctx.to_dict())

    assert torch.equal(decoded, codes)
    assert torch.equal(restored.decode(payload, shape=tuple(codes.shape), semantic_indices=semantic_indices), codes)
    assert semantic_leftctx.encoded_bits(codes, semantic_indices=semantic_indices) <= codes.numel() * 4
    with pytest.raises(ValueError, match="semantic_indices shape"):
        semantic_leftctx.decode(payload, shape=tuple(codes.shape), semantic_indices=torch.zeros(1, 3, dtype=torch.long))
