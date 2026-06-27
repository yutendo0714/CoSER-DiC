import pytest
import torch

from coserdic.entropy import StaticResidualGridHuffmanCode, StaticResidualGridPositionHuffmanCode, UniformResidualGridCode


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
