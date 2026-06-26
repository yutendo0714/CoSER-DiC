from coserdic.entropy import CoSERBitstream, CoSERHeader


def test_bitstream_roundtrip_and_bpp() -> None:
    coder = CoSERBitstream()
    header = CoSERHeader(
        codec_version="0.1",
        image_height=32,
        image_width=16,
        padded_height=64,
        padded_width=64,
        color_space="RGB",
        rate_level_id=1,
        perception_level_id=0,
        semantic_shape=(2, 2),
        detail_shape=(4, 2, 2),
        entropy_model_version="test",
        cdf_precision=16,
    )
    stream = coder.pack(
        header,
        semantic_hyper=b"sh",
        semantic_tokens=b"tokens",
        detail_hyper=b"dh",
        detail_latents=b"latents",
    )
    unpacked = coder.unpack(stream)

    assert unpacked.header.image_height == 32
    assert unpacked.semantic_tokens == b"tokens"
    assert unpacked.detail_latents == b"latents"
    assert coder.actual_bpp(stream, 32, 16) == 8.0 * len(stream) / (32 * 16)


def test_bitstream_checksum_rejects_corruption() -> None:
    coder = CoSERBitstream()
    header = CoSERHeader(
        codec_version="0.1",
        image_height=8,
        image_width=8,
        padded_height=8,
        padded_width=8,
        color_space="RGB",
        rate_level_id=0,
        perception_level_id=0,
        semantic_shape=(1, 1),
        detail_shape=(1, 1, 1),
        entropy_model_version="test",
        cdf_precision=16,
    )
    stream = bytearray(coder.pack(header, semantic_tokens=b"abc"))
    stream[-1] ^= 1

    try:
        coder.unpack(bytes(stream))
    except ValueError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("corrupted stream should fail")

