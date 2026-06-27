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


def test_compact_bitstream_roundtrip_is_smaller_than_json_header() -> None:
    json_coder = CoSERBitstream()
    compact_coder = CoSERBitstream(header_codec="compact")
    header = CoSERHeader(
        codec_version="stage3-uniform-residual-mvp",
        image_height=256,
        image_width=256,
        padded_height=256,
        padded_width=256,
        color_space="RGB",
        rate_level_id=0,
        perception_level_id=0,
        semantic_shape=(16, 16),
        detail_shape=(3, 8, 8),
        entropy_model_version="s2sem-lteh0+s3res-poshuff",
        cdf_precision=16,
        seed=42,
    )

    json_stream = json_coder.pack(header, semantic_tokens=b"a" * 80, detail_latents=b"b" * 48)
    compact_stream = compact_coder.pack(header, semantic_tokens=b"a" * 80, detail_latents=b"b" * 48)
    decoded = json_coder.unpack(compact_stream)

    assert decoded.header.codec_version == "stage3-uniform-residual-mvp"
    assert decoded.header.semantic_shape == (16, 16)
    assert decoded.header.detail_shape == (3, 8, 8)
    assert decoded.semantic_tokens == b"a" * 80
    assert decoded.detail_latents == b"b" * 48
    assert len(compact_stream) < len(json_stream)


def test_compact_bitstream_checksum_rejects_corruption() -> None:
    coder = CoSERBitstream(header_codec="compact")
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


def test_compact_bitstream_crc32_checksum_roundtrip_and_corruption() -> None:
    sha_coder = CoSERBitstream(header_codec="compact")
    crc_coder = CoSERBitstream(header_codec="compact", checksum_codec="crc32")
    header = CoSERHeader(
        codec_version="stage3-uniform-residual-mvp",
        image_height=256,
        image_width=256,
        padded_height=256,
        padded_width=256,
        color_space="RGB",
        rate_level_id=0,
        perception_level_id=0,
        semantic_shape=(16, 16),
        detail_shape=(3, 8, 8),
        entropy_model_version="s2sem-lteh0+s3res-poshuff",
        cdf_precision=0,
    )

    sha_stream = sha_coder.pack(header, semantic_tokens=b"a" * 80, detail_latents=b"b" * 48)
    crc_stream = crc_coder.pack(header, semantic_tokens=b"a" * 80, detail_latents=b"b" * 48)
    unpacked = sha_coder.unpack(crc_stream)

    assert unpacked.header.checksum is not None
    assert unpacked.header.checksum.startswith("crc32:")
    assert unpacked.semantic_tokens == b"a" * 80
    assert unpacked.detail_latents == b"b" * 48
    assert len(crc_stream) == len(sha_stream) - 28

    corrupted = bytearray(crc_stream)
    corrupted[-1] ^= 1
    try:
        sha_coder.unpack(bytes(corrupted))
    except ValueError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("corrupted crc32 stream should fail")


def test_compact_v3_header_is_smaller_and_decodes_v2() -> None:
    coder = CoSERBitstream(header_codec="compact", checksum_codec="crc32")
    header = CoSERHeader(
        codec_version="s3urg0",
        image_height=256,
        image_width=256,
        padded_height=256,
        padded_width=256,
        color_space="RGB",
        rate_level_id=0,
        perception_level_id=0,
        semantic_shape=(16, 16),
        detail_shape=(3, 32, 32),
        entropy_model_version="s3urg-d32-b5-r025",
        cdf_precision=0,
    )
    payload = b"a" * 88 + b"b" * 68
    checked = coder.unpack(coder.pack(header, semantic_tokens=b"a" * 88, detail_latents=b"b" * 68)).header
    legacy_stream = (
        b"COSERDI2\0"
        + CoSERBitstream._encode_compact_v2_header(checked)
        + payload
    )
    v3_stream = coder.pack(header, semantic_tokens=b"a" * 88, detail_latents=b"b" * 68)

    decoded_legacy = coder.unpack(legacy_stream)
    decoded_v3 = coder.unpack(v3_stream)

    assert decoded_legacy.header.codec_version == "s3urg0"
    assert decoded_v3.header.entropy_model_version == "s3urg-d32-b5-r025"
    assert decoded_v3.semantic_tokens == b"a" * 88
    assert decoded_v3.detail_latents == b"b" * 68
    assert len(v3_stream) < len(legacy_stream) - 40
