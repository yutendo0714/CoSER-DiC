import pytest

from coserdic.entropy import CoSERBitstream, CoSERHeader, assert_no_forbidden_side_info
from coserdic.metrics import estimated_actual_gap_ratio


def make_header() -> CoSERHeader:
    return CoSERHeader(
        codec_version="mvp-test",
        image_height=64,
        image_width=64,
        padded_height=64,
        padded_width=64,
        color_space="RGB",
        rate_level_id=1,
        perception_level_id=0,
        semantic_shape=(2, 2),
        detail_shape=(64, 2, 2),
        entropy_model_version="test",
        cdf_precision=16,
    )


def test_semantic_and_detail_stream_roundtrip() -> None:
    coder = CoSERBitstream()
    stream = coder.pack(
        make_header(),
        semantic_hyper=b"semantic-hyper",
        semantic_tokens=b"semantic-token-stream",
        detail_hyper=b"detail-hyper",
        detail_latents=b"detail-latent-stream",
    )
    decoded = coder.unpack(stream)
    assert decoded.semantic_tokens == b"semantic-token-stream"
    assert decoded.detail_latents == b"detail-latent-stream"
    assert decoded.header.semantic_shape == (2, 2)
    assert decoded.header.detail_shape == (64, 2, 2)


def test_deterministic_bitstream_unpack() -> None:
    coder = CoSERBitstream()
    stream = coder.pack(make_header(), semantic_tokens=b"a", detail_latents=b"b")
    first = coder.unpack(stream)
    second = coder.unpack(stream)
    assert first == second


def test_estimated_actual_gap_ratio() -> None:
    assert estimated_actual_gap_ratio(0.03, 0.033) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        estimated_actual_gap_ratio(0.0, 0.03)


def test_decoder_leakage_policy_rejects_original_derived_side_info() -> None:
    assert_no_forbidden_side_info({"rate_level": 1, "seed": 42})
    with pytest.raises(ValueError, match="caption"):
        assert_no_forbidden_side_info({"caption": "free original-derived caption"})

