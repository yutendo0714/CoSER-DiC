import pytest
import torch

from coserdic.entropy import (
    SUPPORTED_SEMANTIC_TOKEN_CODECS,
    decode_semantic_tokens,
    encode_semantic_tokens,
    semantic_bits_per_token,
)


def test_semantic_token_codecs_roundtrip() -> None:
    tokens = torch.tensor([[0, 1, 8191], [7, 512, 4096]], dtype=torch.long)
    for codec in SUPPORTED_SEMANTIC_TOKEN_CODECS:
        payload = encode_semantic_tokens(tokens, codebook_size=8192, codec=codec)
        decoded = decode_semantic_tokens(payload, shape=tuple(tokens.shape), codebook_size=8192, codec=codec)
        assert torch.equal(decoded, tokens)


def test_fixed_bits_uses_codebook_width() -> None:
    tokens = torch.arange(64, dtype=torch.long).reshape(8, 8)
    payload = encode_semantic_tokens(tokens, codebook_size=8192, codec="fixed_bits")
    assert semantic_bits_per_token(8192) == 13
    assert len(payload) == (64 * 13 + 7) // 8


def test_semantic_token_validation() -> None:
    with pytest.raises(ValueError, match="out of range"):
        encode_semantic_tokens(torch.tensor([8192], dtype=torch.long), codebook_size=8192)
    with pytest.raises(TypeError):
        encode_semantic_tokens(torch.tensor([1.0]), codebook_size=8192)
