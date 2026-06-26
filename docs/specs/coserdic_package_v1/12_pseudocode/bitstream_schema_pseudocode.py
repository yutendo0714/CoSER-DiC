"""Pseudo-code for CoSER-DiC bitstream schema."""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CoSERHeader:
    magic_number: bytes
    codec_version: str
    image_height: int
    image_width: int
    padded_height: int
    padded_width: int
    color_space: str
    rate_level_id: int
    perception_level_id: int
    semantic_shape: Tuple[int, int]
    detail_shape: Tuple[int, int, int]
    entropy_model_version: str
    cdf_precision: int
    semantic_hyper_len: int
    semantic_token_len: int
    detail_hyper_len: int
    detail_latent_len: int
    control_len: int
    checksum: Optional[str]
    seed: Optional[int]


class CoSERBitstreamCoder:
    def pack(self, header, semantic_hyper, semantic_tokens, detail_hyper, detail_latents, control=b""):
        raise NotImplementedError

    def unpack(self, bitstream: bytes):
        """
        Returns:
            header, semantic_hyper_stream, semantic_token_stream,
            detail_hyper_stream, detail_latent_stream, control_stream
        """
        raise NotImplementedError

    def compute_actual_bpp(self, bitstream: bytes, height: int, width: int) -> float:
        return 8.0 * len(bitstream) / (height * width)
