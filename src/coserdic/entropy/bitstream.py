from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from typing import Any


MAGIC = b"COSERDIC\0"
HEADER_LEN_STRUCT = struct.Struct(">I")


@dataclass(frozen=True)
class CoSERHeader:
    codec_version: str
    image_height: int
    image_width: int
    padded_height: int
    padded_width: int
    color_space: str
    rate_level_id: int
    perception_level_id: int
    semantic_shape: tuple[int, ...]
    detail_shape: tuple[int, ...]
    entropy_model_version: str
    cdf_precision: int
    semantic_hyper_len: int = 0
    semantic_token_len: int = 0
    detail_hyper_len: int = 0
    detail_latent_len: int = 0
    control_len: int = 0
    checksum: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class PackedStreams:
    header: CoSERHeader
    semantic_hyper: bytes
    semantic_tokens: bytes
    detail_hyper: bytes
    detail_latents: bytes
    control: bytes = b""


class CoSERBitstream:
    """Length-delimited CoSER-DiC stream container.

    This is not the final entropy coder. It is the outer container that makes
    actual transmitted bytes auditable from day zero.
    """

    def pack(
        self,
        header: CoSERHeader,
        semantic_hyper: bytes = b"",
        semantic_tokens: bytes = b"",
        detail_hyper: bytes = b"",
        detail_latents: bytes = b"",
        control: bytes = b"",
    ) -> bytes:
        payload = semantic_hyper + semantic_tokens + detail_hyper + detail_latents + control
        checked_header = CoSERHeader(
            **{
                **asdict(header),
                "semantic_hyper_len": len(semantic_hyper),
                "semantic_token_len": len(semantic_tokens),
                "detail_hyper_len": len(detail_hyper),
                "detail_latent_len": len(detail_latents),
                "control_len": len(control),
                "checksum": hashlib.sha256(payload).hexdigest(),
            }
        )
        header_bytes = self._encode_header(checked_header)
        return MAGIC + HEADER_LEN_STRUCT.pack(len(header_bytes)) + header_bytes + payload

    def unpack(self, bitstream: bytes) -> PackedStreams:
        if not bitstream.startswith(MAGIC):
            raise ValueError("invalid CoSER-DiC magic number")
        offset = len(MAGIC)
        (header_len,) = HEADER_LEN_STRUCT.unpack(bitstream[offset : offset + HEADER_LEN_STRUCT.size])
        offset += HEADER_LEN_STRUCT.size
        header = self._decode_header(bitstream[offset : offset + header_len])
        offset += header_len

        semantic_hyper, offset = self._take(bitstream, offset, header.semantic_hyper_len)
        semantic_tokens, offset = self._take(bitstream, offset, header.semantic_token_len)
        detail_hyper, offset = self._take(bitstream, offset, header.detail_hyper_len)
        detail_latents, offset = self._take(bitstream, offset, header.detail_latent_len)
        control, offset = self._take(bitstream, offset, header.control_len)
        if offset != len(bitstream):
            raise ValueError("trailing bytes after CoSER-DiC payload")

        payload = semantic_hyper + semantic_tokens + detail_hyper + detail_latents + control
        checksum = hashlib.sha256(payload).hexdigest()
        if header.checksum and checksum != header.checksum:
            raise ValueError("CoSER-DiC bitstream checksum mismatch")

        return PackedStreams(
            header=header,
            semantic_hyper=semantic_hyper,
            semantic_tokens=semantic_tokens,
            detail_hyper=detail_hyper,
            detail_latents=detail_latents,
            control=control,
        )

    @staticmethod
    def actual_bpp(bitstream: bytes, height: int, width: int) -> float:
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        return 8.0 * len(bitstream) / float(height * width)

    @staticmethod
    def _take(data: bytes, offset: int, length: int) -> tuple[bytes, int]:
        end = offset + length
        if end > len(data):
            raise ValueError("truncated CoSER-DiC bitstream")
        return data[offset:end], end

    @staticmethod
    def _encode_header(header: CoSERHeader) -> bytes:
        return json.dumps(asdict(header), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _decode_header(data: bytes) -> CoSERHeader:
        raw: dict[str, Any] = json.loads(data.decode("utf-8"))
        raw["semantic_shape"] = tuple(raw["semantic_shape"])
        raw["detail_shape"] = tuple(raw["detail_shape"])
        return CoSERHeader(**raw)

