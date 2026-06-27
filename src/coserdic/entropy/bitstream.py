from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from typing import Any


MAGIC = b"COSERDIC\0"
COMPACT_MAGIC = b"COSERDI2\0"
HEADER_LEN_STRUCT = struct.Struct(">I")
U8_STRUCT = struct.Struct(">B")
U16_STRUCT = struct.Struct(">H")
U32_STRUCT = struct.Struct(">I")
I16_STRUCT = struct.Struct(">h")
I64_STRUCT = struct.Struct(">q")

_CODEC_VERSION_IDS = {
    "s2lte0": 1,
    "s2sth0": 2,
    "s3urg0": 3,
    "s3rae0": 4,
    "stage1-semantic-vq-mvp": 5,
    "stage3-uniform-residual-mvp": 6,
    "0.1": 7,
    "mvp-test": 8,
}
_COLOR_SPACE_IDS = {
    "RGB": 1,
    "YCbCr": 2,
}
_ENTROPY_MODEL_VERSION_IDS = {
    "s2sem-lteh0": 1,
    "s2sem-fxb0": 2,
    "s2sem-sth0": 3,
    "s3urg-d32-b5-r025": 4,
    "s3urg-d32-b4-r025": 5,
    "s3urg-d32-b4-r050": 6,
    "s3urg-d32-b6-r025": 7,
    "s3urg-d16-b4-r025": 8,
    "s3rae-d32-c3-b5-r025": 9,
}

_CODEC_VERSION_STRINGS = {value: key for key, value in _CODEC_VERSION_IDS.items()}
_COLOR_SPACE_STRINGS = {value: key for key, value in _COLOR_SPACE_IDS.items()}
_ENTROPY_MODEL_VERSION_STRINGS = {value: key for key, value in _ENTROPY_MODEL_VERSION_IDS.items()}


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

    def __init__(self, header_codec: str = "json", checksum_codec: str = "sha256") -> None:
        if header_codec not in {"json", "compact"}:
            raise ValueError("header_codec must be 'json' or 'compact'")
        if checksum_codec not in {"sha256", "crc32"}:
            raise ValueError("checksum_codec must be 'sha256' or 'crc32'")
        self.header_codec = header_codec
        self.checksum_codec = checksum_codec

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
                "checksum": self._checksum(payload),
            }
        )
        if self.header_codec == "compact":
            header_bytes = self._encode_compact_header(checked_header)
            return COMPACT_MAGIC + header_bytes + payload
        header_bytes = self._encode_json_header(checked_header)
        return MAGIC + HEADER_LEN_STRUCT.pack(len(header_bytes)) + header_bytes + payload

    def unpack(self, bitstream: bytes) -> PackedStreams:
        if bitstream.startswith(COMPACT_MAGIC):
            offset = len(COMPACT_MAGIC)
            header, offset = self._decode_compact_header(bitstream, offset)
        elif bitstream.startswith(MAGIC):
            offset = len(MAGIC)
            (header_len,) = HEADER_LEN_STRUCT.unpack(bitstream[offset : offset + HEADER_LEN_STRUCT.size])
            offset += HEADER_LEN_STRUCT.size
            header = self._decode_json_header(bitstream[offset : offset + header_len])
            offset += header_len
        else:
            raise ValueError("invalid CoSER-DiC magic number")

        semantic_hyper, offset = self._take(bitstream, offset, header.semantic_hyper_len)
        semantic_tokens, offset = self._take(bitstream, offset, header.semantic_token_len)
        detail_hyper, offset = self._take(bitstream, offset, header.detail_hyper_len)
        detail_latents, offset = self._take(bitstream, offset, header.detail_latent_len)
        control, offset = self._take(bitstream, offset, header.control_len)
        if offset != len(bitstream):
            raise ValueError("trailing bytes after CoSER-DiC payload")

        payload = semantic_hyper + semantic_tokens + detail_hyper + detail_latents + control
        if header.checksum and not _checksum_matches(payload, header.checksum):
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
    def bytes_to_bpp(num_bytes: int, height: int, width: int) -> float:
        if num_bytes < 0:
            raise ValueError("num_bytes must be non-negative")
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        return 8.0 * float(num_bytes) / float(height * width)

    @classmethod
    def actual_payload_bpp(cls, payloads: bytes | list[bytes] | tuple[bytes, ...], height: int, width: int) -> float:
        """CompressAI-style paper bpp from entropy-coded payload streams only."""
        if isinstance(payloads, (bytes, bytearray, memoryview)):
            payload_bytes = len(payloads)
        else:
            payload_bytes = sum(len(payload) for payload in payloads)
        return cls.bytes_to_bpp(payload_bytes, height, width)

    @classmethod
    def debug_full_stream_bpp(cls, bitstream: bytes, height: int, width: int) -> float:
        """Development-only bpp including the CoSER container/header/checksum."""
        return cls.bytes_to_bpp(len(bitstream), height, width)

    @classmethod
    def actual_bpp(cls, bitstream: bytes, height: int, width: int) -> float:
        """Backward-compatible alias for full byte-count bpp.

        Prefer actual_payload_bpp for paper LIC comparisons, and
        debug_full_stream_bpp when intentionally measuring the CoSER container.
        """
        return cls.debug_full_stream_bpp(bitstream, height, width)

    @staticmethod
    def _take(data: bytes, offset: int, length: int) -> tuple[bytes, int]:
        end = offset + length
        if end > len(data):
            raise ValueError("truncated CoSER-DiC bitstream")
        return data[offset:end], end

    @staticmethod
    def _make_checksum(payload: bytes, codec: str | None = None) -> str:
        codec = codec or "sha256"
        if codec == "crc32":
            return f"crc32:{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
        if codec == "sha256":
            return hashlib.sha256(payload).hexdigest()
        raise ValueError(f"unsupported checksum codec: {codec}")

    def _checksum(self, payload: bytes) -> str:
        return self.__class__._make_checksum(payload, self.checksum_codec)

    @staticmethod
    def _encode_json_header(header: CoSERHeader) -> bytes:
        return json.dumps(asdict(header), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _decode_json_header(data: bytes) -> CoSERHeader:
        raw: dict[str, Any] = json.loads(data.decode("utf-8"))
        raw["semantic_shape"] = tuple(raw["semantic_shape"])
        raw["detail_shape"] = tuple(raw["detail_shape"])
        return CoSERHeader(**raw)

    @staticmethod
    def _encode_compact_header(header: CoSERHeader) -> bytes:
        return CoSERBitstream._encode_compact_v3_header(header)

    @staticmethod
    def _encode_compact_v2_header(header: CoSERHeader) -> bytes:
        out = bytearray()
        checksum_kind = _checksum_kind(header.checksum)
        out.extend(U8_STRUCT.pack(2 if checksum_kind == "crc32" else 1))
        for value in (header.image_height, header.image_width, header.padded_height, header.padded_width):
            if int(value) < 0:
                raise ValueError("image dimensions must be non-negative")
            out.extend(U32_STRUCT.pack(int(value)))
        out.extend(I16_STRUCT.pack(int(header.rate_level_id)))
        out.extend(I16_STRUCT.pack(int(header.perception_level_id)))
        out.extend(U16_STRUCT.pack(int(header.cdf_precision)))
        _put_shape(out, header.semantic_shape)
        _put_shape(out, header.detail_shape)
        for value in (
            header.semantic_hyper_len,
            header.semantic_token_len,
            header.detail_hyper_len,
            header.detail_latent_len,
            header.control_len,
        ):
            out.extend(U32_STRUCT.pack(int(value)))
        flags = 0
        if header.checksum:
            flags |= 1
        if checksum_kind == "crc32":
            flags |= 4
        if header.seed is not None:
            flags |= 2
        out.extend(U8_STRUCT.pack(flags))
        if header.seed is not None:
            out.extend(I64_STRUCT.pack(int(header.seed)))
        if header.checksum:
            out.extend(bytes.fromhex(_checksum_hex(header.checksum)))
        _put_string(out, header.codec_version)
        _put_string(out, header.color_space)
        _put_string(out, header.entropy_model_version)
        return bytes(out)

    @staticmethod
    def _encode_compact_v3_header(header: CoSERHeader) -> bytes:
        out = bytearray()
        checksum_kind = _checksum_kind(header.checksum)
        flags = 0
        if header.checksum:
            flags |= 1
        if header.seed is not None:
            flags |= 2
        if checksum_kind == "crc32":
            flags |= 4

        out.extend(U8_STRUCT.pack(3))
        out.extend(U8_STRUCT.pack(flags))
        for value in (header.image_height, header.image_width, header.padded_height, header.padded_width):
            _put_varuint(out, _require_non_negative(value, "image dimensions"))
        _put_varint(out, int(header.rate_level_id))
        _put_varint(out, int(header.perception_level_id))
        _put_varuint(out, _require_non_negative(header.cdf_precision, "cdf_precision"))
        _put_varshape(out, header.semantic_shape)
        _put_varshape(out, header.detail_shape)
        for value in (
            header.semantic_hyper_len,
            header.semantic_token_len,
            header.detail_hyper_len,
            header.detail_latent_len,
            header.control_len,
        ):
            _put_varuint(out, _require_non_negative(value, "payload lengths"))
        if header.seed is not None:
            _put_varint(out, int(header.seed))
        if header.checksum:
            out.extend(bytes.fromhex(_checksum_hex(header.checksum)))
        _put_table_string(out, header.codec_version, _CODEC_VERSION_IDS)
        _put_table_string(out, header.color_space, _COLOR_SPACE_IDS)
        _put_table_string(out, header.entropy_model_version, _ENTROPY_MODEL_VERSION_IDS)
        return bytes(out)

    @staticmethod
    def _decode_compact_header(data: bytes, offset: int) -> tuple[CoSERHeader, int]:
        version, offset = _get_u8(data, offset)
        if version == 3:
            return CoSERBitstream._decode_compact_v3_header(data, offset)
        if version not in {1, 2}:
            raise ValueError(f"unsupported compact CoSER-DiC header version: {version}")
        image_height, offset = _get_u32(data, offset)
        image_width, offset = _get_u32(data, offset)
        padded_height, offset = _get_u32(data, offset)
        padded_width, offset = _get_u32(data, offset)
        rate_level_id, offset = _get_i16(data, offset)
        perception_level_id, offset = _get_i16(data, offset)
        cdf_precision, offset = _get_u16(data, offset)
        semantic_shape, offset = _get_shape(data, offset)
        detail_shape, offset = _get_shape(data, offset)
        semantic_hyper_len, offset = _get_u32(data, offset)
        semantic_token_len, offset = _get_u32(data, offset)
        detail_hyper_len, offset = _get_u32(data, offset)
        detail_latent_len, offset = _get_u32(data, offset)
        control_len, offset = _get_u32(data, offset)
        flags, offset = _get_u8(data, offset)
        seed = None
        if flags & 2:
            seed, offset = _get_i64(data, offset)
        checksum = None
        if flags & 1:
            if version >= 2 and flags & 4:
                checksum_bytes = data[offset : offset + 4]
                if len(checksum_bytes) != 4:
                    raise ValueError("truncated compact crc32 checksum")
                checksum = f"crc32:{checksum_bytes.hex()}"
                offset += 4
            else:
                checksum_bytes = data[offset : offset + 32]
                if len(checksum_bytes) != 32:
                    raise ValueError("truncated compact checksum")
                checksum = checksum_bytes.hex()
                offset += 32
        codec_version, offset = _get_string(data, offset)
        color_space, offset = _get_string(data, offset)
        entropy_model_version, offset = _get_string(data, offset)
        return (
            CoSERHeader(
                codec_version=codec_version,
                image_height=image_height,
                image_width=image_width,
                padded_height=padded_height,
                padded_width=padded_width,
                color_space=color_space,
                rate_level_id=rate_level_id,
                perception_level_id=perception_level_id,
                semantic_shape=semantic_shape,
                detail_shape=detail_shape,
                entropy_model_version=entropy_model_version,
                cdf_precision=cdf_precision,
                semantic_hyper_len=semantic_hyper_len,
                semantic_token_len=semantic_token_len,
                detail_hyper_len=detail_hyper_len,
                detail_latent_len=detail_latent_len,
                control_len=control_len,
                checksum=checksum,
                seed=seed,
            ),
            offset,
        )

    @staticmethod
    def _decode_compact_v3_header(data: bytes, offset: int) -> tuple[CoSERHeader, int]:
        flags, offset = _get_u8(data, offset)
        image_height, offset = _get_varuint(data, offset)
        image_width, offset = _get_varuint(data, offset)
        padded_height, offset = _get_varuint(data, offset)
        padded_width, offset = _get_varuint(data, offset)
        rate_level_id, offset = _get_varint(data, offset)
        perception_level_id, offset = _get_varint(data, offset)
        cdf_precision, offset = _get_varuint(data, offset)
        semantic_shape, offset = _get_varshape(data, offset)
        detail_shape, offset = _get_varshape(data, offset)
        semantic_hyper_len, offset = _get_varuint(data, offset)
        semantic_token_len, offset = _get_varuint(data, offset)
        detail_hyper_len, offset = _get_varuint(data, offset)
        detail_latent_len, offset = _get_varuint(data, offset)
        control_len, offset = _get_varuint(data, offset)
        seed = None
        if flags & 2:
            seed, offset = _get_varint(data, offset)
        checksum = None
        if flags & 1:
            if flags & 4:
                checksum_bytes = data[offset : offset + 4]
                if len(checksum_bytes) != 4:
                    raise ValueError("truncated compact crc32 checksum")
                checksum = f"crc32:{checksum_bytes.hex()}"
                offset += 4
            else:
                checksum_bytes = data[offset : offset + 32]
                if len(checksum_bytes) != 32:
                    raise ValueError("truncated compact checksum")
                checksum = checksum_bytes.hex()
                offset += 32
        codec_version, offset = _get_table_string(data, offset, _CODEC_VERSION_STRINGS)
        color_space, offset = _get_table_string(data, offset, _COLOR_SPACE_STRINGS)
        entropy_model_version, offset = _get_table_string(data, offset, _ENTROPY_MODEL_VERSION_STRINGS)
        return (
            CoSERHeader(
                codec_version=codec_version,
                image_height=image_height,
                image_width=image_width,
                padded_height=padded_height,
                padded_width=padded_width,
                color_space=color_space,
                rate_level_id=rate_level_id,
                perception_level_id=perception_level_id,
                semantic_shape=semantic_shape,
                detail_shape=detail_shape,
                entropy_model_version=entropy_model_version,
                cdf_precision=cdf_precision,
                semantic_hyper_len=semantic_hyper_len,
                semantic_token_len=semantic_token_len,
                detail_hyper_len=detail_hyper_len,
                detail_latent_len=detail_latent_len,
                control_len=control_len,
                checksum=checksum,
                seed=seed,
            ),
            offset,
        )


def _put_string(out: bytearray, value: str) -> None:
    raw = value.encode("utf-8")
    if len(raw) > 65535:
        raise ValueError("compact header string is too long")
    out.extend(U16_STRUCT.pack(len(raw)))
    out.extend(raw)


def _get_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = _get_u16(data, offset)
    raw = data[offset : offset + length]
    if len(raw) != length:
        raise ValueError("truncated compact string")
    return raw.decode("utf-8"), offset + length


def _put_table_string(out: bytearray, value: str, table: dict[str, int]) -> None:
    string_id = table.get(value, 0)
    _put_varuint(out, string_id)
    if string_id == 0:
        raw = value.encode("utf-8")
        _put_varuint(out, len(raw))
        out.extend(raw)


def _get_table_string(data: bytes, offset: int, table: dict[int, str]) -> tuple[str, int]:
    string_id, offset = _get_varuint(data, offset)
    if string_id:
        if string_id not in table:
            raise ValueError(f"unknown compact string table id: {string_id}")
        return table[string_id], offset
    length, offset = _get_varuint(data, offset)
    raw = data[offset : offset + length]
    if len(raw) != length:
        raise ValueError("truncated compact table string")
    return raw.decode("utf-8"), offset + length


def _checksum_kind(checksum: str | None) -> str:
    if checksum is None:
        return "none"
    if checksum.startswith("crc32:"):
        return "crc32"
    if checksum.startswith("sha256:"):
        return "sha256"
    return "sha256"


def _checksum_hex(checksum: str) -> str:
    if checksum.startswith("crc32:") or checksum.startswith("sha256:"):
        return checksum.split(":", 1)[1]
    return checksum


def _checksum_matches(payload: bytes, expected: str) -> bool:
    kind = _checksum_kind(expected)
    if kind == "crc32":
        return expected == CoSERBitstream._make_checksum(payload, "crc32")
    if kind == "sha256":
        digest = CoSERBitstream._make_checksum(payload, "sha256")
        return _checksum_hex(expected) == digest
    return False


def _put_shape(out: bytearray, shape: tuple[int, ...]) -> None:
    if len(shape) > 255:
        raise ValueError("compact header shape has too many dimensions")
    out.extend(U8_STRUCT.pack(len(shape)))
    for value in shape:
        if int(value) < 0 or int(value) > 65535:
            raise ValueError("compact header shape dimensions must fit uint16")
        out.extend(U16_STRUCT.pack(int(value)))


def _get_shape(data: bytes, offset: int) -> tuple[tuple[int, ...], int]:
    ndim, offset = _get_u8(data, offset)
    values = []
    for _ in range(ndim):
        value, offset = _get_u16(data, offset)
        values.append(value)
    return tuple(values), offset


def _put_varshape(out: bytearray, shape: tuple[int, ...]) -> None:
    _put_varuint(out, len(shape))
    for value in shape:
        _put_varuint(out, _require_non_negative(value, "shape dimensions"))


def _get_varshape(data: bytes, offset: int) -> tuple[tuple[int, ...], int]:
    ndim, offset = _get_varuint(data, offset)
    values = []
    for _ in range(ndim):
        value, offset = _get_varuint(data, offset)
        values.append(value)
    return tuple(values), offset


def _put_varuint(out: bytearray, value: int) -> None:
    if value < 0:
        raise ValueError("varuint values must be non-negative")
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def _get_varuint(data: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        byte, offset = _get_u8(data, offset)
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("compact varuint is too large")


def _put_varint(out: bytearray, value: int) -> None:
    encoded = value << 1 if value >= 0 else ((-value) << 1) - 1
    _put_varuint(out, encoded)


def _get_varint(data: bytes, offset: int) -> tuple[int, int]:
    raw, offset = _get_varuint(data, offset)
    return (raw >> 1) ^ -(raw & 1), offset


def _require_non_negative(value: int, field_name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _get_u8(data: bytes, offset: int) -> tuple[int, int]:
    return _unpack_at(U8_STRUCT, data, offset)


def _get_u16(data: bytes, offset: int) -> tuple[int, int]:
    return _unpack_at(U16_STRUCT, data, offset)


def _get_u32(data: bytes, offset: int) -> tuple[int, int]:
    return _unpack_at(U32_STRUCT, data, offset)


def _get_i16(data: bytes, offset: int) -> tuple[int, int]:
    return _unpack_at(I16_STRUCT, data, offset)


def _get_i64(data: bytes, offset: int) -> tuple[int, int]:
    return _unpack_at(I64_STRUCT, data, offset)


def _unpack_at(fmt: struct.Struct, data: bytes, offset: int) -> tuple[int, int]:
    end = offset + fmt.size
    if end > len(data):
        raise ValueError("truncated compact CoSER-DiC header")
    (value,) = fmt.unpack(data[offset:end])
    return int(value), end
