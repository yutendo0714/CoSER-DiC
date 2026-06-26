from __future__ import annotations

from collections.abc import Mapping


FORBIDDEN_DECODER_SIDE_INFO = {
    "original",
    "original_image",
    "caption",
    "segmentation",
    "segmentation_map",
    "ocr_map",
    "ocr_mask",
    "face_mask",
    "importance_map",
    "encoder_only_tensor",
    "debug_tensor",
}


def assert_no_forbidden_side_info(metadata: Mapping[str, object]) -> None:
    present = sorted(FORBIDDEN_DECODER_SIDE_INFO.intersection(metadata.keys()))
    if present:
        joined = ", ".join(present)
        raise ValueError(f"forbidden decoder side information present: {joined}")

