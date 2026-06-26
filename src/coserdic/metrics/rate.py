from __future__ import annotations


def bytes_to_bpp(num_bytes: int, height: int, width: int) -> float:
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    return 8.0 * float(num_bytes) / float(height * width)


def actual_bpp(bitstream: bytes, height: int, width: int) -> float:
    return bytes_to_bpp(len(bitstream), height, width)


def estimated_actual_gap_ratio(estimated_bpp: float, actual_bpp_value: float) -> float:
    if estimated_bpp <= 0:
        raise ValueError("estimated_bpp must be positive")
    return (actual_bpp_value - estimated_bpp) / estimated_bpp
