from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.compute_image_distribution_metrics import crop_positions, extract_patches_to_dir


def test_crop_positions_cover_last_patch() -> None:
    assert crop_positions(512, patch_size=256, stride=256) == [0, 256]
    assert crop_positions(512, patch_size=256, stride=192) == [0, 192, 256]
    assert crop_positions(128, patch_size=256, stride=128) == [0]


def test_extract_patches_to_dir_writes_overlapped_grid(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "patches"
    input_dir.mkdir()
    Image.new("RGB", (512, 512), color=(128, 64, 32)).save(input_dir / "sample.png")

    count = extract_patches_to_dir(input_dir, output_dir, patch_size=256, stride=128)

    assert count == 9
    assert len(list(output_dir.glob("*.png"))) == 9
