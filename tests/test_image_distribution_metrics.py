from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.compute_image_distribution_metrics import (
    crop_positions,
    extract_gencodec_patches_to_dir,
    extract_patches_to_dir,
    fixed_grid_positions,
)


def test_crop_positions_cover_last_patch() -> None:
    assert crop_positions(512, patch_size=256, stride=256) == [0, 256]
    assert crop_positions(512, patch_size=256, stride=192) == [0, 192, 256]
    assert crop_positions(128, patch_size=256, stride=128) == [0]


def test_fixed_grid_positions_do_not_force_edge_patch() -> None:
    assert fixed_grid_positions(512, patch_size=256) == [0, 256]
    assert fixed_grid_positions(512, patch_size=256, offset=128) == [128]
    assert fixed_grid_positions(500, patch_size=256) == [0]
    assert fixed_grid_positions(128, patch_size=256) == []


def test_extract_patches_to_dir_writes_overlapped_grid(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "patches"
    input_dir.mkdir()
    Image.new("RGB", (512, 512), color=(128, 64, 32)).save(input_dir / "sample.png")

    count = extract_patches_to_dir(input_dir, output_dir, patch_size=256, stride=128)

    assert count == 9
    assert len(list(output_dir.glob("*.png"))) == 9


def test_extract_gencodec_patches_uses_dataset_patch_sizes_and_half_shift(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "patches"
    input_dir.mkdir()
    Image.new("RGB", (512, 512), color=(128, 64, 32)).save(input_dir / "image00000_kodim01.png")
    Image.new("RGB", (512, 512), color=(32, 64, 128)).save(input_dir / "image00024_clic_sample.png")

    summary = extract_gencodec_patches_to_dir(
        input_dir,
        output_dir,
        kodak_patch_size=64,
        other_patch_size=256,
        fid_patch_num=2,
    )

    assert summary["num_images_by_patch_group"] == {"kodak": 1, "other": 1}
    assert summary["num_patches_by_patch_group"] == {"kodak": 113, "other": 5}
    assert summary["num_patches_by_patch_size"] == {"64": 113, "256": 5}
    assert summary["num_patches"] == 118
    assert len(list(output_dir.glob("*.png"))) == 118


def test_extract_gencodec_patches_can_filter_dataset(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "patches"
    input_dir.mkdir()
    Image.new("RGB", (512, 512), color=(128, 64, 32)).save(input_dir / "image00000_kodim01.png")
    Image.new("RGB", (512, 512), color=(32, 64, 128)).save(input_dir / "image00452_0801.png")

    summary = extract_gencodec_patches_to_dir(
        input_dir,
        output_dir,
        kodak_patch_size=64,
        other_patch_size=256,
        fid_patch_num=2,
        dataset_filter="div2k_val",
    )

    assert summary["num_images_by_dataset"] == {"div2k_val": 1}
    assert summary["num_patches_by_patch_size"] == {"256": 5}
    assert summary["num_patches"] == 5
    assert len(list(output_dir.glob("*.png"))) == 5
