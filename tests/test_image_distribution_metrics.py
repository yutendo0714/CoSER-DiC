from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from scripts.compute_image_distribution_metrics import (
    crop_positions,
    extract_gencodec_patches_to_dir,
    extract_patches_to_dir,
    fixed_grid_positions,
    maybe_prepare_patch_inputs,
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


def test_extract_gencodec_patches_uses_manifest_stage4_group(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "patches"
    manifest = tmp_path / "manifest.jsonl"
    input_dir.mkdir()
    stage4_path = input_dir / "image00000_stage4.png"
    Image.new("RGB", (512, 512), color=(32, 64, 128)).save(stage4_path)
    manifest.write_text(
        json.dumps(
            {
                "source_path": "/dpl/clic/professional/test/example.png",
                "stage4": str(stage4_path),
            }
        )
        + "\n"
    )

    summary = extract_gencodec_patches_to_dir(
        input_dir,
        output_dir,
        kodak_patch_size=64,
        other_patch_size=128,
        fid_patch_num=2,
        dataset_filter="clic2020_test",
        manifest_jsonl=manifest,
    )

    assert summary["num_images_by_dataset"] == {"clic2020_test": 1}
    assert summary["num_patches_by_patch_size"] == {"128": 25}
    assert summary["num_patches"] == 25


def test_maybe_prepare_patch_inputs_can_reuse_existing_cache(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    cache_dir = tmp_path / "patch_cache"
    reference_dir.mkdir()
    candidate_dir.mkdir()
    Image.new("RGB", (128, 128), color=(128, 64, 32)).save(reference_dir / "sample.png")
    Image.new("RGB", (128, 128), color=(32, 64, 128)).save(candidate_dir / "sample.png")

    args = argparse.Namespace(
        reference_dir=reference_dir,
        candidate_dir=candidate_dir,
        samples_find_deep=False,
        patch_protocol="single_grid",
        patch_size=64,
        patch_stride=64,
        patch_cache_dir=cache_dir,
        reuse_patch_cache=False,
        overwrite_patch_cache=False,
        gencodec_kodak_patch_size=64,
        gencodec_other_patch_size=256,
        gencodec_dataset_filter="all",
        fid_patch_num=2,
        manifest_jsonl=None,
        max_images=0,
    )

    _, _, first_summary = maybe_prepare_patch_inputs(args, tmp_path / "first.json")
    args.reuse_patch_cache = True
    _, _, reused_summary = maybe_prepare_patch_inputs(args, tmp_path / "second.json")

    assert first_summary["patch_cache_reused"] is False
    assert reused_summary["patch_cache_reused"] is True
    assert reused_summary["num_reference_patches"] == 4
    assert reused_summary["num_candidate_patches"] == 4


def test_distribution_metric_cli_deletes_generated_patch_cache_after_dry_run(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    cache_dir = tmp_path / "patch_cache"
    output_json = tmp_path / "metrics.json"
    reference_dir.mkdir()
    candidate_dir.mkdir()
    Image.new("RGB", (128, 128), color=(128, 64, 32)).save(reference_dir / "sample.png")
    Image.new("RGB", (128, 128), color=(32, 64, 128)).save(candidate_dir / "sample.png")

    subprocess.run(
        [
            sys.executable,
            "scripts/compute_image_distribution_metrics.py",
            "--reference-dir",
            str(reference_dir),
            "--candidate-dir",
            str(candidate_dir),
            "--patch-size",
            "64",
            "--patch-stride",
            "64",
            "--patch-cache-dir",
            str(cache_dir),
            "--delete-patch-cache-after",
            "--dry-run",
            "--output-json",
            str(output_json),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text())
    assert payload["patch_cache_deleted_after"] is True
    assert not cache_dir.exists()


def test_torchmetrics_backend_requires_no_kid(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    reference_dir.mkdir()
    candidate_dir.mkdir()
    for index in range(2):
        Image.new("RGB", (64, 64), color=(index, 64, 32)).save(reference_dir / f"sample{index}.png")
        Image.new("RGB", (64, 64), color=(32, 64, index)).save(candidate_dir / f"sample{index}.png")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_image_distribution_metrics.py",
            "--reference-dir",
            str(reference_dir),
            "--candidate-dir",
            str(candidate_dir),
            "--metric-backend",
            "torchmetrics_fid",
            "--no-fid",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "torchmetrics_fid backend supports FID only" in result.stderr
