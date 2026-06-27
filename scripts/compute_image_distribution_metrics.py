from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from coserdic.datasets.image_folder import IMAGE_SUFFIXES


def list_image_files(root: Path, *, deep: bool = False) -> list[Path]:
    iterator = root.rglob("*") if deep else root.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def crop_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def extract_patches_to_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    patch_size: int,
    stride: int,
    deep: bool = False,
    limit: int = 0,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    image_paths = list_image_files(input_dir, deep=deep)
    if limit > 0:
        image_paths = image_paths[:limit]
    for image_index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        if image.width < patch_size or image.height < patch_size:
            image = image.resize((max(image.width, patch_size), max(image.height, patch_size)), Image.Resampling.BICUBIC)
        xs = crop_positions(image.width, patch_size, stride)
        ys = crop_positions(image.height, patch_size, stride)
        for patch_index, (top, left) in enumerate((top, left) for top in ys for left in xs):
            patch = image.crop((left, top, left + patch_size, top + patch_size))
            patch.save(output_dir / f"image{image_index:05d}_patch{patch_index:04d}_{image_path.stem}.png")
            count += 1
    return count


def count_images(path: Path, *, deep: bool = False) -> int:
    return len(list_image_files(path, deep=deep))


def maybe_prepare_patch_inputs(args: argparse.Namespace, output_json: Path | None) -> tuple[Path, Path, dict[str, Any]]:
    if args.patch_size <= 0:
        reference_count = count_images(args.reference_dir, deep=args.samples_find_deep)
        candidate_count = count_images(args.candidate_dir, deep=args.samples_find_deep)
        return args.reference_dir, args.candidate_dir, {
            "mode": "image",
            "num_reference_images": reference_count,
            "num_candidate_images": candidate_count,
        }

    cache_root = args.patch_cache_dir
    if cache_root is None:
        base = output_json.parent if output_json is not None else Path("results/analysis/image_distribution_metrics")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_root = base / f"patch_cache_ps{args.patch_size}_s{args.patch_stride}_{stamp}"
    if cache_root.exists():
        if not args.overwrite_patch_cache:
            raise FileExistsError(f"patch cache already exists: {cache_root}")
        shutil.rmtree(cache_root)
    reference_patch_dir = cache_root / "reference"
    candidate_patch_dir = cache_root / "candidate"
    reference_patch_count = extract_patches_to_dir(
        args.reference_dir,
        reference_patch_dir,
        patch_size=args.patch_size,
        stride=args.patch_stride,
        deep=args.samples_find_deep,
        limit=args.max_images,
    )
    candidate_patch_count = extract_patches_to_dir(
        args.candidate_dir,
        candidate_patch_dir,
        patch_size=args.patch_size,
        stride=args.patch_stride,
        deep=args.samples_find_deep,
        limit=args.max_images,
    )
    return reference_patch_dir, candidate_patch_dir, {
        "mode": "patch",
        "patch_size": args.patch_size,
        "patch_stride": args.patch_stride,
        "patch_cache_dir": str(cache_root),
        "num_reference_images": count_images(args.reference_dir, deep=args.samples_find_deep),
        "num_candidate_images": count_images(args.candidate_dir, deep=args.samples_find_deep),
        "num_reference_patches": reference_patch_count,
        "num_candidate_patches": candidate_patch_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--kid-subset-size", type=int, default=0)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--no-fid", action="store_true")
    parser.add_argument("--no-kid", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--samples-find-deep", action="store_true")
    parser.add_argument("--patch-size", type=int, default=0)
    parser.add_argument("--patch-stride", type=int, default=0)
    parser.add_argument("--patch-cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite-patch-cache", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.reference_dir.exists():
        raise FileNotFoundError(args.reference_dir)
    if not args.candidate_dir.exists():
        raise FileNotFoundError(args.candidate_dir)
    if args.patch_size > 0 and args.patch_stride <= 0:
        args.patch_stride = args.patch_size

    output_json = args.output_json
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
    reference_input, candidate_input, input_summary = maybe_prepare_patch_inputs(args, output_json)
    sample_count = min(
        int(input_summary.get("num_reference_patches", input_summary["num_reference_images"])),
        int(input_summary.get("num_candidate_patches", input_summary["num_candidate_images"])),
    )
    if sample_count <= 0:
        raise ValueError("no images or patches found for distribution metrics")

    fid = not args.no_fid
    kid = not args.no_kid
    kid_subset_size = args.kid_subset_size or min(1000, sample_count)
    if kid and kid_subset_size > sample_count:
        kid_subset_size = sample_count
    if kid and kid_subset_size < 2:
        raise ValueError("KID requires at least 2 samples; pass --no-kid or provide more images/patches")

    payload: dict[str, Any] = {
        "label": args.label,
        "reference_dir": str(args.reference_dir),
        "candidate_dir": str(args.candidate_dir),
        "reference_input": str(reference_input),
        "candidate_input": str(candidate_input),
        "fid_requested": fid,
        "kid_requested": kid,
        "kid_subset_size": kid_subset_size if kid else 0,
        "kid_subsets": args.kid_subsets if kid else 0,
        "batch_size": args.batch_size,
        "cuda": not args.no_cuda,
        **input_summary,
    }
    if not args.dry_run:
        from torch_fidelity import calculate_metrics

        metrics = calculate_metrics(
            input1=str(reference_input),
            input2=str(candidate_input),
            cuda=not args.no_cuda,
            batch_size=args.batch_size,
            fid=fid,
            kid=kid,
            kid_subset_size=kid_subset_size,
            kid_subsets=args.kid_subsets,
            samples_find_deep=False,
            samples_find_ext="png,jpg,jpeg",
            samples_ext_lossy="jpg,jpeg",
            verbose=True,
        )
        payload["metrics"] = {key: float(value) for key, value in metrics.items()}
    else:
        payload["metrics"] = {}

    if output_json is not None:
        output_json.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
