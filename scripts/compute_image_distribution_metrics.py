from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

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


def fixed_grid_positions(length: int, patch_size: int, *, offset: int = 0) -> list[int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if length - offset < patch_size:
        return []
    return list(range(offset, length - patch_size + 1, patch_size))


def infer_gencodec_patch_group(path_or_name: str | Path) -> str:
    dataset = infer_gencodec_dataset(path_or_name)
    if dataset == "kodak":
        return "kodak"
    return "other"


def infer_gencodec_dataset(path_or_name: str | Path) -> str:
    text = str(path_or_name).lower()
    name = Path(path_or_name).name.lower()
    stem = Path(path_or_name).stem.lower()
    if "/kodak/" in text or name.startswith("kodim") or "_kodim" in name:
        return "kodak"
    if "/clic/" in text or "clic" in name:
        return "clic2020_test"
    numeric_tokens = [token for token in stem.replace("-", "_").split("_") if token.isdigit()]
    looks_like_div2k_val = any(801 <= int(token) <= 900 for token in numeric_tokens)
    if "/div2k/" in text or looks_like_div2k_val:
        return "div2k_val"
    return "other"


def dataset_matches_filter(dataset: str, dataset_filter: str) -> bool:
    if dataset_filter == "all":
        return True
    if dataset_filter == "other":
        return dataset != "kodak"
    return dataset == dataset_filter


def load_manifest_patch_groups(manifest_jsonl: Path | None) -> dict[str, str]:
    if manifest_jsonl is None or not manifest_jsonl.exists():
        return {}
    groups: dict[str, str] = {}
    with manifest_jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source_path") or row.get("path") or ""
            group = infer_gencodec_dataset(source)
            for key in ("reference", "semantic_only", "stage3", "stage4", "stage4_image"):
                image_path = row.get(key)
                if image_path:
                    groups[Path(image_path).name] = group
    return groups


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


def extract_gencodec_patches_to_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    kodak_patch_size: int = 64,
    other_patch_size: int = 256,
    fid_patch_num: int = 2,
    dataset_filter: str = "all",
    manifest_jsonl: Path | None = None,
    deep: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    if kodak_patch_size <= 0 or other_patch_size <= 0:
        raise ValueError("gencodec patch sizes must be positive")
    if fid_patch_num <= 0:
        raise ValueError("fid_patch_num must be positive")
    for patch_size in (kodak_patch_size, other_patch_size):
        if patch_size % fid_patch_num != 0:
            raise ValueError("patch sizes must be divisible by fid_patch_num for GenCodec-style shifts")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_groups = load_manifest_patch_groups(manifest_jsonl)
    image_paths = list_image_files(input_dir, deep=deep)
    if limit > 0:
        image_paths = image_paths[:limit]

    total_patches = 0
    images_by_dataset: Counter[str] = Counter()
    images_by_group: Counter[str] = Counter()
    patches_by_dataset: Counter[str] = Counter()
    patches_by_group: Counter[str] = Counter()
    patches_by_patch_size: Counter[int] = Counter()
    skipped_shift_grids = 0
    unit_i_values = list(range(1, fid_patch_num))

    for image_index, image_path in enumerate(image_paths):
        dataset = manifest_groups.get(image_path.name) or infer_gencodec_dataset(image_path)
        if not dataset_matches_filter(dataset, dataset_filter):
            continue
        group = "kodak" if dataset == "kodak" else "other"
        patch_size = kodak_patch_size if group == "kodak" else other_patch_size
        images_by_dataset[dataset] += 1
        images_by_group[group] += 1

        image = Image.open(image_path).convert("RGB")
        offsets: list[tuple[int, int]] = [(0, 0)]
        unit = patch_size // fid_patch_num
        for unit_i in unit_i_values:
            limit_size = (2.0 - unit_i / fid_patch_num) * patch_size
            if image.height >= limit_size and image.width >= limit_size:
                offset = unit * unit_i
                offsets.append((offset, offset))
            else:
                skipped_shift_grids += 1

        patch_index = 0
        for offset_y, offset_x in offsets:
            xs = fixed_grid_positions(image.width, patch_size, offset=offset_x)
            ys = fixed_grid_positions(image.height, patch_size, offset=offset_y)
            for top in ys:
                for left in xs:
                    patch = image.crop((left, top, left + patch_size, top + patch_size))
                    patch.save(
                        output_dir
                        / (
                            f"image{image_index:05d}_ps{patch_size:04d}"
                            f"_oy{offset_y:04d}_ox{offset_x:04d}"
                            f"_patch{patch_index:04d}_{image_path.stem}.png"
                        )
                    )
                    patch_index += 1
                    total_patches += 1
                    patches_by_dataset[dataset] += 1
                    patches_by_group[group] += 1
                    patches_by_patch_size[patch_size] += 1

    return {
        "num_patches": total_patches,
        "dataset_filter": dataset_filter,
        "num_images_by_dataset": dict(sorted(images_by_dataset.items())),
        "num_images_by_patch_group": dict(sorted(images_by_group.items())),
        "num_patches_by_dataset": dict(sorted(patches_by_dataset.items())),
        "num_patches_by_patch_group": dict(sorted(patches_by_group.items())),
        "num_patches_by_patch_size": {str(key): value for key, value in sorted(patches_by_patch_size.items())},
        "skipped_shift_grids": skipped_shift_grids,
    }


def count_images(path: Path, *, deep: bool = False) -> int:
    return len(list_image_files(path, deep=deep))


class UInt8ImageDataset(Dataset):
    def __init__(self, root: Path, *, deep: bool = False) -> None:
        self.paths = list_image_files(root, deep=deep)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        return TF.pil_to_tensor(image)


@torch.no_grad()
def calculate_torchmetrics_fid(
    reference_input: Path,
    candidate_input: Path,
    *,
    batch_size: int,
    cuda: bool,
) -> dict[str, float]:
    from torchmetrics.image.fid import FrechetInceptionDistance

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    fid_metric = FrechetInceptionDistance(normalize=False).to(device)
    for root, real in ((reference_input, True), (candidate_input, False)):
        dataset = UInt8ImageDataset(root)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for batch in loader:
            fid_metric.update(batch.to(device), real=real)
    return {"torchmetrics_frechet_inception_distance": float(fid_metric.compute().item())}


def maybe_prepare_patch_inputs(args: argparse.Namespace, output_json: Path | None) -> tuple[Path, Path, dict[str, Any]]:
    if args.patch_protocol == "single_grid" and args.patch_size <= 0:
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
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if args.patch_protocol == "gencodec":
            cache_root = base / (
                "patch_cache_gencodec"
                f"_kodak{args.gencodec_kodak_patch_size}"
                f"_other{args.gencodec_other_patch_size}"
                f"_n{args.fid_patch_num}_{stamp}"
            )
        else:
            cache_root = base / f"patch_cache_ps{args.patch_size}_s{args.patch_stride}_{stamp}"
    if cache_root.exists():
        reference_patch_dir = cache_root / "reference"
        candidate_patch_dir = cache_root / "candidate"
        if args.reuse_patch_cache and reference_patch_dir.exists() and candidate_patch_dir.exists():
            reference_patch_count = count_images(reference_patch_dir)
            candidate_patch_count = count_images(candidate_patch_dir)
            summary: dict[str, Any] = {
                "mode": "patch",
                "patch_protocol": args.patch_protocol,
                "patch_cache_dir": str(cache_root),
                "patch_cache_reused": True,
                "num_reference_images": count_images(args.reference_dir, deep=args.samples_find_deep),
                "num_candidate_images": count_images(args.candidate_dir, deep=args.samples_find_deep),
                "num_reference_patches": reference_patch_count,
                "num_candidate_patches": candidate_patch_count,
            }
            if args.patch_protocol == "gencodec":
                summary.update(
                    {
                        "gencodec_kodak_patch_size": args.gencodec_kodak_patch_size,
                        "gencodec_other_patch_size": args.gencodec_other_patch_size,
                        "gencodec_dataset_filter": args.gencodec_dataset_filter,
                        "fid_patch_num": args.fid_patch_num,
                    }
                )
            else:
                summary.update({"patch_size": args.patch_size, "patch_stride": args.patch_stride})
            return reference_patch_dir, candidate_patch_dir, summary
        if not args.overwrite_patch_cache:
            raise FileExistsError(f"patch cache already exists: {cache_root}")
        shutil.rmtree(cache_root)
    reference_patch_dir = cache_root / "reference"
    candidate_patch_dir = cache_root / "candidate"
    if args.patch_protocol == "gencodec":
        manifest_jsonl = args.manifest_jsonl
        if manifest_jsonl is None:
            candidate_manifest = args.reference_dir.parent / "manifest.jsonl"
            if candidate_manifest.exists():
                manifest_jsonl = candidate_manifest
        reference_patch_summary = extract_gencodec_patches_to_dir(
            args.reference_dir,
            reference_patch_dir,
            kodak_patch_size=args.gencodec_kodak_patch_size,
            other_patch_size=args.gencodec_other_patch_size,
            fid_patch_num=args.fid_patch_num,
            dataset_filter=args.gencodec_dataset_filter,
            manifest_jsonl=manifest_jsonl,
            deep=args.samples_find_deep,
            limit=args.max_images,
        )
        candidate_patch_summary = extract_gencodec_patches_to_dir(
            args.candidate_dir,
            candidate_patch_dir,
            kodak_patch_size=args.gencodec_kodak_patch_size,
            other_patch_size=args.gencodec_other_patch_size,
            fid_patch_num=args.fid_patch_num,
            dataset_filter=args.gencodec_dataset_filter,
            manifest_jsonl=manifest_jsonl,
            deep=args.samples_find_deep,
            limit=args.max_images,
        )
        return reference_patch_dir, candidate_patch_dir, {
            "mode": "patch",
            "patch_protocol": "gencodec",
            "gencodec_kodak_patch_size": args.gencodec_kodak_patch_size,
            "gencodec_other_patch_size": args.gencodec_other_patch_size,
            "gencodec_dataset_filter": args.gencodec_dataset_filter,
            "fid_patch_num": args.fid_patch_num,
            "patch_cache_dir": str(cache_root),
            "patch_cache_reused": False,
            "manifest_jsonl": str(manifest_jsonl) if manifest_jsonl is not None else "",
            "num_reference_images": count_images(args.reference_dir, deep=args.samples_find_deep),
            "num_candidate_images": count_images(args.candidate_dir, deep=args.samples_find_deep),
            "num_reference_patches": reference_patch_summary["num_patches"],
            "num_candidate_patches": candidate_patch_summary["num_patches"],
            "reference_patch_summary": reference_patch_summary,
            "candidate_patch_summary": candidate_patch_summary,
        }

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
        "patch_protocol": "single_grid",
        "patch_size": args.patch_size,
        "patch_stride": args.patch_stride,
        "patch_cache_dir": str(cache_root),
        "patch_cache_reused": False,
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
    parser.add_argument("--metric-backend", choices=("torch_fidelity", "torchmetrics_fid"), default="torch_fidelity")
    parser.add_argument("--no-fid", action="store_true")
    parser.add_argument("--no-kid", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--samples-find-deep", action="store_true")
    parser.add_argument("--patch-protocol", choices=("single_grid", "gencodec"), default="single_grid")
    parser.add_argument("--patch-size", type=int, default=0)
    parser.add_argument("--patch-stride", type=int, default=0)
    parser.add_argument("--fid-patch-num", type=int, default=2)
    parser.add_argument("--gencodec-kodak-patch-size", type=int, default=64)
    parser.add_argument("--gencodec-other-patch-size", type=int, default=256)
    parser.add_argument(
        "--gencodec-dataset-filter",
        choices=("all", "kodak", "clic2020_test", "div2k_val", "other"),
        default="all",
    )
    parser.add_argument("--manifest-jsonl", type=Path, default=None)
    parser.add_argument("--patch-cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite-patch-cache", action="store_true")
    parser.add_argument("--reuse-patch-cache", action="store_true")
    parser.add_argument("--delete-patch-cache-after", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.reference_dir.exists():
        raise FileNotFoundError(args.reference_dir)
    if not args.candidate_dir.exists():
        raise FileNotFoundError(args.candidate_dir)
    if args.patch_protocol == "single_grid" and args.patch_size > 0 and args.patch_stride <= 0:
        args.patch_stride = args.patch_size
    if (
        args.patch_protocol == "gencodec"
        and args.gencodec_dataset_filter == "all"
        and args.gencodec_kodak_patch_size != args.gencodec_other_patch_size
        and not args.dry_run
        and (not args.no_fid or not args.no_kid)
    ):
        raise ValueError(
            "GenCodec Kodak and non-Kodak patch sizes differ. Run distribution metrics per dataset "
            "with --gencodec-dataset-filter kodak/clic2020_test/div2k_val, or use --dry-run only."
        )

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
    if args.metric_backend == "torchmetrics_fid" and kid and not args.dry_run:
        raise ValueError("torchmetrics_fid backend supports FID only; pass --no-kid")

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
        "metric_backend": args.metric_backend,
        **input_summary,
    }
    if not args.dry_run:
        if args.metric_backend == "torchmetrics_fid":
            if not fid:
                metrics = {}
            else:
                metrics = calculate_torchmetrics_fid(
                    reference_input,
                    candidate_input,
                    batch_size=args.batch_size,
                    cuda=not args.no_cuda,
                )
            payload["metrics"] = metrics
        else:
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
    if (
        args.delete_patch_cache_after
        and input_summary.get("mode") == "patch"
        and not input_summary.get("patch_cache_reused", False)
        and input_summary.get("patch_cache_dir")
    ):
        shutil.rmtree(Path(str(input_summary["patch_cache_dir"])), ignore_errors=True)
        payload["patch_cache_deleted_after"] = True
        if output_json is not None:
            output_json.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
