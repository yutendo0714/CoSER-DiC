from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .image_folder import list_images


@dataclass(frozen=True)
class EvalDatasetSelection:
    key: str
    display_name: str
    paths: tuple[Path, ...]
    expected_count: int | None
    source_roots: tuple[Path, ...]
    notes: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.paths)

    @property
    def status(self) -> str:
        if self.expected_count is None:
            return "unchecked"
        return "ok" if self.count == self.expected_count else "count_mismatch"

    def to_summary(self) -> dict[str, object]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "count": self.count,
            "expected_count": self.expected_count,
            "status": self.status,
            "source_roots": [str(root) for root in self.source_roots],
            "first_path": str(self.paths[0]) if self.paths else "",
            "last_path": str(self.paths[-1]) if self.paths else "",
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EvalProtocolSpec:
    name: str
    display_name: str
    dataset_keys: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    bpp_metric: str
    default_crop_size: int | None
    notes: tuple[str, ...]


class EvalProtocolCountError(ValueError):
    pass


EVAL_PROTOCOLS: dict[str, EvalProtocolSpec] = {
    "coser_common_lic": EvalProtocolSpec(
        name="coser_common_lic",
        display_name="CoSER common LIC protocol",
        dataset_keys=("kodak", "clic_professional_valid", "div2k_val"),
        primary_metrics=("PSNR", "MS-SSIM", "LPIPS", "DISTS", "FID/KID when sample handling is defined"),
        bpp_metric="actual_payload_bpp",
        default_crop_size=None,
        notes=(
            "Internal CoSER LIC protocol for Kodak 24, CLIC Professional Validation 41, and DIV2K validation 100.",
            "Do not merge this table with CLIC2020 test 428 comparisons.",
        ),
    ),
    "coser_common_lic_plus_mobile": EvalProtocolSpec(
        name="coser_common_lic_plus_mobile",
        display_name="CoSER common LIC protocol plus CLIC Mobile Validation",
        dataset_keys=("kodak", "clic_professional_valid", "clic_mobile_valid", "div2k_val"),
        primary_metrics=("PSNR", "MS-SSIM", "LPIPS", "DISTS", "FID/KID when sample handling is defined"),
        bpp_metric="actual_payload_bpp",
        default_crop_size=None,
        notes=(
            "Internal CoSER robustness/generalization protocol with CLIC Professional and Mobile validation split separately.",
            "The CLIC validation splits are not the CLIC2020 test set used by GenCodec/CoD-style reproduction.",
        ),
    ),
    "cod_reproduction_512": EvalProtocolSpec(
        name="cod_reproduction_512",
        display_name="CoD paper reproduction protocol, 512x512",
        dataset_keys=("kodak", "clic2020_test", "div2k_val"),
        primary_metrics=("PSNR", "LPIPS", "DISTS", "FID"),
        bpp_metric="actual_payload_bpp for CoSER; official nominal/file bpp must be labeled for external codecs",
        default_crop_size=512,
        notes=(
            "Primary CoSER low-bitrate generative comparison protocol for CoD-style tables.",
            "Resize if needed and center-crop every image to 512x512 before evaluation.",
            "Use CLIC2020 test 428 as professional/test plus mobile/test.",
            "Use DIV2K validation images 0801-0900, not the first 100 files from a mixed DIV2K root.",
            "For CoD paper patch FID, run per dataset and label exact settings: Kodak512 uses 64px patches with split=2; CLIC2020 uses 128px patches with split=2.",
            "DIV2K patch FID should be reported only when the chosen patch size/backend is explicitly labeled.",
            "Patch-based or overlapped-patch FID must label patch size, shift/split count, and backend.",
        ),
    ),
    "gencodec_reproduction": EvalProtocolSpec(
        name="gencodec_reproduction",
        display_name="Legacy alias for CoD 512 reproduction protocol",
        dataset_keys=("kodak", "clic2020_test", "div2k_val"),
        primary_metrics=("PSNR", "LPIPS", "DISTS", "FID"),
        bpp_metric="actual_payload_bpp for CoSER; official nominal/file bpp must be labeled for external codecs",
        default_crop_size=512,
        notes=(
            "Legacy name retained for old experiment commands. Prefer cod_reproduction_512 for new runs.",
            "Same dataset and 512x512 center-crop preprocessing as cod_reproduction_512.",
            "Do not use this name for CoD-Lite full-resolution comparison tables.",
        ),
    ),
}


def available_eval_protocols() -> tuple[str, ...]:
    return tuple(sorted(EVAL_PROTOCOLS))


def resolve_eval_protocol(
    protocol: str,
    *,
    dpl_root: str | Path = "/dpl",
    dataset_keys: Iterable[str] | None = None,
    strict_expected_counts: bool = False,
) -> list[EvalDatasetSelection]:
    if protocol not in EVAL_PROTOCOLS:
        raise KeyError(f"unknown eval protocol: {protocol}")
    spec = EVAL_PROTOCOLS[protocol]
    keys = tuple(dataset_keys) if dataset_keys is not None else spec.dataset_keys
    allowed = set(spec.dataset_keys)
    unknown = [key for key in keys if key not in allowed]
    if unknown:
        raise KeyError(f"{protocol} does not contain dataset(s): {', '.join(unknown)}")
    selections = [resolve_eval_dataset(key, dpl_root=dpl_root) for key in keys]
    if strict_expected_counts:
        validate_expected_counts(selections)
    return selections


def resolve_eval_dataset(key: str, *, dpl_root: str | Path = "/dpl") -> EvalDatasetSelection:
    root = Path(dpl_root)
    if key == "kodak":
        path = root / "kodak"
        return EvalDatasetSelection(
            key=key,
            display_name="Kodak 24",
            paths=tuple(_list_images_if_exists(path)),
            expected_count=24,
            source_roots=(path,),
        )
    if key == "clic_professional_valid":
        path = root / "clic" / "professional" / "valid"
        return EvalDatasetSelection(
            key=key,
            display_name="CLIC Professional Validation 41",
            paths=tuple(_list_images_if_exists(path)),
            expected_count=41,
            source_roots=(path,),
        )
    if key == "clic_mobile_valid":
        path = root / "clic" / "mobile" / "valid"
        return EvalDatasetSelection(
            key=key,
            display_name="CLIC Mobile Validation 61",
            paths=tuple(_list_images_if_exists(path)),
            expected_count=61,
            source_roots=(path,),
        )
    if key == "clic2020_test":
        professional = root / "clic" / "professional" / "test"
        mobile = root / "clic" / "mobile" / "test"
        paths = tuple(_list_images_if_exists(professional) + _list_images_if_exists(mobile))
        return EvalDatasetSelection(
            key=key,
            display_name="CLIC2020 test 428",
            paths=paths,
            expected_count=428,
            source_roots=(professional, mobile),
            notes=("professional/test and mobile/test are concatenated in that order.",),
        )
    if key == "div2k_val":
        return _resolve_div2k_val(root)
    raise KeyError(f"unknown eval dataset: {key}")


def flatten_selection_paths(selections: Iterable[EvalDatasetSelection]) -> list[Path]:
    return [path for selection in selections for path in selection.paths]


def protocol_summary(protocol: str, selections: Iterable[EvalDatasetSelection]) -> dict[str, object]:
    spec = EVAL_PROTOCOLS[protocol]
    selection_list = list(selections)
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "dataset_keys": [selection.key for selection in selection_list],
        "dataset_counts": {selection.key: selection.count for selection in selection_list},
        "expected_counts": {selection.key: selection.expected_count for selection in selection_list},
        "count_status": {selection.key: selection.status for selection in selection_list},
        "total_images": sum(selection.count for selection in selection_list),
        "primary_metrics": list(spec.primary_metrics),
        "bpp_metric": spec.bpp_metric,
        "default_crop_size": spec.default_crop_size,
        "notes": list(spec.notes),
        "datasets": [selection.to_summary() for selection in selection_list],
    }


def validate_expected_counts(selections: Iterable[EvalDatasetSelection]) -> None:
    mismatches = [
        selection
        for selection in selections
        if selection.expected_count is not None and selection.count != selection.expected_count
    ]
    if mismatches:
        details = ", ".join(
            f"{selection.key}: got {selection.count}, expected {selection.expected_count}" for selection in mismatches
        )
        raise EvalProtocolCountError(details)


def _resolve_div2k_val(dpl_root: Path) -> EvalDatasetSelection:
    mixed_root = dpl_root / "div2k"
    explicit_candidates = (
        dpl_root / "DIV2K_valid_HR",
        mixed_root / "DIV2K_valid_HR",
        mixed_root / "valid",
        mixed_root / "val",
    )
    for candidate in explicit_candidates:
        paths = _list_images_if_exists(candidate)
        if paths:
            return EvalDatasetSelection(
                key="div2k_val",
                display_name="DIV2K validation 100",
                paths=tuple(paths),
                expected_count=100,
                source_roots=(candidate,),
                notes=("Resolved from an explicit validation directory.",),
            )
    mixed_paths = _list_images_if_exists(mixed_root)
    val_paths = [path for path in mixed_paths if _div2k_index(path) in range(801, 901)]
    return EvalDatasetSelection(
        key="div2k_val",
        display_name="DIV2K validation 100",
        paths=tuple(val_paths),
        expected_count=100,
        source_roots=(mixed_root,),
        notes=("Resolved by filename indices 0801-0900 from a mixed DIV2K root.",),
    )


def _div2k_index(path: Path) -> int | None:
    match = re.match(r"^(\d{4})", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _list_images_if_exists(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return list_images(path)
