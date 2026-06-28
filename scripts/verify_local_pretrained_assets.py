from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def is_required(item: dict[str, Any]) -> bool:
    return bool(
        item.get("required_for_mvp_stage4")
        or item.get("required_for_mvp_baseline")
        or item.get("required_for_mvp_parallel_track")
    )


def iter_expected_assets(cfg: dict[str, Any]) -> list[tuple[str, Path, int | None, bool]]:
    assets: list[tuple[str, Path, int | None, bool]] = []

    cod_lite = cfg.get("cod_lite", {})
    local_dir = Path(cod_lite.get("local_dir", "external/pretrained/CoD_Lite"))
    pretrain = cod_lite.get("diffusion_pretrain", {})
    if pretrain:
        assets.append(("cod_lite:diffusion_pretrain", Path(pretrain["path"]), None, is_required(pretrain)))
    for key, item in cod_lite.get("baseline_checkpoints", {}).items():
        required = is_required(item)
        assets.append((f"cod_lite:{key}:checkpoint", Path(item["path"]), item.get("size_bytes"), required))
        config_filename = item.get("config_filename")
        if config_filename:
            assets.append((f"cod_lite:{key}:config", local_dir / config_filename, None, required))

    cod = cfg.get("cod", {})
    local_dir = Path(cod.get("local_dir", "external/pretrained/CoD"))
    for group in ("backbone_checkpoints", "one_step_checkpoints"):
        for key, item in cod.get(group, {}).items():
            required = is_required(item)
            assets.append((f"cod:{key}:checkpoint", Path(item["path"]), item.get("size_bytes"), required))
            config_filename = item.get("config_filename")
            if config_filename:
                assets.append((f"cod:{key}:config", local_dir / config_filename, None, required))

    rdvq = cfg.get("rdvq", {})
    for key, item in rdvq.get("checkpoints", {}).items():
        assets.append((f"rdvq:{key}:checkpoint", Path(item["path"]), item.get("size_bytes"), is_required(item)))

    return assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths/pretrained.yaml")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any required asset is missing.")
    parser.add_argument("--required-only", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    missing = 0
    mismatched = 0
    for name, path, expected_size, required in iter_expected_assets(cfg):
        if args.required_only and not required:
            continue
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        if not exists:
            status = "MISSING"
            if required:
                missing += 1
        elif expected_size is not None and size != int(expected_size):
            status = f"SIZE_MISMATCH expected={expected_size}"
            if required:
                mismatched += 1
        else:
            status = "OK"
        required_label = "required" if required else "optional"
        print(f"{status:32s} {required_label:8s} {name:36s} size={size:>12} path={path}")
    if args.strict and (missing or mismatched):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
