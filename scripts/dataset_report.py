from __future__ import annotations

import argparse
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/paths.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    for name, item in sorted(cfg["datasets"].items()):
        path = Path(item["path"])
        print(f"{name:24s} exists={path.exists()} images={count_images(path):7d} path={path}")


if __name__ == "__main__":
    main()

