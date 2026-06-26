from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/baselines/registry.yaml")
    parser.add_argument("--dest", default="external/repos")
    parser.add_argument("--priority", choices=["A", "B", "C"], default="A")
    parser.add_argument("--name", action="append", help="Clone a specific baseline name.")
    args = parser.parse_args()

    registry = yaml.safe_load(Path(args.registry).read_text())
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    allowed = {"A": {"A"}, "B": {"A", "B"}, "C": {"A", "B", "C"}}[args.priority]
    selected = []
    for name, item in registry["baselines"].items():
        if args.name and name not in args.name:
            continue
        if not item.get("clone", False):
            continue
        if item["priority"] not in allowed:
            continue
        selected.append((name, item))

    for name, item in selected:
        target = dest / name
        if target.exists():
            print(f"skip {name}: {target} already exists")
            continue
        run(["git", "clone", "--depth", "1", item["repo"], str(target)])


if __name__ == "__main__":
    main()

