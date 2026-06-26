from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, list_repo_files


DEFAULT_KEYS = ["diffusion_pretrain", "bpp_0_0039", "bpp_0_0078", "bpp_0_0156", "bpp_0_0312"]


def load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def resolve_files(cfg: dict, keys: list[str], include_yaml: bool) -> list[str]:
    cod = cfg["cod_lite"]
    selected: list[str] = []
    for key in keys:
        if key == "diffusion_pretrain":
            selected.append(cod["diffusion_pretrain"]["filename"])
            continue
        item = cod["baseline_checkpoints"][key]
        selected.append(item["filename"])
        if include_yaml:
            selected.append(item["config_filename"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths/pretrained.yaml")
    parser.add_argument("--all", action="store_true", help="Download pretrain and MVP baseline checkpoints.")
    parser.add_argument("--pretrain-only", action="store_true", help="Download only CoD_Lite_pretrain.pt.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-yaml", action="store_true", help="Also download CoD-Lite YAML files.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    repo = cfg["cod_lite"]["hf_repo"]
    local_dir = Path(cfg["cod_lite"]["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    available = set(list_repo_files(repo))
    keys = ["diffusion_pretrain"] if args.pretrain_only else DEFAULT_KEYS
    if not args.all and not args.pretrain_only:
        keys = ["diffusion_pretrain"]

    filenames = resolve_files(cfg, keys, include_yaml=args.include_yaml)
    for filename in filenames:
        if filename not in available:
            raise FileNotFoundError(f"{filename} not found in {repo}")
        target = local_dir / filename
        if args.dry_run:
            print(f"would download {repo}:{filename} -> {target}")
            continue
        path = hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
        print(f"downloaded {filename} -> {path}")


if __name__ == "__main__":
    main()

