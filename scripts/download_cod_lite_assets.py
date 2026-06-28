from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, list_repo_files


COD_LITE_DEFAULT_KEYS = ["diffusion_pretrain", "bpp_0_0039", "bpp_0_0078", "bpp_0_0156", "bpp_0_0312"]
COD_DEFAULT_KEYS = ["pixel_vpred", "one_step_bpp_0_0039"]


def load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def resolve_cod_lite_files(cfg: dict, keys: list[str], include_yaml: bool) -> list[str]:
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


def resolve_cod_files(cfg: dict, keys: list[str], include_yaml: bool) -> list[str]:
    cod = cfg["cod"]
    selected: list[str] = []
    for key in keys:
        if key in cod["backbone_checkpoints"]:
            item = cod["backbone_checkpoints"][key]
        elif key.startswith("one_step_"):
            item = cod["one_step_checkpoints"][key.removeprefix("one_step_")]
        else:
            raise KeyError(f"unknown CoD asset key: {key}")
        selected.append(item["filename"])
        if include_yaml:
            selected.append(item["config_filename"])
    return selected


def download_files(*, repo: str, local_dir: Path, filenames: list[str], dry_run: bool) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    available = set(list_repo_files(repo))
    for filename in filenames:
        if filename not in available:
            raise FileNotFoundError(f"{filename} not found in {repo}")
        target = local_dir / filename
        if dry_run:
            print(f"would download {repo}:{filename} -> {target}")
            continue
        path = hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
        print(f"downloaded {filename} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths/pretrained.yaml")
    parser.add_argument("--family", choices=("cod_lite", "cod", "both"), default="cod_lite")
    parser.add_argument("--all", action="store_true", help="Download pretrain and MVP baseline checkpoints.")
    parser.add_argument("--pretrain-only", action="store_true", help="Download only CoD_Lite_pretrain.pt.")
    parser.add_argument(
        "--cod-key",
        action="append",
        default=None,
        help="CoD asset key, e.g. pixel_vpred, latent_vpred, latent_vpred_64bits, one_step_bpp_0_0039.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-yaml", action="store_true", help="Also download CoD-Lite YAML files.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.family in {"cod_lite", "both"}:
        repo = cfg["cod_lite"]["hf_repo"]
        local_dir = Path(cfg["cod_lite"]["local_dir"])
        keys = ["diffusion_pretrain"] if args.pretrain_only else COD_LITE_DEFAULT_KEYS
        if not args.all and not args.pretrain_only:
            keys = ["diffusion_pretrain"]
        filenames = resolve_cod_lite_files(cfg, keys, include_yaml=args.include_yaml)
        download_files(repo=repo, local_dir=local_dir, filenames=filenames, dry_run=args.dry_run)

    if args.family in {"cod", "both"}:
        repo = cfg["cod"]["hf_repo"]
        local_dir = Path(cfg["cod"]["local_dir"])
        keys = args.cod_key or (COD_DEFAULT_KEYS if args.all else ["pixel_vpred"])
        filenames = resolve_cod_files(cfg, keys, include_yaml=args.include_yaml)
        download_files(repo=repo, local_dir=local_dir, filenames=filenames, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
