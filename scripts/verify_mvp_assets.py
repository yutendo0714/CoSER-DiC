from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml
from huggingface_hub import list_repo_files


def git_head(repo: str) -> str:
    out = subprocess.check_output(["git", "ls-remote", repo, "HEAD"], text=True)
    return out.split()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/baselines/registry.yaml")
    parser.add_argument("--pretrained", default="configs/paths/pretrained.yaml")
    args = parser.parse_args()

    registry = yaml.safe_load(Path(args.registry).read_text())
    pretrained = yaml.safe_load(Path(args.pretrained).read_text())

    print("MVP Priority A repositories")
    for name, item in registry["baselines"].items():
        if item.get("priority") != "A":
            continue
        repo = item.get("repo")
        if not repo:
            continue
        try:
            print(f"{name:24s} {git_head(repo)} {repo}")
        except Exception as exc:
            print(f"{name:24s} ERROR {type(exc).__name__}: {exc}")

    print("\nCoD-Lite Hugging Face assets")
    cod = pretrained["cod_lite"]
    files = set(list_repo_files(cod["hf_repo"]))
    wanted = [cod["diffusion_pretrain"]["filename"]]
    for item in cod["baseline_checkpoints"].values():
        wanted.append(item["filename"])
        wanted.append(item["config_filename"])
    for filename in wanted:
        print(f"{filename:32s} {'OK' if filename in files else 'MISSING'}")

    print("\nLocal pretrained paths")
    for key, item in pretrained.items():
        if key == "root":
            continue
        local_dir = Path(item["local_dir"])
        print(f"{key:16s} exists={local_dir.exists()} path={local_dir}")


if __name__ == "__main__":
    main()

