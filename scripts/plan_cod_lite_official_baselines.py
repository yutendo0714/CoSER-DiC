from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

try:
    from summarize_per_image_metrics import grouped_dataset_key
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts.summarize_per_image_metrics import grouped_dataset_key


PYTHON = ".venv/bin/python"


def gpu_preflight_command(python: str = PYTHON) -> str:
    return shlex.join([python, "scripts/check_gpu_ready.py", "--min-devices", "1"])


BPP_SETTINGS = {
    "0_0039": {
        "label": "0039",
        "nominal_bpp": 0.00390625,
        "checkpoint": "external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0039.pt",
        "config": "external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0039.yaml",
    },
    "0_0078": {
        "label": "0078",
        "nominal_bpp": 0.0078125,
        "checkpoint": "external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0078.pt",
        "config": "external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0078.yaml",
    },
    "0_0156": {
        "label": "0156",
        "nominal_bpp": 0.015625,
        "checkpoint": "external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt",
        "config": "external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml",
    },
    "0_0312": {
        "label": "0312",
        "nominal_bpp": 0.03125,
        "checkpoint": "external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.pt",
        "config": "external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.yaml",
    },
}

COD_ONE_STEP_BPP_SETTINGS = {
    "0_0039": {
        "label": "0039",
        "nominal_bpp": 0.00390625,
        "checkpoint": "external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.pt",
        "config": "external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.yaml",
    },
    "0_0312": {
        "label": "0312",
        "nominal_bpp": 0.03125,
        "checkpoint": "external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0312.pt",
        "config": "external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0312.yaml",
    },
    "0_1250": {
        "label": "1250",
        "nominal_bpp": 0.125,
        "checkpoint": "external/pretrained/CoD/finetuned_one_step_cod/bpp_0_1250.pt",
        "config": "external/pretrained/CoD/finetuned_one_step_cod/bpp_0_1250.yaml",
    },
}

CODEC_PRESETS = {
    "cod_lite": {
        "tag": "cod_lite",
        "display_name": "CoD-Lite",
        "official_repo": "external/repos/GenCodec/CoD_Lite",
        "official_module": "finetuned_one_step_codec.inference",
        "bpp_settings": BPP_SETTINGS,
    },
    "cod_one_step": {
        "tag": "cod_one_step",
        "display_name": "CoD one-step",
        "official_repo": "external/repos/GenCodec/CoD",
        "official_module": "downstream.finetuned_one_step_cod",
        "bpp_settings": COD_ONE_STEP_BPP_SETTINGS,
    },
}

SPLIT_PATCH_SETTINGS = {
    "kodak": {"fid_patch_size": 64, "fid_patch_num": 2},
    "clic2020_test": {"fid_patch_size": 256, "fid_patch_num": 2},
    "div2k_val": {"fid_patch_size": 256, "fid_patch_num": 2},
}


def normalize_bpp_key(value: str, settings: dict[str, dict[str, object]] | None = None) -> str:
    settings = settings or BPP_SETTINGS
    key = value.strip().replace(".", "_")
    if key in settings:
        return key
    if key.startswith("bpp_"):
        key = key.removeprefix("bpp_")
    if key in settings:
        return key
    raise ValueError(f"unknown CoD-Lite bpp key: {value}")


def read_manifest(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"manifest must contain JSON objects: {path}")
    return rows


def write_split_manifests(rows: list[dict[str, object]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, object]]] = {"full552": list(rows)}
    for row in rows:
        source = str(row.get("source_path", row.get("path", "")))
        key = grouped_dataset_key(source, "cod_reproduction_512")
        if key in SPLIT_PATCH_SETTINGS:
            groups.setdefault(key, []).append(row)
    paths: dict[str, Path] = {}
    for key, group_rows in groups.items():
        path = output_dir / f"{key}.manifest.jsonl"
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in group_rows) + "\n")
        paths[key] = path
    return paths


def build_eval_command(
    *,
    checkpoint: str,
    config: str,
    manifest: Path,
    run_name: str,
    output_dir: str,
    fid_patch_size: int,
    fid_patch_num: int,
    limit: int,
    cod_lite_repo: str,
    official_module: str,
    codec_name: str,
    skip_codec: bool,
) -> list[str]:
    command = [
        PYTHON,
        "scripts/eval_cod_lite_official_baseline.py",
        "--checkpoint",
        checkpoint,
        "--config",
        config,
        "--cod-lite-repo",
        cod_lite_repo,
        "--official-module",
        official_module,
        "--codec-name",
        codec_name,
        "--manifest",
        str(manifest),
        "--run-name",
        run_name,
        "--output-dir",
        output_dir,
        "--fid-patch-size",
        str(fid_patch_size),
        "--fid-patch-num",
        str(fid_patch_num),
    ]
    if limit > 0:
        command.extend(["--limit", str(limit)])
    if skip_codec:
        command.append("--skip-codec")
    return command


def build_curve_command(
    *,
    name: str,
    dataset: str,
    output_json: str,
    summaries: list[tuple[str, Path]],
    include_fid: bool,
) -> list[str]:
    metrics = ["psnr", "ms_ssim", "lpips", "dists"]
    if include_fid:
        metrics.append("fid")
    command = [
        PYTHON,
        "scripts/collect_bd_curve_points.py",
        "--name",
        name,
        "--dataset",
        dataset,
        "--stage",
        "direct",
        "--bpp-policy",
        "actual_payload_bpp from official CoD-Lite .cod payload size, excluding the 4-byte width/height header; fixed model/codebook weights excluded",
    ]
    for metric in metrics:
        command.extend(["--metric", metric])
    for label, summary in summaries:
        command.extend(["--input", f"{label}={summary}"])
    command.extend(["--output-json", output_json])
    return command


def build_plan(args: argparse.Namespace) -> dict[str, object]:
    codec_preset = CODEC_PRESETS[args.codec]
    bpp_settings = codec_preset["bpp_settings"]
    if not isinstance(bpp_settings, dict):
        raise ValueError(f"invalid bpp settings for codec {args.codec}")
    codec_tag = args.codec_tag or str(codec_preset["tag"])
    codec_name = args.codec_name or str(codec_preset["display_name"])
    official_repo = args.official_repo or args.cod_lite_repo or str(codec_preset["official_repo"])
    official_module = args.official_module or str(codec_preset["official_module"])
    rows = read_manifest(Path(args.manifest))
    split_manifests = write_split_manifests(rows, Path(args.split_manifest_dir))
    bpp_keys = [normalize_bpp_key(value, bpp_settings) for value in args.bpp] if args.bpp else list(bpp_settings)
    scopes = set(args.scope)
    if not scopes:
        scopes = {"aggregate", "splits"}

    settings: list[dict[str, object]] = []
    curve_inputs: dict[str, list[tuple[str, Path]]] = {"full552": []}
    for split_key in SPLIT_PATCH_SETTINGS:
        curve_inputs[split_key] = []

    for bpp_key in bpp_keys:
        bpp = bpp_settings[bpp_key]
        bpp_label = str(bpp["label"])
        if "aggregate" in scopes:
            run_name = f"{args.run_prefix}_{codec_tag}_{bpp_label}_cod512_full552_nofid"
            summary = Path(args.output_dir) / run_name / "summary.json"
            command = build_eval_command(
                checkpoint=str(bpp["checkpoint"]),
                config=str(bpp["config"]),
                manifest=split_manifests["full552"],
                run_name=run_name,
                output_dir=args.output_dir,
                fid_patch_size=0,
                fid_patch_num=2,
                limit=args.limit,
                cod_lite_repo=official_repo,
                official_module=official_module,
                codec_name=codec_name,
                skip_codec=args.skip_codec,
            )
            settings.append(
                {
                    "kind": "aggregate",
                    "split": "full552",
                    "bpp_key": bpp_key,
                    "nominal_bpp": bpp["nominal_bpp"],
                    "run_name": run_name,
                    "manifest": str(split_manifests["full552"]),
                    "summary": str(summary),
                    "fid_patch_size": 0,
                    "fid_patch_num": 2,
                    "command": command,
                    "command_text": shlex.join(command),
                }
            )
            curve_inputs["full552"].append((f"{codec_tag}_{bpp_label}", summary))

        if "splits" in scopes:
            for split_key, patch in SPLIT_PATCH_SETTINGS.items():
                if split_key not in split_manifests:
                    continue
                run_name = (
                    f"{args.run_prefix}_{codec_tag}_{bpp_label}_cod512_"
                    f"{split_key}_patch{patch['fid_patch_size']}"
                )
                summary = Path(args.output_dir) / run_name / "summary.json"
                command = build_eval_command(
                    checkpoint=str(bpp["checkpoint"]),
                    config=str(bpp["config"]),
                    manifest=split_manifests[split_key],
                    run_name=run_name,
                    output_dir=args.output_dir,
                    fid_patch_size=int(patch["fid_patch_size"]),
                    fid_patch_num=int(patch["fid_patch_num"]),
                    limit=args.limit,
                    cod_lite_repo=official_repo,
                    official_module=official_module,
                    codec_name=codec_name,
                    skip_codec=args.skip_codec,
                )
                settings.append(
                    {
                        "kind": "split",
                        "split": split_key,
                        "bpp_key": bpp_key,
                        "nominal_bpp": bpp["nominal_bpp"],
                        "run_name": run_name,
                        "manifest": str(split_manifests[split_key]),
                        "summary": str(summary),
                        "fid_patch_size": patch["fid_patch_size"],
                        "fid_patch_num": patch["fid_patch_num"],
                        "command": command,
                        "command_text": shlex.join(command),
                    }
                )
                curve_inputs[split_key].append((f"{codec_tag}_{bpp_label}", summary))

    curve_commands: list[dict[str, object]] = []
    curve_root = Path(args.curve_output_dir)
    if "aggregate" in scopes and curve_inputs["full552"]:
        output_json = curve_root / f"{args.run_prefix}_{codec_tag}_cod512_full552_curve.json"
        command = build_curve_command(
            name=f"{args.run_prefix}_{codec_tag}_cod512_full552",
            dataset=f"CoD 512 reproduction full552 ({codec_name}): Kodak24 + CLIC2020 test428 + DIV2K val100",
            output_json=str(output_json),
            summaries=curve_inputs["full552"],
            include_fid=False,
        )
        curve_commands.append(
            {
                "split": "full552",
                "include_fid": False,
                "output_json": str(output_json),
                "command": command,
                "command_text": shlex.join(command),
            }
        )
    if "splits" in scopes:
        for split_key, summaries in curve_inputs.items():
            if split_key == "full552" or not summaries:
                continue
            output_json = curve_root / f"{args.run_prefix}_{codec_tag}_cod512_{split_key}_curve.json"
            command = build_curve_command(
                name=f"{args.run_prefix}_{codec_tag}_cod512_{split_key}",
                dataset=f"CoD 512 reproduction split ({codec_name}): {split_key}",
                output_json=str(output_json),
                summaries=summaries,
                include_fid=True,
            )
            curve_commands.append(
                {
                    "split": split_key,
                    "include_fid": True,
                    "output_json": str(output_json),
                    "command": command,
                    "command_text": shlex.join(command),
                }
            )

    return {
        "run_prefix": args.run_prefix,
        "codec": args.codec,
        "codec_tag": codec_tag,
        "codec_name": codec_name,
        "official_repo": official_repo,
        "official_module": official_module,
        "manifest": args.manifest,
        "split_manifest_dir": args.split_manifest_dir,
        "split_manifests": {key: str(path) for key, path in split_manifests.items()},
        "policy": {
            "aggregate": "full552 aggregate uses fid_patch_size=0; use it for PSNR/MS-SSIM/LPIPS/DISTS curves.",
            "splits": "FID is protocol-specific per split: Kodak patch64, CLIC2020/DIV2K patch256, fid_patch_num=2.",
            "bpp": f"actual_payload_bpp from official {codec_name} .cod payload bytes excluding the 4-byte width/height header.",
        },
        "settings": settings,
        "curve_commands": curve_commands,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan official CoD-Lite baseline reproduction commands.")
    parser.add_argument("--codec", choices=tuple(CODEC_PRESETS), default="cod_lite")
    parser.add_argument("--codec-tag", default="")
    parser.add_argument("--codec-name", default="")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bpp", action="append", default=[])
    parser.add_argument("--scope", action="append", choices=("aggregate", "splits"), default=[])
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output-dir", default="results/baselines/cod_lite_official")
    parser.add_argument("--curve-output-dir", default="results/baselines/cod_lite_official/curves")
    parser.add_argument("--split-manifest-dir", default="")
    parser.add_argument("--cod-lite-repo", default="")
    parser.add_argument("--official-repo", default="")
    parser.add_argument("--official-module", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-codec", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-sh", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.split_manifest_dir:
        args.split_manifest_dir = str(Path(args.output_dir) / "manifests" / args.run_prefix)
    payload = build_plan(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_sh:
        sh_path = Path(args.output_sh)
        sh_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "cd /workspace/CoSER-DiC",
            "",
            gpu_preflight_command(),
            "",
        ]
        for row in payload["settings"]:
            lines.append(str(row["command_text"]))
        lines.append("")
        for row in payload["curve_commands"]:
            lines.append(str(row["command_text"]))
        sh_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
