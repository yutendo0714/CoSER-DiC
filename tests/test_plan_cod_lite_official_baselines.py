from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from scripts.plan_cod_lite_official_baselines import build_plan, normalize_bpp_key, read_manifest


def _write_manifest(path: Path) -> None:
    rows = [
        {"source_path": "/dpl/kodak/kodim01.png", "reference": "ref/kodak.png"},
        {"source_path": "/dpl/clic/professional/test/a.png", "reference": "ref/clic_a.png"},
        {"source_path": "/dpl/clic/mobile/test/b.png", "reference": "ref/clic_b.png"},
        {"source_path": "/dpl/div2k/0801.png", "reference": "ref/div2k.png"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _args(tmp_path: Path, manifest: Path) -> Namespace:
    return Namespace(
        manifest=str(manifest),
        codec="cod_lite",
        codec_tag="",
        codec_name="",
        bpp=["0_0156"],
        scope=["aggregate", "splits"],
        run_prefix="test_codlite",
        output_dir=str(tmp_path / "evals"),
        curve_output_dir=str(tmp_path / "curves"),
        split_manifest_dir=str(tmp_path / "manifests"),
        cod_lite_repo="external/repos/GenCodec/CoD_Lite",
        official_repo="",
        official_module="",
        limit=0,
        skip_codec=False,
        output_json=str(tmp_path / "plan.json"),
        output_sh="",
    )


def test_normalize_bpp_key_accepts_dot_or_underscore() -> None:
    assert normalize_bpp_key("0.0156") == "0_0156"
    assert normalize_bpp_key("bpp_0_0312") == "0_0312"


def test_build_plan_writes_split_manifests_and_patch_specific_commands(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)

    payload = build_plan(_args(tmp_path, manifest))

    split_manifests = payload["split_manifests"]
    assert set(split_manifests) == {"clic2020_test", "div2k_val", "full552", "kodak"}
    assert len(read_manifest(Path(split_manifests["full552"]))) == 4
    assert len(read_manifest(Path(split_manifests["clic2020_test"]))) == 2

    settings = payload["settings"]
    aggregate = next(row for row in settings if row["kind"] == "aggregate")
    kodak = next(row for row in settings if row["split"] == "kodak")
    clic = next(row for row in settings if row["split"] == "clic2020_test")

    assert aggregate["fid_patch_size"] == 0
    assert "--fid-patch-size" in aggregate["command"]
    assert aggregate["command"][aggregate["command"].index("--fid-patch-size") + 1] == "0"
    assert kodak["fid_patch_size"] == 64
    assert clic["fid_patch_size"] == 256

    curve_commands = payload["curve_commands"]
    full_curve = next(row for row in curve_commands if row["split"] == "full552")
    kodak_curve = next(row for row in curve_commands if row["split"] == "kodak")
    assert "--bpp-policy" in full_curve["command"]
    assert "--metric" in kodak_curve["command"]
    assert "fid" in kodak_curve["command"]
    assert "fid" not in full_curve["command"]


def test_build_plan_supports_cod_one_step_preset(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    args = _args(tmp_path, manifest)
    _write_manifest(manifest)
    args.codec = "cod_one_step"
    args.codec_tag = ""
    args.codec_name = ""
    args.official_repo = ""
    args.official_module = ""
    args.cod_lite_repo = ""
    args.bpp = ["0_0312"]
    args.scope = ["aggregate"]

    payload = build_plan(args)

    row = payload["settings"][0]
    assert payload["codec"] == "cod_one_step"
    assert payload["official_module"] == "downstream.finetuned_one_step_cod"
    assert row["run_name"] == "test_codlite_cod_one_step_0312_cod512_full552_nofid"
    assert "external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0312.pt" in row["command"]
    assert "--codec-name" in row["command"]
    assert row["command"][row["command"].index("--codec-name") + 1] == "CoD one-step"


def test_cli_writes_plan_and_shell(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "plan.json"
    shell = tmp_path / "plan.sh"
    _write_manifest(manifest)

    subprocess.run(
        [
            sys.executable,
            "scripts/plan_cod_lite_official_baselines.py",
            "--manifest",
            str(manifest),
            "--bpp",
            "0_0156",
            "--scope",
            "aggregate",
            "--run-prefix",
            "test_codlite",
            "--output-dir",
            str(tmp_path / "evals"),
            "--curve-output-dir",
            str(tmp_path / "curves"),
            "--output-json",
            str(output),
            "--output-sh",
            str(shell),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    assert len(payload["settings"]) == 1
    assert payload["settings"][0]["run_name"] == "test_codlite_cod_lite_0156_cod512_full552_nofid"
    script_text = shell.read_text()
    assert "check_gpu_ready.py" in script_text
    assert "eval_cod_lite_official_baseline.py" in script_text
    assert "collect_bd_curve_points.py" in script_text
