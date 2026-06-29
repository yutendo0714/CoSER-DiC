from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.eval_cod_lite_official_baseline import run_official_cli


def test_run_official_cli_uses_configurable_module_and_repo(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(cmd, cwd, env, check):  # noqa: ANN001
        calls.append({"cmd": cmd, "cwd": cwd, "env": env, "check": check})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    repo = tmp_path / "repo"
    repo.mkdir()
    ckpt = tmp_path / "model.pt"
    cfg = tmp_path / "model.yaml"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    ckpt.write_text("")
    cfg.write_text("")

    run_official_cli(
        repo=repo,
        module="downstream.finetuned_one_step_cod",
        checkpoint=ckpt,
        config=cfg,
        mode="compress",
        input_dir=input_dir,
        output_dir=output_dir,
    )

    call = calls[0]
    assert call["cwd"] == repo
    assert call["check"] is True
    assert call["cmd"][2] == "downstream.finetuned_one_step_cod"
    assert "--ckpt" in call["cmd"]
    assert str(ckpt.resolve()) in call["cmd"]
    assert str(repo) in call["env"]["PYTHONPATH"]
