from __future__ import annotations

import json
from pathlib import Path

from scripts.promote_stage5_control_candidates import (
    find_plan_row,
    gpu_preflight_command,
    load_plan_rows,
    promote_command,
    selected_candidates,
)


def test_promote_command_updates_limit_run_name_and_recon_flags() -> None:
    command = [
        "python",
        "scripts/eval_stage4_cod_lite_adapter.py",
        "--run-name",
        "candidate_limit64",
        "--limit",
        "64",
        "--batch-size",
        "4",
    ]

    run_name, promoted = promote_command(
        command,
        run_suffix="_full552",
        limit=0,
        batch_size=2,
        num_workers=8,
        output_dir="results/full552",
        save_reconstructions=True,
        save_reconstruction_limit=12,
        save_reconstruction_kinds="reference,stage4,quad",
        extra_args=["--wandb-mode", "offline"],
    )

    assert run_name == "candidate_limit64_full552"
    assert promoted[promoted.index("--run-name") + 1] == "candidate_limit64_full552"
    assert promoted[promoted.index("--limit") + 1] == "0"
    assert promoted[promoted.index("--batch-size") + 1] == "2"
    assert promoted[promoted.index("--num-workers") + 1] == "8"
    assert "--save-reconstructions" in promoted
    assert promoted[-2:] == ["--wandb-mode", "offline"]


def test_gpu_preflight_command_uses_project_checker() -> None:
    assert gpu_preflight_command("python") == "python scripts/check_gpu_ready.py --min-devices 1"


def test_load_plan_rows_and_match_selection(tmp_path: Path) -> None:
    summary = "results/stage4_cod_lite_adapter_eval/run_a/summary.json"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "settings": [
                    {
                        "run_name": "run_a",
                        "summary": summary,
                        "command": [
                            "python",
                            "scripts/eval_stage4_cod_lite_adapter.py",
                            "--run-name",
                            "run_a",
                            "--limit",
                            "64",
                        ],
                    }
                ]
            }
        )
    )

    lookup = load_plan_rows([plan])
    row = find_plan_row({"label": "run_a", "summary_path": summary}, lookup)

    assert row is not None
    assert row["run_name"] == "run_a"


def test_selected_candidates_filters_guard_and_pareto() -> None:
    selection = {
        "recommended": [
            {"label": "bad_guard", "stage3_guard_pass": False, "pareto": True},
            {"label": "bad_basis", "stage3_guard_pass": True, "basis_guard_pass": False, "pareto": True},
            {"label": "bad_pareto", "stage3_guard_pass": True, "pareto": False},
            {"label": "good", "stage3_guard_pass": True, "basis_guard_pass": True, "pareto": True},
        ]
    }

    rows = selected_candidates(selection, top_k=4, require_guard=True, require_pareto=True)

    assert [row["label"] for row in rows] == ["good"]
