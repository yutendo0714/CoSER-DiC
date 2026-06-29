from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


PYTHON = ".venv/bin/python"


def gpu_preflight_command(python: str = PYTHON) -> str:
    return shlex.join([python, "scripts/check_gpu_ready.py", "--min-devices", "1"])


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


def normalize_path_key(value: object) -> str:
    return Path(str(value)).as_posix()


def command_arg(command: list[str], flag: str) -> str:
    if flag not in command:
        return ""
    index = command.index(flag)
    if index + 1 >= len(command):
        return ""
    return command[index + 1]


def set_command_arg(command: list[str], flag: str, value: str) -> list[str]:
    updated = list(command)
    if flag in updated:
        index = updated.index(flag)
        if index + 1 >= len(updated):
            raise ValueError(f"{flag} has no value")
        updated[index + 1] = value
    else:
        updated.extend([flag, value])
    return updated


def ensure_flag(command: list[str], flag: str) -> list[str]:
    if flag in command:
        return list(command)
    return [*command, flag]


def append_extra_args(command: list[str], extra_args: list[str]) -> list[str]:
    if not extra_args:
        return list(command)
    return [*command, *extra_args]


def load_plan_rows(paths: list[Path]) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for path in paths:
        payload = read_json(path)
        settings = payload.get("settings")
        if not isinstance(settings, list):
            raise ValueError(f"sweep plan has no settings list: {path}")
        for row in settings:
            if not isinstance(row, dict):
                continue
            command = row.get("command")
            if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                continue
            keys = []
            summary = row.get("summary")
            if isinstance(summary, str):
                keys.append(normalize_path_key(summary))
            run_name = row.get("run_name")
            if isinstance(run_name, str):
                keys.append(run_name)
            command_run_name = command_arg(command, "--run-name")
            if command_run_name:
                keys.append(command_run_name)
            for key in keys:
                lookup[key] = row
    return lookup


def find_plan_row(candidate: dict[str, object], lookup: dict[str, dict[str, object]]) -> dict[str, object] | None:
    keys = []
    summary_path = candidate.get("summary_path")
    if isinstance(summary_path, str):
        keys.append(normalize_path_key(summary_path))
    label = candidate.get("label")
    if isinstance(label, str):
        keys.append(label)
    for key in keys:
        if key in lookup:
            return lookup[key]
    for key in keys:
        for lookup_key, row in lookup.items():
            if key and (lookup_key.endswith(key) or key.endswith(lookup_key)):
                return row
    return None


def selected_candidates(
    selection: dict[str, object],
    *,
    top_k: int,
    require_guard: bool,
    require_pareto: bool,
) -> list[dict[str, object]]:
    raw = selection.get("recommended", selection.get("candidates", []))
    if not isinstance(raw, list):
        raise ValueError("selection JSON must contain recommended or candidates list")
    rows: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if require_guard and item.get("stage3_guard_pass") is False:
            continue
        if require_guard and item.get("basis_guard_pass") is False:
            continue
        if require_guard and item.get("promotion_guard_pass") is False:
            continue
        if require_pareto and item.get("pareto") is False:
            continue
        rows.append(item)
        if top_k > 0 and len(rows) >= top_k:
            break
    return rows


def promote_command(
    command: list[str],
    *,
    run_suffix: str,
    limit: int,
    batch_size: int,
    num_workers: int,
    output_dir: str,
    save_reconstructions: bool,
    save_reconstruction_limit: int,
    save_reconstruction_kinds: str,
    extra_args: list[str],
) -> tuple[str, list[str]]:
    promoted = list(command)
    old_run_name = command_arg(promoted, "--run-name")
    if not old_run_name:
        raise ValueError("candidate command has no --run-name")
    new_run_name = old_run_name if old_run_name.endswith(run_suffix) else f"{old_run_name}{run_suffix}"
    promoted = set_command_arg(promoted, "--run-name", new_run_name)
    promoted = set_command_arg(promoted, "--limit", str(limit))
    if batch_size > 0:
        promoted = set_command_arg(promoted, "--batch-size", str(batch_size))
    if num_workers >= 0:
        promoted = set_command_arg(promoted, "--num-workers", str(num_workers))
    if output_dir:
        promoted = set_command_arg(promoted, "--output-dir", output_dir)
    if save_reconstructions:
        promoted = ensure_flag(promoted, "--save-reconstructions")
        promoted = set_command_arg(promoted, "--save-reconstruction-limit", str(save_reconstruction_limit))
        promoted = set_command_arg(promoted, "--save-reconstruction-kinds", save_reconstruction_kinds)
    promoted = append_extra_args(promoted, extra_args)
    return new_run_name, promoted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--sweep-plan", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--run-suffix", default="_full552")
    parser.add_argument("--limit", type=int, default=0, help="0 means full eval for eval_stage4_cod_lite_adapter.py")
    parser.add_argument("--batch-size", type=int, default=0, help="<=0 keeps original")
    parser.add_argument("--num-workers", type=int, default=-1, help="<0 keeps original")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--include-unguarded", action="store_true")
    parser.add_argument("--include-non-pareto", action="store_true")
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument("--save-reconstruction-limit", type=int, default=24)
    parser.add_argument("--save-reconstruction-kinds", default="reference,stage3,stage4,quad")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-sh", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = read_json(Path(args.selection))
    plan_lookup = load_plan_rows([Path(path) for path in args.sweep_plan])
    candidates = selected_candidates(
        selection,
        top_k=args.top_k,
        require_guard=not args.include_unguarded,
        require_pareto=not args.include_non_pareto,
    )
    rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for candidate in candidates:
        plan_row = find_plan_row(candidate, plan_lookup)
        if plan_row is None:
            missing.append({"label": candidate.get("label", ""), "summary_path": candidate.get("summary_path", "")})
            continue
        command = plan_row.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError(f"matched plan row has no string command for {candidate.get('label', '')}")
        run_name, promoted = promote_command(
            command,
            run_suffix=args.run_suffix,
            limit=args.limit,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            save_reconstructions=args.save_reconstructions,
            save_reconstruction_limit=args.save_reconstruction_limit,
            save_reconstruction_kinds=args.save_reconstruction_kinds,
            extra_args=args.extra_arg,
        )
        rows.append(
            {
                "source_label": candidate.get("label", ""),
                "source_summary": candidate.get("summary_path", ""),
                "promoted_run_name": run_name,
                "stage3_guard_pass": candidate.get("stage3_guard_pass", None),
                "pareto": candidate.get("pareto", None),
                "actual_payload_bpp": candidate.get("actual_payload_bpp", None),
                "lpips_gap_vs_reference_at_bpp": candidate.get("lpips_gap_vs_reference_at_bpp", None),
                "dists_gap_vs_reference_at_bpp": candidate.get("dists_gap_vs_reference_at_bpp", None),
                "command": promoted,
                "command_text": shlex.join(promoted),
            }
        )

    payload = {
        "selection": args.selection,
        "sweep_plans": args.sweep_plan,
        "promotion_policy": {
            "require_guard": not args.include_unguarded,
            "require_pareto": not args.include_non_pareto,
            "limit": args.limit,
            "run_suffix": args.run_suffix,
        },
        "commands": rows,
        "missing_matches": missing,
    }
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
        lines.extend(row["command_text"] for row in rows)
        sh_path.write_text("\n".join(str(line) for line in lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
