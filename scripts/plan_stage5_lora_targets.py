from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


def candidate_modules(payload: dict[str, object]) -> list[dict[str, object]]:
    rows = payload.get("lora_candidate_modules")
    if not isinstance(rows, list):
        raise ValueError("inspect JSON must contain lora_candidate_modules")
    return [row for row in rows if isinstance(row, dict) and bool(row.get("lora_supported", False))]


def block_index(name: str) -> int | None:
    match = re.match(r"^blocks\.(\d+)\.", name)
    if not match:
        return None
    return int(match.group(1))


def escaped_exact_pattern(names: list[str]) -> str:
    if not names:
        return r"(?!)"
    return "^(" + "|".join(re.escape(name) for name in names) + ")$"


def filter_modules(
    modules: list[dict[str, object]],
    *,
    preset: str,
    last_blocks: int,
) -> tuple[list[dict[str, object]], str]:
    modules = [row for row in modules if bool(row.get("lora_supported", True))]
    if preset == "denoiser_tail":
        block_indices = [block_index(str(row.get("name", ""))) for row in modules]
        valid_indices = [index for index in block_indices if index is not None]
        if not valid_indices:
            return [], "last denoiser blocks only; no blocks.* modules found"
        start = max(valid_indices) - max(int(last_blocks), 1) + 1
        rows = [
            row
            for row in modules
            if (idx := block_index(str(row.get("name", "")))) is not None and idx >= start
        ]
        return rows, f"last {last_blocks} denoiser blocks; avoids y_embedder condition codec"
    if preset == "denoiser_all":
        return [
            row for row in modules if str(row.get("name", "")).startswith("blocks.")
        ], "all denoiser blocks; largest no-y_embedder LoRA candidate"
    if preset == "denoiser_attention":
        return [
            row
            for row in modules
            if str(row.get("name", "")).startswith("blocks.")
            and any(token in str(row.get("name", "")) for token in (".ca.1", ".q", ".k", ".v", ".proj_out"))
        ], "attention / channel-attention-like denoiser modules"
    if preset == "denoiser_mlp":
        return [
            row
            for row in modules
            if str(row.get("name", "")).startswith("blocks.")
            and any(token in str(row.get("name", "")) for token in (".conv4", ".conv5"))
        ], "denoiser feed-forward / MLP-like conv modules"
    if preset == "dec_net":
        return [
            row for row in modules if str(row.get("name", "")).startswith("dec_net.")
        ], "small dec_net linear modules; cheap stability probe"
    if preset == "embedder_small":
        prefixes = ("x_embedder", "s_embedder", "t_embedder", "final_layer")
        return [
            row for row in modules if str(row.get("name", "")).startswith(prefixes)
        ], "small non-y_embedder conditioning/time/output modules"
    if preset == "y_decoder":
        return [
            row for row in modules if str(row.get("name", "")).startswith("y_embedder.decoder")
        ], "condition-codec decoder only; use after denoiser-only probes"
    raise ValueError(f"unknown preset: {preset}")


def summarize_setting(
    rows: list[dict[str, object]],
    *,
    preset: str,
    rationale: str,
    rank: int,
) -> dict[str, object]:
    names = [str(row["name"]) for row in rows]
    return {
        "preset": preset,
        "rank": int(rank),
        "module_count": len(rows),
        "module_names": names,
        "regex": escaped_exact_pattern(names),
        "target_types": sorted({str(row.get("type", "")).lower() for row in rows if row.get("type")}),
        "lora_param_count": int(sum(int(row.get("lora_param_count", 0)) for row in rows)),
        "base_param_count": int(sum(int(row.get("base_param_count", 0)) for row in rows)),
        "rationale": rationale,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan Stage 5 LoRA target regexes from inspect_cod_lite_backbone_params.py JSON."
    )
    parser.add_argument("--inspect-json", required=True)
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        choices=(
            "denoiser_tail",
            "denoiser_all",
            "denoiser_attention",
            "denoiser_mlp",
            "dec_net",
            "embedder_small",
            "y_decoder",
        ),
    )
    parser.add_argument("--last-blocks", type=int, default=6)
    parser.add_argument("--rank", type=int, default=0, help="Override rank metadata; 0 uses inspect JSON lora_rank.")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = read_json(Path(args.inspect_json))
    rank = int(args.rank or payload.get("lora_rank", 4))
    if rank <= 0:
        raise ValueError("rank must be positive")
    modules = candidate_modules(payload)
    presets = args.preset or ["denoiser_tail", "dec_net", "denoiser_mlp", "denoiser_all"]
    settings: list[dict[str, object]] = []
    for preset in presets:
        rows, rationale = filter_modules(modules, preset=preset, last_blocks=args.last_blocks)
        settings.append(summarize_setting(rows, preset=preset, rationale=rationale, rank=rank))
    output = {
        "source": args.inspect_json,
        "checkpoint": payload.get("checkpoint", ""),
        "config": payload.get("config", ""),
        "rank": rank,
        "last_blocks": int(args.last_blocks),
        "settings": settings,
        "policy": (
            "Prefer denoiser-only LoRA first. Treat y_embedder condition-codec LoRA as a later, riskier ablation."
        ),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
