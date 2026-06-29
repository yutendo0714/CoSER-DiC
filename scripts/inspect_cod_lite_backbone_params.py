from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from torch import nn


def compile_patterns(items: list[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for item in items:
        for raw_pattern in item.split(","):
            pattern = raw_pattern.strip()
            if pattern:
                patterns.append(re.compile(pattern))
    return patterns


def prepend_sys_path(path: Path) -> None:
    path_str = str(path.resolve())
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def choose_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; use --device cpu for CPU-only inspection.")
    if value not in {"cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu, or cuda")
    return value


def load_cod_lite_net(*, repo_root: Path, checkpoint_path: Path, config_path: Path, device: str) -> nn.Module:
    prepend_sys_path(repo_root)
    import yaml
    from omegaconf import OmegaConf
    from cod.utils.test_utils import instantiate_class, load_model

    with config_path.open("r") as handle:
        config = OmegaConf.create(yaml.safe_load(handle))
    net = instantiate_class(config.model.net).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    net = load_model(ckpt, net, prefix="net.")
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def direct_parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters(recurse=False)))


def lora_param_count(module: nn.Module, *, rank: int) -> int:
    if isinstance(module, nn.Linear):
        return int(rank * (module.in_features + module.out_features))
    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            return 0
        kernel_h, kernel_w = module.kernel_size
        return int(rank * (module.in_channels * kernel_h * kernel_w + module.out_channels))
    return 0


def module_row(name: str, module: nn.Module, *, lora_rank: int) -> dict[str, object] | None:
    if isinstance(module, nn.Linear):
        return {
            "name": name,
            "type": "Linear",
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "base_param_count": direct_parameter_count(module),
            "lora_supported": True,
            "lora_rank": int(lora_rank),
            "lora_param_count": lora_param_count(module, rank=lora_rank),
        }
    if isinstance(module, nn.Conv2d):
        return {
            "name": name,
            "type": "Conv2d",
            "in_channels": int(module.in_channels),
            "out_channels": int(module.out_channels),
            "kernel_size": list(module.kernel_size),
            "stride": list(module.stride),
            "groups": int(module.groups),
            "base_param_count": direct_parameter_count(module),
            "lora_supported": module.groups == 1,
            "lora_rank": int(lora_rank),
            "lora_param_count": lora_param_count(module, rank=lora_rank),
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect official CoD-Lite backbone parameter names after CoSER wrapper/eval cache conversion."
    )
    parser.add_argument("--cod-lite-repo", default="external/repos/GenCodec/CoD_Lite")
    parser.add_argument("--cod-lite-checkpoint", required=True)
    parser.add_argument("--cod-lite-config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if args.lora_rank <= 0:
        raise ValueError("--lora-rank must be positive")
    device = choose_device(args.device)

    net = load_cod_lite_net(
        repo_root=Path(args.cod_lite_repo),
        checkpoint_path=Path(args.cod_lite_checkpoint),
        config_path=Path(args.cod_lite_config),
        device=device,
    )

    patterns = compile_patterns(args.pattern)
    rows: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    for name, parameter in net.named_parameters():
        row = {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "requires_grad": bool(parameter.requires_grad),
        }
        rows.append(row)
        if patterns and any(pattern.search(name) for pattern in patterns):
            matched_rows.append(row)

    module_rows: list[dict[str, object]] = []
    matched_module_rows: list[dict[str, object]] = []
    for name, module in net.named_modules():
        row = module_row(name, module, lora_rank=args.lora_rank)
        if row is None:
            continue
        module_rows.append(row)
        if patterns and any(pattern.search(name) for pattern in patterns):
            matched_module_rows.append(row)

    summary = {
        "checkpoint": args.cod_lite_checkpoint,
        "config": args.cod_lite_config,
        "device": device,
        "total_param_count": int(sum(int(row["numel"]) for row in rows)),
        "total_named_tensors": len(rows),
        "patterns": args.pattern,
        "matched_param_count": int(sum(int(row["numel"]) for row in matched_rows)),
        "matched_named_tensors": len(matched_rows),
        "lora_rank": int(args.lora_rank),
        "lora_candidate_modules": module_rows,
        "matched_lora_candidate_modules": matched_module_rows,
        "matched_lora_candidate_count": len(matched_module_rows),
        "matched_lora_param_count": int(
            sum(int(row["lora_param_count"]) for row in matched_module_rows if bool(row["lora_supported"]))
        ),
        "matched": matched_rows,
    }

    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"matched", "lora_candidate_modules", "matched_lora_candidate_modules"}
            },
            indent=2,
        )
    )
    if patterns:
        print("\nMatched parameter names:")
        for row in matched_rows[: max(args.top_n, 0)]:
            print(f"{row['numel']:>12}  {row['name']}  {row['shape']}")
        if len(matched_rows) > args.top_n:
            print(f"... {len(matched_rows) - args.top_n} more")
        print("\nMatched LoRA candidate modules:")
        for row in matched_module_rows[: max(args.top_n, 0)]:
            suffix = "" if row["lora_supported"] else "  [unsupported grouped conv]"
            print(
                f"{int(row['lora_param_count']):>12}  {row['type']}  {row['name']}"
                f"  base={row['base_param_count']}{suffix}"
            )
        if len(matched_module_rows) > args.top_n:
            print(f"... {len(matched_module_rows) - args.top_n} more")
    else:
        print("\nAll parameter names:")
        for row in rows[: max(args.top_n, 0)]:
            print(f"{row['numel']:>12}  {row['name']}  {row['shape']}")
        if len(rows) > args.top_n:
            print(f"... {len(rows) - args.top_n} more")
        print("\nAll LoRA candidate modules:")
        for row in module_rows[: max(args.top_n, 0)]:
            suffix = "" if row["lora_supported"] else "  [unsupported grouped conv]"
            print(
                f"{int(row['lora_param_count']):>12}  {row['type']}  {row['name']}"
                f"  base={row['base_param_count']}{suffix}"
            )
        if len(module_rows) > args.top_n:
            print(f"... {len(module_rows) - args.top_n} more")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
