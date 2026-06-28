from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


ARTIFACT_KEYS = ("reference_image", "semantic_only_image", "stage3_image", "stage4_image", "triptych_image")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "path" not in row:
            raise ValueError(f"{path}:{line_number} has no path")
        rows.append(row)
    return rows


def merge_artifact_rows(rows: list[dict[str, Any]], artifact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_path = {str(row.get("path", "")): row for row in artifact_rows if row.get("path")}
    by_index = {int(row["index"]): row for row in artifact_rows if "index" in row}
    merged: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        artifact = by_path.get(str(row.get("path", "")))
        if artifact is None and "index" in row:
            artifact = by_index.get(int(row["index"]))
        if artifact is not None:
            for key in ARTIFACT_KEYS:
                if key not in output and key in artifact:
                    output[key] = artifact[key]
        merged.append(output)
    return merged


def sorted_rows(rows: list[dict[str, Any]], metric: str, *, direction: str) -> list[dict[str, Any]]:
    missing = [row.get("path", row.get("index", "")) for row in rows if metric not in row]
    if missing:
        raise KeyError(f"{metric} is missing from {len(missing)} rows; first missing key: {missing[0]}")
    reverse = direction == "high"
    if direction not in {"high", "low"}:
        raise ValueError(f"unknown direction: {direction}")
    return sorted(rows, key=lambda row: float(row[metric]), reverse=reverse)


def maybe_copy(path_value: str, artifact_dir: Path | None, prefix: str) -> str:
    if not path_value:
        return ""
    src = Path(path_value)
    if artifact_dir is None:
        return str(src)
    if not src.exists():
        return str(src)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dst = artifact_dir / f"{prefix}_{src.name}"
    shutil.copy2(src, dst)
    return str(dst)


def write_gallery(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    output_md: Path,
    artifact_dir: Path | None,
    title: str,
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"Sorted by `{metric}`.",
        "",
        "| rank | metric | bpp | source | reference | semantic | stage3 | stage4 | triptych |",
        "| ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        prefix = f"rank{rank:02d}"
        reference = maybe_copy(str(row.get("reference_image", "")), artifact_dir, f"{prefix}_reference")
        semantic = maybe_copy(str(row.get("semantic_only_image", "")), artifact_dir, f"{prefix}_semantic")
        stage3 = maybe_copy(str(row.get("stage3_image", "")), artifact_dir, f"{prefix}_stage3")
        stage4 = maybe_copy(str(row.get("stage4_image", "")), artifact_dir, f"{prefix}_stage4")
        triptych = maybe_copy(str(row.get("triptych_image", "")), artifact_dir, f"{prefix}_triptych")
        source = str(row.get("path", ""))
        metric_value = float(row[metric])
        bpp = float(row.get("actual_payload_bpp", 0.0))
        lines.append(
            "| "
            f"{rank} | {metric_value:.6g} | {bpp:.6g} | `{source}` | "
            f"{markdown_image(reference, output_md.parent)} | {markdown_image(semantic, output_md.parent)} | "
            f"{markdown_image(stage3, output_md.parent)} | {markdown_image(stage4, output_md.parent)} | "
            f"{markdown_image(triptych, output_md.parent)} |"
        )
    output_md.write_text("\n".join(lines) + "\n")


def markdown_image(path_value: str, base_dir: Path | None = None) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    link_path = path
    if base_dir is not None:
        link_path = Path(os.path.relpath(path, start=base_dir))
    return f"[{path.name}]({link_path.as_posix()})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--artifact-input", type=Path, default=None)
    parser.add_argument("--metric", default="stage3_lpips_alex_delta_vs_semantic_only")
    parser.add_argument("--direction", choices=["high", "low"], default="high")
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--title", default="")
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.artifact_input is not None:
        rows = merge_artifact_rows(rows, read_jsonl(args.artifact_input))
    rows = sorted_rows(rows, args.metric, direction=args.direction)[: args.top_k]
    title = args.title or f"Worst cases by {args.metric}"
    write_gallery(rows, metric=args.metric, output_md=args.output_md, artifact_dir=args.artifact_dir, title=title)
    print(json.dumps({"output_md": str(args.output_md), "artifact_dir": str(args.artifact_dir or ""), "count": len(rows)}))


if __name__ == "__main__":
    main()
