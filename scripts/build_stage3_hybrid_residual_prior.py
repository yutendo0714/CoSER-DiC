from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from coserdic.entropy import (
    StaticResidualGridHybridHuffmanCode,
    StaticResidualGridPositionHuffmanCode,
    StaticResidualGridSemanticPositionHuffmanCode,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-prior", required=True)
    parser.add_argument("--semantic-position-prior", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    position_path = Path(args.position_prior)
    semantic_path = Path(args.semantic_position_prior)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    position_code = StaticResidualGridPositionHuffmanCode.from_dict(json.loads(position_path.read_text()))
    semantic_code = StaticResidualGridSemanticPositionHuffmanCode.from_dict(json.loads(semantic_path.read_text()))
    hybrid_code = StaticResidualGridHybridHuffmanCode(
        position_code=position_code,
        semantic_position_code=semantic_code,
    )

    prior_path = out_dir / "static_residual_grid_hybrid_huffman_prior.json"
    summary_path = out_dir / "summary.json"
    prior_path.write_text(json.dumps(hybrid_code.to_dict(), indent=2) + "\n")
    summary = {
        "run_name": args.run_name or out_dir.name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "position_prior": str(position_path),
        "semantic_position_prior": str(semantic_path),
        "prior": str(prior_path),
        "bits": int(hybrid_code.bits),
        "value_range": float(hybrid_code.value_range),
        "detail_shape": list(hybrid_code.detail_shape),
        "payload_codec": "hybrid_huffman",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": summary, "artifacts": {"prior": str(prior_path), "summary": str(summary_path)}}, indent=2))


if __name__ == "__main__":
    main()
