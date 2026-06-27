from __future__ import annotations

import argparse
import json
from pathlib import Path

from coserdic.datasets.eval_protocols import (
    available_eval_protocols,
    protocol_summary,
    resolve_eval_protocol,
    validate_expected_counts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", action="append", choices=available_eval_protocols(), default=None)
    parser.add_argument("--dpl-root", default="/dpl")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    protocols = args.protocol or list(available_eval_protocols())
    payload: dict[str, object] = {
        "dpl_root": args.dpl_root,
        "protocols": {},
    }
    for protocol in protocols:
        selections = resolve_eval_protocol(protocol, dpl_root=args.dpl_root, strict_expected_counts=False)
        summary = protocol_summary(protocol, selections)
        payload["protocols"][protocol] = summary
        print(f"{protocol}: {summary['total_images']} images")
        for dataset in summary["datasets"]:
            expected = dataset["expected_count"]
            expected_text = "unchecked" if expected is None else str(expected)
            print(
                "  "
                f"{dataset['key']:24s} "
                f"count={dataset['count']:4d} "
                f"expected={expected_text:>9s} "
                f"status={dataset['status']:14s} "
                f"first={dataset['first_path']}"
            )
        print()
        if args.strict:
            validate_expected_counts(selections)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
