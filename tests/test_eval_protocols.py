from __future__ import annotations

from pathlib import Path

import pytest

from coserdic.datasets.eval_protocols import (
    EvalProtocolCountError,
    flatten_selection_paths,
    resolve_eval_dataset,
    resolve_eval_protocol,
    validate_expected_counts,
)


def touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_div2k_val_uses_0801_to_0900_from_mixed_root(tmp_path: Path) -> None:
    dpl = tmp_path / "dpl"
    for index in [1, 2, 800, 801, 850, 900]:
        touch_image(dpl / "div2k" / f"{index:04d}.png")

    selection = resolve_eval_dataset("div2k_val", dpl_root=dpl)

    assert [path.name for path in selection.paths] == ["0801.png", "0850.png", "0900.png"]
    assert selection.expected_count == 100
    assert selection.status == "count_mismatch"


def test_clic2020_test_concatenates_professional_and_mobile(tmp_path: Path) -> None:
    dpl = tmp_path / "dpl"
    touch_image(dpl / "clic" / "professional" / "test" / "p001.png")
    touch_image(dpl / "clic" / "mobile" / "test" / "m001.png")

    selection = resolve_eval_dataset("clic2020_test", dpl_root=dpl)

    assert [path.name for path in selection.paths] == ["p001.png", "m001.png"]
    assert selection.expected_count == 428
    assert selection.status == "count_mismatch"


def test_protocol_can_resolve_selected_dataset_subset_without_strict_counts(tmp_path: Path) -> None:
    dpl = tmp_path / "dpl"
    touch_image(dpl / "clic" / "professional" / "test" / "p001.png")
    touch_image(dpl / "clic" / "mobile" / "test" / "m001.png")

    selections = resolve_eval_protocol(
        "gencodec_reproduction",
        dpl_root=dpl,
        dataset_keys=["clic2020_test"],
        strict_expected_counts=False,
    )

    assert len(selections) == 1
    assert [path.name for path in flatten_selection_paths(selections)] == ["p001.png", "m001.png"]


def test_strict_count_validation_raises_on_incomplete_protocol(tmp_path: Path) -> None:
    dpl = tmp_path / "dpl"
    touch_image(dpl / "kodak" / "kodim01.png")
    selections = resolve_eval_protocol("coser_common_lic", dpl_root=dpl, strict_expected_counts=False)

    with pytest.raises(EvalProtocolCountError):
        validate_expected_counts(selections)
