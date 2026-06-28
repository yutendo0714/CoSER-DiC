from __future__ import annotations

import json

from scripts.summarize_per_image_metrics import dataset_key, grouped_dataset_key, read_jsonl, summarize_rows


def test_dataset_key_known_protocol_paths() -> None:
    assert dataset_key("/dpl/kodak/kodim01.png") == "kodak"
    assert dataset_key("/dpl/clic/professional/valid/sample.png") == "clic_professional_valid"
    assert dataset_key("/dpl/clic/mobile/valid/sample.png") == "clic_mobile_valid"
    assert dataset_key("/dpl/clic/professional/test/sample.png") == "clic2020_test_professional"
    assert dataset_key("/dpl/clic/mobile/test/sample.png") == "clic2020_test_mobile"
    assert dataset_key("/dpl/div2k/0801.png") == "div2k_val"
    assert dataset_key("/dpl/div2k/0001.png") == "div2k_train"


def test_grouped_dataset_key_combines_gencodec_clic_test() -> None:
    assert (
        grouped_dataset_key("/dpl/clic/professional/test/sample.png", "cod_reproduction_512")
        == "clic2020_test"
    )
    assert (
        grouped_dataset_key("/dpl/clic/professional/test/sample.png", "gencodec_reproduction")
        == "clic2020_test"
    )
    assert grouped_dataset_key("/dpl/clic/mobile/test/sample.png", "gencodec_reproduction") == "clic2020_test"
    assert grouped_dataset_key("/dpl/kodak/kodim01.png", "gencodec_reproduction") == "kodak"
    assert grouped_dataset_key("/dpl/clic/professional/test/sample.png", "path") == "clic2020_test_professional"


def test_summarize_rows_ignores_missing_metrics(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"path": "/dpl/kodak/kodim01.png", "actual_payload_bpp": 0.1}),
                json.dumps({"path": "/dpl/kodak/kodim02.png", "actual_payload_bpp": 0.3}),
            ]
        )
        + "\n"
    )

    rows = read_jsonl(path)
    summary = summarize_rows(rows, ("actual_payload_bpp", "stage3_lpips_alex"))

    assert summary["count"] == 2
    assert summary["actual_payload_bpp"]["mean"] == 0.2
    assert "stage3_lpips_alex" not in summary


def test_summarize_rows_marks_boolean_metrics() -> None:
    rows = [
        {"path": "/dpl/kodak/kodim01.png", "semantic_token_roundtrip": True},
        {"path": "/dpl/kodak/kodim02.png", "semantic_token_roundtrip": False},
    ]

    summary = summarize_rows(rows, ("semantic_token_roundtrip",))

    assert summary["semantic_token_roundtrip"]["mean"] == 0.5
    assert summary["semantic_token_roundtrip"]["all_true"] is False
    assert summary["semantic_token_roundtrip"]["true_count"] == 1
    assert summary["semantic_token_roundtrip"]["false_count"] == 1
