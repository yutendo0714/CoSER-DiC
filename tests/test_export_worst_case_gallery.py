from __future__ import annotations

from scripts.export_worst_case_gallery import merge_artifact_rows, sorted_rows


def test_merge_artifact_rows_uses_path_key_first() -> None:
    rows = [{"index": 0, "path": "/data/a.png", "metric": 1.0}]
    artifacts = [{"index": 99, "path": "/data/a.png", "stage3_image": "stage3/a.png"}]

    merged = merge_artifact_rows(rows, artifacts)

    assert merged[0]["stage3_image"] == "stage3/a.png"


def test_sorted_rows_supports_high_and_low_direction() -> None:
    rows = [{"path": "a", "metric": 2.0}, {"path": "b", "metric": 1.0}]

    assert [row["path"] for row in sorted_rows(rows, "metric", direction="high")] == ["a", "b"]
    assert [row["path"] for row in sorted_rows(rows, "metric", direction="low")] == ["b", "a"]
