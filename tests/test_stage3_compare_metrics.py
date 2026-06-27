from __future__ import annotations

import json

from scripts.compare_stage3_per_image_metrics import compare, read_jsonl


def test_compare_stage3_per_image_metrics(tmp_path) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    reference.write_text(
        "\n".join(
            [
                json.dumps({"path": "a.png", "actual_payload_bpp": 0.10, "stage3_psnr": 20.0}),
                json.dumps({"path": "b.png", "actual_payload_bpp": 0.20, "stage3_psnr": 21.0}),
            ]
        )
        + "\n"
    )
    candidate.write_text(
        "\n".join(
            [
                json.dumps({"path": "a.png", "actual_payload_bpp": 0.09, "stage3_psnr": 20.5}),
                json.dumps({"path": "b.png", "actual_payload_bpp": 0.22, "stage3_psnr": 20.8}),
            ]
        )
        + "\n"
    )

    summary, rows = compare(
        read_jsonl(reference),
        read_jsonl(candidate),
        reference_label="ref",
        candidate_label="cand",
        rate_metric="actual_payload_bpp",
        quality_metric="stage3_psnr",
        quality_direction="higher",
    )

    assert summary["num_images"] == 2
    assert summary["candidate_rate_win_count"] == 1
    assert summary["candidate_quality_win_count"] == 1
    assert abs(summary["candidate_minus_reference_rate_mean"] - 0.005) < 1.0e-9
    assert rows[0]["cand_minus_ref_actual_payload_bpp"] == -0.010000000000000009


def test_compare_stage3_per_image_metrics_lower_is_better() -> None:
    reference_rows = {
        "a.png": {"actual_payload_bpp": 0.10, "stage3_lpips_alex": 0.30},
        "b.png": {"actual_payload_bpp": 0.20, "stage3_lpips_alex": 0.40},
    }
    candidate_rows = {
        "a.png": {"actual_payload_bpp": 0.11, "stage3_lpips_alex": 0.25},
        "b.png": {"actual_payload_bpp": 0.21, "stage3_lpips_alex": 0.45},
    }

    summary, _ = compare(
        reference_rows,
        candidate_rows,
        reference_label="ref",
        candidate_label="cand",
        rate_metric="actual_payload_bpp",
        quality_metric="stage3_lpips_alex",
        quality_direction="lower",
    )

    assert summary["quality_direction"] == "lower"
    assert summary["candidate_quality_win_count"] == 1
