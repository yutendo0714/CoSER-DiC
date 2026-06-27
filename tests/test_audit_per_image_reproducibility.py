from __future__ import annotations

from scripts.audit_per_image_reproducibility import audit_rows


def test_audit_rows_passes_with_strict_match_and_small_tolerance_delta() -> None:
    reference = {"a": {"payload": 1, "psnr": 10.0}}
    candidate = {"a": {"payload": 1, "psnr": 10.000001}}

    payload = audit_rows(
        reference,
        candidate,
        strict_fields=("payload",),
        tolerance_fields=("psnr",),
        atol=1.0e-5,
    )

    assert payload["pass"] is True
    assert payload["strict_mismatch_count"] == 0
    assert payload["tolerance_mismatch_count"] == 0


def test_audit_rows_fails_on_strict_mismatch() -> None:
    reference = {"a": {"payload": 1, "psnr": 10.0}}
    candidate = {"a": {"payload": 2, "psnr": 10.0}}

    payload = audit_rows(
        reference,
        candidate,
        strict_fields=("payload",),
        tolerance_fields=("psnr",),
        atol=1.0e-5,
    )

    assert payload["pass"] is False
    assert payload["strict_mismatch_count"] == 1
