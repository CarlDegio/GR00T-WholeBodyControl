from __future__ import annotations

import pytest

from gear_sonic.runtime.vla_timing import VlaTimingWindow


def test_vla_timing_window_reports_fixed_rolling_statistics() -> None:
    window = VlaTimingWindow(window_size=3)
    window.record({"camera_read": 1.0, "worker_total": 10.0}, received_ns=1_000_000)
    window.record({"camera_read": 3.0, "worker_total": 20.0}, received_ns=2_000_000)
    window.record({"camera_read": 5.0, "worker_total": 30.0}, received_ns=3_000_000)

    snapshot = window.snapshot(now_ns=4_000_000)
    camera = snapshot["segments_ms"]["camera_read"]

    assert snapshot["sample_count"] == 3
    assert snapshot["last_sample_age_ms"] == 1.0
    assert camera == {
        "last": 5.0,
        "mean": 3.0,
        "p50": 3.0,
        "p95": pytest.approx(4.8),
        "count": 3,
    }


def test_vla_timing_rejects_negative_measurements() -> None:
    window = VlaTimingWindow()
    with pytest.raises(ValueError, match="invalid VLA timing"):
        window.record({"worker_total": -1.0})
