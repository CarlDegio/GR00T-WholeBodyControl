import msgpack
import pytest

from gear_sonic.scripts.benchmark_camera_stream import CameraStreamStats


def _packed_message(timestamp: float, padding_length: int) -> bytes:
    return msgpack.packb(
        {
            "timestamps": {
                "ego_view": timestamp,
                "left_wrist": timestamp,
            },
            "images": {
                "ego_view": b"x",
                "left_wrist": b"y",
            },
            "padding": "p" * padding_length,
        },
        use_bin_type=True,
    )


def test_stream_stats_reports_rates_throughput_latency_and_decode_cost():
    first = _packed_message(timestamp=9.9, padding_length=6)
    second = _packed_message(timestamp=10.9, padding_length=45)
    assert len(first) == 100
    assert len(second) == 140

    stats = CameraStreamStats()
    stats.add_message(first, received_at=10.0, decode_ms=2.0)
    stats.add_message(second, received_at=11.0, decode_ms=4.0)

    summary = stats.summary(elapsed_s=2.0)

    assert summary["messages"] == 2
    assert summary["message_fps"] == 1.0
    assert summary["wire_bytes_per_second"] == 120.0
    assert summary["images_per_second"] == 2.0
    assert summary["streams"]["ego_view"]["unique_fps"] == 1.0
    assert summary["streams"]["left_wrist"]["unique_fps"] == 1.0
    assert summary["latency_ms"]["p50"] == pytest.approx(100.0)
    assert summary["decode_ms"]["mean"] == 3.0


def test_stream_stats_rejects_nonpositive_elapsed_time():
    stats = CameraStreamStats()

    with pytest.raises(ValueError, match="elapsed_s must be positive"):
        stats.summary(elapsed_s=0.0)
