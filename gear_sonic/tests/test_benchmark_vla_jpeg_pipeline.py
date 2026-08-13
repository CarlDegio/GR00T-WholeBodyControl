import argparse
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gear_sonic.runtime.client import SnapshotUnavailableError
from gear_sonic.scripts import benchmark_vla_jpeg_pipeline as benchmark

OPENPI_REPO = Path("/home/user/Project/openpi_sonic")


def test_vla_stats_report_unique_rates_throughput_and_latency():
    stats = benchmark.VlaPipelineStats()
    values = {
        "encoded_frames": {
            "ego_view": (1, 101),
            "chest_view": (2, 102),
            "left_wrist": (3, 103),
            "right_wrist": (4, 104),
        },
        "decoded_frames": {
            "ego_view": (5, 101),
            "ego_view_depth": (6, 101),
        },
        "request_bytes": 1000,
        "camera_rpc_ms": 1.0,
        "jpeg_prepare_ms": 0.1,
        "request_pack_ms": 0.2,
        "openpi_decode_ms": 4.0,
    }
    stats.record(**values)
    stats.record(**values)

    summary = stats.summary(2.0)

    assert summary["vla_requests"] == 2
    assert summary["vla_request_fps"] == 1.0
    assert summary["vla_request_bytes"] == 2000
    assert summary["vla_bytes_per_request"] == 1000.0
    assert summary["vla_mbit_per_second"] == 0.008
    assert summary["encoded_streams"]["ego_view"]["gateway_publication_fps"] == 0.5
    assert summary["encoded_streams"]["ego_view"]["physical_unique_fps"] == 0.5
    assert summary["decoded_streams"]["ego_view_depth"]["gateway_publication_fps"] == 0.5
    assert summary["decoded_streams"]["ego_view_depth"]["physical_unique_fps"] == 0.5
    assert summary["decoded_physical_images_per_second"] == 1.0
    assert summary["latency_ms"]["gateway_rpc"]["mean"] == 1.0
    assert summary["latency_ms"]["jpeg_prepare"]["p95"] == 0.1
    assert summary["latency_ms"]["request_pack"]["p50"] == 0.2
    assert summary["latency_ms"]["openpi_decode"]["p50"] == 4.0


def test_vla_stats_separates_gateway_publications_from_physical_timestamps():
    stats = benchmark.VlaPipelineStats()
    base = {
        "request_bytes": 10,
        "camera_rpc_ms": 1.0,
        "jpeg_prepare_ms": 0.1,
        "request_pack_ms": 0.2,
        "openpi_decode_ms": 4.0,
    }
    stats.record(
        encoded_frames={"ego_view": (10, 1_000)},
        decoded_frames={"ego_view": (20, 1_000)},
        **base,
    )
    stats.record(
        encoded_frames={"ego_view": (11, 1_000)},
        decoded_frames={"ego_view": (21, 1_000)},
        **base,
    )
    stats.record(
        encoded_frames={"ego_view": (13, 900)},
        decoded_frames={"ego_view": (23, 900)},
        **base,
    )

    summary = stats.summary(2.0)
    encoded = summary["encoded_streams"]["ego_view"]
    decoded = summary["decoded_streams"]["ego_view"]
    assert encoded == {
        "gateway_publications": 3,
        "gateway_publication_fps": 1.5,
        "physical_unique_frames": 2,
        "physical_unique_fps": 1.0,
        "source_timestamp_reuses": 1,
        "source_timestamp_regressions": 1,
        "source_timestamp_missing": 0,
        "observed_gateway_sequence_gaps": 1,
        "gateway_sequence_regressions": 0,
    }
    assert decoded["physical_unique_frames"] == 2
    assert summary["decoded_physical_images"] == 2
    assert summary["decoded_physical_images_per_second"] == 1.0
    assert "transport_drops" not in summary


def test_vla_stats_reports_snapshot_rejections_without_calling_them_drops():
    stats = benchmark.VlaPipelineStats()

    stats.record_snapshot_rejection("encoded", "snapshot exceeds requested skew")
    stats.record_snapshot_rejection("encoded", "snapshot exceeds requested skew")
    stats.record_snapshot_rejection("decoded", "missing streams")

    summary = stats.summary(1.0)
    assert summary["snapshot_rejections"] == {
        "total": 3,
        "encoded": 2,
        "decoded": 1,
        "reasons": {
            "missing streams": 1,
            "snapshot exceeds requested skew": 2,
        },
    }
    assert summary["transport_drop_observability"] == (
        "PUB/HWM transport drops are unobservable without a producer frame identifier"
    )


def test_codec_stats_ignore_duplicate_encoded_sequences():
    stats = benchmark.VlaPipelineStats()

    stats.record_codec(
        stream="ego_view", sequence=9, new_codec_ms=0.05, old_codec_ms=2.0
    )
    stats.record_codec(
        stream="ego_view", sequence=9, new_codec_ms=100.0, old_codec_ms=200.0
    )

    comparison = stats.summary(1.0)["codec_comparison_ms"]
    assert comparison["samples"] == 1
    assert comparison["new_p50"] == 0.05
    assert comparison["new_p95"] == 0.05
    assert comparison["old_p50"] == 2.0
    assert comparison["old_p95"] == 2.0


def test_prepare_video_pairs_each_jpeg_with_its_image_shape():
    camera = {
        "images": {
            "ego_view": b"ego",
            "chest_view": b"chest",
            "left_wrist": b"left",
            "right_wrist": b"right",
        },
        "image_shapes": {
            "ego_view": (10, 20, 3),
            "chest_view": (11, 21, 3),
            "left_wrist": (12, 22, 3),
            "right_wrist": (13, 23, 3),
        },
    }

    video, codec_ms = benchmark._prepare_video(camera)

    assert video["ego_view"]["data"] == b"ego"
    assert video["ego_view"]["shape"] == (1, 1, 10, 20, 3)
    assert video["right_wrist"]["shape"] == (1, 1, 13, 23, 3)
    assert set(codec_ms) == set(camera["images"])
    assert all(value >= 0.0 for value in codec_ms.values())


def test_vla_stats_rejects_nonpositive_elapsed_time():
    stats = benchmark.VlaPipelineStats()

    with pytest.raises(ValueError, match="elapsed_s must be positive"):
        stats.summary(0.0)


def test_load_openpi_decoder_executes_existing_dataclass_module():
    module = benchmark.load_openpi_decoder(OPENPI_REPO)

    assert module.__file__ == str(
        OPENPI_REPO / "scripts" / "serve_g1_sonic_zmq_policy.py"
    )
    assert callable(module._decode_jpeg_rgb_video)


def test_load_openpi_decoder_does_not_write_bytecode_into_repository(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "serve_g1_sonic_zmq_policy.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Config:\n"
        "    value: str\n"
        "def _decode_jpeg_rgb_video(value, *, name):\n"
        "    return value\n"
    )

    benchmark.load_openpi_decoder(tmp_path)

    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == [
        Path("scripts"),
        Path("scripts/serve_g1_sonic_zmq_policy.py"),
    ]


def test_benchmark_script_runs_directly_from_worktree():
    repository = Path(__file__).parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            "gear_sonic/scripts/benchmark_vla_jpeg_pipeline.py",
            "--help",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--sensor-gateway-endpoint" in completed.stdout


def test_run_benchmark_does_not_create_client_when_decoder_load_fails(
    tmp_path, monkeypatch
):
    created_clients = []

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            created_clients.append(self)

    monkeypatch.setattr(benchmark, "SensorGatewayClient", FakeClient)
    args = argparse.Namespace(
        sensor_gateway_endpoint="inproc://unused",
        request_timeout_ms=1000,
        openpi_repo=tmp_path,
        duration_seconds=1.0,
        output_json=tmp_path / "result.json",
    )

    with pytest.raises(FileNotFoundError, match="OpenPI decoder module not found"):
        benchmark.run_benchmark(args)

    assert created_clients == []


def test_load_openpi_decoder_restores_module_registry_when_decoder_missing(
    tmp_path,
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "serve_g1_sonic_zmq_policy.py").write_text("value = 1\n")
    module_name = "benchmark_openpi_serve_g1_sonic_zmq_policy"
    previous = sys.modules.pop(module_name, None)
    try:
        with pytest.raises(AttributeError, match="has no JPEG decoder"):
            benchmark.load_openpi_decoder(tmp_path)
        assert module_name not in sys.modules
    finally:
        if previous is not None:
            sys.modules[module_name] = previous


def test_run_benchmark_counts_snapshot_rejections_instead_of_aborting(tmp_path):
    class RejectingClient:
        def read_snapshot(self, _request, *, retries):
            assert retries == 0
            raise SnapshotUnavailableError("snapshot exceeds requested skew")

    args = argparse.Namespace(
        sensor_gateway_endpoint="inproc://unused",
        request_timeout_ms=1000,
        openpi_repo=tmp_path,
        duration_seconds=0.001,
        output_json=tmp_path / "rejections.json",
    )

    result = benchmark.run_benchmark(
        args,
        client=RejectingClient(),
        openpi_module=SimpleNamespace(_decode_jpeg_rgb_video=lambda value, *, name: value),
    )

    assert result["vla_requests"] == 0
    assert result["snapshot_rejections"]["encoded"] > 0
    assert result["snapshot_rejections"]["decoded"] == 0
