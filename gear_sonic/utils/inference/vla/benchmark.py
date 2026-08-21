"""Benchmark the binary-JPEG SensorGateway-to-OpenPI protocol pipeline.

This utility intentionally runs one sequential loop. It measures the protocol
without changing or reusing the production VLA ingress worker.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import ModuleType
from typing import Any, Mapping

import cv2
import msgpack_numpy as mnp
import numpy as np

from gear_sonic.camera.constants import PRODUCTION_JPEG_QUALITY
from gear_sonic.runtime.gateway.sensor_client import (
    SensorGatewayClient,
    SnapshotUnavailableError,
)
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest
from gear_sonic.utils.inference.vla.ingress import (
    VLA_CAMERA_NAMES,
    VLA_CAMERA_STREAMS,
    camera_message_from_snapshot,
)
from gear_sonic.utils.inference.vla.service import (
    wrap_camera_jpeg_for_video,
)

DECODED_CAMERA_NAMES = (
    "ego_view",
    "ego_view_depth",
    "chest_view",
    "chest_view_depth",
    "left_wrist",
    "right_wrist",
)
DECODED_CAMERA_STREAMS = tuple(f"camera/{name}" for name in DECODED_CAMERA_NAMES)
DEFAULT_OPENPI_REPO = Path("/home/user/Project/openpi_sonic")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


class VlaPipelineStats:
    """Accumulate request, unique-frame, latency, and codec measurements."""

    def __init__(self) -> None:
        self.request_count = 0
        self.total_request_bytes = 0
        self.gateway_sequences: dict[str, dict[str, set[int]]] = {
            "encoded": defaultdict(set),
            "decoded": defaultdict(set),
        }
        self.source_timestamps: dict[str, dict[str, set[int]]] = {
            "encoded": defaultdict(set),
            "decoded": defaultdict(set),
        }
        self.last_gateway_sequence: dict[str, dict[str, int]] = {
            "encoded": {},
            "decoded": {},
        }
        self.last_source_timestamp: dict[str, dict[str, int]] = {
            "encoded": {},
            "decoded": {},
        }
        self.stream_counters: dict[str, dict[str, Counter[str]]] = {
            "encoded": defaultdict(Counter),
            "decoded": defaultdict(Counter),
        }
        self.snapshot_rejections: Counter[str] = Counter()
        self.snapshot_rejection_reasons: Counter[str] = Counter()
        self.latency_ms: dict[str, list[float]] = {
            "gateway_rpc": [],
            "jpeg_prepare": [],
            "request_pack": [],
            "openpi_decode": [],
        }
        self.codec_sequences: dict[str, set[int]] = defaultdict(set)
        self.new_codec_ms: list[float] = []
        self.old_codec_ms: list[float] = []

    def record(
        self,
        *,
        encoded_frames: Mapping[str, tuple[int, int]],
        decoded_frames: Mapping[str, tuple[int, int]],
        request_bytes: int,
        camera_rpc_ms: float,
        jpeg_prepare_ms: float,
        request_pack_ms: float,
        openpi_decode_ms: float,
    ) -> None:
        self.request_count += 1
        self.total_request_bytes += int(request_bytes)
        self._record_frames("encoded", encoded_frames)
        self._record_frames("decoded", decoded_frames)
        self.latency_ms["gateway_rpc"].append(float(camera_rpc_ms))
        self.latency_ms["jpeg_prepare"].append(float(jpeg_prepare_ms))
        self.latency_ms["request_pack"].append(float(request_pack_ms))
        self.latency_ms["openpi_decode"].append(float(openpi_decode_ms))

    def _record_frames(
        self,
        stage: str,
        frames: Mapping[str, tuple[int, int]],
    ) -> None:
        for stream, (raw_sequence, raw_source_timestamp) in frames.items():
            sequence = int(raw_sequence)
            source_timestamp = int(raw_source_timestamp)
            sequences = self.gateway_sequences[stage][stream]
            if sequence in sequences:
                continue

            previous_sequence = self.last_gateway_sequence[stage].get(stream)
            counters = self.stream_counters[stage][stream]
            if previous_sequence is not None:
                if sequence > previous_sequence:
                    counters["observed_gateway_sequence_gaps"] += max(
                        0, sequence - previous_sequence - 1
                    )
                else:
                    counters["gateway_sequence_regressions"] += 1
            sequences.add(sequence)
            self.last_gateway_sequence[stage][stream] = sequence

            if source_timestamp <= 0:
                counters["source_timestamp_missing"] += 1
                continue
            previous_timestamp = self.last_source_timestamp[stage].get(stream)
            if previous_timestamp is not None:
                if source_timestamp == previous_timestamp:
                    counters["source_timestamp_reuses"] += 1
                elif source_timestamp < previous_timestamp:
                    counters["source_timestamp_regressions"] += 1
            self.source_timestamps[stage][stream].add(source_timestamp)
            self.last_source_timestamp[stage][stream] = source_timestamp

    def record_snapshot_rejection(self, stage: str, reason: str) -> None:
        if stage not in self.gateway_sequences:
            raise ValueError(f"unsupported snapshot stage: {stage!r}")
        self.snapshot_rejections[stage] += 1
        self.snapshot_rejection_reasons[str(reason)] += 1

    def record_codec(
        self,
        *,
        stream: str,
        sequence: int,
        new_codec_ms: float,
        old_codec_ms: float,
    ) -> bool:
        sequence = int(sequence)
        if sequence in self.codec_sequences[stream]:
            return False
        self.codec_sequences[stream].add(sequence)
        self.new_codec_ms.append(float(new_codec_ms))
        self.old_codec_ms.append(float(old_codec_ms))
        return True

    def _stream_summary(self, stage: str, elapsed_s: float) -> dict[str, dict[str, float | int]]:
        return {
            stream: {
                "gateway_publications": len(sequences),
                "gateway_publication_fps": len(sequences) / elapsed_s,
                "physical_unique_frames": len(self.source_timestamps[stage][stream]),
                "physical_unique_fps": len(self.source_timestamps[stage][stream])
                / elapsed_s,
                "source_timestamp_reuses": self.stream_counters[stage][stream][
                    "source_timestamp_reuses"
                ],
                "source_timestamp_regressions": self.stream_counters[stage][stream][
                    "source_timestamp_regressions"
                ],
                "source_timestamp_missing": self.stream_counters[stage][stream][
                    "source_timestamp_missing"
                ],
                "observed_gateway_sequence_gaps": self.stream_counters[stage][stream][
                    "observed_gateway_sequence_gaps"
                ],
                "gateway_sequence_regressions": self.stream_counters[stage][stream][
                    "gateway_sequence_regressions"
                ],
            }
            for stream, sequences in sorted(self.gateway_sequences[stage].items())
        }

    def summary(self, elapsed_s: float) -> dict[str, object]:
        if elapsed_s <= 0.0:
            raise ValueError("elapsed_s must be positive")

        encoded_streams = self._stream_summary("encoded", elapsed_s)
        decoded_streams = self._stream_summary("decoded", elapsed_s)
        request_bytes_per_second = self.total_request_bytes / elapsed_s
        new_codec = _distribution(self.new_codec_ms)
        old_codec = _distribution(self.old_codec_ms)
        return {
            "elapsed_seconds": elapsed_s,
            "vla_requests": self.request_count,
            "vla_request_fps": self.request_count / elapsed_s,
            "vla_request_bytes": self.total_request_bytes,
            "vla_bytes_per_request": (
                self.total_request_bytes / self.request_count
                if self.request_count
                else 0.0
            ),
            "vla_bytes_per_second": request_bytes_per_second,
            "vla_mbit_per_second": request_bytes_per_second * 8.0 / 1_000_000.0,
            "encoded_streams": encoded_streams,
            "decoded_streams": decoded_streams,
            "decoded_physical_images": sum(
                stream["physical_unique_frames"] for stream in decoded_streams.values()
            ),
            "decoded_physical_images_per_second": sum(
                stream["physical_unique_frames"] for stream in decoded_streams.values()
            )
            / elapsed_s,
            "snapshot_rejections": {
                "total": sum(self.snapshot_rejections.values()),
                "encoded": self.snapshot_rejections["encoded"],
                "decoded": self.snapshot_rejections["decoded"],
                "reasons": dict(sorted(self.snapshot_rejection_reasons.items())),
            },
            "transport_drop_observability": (
                "PUB/HWM transport drops are unobservable without a producer frame identifier"
            ),
            "latency_ms": {
                name: _distribution(values) for name, values in self.latency_ms.items()
            },
            "codec_comparison_ms": {
                "samples": len(self.new_codec_ms),
                "new": new_codec,
                "old": old_codec,
                "new_mean": new_codec["mean"],
                "new_p50": new_codec["p50"],
                "new_p95": new_codec["p95"],
                "old_mean": old_codec["mean"],
                "old_p50": old_codec["p50"],
                "old_p95": old_codec["p95"],
            },
        }


def _prepare_video(
    camera: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Wrap each existing JPEG and retain a per-image codec micro-timing."""
    video: dict[str, dict[str, Any]] = {}
    codec_ms: dict[str, float] = {}
    for name in VLA_CAMERA_NAMES:
        started_ns = time.perf_counter_ns()
        video[name] = wrap_camera_jpeg_for_video(
            camera["images"][name], camera["image_shapes"][name]
        )
        codec_ms[name] = _elapsed_ms(started_ns)
    return video, codec_ms


def _old_reencode_rgb_video(decoded: np.ndarray) -> bytes:
    """Reproduce the removed RGB-to-JPEG quality-95 codec step."""
    frame = np.asarray(decoded)
    if frame.ndim == 5 and frame.shape[:2] == (1, 1):
        frame = frame[0, 0]
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"decoded OpenPI video must be uint8 RGB HWC, got {frame.shape} {frame.dtype}")
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), PRODUCTION_JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed in old codec benchmark")
    return encoded.tobytes()


def load_openpi_decoder(openpi_repo: Path) -> ModuleType:
    """Load the existing OpenPI server decoder without modifying its repository."""
    module_path = openpi_repo / "scripts" / "serve_g1_sonic_zmq_policy.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"OpenPI decoder module not found: {module_path}")
    module_name = "benchmark_openpi_serve_g1_sonic_zmq_policy"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load OpenPI decoder module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.modules[module_name] = module
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
        if not callable(getattr(module, "_decode_jpeg_rgb_video", None)):
            raise AttributeError(f"OpenPI module has no JPEG decoder: {module_path}")
    except BaseException:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


def _frame_observations(
    snapshot: Any,
    streams: tuple[str, ...],
    names: tuple[str, ...],
) -> dict[str, tuple[int, int]]:
    return {
        name: (
            int(snapshot.snapshot.frames[stream].metadata.sequence),
            int(snapshot.snapshot.frames[stream].source_timestamp_ns),
        )
        for stream, name in zip(streams, names, strict=True)
    }


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-gateway-endpoint", required=True)
    parser.add_argument("--openpi-repo", type=Path, default=DEFAULT_OPENPI_REPO)
    parser.add_argument("--duration-seconds", type=_positive_float, default=60.0)
    parser.add_argument("--request-timeout-ms", type=int, default=1000)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def run_benchmark(
    args: argparse.Namespace,
    *,
    client: SensorGatewayClient | None = None,
    openpi_module: ModuleType | None = None,
) -> dict[str, object]:
    """Run the encoded request and decoded-rate measurements sequentially."""
    if openpi_module is None:
        openpi_module = load_openpi_decoder(args.openpi_repo)
    own_client = client is None
    if client is None:
        client = SensorGatewayClient(
            args.sensor_gateway_endpoint,
            request_timeout_ms=args.request_timeout_ms,
        )

    encoded_request = SnapshotRequest(
        streams=VLA_CAMERA_STREAMS,
        max_age_ms=1000.0,
        max_skew_ms=5.0,
    )
    decoded_request = SnapshotRequest(
        streams=DECODED_CAMERA_STREAMS,
        max_age_ms=1000.0,
        max_skew_ms=5.0,
    )
    stats = VlaPipelineStats()
    seen_codec_sequences: dict[str, set[int]] = defaultdict(set)
    started_ns = time.perf_counter_ns()
    deadline_ns = started_ns + int(args.duration_seconds * 1_000_000_000)
    try:
        while time.perf_counter_ns() < deadline_ns:
            rpc_started_ns = time.perf_counter_ns()
            try:
                encoded = client.read_snapshot(encoded_request, retries=0)
            except SnapshotUnavailableError as exc:
                stats.record_snapshot_rejection("encoded", str(exc))
                continue
            camera_rpc_ms = _elapsed_ms(rpc_started_ns)

            prepare_started_ns = time.perf_counter_ns()
            camera = camera_message_from_snapshot(encoded)
            video, new_codec_ms = _prepare_video(camera)
            jpeg_prepare_ms = _elapsed_ms(prepare_started_ns)

            pack_started_ns = time.perf_counter_ns()
            packed = mnp.packb(
                {"endpoint": "get_action", "data": {"observation": {"video": video}}}
            )
            request_pack_ms = _elapsed_ms(pack_started_ns)

            decode_started_ns = time.perf_counter_ns()
            decoded_video = {
                name: openpi_module._decode_jpeg_rgb_video(
                    video[name], name=f"video.{name}"
                )
                for name in VLA_CAMERA_NAMES
            }
            openpi_decode_ms = _elapsed_ms(decode_started_ns)

            encoded_frames = _frame_observations(
                encoded, VLA_CAMERA_STREAMS, VLA_CAMERA_NAMES
            )
            for name, (sequence, _source_timestamp) in encoded_frames.items():
                if sequence in seen_codec_sequences[name]:
                    continue
                seen_codec_sequences[name].add(sequence)
                old_started_ns = time.perf_counter_ns()
                _old_reencode_rgb_video(decoded_video[name])
                old_codec_ms = _elapsed_ms(old_started_ns)
                stats.record_codec(
                    stream=name,
                    sequence=sequence,
                    new_codec_ms=new_codec_ms[name],
                    old_codec_ms=old_codec_ms,
                )

            try:
                decoded = client.read_snapshot(decoded_request, retries=0)
            except SnapshotUnavailableError as exc:
                stats.record_snapshot_rejection("decoded", str(exc))
                continue
            decoded_frames = _frame_observations(
                decoded, DECODED_CAMERA_STREAMS, DECODED_CAMERA_NAMES
            )
            stats.record(
                encoded_frames=encoded_frames,
                decoded_frames=decoded_frames,
                request_bytes=len(packed),
                camera_rpc_ms=camera_rpc_ms,
                jpeg_prepare_ms=jpeg_prepare_ms,
                request_pack_ms=request_pack_ms,
                openpi_decode_ms=openpi_decode_ms,
            )
    finally:
        if own_client:
            client.close()

    elapsed_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
    result = {
        "sensor_gateway_endpoint": args.sensor_gateway_endpoint,
        "openpi_decoder": str(
            args.openpi_repo / "scripts" / "serve_g1_sonic_zmq_policy.py"
        ),
        "requested_duration_seconds": args.duration_seconds,
        "single_threaded_sequential": True,
        **stats.summary(elapsed_s),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved benchmark JSON to {args.output_json}")


if __name__ == "__main__":
    main()
