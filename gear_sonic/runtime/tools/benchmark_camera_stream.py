"""Benchmark a composed camera ZMQ stream from the receiving PC."""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import cv2
import msgpack
import numpy as np
import zmq

DEFAULT_EXPECTED_STREAMS = (
    "ego_view",
    "ego_view_depth",
    "chest_view",
    "chest_view_depth",
    "left_wrist",
    "right_wrist",
)


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


def _base64_counterfactual_size(message: dict[str, Any]) -> int:
    """Pack the same message after replacing binary image values with Base64."""
    counterfactual = dict(message)
    counterfactual["images"] = {
        name: base64.b64encode(data).decode("ascii")
        if isinstance(data, bytes | bytearray)
        else data
        for name, data in message.get("images", {}).items()
    }
    return len(msgpack.packb(counterfactual, use_bin_type=True))


class CameraStreamStats:
    """Accumulate receiver-side camera stream statistics."""

    def __init__(self) -> None:
        self.message_count = 0
        self.total_wire_bytes = 0
        self.total_base64_counterfactual_wire_bytes = 0
        self.binary_image_count = 0
        self.image_count = 0
        self.latency_ms: list[float] = []
        self.decode_ms: list[float] = []
        self.stream_samples: dict[str, int] = defaultdict(int)
        self.stream_unique_timestamps: dict[str, set[float]] = defaultdict(set)
        self.stream_latency_ms: dict[str, list[float]] = defaultdict(list)
        self.image_shapes: dict[str, list[int]] = {}
        self.image_dtypes: dict[str, str] = {}

    def add_message(self, packed: bytes, received_at: float, decode_ms: float) -> None:
        message = msgpack.unpackb(packed, raw=False)
        timestamps = message.get("timestamps", {})
        images = message.get("images", {})

        self.message_count += 1
        self.total_wire_bytes += len(packed)
        self.total_base64_counterfactual_wire_bytes += _base64_counterfactual_size(
            message
        )
        self.binary_image_count += sum(
            isinstance(encoded, bytes | bytearray) for encoded in images.values()
        )
        self.image_count += len(images)
        self.decode_ms.append(decode_ms)

        for stream, timestamp in timestamps.items():
            latency_ms = (received_at - float(timestamp)) * 1000.0
            self.stream_samples[stream] += 1
            self.stream_unique_timestamps[stream].add(float(timestamp))
            self.stream_latency_ms[stream].append(latency_ms)
            self.latency_ms.append(latency_ms)

    def record_decoded_images(self, decoded: dict[str, np.ndarray]) -> None:
        for stream, image in decoded.items():
            self.image_shapes[stream] = list(image.shape)
            self.image_dtypes[stream] = str(image.dtype)

    def summary(self, elapsed_s: float) -> dict[str, object]:
        if elapsed_s <= 0:
            raise ValueError("elapsed_s must be positive")

        streams = {}
        for stream in sorted(self.stream_samples):
            unique_frames = len(self.stream_unique_timestamps[stream])
            streams[stream] = {
                "samples": self.stream_samples[stream],
                "unique_frames": unique_frames,
                "unique_fps": unique_frames / elapsed_s,
                "latency_ms": _distribution(self.stream_latency_ms[stream]),
            }

        wire_bytes_per_second = self.total_wire_bytes / elapsed_s
        counterfactual_bytes = self.total_base64_counterfactual_wire_bytes
        savings_bytes = counterfactual_bytes - self.total_wire_bytes
        return {
            "elapsed_seconds": elapsed_s,
            "messages": self.message_count,
            "message_fps": self.message_count / elapsed_s,
            "images": self.image_count,
            "binary_images": self.binary_image_count,
            "images_per_second": self.image_count / elapsed_s,
            "wire_bytes": self.total_wire_bytes,
            "actual_wire_bytes": self.total_wire_bytes,
            "wire_bytes_per_second": wire_bytes_per_second,
            "wire_mib_per_second": wire_bytes_per_second / (1024.0**2),
            "wire_mbit_per_second": wire_bytes_per_second * 8.0 / 1_000_000.0,
            "mean_message_bytes": (
                self.total_wire_bytes / self.message_count if self.message_count else 0.0
            ),
            "base64_counterfactual_wire_bytes": counterfactual_bytes,
            "base64_counterfactual_wire_bytes_per_second": (
                counterfactual_bytes / elapsed_s
            ),
            "base64_counterfactual_wire_mbit_per_second": (
                counterfactual_bytes / elapsed_s * 8.0 / 1_000_000.0
            ),
            "binary_wire_savings_bytes": savings_bytes,
            "binary_wire_savings_percent": (
                savings_bytes / counterfactual_bytes * 100.0
                if counterfactual_bytes
                else 0.0
            ),
            "latency_ms": _distribution(self.latency_ms),
            "decode_ms": _distribution(self.decode_ms),
            "streams": streams,
            "image_shapes": self.image_shapes,
            "image_dtypes": self.image_dtypes,
        }


def _decode_image(encoded: Any) -> np.ndarray:
    if isinstance(encoded, str):
        encoded = base64.b64decode(encoded)
    if not isinstance(encoded, bytes | bytearray):
        raise TypeError(f"unsupported encoded image type: {type(encoded).__name__}")
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("OpenCV failed to decode an image payload")
    return image


def decode_message_images(packed: bytes) -> dict[str, np.ndarray]:
    message = msgpack.unpackb(packed, raw=False)
    return {
        stream: _decode_image(encoded)
        for stream, encoded in message.get("images", {}).items()
    }


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.123.164")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--warmup-seconds", type=_positive_float, default=5.0)
    parser.add_argument("--duration-seconds", type=_positive_float, default=60.0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--expected-streams",
        default=",".join(DEFAULT_EXPECTED_STREAMS),
        help="Comma-separated image keys required in the measured stream",
    )
    return parser.parse_args()


def _receive(socket: zmq.Socket, timeout_ms: int = 5000) -> bytes:
    if not socket.poll(timeout_ms):
        raise TimeoutError("no camera message received within 5 seconds")
    return socket.recv()


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    endpoint = f"tcp://{args.host}:{args.port}"
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVHWM, 100)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)

    try:
        warmup_deadline = time.perf_counter() + args.warmup_seconds
        while time.perf_counter() < warmup_deadline:
            decode_message_images(_receive(socket))

        stats = CameraStreamStats()
        started_at = time.perf_counter()
        deadline = started_at + args.duration_seconds
        while time.perf_counter() < deadline:
            packed = _receive(socket)
            received_at = time.time()
            decode_started_at = time.perf_counter()
            decoded = decode_message_images(packed)
            decode_ms = (time.perf_counter() - decode_started_at) * 1000.0
            stats.add_message(packed, received_at=received_at, decode_ms=decode_ms)
            stats.record_decoded_images(decoded)
        elapsed_s = time.perf_counter() - started_at
    finally:
        socket.close()
        context.term()

    summary = stats.summary(elapsed_s)
    expected_streams = tuple(
        stream.strip() for stream in args.expected_streams.split(",") if stream.strip()
    )
    observed_streams = set(summary["streams"])
    missing_streams = sorted(set(expected_streams) - observed_streams)
    if missing_streams:
        raise RuntimeError(f"missing expected camera streams: {', '.join(missing_streams)}")

    result = {
        "label": args.label,
        "endpoint": endpoint,
        "warmup_seconds": args.warmup_seconds,
        "requested_duration_seconds": args.duration_seconds,
        "expected_streams": list(expected_streams),
        **summary,
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
