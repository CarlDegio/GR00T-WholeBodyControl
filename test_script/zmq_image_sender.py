#!/usr/bin/env python3
"""Benchmark three resized camera images over ZMQ.

The script loads three local images, resizes them to 224x224, optionally
encodes/compresses them, sends them to a ZMQ REP receiver, and prints timing
statistics for resize, encode, communication, decode, and total latency.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any

import cv2
import numpy as np
import zmq


CAMERA_NAMES = ("head", "left", "right")
CODECS = ("turbojpeg", "opencv_jpeg", "blosc_zstd", "raw")
IMAGE_SIZE = (224, 224)


@dataclass
class IterationStats:
    codec: str
    quality: int | None
    payload_bytes: int
    resize_ms: float
    encode_ms: float
    comm_ms: float
    decode_ms: float
    total_ms: float


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def jpeg_quality(value: str) -> int:
    parsed = int(value)
    if parsed < 85 or parsed > 90:
        raise argparse.ArgumentTypeError("JPEG quality must be between 85 and 90")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize three local images to 224x224 and benchmark ZMQ transfer "
            "with TurboJPEG, OpenCV JPEG, Blosc+Zstd, or raw payloads."
        )
    )
    parser.add_argument("--server-host", default="124.220.175.102", help="Receiver host")
    parser.add_argument("--server-port", type=int, default=29999, help="Receiver port")
    parser.add_argument("--head-image", required=True, help="Path to the head camera image")
    parser.add_argument("--left-image", required=True, help="Path to the left camera image")
    parser.add_argument("--right-image", required=True, help="Path to the right camera image")
    parser.add_argument(
        "--codec",
        choices=("all", *CODECS),
        default="all",
        help="Encoding method to test",
    )
    parser.add_argument(
        "--quality",
        type=jpeg_quality,
        default=90,
        help="JPEG quality for turbojpeg/opencv_jpeg, restricted to 85-90",
    )
    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=100,
        help="Measured iterations per codec",
    )
    parser.add_argument(
        "--warmup",
        type=nonnegative_int,
        default=5,
        help="Warmup iterations per codec, not included in statistics",
    )
    parser.add_argument(
        "--timeout-ms",
        type=positive_int,
        default=5000,
        help="ZMQ send/receive timeout in milliseconds",
    )
    parser.add_argument(
        "--blosc-clevel",
        type=int,
        default=5,
        choices=range(0, 10),
        metavar="[0-9]",
        help="Blosc compression level for blosc_zstd",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional CSV path for per-iteration measured results",
    )
    return parser.parse_args()


def require_turbojpeg() -> Any:
    try:
        import turbojpeg
    except ImportError as exc:
        raise RuntimeError(
            "turbojpeg codec requested but the turbojpeg/PyTurboJPEG package is not installed"
        ) from exc

    if hasattr(turbojpeg, "TurboJPEG"):
        return ("py_turbojpeg", turbojpeg.TurboJPEG())
    if hasattr(turbojpeg, "compress"):
        return ("turbojpeg_module", turbojpeg)
    raise RuntimeError("Unsupported turbojpeg package API")


def require_blosc() -> Any:
    try:
        import blosc

        return ("blosc", blosc)
    except ImportError:
        pass

    try:
        import blosc2

        return ("blosc2", blosc2)
    except ImportError as exc:
        raise RuntimeError("blosc_zstd codec requested but neither blosc nor blosc2 is installed") from exc


def load_images(paths: dict[str, str]) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for name, raw_path in paths.items():
        path = Path(raw_path).expanduser()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read {name} image: {path}")
        images[name] = image
    return images


def resize_images(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        for name, image in images.items()
    }


def turbojpeg_encode(helper: Any, image: np.ndarray, quality: int) -> bytes:
    api_name, turbo = helper
    if api_name == "py_turbojpeg":
        try:
            from turbojpeg import TJPF_BGR
        except ImportError:
            TJPF_BGR = None
        if TJPF_BGR is None:
            return turbo.encode(image, quality=quality)
        return turbo.encode(image, quality=quality, pixel_format=TJPF_BGR)

    try:
        return turbo.compress(image, quality=quality, pixelformat="BGR")
    except TypeError:
        return turbo.compress(image, quality=quality)


def blosc_zstd_compress(helper: Any, data: bytes, clevel: int) -> bytes:
    api_name, module = helper
    if api_name == "blosc":
        return module.compress(data, typesize=1, clevel=clevel, shuffle=module.SHUFFLE, cname="zstd")

    return module.compress2(
        data,
        clevel=clevel,
        codec=module.Codec.ZSTD,
        filter=module.Filter.SHUFFLE,
    )


def encode_images(
    images: dict[str, np.ndarray],
    codec: str,
    quality: int,
    blosc_clevel: int,
    helpers: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], int]:
    encoded: dict[str, dict[str, Any]] = {}
    total_bytes = 0

    for name, image in images.items():
        if codec == "turbojpeg":
            payload = turbojpeg_encode(helpers["turbojpeg"], image, quality)
        elif codec == "opencv_jpeg":
            ok, buffer = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if not ok:
                raise RuntimeError(f"cv2.imencode failed for {name}")
            payload = buffer.tobytes()
        elif codec == "blosc_zstd":
            payload = blosc_zstd_compress(helpers["blosc_zstd"], image.tobytes(), blosc_clevel)
        elif codec == "raw":
            payload = image.tobytes()
        else:
            raise ValueError(f"Unsupported codec: {codec}")

        total_bytes += len(payload)
        encoded[name] = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "payload": payload,
        }

    return encoded, total_bytes


def make_request(
    seq: int,
    codec: str,
    quality: int | None,
    blosc_clevel: int,
    image_metadata: dict[str, dict[str, Any]],
    resize_ms: float,
    encode_ms: float,
) -> dict[str, Any]:
    return {
        "seq": seq,
        "codec": codec,
        "quality": quality,
        "blosc_clevel": blosc_clevel,
        "image_size": list(IMAGE_SIZE),
        "resize_ms": resize_ms,
        "encode_ms": encode_ms,
        "client_send_time_ns": time.perf_counter_ns(),
        "images": image_metadata,
    }


def run_iteration(
    socket: zmq.Socket,
    source_images: dict[str, np.ndarray],
    seq: int,
    codec: str,
    quality: int,
    blosc_clevel: int,
    helpers: dict[str, Any],
) -> IterationStats:
    resize_start = time.perf_counter_ns()
    resized_images = resize_images(source_images)
    resize_end = time.perf_counter_ns()

    encode_start = time.perf_counter_ns()
    encoded_images, payload_bytes = encode_images(
        resized_images,
        codec=codec,
        quality=quality,
        blosc_clevel=blosc_clevel,
        helpers=helpers,
    )
    encode_end = time.perf_counter_ns()

    resize_ms = ns_to_ms(resize_end - resize_start)
    encode_ms = 0.0 if codec == "raw" else ns_to_ms(encode_end - encode_start)
    quality_value = quality if codec in {"turbojpeg", "opencv_jpeg"} else None
    payload_frames: list[bytes] = []
    image_metadata: dict[str, dict[str, Any]] = {}
    for name in CAMERA_NAMES:
        item = encoded_images[name]
        payload_frames.append(item["payload"])
        image_metadata[name] = {
            "shape": item["shape"],
            "dtype": item["dtype"],
            "payload_index": len(payload_frames) - 1,
            "payload_bytes": len(item["payload"]),
        }

    request = make_request(
        seq=seq,
        codec=codec,
        quality=quality_value,
        blosc_clevel=blosc_clevel,
        image_metadata=image_metadata,
        resize_ms=resize_ms,
        encode_ms=encode_ms,
    )

    send_start = time.perf_counter_ns()
    socket.send_multipart([json.dumps(request).encode("utf-8"), *payload_frames])
    reply = json.loads(socket.recv().decode("utf-8"))
    send_end = time.perf_counter_ns()

    if not reply.get("ok", False):
        raise RuntimeError(f"Receiver returned error for seq={seq}: {reply.get('error')}")
    if reply.get("seq") != seq:
        raise RuntimeError(f"Receiver sequence mismatch: expected {seq}, got {reply.get('seq')}")

    decode_ms = 0.0 if codec == "raw" else float(reply.get("decode_ms", 0.0))
    comm_ms = ns_to_ms(send_end - send_start) - decode_ms
    if comm_ms < 0:
        comm_ms = 0.0

    return IterationStats(
        codec=codec,
        quality=quality_value,
        payload_bytes=payload_bytes,
        resize_ms=resize_ms,
        encode_ms=encode_ms,
        comm_ms=comm_ms,
        decode_ms=decode_ms,
        total_ms=encode_ms + comm_ms + decode_ms,
    )


def ns_to_ms(value_ns: int) -> float:
    return value_ns / 1_000_000.0


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * ratio))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def print_summary(codec: str, stats: list[IterationStats]) -> None:
    if not stats:
        return

    payload_kb = statistics.fmean(item.payload_bytes for item in stats) / 1024.0
    quality = stats[0].quality if stats[0].quality is not None else "-"

    print(f"\n[{codec}] measured frames={len(stats)} quality={quality} avg_payload={payload_kb:.1f} KiB")
    print(
        "metric           mean_ms    p50_ms    p95_ms    min_ms    max_ms\n"
        "----------------------------------------------------------------"
    )
    for label, values in (
        ("resize", [item.resize_ms for item in stats]),
        ("encode", [item.encode_ms for item in stats]),
        ("comm", [item.comm_ms for item in stats]),
        ("decode", [item.decode_ms for item in stats]),
        ("total", [item.total_ms for item in stats]),
    ):
        summary = summarize(values)
        print(
            f"{label:<12}"
            f"{summary['mean']:>10.3f}"
            f"{summary['p50']:>10.3f}"
            f"{summary['p95']:>10.3f}"
            f"{summary['min']:>10.3f}"
            f"{summary['max']:>10.3f}"
        )


def save_csv(path: str, stats: list[IterationStats]) -> None:
    lines = [
        "codec,quality,payload_bytes,resize_ms,encode_ms,comm_ms,decode_ms,total_ms",
    ]
    for item in stats:
        quality = "" if item.quality is None else str(item.quality)
        lines.append(
            f"{item.codec},{quality},{item.payload_bytes},"
            f"{item.resize_ms:.6f},{item.encode_ms:.6f},{item.comm_ms:.6f},"
            f"{item.decode_ms:.6f},{item.total_ms:.6f}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_helpers(codecs: list[str]) -> dict[str, Any]:
    helpers: dict[str, Any] = {}
    if "turbojpeg" in codecs:
        helpers["turbojpeg"] = require_turbojpeg()
    if "blosc_zstd" in codecs:
        helpers["blosc_zstd"] = require_blosc()
    return helpers


def main() -> None:
    args = parse_args()
    codecs = list(CODECS) if args.codec == "all" else [args.codec]
    helpers = build_helpers(codecs)

    source_images = load_images(
        {
            "head": args.head_image,
            "left": args.left_image,
            "right": args.right_image,
        }
    )

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    endpoint = f"tcp://{args.server_host}:{args.server_port}"
    socket.connect(endpoint)
    print(f"Connected to {endpoint}")

    all_stats: list[IterationStats] = []
    seq = 0
    try:
        for codec in codecs:
            print(f"\nRunning codec={codec} warmup={args.warmup} iterations={args.iterations}")
            measured: list[IterationStats] = []
            for index in range(args.warmup + args.iterations):
                seq += 1
                item = run_iteration(
                    socket=socket,
                    source_images=source_images,
                    seq=seq,
                    codec=codec,
                    quality=args.quality,
                    blosc_clevel=args.blosc_clevel,
                    helpers=helpers,
                )
                if index >= args.warmup:
                    measured.append(item)
                    all_stats.append(item)
            print_summary(codec, measured)
    finally:
        socket.close()
        context.term()

    if args.save_csv:
        save_csv(args.save_csv, all_stats)
        print(f"\nSaved per-iteration CSV to {args.save_csv}")


if __name__ == "__main__":
    main()
