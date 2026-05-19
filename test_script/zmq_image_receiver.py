#!/usr/bin/env python3
"""Receive and decode three resized camera images over ZMQ."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import cv2
import numpy as np
import zmq


CODECS = ("turbojpeg", "opencv_jpeg", "blosc_zstd", "raw")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bind a ZMQ REP socket, receive image benchmark requests, decode them, "
            "and reply with decode timing."
        )
    )
    parser.add_argument("--bind-host", default="127.0.0.1", help="Host/IP to bind")
    parser.add_argument("--bind-port", type=int, default=29999, help="Port to bind")
    parser.add_argument(
        "--timeout-ms",
        type=positive_int,
        default=0,
        help="Receive timeout in milliseconds, 0 means wait forever",
    )
    parser.add_argument(
        "--report-interval",
        type=positive_int,
        default=50,
        help="Print one receiver status line every N messages",
    )
    return parser.parse_args()


def require_turbojpeg() -> Any:
    try:
        import turbojpeg
    except ImportError as exc:
        raise RuntimeError(
            "Received turbojpeg payload but the turbojpeg/PyTurboJPEG package is not installed"
        ) from exc

    if hasattr(turbojpeg, "TurboJPEG"):
        return ("py_turbojpeg", turbojpeg.TurboJPEG())
    if hasattr(turbojpeg, "decompress"):
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
        raise RuntimeError("Received blosc_zstd payload but neither blosc nor blosc2 is installed") from exc


def turbojpeg_decode(helper: Any, payload: bytes) -> np.ndarray:
    api_name, turbo = helper
    if api_name == "py_turbojpeg":
        return turbo.decode(payload)
    return turbo.decompress(payload)


def blosc_decompress(helper: Any, payload: bytes) -> bytes:
    api_name, module = helper
    if api_name == "blosc":
        return module.decompress(payload)
    return module.decompress2(payload)


def decode_image(
    codec: str,
    item: dict[str, Any],
    payload_frames: list[bytes],
    helpers: dict[str, Any],
) -> np.ndarray:
    payload = payload_frames[item["payload_index"]]
    shape = tuple(item["shape"])
    dtype = np.dtype(item["dtype"])

    if codec == "turbojpeg":
        if "turbojpeg" not in helpers:
            helpers["turbojpeg"] = require_turbojpeg()
        image = turbojpeg_decode(helpers["turbojpeg"], payload)
        return np.asarray(image)

    if codec == "opencv_jpeg":
        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("cv2.imdecode failed")
        return image

    if codec == "blosc_zstd":
        if "blosc_zstd" not in helpers:
            helpers["blosc_zstd"] = require_blosc()
        raw = blosc_decompress(helpers["blosc_zstd"], payload)
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    if codec == "raw":
        return np.frombuffer(payload, dtype=dtype).reshape(shape)

    raise ValueError(f"Unsupported codec: {codec}")


def ns_to_ms(value_ns: int) -> float:
    return value_ns / 1_000_000.0


def handle_request(
    request: dict[str, Any],
    payload_frames: list[bytes],
    helpers: dict[str, Any],
) -> dict[str, Any]:
    seq = request.get("seq")
    codec = request.get("codec")
    if codec not in CODECS:
        raise ValueError(f"Unsupported codec: {codec}")

    if codec == "raw":
        payload_bytes = sum(len(payload) for payload in payload_frames)
        return {
            "seq": seq,
            "ok": True,
            "codec": codec,
            "decode_ms": 0.0,
            "payload_bytes": payload_bytes,
            "image_shapes": {
                name: item.get("shape", [])
                for name, item in request.get("images", {}).items()
            },
        }

    decode_start = time.perf_counter_ns()
    decoded: dict[str, np.ndarray] = {}
    for name, item in request.get("images", {}).items():
        decoded[name] = decode_image(codec, item, payload_frames, helpers)
    decode_end = time.perf_counter_ns()

    decode_ms = 0.0 if codec == "raw" else ns_to_ms(decode_end - decode_start)
    payload_bytes = sum(len(payload) for payload in payload_frames)
    return {
        "seq": seq,
        "ok": True,
        "codec": codec,
        "decode_ms": decode_ms,
        "payload_bytes": payload_bytes,
        "image_shapes": {name: list(image.shape) for name, image in decoded.items()},
    }


def main() -> None:
    args = parse_args()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    if args.timeout_ms > 0:
        socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    endpoint = f"tcp://{args.bind_host}:{args.bind_port}"
    socket.bind(endpoint)
    print(f"Receiver listening on {endpoint}")

    helpers: dict[str, Any] = {}
    count = 0
    try:
        while True:
            try:
                frames = socket.recv_multipart()
                request = json.loads(frames[0].decode("utf-8"))
                reply = handle_request(request, frames[1:], helpers)
            except zmq.Again:
                continue
            except Exception as exc:
                seq = None
                try:
                    seq = request.get("seq")  # type: ignore[name-defined]
                except Exception:
                    pass
                reply = {"seq": seq, "ok": False, "error": str(exc)}

            socket.send(json.dumps(reply).encode("utf-8"))
            count += 1
            if count % args.report_interval == 0:
                print(
                    f"received={count} codec={reply.get('codec')} "
                    f"payload={reply.get('payload_bytes', 0) / 1024.0:.1f} KiB "
                    f"decode_ms={reply.get('decode_ms', 0.0):.3f} ok={reply.get('ok')}"
                )
    except KeyboardInterrupt:
        print("\nReceiver stopped")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
