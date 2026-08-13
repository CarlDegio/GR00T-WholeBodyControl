"""Best-effort visualization transport into the read-only SensorGateway."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import zmq

from gear_sonic.camera.constants import PRODUCTION_JPEG_QUALITY

VISUALIZATION_SCHEMA = "sonic.visualization_frame"
VISUALIZATION_STREAMS = (
    "visualization/navdp_navigation",
    "visualization/navdp_head_rgbd",
    "visualization/lingbot_depth",
)


class VisualizationPublisher:
    """Non-blocking JPEG publisher; display transport never stalls control."""

    def __init__(
        self, endpoint: str, *, jpeg_quality: int = PRODUCTION_JPEG_QUALITY
    ) -> None:
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self.endpoint = endpoint
        self.jpeg_quality = int(jpeg_quality)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)
        self.socket.connect(endpoint)
        self.sequence = 0

    def publish(self, stream: str, frame_bgr: np.ndarray) -> bool:
        if stream not in VISUALIZATION_STREAMS:
            raise ValueError(f"unsupported visualization stream: {stream}")
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("visualization frame must be uint8 HxWx3 BGR")
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            return False
        metadata = {
            "type": VISUALIZATION_SCHEMA,
            "version": 1,
            "stream": stream,
            "sequence": self.sequence,
            "timestamp_ns": time.monotonic_ns(),
            "shape": list(frame.shape),
            "encoding": "jpeg",
        }
        self.sequence += 1
        try:
            self.socket.send_multipart(
                [json.dumps(metadata, separators=(",", ":")).encode(), encoded.tobytes()],
                flags=zmq.DONTWAIT,
            )
            return True
        except zmq.Again:
            return False

    def close(self) -> None:
        self.socket.close()
        self.context.term()
