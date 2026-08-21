from __future__ import annotations

import json
import time

import cv2
import numpy as np
import zmq

from gear_sonic.runtime.gateway.sensor import SensorGatewayCore, VisualizationZmqIngress
from gear_sonic.runtime.gateway.shared_memory import read_shared_memory_frame
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest
from gear_sonic.runtime.gateway.visualization import (
    NAVDP_ACTOR_RAY_STREAM,
    NAVDP_SLAM_2D_STREAM,
    VISUALIZATION_SCHEMA,
    VISUALIZATION_STREAMS,
    VisualizationPublisher,
)


def test_visualization_contract_exposes_split_navdp_panels() -> None:
    assert NAVDP_ACTOR_RAY_STREAM in VISUALIZATION_STREAMS
    assert NAVDP_SLAM_2D_STREAM in VISUALIZATION_STREAMS


def test_visualization_ingress_stores_jpeg_in_sensor_gateway_shared_memory() -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=2, history_size=4)
    endpoint = "inproc://visualization-test"
    ingress = VisualizationZmqIngress(context, endpoint, core)
    sender = context.socket(zmq.PUSH)
    sender.connect(endpoint)
    image = np.full((40, 80, 3), (10, 80, 180), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    metadata = {
        "type": VISUALIZATION_SCHEMA,
        "version": 1,
        "stream": NAVDP_ACTOR_RAY_STREAM,
        "sequence": 4,
        "timestamp_ns": time.monotonic_ns(),
        "shape": list(image.shape),
        "encoding": "jpeg",
    }

    sender.send_multipart([json.dumps(metadata).encode(), encoded.tobytes()])
    assert ingress.poll_once(timeout_ms=100)
    snapshot = core.select(
        SnapshotRequest(
            streams=(NAVDP_ACTOR_RAY_STREAM,),
            max_age_ms=1000,
            max_skew_ms=0,
        )
    )
    frame = snapshot.frames[NAVDP_ACTOR_RAY_STREAM]
    restored = cv2.imdecode(read_shared_memory_frame(frame), cv2.IMREAD_COLOR)

    assert snapshot.complete
    assert restored.shape == image.shape
    assert frame.attributes["encoding"] == "jpeg"

    sender.close()
    ingress.close()
    core.close()
    context.term()


def test_visualization_default_passes_quality_95_to_jpeg_encoder(monkeypatch) -> None:
    """Catch visualization output using a lower default JPEG quality."""
    imencode_calls = []

    def capture_imencode(extension, image, parameters):
        imencode_calls.append((extension, parameters))
        return True, np.array([0], dtype=np.uint8)

    monkeypatch.setattr(cv2, "imencode", capture_imencode)
    publisher = VisualizationPublisher("inproc://visualization-quality-test")
    try:
        publisher.publish(
            NAVDP_ACTOR_RAY_STREAM, np.zeros((8, 8, 3), dtype=np.uint8)
        )
    finally:
        publisher.close()

    assert imencode_calls == [(".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95])]
