from __future__ import annotations

import json
import time

import cv2
import numpy as np
import zmq

from gear_sonic.runtime.sensor_gateway import SensorGatewayCore, VisualizationZmqIngress
from gear_sonic.runtime.shared_memory import read_shared_memory_frame
from gear_sonic.runtime.snapshot import SnapshotRequest
from gear_sonic.runtime.visualization import VISUALIZATION_SCHEMA


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
        "stream": "visualization/navdp_navigation",
        "sequence": 4,
        "timestamp_ns": time.monotonic_ns(),
        "shape": list(image.shape),
        "encoding": "jpeg",
    }

    sender.send_multipart([json.dumps(metadata).encode(), encoded.tobytes()])
    assert ingress.poll_once(timeout_ms=100)
    snapshot = core.select(
        SnapshotRequest(
            streams=("visualization/navdp_navigation",),
            max_age_ms=1000,
            max_skew_ms=0,
        )
    )
    frame = snapshot.frames["visualization/navdp_navigation"]
    restored = cv2.imdecode(read_shared_memory_frame(frame), cv2.IMREAD_COLOR)

    assert snapshot.complete
    assert restored.shape == image.shape
    assert frame.attributes["encoding"] == "jpeg"

    sender.close()
    ingress.close()
    core.close()
    context.term()
