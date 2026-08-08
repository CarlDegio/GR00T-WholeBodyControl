from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import zmq

from gear_sonic.runtime.client import (
    SensorGatewayClient,
    SensorGatewayClientError,
    SensorGatewayTimeoutError,
    SnapshotUnavailableError,
)
from gear_sonic.runtime.sensor_gateway import SensorGatewayCore, SensorGatewayRpc
from gear_sonic.runtime.shared_memory import FrameOverwrittenError
from gear_sonic.runtime.snapshot import SnapshotRequest


class _RpcThread:
    def __init__(self, context: zmq.Context, endpoint: str, core: SensorGatewayCore) -> None:
        self.context = context
        self.endpoint = endpoint
        self.core = core
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        rpc = SensorGatewayRpc(self.context, self.endpoint, self.core)
        self.ready.set()
        try:
            while not self.stop.is_set():
                rpc.serve_once(timeout_ms=10)
        finally:
            rpc.close()

    def start(self) -> None:
        self.thread.start()
        assert self.ready.wait(1.0)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(1.0)
        assert not self.thread.is_alive()


def test_client_reads_health_and_materializes_a_complete_snapshot() -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=8, history_size=16)
    server = _RpcThread(context, "inproc://sensor-client", core)
    server.start()
    now_ns = time.monotonic_ns()
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    odometry = np.arange(13, dtype=np.float64)
    core.publish_array("camera/chest_view", rgb, received_ns=now_ns)
    core.publish_array("ros/odometry", odometry, received_ns=now_ns)
    client = SensorGatewayClient(
        "inproc://sensor-client",
        context=context,
        request_timeout_ms=100,
    )
    try:
        assert client.ping()
        health = client.health()
        assert health["streams"]["camera/chest_view"]["state"] == "healthy"

        materialized = client.read_snapshot(
            SnapshotRequest(
                streams=("camera/chest_view", "ros/odometry"),
                max_age_ms=1000.0,
                max_skew_ms=10.0,
            )
        )

        assert materialized.attempts == 1
        np.testing.assert_array_equal(materialized.arrays["camera/chest_view"], rgb)
        np.testing.assert_array_equal(materialized.arrays["ros/odometry"], odometry)
        with pytest.raises(TypeError):
            materialized.arrays["extra"] = np.zeros(1)
    finally:
        client.close()
        server.close()
        core.close()
        context.term()


def test_client_retries_the_whole_snapshot_after_an_overwritten_frame() -> None:
    context = zmq.Context()
    core = SensorGatewayCore(slot_count=8, history_size=16)
    server = _RpcThread(context, "inproc://sensor-client-retry", core)
    server.start()
    now_ns = time.monotonic_ns()
    values = np.arange(10, dtype=np.float32)
    core.publish_array("ros/livox_imu", values, received_ns=now_ns)
    calls = 0

    def flaky_reader(frame):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FrameOverwrittenError("simulated overwrite")
        from gear_sonic.runtime.shared_memory import read_shared_memory_frame

        return read_shared_memory_frame(frame)

    client = SensorGatewayClient(
        "inproc://sensor-client-retry",
        context=context,
        request_timeout_ms=100,
        frame_reader=flaky_reader,
    )
    try:
        materialized = client.read_snapshot(
            SnapshotRequest(
                streams=("ros/livox_imu",),
                max_age_ms=1000.0,
                max_skew_ms=1.0,
            ),
            retries=1,
            retry_backoff_s=0.0,
        )

        assert materialized.attempts == 2
        assert calls == 2
        np.testing.assert_array_equal(materialized.arrays["ros/livox_imu"], values)
    finally:
        client.close()
        server.close()
        core.close()
        context.term()


def test_client_reports_missing_stream_and_recovers_req_socket_after_timeout() -> None:
    context = zmq.Context()
    missing_client = SensorGatewayClient(
        "inproc://missing-gateway",
        context=context,
        request_timeout_ms=20,
    )
    with pytest.raises(SensorGatewayTimeoutError):
        missing_client.health()
    missing_client.close()

    core = SensorGatewayCore(slot_count=8, history_size=16)
    server = _RpcThread(context, "inproc://sensor-client-missing-stream", core)
    server.start()
    client = SensorGatewayClient(
        "inproc://sensor-client-missing-stream",
        context=context,
        request_timeout_ms=100,
    )
    try:
        with pytest.raises(SnapshotUnavailableError, match="missing streams"):
            client.read_snapshot(
                SnapshotRequest(
                    streams=("camera/missing",),
                    max_age_ms=100.0,
                    max_skew_ms=10.0,
                ),
                retries=0,
            )
        client.close()
        with pytest.raises(SensorGatewayClientError, match="closed"):
            client.health()
    finally:
        server.close()
        core.close()
        context.term()
