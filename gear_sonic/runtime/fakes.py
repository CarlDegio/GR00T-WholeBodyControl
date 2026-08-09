"""Protocol-compatible fake services used by local integration tests."""

from __future__ import annotations

import io
import threading
import time
from typing import Any, Callable, Mapping

import msgpack
import numpy as np
import zmq


def _encode_policy_object(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        output = io.BytesIO()
        np.save(output, value, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}
    raise TypeError(f"unsupported fake policy value: {type(value).__name__}")


def _decode_policy_object(value: Any) -> Any:
    if isinstance(value, dict) and "__ndarray_class__" in value:
        return np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
    return value


def pack_policy_message(value: Any) -> bytes:
    return msgpack.packb(value, default=_encode_policy_object, use_bin_type=True)


def unpack_policy_message(payload: bytes) -> Any:
    return msgpack.unpackb(payload, object_hook=_decode_policy_object, raw=False)


class FakePolicyServer:
    """Small compatible subset of the OpenPI REQ/REP policy server."""

    def __init__(
        self,
        context: zmq.Context,
        endpoint: str,
        *,
        action_factory: Callable[[Mapping[str, Any]], Any] | None = None,
        response_delay_s: float = 0.0,
    ) -> None:
        self.context = context
        self.endpoint = endpoint
        self.action_factory = action_factory or self._default_action
        self.response_delay_s = float(response_delay_s)
        if self.response_delay_s < 0.0:
            raise ValueError("response_delay_s cannot be negative")
        self.request_count = 0
        self.last_endpoint = ""
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _default_action(_request: Mapping[str, Any]) -> Any:
        return [
            {"action": np.zeros((1, 1, 29), dtype=np.float32)},
            {"source": "fake_policy"},
        ]

    def start(self, timeout_s: float = 2.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("fake policy server is already running")
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise TimeoutError("fake policy server did not become ready")

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("fake policy server did not stop")
        self._thread = None

    def _handle(self, request: Mapping[str, Any]) -> Any:
        endpoint = str(request.get("endpoint", "get_action"))
        self.last_endpoint = endpoint
        self.request_count += 1
        if endpoint == "ping":
            return {"status": "ok", "message": "Fake PolicyServer is running"}
        if endpoint == "reset":
            return {"status": "ok"}
        if endpoint == "get_modality_config":
            return {}
        if endpoint == "get_action":
            return self.action_factory(request.get("data", {}))
        return {"error": f"Unknown endpoint: {endpoint}"}

    def _run(self) -> None:
        socket = self.context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.endpoint)
        self._ready.set()
        try:
            while not self._stop.is_set():
                if not socket.poll(20, zmq.POLLIN):
                    continue
                request = unpack_policy_message(socket.recv())
                response = self._handle(request)
                if self.response_delay_s > 0.0 and self._stop.wait(self.response_delay_s):
                    break
                socket.send(pack_policy_message(response))
        finally:
            socket.close(linger=0)


class FakeCameraServer:
    """Camera PUB endpoint that accepts the existing serialized camera dictionary."""

    def __init__(self, context: zmq.Context, endpoint: str) -> None:
        self.socket = context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(endpoint)
        self.message_count = 0

    def publish(self, serialized_frame: Mapping[str, Any]) -> None:
        self.socket.send(msgpack.packb(dict(serialized_frame), use_bin_type=True))
        self.message_count += 1

    def close(self) -> None:
        self.socket.close(linger=0)


class FakeCppService:
    """C++-side state publisher and raw command receiver."""

    def __init__(
        self,
        context: zmq.Context,
        *,
        state_endpoint: str,
        command_endpoint: str,
    ) -> None:
        self.state_socket = context.socket(zmq.PUB)
        self.state_socket.setsockopt(zmq.LINGER, 0)
        self.state_socket.bind(state_endpoint)
        self.command_socket = context.socket(zmq.SUB)
        self.command_socket.setsockopt(zmq.LINGER, 0)
        for topic in (b"command", b"planner", b"pose"):
            self.command_socket.setsockopt(zmq.SUBSCRIBE, topic)
        self.command_socket.connect(command_endpoint)
        self.state_message_count = 0

    def publish_state(self, state: Mapping[str, Any], *, topic: str = "g1_debug") -> None:
        payload = msgpack.packb(dict(state), use_bin_type=True)
        self.state_socket.send(topic.encode("utf-8") + payload)
        self.state_message_count += 1

    def receive_command(self, timeout_ms: int = 0) -> bytes | None:
        if self.command_socket.poll(timeout_ms, zmq.POLLIN):
            return self.command_socket.recv()
        return None

    def close(self) -> None:
        self.command_socket.close(linger=0)
        self.state_socket.close(linger=0)
