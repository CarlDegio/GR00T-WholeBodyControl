"""
ZMQ utilities for subscribing to robot state and config from the C++ deploy process.

Provides:
- ``ZMQStateSubscriber`` — non-blocking SUB on the ``g1_debug`` topic
"""

import zmq

from gear_sonic.runtime.protocol.cpp_state import decode_cpp_state_payload
from gear_sonic.runtime.zmq_sockets import connect_subscriber

STATE_ZMQ_TOPIC = "g1_debug"
DEFAULT_STATE_ZMQ_PORT = 5557


class ZMQStateSubscriber:
    """Non-blocking SUB on the ``g1_debug`` ZMQ topic for robot state.

    Uses ``zmq.CONFLATE`` so only the latest message is kept.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_STATE_ZMQ_PORT,
        topic: str = STATE_ZMQ_TOPIC,
    ):
        self._ctx = zmq.Context()
        self._socket = connect_subscriber(
            self._ctx, f"tcp://{host}:{port}", topic=topic,
            conflate=True, receive_timeout_ms=0,
        )
        self._topic = topic
        self._topic_bytes = topic.encode("utf-8")
        self._msg = None
        print(f"[ZMQStateSubscriber] Connected to tcp://{host}:{port} (topic: {topic})")

    def _poll(self):
        """Poll for latest message (non-blocking)."""
        try:
            raw = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return

        self._msg = decode_cpp_state_payload(raw[len(self._topic_bytes):])

    def get_msg(self, clear: bool = True):
        """Return the latest state message (or ``None``)."""
        self._poll()
        msg = self._msg
        if clear:
            self._msg = None
        return msg

    def close(self):
        self._socket.close()
        self._ctx.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
