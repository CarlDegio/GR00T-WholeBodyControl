"""
ZMQ utilities for subscribing to robot state and config from the C++ deploy process.

Provides:
- ``ZMQStateSubscriber`` — non-blocking SUB on the ``g1_debug`` topic
"""

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

STATE_ZMQ_TOPIC = "g1_debug"
DEFAULT_STATE_ZMQ_PORT = 5557


def _unpack_msgpack_zmq(raw: bytes, topic: str) -> dict:
    """Strip a ZMQ topic prefix and decode the msgpack payload."""
    payload = raw[len(topic):]
    return msgpack.unpackb(payload, raw=False)


def _convert_lists_to_numpy(data: dict) -> dict:
    """Convert list values in a dict to numpy arrays."""
    if not isinstance(data, dict):
        return data
    result = {}
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            result[key] = np.array(value)
        elif isinstance(value, dict):
            result[key] = _convert_lists_to_numpy(value)
        else:
            result[key] = value
    return result


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
        mnp.patch()
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        self._topic = topic
        self._msg = None
        print(f"[ZMQStateSubscriber] Connected to tcp://{host}:{port} (topic: {topic})")

    def _poll(self):
        """Poll for latest message (non-blocking)."""
        try:
            raw = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return

        msg = _unpack_msgpack_zmq(raw, self._topic)
        msg = _convert_lists_to_numpy(msg)
        self._msg = msg

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
