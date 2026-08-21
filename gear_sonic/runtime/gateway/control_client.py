"""Typed clients for ControlGateway intents and dispatched commands."""

from __future__ import annotations

import time
from typing import Callable, Mapping

import zmq

from gear_sonic.runtime.protocol import OperatorCommand
from gear_sonic.runtime.gateway.control import ControlGatewayCore


class ControlGatewayIntentClient:
    """Submit typed component intents to the ControlGateway PULL endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        source: str,
        context: zmq.Context | None = None,
        ttl_ms: int = 1000,
        latest_only: bool = False,
    ) -> None:
        if not endpoint.startswith(("tcp://", "inproc://")):
            raise ValueError(f"unsupported ControlGateway endpoint: {endpoint}")
        self.endpoint = endpoint
        self._owns_context = context is None
        self._context = zmq.Context() if context is None else context
        self._socket = self._context.socket(zmq.PUSH)
        if latest_only:
            # Velocity intents describe the desired state now, not a motion
            # sequence. If the gateway falls behind, retain only the newest
            # intent so a later stop can replace queued motion immediately.
            self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.setsockopt(zmq.SNDHWM, 1)
        else:
            self._socket.setsockopt(zmq.SNDHWM, 100)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._core = ControlGatewayCore(source=source, ttl_ms=ttl_ms)
        self._closed = False

    def send(self, name: str, parameters: Mapping[str, object]) -> OperatorCommand:
        event = self._core.accept_command(
            name,
            parameters=dict(parameters),
        )
        self._socket.send_string(event.command.to_json())
        return event.command

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close()
        if self._owns_context:
            self._context.term()


class ControlGatewaySubscriber:
    """Non-blocking typed-command subscriber with TTL/order validation."""

    def __init__(
        self,
        endpoint: str,
        *,
        context: zmq.Context | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        accepted_names: set[str] | None = None,
    ) -> None:
        if not endpoint.startswith("tcp://") and not endpoint.startswith("inproc://"):
            raise ValueError(f"unsupported ControlGateway endpoint: {endpoint}")
        self.endpoint = endpoint
        self._owns_context = context is None
        self._context = zmq.Context() if context is None else context
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        # Do not conflate the shared command stream: a navigation key must not
        # overwrite a preceding POSE/PLANNER transition before consumers can
        # apply their own accepted-name filter.
        self._socket.setsockopt(zmq.RCVHWM, 100)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._monotonic_ns = monotonic_ns
        self._accepted_names = None if accepted_names is None else frozenset(accepted_names)
        self._last_sequence_by_source: dict[str, int] = {}
        self._closed = False

    def decode_command(self, payload: str | bytes) -> OperatorCommand | None:
        """Return a validated typed command while retaining TTL/order checks."""

        command = OperatorCommand.from_json(payload)
        now_ns = int(self._monotonic_ns())
        if command.metadata.is_expired(now_ns):
            return None
        previous = self._last_sequence_by_source.get(command.metadata.source)
        if previous is not None and command.metadata.sequence <= previous:
            return None
        self._last_sequence_by_source[command.metadata.source] = command.metadata.sequence
        accepted_names = getattr(self, "_accepted_names", None)
        if accepted_names is not None and command.name not in accepted_names:
            return None
        return command

    def read_command(self) -> OperatorCommand | None:
        try:
            payload = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return None
        try:
            return self.decode_command(payload)
        except (KeyError, TypeError, ValueError):
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close()
        if self._owns_context:
            self._context.term()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
