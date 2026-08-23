"""Explicit ZeroMQ socket constructors shared by runtime services."""

from __future__ import annotations

from typing import Any

import zmq


def _open(
    context: zmq.Context, kind: int, endpoint: str, *, bind: bool,
    receive: bool = False, topic: str | bytes | None = None,
    conflate: bool = False, high_water_mark: int | None = None,
    receive_timeout_ms: int | None = None, immediate: bool = False,
    linger_ms: int | None = None,
) -> zmq.Socket:
    socket = context.socket(kind)
    if topic is not None:
        if isinstance(topic, bytes):
            socket.setsockopt(zmq.SUBSCRIBE, topic)
        else:
            socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    if conflate:
        socket.setsockopt(zmq.CONFLATE, 1)
    if high_water_mark is not None:
        socket.setsockopt(zmq.RCVHWM if receive else zmq.SNDHWM, high_water_mark)
    if receive_timeout_ms is not None:
        socket.setsockopt(zmq.RCVTIMEO, receive_timeout_ms)
    if immediate:
        socket.setsockopt(zmq.IMMEDIATE, 1)
    if linger_ms is not None:
        socket.setsockopt(zmq.LINGER, linger_ms)
    (socket.bind if bind else socket.connect)(endpoint)
    return socket


def connect_subscriber(
    context: zmq.Context, endpoint: str, *, topic: str | bytes = "", **options: Any,
) -> zmq.Socket:
    return _open(
        context, zmq.SUB, endpoint, bind=False, receive=True,
        topic=topic, **options,
    )


def connect_push(
    context: zmq.Context, endpoint: str, **options: Any,
) -> zmq.Socket:
    return _open(context, zmq.PUSH, endpoint, bind=False, **options)


def bind_publisher(
    context: zmq.Context, endpoint: str, **options: Any,
) -> zmq.Socket:
    return _open(context, zmq.PUB, endpoint, bind=True, **options)


def bind_pull(
    context: zmq.Context, endpoint: str, **options: Any,
) -> zmq.Socket:
    return _open(context, zmq.PULL, endpoint, bind=True, receive=True, **options)
