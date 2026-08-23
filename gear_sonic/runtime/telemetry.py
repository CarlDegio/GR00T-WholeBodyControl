"""Function-only, non-blocking runtime events and performance metrics."""

from __future__ import annotations

from collections import deque
from logging.handlers import RotatingFileHandler
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import zmq

from gear_sonic.runtime.zmq_sockets import connect_push

EVENT_SCHEMA = "sonic.runtime_event"
METRICS_SCHEMA = "sonic.runtime_metrics"
SCHEMA_VERSION = 1
VLA_TIMING_SEGMENTS = (
    "frame_age",
    "camera_read",
    "state_read",
    "jpeg_prepare",
    "observation_build",
    "request_pack",
    "policy_roundtrip",
    "response_unpack",
    "action_postprocess",
    "worker_total",
    "action_ready",
)
LAVIRA_TIMING_SEGMENTS = (
    "camera_rgbd",
    "image_encode",
    "api_inference",
    "postprocess",
    "total",
)
NAVDP_TIMING_SEGMENTS = (
    "policy_inference",
    "mpc_solve",
    "trajectory_age",
    "odometry_age",
    "control_loop",
)
BASE_POSE_TIMING_SEGMENTS = ("worker_to_control", "control_update")
RUNTIME_METRIC_SEGMENTS = {
    "vla": VLA_TIMING_SEGMENTS,
    "lavira": LAVIRA_TIMING_SEGMENTS,
    "navdp": NAVDP_TIMING_SEGMENTS,
    "base_pose": BASE_POSE_TIMING_SEGMENTS,
}
_LEVEL_COLORS = dict(DEBUG=34, INFO=32, WARNING=33, ERROR=31, CRITICAL=31)
_COMPONENT_LABELS = dict(
    base_pose="BASEPOSE", control_gateway="CONTROL", planner_executor="PLANNER"
)


def default_inference_log_dir() -> Path:
    """Return the shared directory used by inference logs and audit artifacts."""

    return Path(__file__).resolve().parents[2] / "outputs" / "logs" / "inference"


def configure_file_logging(
    component: str,
    *,
    log_dir: str | Path | None = None,
) -> logging.Logger:
    """Return one bounded, file-only logger for an inference component."""
    logger = logging.getLogger(f"sonic.{component}")
    if logger.handlers:
        return logger
    directory = (
        Path(log_dir) if log_dir is not None else default_inference_log_dir()
    )
    directory.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        directory / f"{component}.log", maxBytes=10 << 20, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def open_telemetry_publisher(endpoint: str, *, high_water_mark: int = 4) -> zmq.Socket:
    return connect_push(
        zmq.Context.instance(), endpoint, high_water_mark=high_water_mark,
        immediate=True, linger_ms=0,
    )


def build_event(
    component: str,
    level: int,
    code: str,
    message: str,
    **fields: object,
) -> dict[str, object]:
    return {
        "type": EVENT_SCHEMA,
        "version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "level": logging.getLevelName(int(level)),
        "component": component,
        "code": code,
        "message": message,
        "fields": {key: value for key, value in fields.items() if value is not None},
    }


def format_event(payload: Mapping[str, object], *, color: bool = False) -> str:
    if payload.get("type") != EVENT_SCHEMA or payload.get("version") != SCHEMA_VERSION:
        raise ValueError("invalid runtime event message")
    level = str(payload["level"]).upper()
    component = str(payload["component"])
    source = _COMPONENT_LABELS.get(component, component.upper())
    fields = dict(payload.get("fields", {}))
    details = " ".join(
        f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
        for key, value in sorted(fields.items())
    )
    if color:
        level_tag = f"\033[1;{_LEVEL_COLORS.get(level, 37)}m[{level}]\033[0m"
        source_tag = f"\033[1;36m[{source}]\033[0m"
    else:
        level_tag, source_tag = f"[{level}]", f"[{source}]"
    return f"{level_tag}{source_tag} {payload['code']} — {payload['message']}" + (
        f" | {details}" if details else ""
    )


def emit_event(
    payload: Mapping[str, object],
    *,
    socket: zmq.Socket | None = None,
    display: bool = False,
    logger: logging.Logger | None = None,
) -> bool:
    line = format_event(payload, color=display and sys.stdout.isatty())
    if logger is not None:
        level = getattr(logging, str(payload["level"]).upper(), logging.INFO)
        logger.log(
            level,
            "%s | %s | %s",
            payload["code"],
            payload["message"],
            payload.get("fields") or "",
        )
    if display:
        print(line, flush=True)
    if socket is None:
        return True
    try:
        socket.send_json(dict(payload), flags=zmq.DONTWAIT)
        return True
    except zmq.ZMQError:
        return False


def _valid_metrics(values: Mapping[str, float], names: Sequence[str]) -> dict[str, float]:
    allowed = frozenset(names)
    result = {name: float(value) for name, value in values.items() if name in allowed}
    if not result:
        raise ValueError("runtime metric sample has no recognized values")
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("runtime metric values must be finite and non-negative")
    return result


def publish_metrics(
    socket: zmq.Socket,
    component: str,
    values: Mapping[str, float],
    *,
    allowed_names: Sequence[str],
    activate: bool = False,
) -> bool:
    if activate and values:
        raise ValueError("runtime metric activation cannot contain values")
    payload = {
        "type": METRICS_SCHEMA,
        "version": SCHEMA_VERSION,
        "component": component,
        "timestamp_ns": time.monotonic_ns(),
        "values": {} if activate else _valid_metrics(values, allowed_names),
        "activate": bool(activate),
    }
    try:
        socket.send_json(payload, flags=zmq.DONTWAIT)
        return True
    except zmq.ZMQError:
        return False


def create_metrics_window(names: Sequence[str], window_size: int = 100) -> dict:
    if window_size < 2:
        raise ValueError("metrics window_size must be at least two")
    return {
        "size": int(window_size),
        "count": 0,
        "received_ns": None,
        "samples": {name: deque(maxlen=int(window_size)) for name in names},
    }


def record_metrics(window: dict, values: Mapping[str, float], received_ns: int) -> None:
    for name, value in _valid_metrics(values, tuple(window["samples"])).items():
        window["samples"][name].append(value)
    window["count"] += 1
    window["received_ns"] = int(received_ns)


def _percentile(values: deque[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def metrics_snapshot(window: dict, *, now_ns: int | None = None) -> dict:
    now = time.monotonic_ns() if now_ns is None else int(now_ns)
    received = window["received_ns"]
    return {
        "sample_count": window["count"],
        "last_sample_age_ms": (
            None if received is None else max(0, now - received) / 1_000_000.0
        ),
        "window_size": window["size"],
        "values": {
            name: {
                "last": values[-1] if values else None,
                "mean": sum(values) / len(values) if values else None,
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "count": len(values),
            }
            for name, values in window["samples"].items()
        },
    }


def poll_metrics(
    socket: zmq.Socket, windows: Mapping[str, dict]
) -> tuple[str, int] | None:
    if not socket.poll(0, zmq.POLLIN):
        return None
    received_ns = time.monotonic_ns()
    payload = socket.recv_json()
    if payload.get("type") != METRICS_SCHEMA or payload.get("version") != SCHEMA_VERSION:
        raise ValueError("invalid runtime metrics message")
    component = str(payload.get("component", ""))
    if component not in windows:
        raise ValueError(f"unknown runtime metrics component: {component}")
    activate = payload.get("activate", False)
    values = payload.get("values")
    if not isinstance(activate, bool) or not isinstance(values, Mapping):
        raise ValueError("invalid runtime metrics message")
    if values:
        record_metrics(windows[component], values, received_ns)
    elif not activate:
        raise ValueError("runtime metric sample has no recognized values")
    return component, received_ns
