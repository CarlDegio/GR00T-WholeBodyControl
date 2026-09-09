from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest
import zmq

from gear_sonic.runtime.telemetry import (
    BASE_POSE_TIMING_SEGMENTS,
    LAVIRA_TIMING_SEGMENTS,
    NAVDP_TIMING_SEGMENTS,
    RUNTIME_METRIC_SEGMENTS,
    VLA_TIMING_SEGMENTS,
    build_event,
    configure_file_logging,
    create_metrics_window,
    emit_event,
    format_event,
    metrics_snapshot,
    poll_metrics,
    publish_metrics,
    record_metrics,
)


def test_timing_uses_jpeg_prepare() -> None:
    window = create_metrics_window(VLA_TIMING_SEGMENTS, window_size=2)
    record_metrics(
        window,
        {"jpeg_prepare": 0.05, "worker_total": 1.0, "action_ready": 1.2},
        received_ns=1,
    )

    snapshot = metrics_snapshot(window)

    assert snapshot["values"]["jpeg_prepare"]["last"] == 0.05
    assert snapshot["values"]["action_ready"]["last"] == 1.2
    assert "jpeg_encode" not in snapshot["values"]


def test_vla_timing_window_reports_fixed_rolling_statistics() -> None:
    window = create_metrics_window(VLA_TIMING_SEGMENTS, window_size=3)
    record_metrics(window, {"camera_read": 1.0, "worker_total": 10.0}, 1_000_000)
    record_metrics(window, {"camera_read": 3.0, "worker_total": 20.0}, 2_000_000)
    record_metrics(window, {"camera_read": 5.0, "worker_total": 30.0}, 3_000_000)

    snapshot = metrics_snapshot(window, now_ns=4_000_000)
    camera = snapshot["values"]["camera_read"]

    assert snapshot["sample_count"] == 3
    assert snapshot["last_sample_age_ms"] == 1.0
    assert camera == {
        "last": 5.0,
        "mean": 3.0,
        "p50": 3.0,
        "p95": pytest.approx(4.8),
        "count": 3,
    }


def test_vla_timing_rejects_negative_measurements() -> None:
    window = create_metrics_window(VLA_TIMING_SEGMENTS)
    with pytest.raises(ValueError, match="finite and non-negative"):
        record_metrics(window, {"worker_total": -1.0}, received_ns=1)


def test_runtime_event_has_one_stable_pane_format() -> None:
    event = build_event(
        "navdp",
        logging.WARNING,
        "MPC_TIMEOUT",
        "solve timed out",
        elapsed_ms=83,
    )

    line = format_event(event)

    assert line == "[WARNING][NAVDP] MPC_TIMEOUT — solve timed out | elapsed_ms=83"
    assert format_event(event, color=True).startswith(
        "\033[1;33m[WARNING]\033[0m\033[1;36m[NAVDP]\033[0m"
    )
    socket = Mock()
    logger = Mock()
    assert emit_event(event, socket=socket, logger=logger)
    assert socket.send_json.call_args.kwargs["flags"] == zmq.DONTWAIT
    logger.log.assert_called_once()


def test_component_file_logging_is_bounded_and_idempotent(tmp_path) -> None:
    component = f"test_{tmp_path.name}"
    logger = configure_file_logging(component, log_dir=tmp_path)

    assert configure_file_logging(component, log_dir=tmp_path) is logger
    assert len(logger.handlers) == 1
    logger.info("ready")
    logger.handlers[0].flush()

    assert "ready" in (tmp_path / f"{component}.log").read_text()


@pytest.mark.parametrize("already_suppressed", [False, True])
def test_component_can_keep_file_logging_in_minimal_experiments(
    tmp_path, monkeypatch, already_suppressed,
) -> None:
    monkeypatch.setenv("SONIC_EXPERIMENT_MINIMAL_LOGGING", "1")
    component = f"test_base_pose_{tmp_path.name}"
    other_component = f"test_navdp_{tmp_path.name}"
    other = configure_file_logging(other_component, log_dir=tmp_path)
    if already_suppressed:
        configure_file_logging(component, log_dir=tmp_path)
    logger = configure_file_logging(component, log_dir=tmp_path, force=True)
    try:
        logger.info("pulse stopped; settling 0.5 s")
        other.info("still minimal")
        assert configure_file_logging(
            component, log_dir=tmp_path, force=True,
        ) is logger
        assert len(logger.handlers) == 1
        logger.handlers[0].flush()
        assert "pulse stopped; settling 0.5 s" in (
            tmp_path / f"{component}.log"
        ).read_text()
        assert not (tmp_path / f"{other_component}.log").exists()
    finally:
        for target in (logger, other):
            for handler in tuple(target.handlers):
                target.removeHandler(handler)
                handler.close()


def test_metrics_publish_is_non_blocking_and_drops_when_unavailable() -> None:
    socket = Mock()
    socket.send_json.side_effect = zmq.Again()

    sent = publish_metrics(
        socket,
        "vla",
        {"worker_total": 12.0},
        allowed_names=VLA_TIMING_SEGMENTS,
    )

    assert not sent
    assert socket.send_json.call_args.kwargs["flags"] == zmq.DONTWAIT


def _metric_message(component: str, values: dict, *, activate: bool = False) -> dict:
    return {
        "type": "sonic.runtime_metrics",
        "version": 1,
        "component": component,
        "values": values,
        "activate": activate,
    }


def test_metrics_activate_and_route_components_without_cross_contamination() -> None:
    sender = Mock()
    publish_metrics(
        sender, "lavira", {}, allowed_names=LAVIRA_TIMING_SEGMENTS, activate=True
    )
    windows = {
        component: create_metrics_window(names)
        for component, names in RUNTIME_METRIC_SEGMENTS.items()
    }
    receiver = Mock()
    receiver.poll.return_value = True
    receiver.recv_json.side_effect = [
        sender.send_json.call_args.args[0],
        _metric_message("vla", {"worker_total": 12.0}),
        _metric_message("lavira", {"total": 250.0}),
        _metric_message("navdp", {"mpc_solve": 8.0, "odometry_age": 4.0}),
        _metric_message(
            "base_pose",
            {"worker_to_control": 3.0, "control_update": 1.0},
        ),
    ]

    assert poll_metrics(receiver, windows)[0] == "lavira"
    assert metrics_snapshot(windows["lavira"])["sample_count"] == 0
    assert poll_metrics(receiver, windows)[0] == "vla"
    assert poll_metrics(receiver, windows)[0] == "lavira"
    assert poll_metrics(receiver, windows)[0] == "navdp"
    assert poll_metrics(receiver, windows)[0] == "base_pose"

    assert metrics_snapshot(windows["vla"])["sample_count"] == 1
    assert metrics_snapshot(windows["lavira"])["sample_count"] == 1
    assert metrics_snapshot(windows["lavira"])["values"]["total"]["last"] == 250.0
    navdp = metrics_snapshot(windows["navdp"])
    assert tuple(navdp["values"]) == NAVDP_TIMING_SEGMENTS
    assert navdp["values"]["mpc_solve"]["last"] == 8.0
    base_pose = metrics_snapshot(windows["base_pose"])
    assert tuple(base_pose["values"]) == BASE_POSE_TIMING_SEGMENTS
    assert base_pose["values"]["worker_to_control"]["last"] == 3.0


def test_metrics_reject_unknown_component() -> None:
    receiver = Mock()
    receiver.poll.return_value = True
    receiver.recv_json.return_value = _metric_message("unknown", {"total": 1.0})

    with pytest.raises(ValueError, match="unknown runtime metrics component"):
        poll_metrics(receiver, {})
