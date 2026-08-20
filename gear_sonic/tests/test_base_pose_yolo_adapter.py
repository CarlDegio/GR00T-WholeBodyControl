from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from gear_sonic.scripts.base_pose_agent import BasePoseAgentConfig
from gear_sonic.scripts.base_pose_yolo_agent import GatewayRawServoAdapter
from gear_sonic.utils.inference.base_pose_visual_servo import RawServoEvent
from gear_sonic.utils.inference.base_pose_visual_servo import (
    RawServoCalibration,
    RawServoObservation,
    ServoCommand,
    ServoPhase,
    TableGeometry,
    TargetGeometry,
    VisualServoController,
)
from gear_sonic.utils.inference.base_pose_visual_servo_diagnostics import (
    DetectionFrameData,
)
from gear_sonic.utils.inference.base_pose import AlignedRGBDSnapshot, BasePoseCameraError


def _servo_observation(
    *,
    yaw_rad: float = 0.0,
    include_table: bool = True,
) -> RawServoObservation:
    return RawServoObservation(
        target=TargetGeometry(1.2, 0.0, (1.2, 0.0, 0.5), 100, 0.9, 1.2),
        table=(
            TableGeometry(
                yaw_error_rad=yaw_rad,
                line_length_px=100.0,
                valid_depth_samples=20,
                line_center_px=(320.0, 200.0),
            )
            if include_table
            else None
        ),
        camera_timestamp=1.0,
        target_track_id=1,
        surface_track_id=2 if include_table else None,
        target_bbox_xyxy=(240.0, 120.0, 400.0, 360.0),
    )


def _orientation(
    actual_heading_rad: float,
    heading_setpoint_rad: float | None = None,
) -> dict[str, float]:
    return {
        "actual_heading_rad": actual_heading_rad,
        "heading_setpoint_rad": (
            actual_heading_rad
            if heading_setpoint_rad is None
            else heading_setpoint_rad
        ),
        "state_age_s": 0.01,
        "telemetry_age_s": 0.01,
    }


def test_yolo_adapter_uses_gateway_generation_directly(tmp_path) -> None:
    intents: list[tuple[str, dict[str, object]]] = []
    adapter = GatewayRawServoAdapter(
        BasePoseAgentConfig(
            task="align to the basket",
            output_root=str(tmp_path),
        ),
        submit_intent=lambda name, values: intents.append((name, dict(values))),
        monotonic=lambda: 10.0,
    )

    assert adapter.start(7, now=10.0)
    assert adapter.runtime.generation == 7
    assert intents[-1][0] == "base_pose_velocity"
    assert intents[-1][1]["generation"] == 7
    assert intents[-1][1]["motion_profile"] == "yoloe_servo"
    assert intents[-1][1]["velocity"] == [0.0, 0.0, 0.0]


def test_yolo_adapter_ignores_stale_and_idle_global_cancels(tmp_path) -> None:
    logs: list[str] = []
    adapter = GatewayRawServoAdapter(
        BasePoseAgentConfig(
            task="align to the basket",
            output_root=str(tmp_path),
        ),
        submit_intent=lambda _name, _values: None,
        logger=logs.append,
    )

    assert not adapter.cancel(1, "unrelated_navigation_cancel", now=1.0)
    assert adapter.runtime.phase == "idle"
    assert adapter.runtime.generation == 1
    assert adapter.start(2, now=1.1)
    assert not adapter.cancel(1, "stale_cancel", now=1.2)
    assert adapter.runtime.phase == "inference"
    assert adapter.cancel(3, "base_pose_velocity_timeout", now=1.3)
    assert adapter.runtime.phase == "idle"
    assert adapter.runtime.generation == 3
    assert not any("STOP unrelated_navigation_cancel" in line for line in logs)
    assert any("ignored stale cancel" in line for line in logs)
    assert any("STOP base_pose_velocity_timeout" in line for line in logs)


def test_yolo_adapter_forwards_viewer_overlay_without_touching_velocity(
    tmp_path,
) -> None:
    intents: list[tuple[str, dict[str, object]]] = []
    adapter = GatewayRawServoAdapter(
        BasePoseAgentConfig(
            task="align to the red tote",
            output_root=str(tmp_path),
        ),
        submit_intent=lambda name, values: intents.append((name, dict(values))),
    )
    assert adapter.start(2, now=1.0)
    overlay = {
        "target_bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
        "target_lateral_anchor_px": [20.0, 30.0],
        "table_edge_endpoints_px": [[0.0, 35.0], [50.0, 35.0]],
        "desk_mask_row_spans": [[35, 0, 50]],
        "image_size": [64, 48],
    }

    adapter._publish(
        {
            "action": "visual_servo",
            "camera_stream": "ego_view",
            "velocity": {"vx": 0.0, "vy": 0.0, "wz": 0.2},
            "viewer_overlay": overlay,
        }
    )

    assert intents[-1][0] == "base_pose_velocity"
    assert intents[-1][1]["velocity"] == [0.0, 0.0, 0.2]
    assert intents[-1][1]["viewer_overlay"] == overlay


def test_mixed_camera_overlay_keeps_mode_border_and_routes_geometry(
    tmp_path,
) -> None:
    intents: list[tuple[str, dict[str, object]]] = []
    adapter = GatewayRawServoAdapter(
        BasePoseAgentConfig(
            task="align to the basket",
            output_root=str(tmp_path),
        ),
        submit_intent=lambda name, values: intents.append((name, dict(values))),
    )
    assert adapter.start(2, now=1.0)
    runtime = adapter.runtime
    runtime.active_camera_stream = "ego_view"
    runtime.control_source_stream = "chest_view"
    desk_mask = np.zeros((48, 64), dtype=np.uint8)
    desk_mask[32:34, 4:60] = 1
    observation = replace(
        _servo_observation(),
        image_width=64,
        image_height=48,
        desk_mask=desk_mask,
        table=replace(
            _servo_observation().table,
            line_endpoints_px=((5.0, 35.0), (55.0, 35.0)),
        ),
        table_camera_stream="ego_view",
    )

    runtime._set_viewer_overlay(
        observation,
        {"yaw_source": {"stream": "ego_view", "valid": True}},
    )
    runtime._publish(ServoCommand(0.0, 0.0, 0.0), "visual_servo")

    parameters = intents[-1][1]
    assert parameters["camera_stream"] == "ego_view"
    overlay = parameters["viewer_overlay"]
    assert overlay["target_camera_stream"] == "chest_view"
    assert overlay["table_camera_stream"] == "ego_view"
    assert overlay["table_edge_endpoints_px"] == [
        [5.0, 35.0],
        [55.0, 35.0],
    ]
    assert overlay["desk_mask_row_spans"]


@pytest.mark.parametrize(
    ("frame_kwargs", "viewer_attribute", "expected"),
    (
        (
            {"target_bbox_xyxy": (1.0, 2.0, 10.0, 20.0)},
            "viewer_target_bbox_xyxy",
            (1.0, 2.0, 10.0, 20.0),
        ),
        (
            {
                "surface_mask": np.pad(
                    np.ones((1, 4), dtype=np.uint8),
                    ((3, 44), (5, 55)),
                )
            },
            "viewer_desk_mask_row_spans",
            ((3, 5, 9),),
        ),
        (
            {
                "table_geometry": {
                    "line_endpoints_px": ((2.0, 30.0), (60.0, 31.0))
                }
            },
            "viewer_table_edge_endpoints_px",
            ((2.0, 30.0), (60.0, 31.0)),
        ),
    ),
)
def test_incomplete_yoloe_frame_keeps_each_available_viewer_overlay(
    tmp_path,
    frame_kwargs: dict[str, object],
    viewer_attribute: str,
    expected: object,
) -> None:
    adapter = GatewayRawServoAdapter(
        BasePoseAgentConfig(
            task="align to the basket",
            output_root=str(tmp_path),
        ),
        submit_intent=lambda _name, _values: None,
    )
    assert adapter.start(2, now=1.0)
    assert adapter.runtime.accept_event(
        RawServoEvent(
            2,
            "initialized",
            observation=_servo_observation(),
            details={"live_stream": "ego_view"},
        ),
        now=1.1,
    )
    frame = DetectionFrameData(
        frame_index=1,
        camera_timestamp=1.2,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        perception_kind="invalid",
        perception_error="partial detection",
        **frame_kwargs,
    )

    assert adapter.runtime.accept_event(
        RawServoEvent(
            2,
            "invalid",
            error="partial detection",
            frame=frame,
        ),
        now=1.2,
    )

    assert getattr(adapter.runtime, viewer_attribute) == expected
    assert adapter.runtime.viewer_image_size == (64, 48)


def test_yolo_adapter_injects_agent_near_orientation_provider(tmp_path) -> None:
    provider = lambda _now: {
        "actual_heading_rad": 0.1,
        "heading_setpoint_rad": 0.2,
        "state_age_s": 0.01,
        "telemetry_age_s": 0.01,
    }

    adapter = GatewayRawServoAdapter(
        BasePoseAgentConfig(
            task="align to the basket",
            output_root=str(tmp_path),
        ),
        submit_intent=lambda _name, _values: None,
        orientation_provider=provider,
    )

    assert adapter.runtime.orientation_provider is provider


def test_yolo_adapter_reports_terminal_worker_failure_once(tmp_path) -> None:
    intents: list[tuple[str, dict[str, object]]] = []
    adapter = GatewayRawServoAdapter(
        BasePoseAgentConfig(
            task="align to the basket",
            output_root=str(tmp_path),
        ),
        submit_intent=lambda name, values: intents.append((name, dict(values))),
        monotonic=lambda: 10.0,
    )
    assert adapter.start(3, now=10.0)
    adapter.runtime.events.put(
        RawServoEvent(3, "error", error="tracker failed", hard=True)
    )

    adapter.tick(now=10.1)
    adapter.tick(now=10.2)

    statuses = [values for name, values in intents if name == "base_pose_status"]
    assert statuses == [
        {"generation": 3, "state": "failed", "reason": "tracker failed"}
    ]


def test_dual_worker_failure_stops_before_reporting_failed(tmp_path) -> None:
    intents: list[tuple[str, dict[str, object]]] = []
    adapter = GatewayRawServoAdapter(
        BasePoseAgentConfig(
            task="align to the basket",
            output_root=str(tmp_path),
        ),
        submit_intent=lambda name, values: intents.append((name, dict(values))),
        monotonic=lambda: 10.0,
    )
    assert adapter.start(4, now=10.0)
    adapter.runtime.events.put(
        RawServoEvent(
            4,
            "error",
            error="both-camera text failover exhausted",
            hard=True,
        )
    )

    adapter.tick(now=10.1)
    adapter.tick(now=10.2)

    velocities = [
        values for name, values in intents if name == "base_pose_velocity"
    ]
    assert velocities
    assert all(values["velocity"] == [0.0, 0.0, 0.0] for values in velocities)
    assert [values["action"] for values in velocities].count("stop") == 3
    statuses = [values for name, values in intents if name == "base_pose_status"]
    assert statuses == [
        {
            "generation": 4,
            "state": "failed",
            "reason": "both-camera text failover exhausted",
        }
    ]
    assert intents[-1][0] == "base_pose_status"


def test_yolo_servo_outputs_only_bounded_yaw_during_initial_alignment() -> None:
    controller = VisualServoController()
    observation = RawServoObservation(
        target=TargetGeometry(1.2, 0.0, (1.2, 0.0, 0.5), 100, 0.9, 1.2),
        table=TableGeometry(
            yaw_error_rad=0.5,
            line_length_px=100.0,
            valid_depth_samples=20,
            line_center_px=(320.0, 200.0),
        ),
        camera_timestamp=1.0,
        target_track_id=1,
        surface_track_id=2,
        target_bbox_xyxy=(240.0, 120.0, 400.0, 360.0),
    )

    command = controller.update(observation, now=0.1)

    assert controller.phase is ServoPhase.YAW_ALIGN
    assert command.vx == 0.0
    assert command.vy == 0.0
    assert 0.0 < abs(command.wz) <= 0.3


def test_yolo_calibration_rejects_non_metric_raw_depth() -> None:
    calibration = RawServoCalibration(2, 2, 100.0, 100.0, 1.0, 1.0)
    snapshot = AlignedRGBDSnapshot(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_raw=np.ones((2, 2), dtype=np.uint16),
        fx=100.0,
        fy=100.0,
        cx=1.0,
        cy=1.0,
        depth_scale_m=0.0,
        depth_aligned_to="ego_view",
        depth_source="realsense",
        timestamp=1.0,
    )

    with pytest.raises(BasePoseCameraError, match="depth scale"):
        calibration.validate_snapshot(snapshot)
