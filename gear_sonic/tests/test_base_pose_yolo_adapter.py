from __future__ import annotations

import json
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
    validate_raw_servo_target,
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
            TableGeometry(yaw_rad, 1.0, 100, 0.01, (1.0, 0.0))
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


def test_yolo_adapter_rebases_internal_worker_to_gateway_generation(tmp_path) -> None:
    intents: list[tuple[str, dict[str, object]]] = []
    adapter = GatewayRawServoAdapter(
        BasePoseAgentConfig(
            task="align to the basket",
            mode="raw_yoloe_servo",
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


def test_yolo_adapter_forwards_viewer_overlay_without_touching_velocity(
    tmp_path,
) -> None:
    intents: list[tuple[str, dict[str, object]]] = []
    adapter = GatewayRawServoAdapter(
        BasePoseAgentConfig(
            task="align to the red tote",
            mode="raw_yoloe_servo",
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
        json.dumps(
            {
                "action": "visual_servo",
                "camera_stream": "ego_view",
                "velocity": {"vx": 0.0, "vy": 0.0, "wz": 0.2},
                "viewer_overlay": overlay,
            }
        )
    )

    assert intents[-1][0] == "base_pose_velocity"
    assert intents[-1][1]["velocity"] == [0.0, 0.0, 0.2]
    assert intents[-1][1]["viewer_overlay"] == overlay


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
            mode="raw_yoloe_servo",
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
            mode="raw_yoloe_servo",
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


def test_yolo_grounding_contract_preserves_model_target_prompt() -> None:
    target = validate_raw_servo_target(
        {
            "status": "READY",
            "primary_target": {
                "text_prompt": "red tote returned by codex",
                "bbox_2d": [250, 200, 750, 800],
            },
            "manipulation_anchor": "basket opening",
            "selection_reason": "stable task destination",
            "confidence": 0.9,
            "limitations": "",
        }
    )

    assert target.target_prompt == "red tote returned by codex"
    assert target.target_bbox == (250.0, 200.0, 750.0, 800.0)


def test_yolo_servo_outputs_only_bounded_yaw_during_initial_alignment() -> None:
    controller = VisualServoController(post_stop_sample_s=0.0)
    observation = RawServoObservation(
        target=TargetGeometry(1.2, 0.0, (1.2, 0.0, 0.5), 100, 0.9, 1.2),
        table=TableGeometry(0.5, 1.0, 100, 0.01, (1.0, 0.0)),
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


@pytest.mark.parametrize("phase", [ServoPhase.YAW_ALIGN, ServoPhase.YAW_TRIM])
@pytest.mark.parametrize(
    ("actual_heading_deg", "expected_error_deg", "expected_wz"),
    [(15.0, 15.0, 0.10), (45.0, -15.0, -0.10)],
)
def test_yolo_optional_table_yaw_closes_against_agent_near_actual_heading(
    phase: ServoPhase,
    actual_heading_deg: float,
    expected_error_deg: float,
    expected_wz: float,
) -> None:
    controller = VisualServoController(
        ema_alpha=1.0,
        allow_missing_table=True,
    )
    controller.reset(1.0)
    controller.phase = phase

    controller.update(
        _servo_observation(yaw_rad=math.radians(20.0)),
        now=1.1,
        orientation=_orientation(math.radians(10.0)),
    )
    target_heading = math.radians(30.0)
    assert controller.desired_heading_rad == pytest.approx(target_heading)

    # The integrated SONIC setpoint may reach the target before the physical
    # base; agent-near closes this fallback against measured g1_debug yaw.
    controller.current = ServoCommand(0.0, 0.0, 0.0, 0.15)
    command = controller.update(
        _servo_observation(include_table=False),
        now=1.2,
        orientation=_orientation(
            math.radians(actual_heading_deg),
            target_heading,
        ),
    )

    assert controller.last_errors[2] == pytest.approx(
        math.radians(expected_error_deg)
    )
    assert controller.heading_setpoint_error_rad is None
    assert controller.yaw_error_source == "propagated_actual_heading"
    assert command.wz == pytest.approx(expected_wz)


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
