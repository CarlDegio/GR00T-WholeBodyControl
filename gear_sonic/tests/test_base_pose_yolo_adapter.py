from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.scripts.base_pose_agent import BasePoseAgentConfig
from gear_sonic.scripts.base_pose_yolo_agent import GatewayRawServoAdapter
from gear_sonic.utils.inference.base_pose_visual_servo import RawServoEvent
from gear_sonic.utils.inference.base_pose_visual_servo import (
    RawServoCalibration,
    RawServoObservation,
    ServoPhase,
    TableGeometry,
    TargetGeometry,
    VisualServoController,
    validate_raw_servo_target,
)
from gear_sonic.utils.inference.base_pose import AlignedRGBDSnapshot, BasePoseCameraError


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


def test_yolo_grounding_contract_keeps_one_tight_normalized_target() -> None:
    target = validate_raw_servo_target(
        {
            "status": "READY",
            "primary_target": {
                "text_prompt": "blue basket",
                "bbox_2d": [250, 200, 750, 800],
            },
            "manipulation_anchor": "basket opening",
            "selection_reason": "stable task destination",
            "confidence": 0.9,
            "limitations": "",
        }
    )

    assert target.target_prompt == "blue basket"
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
