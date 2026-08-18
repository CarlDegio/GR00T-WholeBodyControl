from __future__ import annotations

from gear_sonic.scripts.base_pose_agent import BasePoseAgentConfig
from gear_sonic.scripts.base_pose_yolo_agent import GatewayRawServoAdapter
from gear_sonic.utils.inference.base_pose_visual_servo import RawServoEvent
from gear_sonic.utils.inference.base_pose_visual_servo import (
    RawServoObservation,
    ServoPhase,
    TableGeometry,
    TargetGeometry,
    VisualServoController,
    validate_raw_servo_target,
)


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
