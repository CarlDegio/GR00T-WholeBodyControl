from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.base_pose import (
    BASE_POSE_OUTPUT_SCHEMA,
    BasePoseCameraError,
    BasePoseConfig,
    BasePoseObservation,
    BasePosePlanner,
    BasePoseResult,
    BasePoseSequenceController,
    BasePoseValidationError,
    SensorGatewayBasePoseCamera,
    build_base_pose_prompt,
    plan_to_segments,
    validate_base_pose_plan,
)
from gear_sonic.base_pose.policy import DEPTH_QUERY_SCHEMA


def command(step: int, action: str, value: float) -> dict[str, object]:
    return {
        "step": step,
        "action": action,
        "value": value,
        "unit": "degrees" if action.startswith("ROTATE_") else "meters",
        "purpose": action.lower(),
    }


def plan(
    commands: list[dict[str, object]] | None = None,
    *,
    status: str = "ADJUST",
) -> dict[str, object]:
    if commands is None:
        commands = [command(1, "ROTATE_LEFT", 30.0)]
    return {
        "status": status,
        "task_interpretation": {
            "primary_target": "medicine bottle",
            "secondary_targets": ["blue basket"],
            "manipulation_anchor": "bottle and basket opening",
            "interaction_direction": "face the combined workspace",
            "selection_reason": "supports pickup and placement",
        },
        "current_alignment": {
            "horizontal_position": "LEFT",
            "distance_estimate": "SUITABLE",
            "orientation_estimate": "TURNED_RIGHT",
            "theta": 20.0,
        },
        "desired_final_pose": {
            "target_alignment": "workspace centered",
            "target_distance": "reachable table standoff",
            "target_orientation": "face workspace",
        },
        "command_sequence": commands,
        "expected_result": "centered, reachable, and facing the workspace",
        "confidence": 0.8,
        "limitations": "visual estimate",
    }


def observation(*, depth: bool = False, stream: str = "ego_view") -> BasePoseObservation:
    return BasePoseObservation(
        rgb=np.zeros((12, 16, 3), dtype=np.uint8),
        depth_raw=np.full((12, 16), 900, dtype=np.uint16) if depth else None,
        fx=100.0,
        fy=100.0,
        cx=7.5,
        cy=5.5,
        depth_scale_m=0.001 if depth else None,
        depth_aligned_to=stream if depth else None,
        depth_source="lingbot-depth" if depth else None,
        timestamp=1.0,
    )


class FakeVisionClient:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["schema"] is DEPTH_QUERY_SCHEMA:
            return {
                "task_interpretation": {
                    "primary_target": "medicine bottle",
                    "secondary_targets": ["basket"],
                    "manipulation_anchors": ["bottle"],
                    "selection_reason": "measure grasp anchor",
                },
                "depth_queries": [
                    {
                        "query_id": "bottle",
                        "task_role": "PRIMARY_TARGET",
                        "target_or_anchor": "medicine bottle",
                        "bbox_2d": [250, 250, 750, 750],
                        "sampling_reason": "estimate range",
                    }
                ],
                "confidence": 0.9,
                "limitations": "none",
            }
        assert kwargs["schema"] is BASE_POSE_OUTPUT_SCHEMA
        return self.value


def test_planner_returns_exact_plan_without_persistent_diagnostics(tmp_path: Path) -> None:
    expected = plan()
    client = FakeVisionClient(expected)
    planner = BasePosePlanner(
        BasePoseConfig(task="pick and place", output_root=str(tmp_path)),
        client=client,  # type: ignore[arg-type]
    )

    result = planner.plan(observation())

    assert result.plan == expected
    assert result.output_dir is None
    assert result.raw_output == expected
    assert result.backend == "codex"
    assert len(client.calls) == 1
    assert len(client.calls[0]["image_paths"]) == 1
    assert list(tmp_path.iterdir()) == []


def test_rgbd_and_depth_query_keep_agent_near_modalities() -> None:
    rgbd_client = FakeVisionClient(plan())
    rgbd = BasePosePlanner(
        BasePoseConfig(task="align", mode="rgbd"),
        client=rgbd_client,  # type: ignore[arg-type]
    ).plan(observation(depth=True))
    assert rgbd.plan["status"] == "ADJUST"
    assert len(rgbd_client.calls[0]["image_paths"]) == 3

    query_client = FakeVisionClient(plan())
    queried = BasePosePlanner(
        BasePoseConfig(task="align", mode="rgb_depth_query"),
        client=query_client,  # type: ignore[arg-type]
    ).plan(observation(depth=True))
    assert len(query_client.calls) == 2
    assert queried.depth_query_selection is not None
    assert queried.depth_measurements is not None
    assert queried.depth_measurements[0]["center_7x7_median_mm"] == 900.0


def test_depth_modes_do_not_fallback_on_wrong_alignment_or_source() -> None:
    client = FakeVisionClient(plan())
    config = BasePoseConfig(task="align", mode="rgbd")
    wrong = observation(depth=True)
    wrong = BasePoseObservation(
        **{**wrong.__dict__, "depth_source": "raw-camera-depth"}
    )
    with pytest.raises(BasePoseCameraError, match="depth_source"):
        BasePosePlanner(config, client=client).plan(wrong)  # type: ignore[arg-type]
    assert client.calls == []


def test_sensor_gateway_adapter_preserves_lingbot_alignment_metadata() -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    depth = np.full((4, 6), 1000, dtype=np.uint16)
    info = {
        "fx": 100.0,
        "fy": 101.0,
        "cx": 2.5,
        "cy": 1.5,
        "width": 6,
        "height": 4,
        "depth_scale_m": 0.001,
        "depth_aligned_to": "chest_view",
    }
    frames = {
        "camera/chest_view": SimpleNamespace(
            attributes={"camera_info": info}, source_timestamp_ns=1_000_000_000
        ),
        "derived/lingbot_depth": SimpleNamespace(
            attributes={"camera_info": info, "depth_source": "lingbot-depth"},
            source_timestamp_ns=1_000_000_000,
        ),
    }
    materialized = SimpleNamespace(
        snapshot=SimpleNamespace(frames=frames),
        arrays={"camera/chest_view": rgb, "derived/lingbot_depth": depth},
    )
    camera = SensorGatewayBasePoseCamera(
        "inproc://unused",
        camera_stream="chest_view",
        require_depth=True,
        client=SimpleNamespace(),  # type: ignore[arg-type]
    )

    decoded = camera._decode(materialized)

    assert decoded.depth_source == "lingbot-depth"
    assert decoded.depth_aligned_to == "chest_view"
    assert decoded.depth_scale_m == 0.001
    assert decoded.timestamp == 1.0


def test_validator_preserves_agent_near_execution_thresholds() -> None:
    value = plan(
        [
            command(1, "ROTATE_LEFT", 2.0),
            command(2, "MOVE_BACKWARD", 0.1),
        ]
    )
    assert validate_base_pose_plan(value) == value
    value["command_sequence"][0]["value"] = 1.99  # type: ignore[index]
    with pytest.raises(BasePoseValidationError, match="rotation command"):
        validate_base_pose_plan(value)


def test_prompt_preserves_agent_near_minimum_command_rules() -> None:
    prompt = build_base_pose_prompt(
        BasePoseConfig(task="adjust pose"), observation()
    )

    assert (
        "Every MOVE_FORWARD or MOVE_BACKWARD command must specify a distance "
        "greater than or equal to 0.3 meters."
    ) in prompt
    assert (
        "Every ROTATE_LEFT or ROTATE_RIGHT command must specify an angle "
        "greater than or equal to 30 degrees."
    ) in prompt
    assert (
        "theta is an estimated geometric quantity used for scene reasoning "
        "and is not itself a motion command."
    ) in prompt
    assert "Do not output direct lateral movement." in prompt


def test_plan_to_segments_and_controller_preserve_order_and_pause() -> None:
    value = plan(
        [
            command(1, "ROTATE_RIGHT", 30.0),
            command(2, "MOVE_FORWARD", 0.3),
        ]
    )
    segments = plan_to_segments(value)
    assert [item.action for item in segments] == ["ROTATE_RIGHT", "MOVE_FORWARD"]
    assert segments[0].command.velocity == (0.0, 0.0, -0.4)
    assert segments[1].command.duration == pytest.approx(1.0)

    controller = BasePoseSequenceController(transition_pause=0.5)
    assert controller.start(BasePoseResult(value, None, {}), now=0.0)
    assert controller.step(0.0)[0] == "ROTATE_RIGHT"
    assert controller.step(segments[0].command.duration)[0] == "hold"
    assert controller.step(segments[0].command.duration + 0.5)[0] == "MOVE_FORWARD"

