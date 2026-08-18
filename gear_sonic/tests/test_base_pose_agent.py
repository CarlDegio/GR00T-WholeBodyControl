from __future__ import annotations

from gear_sonic.base_pose import BasePoseResult
from gear_sonic.scripts.base_pose_agent import (
    BasePoseAgentConfig,
    BasePoseAgentRuntime,
    WorkerResult,
)


def plan() -> dict[str, object]:
    return {
        "status": "ADJUST",
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
        "command_sequence": [
            {
                "step": 1,
                "action": "MOVE_FORWARD",
                "value": 0.1,
                "unit": "meters",
                "purpose": "approach",
            }
        ],
        "expected_result": "centered and reachable",
        "confidence": 0.8,
        "limitations": "visual estimate",
    }


def test_agent_runtime_publishes_generation_scoped_velocity_and_terminal_status() -> None:
    intents: list[tuple[str, dict[str, object]]] = []
    runtime = BasePoseAgentRuntime(
        BasePoseAgentConfig(task="align", planner_hz=20.0),
        submit_intent=lambda name, parameters: intents.append((name, dict(parameters))),
    )
    result = BasePoseResult(plan(), None, {})

    assert runtime.start(1)
    runtime.results.put(WorkerResult(1, result, None))
    runtime.tick(0.0)
    assert intents[0][0] == "base_pose_velocity"
    assert intents[0][1]["generation"] == 1
    assert intents[0][1]["velocity"] == [0.3, 0.0, 0.0]

    runtime.tick(1.0)
    assert [name for name, _ in intents[-2:]] == [
        "base_pose_velocity",
        "base_pose_status",
    ]
    assert intents[-1][1]["state"] == "reached"
    assert runtime.state == "idle"


def test_agent_cancel_discards_late_model_result() -> None:
    intents: list[tuple[str, dict[str, object]]] = []
    runtime = BasePoseAgentRuntime(
        BasePoseAgentConfig(task="align"),
        submit_intent=lambda name, parameters: intents.append((name, dict(parameters))),
    )
    runtime.start(1)
    runtime.cancel(2, "operator_stop")
    runtime.results.put(WorkerResult(1, BasePoseResult(plan(), None, {}), None))

    runtime.tick(2.0)

    assert runtime.state == "idle"
    assert intents == []
