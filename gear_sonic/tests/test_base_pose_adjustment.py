"""Software-only contract tests for head-vision base-pose adjustment."""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.scripts.base_pose_planner import (
    BasePosePlannerConfig,
    BasePosePlannerRuntime,
    BasePoseSequenceController,
    WorkerResult,
    plan_to_segments,
)
import gear_sonic.utils.inference.base_pose as base_pose_module
from gear_sonic.scripts.lavira_sonic_relay import (
    PlannerState,
    extract_frozen_planner_pose,
)
from gear_sonic.utils.inference.base_pose import (
    AlignedRGBDCamera,
    AlignedRGBDSnapshot,
    BasePoseCameraError,
    BasePoseConfig,
    BasePoseResult,
    BasePoseRunner,
    BasePoseValidationError,
    CodexStructuredVisionClient,
    QwenVLStructuredVisionClient,
    _dashscope_api_key,
    build_base_pose_prompt,
    query_depth_regions,
    validate_base_pose_plan,
)


def plan(
    commands: list[dict[str, object]] | None = None,
    *,
    status: str = "ADJUST",
) -> dict[str, object]:
    if commands is None:
        commands = [
            {
                "step": 1,
                "action": "ROTATE_LEFT",
                "value": 2.0,
                "unit": "degrees",
                "purpose": "align anchor",
            }
        ]
    return {
        "status": status,
        "task_interpretation": {
            "primary_target": "cup",
            "secondary_targets": ["tray"],
            "manipulation_anchor": "cup body and tray opening",
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
        "confidence": 0.7,
        "limitations": "visual estimate",
    }


def command(step: int, action: str, value: float) -> dict[str, object]:
    return {
        "step": step,
        "action": action,
        "value": value,
        "unit": "degrees" if action.startswith("ROTATE_") else "meters",
        "purpose": action.lower(),
    }


def result(value: dict[str, object] | None = None, output_dir: str = "/tmp/base"):
    return BasePoseResult(
        plan=plan() if value is None else value,
        output_dir=output_dir,
        timing_s={},
    )


def test_prompt_requires_point_three_meter_minimum_and_indirect_small_correction() -> None:
    snapshot = AlignedRGBDSnapshot(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_raw=None,
        fx=1.0,
        fy=1.0,
        cx=0.5,
        cy=0.5,
        depth_scale_m=None,
        depth_aligned_to=None,
        depth_source=None,
        timestamp=1.0,
    )

    prompt = build_base_pose_prompt(BasePoseConfig(task="adjust pose"), snapshot)

    assert (
        "Every MOVE_FORWARD or MOVE_BACKWARD command must specify a distance "
        "greater than or equal to 0.3 meters."
    ) in prompt
    assert (
        "If the desired positional correction is smaller than 0.3 meters, do not "
        "approximate it using an invalid smaller translation."
    ) in prompt
    assert (
        "When geometrically appropriate and safe, use a combination of backward "
        "movement, rotation, forward movement, and a final corrective rotation to "
        "produce the required positional change."
    ) in prompt
    assert (
        "Every translation-command value must be greater than or equal to 0.3 "
        "meters."
    ) in prompt
    assert '"value": 0.0' not in prompt
    assert '"theta": 0.0' in prompt
    assert (
        "theta is an estimated geometric quantity used for scene reasoning "
        "and is not itself a motion command."
    ) in prompt


def test_validator_accepts_exact_two_degree_and_point_one_meter_minima() -> None:
    value = plan(
        [
            command(1, "ROTATE_LEFT", 2.0),
            command(2, "MOVE_BACKWARD", 0.10),
            command(3, "ROTATE_RIGHT", 90.0),
            command(4, "MOVE_FORWARD", 1.50),
        ]
    )

    assert validate_base_pose_plan(value) == value

    value["command_sequence"][0]["value"] = 1.999  # type: ignore[index]
    with pytest.raises(BasePoseValidationError, match="rotation command"):
        validate_base_pose_plan(value)


def test_validator_requires_valid_theta() -> None:
    missing = plan()
    missing["current_alignment"].pop("theta")  # type: ignore[index]
    with pytest.raises(BasePoseValidationError, match="invalid object schema"):
        validate_base_pose_plan(missing)

    negative = plan()
    negative["current_alignment"]["theta"] = -0.1  # type: ignore[index]
    with pytest.raises(BasePoseValidationError, match="non-negative"):
        validate_base_pose_plan(negative)

    unknown = plan()
    unknown["current_alignment"].update(  # type: ignore[union-attr]
        {"orientation_estimate": "UNKNOWN", "theta": 1.0}
    )
    with pytest.raises(BasePoseValidationError, match="must be 0"):
        validate_base_pose_plan(unknown)

    unknown["current_alignment"]["theta"] = 0.0  # type: ignore[index]
    assert validate_base_pose_plan(unknown)["current_alignment"]["theta"] == 0.0


@pytest.mark.parametrize(
    ("commands", "message"),
    [
        ([command(1, "MOVE_FORWARD", 0.099)], "translation command"),
        (
            [command(index + 1, "ROTATE_LEFT", 30.1) for index in range(6)],
            "total rotation",
        ),
        (
            [command(index + 1, "MOVE_FORWARD", 0.61) for index in range(5)],
            "total translation",
        ),
        (
            [command(index + 1, "ROTATE_LEFT", 2.0) for index in range(9)],
            "exceeds 8 steps",
        ),
    ],
)
def test_validator_rejects_whole_out_of_bounds_sequence(commands, message) -> None:
    with pytest.raises(BasePoseValidationError, match=message):
        validate_base_pose_plan(plan(commands))


@pytest.mark.parametrize("status", ["READY", "UNSURE", "UNSAFE"])
def test_non_adjust_statuses_require_empty_commands(status: str) -> None:
    assert validate_base_pose_plan(plan([], status=status))["command_sequence"] == []
    with pytest.raises(BasePoseValidationError, match="empty command_sequence"):
        validate_base_pose_plan(plan(status=status))


def test_adapter_preserves_order_sign_and_exact_model_duration() -> None:
    model_plan = plan(
        [
            command(1, "ROTATE_LEFT", 30.0),
            command(2, "MOVE_BACKWARD", 0.3),
            command(3, "ROTATE_RIGHT", 10.0),
            command(4, "MOVE_FORWARD", 0.6),
        ]
    )

    segments = plan_to_segments(model_plan, rotation_speed=0.4, translation_speed=0.3)

    assert [segment.action for segment in segments] == [
        "ROTATE_LEFT",
        "MOVE_BACKWARD",
        "ROTATE_RIGHT",
        "MOVE_FORWARD",
    ]
    assert [segment.command.wz for segment in segments] == pytest.approx(
        [0.4, 0.0, -0.4, 0.0]
    )
    assert [segment.command.vx for segment in segments] == pytest.approx(
        [0.0, -0.3, 0.0, 0.3]
    )
    assert [segment.command.duration for segment in segments] == pytest.approx(
        [math.radians(30.0) / 0.4, 1.0, math.radians(10.0) / 0.4, 2.0]
    )


def test_adapter_scales_model_translations_and_rotations() -> None:
    model_plan = plan(
        [
            command(1, "ROTATE_LEFT", 30.0),
            command(2, "MOVE_FORWARD", 0.4),
        ]
    )

    segments = plan_to_segments(
        model_plan,
        rotation_speed=0.5,
        translation_speed=0.2,
        rotation_scale=1.5,
        translation_scale=2.0,
    )

    assert [segment.command.duration for segment in segments] == pytest.approx(
        [math.radians(45.0) / 0.5, 0.8 / 0.2]
    )


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
@pytest.mark.parametrize("scale_name", ["rotation_scale", "translation_scale"])
def test_adapter_rejects_invalid_motion_scale(
    scale_name: str, scale: float
) -> None:
    with pytest.raises(ValueError, match=scale_name):
        plan_to_segments(plan([]), **{scale_name: scale})


def test_sequence_controller_inserts_idle_and_space_can_cancel_every_phase() -> None:
    controller = BasePoseSequenceController(transition_pause=0.5)
    assert controller.start(
        result(plan([command(1, "ROTATE_LEFT", 2.0), command(2, "MOVE_FORWARD", 0.1)])),
        10.0,
    )
    rotation_duration = math.radians(2.0) / 0.4
    action, idle = controller.step(10.0 + rotation_duration)
    assert action == "hold"
    assert idle.velocity == (0.0, 0.0, 0.0)
    assert controller.phase == "pause"

    controller.cancel()
    assert not controller.active
    assert controller.step(11.0)[1].velocity == (0.0, 0.0, 0.0)


def test_runtime_space_drops_late_model_result_and_never_restarts_motion(
    tmp_path,
) -> None:
    messages: list[str] = []
    runtime = BasePosePlannerRuntime(
        BasePosePlannerConfig(task="put cup in tray", output_root=str(tmp_path)),
        publish=messages.append,
        logger=lambda _message: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    generation = runtime.pending_generation
    assert runtime.handle_key(" ", now=1.1) == "cancelled"

    late = WorkerResult(
        generation=generation,  # type: ignore[arg-type]
        result=result(output_dir=str(tmp_path / "late")),
        error=None,
    )
    assert not runtime.accept_worker_result(late, now=1.2)
    assert runtime.phase == "idle"
    assert all(
        json.loads(message)["velocity"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        for message in messages[-3:]
    )


@pytest.mark.parametrize(
    ("action", "value"),
    [
        ("ROTATE_LEFT", 5.0),
        ("ROTATE_RIGHT", 5.0),
        ("MOVE_FORWARD", 0.2),
        ("MOVE_BACKWARD", 0.2),
    ],
)
def test_space_cancels_each_motion_direction_immediately(
    tmp_path: Path, action: str, value: float
) -> None:
    messages: list[str] = []
    runtime = BasePosePlannerRuntime(
        BasePosePlannerConfig(task="adjust", output_root=str(tmp_path)),
        publish=messages.append,
        logger=lambda _message: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    generation = runtime.pending_generation
    assert runtime.accept_worker_result(
        WorkerResult(
            generation=generation,  # type: ignore[arg-type]
            result=result(
                plan([command(1, action, value)]),
                output_dir=str(tmp_path / action),
            ),
            error=None,
        ),
        now=1.1,
    )
    assert runtime.phase == "motion"

    runtime.handle_key(" ", now=1.2)

    assert runtime.phase == "idle"
    assert all(
        json.loads(message)["velocity"]
        == {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        for message in messages[-3:]
    )


def test_space_cancels_during_inter_step_idle_pause(tmp_path: Path) -> None:
    runtime = BasePosePlannerRuntime(
        BasePosePlannerConfig(task="adjust", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _message: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    generation = runtime.pending_generation
    assert runtime.accept_worker_result(
        WorkerResult(
            generation=generation,  # type: ignore[arg-type]
            result=result(
                plan(
                    [
                        command(1, "ROTATE_LEFT", 2.0),
                        command(2, "MOVE_FORWARD", 0.1),
                    ]
                ),
                output_dir=str(tmp_path / "pause"),
            ),
            error=None,
        ),
        now=1.0,
    )
    runtime.publish_due(1.0 + math.radians(2.0) / 0.4)
    assert runtime.controller.phase == "pause"

    runtime.handle_key(" ", now=1.2)

    assert runtime.phase == "idle"


def test_space_discards_not_yet_started_request_so_fresh_n_can_queue(
    tmp_path: Path,
) -> None:
    runtime = BasePosePlannerRuntime(
        BasePosePlannerConfig(task="adjust", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _message: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    assert runtime.handle_key(" ", now=1.1) == "cancelled"

    assert runtime.handle_key("n", now=1.2) == "started"


def test_runtime_exposes_no_planner_ready_state() -> None:
    assert "planner_ready_file" not in BasePosePlannerConfig.__dataclass_fields__


def test_space_publishes_exact_zero_stop_sequence(tmp_path: Path) -> None:
    messages: list[str] = []
    logs: list[str] = []
    config = BasePosePlannerConfig(
        task="adjust",
        output_root=str(tmp_path),
        final_stop_count=4,
    )
    runtime = BasePosePlannerRuntime(
        config,
        publish=messages.append,
        logger=logs.append,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    count_before_space = len(messages)

    assert runtime.handle_key(" ", now=1.1) == "cancelled"

    stops = [json.loads(message) for message in messages[count_before_space:]]
    assert len(stops) == config.final_stop_count
    assert all(
        stop["velocity"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        for stop in stops
    )
    assert all(stop["action"] == "stop" for stop in stops)
    assert logs[-1] == "[BasePose] STOP operator_space"


def snapshot(*, depth: bool = True) -> AlignedRGBDSnapshot:
    return AlignedRGBDSnapshot(
        rgb=np.zeros((10, 20, 3), dtype=np.uint8),
        depth_raw=(np.full((10, 20), 1250, dtype=np.uint16) if depth else None),
        fx=100.0,
        fy=100.0,
        cx=9.5,
        cy=4.5,
        depth_scale_m=0.001 if depth else None,
        depth_aligned_to="ego_view" if depth else None,
        depth_source="lingbot-depth" if depth else None,
        timestamp=12.0,
    )


def test_dynamic_ego_rgbd_decode_requires_aligned_pure_lingbot_uint16() -> None:
    source = snapshot()
    schema = ImageMessageSchema(
        timestamps={"ego_view": 12.0, "ego_view_depth": 12.0},
        images={"ego_view": source.rgb, "ego_view_depth": source.depth_raw},
        camera_info={
            "ego_view": {
                "fx": source.fx,
                "fy": source.fy,
                "cx": source.cx,
                "cy": source.cy,
                "width": 20,
                "height": 10,
                "depth_scale_m": 0.001,
                "depth_aligned_to": "ego_view",
                "depth_source": "lingbot-depth",
            }
        },
    )
    camera = object.__new__(AlignedRGBDCamera)
    camera.stream_name = "ego_view"
    camera.depth_key = "ego_view_depth"
    camera.require_depth = True
    camera.required_depth_source = "lingbot-depth"

    decoded = camera.decode_payload(schema.serialize())

    assert decoded.depth_raw.dtype == np.uint16
    assert decoded.depth_aligned_to == "ego_view"
    assert decoded.depth_source == "lingbot-depth"

    bad = schema.serialize()
    bad["camera_info"]["ego_view"]["depth_source"] = "realsense"
    with pytest.raises(BasePoseCameraError, match="depth_source"):
        camera.decode_payload(bad)


def test_roi_depth_statistics_and_intrinsic_backprojection() -> None:
    selection = {
        "depth_queries": [
            {
                "query_id": "cup",
                "task_role": "PRIMARY_TARGET",
                "target_or_anchor": "cup center",
                "bbox_2d": [250.0, 200.0, 750.0, 800.0],
                "sampling_reason": "pickup anchor",
            }
        ]
    }

    measurement = query_depth_regions(selection, snapshot())[0]

    assert measurement["bbox_pixel"] == [5, 2, 15, 8]
    assert measurement["valid_ratio"] == pytest.approx(1.0)
    assert measurement["center_7x7_median_mm"] == pytest.approx(1250.0)
    assert measurement["depth_percentiles_mm"] == {
        "p10": 1250.0,
        "p25": 1250.0,
        "p50": 1250.0,
        "p75": 1250.0,
        "p90": 1250.0,
    }
    assert measurement["camera_xyz_m"] == pytest.approx(
        {"x_right": 0.00625, "y_down": 0.00625, "z_forward": 1.25}
    )


@pytest.mark.parametrize(
    ("fast", "feature_config", "expects_fast_tier"),
    [
        (True, "features.fast_mode=true", True),
        (False, "features.fast_mode=false", False),
    ],
)
def test_codex_client_sets_explicit_fast_mode(
    tmp_path: Path,
    fast: bool,
    feature_config: str,
    expects_fast_tier: bool,
) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object):
        calls.append(list(command))
        if command[1:] == ["login", "status"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="Logged in with ChatGPT", stderr=""
            )
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(plan()), stderr=""
        )

    image_path = tmp_path / "rgb.png"
    image_path.write_bytes(b"png bytes")
    client = (
        CodexStructuredVisionClient(runner=runner)
        if fast
        else CodexStructuredVisionClient(fast=False, runner=runner)
    )

    assert client.run(
        prompt="plan",
        image_paths=[image_path],
        schema={"type": "object"},
        schema_filename="plan.schema.json",
        cwd=tmp_path,
    ) == plan()

    configs = [
        command[index + 1]
        for command in calls[1:]
        for index, value in enumerate(command[:-1])
        if value == "--config"
    ]
    assert feature_config in configs
    assert ('service_tier="fast"' in configs) is expects_fast_tier


def test_dashscope_key_prefers_environment_then_virtualenv_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DASHSCOPE_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    assert _dashscope_api_key(env_file) == "file-key"

    monkeypatch.setenv("DASHSCOPE_API_KEY", "environment-key")
    assert _dashscope_api_key(env_file) == "environment-key"


def test_qwenvl_plus_client_streams_images_schema_and_json(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    expected = plan()

    class FakeCompletions:
        def create(self, **kwargs: object):
            calls.append(kwargs)
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    reasoning_content="inspect image",
                                    content=None,
                                )
                            )
                        ]
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    reasoning_content=None,
                                    content=json.dumps(expected),
                                )
                            )
                        ]
                    ),
                    SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=10)),
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    image_path = tmp_path / "rgb.png"
    image_path.write_bytes(b"png bytes")
    schema = {"type": "object", "required": ["status"]}
    depth_path = tmp_path / "depth.png"
    depth_path.write_bytes(b"depth png bytes")
    qwen = QwenVLStructuredVisionClient(client=client)

    result = qwen.run(
        prompt="plan from ATTACHED_IMAGE_1",
        image_paths=[image_path, depth_path],
        schema=schema,
        schema_filename="plan.schema.json",
        cwd=tmp_path,
    )

    assert result == expected
    assert qwen.last_reasoning_content == "inspect image"
    assert qwen.last_answer_content == json.dumps(expected)
    assert calls[0]["model"] == "qwen3-vl-plus"
    assert calls[0]["stream"] is True
    assert calls[0]["timeout"] == 600.0
    assert calls[0]["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": 500,
    }
    messages = calls[0]["messages"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "plan from ATTACHED_IMAGE_1" in content[-1]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "ATTACHED_IMAGE_2 is image_url item 2 above." in content[-1]["text"]
    assert "Return only one JSON object matching this schema" in content[-1]["text"]
    assert json.loads((tmp_path / "plan.schema.json").read_text()) == schema


def test_runner_selects_qwenvl_plus_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fake_client = object()

    def build_qwen(**kwargs: object) -> object:
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(base_pose_module, "QwenVLStructuredVisionClient", build_qwen)
    runner = BasePoseRunner(
        BasePoseConfig(
            task="adjust pose",
            vision_backend="qwenvl",
            qwenvl_model="qwen3-vl-plus",
            qwenvl_base_url="https://dashscope.example/v1",
            codex_timeout_seconds=123.0,
        ),
        camera=FakeCamera(snapshot(depth=False)),  # type: ignore[arg-type]
    )

    assert runner.client is fake_client
    assert captured == {
        "model": "qwen3-vl-plus",
        "base_url": "https://dashscope.example/v1",
        "timeout_seconds": 123.0,
    }


class FakeCamera:
    def __init__(self, value: AlignedRGBDSnapshot):
        self.value = value
        self.captures = 0

    def capture(self):
        self.captures += 1
        return self.value

    def close(self):
        pass


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.mark.parametrize(
    ("mode", "depth", "image_counts"),
    [
        ("rgb", False, [1]),
        ("rgbd", True, [3]),
        ("rgb_depth_query", True, [1, 1]),
    ],
)
def test_three_modes_use_expected_attachments_and_one_snapshot(
    tmp_path: Path, mode: str, depth: bool, image_counts: list[int]
) -> None:
    responses = [plan()]
    if mode == "rgb_depth_query":
        responses.insert(
            0,
            {
                "task_interpretation": {
                    "primary_target": "cup",
                    "secondary_targets": ["tray"],
                    "manipulation_anchors": ["cup body", "tray opening"],
                    "selection_reason": "complete workspace",
                },
                "depth_queries": [],
                "confidence": 0.5,
                "limitations": "none",
            },
        )
    camera = FakeCamera(snapshot(depth=depth))
    client = FakeClient(responses)
    runner = BasePoseRunner(
        BasePoseConfig(
            task="put cup in tray",
            mode=mode,  # type: ignore[arg-type]
            output_root=str(tmp_path),
        ),
        camera=camera,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
    )

    base_result = runner.run_once()

    assert camera.captures == 1
    assert [len(call["image_paths"]) for call in client.calls] == image_counts
    assert '"theta": 0.0' in client.calls[-1]["prompt"]
    assert Path(base_result.output_dir, "rgb.png").is_file()
    assert Path(base_result.output_dir, "validation.json").is_file()
    if depth:
        assert Path(base_result.output_dir, "depth_uint16_mm.png").is_file()
        assert Path(base_result.output_dir, "depth_color_0_3m.png").is_file()


def test_current_upper_body_and_hands_are_latched_into_idle_planner_message() -> None:
    body = np.arange(29, dtype=np.float64) / 10.0
    left = np.arange(7, dtype=np.float64)
    right = np.arange(7, dtype=np.float64) + 10.0
    frozen = extract_frozen_planner_pose(
        {
            "body_q_measured": body,
            "left_hand_q_measured": left,
            "right_hand_q_measured": right,
        }
    )

    message = PlannerState().message(
        np.zeros(3, dtype=np.float32), 0.05, frozen_pose=frozen
    )
    header = json.loads(message[7 : 7 + 1280].rstrip(b"\x00"))
    payload = message[7 + 1280 :]

    assert struct.unpack_from("<i", payload, 0)[0] == 0
    assert [field["name"] for field in header["fields"]][-4:] == [
        "upper_body_position",
        "upper_body_velocity",
        "left_hand_joints",
        "right_hand_joints",
    ]
    assert len(frozen.upper_body_position) == 17
    assert frozen.left_hand_position == tuple(left)
    assert frozen.right_hand_position == tuple(right)
