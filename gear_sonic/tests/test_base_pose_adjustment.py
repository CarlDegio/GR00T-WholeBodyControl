"""Contracts shared by YOLOE base-pose camera, vision, and relay paths."""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.camera.sensor_server import ImageMessageSchema
import gear_sonic.scripts.lavira_sonic_relay as sonic_relay
from gear_sonic.scripts.lavira_sonic_relay import (
    PlannerState,
    extract_frozen_planner_pose,
)
from gear_sonic.utils.inference.base_pose import (
    AlignedRGBDCamera,
    AlignedRGBDSnapshot,
    CodexStructuredVisionClient,
    DualAlignedRGBDCamera,
    QwenVLStructuredVisionClient,
    _dashscope_api_key,
)
from gear_sonic.utils.teleop.sonic_orientation_telemetry import OrientationTracker


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


def test_ego_rgbd_decode_uses_saved_calibration_without_packet_camera_info() -> None:
    source = snapshot()
    schema = ImageMessageSchema(
        timestamps={"ego_view": 12.0, "ego_view_depth": 12.0},
        images={"ego_view": source.rgb, "ego_view_depth": source.depth_raw},
    )
    camera = object.__new__(AlignedRGBDCamera)
    camera.stream_name = "ego_view"
    camera.depth_key = "ego_view_depth"
    camera.require_depth = True
    camera.required_depth_source = None
    camera._configured_camera_info = {
        "fx": source.fx,
        "fy": source.fy,
        "cx": source.cx,
        "cy": source.cy,
        "width": 20,
        "height": 10,
        "depth_scale_m": 0.001,
        "depth_aligned_to": "ego_view",
    }

    decoded = camera.decode_payload(schema.serialize())

    assert decoded.fx == 100.0
    assert decoded.fy == 100.0
    assert decoded.depth_scale_m == 0.001
    assert decoded.depth_aligned_to == "ego_view"


def _dual_stream_decoder(
    stream_name: str,
    *,
    width: int = 20,
    height: int = 10,
) -> AlignedRGBDCamera:
    decoder = object.__new__(AlignedRGBDCamera)
    decoder.stream_name = stream_name
    decoder.depth_key = f"{stream_name}_depth"
    decoder.require_depth = True
    decoder.required_depth_source = None
    decoder._configured_camera_info = {
        "fx": 100.0,
        "fy": 101.0,
        "cx": 9.5,
        "cy": 4.5,
        "width": width,
        "height": height,
        "depth_scale_m": 0.001,
        "depth_aligned_to": stream_name,
    }
    return decoder


def test_dual_rgbd_decodes_both_streams_from_one_composed_payload() -> None:
    assert DualAlignedRGBDCamera is not None
    camera = object.__new__(DualAlignedRGBDCamera)
    camera.stream_names = ("ego_view", "chest_view")
    camera._decoders = {
        stream: _dual_stream_decoder(stream)
        for stream in camera.stream_names
    }
    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    head_depth = np.full((10, 20), 1100, dtype=np.uint16)
    chest_depth = np.full((10, 20), 1400, dtype=np.uint16)
    payload = ImageMessageSchema(
        timestamps={"ego_view": 12.0, "chest_view": 12.0},
        images={
            "ego_view": rgb,
            "ego_view_depth": head_depth,
            "chest_view": rgb + 1,
            "chest_view_depth": chest_depth,
        },
    ).serialize()

    capture = camera.decode_payload(payload)

    assert set(capture.snapshots) == {"ego_view", "chest_view"}
    assert capture.errors == {}
    assert capture.snapshots["ego_view"].depth_raw is not None
    assert capture.snapshots["chest_view"].depth_raw is not None
    assert int(capture.snapshots["ego_view"].depth_raw[0, 0]) == 1100
    assert int(capture.snapshots["chest_view"].depth_raw[0, 0]) == 1400


def test_dual_rgbd_isolates_one_malformed_stream() -> None:
    assert DualAlignedRGBDCamera is not None
    camera = object.__new__(DualAlignedRGBDCamera)
    camera.stream_names = ("ego_view", "chest_view")
    camera._decoders = {
        stream: _dual_stream_decoder(stream)
        for stream in camera.stream_names
    }
    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    payload = ImageMessageSchema(
        timestamps={"ego_view": 12.0, "chest_view": 12.0},
        images={
            "ego_view": rgb,
            "ego_view_depth": np.full((10, 20), 1100, dtype=np.uint16),
            "chest_view": rgb,
            "chest_view_depth": np.full((9, 20), 1400, dtype=np.uint16),
        },
    ).serialize()

    capture = camera.decode_payload(payload)

    assert set(capture.snapshots) == {"ego_view"}
    assert "chest_view" in capture.errors
    assert "shapes do not match" in capture.errors["chest_view"]




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
    expected = {"bbox_2d": [250, 200, 750, 800], "confidence": 0.9}

    def runner(command: list[str], **_kwargs: object):
        calls.append(list(command))
        if command[1:] == ["login", "status"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="Logged in with ChatGPT", stderr=""
            )
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(expected), stderr=""
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
    ) == expected

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
    expected = {"bbox_2d": [250, 200, 750, 800], "confidence": 0.9}

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


def test_qwenvl_client_disables_thinking_without_budget(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    expected = {"status": "READY"}

    class FakeCompletions:
        def create(self, **kwargs: object):
            calls.append(kwargs)
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    reasoning_content=None,
                                    content=json.dumps(expected),
                                )
                            )
                        ]
                    )
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    image_path = tmp_path / "rgb.png"
    image_path.write_bytes(b"png bytes")
    qwen = QwenVLStructuredVisionClient(
        model="qwen3-vl-8b-instruct",
        enable_thinking=False,
        client=client,
    )

    result = qwen.run(
        prompt="locate target",
        image_paths=[image_path],
        schema={"type": "object", "required": ["status"]},
        schema_filename="target.schema.json",
        cwd=tmp_path,
    )

    assert result == expected
    assert calls[0]["extra_body"] == {"enable_thinking": False}


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


def test_relay_integrates_wz_into_planner_facing_target() -> None:
    planner = PlannerState()
    planner.message(np.array([0.0, 0.0, 0.2], dtype=np.float32), 0.05)
    message = planner.message(
        np.array([0.0, 0.0, 0.2], dtype=np.float32), 0.05
    )
    payload = message[7 + 1280 :]
    facing = struct.unpack_from("<fff", payload, 16)

    assert planner.heading == pytest.approx(0.02)
    assert facing == pytest.approx(
        (math.cos(0.02), math.sin(0.02), 0.0), abs=1.0e-6
    )


def test_relay_orientation_sample_uses_post_integration_heading() -> None:
    planner = PlannerState()
    tracker = OrientationTracker()
    state = {
        "base_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "body_q_measured": np.arange(29, dtype=np.float64),
        "left_hand_q_measured": np.arange(7, dtype=np.float64),
        "right_hand_q_measured": np.arange(7, dtype=np.float64),
    }

    frozen = sonic_relay.process_robot_state(
        state,
        now_monotonic_s=10.0,
        heading_setpoint_rad=planner.heading,
        orientation_tracker=tracker,
        freeze_current_upper_body=False,
    )
    planner.message(np.array([0.0, 0.0, 0.15], dtype=np.float32), 0.05)
    sample = tracker.sample(10.05, planner.heading)

    assert frozen is None
    assert sample.actual_yaw_rad == pytest.approx(0.0)
    assert sample.heading_setpoint_rad == pytest.approx(0.0075)
    assert sample.heading_lag_rad == pytest.approx(0.0075)


def test_relay_state_processing_freezes_pose_only_when_explicitly_requested() -> None:
    state = {
        "base_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "body_q_measured": np.arange(29, dtype=np.float64),
        "left_hand_q_measured": np.arange(7, dtype=np.float64),
        "right_hand_q_measured": np.arange(7, dtype=np.float64),
    }

    frozen = sonic_relay.process_robot_state(
        state,
        now_monotonic_s=1.0,
        heading_setpoint_rad=0.0,
        orientation_tracker=None,
        freeze_current_upper_body=True,
    )

    assert frozen is not None
    assert len(frozen.upper_body_position) == 17


def test_relay_orientation_telemetry_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["lavira_sonic_relay.py"])

    args = sonic_relay.parse_args()

    assert args.orientation_telemetry_output == ""
