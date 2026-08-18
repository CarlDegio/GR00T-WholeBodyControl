"""Behavioral tests for single-cycle RGB-D ObjectNav inference."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.utils.inference.object_nav import (
    ComposedRGBDCamera,
    ObjectNavCameraError,
    ObjectNavConfig,
    ObjectNavResult,
    ObjectNavRunner,
    QwenVLBBoxClient,
    RGBDSnapshot,
    get_qwenvl_policy_prompt,
    validate_object_nav_policy,
)


def snapshot(index: int = 1, depth_mm: int = 2000) -> RGBDSnapshot:
    depth = np.full((5, 5), depth_mm, dtype=np.uint16)
    return RGBDSnapshot(
        rgb_bgr=np.full((5, 5, 3), index, dtype=np.uint8),
        depth_raw=depth,
        depth_mm=depth.astype(np.float32),
        fx=100.0,
        fy=100.0,
        cx=2.0,
        cy=2.0,
        depth_scale_m=0.001,
        depth_aligned_to="chest_view",
        timestamp=float(index),
    )


def policy(*, action: str = "NAVIGATE", confidence: float = 0.9) -> dict[str, object]:
    boxable = action == "NAVIGATE"
    return {
        "visual_check": "chair visible",
        "action": action,
        "bbox_2d": [450, 450, 550, 550] if boxable else None,
        "target": "red chair",
        "target_type": "global_target",
        "estimated_distance_m": 2.5 if boxable else None,
        "target_center_normalized": [500.0, 500.0] if boxable else None,
        "target_center_pixel": [2.0, 2.0] if boxable else None,
        "horizontal_offset_pixel": 0.0 if boxable else None,
        "camera_bearing_deg": 0.0 if boxable else None,
        "rotation_direction": "CENTERED" if boxable else None,
        "rotation_angle_deg": 0.0 if boxable else None,
        "confidence": confidence,
        "distance_confidence": 0.7,
        "stop_reasoning": "target reached" if action == "STOP" else "",
    }


def qwen_policy(*, action: str = "NAVIGATE") -> dict[str, object]:
    return {
        "action": action,
        "bbox_2d": [450, 450, 550, 550] if action == "NAVIGATE" else None,
        "target": "red chair",
        "target_type": "global_target",
        "confidence": 0.9,
        "stop_reasoning": "target reached" if action == "STOP" else "",
    }


class FakeCamera:
    def __init__(self, snapshots: list[RGBDSnapshot]):
        self.snapshots = iter(snapshots)
        self.capture_count = 0
        self.closed = False

    def capture_aligned_rgbd(self) -> RGBDSnapshot:
        self.capture_count += 1
        return next(self.snapshots)

    def close(self) -> None:
        self.closed = True


class FakePolicyClient:
    def __init__(
        self,
        policy_value: dict[str, object] | None = None,
        error: Exception | None = None,
    ):
        self.policy_value = policy_value
        self.error = error
        self.calls: list[dict[str, object]] = []

    def locate(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.policy_value is not None
        return self.policy_value


def run_with_fakes(
    tmp_path: Path,
    policy_value: dict[str, object],
    *,
    target_standoff_distance: float = 0.0,
) -> tuple[ObjectNavResult, FakeCamera, FakePolicyClient]:
    camera = FakeCamera([snapshot(index) for index in range(1, 6)])
    policy_client = FakePolicyClient(policy_value)
    runner = ObjectNavRunner(
        ObjectNavConfig(
            mission="find the chair",
            global_target="chair",
            target_standoff_distance=target_standoff_distance,
            output_root=str(tmp_path),
        ),
        camera=camera,
        policy_client=policy_client,
    )
    return runner.run_once(), camera, policy_client


def test_config_preserves_agentnav_automatic_defaults() -> None:
    config = ObjectNavConfig(mission="find chair", global_target="chair")

    assert config.camera_timeout_ms == 3000
    assert config.qwenvl_timeout_seconds == 180.0
    assert config.qwenvl_model == "qwen3-vl-32b-instruct"
    assert config.min_confidence == 0.6
    assert config.target_standoff_distance == 0.0
    assert not hasattr(config, "vision_backend")
    assert not hasattr(config, "model")


def test_default_runner_constructs_only_qwenvl_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = object()
    captured: dict[str, object] = {}

    def build_qwenvl(**kwargs: object) -> object:
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(
        "gear_sonic.utils.inference.object_nav.QwenVLBBoxClient",
        build_qwenvl,
    )
    runner = ObjectNavRunner(
        ObjectNavConfig(
            "find chair",
            "chair",
            qwenvl_model="qwen-test",
            qwenvl_base_url="https://qwen.invalid/v1",
            qwenvl_timeout_seconds=42.0,
            output_root=str(tmp_path),
        ),
        camera=FakeCamera([snapshot(index) for index in range(1, 6)]),
    )

    assert runner.policy_client is marker
    assert captured == {
        "model": "qwen-test",
        "base_url": "https://qwen.invalid/v1",
        "timeout_seconds": 42.0,
    }


def test_object_nav_result_is_immutable() -> None:
    result = ObjectNavResult("FAILED", {}, {"commands": []}, {}, "/tmp/output")

    with pytest.raises(FrozenInstanceError):
        result.outcome = "NAVIGATE"  # type: ignore[misc]


def test_qwenvl_prompt_uses_explicit_inputs_and_image_size() -> None:
    current = replace(
        snapshot(), rgb_bgr=np.zeros((10, 20, 3), dtype=np.uint8), fx=123.5
    )
    prompt = get_qwenvl_policy_prompt(
        'find the "red chair"\nthen stop', "red chair", current
    )

    assert r'find the \"red chair\"\nthen stop' in prompt
    assert "width=20, height=10" in prompt
    assert "Never return STOP merely" in prompt


@pytest.mark.parametrize("key", sorted(policy()))
def test_policy_rejects_every_missing_required_key(key: str) -> None:
    value = policy()
    del value[key]

    with pytest.raises(ValueError, match="schema"):
        validate_object_nav_policy(value)


def test_policy_rejects_extra_keys() -> None:
    value = policy()
    value["reasoning"] = "hidden"

    with pytest.raises(ValueError, match="schema"):
        validate_object_nav_policy(value)


@pytest.mark.parametrize("key", ["confidence", "distance_confidence"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True])
def test_policy_rejects_non_finite_or_boolean_confidence(key: str, invalid: object) -> None:
    value = policy()
    value[key] = invalid

    with pytest.raises(ValueError, match=key):
        validate_object_nav_policy(value)


@pytest.mark.parametrize(
    "bbox", [[500, 100, 500, 900], [600, 100, 500, 900], [100, 900, 500, 100]]
)
def test_policy_rejects_invalid_bbox_corner_ordering(bbox: list[int]) -> None:
    value = policy()
    value["bbox_2d"] = bbox

    with pytest.raises(ValueError, match="bbox"):
        validate_object_nav_policy(value)


@pytest.mark.parametrize(
    "target_type", ["global_target", "intermediate_landmark", "traversable_opening"]
)
def test_policy_accepts_each_supported_target_type(target_type: str) -> None:
    value = policy()
    value["target_type"] = target_type

    assert validate_object_nav_policy(value) == value


def test_policy_rejects_unknown_target_type_and_non_global_stop() -> None:
    value = policy()
    value["target_type"] = "obstacle"
    with pytest.raises(ValueError, match="target_type"):
        validate_object_nav_policy(value)

    value = policy(action="STOP")
    value["target_type"] = "intermediate_landmark"
    with pytest.raises(ValueError, match="global target"):
        validate_object_nav_policy(value)


@pytest.mark.parametrize("direction", ["LEFT", "RIGHT", "CENTERED"])
def test_policy_accepts_each_supported_rotation_direction(direction: str) -> None:
    value = policy()
    value["rotation_direction"] = direction

    assert validate_object_nav_policy(value) == value


def test_policy_rejects_unknown_rotation_direction() -> None:
    value = policy()
    value["rotation_direction"] = "FORWARD"

    with pytest.raises(ValueError, match="rotation_direction"):
        validate_object_nav_policy(value)


def test_composed_camera_rejects_malformed_rgbd_payload() -> None:
    malformed = ImageMessageSchema(
        timestamps={"chest_view": 1.0},
        images={"chest_view": np.zeros((2, 2, 3), dtype=np.uint8)},
        camera_info={"chest_view": {}},
    ).serialize()

    with pytest.raises(ObjectNavCameraError, match="RGB and depth"):
        ComposedRGBDCamera.decode_payload(malformed)


def test_qwenvl_client_sends_local_image_and_validates_policy(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(qwen_policy()))
                    )
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"png bytes")
    ticks = iter([4.0, 5.25])
    qwen = QwenVLBBoxClient(client=client, monotonic=lambda: next(ticks))

    result = qwen.locate(
        image_path=image_path,
        mission="find chair",
        global_target="chair",
        snapshot=snapshot(),
        cwd=tmp_path,
    )

    assert result["action"] == "NAVIGATE"
    assert result["bbox_2d"] == [450, 450, 550, 550]
    assert result["estimated_distance_m"] is None
    assert result["target_center_normalized"] == [500.0, 500.0]
    assert result["target_center_pixel"] == [2.0, 2.0]
    assert result["horizontal_offset_pixel"] == 0.0
    assert result["camera_bearing_deg"] == 0.0
    assert result["rotation_direction"] == "CENTERED"
    assert result["rotation_angle_deg"] == 0.0
    assert calls[0]["model"] == "qwen3-vl-32b-instruct"
    assert calls[0]["timeout"] == 180.0
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["extra_body"] == {"enable_thinking": False}
    messages = calls[0]["messages"]
    assert isinstance(messages, list)
    image_url = messages[0]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    prompt = messages[0]["content"][1]["text"]
    assert '"bbox_2d": [x1, y1, x2, y2] or null' in prompt
    assert "Do not output target_center" in prompt
    assert qwen.last_auth_check_seconds == 0.0
    assert qwen.last_api_inference_seconds == pytest.approx(1.25)


def test_qwenvl_derives_right_turn_from_bbox_and_camera_intrinsics(tmp_path: Path) -> None:
    value = qwen_policy()
    value["bbox_2d"] = [700, 400, 800, 600]

    class FakeCompletions:
        def create(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))]
            )

    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"png bytes")
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    result = QwenVLBBoxClient(client=client).locate(
        image_path=image_path,
        mission="find chair",
        global_target="chair",
        snapshot=replace(snapshot(), fx=2.0),
        cwd=tmp_path,
    )

    assert result["target_center_pixel"] == [3.0, 2.0]
    assert result["horizontal_offset_pixel"] == 1.0
    assert result["camera_bearing_deg"] == pytest.approx(26.565, abs=0.001)
    assert result["rotation_direction"] == "RIGHT"
    assert result["rotation_angle_deg"] == pytest.approx(26.565, abs=0.001)


def test_qwenvl_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        QwenVLBBoxClient()


def test_qwenvl_client_uses_isolated_http_proxy_and_ignores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeHttpClient:
        def __init__(self, **kwargs: object):
            captured["httpx"] = kwargs

    class FakeOpenAI:
        def __init__(self, **kwargs: object):
            captured["openai"] = kwargs

    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7890/")
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=FakeHttpClient))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    QwenVLBBoxClient(
        api_key="test-key",
        proxy_url="http://127.0.0.1:7890",
    )

    assert captured["httpx"] == {
        "proxy": "http://127.0.0.1:7890",
        "trust_env": False,
    }
    assert captured["openai"]["http_client"].__class__ is FakeHttpClient


def test_successful_cycle_captures_five_frames_and_writes_diagnostics(
    tmp_path: Path,
) -> None:
    result, camera, policy_client = run_with_fakes(
        tmp_path, policy(), target_standoff_distance=0.5
    )

    assert result.outcome == "NAVIGATE"
    assert len(result.commands["commands"]) == 2
    assert result.geometry["travel"] == 1.5
    assert camera.capture_count == 5
    assert len(policy_client.calls) == 1
    assert policy_client.calls[0]["snapshot"].timestamp == 3.0
    assert "qwen_prediction" in result.geometry
    output_dir = Path(result.output_dir)
    assert sorted(path.name for path in output_dir.glob("depth_raw_*.png")) == [
        "depth_raw_01.png",
        "depth_raw_02.png",
        "depth_raw_03.png",
        "depth_raw_04.png",
        "depth_raw_05.png",
    ]
    timing = result.geometry["timing_s"]
    assert set(timing) == {
        "camera_rgbd",
        "image_io",
        "auth_check",
        "api_inference",
        "postprocess",
        "total",
    }
    assert all(float(value) >= 0.0 for value in timing.values())
    saved_geometry = json.loads(
        (output_dir / "object_nav_geometry.json").read_text(encoding="utf-8")
    )
    assert saved_geometry["timing_s"] == timing
    assert json.loads((output_dir / "camera_info.json").read_text())["frame_count"] == 5
    assert json.loads((output_dir / "object_nav_commands.json").read_text()) == result.commands


def test_object_nav_warmup_runs_iteration_zero_without_returning_motion(
    tmp_path: Path,
) -> None:
    camera = FakeCamera([snapshot(index) for index in range(1, 6)])
    runner = ObjectNavRunner(
        ObjectNavConfig("find chair", "chair", output_root=str(tmp_path)),
        camera=camera,
        policy_client=FakePolicyClient(policy()),
    )

    warmup = runner.warmup()

    assert warmup.outcome == "NAVIGATE"
    assert Path(warmup.output_dir).name == "iteration_0000"


@pytest.mark.parametrize(
    ("policy_value", "expected"),
    [(policy(action="STOP"), "STOP"), (policy(confidence=0.2), "REJECTED")],
)
def test_stop_and_low_confidence_fail_closed(
    tmp_path: Path, policy_value: dict[str, object], expected: str
) -> None:
    result, _, _ = run_with_fakes(tmp_path, policy_value)

    assert result.outcome == expected
    assert result.commands == {"commands": []}
    assert result.error is None


def test_qwenvl_failure_returns_failed_result_and_empty_commands(
    tmp_path: Path,
) -> None:
    camera = FakeCamera([snapshot(index) for index in range(1, 6)])
    policy_client = FakePolicyClient(error=RuntimeError("Qwen-VL unavailable"))
    runner = ObjectNavRunner(
        ObjectNavConfig("find chair", "chair", output_root=str(tmp_path)),
        camera=camera,
        policy_client=policy_client,
    )

    result = runner.run_once()

    assert result.outcome == "FAILED"
    assert result.commands == {"commands": []}
    assert result.error == "Qwen-VL unavailable"
    assert result.geometry["status"] == "failed"
    assert result.geometry["error"] == "Qwen-VL unavailable"
    assert set(result.geometry["timing_s"]) == {
        "camera_rgbd",
        "image_io",
        "auth_check",
        "api_inference",
        "postprocess",
        "total",
    }


def test_injected_camera_lifecycle_remains_with_caller(tmp_path: Path) -> None:
    camera = FakeCamera([snapshot(index) for index in range(1, 6)])
    runner = ObjectNavRunner(
        ObjectNavConfig("find chair", "chair", output_root=str(tmp_path)),
        camera=camera,
        policy_client=FakePolicyClient(policy()),
    )

    runner.close()

    assert camera.closed is False
