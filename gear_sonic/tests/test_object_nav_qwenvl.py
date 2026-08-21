"""Behavioral tests for single-cycle RGB-D ObjectNav inference."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import base64
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.inference.lavira.object_nav import (
    ObjectNavCameraError,
    ObjectNavConfig,
    ObjectNavResult,
    ObjectNavRunner,
    QwenVLBBoxClient,
    RGBDSnapshot,
    SensorGatewayRGBDCamera,
    get_qwenvl_policy_prompt,
    validate_object_nav_policy,
)


def snapshot(index: int = 1, depth_mm: int = 2000) -> RGBDSnapshot:
    depth = np.full((5, 5), depth_mm, dtype=np.uint16)
    return RGBDSnapshot(
        rgb_bgr=np.full((5, 5, 3), index, dtype=np.uint8),
        depth_mm=depth.astype(np.float32),
        fx=100.0,
        cx=2.0,
    )


def policy(*, action: str = "NAVIGATE", confidence: float = 0.9) -> dict[str, object]:
    boxable = action == "NAVIGATE"
    return {
        "action": action,
        "bbox_2d": [450, 450, 550, 550] if boxable else None,
        "target": "red chair",
        "target_type": "global_target",
        "confidence": confidence,
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
) -> tuple[ObjectNavResult, FakeCamera, FakePolicyClient]:
    camera = FakeCamera([snapshot(index) for index in range(1, 6)])
    policy_client = FakePolicyClient(policy_value)
    runner = ObjectNavRunner(
        ObjectNavConfig(
            mission="find the chair",
            global_target="chair",
        ),
        camera=camera,
        policy_client=policy_client,
    )
    return runner.run_once(), camera, policy_client


def test_config_preserves_agentnav_automatic_defaults() -> None:
    config = ObjectNavConfig(mission="find chair", global_target="chair")

    assert config.qwenvl_timeout_seconds == 180.0
    assert config.qwenvl_model == "qwen3-vl-32b-instruct"
    assert config.min_confidence == 0.6
    assert config.max_direct_travel == 8.0
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
        "gear_sonic.utils.inference.lavira.object_nav.QwenVLBBoxClient",
        build_qwenvl,
    )
    runner = ObjectNavRunner(
        ObjectNavConfig(
            "find chair",
            "chair",
            qwenvl_model="qwen-test",
            qwenvl_base_url="https://qwen.invalid/v1",
            qwenvl_timeout_seconds=42.0,
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
    result = ObjectNavResult("FAILED", {}, {})

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


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True])
def test_policy_rejects_non_finite_or_boolean_confidence(invalid: object) -> None:
    value = policy()
    value["confidence"] = invalid

    with pytest.raises(ValueError, match="confidence"):
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


def test_gateway_camera_rejects_mismatched_rgbd_shapes() -> None:
    rgb_stream = SensorGatewayRGBDCamera.RGB_STREAM
    depth_stream = SensorGatewayRGBDCamera.DEPTH_STREAM
    frame = SimpleNamespace(attributes={}, source_timestamp_ns=1)
    malformed = SimpleNamespace(
        snapshot=SimpleNamespace(
            frames={rgb_stream: frame, depth_stream: frame},
        ),
        arrays={
            rgb_stream: np.zeros((2, 2, 3), dtype=np.uint8),
            depth_stream: np.zeros((3, 2), dtype=np.uint16),
        },
    )

    with pytest.raises(ObjectNavCameraError, match="shapes do not match"):
        SensorGatewayRGBDCamera._decode(malformed)


def test_qwenvl_client_sends_in_memory_image_and_validates_policy() -> None:
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
    ticks = iter([3.0, 3.25, 4.0, 5.25])
    qwen = QwenVLBBoxClient(client=client, monotonic=lambda: next(ticks))

    result = qwen.locate(
        mission="find chair",
        global_target="chair",
        snapshot=snapshot(),
    )

    assert result["action"] == "NAVIGATE"
    assert result["bbox_2d"] == [450, 450, 550, 550]
    assert calls[0]["model"] == "qwen3-vl-32b-instruct"
    assert calls[0]["timeout"] == 180.0
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["extra_body"] == {"enable_thinking": False}
    messages = calls[0]["messages"]
    assert isinstance(messages, list)
    image_url = messages[0]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert base64.b64decode(image_url.partition(",")[2]).startswith(b"\x89PNG")
    prompt = messages[0]["content"][1]["text"]
    assert '"bbox_2d": [x1, y1, x2, y2] or null' in prompt
    assert "Do not output target_center" in prompt
    assert qwen.last_image_encode_seconds == pytest.approx(0.25)
    assert qwen.last_api_inference_seconds == pytest.approx(1.25)


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


def test_successful_cycle_logs_diagnostics_without_writing_artifacts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="sonic.lavira"):
        result, camera, policy_client = run_with_fakes(tmp_path, policy())

    assert result.outcome == "NAVIGATE"
    assert result.geometry["mean_range"] == 2.0
    assert camera.capture_count == 5
    assert len(policy_client.calls) == 1
    assert np.all(policy_client.calls[0]["snapshot"].rgb_bgr == 3)
    assert list(tmp_path.iterdir()) == []
    assert "object_nav outcome=NAVIGATE" in caplog.text
    assert "policy=" in caplog.text
    assert "geometry=" in caplog.text
    timing = result.geometry["timing_s"]
    assert set(timing) == {
        "camera_rgbd",
        "image_encode",
        "api_inference",
        "postprocess",
        "total",
    }
    assert all(float(value) >= 0.0 for value in timing.values())


def test_da_release_callback_runs_after_rgbd_capture_before_qwen_policy(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class OrderedPolicyClient(FakePolicyClient):
        def locate(self, **kwargs):
            events.append("policy")
            return super().locate(**kwargs)

    runner = ObjectNavRunner(
        ObjectNavConfig("find chair", "chair"),
        camera=FakeCamera([snapshot(index) for index in range(1, 6)]),
        policy_client=OrderedPolicyClient(policy()),
    )

    runner.run_once(rgbd_capture_complete=lambda: events.append("depth_released"))

    assert events == ["depth_released", "policy"]


@pytest.mark.parametrize(
    ("policy_value", "expected"),
    [(policy(action="STOP"), "STOP"), (policy(confidence=0.2), "REJECTED")],
)
def test_stop_and_low_confidence_fail_closed(
    tmp_path: Path, policy_value: dict[str, object], expected: str
) -> None:
    result, _, _ = run_with_fakes(tmp_path, policy_value)

    assert result.outcome == expected
    assert result.error is None


def test_qwenvl_failure_returns_failed_result(
    tmp_path: Path,
) -> None:
    camera = FakeCamera([snapshot(index) for index in range(1, 6)])
    policy_client = FakePolicyClient(error=RuntimeError("Qwen-VL unavailable"))
    runner = ObjectNavRunner(
        ObjectNavConfig("find chair", "chair"),
        camera=camera,
        policy_client=policy_client,
    )

    result = runner.run_once()

    assert result.outcome == "FAILED"
    assert result.error == "Qwen-VL unavailable"
    assert result.geometry["status"] == "failed"
    assert result.geometry["error"] == "Qwen-VL unavailable"
    assert set(result.geometry["timing_s"]) == {
        "camera_rgbd",
        "image_encode",
        "api_inference",
        "postprocess",
        "total",
    }


def test_injected_camera_lifecycle_remains_with_caller(tmp_path: Path) -> None:
    camera = FakeCamera([snapshot(index) for index in range(1, 6)])
    runner = ObjectNavRunner(
        ObjectNavConfig("find chair", "chair"),
        camera=camera,
        policy_client=FakePolicyClient(policy()),
    )

    runner.close()

    assert camera.closed is False
