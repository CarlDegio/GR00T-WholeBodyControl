from __future__ import annotations

import io
import json
from pathlib import Path
import queue
import re
import shlex
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import requests
import yaml

from baselinelavira import DEFAULT_PROFILE, build_profile
from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.scripts.cleanup_inference import NAVDP_PROCESS_PATTERNS
from gear_sonic.scripts.launch_inference import (
    build_navdp_planner_command,
    build_navdp_server_background_command,
    build_navdp_server_command,
    load_inference_launch_config,
)
from gear_sonic.utils.inference.iplanner.client import (
    encode_frames,
    request_trajectory,
    reset_iplanner,
)
from gear_sonic.utils.inference.iplanner.config import (
    load_iplanner_server_config,
    navigation_backend,
)
from gear_sonic.utils.inference.navdp.gateway import load_navdp_planner_config
from gear_sonic.utils.inference.navdp.navigation import Pose2D
from gear_sonic.utils.inference.navdp.runtime import (
    _consume_inference_results,
    _NavDPRuntimeState,
    _refresh_gateway_camera,
)


def _write_profile(tmp_path, payload=None):
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(payload or build_profile()))
    return path


def _response(payload):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)


def _trajectory():
    return [[[float(x), 0.25 * x, 0.0] for x in range(6)]]


def test_baseline_inherits_all_existing_components_and_endpoints(tmp_path):
    original = yaml.safe_load(DEFAULT_PROFILE.read_text())
    # Simulate another round of camera, alignment and SONIC changes in Groot.
    original["components"]["navdp"]["rgb_stream"] = "camera/ego_view"
    original["components"]["base_pose"]["raw_forward_recenter_yaw_speed_rad_s"] = 0.23
    original["components"]["deploy"]["policy_variant"] = "low_latency"
    path = tmp_path / "current.yaml"
    path.write_text(yaml.safe_dump(original))
    before = path.read_bytes()
    baseline = build_profile(path)

    assert path.read_bytes() == before
    assert baseline["profile"] == "baselinelavira"
    assert baseline["endpoints"] == original["endpoints"]
    assert baseline["ros_topics"] == original["ros_topics"]
    for name, component in original["components"].items():
        assert baseline["components"][name] == component
    assert set(baseline["components"]) - set(original["components"]) == {
        "navigation_policy", "iplanner", "iplanner_navigation",
    }
    merged_path = _write_profile(tmp_path, baseline)
    assert navigation_backend(load_runtime_profile(merged_path)) == "iplanner"
    assert load_navdp_planner_config(str(merged_path)).rgb_stream == "camera/ego_view"
    assert navigation_backend(load_runtime_profile()) == "xnavdp"


def test_baseline_server_uses_original_lavira_and_shared_navigation_worker(tmp_path):
    root = tmp_path / "LaViRA source"
    path = _write_profile(tmp_path, build_profile(
        root=str(root), checkpoint="weights/model.pth", device="cpu", conda_env="planner env",
    ))
    config = load_inference_launch_config(path)
    command = build_navdp_server_command(config)
    args = shlex.split(command)
    assert "gear_sonic.utils.inference.iplanner.server" in args
    assert args[args.index("--root") + 1] == str(root)
    assert args[args.index("--checkpoint") + 1] == str(root / "weights/model.pth")
    assert args[args.index("--config") + 1] == str(root / "configs/iplanner.yaml")
    assert args[args.index("--device") + 1] == "cpu"
    assert args[args.index("-n") + 1] == "planner env"
    assert "eval.src.policy_server" not in command
    assert "--embodiment" not in command and "--real" not in command
    worker = build_navdp_planner_command(config, DEFAULT_PROFILE.parents[2])
    assert "gear_sonic.utils.inference.navdp.service" in worker
    assert str(path) in worker
    background = build_navdp_server_background_command(config)
    assert "[iPlanner server]" in background
    subprocess.run(["bash", "-n", "-c", background], check=True)
    standard = build_navdp_server_command(load_inference_launch_config())
    assert "eval.src.policy_server" in standard
    assert "iplanner.server" not in standard


def test_iplanner_prerequisite_paths_do_not_require_xnavdp_weights(tmp_path):
    path = _write_profile(tmp_path, build_profile(root=str(tmp_path), checkpoint="missing.pth"))
    settings = load_iplanner_server_config(load_runtime_profile(path))
    assets = dict((label, path) for path, label in settings.required_assets())
    assert set(assets) == {"LaViRA iPlanner server", "iPlanner checkpoint", "iPlanner model config"}
    assert assets["iPlanner checkpoint"] == tmp_path / "missing.pth"
    assert not assets["iPlanner checkpoint"].exists()
    assert all("x-navdp" not in str(path) for path in assets.values())


def test_cleanup_matches_only_the_owned_iplanner_entry_point():
    own = "python -m gear_sonic.utils.inference.iplanner.server --root /source"
    external = "python iplanner_server.py --port 8888"
    assert any(re.search(pattern, own) for pattern in NAVDP_PROCESS_PATTERNS)
    assert not any(re.search(pattern, external) for pattern in NAVDP_PROCESS_PATTERNS)


def test_depth_encoding_preserves_units_and_saturates_without_wrapping():
    rgb = np.array([[[1, 20, 250]] * 8], dtype=np.uint8)
    depth = np.array([[0.05, 1.2345, 5.5, 10.0, np.nan, np.inf, -1.0, 0.0]])
    original = depth.copy()
    rgb_bytes, depth_bytes = encode_frames(rgb, depth)
    decoded_rgb = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    decoded_depth = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    assert rgb_bytes.startswith(b"\x89PNG")
    np.testing.assert_array_equal(decoded_rgb, rgb)
    np.testing.assert_array_equal(decoded_depth, [[500, 12345, 55000, 65535, 0, 0, 0, 0]])
    np.testing.assert_array_equal(depth, original)


def test_request_uses_lavira_contract_and_preserves_left_axis(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return _response({"trajectory": _trajectory(), "all_values": [[0.8]]})

    monkeypatch.setattr(requests, "post", post)
    result = request_trajectory(
        "http://planner/", np.zeros((2, 3, 3), np.uint8), np.ones((2, 3)),
        (-8.0, 9.0), timeout=1.75,
    )
    url, request = calls[0]
    assert url == "http://planner/pointgoal_step"
    assert request["timeout"] == 1.75
    assert json.loads(request["data"]["goal_data"]) == {"goal_x": [-5.0], "goal_y": [5.0]}
    assert set(request["data"]) == {"goal_data"}
    assert request["files"]["image"][2] == request["files"]["depth"][2] == "image/png"
    np.testing.assert_array_equal(result, np.asarray(_trajectory())[0, :, :2])


@pytest.mark.parametrize("depth_value", [0.0, -1.0, float("nan"), float("inf")])
def test_all_invalid_depth_is_rejected_before_model_inference(monkeypatch, depth_value):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: pytest.fail("invalid depth reached the model"))
    with pytest.raises(ValueError, match="valid positive depth"):
        request_trajectory("http://planner", np.zeros((2, 3, 3), np.uint8),
                           np.full((2, 3), depth_value), (3.0, 0.0))


@pytest.mark.parametrize("trajectory", [
    [], [[]], [[[0.0, 0.0, 0.0]]],
    [[[0.0, 0.0]] * 6], [[[0.0, 0.0, 0.0]] * 6] * 2,
    [[[float("nan"), 0.0, 0.0]] * 6], [[[0.0, 0.0, float("inf")]] * 6],
])
def test_invalid_trajectories_fail_before_control(monkeypatch, trajectory):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _response({"trajectory": trajectory}))
    with pytest.raises(ValueError, match="invalid iPlanner trajectory"):
        request_trajectory("http://planner", np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), (2, 0))


def test_transport_failure_remains_a_failure(monkeypatch):
    def post(*args, **kwargs):
        raise requests.Timeout("inference timed out")

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(requests.Timeout):
        request_trajectory("http://planner", np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), (2, 0))


def test_camera_reset_preserves_camera_and_selects_iplanner(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return _response({"algo": "iplanner"})

    monkeypatch.setattr(requests, "post", post)
    frame = SimpleNamespace(
        rgb=np.zeros((2, 2, 3), np.uint8), depth_m=np.ones((2, 2)),
        camera_info={"fx": 300, "fy": 301, "cx": 12, "cy": 13},
        source_timestamp_s=123.4,
    )
    state = _NavDPRuntimeState()
    for _ in range(2):
        _refresh_gateway_camera(
            state, gateway=SimpleNamespace(poll_camera=lambda: frame),
            navdp_endpoint="http://planner", config=load_navdp_planner_config(),
            report_event=lambda *a, **kw: None, planner_backend="iplanner",
        )
    assert state.navdp_initialized
    assert state.latest_rgb is frame.rgb and state.latest_depth is frame.depth_m
    assert state.latest_camera_timestamp == 123.4
    assert len(calls) == 1
    assert calls[0][1]["json"] == {
        "intrinsic": [[300.0, 0.0, 12.0], [0.0, 301.0, 13.0], [0.0, 0.0, 1.0]],
        "batch_size": 1, "stop_threshold": 0.1,
    }


def test_reset_rejects_another_model_at_the_endpoint(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _response({"algo": "navdp"}))
    with pytest.raises(ValueError, match="expected a LaViRA iPlanner"):
        reset_iplanner("http://planner", {"fx": 300, "fy": 300, "cx": 0, "cy": 0})


def test_xnavdp_world_reference_and_generation_guard_remain_unchanged():
    results = queue.Queue()
    state = _NavDPRuntimeState(mode="nav_goal", generation=2, segment_id=3)
    trajectory = np.asarray(_trajectory(), dtype=np.float32)[0, :, :2]
    pose = Pose2D(10.0, 20.0, np.pi / 2)
    results.put((1, 3, trajectory * 999, pose, None, 0.1))
    results.put((2, 3, trajectory, pose, None, 0.1))
    _consume_inference_results(
        state, results, send_metrics=lambda *a, **kw: None,
        send_status=lambda *a, **kw: None, report_event=lambda *a, **kw: None,
    )
    assert state.mpc_reference_version == 1
    np.testing.assert_array_equal(state.trajectory, trajectory)
    np.testing.assert_allclose(state.mpc_reference[0], [9.25, 23.0])
    np.testing.assert_allclose(state.mpc_reference[-1], [8.75, 25.0])


def test_client_matches_original_lavira_flask_handlers(monkeypatch):
    """Exercise the original HTTP decoding with a fake model, without weights."""
    import importlib.util
    import sys
    import types
    from urllib.parse import urlsplit

    root = Path(build_profile()["components"]["iplanner"]["root"])
    if not (root / "iplanner_server.py").is_file():
        pytest.skip("local Uni-LaViRA source is not available")
    observations = []

    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class Agent:
        def __init__(self, intrinsic, **kwargs):
            observations.append(intrinsic)

        def step_pointgoal(self, depth, goal):
            observations.append((depth, goal))
            return None, Tensor(_trajectory()), Tensor([[0.8]])

    module = types.ModuleType("iplanner_agent")
    module.IPlannerAgent = Agent
    monkeypatch.setitem(sys.modules, "iplanner_agent", module)
    monkeypatch.setattr(sys, "argv", ["iplanner_server.py", "--device", "cpu"])
    spec = importlib.util.spec_from_file_location("original_lavira_iplanner", root / "iplanner_server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    client = server.app.test_client()

    def post(url, **kwargs):
        if "json" in kwargs:
            response = client.post(urlsplit(url).path, json=kwargs["json"])
        else:
            data = dict(kwargs["data"])
            for name, (filename, payload, mime) in kwargs["files"].items():
                data[name] = (io.BytesIO(payload), filename, mime)
            response = client.post(urlsplit(url).path, data=data, content_type="multipart/form-data")
        assert response.status_code == 200, response.get_data(as_text=True)
        return _response(response.get_json())

    monkeypatch.setattr(requests, "post", post)
    reset_iplanner("http://planner", {"fx": 300, "fy": 301, "cx": 1, "cy": 2})
    result = request_trajectory(
        "http://planner", np.zeros((2, 3, 3), np.uint8), np.full((2, 3), 2.3456), (4.0, -1.5),
    )
    np.testing.assert_allclose(observations[1][0], np.full((1, 2, 3, 1), 2.3456))
    np.testing.assert_array_equal(observations[1][1], [[4.0, -1.5, 0.0]])
    np.testing.assert_array_equal(result, np.asarray(_trajectory())[0, :, :2])
