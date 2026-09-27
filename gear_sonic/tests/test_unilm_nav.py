import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import zmq

from gear_sonic.runtime.gateway.control import ControlGatewayCore, NavigationControlState
from gear_sonic.runtime.gateway.services.control import ControlGatewayRuntime
from gear_sonic.runtime.profile import load_component_config, load_runtime_profile
from gear_sonic.utils.inference.base_pose.agent import load_base_pose_config
from gear_sonic.utils.inference.base_pose.sensor import AlignedRGBDSnapshot
from gear_sonic.utils.inference.base_pose.unilm import (
    Observation,
    Pose,
    PoseTracker,
    UniLMClient,
    UniLMConfig,
    UniLMError,
    lift_point,
    pose_from_odometry,
    robot_geometry,
    wrap,
)
from gear_sonic.utils.inference.base_pose.unilm_runtime import CameraMonitor, ObservationMemory, UniLMRuntime


def observation(timestamp=1.0, depth=1000, pose=Pose(0, 0, 0), camera="chest"):
    return Observation(
        AlignedRGBDSnapshot(
            np.full((48, 64, 3), 120, np.uint8),
            np.full((48, 64), depth, np.uint16),
            40.0,
            40.0,
            32.0,
            24.0,
            0.001,
            "chest_view" if camera == "chest" else "ego_view",
            "realsense" if camera == "chest" else "orbbec",
            timestamp,
        ),
        pose,
    )


def model(responses, calls):
    outputs = iter(responses)

    def create(**kw):
        calls.append(kw)
        content = json.dumps(next(outputs))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.parametrize("camera,height,pitch", [("chest", 1.0, 3), ("head", 1.1, 38)])
def test_camera_height_pitch_and_invalid_depth(camera, height, pitch):
    c, base = UniLMConfig(camera=camera, camera_height_m=height), load_base_pose_config()
    p = lift_point(observation(camera=camera), (500, 500), c, base)
    assert p == pytest.approx([math.cos(math.radians(pitch)), 0, height - math.sin(math.radians(pitch))])
    with pytest.raises(UniLMError, match="no valid depth"):
        lift_point(observation(depth=0, camera=camera), (500, 500), c, base)
    with pytest.raises(UniLMError, match="0,1000"):
        lift_point(observation(camera=camera), (1000, 500), c, base)


def test_chest_projection_uses_chest_offsets_and_yaw_and_rejects_head_depth():
    c, base = UniLMConfig(), load_base_pose_config()
    base.dual_chest_camera_forward_offset_m = 0.12
    base.dual_chest_camera_lateral_offset_m = 0.04
    base.dual_chest_camera_yaw_deg = 90.0
    p = lift_point(observation(), (500, 500), c, base)
    assert p == pytest.approx([0.12, 0.04 + math.cos(math.radians(3)), 1.0 - math.sin(math.radians(3))])
    with pytest.raises(UniLMError, match="not aligned to chest_view"):
        lift_point(observation(camera="head"), (500, 500), c, base)


def test_three_calls_use_selected_depth_pose_and_geometric_heading(tmp_path):
    calls = []
    client = UniLMClient(
        UniLMConfig(), model([{"selected_id": 1}, {"u": 500, "v": 500}, {"x": 0.1, "y": 0.2, "theta": 2.8}], calls)
    )
    observations = [observation(depth=1800), observation(depth=800, pose=Pose(10, -2, math.pi / 2))]
    goal, details = client.infer(
        observations,
        "collect the bottle",
        {},
        load_base_pose_config(),
        deadline=time.monotonic() + 20,
        cancel=threading.Event(),
        directory=tmp_path,
    )
    assert len(calls) == 3
    assert [len(call["messages"][1]["content"]) - 1 for call in calls] == [2, 1, 1]
    assert all(call["model"] == "qwen3.8-flash" for call in calls)
    assert all(call["extra_body"] == {"enable_thinking": False} for call in calls)
    assert (goal.x, goal.y) == pytest.approx((9.8, -1.9))
    target_x = 0.8 * math.cos(math.radians(3))
    assert details["target_local_m"][0] == pytest.approx(target_x)
    assert details["target_local_m"][2] == pytest.approx(1.0 - 0.8 * math.sin(math.radians(3)))
    assert goal.yaw == pytest.approx(wrap(math.pi / 2 + math.atan2(-0.2, target_x - 0.1)))
    assert (tmp_path / "prediction.json").is_file()


def test_format_retry_does_not_restart_the_pipeline(tmp_path):
    calls = []
    client = UniLMClient(
        UniLMConfig(),
        model(
            [{"selected_id": 99}, {"selected_id": 0}, {"u": 500, "v": 500}, {"x": 0.1, "y": 0, "theta": 0}], calls
        ),
    )
    client.infer(
        [observation()],
        "pick bottle",
        {},
        load_base_pose_config(),
        deadline=time.monotonic() + 20,
        cancel=threading.Event(),
        directory=tmp_path,
    )
    assert len(calls) == 4
    assert calls[0]["messages"] == calls[1]["messages"]


def test_missing_depth_ends_without_pose_call(tmp_path):
    calls = []
    client = UniLMClient(UniLMConfig(), model([{"selected_id": 0}, {"u": 500, "v": 500}], calls))
    with pytest.raises(UniLMError, match="no valid depth"):
        client.infer(
            [observation(depth=0)],
            "pick bottle",
            {},
            load_base_pose_config(),
            deadline=time.monotonic() + 20,
            cancel=threading.Event(),
            directory=tmp_path,
        )
    assert len(calls) == 2


def test_cancel_prevents_cloud_calls(tmp_path):
    calls = []
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(UniLMError, match="cancelled"):
        UniLMClient(UniLMConfig(), model([], calls)).infer(
            [observation()],
            "pick bottle",
            {},
            load_base_pose_config(),
            deadline=time.monotonic() + 20,
            cancel=cancel,
            directory=tmp_path,
        )
    assert not calls


def test_nominal_imu_frame_is_not_treated_as_pelvis():
    geometry, transform = robot_geometry(UniLMConfig().urdf_path, 1.1)
    p = pose_from_odometry([2, 3, 0, 0, 0, 0, 1], transform)
    assert (p.x, p.y) == pytest.approx((2 + transform[0, 3], 3 + transform[1, 3]))
    assert abs(transform[2, 3]) > 0.3
    assert geometry["arms"]["left"]["shoulder_xyz_m"][2] < 1.1
    with pytest.raises(UniLMError):
        pose_from_odometry([0, 0, 0, 0, 0, 0, 0], transform)


def test_camera_switch_preserves_robot_standing_geometry_and_odometry_transform():
    head, head_transform = robot_geometry(UniLMConfig().urdf_path, 1.1)
    chest, chest_transform = robot_geometry(UniLMConfig().urdf_path, 1.0)
    assert head.pop("camera_height_m") == 1.1
    assert chest.pop("camera_height_m") == 1.0
    assert chest == head
    np.testing.assert_array_equal(chest_transform, head_transform)


@pytest.mark.parametrize("camera", ["chest", "head"])
def test_camera_monitor_reads_selected_gateway_rgbd_and_source_synchronized_pose(monkeypatch, camera):
    from gear_sonic.runtime.gateway.sensor import SensorGatewayCore
    from gear_sonic.runtime.gateway.sensor_client import SensorGatewayClient
    from gear_sonic.tests.test_sensor_gateway_client import _RpcThread
    import gear_sonic.utils.inference.base_pose.unilm_runtime as module

    context = zmq.Context()
    core = SensorGatewayCore(slot_count=8, history_size=16)
    endpoint = f"inproc://unilm-sensors-{id(core)}"
    monkeypatch.setattr(
        module, "SensorGatewayClient", lambda address, **kw: SensorGatewayClient(address, context=context, **kw)
    )
    server = _RpcThread(context, endpoint, core)
    stamp = time.time_ns()
    frames = {}
    calibrations = {}
    for source, rgb_value, depth_value, focal_length, depth_scale in (
        ("head", 120, 1000, 40, 0.001),
        ("chest", 220, 1800, 60, 0.002),
    ):
        frame = observation(depth=depth_value, camera=source).frame
        frame.rgb.fill(rgb_value)
        frames[source] = frame
        camera_info = {
            "width": 64,
            "height": 48,
            "fx": focal_length,
            "fy": focal_length,
            "cx": 32,
            "cy": 24,
            "depth_scale_m": depth_scale,
            "depth_aligned_to": frame.depth_aligned_to,
            "depth_source": frame.depth_source,
        }
        calibrations[source] = camera_info
        rgb_stream = f"camera/{frame.depth_aligned_to}"
        for stream, array in ((rgb_stream, frame.rgb), (rgb_stream + "_depth", frame.depth_raw)):
            core.publish_array(
                stream,
                array,
                source_timestamp_ns=stamp,
                source_clock="camera_unix",
                attributes={"camera_info": camera_info},
            )
    core.publish_array(
        "ros/odometry",
        np.array([2, 3, 0, 0, 0, 0, 1], np.float64),
        source_timestamp_ns=stamp + 20_000_000,
        source_clock="ros_time",
        attributes={"child_frame_id": "body"},
    )
    monitor = CameraMonitor(
        SimpleNamespace(endpoint_uri=lambda name: endpoint),
        load_base_pose_config(),
        UniLMConfig(camera=camera),
        np.eye(4),
    )
    server.start()
    monitor.thread.start()
    try:
        value = monitor.observations.get(timeout=2)
        assert value.pose == Pose(2, 3, 0)
        assert value.frame.timestamp == pytest.approx(stamp * 1e-9)
        assert value.frame.depth_aligned_to == frames[camera].depth_aligned_to
        assert value.frame.depth_source == frames[camera].depth_source
        assert value.frame.depth_scale_m == calibrations[camera]["depth_scale_m"]
        assert value.frame.fx == calibrations[camera]["fx"]
        assert value.frame.fy == calibrations[camera]["fy"]
        np.testing.assert_array_equal(value.frame.rgb, frames[camera].rgb)
        np.testing.assert_array_equal(value.frame.depth_raw, frames[camera].depth_raw)
        assert monitor.current()[0] == Pose(2, 3, 0)
    finally:
        monitor.close()
        server.close()
        core.close()
        context.term()


def test_navigation_memory_freezes_and_rejects_another_segment():
    m = ObservationMemory()
    m.start((7, 1, 9))
    for i in range(8):
        m.observe(observation(i))
    m.finish((7, 1, 10), True)
    assert not m.frozen
    m.finish((7, 1, 9), True)
    m.observe(observation(100))
    assert [o.frame.timestamp for o in m.frozen[(7, 1, 9)]] == list(range(3, 8))
    m.start((7, 2, 11))
    assert [o.frame.timestamp for o in m.frozen[(7, 1, 9)]] == list(range(3, 8))
    m.clear()
    assert not m.frames
    assert not m.frozen


def test_navigation_history_excludes_failed_later_and_duplicate_frames():
    m = ObservationMemory()
    m.start((7, 1, 6))
    m.observe(observation(100))
    m.finish((7, 1, 6), True)
    m.start((7, 2, 8))
    m.observe(observation(110))
    m.finish((7, 2, 8), False)
    m.start((7, 3, 15), observation(100))
    m.observe(observation(120))
    m.finish((7, 3, 15), True)
    m.start((7, 4, 18))
    m.observe(observation(130))
    m.finish((7, 4, 18), True)
    assert [o.frame.timestamp for o in m.history_through((7, 3, 15))] == [100, 120]
    assert not m.history_through((7, 2, 8))
    assert not m.history_through((7, 3, 16))
    assert not m.history_through((8, 3, 15))


def test_tracker_latches_position_before_final_yaw_and_never_reapproaches():
    tracker = PoseTracker(Pose(0, 0, math.pi / 2), UniLMConfig())
    # At exactly 5 cm we have not entered final alignment yet.
    velocity, done, diagnostics = tracker.update(Pose(-0.05, 0, 0.2), 0, 0)
    assert velocity[2] < 0 and not done
    assert not diagnostics["position_acquired"]
    # Reproduce crossing 15 cm while still far from the final yaw. Even when
    # the approach bearing would reverse the turn, position stays acquired.
    for i, distance in enumerate((0.049, 0.056, 0.048, 0.149, 0.15, 0.157, 0.3, 0.11), start=1):
        velocity, done, diagnostics = tracker.update(Pose(-distance, 0, 0.2), i * 0.05, 0)
        assert velocity == (0, 0, 0) and not done
        assert diagnostics["position_acquired"]
        assert diagnostics["phase"] == "position_settling"
    velocity, done, diagnostics = tracker.update(Pose(-0.3, 0, 0.2), 1.05, 0)
    assert velocity[:2] == (0, 0) and velocity[2] > 0 and not done
    assert diagnostics["position_acquired"]
    assert diagnostics["phase"] == "final_yaw"


@pytest.mark.parametrize("sign", [-1, 1])
def test_tracker_turn_uses_actual_error_when_setpoint_crosses_target(sign):
    tracker = PoseTracker(Pose(0, 0, 0), UniLMConfig())
    tracker.update(Pose(0, 0, -sign * math.radians(10)), 0, 0)
    # The failed run alternated turn direction as heading lag crossed the
    # physical error. A setpoint at/past the target must not stop/reverse yaw.
    for i, lag_degrees in enumerate((9.8, 10.2, 10.0, 9.8, 10.2)):
        lag = sign * math.radians(lag_degrees)
        velocity, done, diagnostics = tracker.update(
            Pose(0, 0, -sign * math.radians(10)), 1.0 + i * 0.05, lag
        )
        assert velocity == pytest.approx((0, 0, sign * math.radians(15)))
        assert not done
        assert diagnostics["phase"] == "final_yaw"
        assert diagnostics["heading_lag_rad"] == lag


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("error_degrees, speed", [(3, 0.1), (45, 0.3)])
def test_tracker_actual_error_turn_preserves_speed_bounds(sign, error_degrees, speed):
    tracker = PoseTracker(Pose(0, 0, 0), UniLMConfig(yaw_tolerance_deg=1.0))
    tracker.update(Pose(0, 0, -sign * math.radians(error_degrees)), 0, 0)
    velocity, done, _ = tracker.update(
        Pose(0, 0, -sign * math.radians(error_degrees)), 1.0, sign * math.radians(50)
    )
    assert velocity == pytest.approx((0, 0, sign * speed))
    assert not done


def test_tracker_settles_using_actual_yaw_despite_heading_lag():
    tracker = PoseTracker(Pose(0, 0, 0), UniLMConfig())
    for now, lag_degrees in ((0, 20), (0.5, -20), (1.0, 20), (1.5, -20), (2.0, 20)):
        velocity, done, diagnostics = tracker.update(
            Pose(0, 0, math.radians(6.9)), now, math.radians(lag_degrees)
        )
        assert velocity == (0, 0, 0)
        assert done == (now == 2.0)
        assert diagnostics["phase"] == (
            "position_settling" if now < 1.0 else "complete" if done else "settling"
        )


def test_tracker_holds_before_walking_despite_heading_lag():
    tracker = PoseTracker(Pose(1, 0, math.pi / 2), UniLMConfig())
    for i in range(20):
        velocity, done, diagnostics = tracker.update(Pose(0, 0, 0), i * 0.05, math.radians(20))
        assert velocity == (0, 0, 0) and not done
        assert diagnostics["phase"] == "approach_settling"
    velocity, done, diagnostics = tracker.update(Pose(0, 0, 0), 1.0, math.radians(20))
    assert velocity == (0.4, 0, 0)
    assert not done
    assert diagnostics["phase"] == "walk"


def test_tracker_finishes_yaw_without_rechecking_distance():
    tracker = PoseTracker(Pose(0, 0, math.pi / 2), UniLMConfig())
    velocity, done, _ = tracker.update(Pose(-0.04, 0, 0), 0, 0)
    assert velocity == (0, 0, 0) and not done
    velocity, done, _ = tracker.update(Pose(-0.16, 0, 0), 1.0, 0)
    assert velocity[2] > 0 and not done
    velocity, done, diagnostics = tracker.update(Pose(-0.16, 0, math.pi / 2), 1.05, 0)
    assert velocity == (0, 0, 0) and not done
    assert diagnostics["phase"] == "settling"
    # A yaw disturbance restarts the stability window without reacquiring XY.
    velocity, done, diagnostics = tracker.update(Pose(-0.2, 0, math.pi / 2 - 0.2), 1.5, 0)
    assert velocity[:2] == (0, 0) and velocity[2] > 0 and not done
    assert diagnostics["phase"] == "final_yaw"
    for now in (1.6, 2.1, 2.61):
        velocity, done, diagnostics = tracker.update(Pose(-0.16, 0, math.pi / 2), now, 0)
        assert velocity == (0, 0, 0)
        assert done == (now == 2.61)
    assert done
    assert diagnostics["distance_m"] > 0.15


@pytest.mark.parametrize("distance", [0.06, 0.19, 0.4, 0.8])
def test_tracker_limits_straight_steps_and_waits_before_restarting(distance):
    tracker = PoseTracker(Pose(distance, 0, 0), UniLMConfig())
    for i in range(71):
        now = i * 0.05
        velocity, done, diagnostics = tracker.update(Pose(0, 0, 0), now, 0)
        walking = 1.0 <= now < 2.0 or now >= 3.5
        assert velocity == ((0.4, 0, 0) if walking else (0, 0, 0))
        assert not done
        assert diagnostics["phase"] == (
            "walk" if walking else "walk_pause" if 2.0 <= now < 2.5 else "approach_settling"
        )
    velocity, done, diagnostics = tracker.update(Pose(distance - 0.04, 0, 0), 3.55, 0)
    assert velocity == (0, 0, 0) and not done
    assert diagnostics["position_acquired"]
    assert diagnostics["phase"] == "position_settling"


def test_tracker_never_turns_while_walking_and_uses_pose_at_end_of_pause():
    tracker = PoseTracker(Pose(1, 0, math.pi / 2), UniLMConfig())
    velocity, done, diagnostics = tracker.update(Pose(0, 0, 0.3), 0, 0)
    assert velocity[:2] == (0, 0) and velocity[2] < 0 and not done
    assert diagnostics["phase"] == "approach"
    for i in range(20):
        velocity, _, _ = tracker.update(Pose(0, 0, 0), 0.5 + i * 0.05, 0)
        assert velocity == (0, 0, 0)
    velocity, _, _ = tracker.update(Pose(0, 0, 0), 1.5, 0)
    assert velocity == (0.4, 0, 0)
    velocity, done, diagnostics = tracker.update(Pose(0.5, 0.2, -0.8), 2.0, 0)
    assert velocity == (0.4, 0, 0) and not done
    assert diagnostics["phase"] == "walk"
    for now in (2.5, 2.75, 2.999):
        velocity, _, diagnostics = tracker.update(Pose(0.5, 0.2, 0), now, 0)
        assert velocity == (0, 0, 0)
        assert diagnostics["phase"] == "walk_pause"
    # A different location after the full pause reverses the required turn.
    velocity, _, diagnostics = tracker.update(Pose(0.5, -0.2, 0), 3.0, 0)
    assert velocity[:2] == (0, 0) and velocity[2] > 0
    assert diagnostics["approach_heading_rad"] == pytest.approx(math.atan2(0.2, 0.5))
    assert diagnostics["phase"] == "approach"


@pytest.mark.parametrize("bad_time", [0.7, 0.85, 1.0])
def test_tracker_rejects_overshoot_in_final_hold_window(bad_time):
    tracker = PoseTracker(Pose(1, 0, 0), UniLMConfig())
    for i in range(21):
        now = i * 0.05
        yaw = math.radians(20 if now >= bad_time - 1e-9 else 4)
        velocity, _, diagnostics = tracker.update(Pose(0, 0, yaw), now, 0)
        assert velocity[0] == 0
        if now < 1.0:
            assert velocity == (0, 0, 0)
        else:
            assert velocity[2] < 0 and diagnostics["phase"] == "approach"
    # Correcting the overshoot must be followed by a new, full hold.
    for i in range(20):
        velocity, _, _ = tracker.update(Pose(0, 0, 0), 1.1 + i * 0.05, 0)
        assert velocity == (0, 0, 0)
    velocity, _, _ = tracker.update(Pose(0, 0, 0), 2.1, 0)
    assert velocity == (0.4, 0, 0)


@pytest.mark.parametrize("recovery_time, can_walk", [(0.7, True), (0.75, False)])
def test_tracker_requires_entire_last_point_three_seconds_not_only_last_sample(recovery_time, can_walk):
    tracker = PoseTracker(Pose(1, 0, 0), UniLMConfig())
    for i in range(21):
        now = i * 0.05
        yaw = math.radians(10 if 0.5 <= now < recovery_time else 0)
        velocity, _, _ = tracker.update(Pose(0, 0, yaw), now, 0)
        if now < 1.0:
            assert velocity == (0, 0, 0)
    assert velocity == ((0.4, 0, 0) if can_walk else (0, 0, 0))


def test_tracker_exactly_seven_degrees_does_not_pass_approach_check():
    tracker = PoseTracker(Pose(1, 0, 0), UniLMConfig())
    velocity, _, diagnostics = tracker.update(Pose(0, 0, math.radians(7)), 0, 0)
    assert velocity[0] == 0 and velocity[2] < 0
    assert diagnostics["phase"] == "approach"


@pytest.mark.parametrize("arrival_time", [0.5, 1.5, 2.25])
def test_tracker_arrival_interrupts_pre_walk_hold_walk_and_pause(arrival_time):
    tracker = PoseTracker(Pose(0.5, 0, 1), UniLMConfig())
    for i in range(round(arrival_time / 0.05)):
        tracker.update(Pose(0, 0, 0), i * 0.05, 0)
    velocity, done, diagnostics = tracker.update(Pose(0.46, 0, 0), arrival_time, 0)
    assert velocity == (0, 0, 0) and not done
    assert diagnostics["position_acquired"]
    assert diagnostics["phase"] == "position_settling"


@pytest.mark.parametrize("arrival_time", [0.05, 0.2, 0.4, 0.65, 1.0])
def test_tracker_stops_on_arrival_then_holds_zero_for_one_second(arrival_time):
    tracker = PoseTracker(Pose(0.5, 0, math.pi / 2), UniLMConfig())
    for i in range(20):
        tracker.update(Pose(0, 0, 0), i * 0.05, 0)
    arrival_time += 1.0
    for now in (1.0, arrival_time - 0.001):
        velocity, _, _ = tracker.update(Pose(0, 0, 0), now, 0)
        assert velocity == (0.4, 0, 0)
    # Even a pending yaw correction must stop in the arrival cycle.
    velocity, done, diagnostics = tracker.update(Pose(0.46, 0, 0.3), arrival_time, 0)
    assert velocity == (0, 0, 0) and not done
    assert diagnostics["position_acquired"]
    assert diagnostics["phase"] == "position_settling"
    # Re-entering or drifting out cannot restart walking or the hold timer.
    for elapsed, x in ((0.05, 0.2), (0.5, 0.46), (0.999, -0.1)):
        velocity, done, diagnostics = tracker.update(Pose(x, 0, 0.3), arrival_time + elapsed, 0)
        assert velocity == (0, 0, 0) and not done
        assert diagnostics["position_acquired"]
        assert diagnostics["phase"] == "position_settling"
    velocity, done, diagnostics = tracker.update(Pose(-0.1, 0, 0.3), arrival_time + 1.0, 0)
    assert velocity[:2] == (0, 0) and velocity[2] > 0 and not done
    assert diagnostics["phase"] == "final_yaw"


def test_runtime_profile_loads_camera_and_controller_bounds():
    config = load_component_config(UniLMConfig, "unilm_nav")
    assert (config.camera, config.camera_height_m) == ("chest", 1.0)
    assert UniLMConfig().position_tolerance_m == config.position_tolerance_m == 0.05
    assert UniLMConfig().position_settle_s == config.position_settle_s == 1.0
    assert (config.min_angular_speed_rad_s, config.max_angular_speed_rad_s) == (0.1, 0.3)
    assert config.yaw_tolerance_deg == 7.0
    assert (config.approach_hold_s, config.approach_stable_s) == (1.0, 0.3)
    assert (config.walk_duration_s, config.walk_pause_s) == (1.0, 0.5)


@pytest.mark.parametrize(
    "settings",
    [
        {"position_tolerance_m": 0},
        {"position_tolerance_m": -0.1},
        {"position_settle_s": 0},
        {"position_settle_s": -0.1},
        {"position_settle_s": float("nan")},
        {"position_settle_s": float("inf")},
        {"min_angular_speed_rad_s": 0.31},
        {"min_angular_speed_rad_s": float("nan")},
        {"approach_hold_s": 0},
        {"approach_stable_s": 1.1},
        {"walk_duration_s": -1},
        {"walk_pause_s": float("nan")},
    ],
)
def test_invalid_controller_bounds_are_rejected(settings):
    with pytest.raises(ValueError):
        UniLMConfig(**settings)


@pytest.mark.parametrize(
    "goal", [Pose(0.8, 0.4, math.radians(-179)), Pose(-0.5, -0.4, math.radians(179)), Pose(0, 0, math.pi / 2)]
)
def test_tracker_acquires_position_and_finishes_yaw_with_sonic_lag_and_existing_envelope(goal):
    c = UniLMConfig()
    tracker = PoseTracker(goal, c)
    p = Pose(0, 0, math.radians(179))
    heading = p.yaw
    gateway = NavigationControlState()
    gateway.handle_key("b", now=0)
    position_acquired_pose = None
    for i in range(2400):
        now = i * 0.05
        velocity, done, diagnostics = tracker.update(p, now, wrap(heading - p.yaw))
        if diagnostics["position_acquired"] and position_acquired_pose is None:
            assert diagnostics["distance_m"] < c.position_tolerance_m
            position_acquired_pose = p
        if position_acquired_pose is not None:
            assert velocity[:2] == (0, 0)
        gateway.accept_base_pose_velocity(
            {
                "generation": 1,
                "skill_id": 0,
                "segment_id": 0,
                "velocity": velocity,
                "motion_profile": "yoloe_servo",
            },
            now=now,
        )
        heading = wrap(heading + velocity[2] * 0.05)
        yaw = wrap(p.yaw + wrap(heading - p.yaw) * 0.25)
        p = Pose(p.x + velocity[0] * math.cos(heading) * 0.05, p.y + velocity[0] * math.sin(heading) * 0.05, yaw)
        if done:
            break
    assert done
    assert position_acquired_pose is not None
    assert (p.x, p.y) == (position_acquired_pose.x, position_acquired_pose.y)
    assert math.hypot(p.x - goal.x, p.y - goal.y) < c.position_tolerance_m
    assert abs(wrap(p.yaw - goal.yaw)) <= math.radians(c.yaw_tolerance_deg)


def runtime(tmp_path, infer=None):
    calls = []
    c = UniLMConfig(output_root=str(tmp_path))
    r = UniLMRuntime(
        c,
        load_base_pose_config(),
        send=lambda n, p: calls.append((n, dict(p))),
        infer=infer or (lambda *a, **k: (Pose(0, 0, 0), {})),
        current_pose=lambda: (Pose(0, 0, 0), time.monotonic()),
        geometry={},
        report=lambda *a, **k: None,
    )
    return r, calls


def test_standalone_uses_five_fresh_frames_and_duplicate_start_is_ignored(tmp_path):
    frames = []

    def infer(obs, *a, **kw):
        frames.extend(obs)
        return Pose(0, 0, 0), {}

    r, calls = runtime(tmp_path, infer)
    r.observe(observation(time.time() - 10))
    params = {"generation": 1, "skill_id": 0, "segment_id": 0}
    r.command("start_base_pose", params, "collect bottle")
    original = r.cancel
    r.command("start_base_pose", params, "collect bottle")
    assert r.cancel is original
    r.observe(observation(r.collect_after - 1))
    for i in range(5):
        r.observe(observation(r.collect_after + 0.1 + i * 0.5))
    for _ in range(100):
        if not r.results.empty():
            break
        time.sleep(0.005)
    assert len(frames) == 5
    saved_config = json.loads((r.directory / "config.json").read_text())
    assert saved_config["unilm"]["camera_height_m"] == 1.0
    assert saved_config["camera_stream"] == "chest_view"
    assert saved_config["depth_stream"] == "camera/chest_view_depth"
    assert saved_config["camera_pitch_deg"] == -3.0
    assert all(p["camera_stream"] == "chest_view" for n, p in calls if n == "base_pose_velocity")
    r.command("cancel_navigation", {"generation": 2}, "")
    r.tick()
    assert r.active is None
    assert not any(n == "start_vla_task" for n, _ in calls)
    assert not any(any(v for v in p.get("velocity", [])) for _, p in calls)


def test_explicit_empty_unilm_task_does_not_fall_back(tmp_path):
    r, calls = runtime(tmp_path, lambda *a, **k: pytest.fail("empty task must not infer"))
    r.command("start_base_pose", {"generation": 1, "task": ""}, "collect the pill bottle")
    assert r.task == ""
    assert r.active is None
    status = next(p for n, p in calls if n == "base_pose_status")
    assert status["reason"] == "manipulation_task_missing"
    assert not any(any(p.get("velocity", [])) for _, p in calls)


def test_missing_navigation_history_fails_once_and_never_falls_back(tmp_path):
    r, calls = runtime(tmp_path)
    params = {
        "generation": 1,
        "skill_id": 2,
        "segment_id": 4,
        "observation_source": "navigation",
        "source_nav_skill_id": 1,
        "source_nav_segment_id": 3,
    }
    r.command("start_base_pose", params, "pick bottle")
    r.command("start_base_pose", params, "pick bottle")
    terminals = [p for n, p in calls if n == "base_pose_status"]
    assert len(terminals) == 1
    assert terminals[0]["reason"] == "navigation_history_missing"
    assert r.active is None


def test_stale_result_and_stale_orientation_cannot_move_robot(tmp_path):
    r, calls = runtime(tmp_path)
    r.command("start_base_pose", {"generation": 1}, "pick bottle")
    old = r.cancel
    r.command("cancel_navigation", {"generation": 2}, "")
    r.command("start_base_pose", {"generation": 3}, "pick bottle")
    r.results.put(((1, 0, 0), old, Pose(1, 0, 0), None))
    r.tick()
    assert r.tracker is None
    r.results.put(((3, 0, 0), r.cancel, Pose(1, 0, 0), None))
    r.tick()
    assert r.active is None
    assert calls[-1][1]["reason"] == "sonic_orientation_stale_or_unavailable"
    assert not any(any(p.get("velocity", [])) for _, p in calls)


def test_out_of_near_field_goal_is_rejected_without_clipping(tmp_path):
    r, calls = runtime(tmp_path)
    r.command("start_base_pose", {"generation": 1}, "pick bottle")
    r.results.put(((1, 0, 0), r.cancel, Pose(100, 0, 0), None))
    r.tick()
    assert calls[-1][1]["reason"] == "predicted_pose_outside_near_field"


def test_navigation_runtime_uses_only_frozen_source_segment(tmp_path):
    seen = []
    r, calls = runtime(tmp_path, lambda frames, *a, **kw: (seen.extend(frames) or Pose(0, 0, 0), {}))
    nav = {"generation": 1, "skill_id": 2, "segment_id": 5}
    r.command("navigation_goal", nav, "pick bottle")
    for i in range(7):
        r.observe(observation(100 + i))
    r.command("navigation_status", {**nav, "state": "reached"}, "")
    r.observe(observation(200))
    r.command(
        "start_base_pose",
        {
            "generation": 1,
            "skill_id": 3,
            "segment_id": 7,
            "observation_source": "navigation",
            "source_nav_skill_id": 2,
            "source_nav_segment_id": 5,
        },
        "pick bottle",
    )
    for _ in range(100):
        if not r.results.empty():
            break
        time.sleep(0.005)
    assert [frame.frame.timestamp for frame in seen] == [102, 103, 104, 105, 106]
    r.terminal("stopped", "test_finished")


def test_incomplete_navigation_history_does_not_call_model(tmp_path):
    r, calls = runtime(tmp_path, lambda *a, **k: pytest.fail("history must contain five frames"))
    nav = {"generation": 1, "skill_id": 2, "segment_id": 5}
    r.command("navigation_goal", nav, "")
    r.observe(observation())
    r.command("navigation_status", {**nav, "state": "reached"}, "")
    r.command(
        "start_base_pose",
        {
            "generation": 1,
            "skill_id": 3,
            "segment_id": 7,
            "observation_source": "navigation",
            "source_nav_skill_id": 2,
            "source_nav_segment_id": 5,
        },
        "pick bottle",
    )
    assert calls[-1][1]["reason"] == "navigation_history_incomplete"


@pytest.mark.parametrize("last_segment_frames", [0, 1, 5])
def test_alignment_uses_all_completed_navigation_segments(tmp_path, last_segment_frames):
    seen = []
    r, calls = runtime(tmp_path, lambda frames, *a, **kw: (seen.extend(frames) or Pose(0, 0, 0), {}))
    first = {"generation": 1, "skill_id": 1, "segment_id": 6}
    last = {"generation": 1, "skill_id": 2, "segment_id": 15}
    r.command("navigation_goal", first, "")
    for i in range(7):
        r.observe(observation(100 + i, depth=800, pose=Pose(1, 2, 0.3)))
    r.command("navigation_status", {**first, "state": "reached"}, "")
    # Independent sensor frames during LA's panorama / VA waits must not enter history.
    r.observe(observation(150))
    r.command("heading_goal", {"generation": 1, "skill_id": 2, "segment_id": 14}, "")
    r.observe(observation(160))
    r.command("navigation_goal", last, "")
    for i in range(last_segment_frames):
        r.observe(observation(200 + i, depth=1200, pose=Pose(3, 4, 0.5)))
    r.command("navigation_status", {**last, "state": "reached"}, "")
    r.observe(observation(250))
    r.command(
        "start_base_pose",
        {
            "generation": 1,
            "skill_id": 3,
            "segment_id": 17,
            "observation_source": "navigation",
            "source_nav_skill_id": 2,
            "source_nav_segment_id": 15,
        },
        "pick bowl",
    )
    for _ in range(100):
        if not r.results.empty():
            break
        time.sleep(0.005)
    try:
        expected = [102, 103, 104, 105, 106] + list(range(200, 200 + last_segment_frames))
        assert [obs.frame.timestamp for obs in seen] == expected
        assert seen[0].pose == Pose(1, 2, 0.3)
        with np.load(r.directory / "observation_0.npz") as saved:
            assert saved["timestamp"] == 102
            assert np.all(saved["depth"] == 800)
            assert saved["pose"] == pytest.approx([1, 2, 0.3])
        assert len(list(r.directory.glob("observation_*.npz"))) == len(expected)
    finally:
        r.terminal("stopped", "test_finished")


@pytest.mark.parametrize("reset", ["start_navigation", "cancel_navigation", "new_generation"])
def test_navigation_history_cannot_leak_across_tasks(tmp_path, reset):
    r, calls = runtime(tmp_path, lambda *a, **kw: pytest.fail("old task history must not infer"))
    nav = {"generation": 1, "skill_id": 1, "segment_id": 6}
    r.command("navigation_goal", nav, "")
    for i in range(5):
        r.observe(observation(100 + i))
    r.command("navigation_status", {**nav, "state": "reached"}, "")
    generation = 2 if reset == "new_generation" else 1
    if reset != "new_generation":
        r.command(reset, {"generation": generation}, "")
    r.command(
        "start_base_pose",
        {
            "generation": generation,
            "skill_id": 3,
            "segment_id": 17,
            "observation_source": "navigation",
            "source_nav_skill_id": 1,
            "source_nav_segment_id": 6,
        },
        "pick bowl",
    )
    assert calls[-1][1]["reason"] == "navigation_history_missing"


def test_timeout_and_lost_odometry_stop_without_retry(tmp_path):
    r, calls = runtime(tmp_path)
    r.command("start_base_pose", {"generation": 1}, "pick bottle")
    r.deadline = time.monotonic() - 1
    r.tick()
    assert calls[-1][1]["reason"] == "unilm_timeout"
    r.command("start_base_pose", {"generation": 2}, "pick bottle")
    r.tracker = PoseTracker(Pose(1, 0, 0), r.config)

    def missing_pose():
        raise UniLMError("odometry_unavailable")

    r.current_pose = missing_pose
    r.tick()
    assert calls[-1][1]["reason"] == "odometry_unavailable"
    assert r.active is None
    assert not any(any(p.get("velocity", [])) for _, p in calls)


def test_runtime_streams_forward_then_stops_at_fastlio_arrival_before_final_turn(tmp_path, monkeypatch):
    import gear_sonic.utils.inference.base_pose.unilm_runtime as module

    r, calls = runtime(tmp_path)
    now = 100.0
    pose = Pose(0, 0, 0)
    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    r.current_pose = lambda: (pose, now)
    r.command("start_base_pose", {"generation": 1}, "pick bottle")
    r.results.put(((1, 0, 0), r.cancel, Pose(0.5, 0, math.pi / 2), None))
    for elapsed, x, expected_phase in (
        (0.0, 0.0, "approach_settling"),
        (0.7, 0.0, "approach_settling"),
        (1.0, 0.0, "walk"),
        (1.25, 0.44, "walk"),
        (1.5, 0.46, "position_settling"),
        (2.0, 0.4, "position_settling"),
        (2.499, 0.46, "position_settling"),
        (2.5, 0.4, "final_yaw"),
    ):
        now = 100.0 + elapsed
        pose = Pose(x, 0, 0)
        r.orientation = SimpleNamespace(heading_lag_rad=0.0, state_age_s=0.0, emitted_at_monotonic_s=now)
        r.tick()
        assert r.active is not None
        assert calls[-1][0] == "base_pose_velocity"
        velocity = calls[-1][1]["velocity"]
        if expected_phase == "walk":
            assert velocity == [0.4, 0, 0]
        elif expected_phase in {"position_settling", "approach_settling"}:
            assert velocity == [0, 0, 0]
        else:
            assert velocity[:2] == [0, 0] and velocity[2] > 0
        assert r.last_tracking["phase"] == expected_phase
    r.terminal("stopped", "test_finished")


def test_standalone_stops_and_logs_actual_pose_after_settling(tmp_path, monkeypatch):
    import gear_sonic.utils.inference.base_pose.unilm_runtime as module

    r, calls = runtime(tmp_path)
    now = time.monotonic()
    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    r.command("start_base_pose", {"generation": 1}, "pick bottle")
    r.results.put(((1, 0, 0), r.cancel, Pose(0, 0, 0), None))
    for _ in range(44):
        r.orientation = SimpleNamespace(heading_lag_rad=0.0, state_age_s=0.0, emitted_at_monotonic_s=now)
        r.tick()
        now += 0.05
    assert r.active is None
    terminal = calls[-1][1]
    assert terminal["state"] == "reached"
    assert terminal["last_tracking"]["pose"] == {"x": 0, "y": 0, "yaw": 0}
    assert all(n in {"base_pose_velocity", "base_pose_status"} for n, _ in calls)
    assert json.loads((Path(terminal["artifacts"]) / "result.json").read_text())["reason"] == "aligned"


@pytest.mark.parametrize("drift_m", [0.16, 3.1])
def test_runtime_reports_aligned_despite_position_drift_during_final_yaw(tmp_path, monkeypatch, drift_m):
    import gear_sonic.utils.inference.base_pose.unilm_runtime as module

    r, calls = runtime(tmp_path)
    now = time.monotonic()
    pose = Pose(-0.04, 0, 0)
    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    r.current_pose = lambda: (pose, now)
    r.command("start_base_pose", {"generation": 1}, "pick bottle")
    r.results.put(((1, 0, 0), r.cancel, Pose(0, 0, math.pi / 2), None))
    r.orientation = SimpleNamespace(heading_lag_rad=0.0, state_age_s=0.0, emitted_at_monotonic_s=now)
    r.tick()
    assert calls[-1][1]["velocity"] == [0, 0, 0]
    assert r.last_tracking["phase"] == "position_settling"
    # Position is diagnostic after acquisition, including the near-field bound.
    pose = Pose(-drift_m, 0, 0.2)
    now += 1.0
    r.orientation = SimpleNamespace(heading_lag_rad=0.0, state_age_s=0.0, emitted_at_monotonic_s=now)
    r.tick()
    assert r.active is not None
    assert calls[-1][1]["velocity"][:2] == [0, 0]
    assert calls[-1][1]["velocity"][2] > 0
    pose = Pose(-drift_m, 0, math.pi / 2)
    for _ in range(22):
        now += 0.05
        r.orientation = SimpleNamespace(heading_lag_rad=0.0, state_age_s=0.0, emitted_at_monotonic_s=now)
        r.tick()
    terminal = calls[-1][1]
    assert r.active is None
    assert (terminal["state"], terminal["reason"]) == ("reached", "aligned")
    assert terminal["last_tracking"]["distance_m"] == pytest.approx(drift_m)
    assert terminal["last_tracking"]["phase"] == "complete"
    assert calls[-2][1]["velocity"] == [0, 0, 0]


@pytest.mark.parametrize("during_tracking", [False, True])
def test_runtime_checks_near_field_distance_before_position_acquisition(tmp_path, during_tracking):
    r, calls = runtime(tmp_path)
    pose = Pose(0, 0, 0) if during_tracking else Pose(-3, 0, 0)
    r.current_pose = lambda: (pose, time.monotonic())
    r.command("start_base_pose", {"generation": 1}, "pick bottle")
    r.results.put(((1, 0, 0), r.cancel, Pose(1, 0, 0), None))
    r.orientation = SimpleNamespace(
        heading_lag_rad=0.0, state_age_s=0.0, emitted_at_monotonic_s=time.monotonic()
    )
    r.tick()
    if during_tracking:
        assert r.active is not None and r.last_tracking["phase"] == "approach_settling"
        pose = Pose(-3, 0, 0)
        r.tick()
    assert r.active is None
    prefix = "tracking" if during_tracking else "predicted"
    assert calls[-1][1]["reason"] == f"{prefix}_pose_outside_near_field"
    assert calls[-2][1]["velocity"] == [0, 0, 0]


@pytest.mark.parametrize(
    "reason",
    ["operator_cancelled", "unilm_timeout", "odometry_unavailable", "sonic_orientation_stale_or_unavailable"],
)
@pytest.mark.parametrize("phase", ["position_settling", "final_yaw"])
def test_runtime_still_stops_after_position_acquisition(tmp_path, monkeypatch, reason, phase):
    import gear_sonic.utils.inference.base_pose.unilm_runtime as module

    r, calls = runtime(tmp_path)
    now = time.monotonic()
    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    r.command("start_base_pose", {"generation": 1}, "pick bottle")
    r.results.put(((1, 0, 0), r.cancel, Pose(0, 0, math.pi / 2), None))
    r.orientation = SimpleNamespace(
        heading_lag_rad=0.0, state_age_s=0.0, emitted_at_monotonic_s=time.monotonic()
    )
    r.tick()
    assert r.tracker.position_acquired and calls[-1][1]["velocity"] == [0, 0, 0]
    if phase == "final_yaw":
        now += 1.0
        r.orientation.emitted_at_monotonic_s = now
        r.tick()
        assert calls[-1][1]["velocity"][2] > 0
    assert r.last_tracking["phase"] == phase
    before_stop = len(calls)
    if reason == "operator_cancelled":
        r.command("cancel_navigation", {"generation": 1}, "")
    elif reason == "unilm_timeout":
        r.deadline = time.monotonic() - 1
    elif reason == "odometry_unavailable":

        def missing_pose():
            raise UniLMError("odometry_unavailable")

        r.current_pose = missing_pose
    else:
        r.orientation.state_age_s = r.config.odometry_timeout_s + 1
    r.tick()
    assert r.active is None and calls[-1][1]["reason"] == reason
    assert calls[-2][1]["velocity"] == [0, 0, 0]
    assert not any(any(p.get("velocity", [])) for _, p in calls[before_stop:])


@pytest.mark.parametrize("elapsed, phase", [(0.5, "approach_settling"), (1.0, "walk"), (2.0, "walk_pause")])
def test_operator_cancel_interrupts_approach_phases(tmp_path, monkeypatch, elapsed, phase):
    import gear_sonic.utils.inference.base_pose.unilm_runtime as module

    r, calls = runtime(tmp_path)
    now = 100.0
    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    r.command("start_base_pose", {"generation": 1}, "pick bottle")
    r.results.put(((1, 0, 0), r.cancel, Pose(0.2, 0, 0), None))
    r.orientation = SimpleNamespace(
        heading_lag_rad=0.0, state_age_s=0.0, emitted_at_monotonic_s=time.monotonic()
    )
    r.tick()
    for dt in sorted({0.5, min(elapsed, 1.0), elapsed}):
        now = 100.0 + dt
        r.orientation.emitted_at_monotonic_s = now
        r.tick()
    assert r.last_tracking["phase"] == phase
    before_cancel = len(calls)
    r.command("cancel_navigation", {"generation": 2}, "")
    r.tick()
    assert r.active is None
    assert calls[-1][1]["state"] == "stopped"
    assert calls[before_cancel][1]["velocity"] == [0, 0, 0]
    assert not any(any(p.get("velocity", [])) for _, p in calls[before_cancel:])


def test_gateway_unilm_start_and_status_preserve_ownership():
    r = object.__new__(ControlGatewayRuntime)
    r.profile = load_runtime_profile()
    r.navigation = NavigationControlState()
    r.navigation.handle_key("n", now=0)
    dispatched = []
    r._send_navigation_stop = lambda **k: None
    r.dispatch_navigation_event = lambda n, p: dispatched.append((n, p))
    r.show_command = lambda *a, **k: None
    parameters = {
        "generation": 1,
        "skill_id": 2,
        "segment_id": 4,
        "backend": "unilm_nav",
        "task": "pick bottle",
        "observation_source": "navigation",
        "source_nav_skill_id": 1,
        "source_nav_segment_id": 2,
    }
    core = ControlGatewayCore(source="lavira_agent")
    r._handle_start_base_pose(core.accept_command("start_base_pose", parameters=parameters).command)
    assert r.navigation.mode == "base_pose_inference"
    assert dispatched[-1] == ("start_base_pose", parameters)
    assert r.navigation.handle_key("b", now=1).mode == "ignored"
    with pytest.raises(ValueError, match="stale or unexpected"):
        r._handle_start_base_pose(core.accept_command("start_base_pose", parameters=parameters).command)
    base_core = ControlGatewayCore(source="base_pose_agent")
    status = {"generation": 1, "skill_id": 2, "segment_id": 3, "state": "reached", "reason": "aligned"}
    with pytest.raises(ValueError, match="stale base-pose segment"):
        r._handle_base_pose_status(base_core.accept_command("base_pose_status", parameters=status).command)
    assert r.navigation.mode == "base_pose_inference"
    status["segment_id"] = 4
    r._handle_base_pose_status(base_core.accept_command("base_pose_status", parameters=status).command)
    assert r.navigation.mode == "lavira_pending"
    assert r.navigation.handle_key("b", now=2).mode == "ignored"


@pytest.mark.parametrize("prompt", ["", "pick the cup"])
def test_standalone_gateway_b_supplies_task_and_n_is_disabled(prompt):
    settings = {
        "launcher": {"run_mode": "unilm_only"},
        "unilm_nav": {"enabled": True},
        "vla": {"vla_prompt": prompt, "vla_trained_prompt": "collect the pill bottle"},
    }
    r = object.__new__(ControlGatewayRuntime)
    r.profile = SimpleNamespace(components=settings, component=settings.__getitem__)
    r.navigation = NavigationControlState()
    r.navigation_pub = SimpleNamespace(send_string=lambda payload: None)
    r.show_command = lambda *a, **k: None
    r.show_event = lambda *a, **k: None
    dispatched = []
    r.dispatch_navigation_event = lambda n, p: dispatched.append((n, p))
    core = ControlGatewayCore(source="operator_console")
    r._handle_navigation_key(core.accept_command("navigation_key", parameters={"key": "n"}).command)
    assert r.navigation.mode == "listen_wasd"
    assert not dispatched
    r._handle_navigation_key(core.accept_command("navigation_key", parameters={"key": "b"}).command)
    assert dispatched == [
        (
            "start_base_pose",
            {
                "generation": 1,
                "skill_id": 0,
                "segment_id": 0,
                "backend": "unilm_nav",
                "task": prompt,
                "observation_source": "current",
            },
        )
    ]


@pytest.mark.parametrize("navigation_mode", ["vln", "object_nav"])
def test_agent_automatically_hands_off_once_without_old_alignment_checks(navigation_mode):
    from gear_sonic.tests.test_lavira_agent import build_agent, grounding, move, task_complete

    agent, _, client, intents, _ = build_agent(
        [move()], groundings=[grounding()] * 3, postchecks=[task_complete()],
        manipulation_prompt="Put the bottle into the blue basket.",
    )
    agent.alignment_backend = "unilm_nav"
    agent.navigation_mode = navigation_mode
    result = agent.run(7)
    assert result.state == "reached"
    names = [n for n, _ in intents]
    assert names.count("start_base_pose") == 1
    assert names.count("start_vla_task") == 1
    assert len(client.la_calls) == 1
    assert not client.alignment_grounding_calls
    assert len(client.postcheck_calls) == 1
    command = next(p for n, p in intents if n == "start_base_pose")
    nav = next(p for n, p in intents if n == "navigation_goal")
    assert command["source_nav_segment_id"] == nav["segment_id"]
    assert command["task"] == "Put the bottle into the blue basket."
    assert client.la_calls[0]["manipulation_prompt"] == command["task"]
    assert client.postcheck_calls[0]["mission"] == command["task"]
    assert "collect the pill bottle" not in json.dumps(intents)


@pytest.mark.parametrize("ready", [True, False])
def test_agent_unilm_completion_proceeds_to_vla_readiness_without_distance_check(ready):
    from gear_sonic.tests.test_lavira_agent import build_agent, grounding, move, task_complete

    agent, _, client, intents, _ = build_agent(
        [move()], groundings=[grounding()] * 3, postchecks=[task_complete()]
    )
    agent.alignment_backend = "unilm_nav"
    original_wait = agent.wait_status
    readiness_calls = []

    def wait(generation, skill, segment, timeout):
        result = original_wait(generation, skill, segment, timeout)
        if intents[-1][0] == "start_base_pose":
            result["last_tracking"] = {"distance_m": 0.16, "yaw_error_rad": 0.0, "phase": "complete"}
        return result

    def readiness(generation, skill):
        assert intents[-1][0] == "start_base_pose"
        readiness_calls.append((generation, skill))
        return ready, "test_not_ready"

    agent.wait_status = wait
    agent.readiness = readiness
    result = agent.run(7)
    assert len(readiness_calls) == 1
    assert not client.alignment_grounding_calls
    assert [name for name, _ in intents].count("start_vla_task") == int(ready)
    if ready:
        assert result.state == "reached"
    else:
        assert result.state == "failed"
        assert result.reason == "manipulate_gate:test_not_ready"


def test_agent_unilm_failure_terminates_without_vla_or_retry():
    from gear_sonic.tests.test_lavira_agent import build_agent, grounding, move

    agent, _, client, intents, _ = build_agent([move()], groundings=[grounding()] * 3)
    agent.alignment_backend = "unilm_nav"
    original = agent.wait_status

    def wait(g, s, seg, timeout, **kw):
        if intents[-1][0] == "start_base_pose":
            return {"generation": g, "skill_id": s, "segment_id": seg, "state": "failed", "reason": "bad_depth"}
        return original(g, s, seg, timeout, **kw)

    agent.wait_status = wait
    result = agent.run(7)
    assert result.state == "failed"
    assert "unilm_failed" in result.reason
    assert [n for n, _ in intents].count("start_base_pose") == 1
    assert not any(n == "start_vla_task" for n, _ in intents)


def test_agent_intermediate_move_does_not_trigger_unilm():
    from gear_sonic.tests.test_lavira_agent import build_agent, decision, grounding, move

    agent, _, _, intents, _ = build_agent(
        [move("trash can", global_target="basket"), decision(None, result="FAIL")], groundings=[grounding()] * 3
    )
    agent.alignment_backend = "unilm_nav"
    agent.navigation_mode = "vln"
    result = agent.run(7)
    assert result.state == "failed"
    assert any(n == "navigation_goal" for n, _ in intents)
    assert not any(n == "start_base_pose" for n, _ in intents)


def test_standalone_launcher_has_no_navigation_prerequisites(tmp_path, monkeypatch):
    import yaml

    from gear_sonic.scripts import launch_inference as launcher
    from gear_sonic.scripts.launch_unilm_nav import standalone_profile

    document = standalone_profile(launcher.default_launch_config_path(), "pick the cup")
    path = tmp_path / "standalone.yaml"
    path.write_text(yaml.safe_dump(document))
    c = launcher.load_inference_launch_config(path)
    assert c.run_mode == "unilm_only"
    assert "navdp" not in launcher._worker_pane_names(c)
    assert not launcher._depth_anything_required(c)
    prompts = load_runtime_profile(path).component("vla")
    assert prompts["vla_prompt"] == "pick the cup"
    assert prompts["vla_trained_prompt"] == load_runtime_profile().component("vla")["vla_trained_prompt"]
    monkeypatch.setattr(
        launcher.socket,
        "create_connection",
        lambda *a, **k: pytest.fail("standalone must not wait for navigation/model TCP services"),
    )
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    launcher._check_prerequisites(c)


def test_shared_environment_keeps_current_checkout_first():
    from gear_sonic.scripts import launch_inference as launcher

    root = Path(__file__).resolve().parents[2]
    command = launcher.build_base_pose_agent_command(launcher.load_inference_launch_config(), root)
    assert str(root.parent / "GR00T-WholeBodyControl/.venv_inference/bin/python") in command
    assert str(root.parent / "GR00T-WholeBodyControl/.env.local") in command
    assert f"export PYTHONPATH={root}" in command
