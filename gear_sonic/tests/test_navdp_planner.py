from __future__ import annotations

import json
import math
import signal
import struct
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.inference.navdp.control import (
    XNAVDP_G1_MPC_DEFAULTS,
    AsyncMpcSolver,
    InternNavMpcController,
    LatestMessageWorker,
    MpcSolveRequest,
    fastlio_heading_target_from_mpc,
    fresh_mpc_control,
    prepare_internnav_world_reference,
    should_abort_nav_for_zero_action,
    xnavdp_adaptive_speed,
    xnavdp_control_to_body_velocity,
)
from gear_sonic.utils.inference.navdp.gateway import (
    NavDPPlannerConfig,
    load_navdp_planner_config,
    _SharedSensors,
    _control_freshness_snapshot,
    _encode_navdp_frames,
    _navdp_request,
    _reset_navdp,
    point_plane_distances,
)
from gear_sonic.utils.inference.navdp.navigation import (
    HeadingGoalController,
    NavigationCommand,
    Pose2D,
    base_goal_to_world,
    build_navigation_message,
    closest_timestamped_pose,
    decode_navigation_message,
    local_goal_from_world,
    local_trajectory_to_world,
    update_slam_map,
)
from gear_sonic.utils.inference.navdp.runtime import (
    _NavDPRuntimeState,
    _NavDPSensorCycle,
    _schedule_navigation_inference,
)
from gear_sonic.utils.inference.navdp.service import (
    _fresh_sonic_yaw,
    _navigation_status_payload,
)
from gear_sonic.utils.inference.navdp.visualization import (
    _VIZ_CENTER,
    actor_ray_from_points,
    actor_ray_velocity_arrow,
    filter_livox_points,
    format_actor_ray_control_text,
    install_shutdown_signal_handlers,
    render_actor_ray_panel,
    render_slam_world_panel,
)
from gear_sonic.utils.planner_control import SonicPlannerState, depth_requires_stop
from gear_sonic.utils.teleop.sonic_orientation_telemetry import (
    OrientationTelemetrySample,
)


def test_navdp_goal_tolerance_matches_production_profile() -> None:
    config = load_navdp_planner_config()
    assert config.goal_tolerance_m == pytest.approx(2.0)
    assert config.rgb_stream == "camera/ego_view"
    assert config.depth_stream == "camera/ego_view_depth"


def test_navdp_runs_unthrottled_inference_with_ten_hz_mpc() -> None:
    config = load_navdp_planner_config()
    assert config.control_hz == pytest.approx(20.0)
    assert config.mpc_hz == pytest.approx(10.0)
    assert config.mpc_result_timeout_s == pytest.approx(0.3)
    assert not hasattr(config, "inference_hz")
    assert config.heading_preview_s == pytest.approx(0.6)
    assert config.heading_angular_speed_rad_s == pytest.approx(0.4)
    assert config.heading_fine_angular_speed_rad_s == pytest.approx(0.2)
    assert config.heading_slowdown_angle_rad == pytest.approx(math.radians(20.0))
    assert config.heading_goal_tolerance_rad == pytest.approx(math.radians(5.0))
    assert config.heading_orientation_timeout_s == pytest.approx(0.3)
    assert not hasattr(config, "radar_timeout_s")
    assert config.trajectory_timeout_s == pytest.approx(2.5)


def test_heading_goal_slows_and_stops_on_sonic_error_across_wraparound() -> None:
    controller = HeadingGoalController()
    controller.start(current_yaw=3.0, delta_rad=math.pi / 2.0, now=10.0)

    crossed_wrap = controller.update(current_yaw=-2.5, now=10.5)
    almost_done = controller.update(current_yaw=-1.8, now=11.0)
    completed = controller.update(current_yaw=-1.7, now=11.1)

    assert crossed_wrap.state == almost_done.state == "active"
    assert crossed_wrap.angular_velocity_rad_s == pytest.approx(0.4)
    assert crossed_wrap.reason == "heading_sonic_yaw_tracking"
    assert almost_done.angular_velocity_rad_s == pytest.approx(0.2)
    assert completed.state == "reached"
    assert completed.angular_velocity_rad_s == 0.0
    assert completed.reason == "heading_sonic_yaw_reached"
    assert controller.accumulated_yaw_rad == pytest.approx(1.583185307179586)


def test_heading_goal_can_force_left_at_the_pi_boundary() -> None:
    controller = HeadingGoalController()
    requested_left_turn = math.pi + math.radians(0.62)
    controller.start(
        current_yaw=-1.5,
        delta_rad=requested_left_turn,
        turn_direction="left",
        now=10.0,
    )

    initial = controller.update(current_yaw=-1.5, now=10.1)
    almost_done = controller.update(current_yaw=1.5, now=17.6)
    completed = controller.update(
        current_yaw=controller.target_rad,
        now=18.0,
    )

    assert initial.state == almost_done.state == "active"
    assert initial.angular_velocity_rad_s == pytest.approx(0.4)
    assert initial.remaining_rad == pytest.approx(requested_left_turn)
    assert almost_done.angular_velocity_rad_s == pytest.approx(0.2)
    assert completed.state == "reached"
    assert completed.angular_velocity_rad_s == 0.0


def test_forced_left_heading_uses_shortest_correction_after_overshoot() -> None:
    controller = HeadingGoalController()
    controller.start(
        current_yaw=0.0,
        delta_rad=math.pi,
        turn_direction="left",
        now=10.0,
    )

    turning_left = controller.update(current_yaw=3.0, now=17.5)
    correcting_right = controller.update(current_yaw=-3.0, now=18.3)

    assert turning_left.angular_velocity_rad_s == pytest.approx(0.2)
    assert correcting_right.state == "active"
    assert correcting_right.angular_velocity_rad_s == pytest.approx(-0.2)
    assert correcting_right.remaining_rad == pytest.approx(-0.14159265358979312)


def test_heading_goal_has_no_global_timeout_but_adjustments_can_be_limited() -> None:
    controller = HeadingGoalController()
    controller.start(current_yaw=-1.2, delta_rad=math.pi / 2.0, now=20.0)

    still_turning = controller.update(current_yaw=-1.2, now=120.0)

    assert still_turning.state == "active"
    assert still_turning.angular_velocity_rad_s == pytest.approx(0.4)

    controller.start(
        current_yaw=-1.2,
        delta_rad=math.pi / 2.0,
        now=200.0,
        max_angular_speed_rad_s=0.2,
        max_duration_s=10.0,
    )
    adjusting = controller.update(current_yaw=-1.2, now=209.9)
    time_limited = controller.update(current_yaw=-1.2, now=210.0)

    assert adjusting.state == "active"
    assert adjusting.angular_velocity_rad_s == pytest.approx(0.2)
    assert time_limited.state == "reached"
    assert time_limited.angular_velocity_rad_s == 0.0
    assert time_limited.reason == "heading_adjustment_time_limit"


def test_heading_goal_only_uses_fresh_sonic_yaw() -> None:
    sample = OrientationTelemetrySample(
        emitted_at_monotonic_s=10.0,
        actual_yaw_rad=1.25,
        actual_heading_rad=0.4,
        heading_setpoint_rad=0.5,
        heading_lag_rad=0.1,
        state_age_s=0.05,
    )

    assert _fresh_sonic_yaw(sample, now_s=10.1, timeout_s=0.3) == pytest.approx(
        1.25
    )
    assert _fresh_sonic_yaw(sample, now_s=10.26, timeout_s=0.3) is None
    assert _fresh_sonic_yaw(None, now_s=10.0, timeout_s=0.3) is None


def test_xnavdp_g1_mpc_defaults_keep_three_second_horizon() -> None:
    defaults = XNAVDP_G1_MPC_DEFAULTS

    assert defaults == {
        "horizon_steps": 30,
        "desired_velocity": 0.4,
        "max_linear_velocity": 0.4,
        "max_angular_velocity": 0.8,
        "reference_gap": 3,
        "dt": 0.1,
        "reference_trajectory_length_m": 2.0,
        "minimum_desired_velocity": 0.0,
        "interpolation_ratio": 50,
        "lookahead_points": 10,
        "linear_control_weight": 5.0,
        "angular_control_weight": 0.02,
        "curvature_speed_gain": 0.15,
    }


def test_xnavdp_speed_mapping_is_unicycle_without_lateral_velocity() -> None:
    assert xnavdp_control_to_body_velocity(0.3, 0.5) == pytest.approx(
        (0.3, 0.0, 0.5)
    )
    assert xnavdp_control_to_body_velocity(-0.2, -0.4) == pytest.approx(
        (-0.2, 0.0, -0.4)
    )


def test_fastlio_heading_target_integrates_the_original_mpc_yaw_rate() -> None:
    heading = fastlio_heading_target_from_mpc(
        fastlio_yaw=0.4,
        mpc_angular_velocity=0.5,
        heading_preview_s=0.6,
    )

    assert heading == pytest.approx(0.70)


def test_xnavdp_request_keeps_lateral_trajectory_axis_unchanged(monkeypatch) -> None:
    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {"trajectory": [[[0.0, 0.0], [0.2, 0.1], [0.4, -0.2]]]}

    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(post=lambda *args, **kwargs: Response()),
    )

    trajectory = _navdp_request(
        "http://127.0.0.1:19999",
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8), dtype=np.float32),
        (1.0, 0.0),
    )

    np.testing.assert_allclose(trajectory, [[0.0, 0.0], [0.2, 0.1], [0.4, -0.2]])


def test_xnavdp_adaptive_speed_matches_length_and_curvature_limits() -> None:
    kwargs = {"max_angular_velocity": 0.8, "curvature_speed_gain": 0.15}
    assert xnavdp_adaptive_speed(2.0, 0.0, **kwargs) == pytest.approx(0.4)
    assert xnavdp_adaptive_speed(1.0, 0.0, **kwargs) == pytest.approx(0.2)
    assert xnavdp_adaptive_speed(0.01, 0.0, **kwargs) == pytest.approx(0.002)
    assert xnavdp_adaptive_speed(2.0, 10.0, **kwargs) == pytest.approx(0.012)


def test_navdp_sensor_state_starts_empty() -> None:
    sensors = _SharedSensors()
    assert sensors.slam_map_xy.shape == (0, 2)
    assert sensors.robot_history.shape == (0, 2)


def test_control_freshness_snapshot_reads_latest_ros_callback_values() -> None:
    sensors = _SharedSensors()
    sensors.pose = Pose2D(1.0, 2.0, 0.3)
    sensors.pose_time = 12.0
    sensors.points = np.array([[0.5, 0.1, 0.2]], dtype=np.float32)

    pose, pose_time, points = _control_freshness_snapshot(sensors)

    assert pose == Pose2D(1.0, 2.0, 0.3)
    assert pose_time == pytest.approx(12.0)
    np.testing.assert_allclose(points, [[0.5, 0.1, 0.2]])


def test_navdp_reset_uses_configured_permissive_pointgoal_stop_threshold(monkeypatch) -> None:
    captured = {}

    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

    def post(url, *, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))

    _reset_navdp(
        "http://127.0.0.1:19999",
        {"fx": 500.0, "fy": 501.0, "cx": 320.0, "cy": 240.0},
        stop_threshold=-4.0,
    )

    assert captured["json"]["stop_threshold"] == [-4.0]


def test_navdp_rgb_uses_official_jpeg_and_depth_uses_png() -> None:
    rgb_bytes, depth_bytes = _encode_navdp_frames(
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8), dtype=np.float32),
    )

    assert rgb_bytes.startswith(b"\xff\xd8")
    assert depth_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_xnavdp_request_sends_fastlio_pose_for_real_trajectory_guidance(monkeypatch) -> None:
    captured = {}

    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {"trajectory": [[[0.0, 0.0], [0.2, 0.0]]]}

    def post(url, *, files, data, timeout):
        captured.update(url=url, files=files, data=data, timeout=timeout)
        return Response()

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))

    trajectory = _navdp_request(
        "http://127.0.0.1:19999",
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8), dtype=np.float32),
        (2.0, -0.5),
        pose=Pose2D(1.25, -2.5, np.pi / 2.0),
    )

    state = json.loads(captured["data"]["state_data"])
    np.testing.assert_allclose(state["robot_pos"], [[1.25, -2.5, 0.0]])
    np.testing.assert_allclose(
        state["robot_quat"], [[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]], atol=1e-6
    )
    np.testing.assert_allclose(trajectory, [[0.0, 0.0], [0.2, 0.0]])


def test_xnavdp_request_timeout_defaults_to_ten_seconds(monkeypatch) -> None:
    captured = {}

    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {"trajectory": [[[0.0, 0.0], [0.2, 0.0]]]}

    def post(url, *, files, data, timeout):
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))

    _navdp_request(
        "http://127.0.0.1:19999",
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8), dtype=np.float32),
        (1.0, 0.0),
    )

    assert captured["timeout"] == pytest.approx(10.0)


def test_actor_ray_draws_raw_navdp_positions_as_yellow_points_not_segments() -> None:
    rays = np.full(180, 3.0, dtype=np.float32)
    trajectory = np.array([[0.0, 0.0], [0.6, 0.0], [1.2, 0.0]], dtype=np.float32)

    panel = render_actor_ray_panel(rays, trajectory=trajectory)

    scale = 205.0 / 3.0
    point_rows = [round(260 - x * scale) for x in (0.0, 0.6, 1.2)]
    assert all(np.array_equal(panel[row, 250], [0, 255, 255]) for row in point_rows)
    between_row = round(260 - 0.3 * scale)
    assert not np.array_equal(panel[between_row, 250], [0, 255, 255])


def test_actor_ray_draws_only_the_first_24_navdp_positions() -> None:
    rays = np.full(180, 3.0, dtype=np.float32)
    trajectory = np.zeros((25, 2), dtype=np.float32)
    trajectory[24] = (1.2, 0.0)

    panel = render_actor_ray_panel(rays, trajectory=trajectory)

    excluded_row = round(260 - 1.2 * (205.0 / 3.0))
    assert not np.array_equal(panel[excluded_row, 250], [0, 255, 255])


def test_sonic_planner_packet_preserves_positive_left_lateral_direction() -> None:
    packet = SonicPlannerState().message((0.0, 0.15, 0.0), dt=0.05)
    values = struct.unpack("<i3f3f2f", packet[len(b"planner") + 1280 :])
    assert values[1:4] == pytest.approx((0.0, 1.0, 0.0))
    assert values[7] == pytest.approx(0.15)


def test_local_navdp_trajectory_rotates_and_translates_into_world_frame() -> None:
    local = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    world = local_trajectory_to_world(local, Pose2D(10.0, 20.0, np.pi / 2))

    np.testing.assert_allclose(world, [[10.0, 20.0], [10.0, 21.0], [9.0, 21.0]], atol=1e-6)


def test_slam_map_voxelizes_and_keeps_only_nearby_absolute_points() -> None:
    existing = np.array([[0.01, 0.01], [20.0, 20.0]], dtype=np.float32)
    incoming = np.array([[0.02, 0.02], [0.49, 0.51], [np.nan, 0.0]], dtype=np.float32)

    result = update_slam_map(
        existing,
        incoming,
        center_xy=(0.0, 0.0),
        voxel_size_m=0.1,
        retain_radius_m=5.0,
        max_points=100,
    )

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[0.02, 0.02], [0.49, 0.51]], atol=1e-6)


def test_slam_world_panel_shows_only_plus_minus_five_meters() -> None:
    kwargs = dict(
        pose=None,
        world_goal=None,
        trajectory_world=None,
        robot_history=None,
        span_m=10.0,
    )
    empty = render_slam_world_panel(np.empty((0, 2), dtype=np.float32), **kwargs)
    inside = render_slam_world_panel(np.array([[4.9, 0.0]], dtype=np.float32), **kwargs)
    outside = render_slam_world_panel(np.array([[5.1, 0.0]], dtype=np.float32), **kwargs)

    assert np.any(inside != empty)
    assert np.array_equal(outside, empty)


def test_navigation_protocol_round_trip_preserves_generation_and_goal() -> None:
    payload = build_navigation_message(
        mode="nav_goal",
        generation=7,
        goal_base=(2.0, 0.4),
        target="blue basket",
        target_type="global_target",
        confidence=0.91,
        timestamp=12.5,
    )

    command = decode_navigation_message(payload)

    assert command == NavigationCommand(
        mode="nav_goal",
        generation=7,
        timestamp=12.5,
        velocity=None,
        goal_base=(2.0, 0.4),
        target="blue basket",
        target_type="global_target",
        confidence=0.91,
    )


def test_world_goal_remains_fixed_as_robot_moves_and_rotates() -> None:
    world = base_goal_to_world((2.0, 0.0), Pose2D(1.0, 2.0, np.pi / 2))
    assert world == pytest.approx((1.0, 4.0))
    assert local_goal_from_world(world, Pose2D(1.0, 3.0, np.pi / 2)) == pytest.approx(
        (1.0, 0.0)
    )


def test_reached_status_returns_navdp_fixed_fastlio_world_goal() -> None:
    config = load_navdp_planner_config()
    state = _NavDPRuntimeState(
        generation=8,
        skill_id=3,
        segment_id=11,
        mode="nav_goal",
        world_goal=(4.25, -1.5),
    )
    cycle = _NavDPSensorCycle(
        pose=Pose2D(3.0, -1.5, math.pi / 2.0),
        pose_time=1.0,
        pose_history=[],
        slam_map_xy=np.empty((0, 2), dtype=np.float32),
        robot_history=np.empty((0, 2), dtype=np.float32),
    )
    statuses = []

    def send_status(status_state, reason, **fields):
        statuses.append((status_state, reason, fields))

    _schedule_navigation_inference(
        state,
        cycle,
        config=config,
        infer=lambda *_args: pytest.fail("goal within tolerance must not infer"),
        send_status=send_status,
    )

    assert state.mode == "stop"
    assert statuses == [(
        "reached",
        "goal_within_2m",
        {},
    )]
    assert _navigation_status_payload(
        state, statuses[0][0], statuses[0][1]
    ) == {
        "type": "sonic_navigation_status",
        "version": 1,
        "generation": 8,
        "skill_id": 3,
        "segment_id": 11,
        "state": "reached",
        "reason": "goal_within_2m",
        "goal_world": {"x": 4.25, "y": -1.5},
    }


def test_livox_filter_uses_translation_without_legacy_yaw_rotation() -> None:
    points = np.array(
        [[1.0, 0.25, -0.416], [0.249, 0.0, 0.0], [0.25, 0.0, 0.0]],
        dtype=np.float32,
    )

    filtered = filter_livox_points(points)

    assert filtered.shape == (2, 3)
    assert filtered[0] == pytest.approx([1.0002835, 0.25003, 0.00018], abs=1e-5)
    assert filtered[1] == pytest.approx([0.2502835, 0.00003, 0.41618], abs=1e-5)


def test_actor_ray_uses_3d_range_and_full_vertical_field() -> None:
    points = np.array([[1.0, 0.0, 2.0]], dtype=np.float32)

    rays = actor_ray_from_points(points)

    assert rays[90] == pytest.approx(np.sqrt(5.0))


def test_runtime_has_no_actor_ray_temporal_filter_state() -> None:
    sensors = _SharedSensors()
    assert not hasattr(sensors, "ray_history")


def test_sonic_directional_packet_can_share_motion_and_facing_heading() -> None:
    packet = SonicPlannerState().directional_message(
        speed=0.2,
        movement_heading=np.pi / 4,
        facing_heading=np.pi / 4,
    )
    values = struct.unpack("<i3f3f2f", packet[len(b"planner") + 1280 :])

    expected = np.sqrt(0.5)
    assert values[1:4] == pytest.approx((expected, expected, 0.0))
    assert values[4:7] == pytest.approx((expected, expected, 0.0))
    assert values[7] == pytest.approx(0.2)


def test_sonic_directional_packet_separates_translation_from_facing() -> None:
    packet = SonicPlannerState().directional_message(
        speed=0.2,
        movement_heading=np.pi / 2,
        facing_heading=0.0,
    )
    values = struct.unpack("<i3f3f2f", packet[len(b"planner") + 1280 :])

    assert values[1:4] == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert values[4:7] == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    assert values[7] == pytest.approx(0.2)


def test_internnav_reference_skips_first_three_points_and_uses_odom_world_frame() -> None:
    local = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.1]],
        dtype=np.float32,
    )

    world = prepare_internnav_world_reference(
        local,
        Pose2D(10.0, 20.0, np.pi / 2),
        interpolation_ratio=1,
    )

    np.testing.assert_allclose(world, [[10.0, 20.3], [9.9, 20.4]], atol=1e-6)


def test_camera_trajectory_uses_closest_timestamped_fastlio_pose() -> None:
    history = [
        (10.0, Pose2D(1.0, 0.0, 0.0)),
        (10.2, Pose2D(2.0, 0.0, 0.1)),
        (10.4, Pose2D(3.0, 0.0, 0.2)),
    ]

    pose = closest_timestamped_pose(history, 10.26)

    assert pose == Pose2D(2.0, 0.0, 0.1)


def test_navdp_runtime_has_no_heading_step_cap() -> None:
    config = load_navdp_planner_config()

    assert not hasattr(config, "max_heading_step_deg")


def test_shutdown_signal_handlers_turn_tmux_termination_into_cleanup(monkeypatch) -> None:
    installed = {}

    monkeypatch.setattr(signal, "signal", lambda signum, handler: installed.setdefault(signum, handler))

    install_shutdown_signal_handlers()

    assert {signal.SIGHUP, signal.SIGINT, signal.SIGTERM} <= installed.keys()
    with pytest.raises(KeyboardInterrupt):
        installed[signal.SIGTERM](signal.SIGTERM, None)


def test_actor_ray_control_text_reports_sent_speed_and_yaw_rate() -> None:
    text = format_actor_ray_control_text((0.18, 0.0, -0.32))

    assert text == "sent speed=0.180 m/s   wz=-0.320 rad/s"


def test_actor_ray_velocity_arrow_uses_vx_length_and_wz_deflection() -> None:
    start, end = actor_ray_velocity_arrow(
        (0.15, 99.0, 0.30),
        max_speed_mps=0.30,
        max_length_px=100,
        preview_s=1.0,
    )

    assert start == _VIZ_CENTER
    assert end[0] < start[0]
    assert end[1] < start[1]
    assert np.linalg.norm(np.subtract(end, start)) == pytest.approx(50.0, abs=1.0)
    assert end == (
        round(start[0] - np.sin(0.30) * 50.0),
        round(start[1] - np.cos(0.30) * 50.0),
    )


def test_mpc_defaults_match_xnavdp_g1_with_sonic_timing() -> None:
    controller = InternNavMpcController(
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    )

    assert controller.horizon_steps == 30
    assert controller.dt == pytest.approx(0.1)
    assert controller.desired_velocity == pytest.approx(0.2)
    assert controller.max_linear_velocity == pytest.approx(0.4)
    assert controller.max_angular_velocity == pytest.approx(0.8)
    assert controller.reference_gap == 3


def test_async_mpc_solver_does_not_block_the_control_thread() -> None:
    started = threading.Event()
    release = threading.Event()

    class Controller:
        def __init__(self, _reference) -> None:
            pass

        def update_reference(self, _reference) -> None:
            pass

        def solve(self, _pose):
            started.set()
            assert release.wait(1.0)
            return 0.2, 0.4

    solver = AsyncMpcSolver(controller_factory=Controller)
    request = MpcSolveRequest(
        generation=3,
        reference_version=5,
        world_reference=np.array([[0.0, 0.0], [1.0, 0.0]]),
        pose=Pose2D(0.0, 0.0, 0.0),
    )

    before = time.perf_counter()
    solver.submit(request)
    assert time.perf_counter() - before < 0.02
    assert started.wait(1.0)
    assert solver.poll_latest() is None

    release.set()
    deadline = time.monotonic() + 1.0
    result = None
    while result is None and time.monotonic() < deadline:
        result = solver.poll_latest()
        time.sleep(0.001)
    solver.close()

    assert result is not None
    assert result.generation == 3
    assert result.reference_version == 5
    assert result.control == pytest.approx((0.2, 0.4))
    assert result.error is None


def test_mpc_control_expires_without_a_recent_success() -> None:
    assert fresh_mpc_control(
        (0.2, 0.4), result_time=10.0, now=10.25, timeout_s=0.3
    ) == pytest.approx((0.2, 0.4))
    assert fresh_mpc_control(
        (0.2, 0.4), result_time=10.0, now=10.31, timeout_s=0.3
    ) == (0.0, 0.0)


def test_latest_message_worker_keeps_ros_callback_non_blocking() -> None:
    started = threading.Event()
    release = threading.Event()
    processed: list[int] = []

    def process(value: int) -> None:
        started.set()
        assert release.wait(1.0)
        processed.append(value)

    worker = LatestMessageWorker(process)
    before = time.perf_counter()
    worker.submit(1)
    assert time.perf_counter() - before < 0.02
    assert started.wait(1.0)

    # While the first message is expensive, only retain the newest arrival.
    worker.submit(2)
    worker.submit(3)
    release.set()
    deadline = time.monotonic() + 1.0
    while processed != [1, 3] and time.monotonic() < deadline:
        time.sleep(0.001)
    worker.close()

    assert processed == [1, 3]


def test_point_plane_distances_match_plane_equation_without_matrix_multiply() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]], dtype=np.float32)
    normal = np.array([0.2, -0.3, 0.5], dtype=np.float32)
    offset = 0.7

    distances = point_plane_distances(points, normal, offset)

    assert distances == pytest.approx(np.abs(points @ normal - offset))


def test_valid_zero_macro_action_aborts_navigation_but_waiting_does_not() -> None:
    assert should_abort_nav_for_zero_action(
        mode="nav_goal", selected_command=(0.0, 0.0, 0.0), command_available=True
    )
    assert not should_abort_nav_for_zero_action(
        mode="nav_goal", selected_command=(0.0, 0.0, 0.0), command_available=False
    )
    assert not should_abort_nav_for_zero_action(
        mode="manual_velocity", selected_command=(0.0, 0.0, 0.0), command_available=True
    )


def test_depth_stop_requires_component_strictly_larger_than_2000_pixels() -> None:
    depth = np.ones((60, 60), dtype=np.float32)
    depth.flat[:2000] = 0.09
    assert not depth_requires_stop(depth)
    depth.flat[2000] = 0.09
    assert depth_requires_stop(depth)

    depth.flat[:2001] = 0.11
    assert not depth_requires_stop(depth)
