from __future__ import annotations

import queue
import json
import struct
import sys
import signal
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest

from gear_sonic.scripts.navdp_planner import (
    NavigationCommand,
    Pose2D,
    SonicPlannerState,
    base_goal_to_world,
    build_navigation_message,
    decode_navigation_message,
    depth_requires_stop,
    filter_livox_points,
    format_navigation_diagnostics,
    integrate_velocity_path,
    local_goal_from_world,
    compose_reasan_navigation_view,
    render_actor_ray_panel,
    render_slam_world_panel,
    local_trajectory_to_world,
    update_slam_map,
    should_abort_nav_for_zero_action,
)
from gear_sonic.scripts import navdp_planner


def test_runtime_queue_dependency_is_imported_at_module_scope() -> None:
    assert navdp_planner.queue is queue


def test_navdp_goal_tolerance_matches_production_profile() -> None:
    assert navdp_planner.NavDPPlannerConfig().goal_tolerance_m == pytest.approx(0.5)


def test_navdp_runs_unthrottled_inference_with_ten_hz_mpc() -> None:
    config = navdp_planner.NavDPPlannerConfig()
    assert config.control_hz == pytest.approx(20.0)
    assert config.mpc_hz == pytest.approx(10.0)
    assert config.mpc_result_timeout_s == pytest.approx(0.3)
    assert not hasattr(config, "inference_hz")
    assert config.heading_preview_s == pytest.approx(0.6)
    assert config.radar_timeout_s == pytest.approx(0.75)
    assert config.trajectory_timeout_s == pytest.approx(2.5)


def test_xnavdp_g1_mpc_defaults_keep_three_second_horizon() -> None:
    defaults = navdp_planner.XNAVDP_G1_MPC_DEFAULTS

    assert defaults == {
        "horizon_steps": 30,
        "desired_velocity": 0.3,
        "max_linear_velocity": 0.3,
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
    assert navdp_planner.xnavdp_control_to_body_velocity(0.3, 0.5) == pytest.approx(
        (0.3, 0.0, 0.5)
    )
    assert navdp_planner.xnavdp_control_to_body_velocity(-0.2, -0.4) == pytest.approx(
        (-0.2, 0.0, -0.4)
    )


def test_sonic_target_heading_integrates_the_original_mpc_yaw_rate() -> None:
    heading = navdp_planner.sonic_heading_from_mpc(
        fastlio_yaw=0.4,
        fastlio_to_sonic_offset=0.2,
        mpc_angular_velocity=0.5,
        heading_preview_s=0.6,
    )

    assert heading == pytest.approx(0.90)


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

    trajectory = navdp_planner._navdp_request(
        "http://127.0.0.1:19999",
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8), dtype=np.float32),
        (1.0, 0.0),
    )

    np.testing.assert_allclose(trajectory, [[0.0, 0.0], [0.2, 0.1], [0.4, -0.2]])


def test_xnavdp_adaptive_speed_matches_length_and_curvature_limits() -> None:
    kwargs = {"max_angular_velocity": 0.8, "curvature_speed_gain": 0.15}
    assert navdp_planner.xnavdp_adaptive_speed(2.0, 0.0, **kwargs) == pytest.approx(0.3)
    assert navdp_planner.xnavdp_adaptive_speed(1.0, 0.0, **kwargs) == pytest.approx(0.15)
    assert navdp_planner.xnavdp_adaptive_speed(0.01, 0.0, **kwargs) == pytest.approx(0.0015)
    assert navdp_planner.xnavdp_adaptive_speed(2.0, 10.0, **kwargs) == pytest.approx(0.012)


def test_navdp_sensor_state_starts_empty() -> None:
    sensors = navdp_planner._SharedSensors()
    assert sensors.slam_map_xy.shape == (0, 2)
    assert sensors.robot_history.shape == (0, 2)


def test_control_freshness_snapshot_reads_latest_ros_callback_values() -> None:
    sensors = navdp_planner._SharedSensors()
    sensors.pose = Pose2D(1.0, 2.0, 0.3)
    sensors.pose_time = 12.0
    sensors.points = np.array([[0.5, 0.1, 0.2]], dtype=np.float32)
    sensors.points_time = 13.0

    pose, pose_time, points, points_time = navdp_planner._control_freshness_snapshot(
        sensors
    )

    assert pose == Pose2D(1.0, 2.0, 0.3)
    assert pose_time == pytest.approx(12.0)
    np.testing.assert_allclose(points, [[0.5, 0.1, 0.2]])
    assert points_time == pytest.approx(13.0)


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

    navdp_planner._reset_navdp(
        "http://127.0.0.1:19999",
        {"fx": 500.0, "fy": 501.0, "cx": 320.0, "cy": 240.0},
        stop_threshold=-4.0,
    )

    assert captured["json"]["stop_threshold"] == [-4.0]


def test_navdp_camera_extraction_uses_aligned_ego_view_rgbd() -> None:
    rgb = np.array([[[250, 20, 5]]], dtype=np.uint8)
    depth = np.array([[1000]], dtype=np.uint16)

    extracted_rgb, extracted_depth, _ = navdp_planner._extract_camera_frame(
        {
            "images": {
                "chest_view": np.zeros_like(rgb),
                "chest_view_depth": np.full_like(depth, 2000),
                "ego_view": rgb,
                "ego_view_depth": depth,
            },
            "camera_info": {"ego_view": {"depth_scale_m": 0.001}},
        }
    )

    np.testing.assert_array_equal(extracted_rgb, rgb)
    np.testing.assert_allclose(extracted_depth, [[1.0]])


def test_head_depth_panel_uses_fixed_zero_to_five_meter_scale() -> None:
    near = navdp_planner.render_head_depth_panel(
        np.array([[1.0, 5.0, np.nan]], dtype=np.float32)
    )
    repeated = navdp_planner.render_head_depth_panel(
        np.array([[1.0, 3.0, np.nan]], dtype=np.float32)
    )

    assert near.shape == (1, 3, 3)
    np.testing.assert_array_equal(near[0, 0], repeated[0, 0])
    np.testing.assert_array_equal(near[0, 2], [0, 0, 0])


def test_head_rgbd_view_uses_black_panels_when_sources_are_missing() -> None:
    missing = navdp_planner.compose_head_rgbd_view(None, None, panel_size=(4, 3))
    rgb_only = navdp_planner.compose_head_rgbd_view(
        np.full((2, 2, 3), 127, dtype=np.uint8), None, panel_size=(4, 3)
    )

    assert missing.shape == (3, 8, 3)
    assert not np.any(missing)
    assert np.any(rgb_only[:, :4])
    assert not np.any(rgb_only[:, 4:])


def test_head_rgb_is_converted_to_bgr_for_opencv_only() -> None:
    view = navdp_planner.compose_head_rgbd_view(
        np.array([[[255, 0, 0]]], dtype=np.uint8), None, panel_size=(1, 1)
    )

    np.testing.assert_array_equal(view[0, 0], [0, 0, 255])


def test_navdp_rgb_uses_official_jpeg_and_depth_uses_png() -> None:
    rgb_bytes, depth_bytes = navdp_planner._encode_navdp_frames(
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

    trajectory = navdp_planner._navdp_request(
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

    navdp_planner._navdp_request(
        "http://127.0.0.1:19999",
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8), dtype=np.float32),
        (1.0, 0.0),
    )

    assert captured["timeout"] == pytest.approx(10.0)


def test_reasan_actor_ray_and_world_map_are_present_beside_physical_view() -> None:
    rays = np.full(180, 3.0, dtype=np.float32)
    rays[90] = 0.5
    trajectory = np.array([[0.0, 0.0], [0.25, 0.05], [0.5, 0.1]], dtype=np.float32)
    actor = render_actor_ray_panel(rays, trajectory=trajectory)
    frame = compose_reasan_navigation_view(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        rays,
        trajectory,
        slam_map_xy=np.array([[10.0, 20.0]], dtype=np.float32),
        pose=Pose2D(10.0, 20.0, 0.0),
        world_goal=(11.0, 20.0),
        robot_history=np.array([[9.5, 20.0], [10.0, 20.0]], dtype=np.float32),
    )

    assert actor.shape == (500, 500, 3)
    assert frame.shape == (500, 1500, 3)
    assert np.any(actor != 20)
    assert np.array_equal(frame[:, 500:1000], actor)
    assert np.any(frame[:, 1000:] != 20)


def test_navdp_prediction_changes_only_actor_ray_panel() -> None:
    rays = np.full(180, 3.0, dtype=np.float32)
    common = dict(
        points_base=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        ranges_m=rays,
        slam_map_xy=np.array([[0.5, 0.5]], dtype=np.float32),
        pose=Pose2D(0.0, 0.0, 0.0),
        world_goal=(2.0, 0.0),
        robot_history=np.array([[-0.2, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    empty = compose_reasan_navigation_view(trajectory=np.empty((0, 2)), **common)
    predicted = compose_reasan_navigation_view(
        trajectory=np.array([[0.0, 0.0], [0.25, 0.1], [0.5, 0.2]], dtype=np.float32),
        **common,
    )

    assert np.array_equal(predicted[:, :500], empty[:, :500])
    assert np.any(predicted[:, 500:1000] != empty[:, 500:1000])
    assert np.array_equal(predicted[:, 1000:], empty[:, 1000:])


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


def test_tmp_client_velocity_integrator_remains_30_steps_for_diagnostics() -> None:
    path = integrate_velocity_path((0.3, 0.0, 0.0))
    assert path.shape == (31, 2)
    assert path[-1] == pytest.approx([0.45, 0.0], abs=1e-5)


def test_sonic_planner_packet_preserves_positive_left_lateral_direction() -> None:
    packet = SonicPlannerState().message((0.0, 0.15, 0.0), dt=0.05)
    values = struct.unpack("<i3f3f2f", packet[len(b"planner") + 1280 :])
    assert values[1:4] == pytest.approx((0.0, 1.0, 0.0))
    assert values[7] == pytest.approx(0.15)


def test_navigation_diagnostics_include_confidence_goals_and_both_paths() -> None:
    text = format_navigation_diagnostics(
        generation=4,
        confidence=0.95,
        local_goal=(1.2, -0.3),
        world_goal=(4.0, 5.0),
        trajectory=np.array([[0.0, 0.0], [0.4, -0.1]], dtype=np.float32),
        velocity=(0.3, -0.05, 0.1),
    )

    assert "generation=4 lavira_confidence=0.950" in text
    assert "local_goal=(1.200, -0.300) world_goal=(4.000, 5.000)" in text
    assert "navdp_local_trajectory=" in text
    assert "sent_velocity=(0.300, -0.050, 0.100)" in text
    assert "sent_velocity_path_30x0.05s=" in text


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
        confidence=pytest.approx(0.91),
    )


def test_world_goal_remains_fixed_as_robot_moves_and_rotates() -> None:
    world = base_goal_to_world((2.0, 0.0), Pose2D(1.0, 2.0, np.pi / 2))
    assert world == pytest.approx((1.0, 4.0))
    assert local_goal_from_world(world, Pose2D(1.0, 3.0, np.pi / 2)) == pytest.approx(
        (1.0, 0.0)
    )


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

    rays = navdp_planner.actor_ray_from_points(points)

    assert rays[90] == pytest.approx(np.sqrt(5.0))


def test_near_radar_point_is_visualized_without_stopping_control_output() -> None:
    velocity, rays, camera_stop = navdp_planner._prepare_control_output(
        (0.3, 0.0, 0.1),
        np.array([[0.09, 0.0, 0.0]], dtype=np.float32),
        np.ones((60, 60), dtype=np.float32),
    )

    assert velocity == pytest.approx((0.3, 0.0, 0.1))
    assert rays[90] == pytest.approx(0.09)
    assert not camera_stop


def test_depth_stop_still_zeros_control_output() -> None:
    depth = np.ones((60, 60), dtype=np.float32)
    depth.flat[:2001] = 0.09

    velocity, _, camera_stop = navdp_planner._prepare_control_output(
        (0.3, 0.0, 0.1),
        np.empty((0, 3), dtype=np.float32),
        depth,
    )

    assert velocity == (0.0, 0.0, 0.0)
    assert camera_stop


def test_runtime_has_no_actor_ray_temporal_filter_state() -> None:
    sensors = navdp_planner._SharedSensors()
    assert not hasattr(sensors, "ray_history")


def test_sonic_arc_packet_uses_same_world_direction_for_motion_and_facing() -> None:
    packet = SonicPlannerState().arc_message(speed=0.2, heading=np.pi / 4)
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

    world = navdp_planner.prepare_internnav_world_reference(
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

    pose = navdp_planner.closest_timestamped_pose(history, 10.26)

    assert pose == Pose2D(2.0, 0.0, 0.1)


def test_navdp_runtime_has_no_heading_step_cap() -> None:
    config = navdp_planner.NavDPPlannerConfig()

    assert not hasattr(config, "max_heading_step_deg")
    assert not hasattr(navdp_planner, "mpc_twist_to_sonic_target")


def test_actor_ray_recorder_publishes_mp4_only_after_it_is_finalized(tmp_path) -> None:
    rays = np.full(180, 3.0, dtype=np.float32)
    trajectory = np.array([[0.0, 0.0], [0.4, 0.1], [0.8, 0.2]], dtype=np.float32)
    panel = render_actor_ray_panel(rays, trajectory=trajectory)
    recorder = navdp_planner.ActorRayVideoRecorder(tmp_path, fps=20.0)

    recorder.write(panel)
    output_path = recorder.output_path
    working_path = recorder.working_path

    assert output_path is not None
    assert output_path.suffix == ".mp4"
    assert not output_path.exists()
    assert working_path is not None and working_path.name.endswith(".recording.mp4")

    recorder.close()

    assert output_path.parent == tmp_path
    assert output_path.name.startswith("actorray_")
    assert output_path.stat().st_size > 0
    assert not working_path.exists()

    import cv2

    capture = cv2.VideoCapture(str(output_path))
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    ok, decoded = capture.read()
    capture.release()
    assert ok
    assert fourcc.lower() in {"avc1", "h264"}
    assert decoded.shape[:2] == panel.shape[:2]


def test_shutdown_signal_handlers_turn_tmux_termination_into_cleanup(monkeypatch) -> None:
    installed = {}

    monkeypatch.setattr(signal, "signal", lambda signum, handler: installed.setdefault(signum, handler))

    navdp_planner.install_shutdown_signal_handlers()

    assert {signal.SIGHUP, signal.SIGINT, signal.SIGTERM} <= installed.keys()
    with pytest.raises(KeyboardInterrupt):
        installed[signal.SIGTERM](signal.SIGTERM, None)


def test_actor_ray_recording_session_creates_one_video_per_navigation(tmp_path) -> None:
    panel = np.zeros((64, 64, 3), dtype=np.uint8)
    session = navdp_planner.ActorRayRecordingSession(tmp_path, fps=20.0)

    session.write(panel)
    assert session.output_path is None

    session.start(generation=7)
    session.write(panel)
    first_path = session.output_path
    assert first_path is not None and not first_path.exists()
    session.stop()
    assert first_path.exists()

    session.start(generation=8)
    session.write(panel)
    second_path = session.output_path
    session.stop()

    assert first_path is not None and first_path.exists()
    assert second_path is not None and second_path.exists()
    assert first_path != second_path
    assert first_path.name.startswith("actorray_g000007_")
    assert second_path.name.startswith("actorray_g000008_")


def test_direction_chain_diagnostics_exposes_every_yaw_sign() -> None:
    text = navdp_planner.format_direction_chain_diagnostics(
        trajectory=np.array([[0.0, 0.0], [0.2, 0.1]], dtype=np.float32),
        mpc_angular_velocity=0.25,
        fastlio_yaw=0.4,
        fastlio_yaw_delta=-0.03,
        sonic_target_heading=0.62,
    )

    assert "path_dy=+0.100" in text
    assert "mpc_wz=+0.250" in text
    assert "fastlio_yaw=+0.400" in text
    assert "fastlio_dyaw=-0.030" in text
    assert "sonic_heading=+0.620" in text


def test_actor_ray_control_text_reports_sent_speed_and_yaw_rate() -> None:
    text = navdp_planner.format_actor_ray_control_text((0.18, 0.0, -0.32))

    assert text == "sent speed=0.180 m/s   wz=-0.320 rad/s"


def test_actor_ray_velocity_arrow_uses_vx_length_and_wz_deflection() -> None:
    start, end = navdp_planner.actor_ray_velocity_arrow(
        (0.15, 99.0, 0.30),
        max_speed_mps=0.30,
        max_length_px=100,
        preview_s=1.0,
    )

    assert start == navdp_planner._VIZ_CENTER
    assert end[0] < start[0]
    assert end[1] < start[1]
    assert np.linalg.norm(np.subtract(end, start)) == pytest.approx(50.0, abs=1.0)
    assert end == (
        round(start[0] - np.sin(0.30) * 50.0),
        round(start[1] - np.cos(0.30) * 50.0),
    )


def test_mpc_defaults_match_xnavdp_g1_with_sonic_timing() -> None:
    controller = navdp_planner.InternNavMpcController(
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    )

    assert controller.horizon_steps == 30
    assert controller.dt == pytest.approx(0.1)
    assert controller.desired_velocity == pytest.approx(0.15)
    assert controller.max_linear_velocity == pytest.approx(0.3)
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

    solver = navdp_planner.AsyncMpcSolver(controller_factory=Controller)
    request = navdp_planner.MpcSolveRequest(
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
    assert navdp_planner.fresh_mpc_control(
        (0.2, 0.4), result_time=10.0, now=10.25, timeout_s=0.3
    ) == pytest.approx((0.2, 0.4))
    assert navdp_planner.fresh_mpc_control(
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

    worker = navdp_planner.LatestMessageWorker(process)
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

    distances = navdp_planner.point_plane_distances(points, normal, offset)

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
