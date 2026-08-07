from __future__ import annotations

import queue
import json
import struct
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from gear_sonic.scripts.navdp_planner import (
    NavigationCommand,
    Pose2D,
    SonicPlannerState,
    apply_hard_safety,
    base_goal_to_world,
    build_navigation_message,
    build_planner_velocity_message,
    camera_point_to_base,
    decode_navigation_message,
    depth_requires_stop,
    filter_livox_points,
    format_navigation_diagnostics,
    integrate_velocity_path,
    local_goal_from_world,
    livox_custom_points_to_numpy,
    compose_reasan_navigation_view,
    render_actor_ray_panel,
    render_slam_world_panel,
    local_trajectory_to_world,
    update_slam_map,
    should_abort_nav_for_lidar,
    should_abort_nav_for_zero_action,
)
from gear_sonic.scripts import navdp_planner


def test_runtime_queue_dependency_is_imported_at_module_scope() -> None:
    assert navdp_planner.queue is queue


def test_humanoid_fastlio_default_odometry_topic_matches_fork() -> None:
    assert navdp_planner.NavDPPlannerConfig().odom_topic == "/Odometry_loc"


def test_navdp_goal_tolerance_is_forty_centimeters() -> None:
    assert navdp_planner.NavDPPlannerConfig().goal_tolerance_m == pytest.approx(0.40)


def test_navdp_runs_unthrottled_inference_with_ten_hz_mpc() -> None:
    config = navdp_planner.NavDPPlannerConfig()
    assert config.control_hz == pytest.approx(20.0)
    assert config.mpc_hz == pytest.approx(10.0)
    assert not hasattr(config, "inference_hz")
    assert config.heading_preview_s == pytest.approx(0.5)


def test_xnavdp_g1_mpc_defaults_keep_three_second_horizon() -> None:
    defaults = navdp_planner.XNAVDP_G1_MPC_DEFAULTS

    assert defaults == {
        "horizon_steps": 30,
        "desired_velocity": 0.3,
        "max_linear_velocity": 0.3,
        "max_angular_velocity": 0.5,
        "reference_gap": 3,
        "dt": 0.1,
        "reference_trajectory_length_m": 2.0,
        "minimum_desired_velocity": 0.05,
        "interpolation_ratio": 50,
        "lookahead_points": 10,
    }


def test_xnavdp_speed_mapping_is_unicycle_without_lateral_velocity() -> None:
    assert navdp_planner.xnavdp_control_to_body_velocity(0.3, 0.5) == pytest.approx(
        (0.3, 0.0, 0.5)
    )
    assert navdp_planner.xnavdp_control_to_body_velocity(-0.2, -0.4) == pytest.approx(
        (-0.2, 0.0, -0.4)
    )


def test_xnavdp_adaptive_speed_matches_length_and_curvature_limits() -> None:
    assert navdp_planner.xnavdp_adaptive_speed(2.0, 0.0) == pytest.approx(0.3)
    assert navdp_planner.xnavdp_adaptive_speed(1.0, 0.0) == pytest.approx(0.15)
    assert navdp_planner.xnavdp_adaptive_speed(0.01, 0.0) == pytest.approx(0.05)
    assert navdp_planner.xnavdp_adaptive_speed(2.0, 10.0) == pytest.approx(0.05)


def test_humanoid_fastlio_registered_cloud_topic_matches_fork() -> None:
    assert navdp_planner.NavDPPlannerConfig().slam_cloud_topic == "/cloud_registered_1"
    sensors = navdp_planner._SharedSensors()
    assert sensors.slam_map_xy.shape == (0, 2)
    assert sensors.robot_history.shape == (0, 2)


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


def test_livox_custom_message_points_convert_to_xyz_array() -> None:
    message = SimpleNamespace(points=[
        SimpleNamespace(x=1.0, y=-0.2, z=0.3),
        SimpleNamespace(x=2.0, y=0.4, z=-0.5),
    ])

    points = livox_custom_points_to_numpy(message)

    assert points.dtype == np.float32
    np.testing.assert_allclose(points, [[1.0, -0.2, 0.3], [2.0, 0.4, -0.5]])


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


def test_camera_optical_point_maps_right_to_negative_base_y() -> None:
    assert camera_point_to_base((0.2, 0.1, 2.0)) == pytest.approx(
        (2.0, -0.2, 0.3)
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


def test_runtime_has_no_actor_ray_temporal_filter_state() -> None:
    sensors = navdp_planner._SharedSensors()
    assert not hasattr(sensors, "ray_history")


def test_forward_135_degree_sector_stops_translation_but_not_pure_turning() -> None:
    points = np.array([[0.09, 0.0, 0.0]], dtype=np.float32)

    assert apply_hard_safety((0.3, 0.0, 0.1), points, camera_stop=False) == (
        0.0,
        0.0,
        0.0,
    )
    assert apply_hard_safety((0.0, 0.0, 0.4), points, camera_stop=False) == (
        0.0,
        0.0,
        0.4,
    )
    assert apply_hard_safety((-0.2, 0.0, 0.0), points, camera_stop=False) == (
        -0.2,
        0.0,
        0.0,
    )
    assert apply_hard_safety(
        (0.3, 0.0, 0.0), np.array([[0.11, 0.0, 0.0]], dtype=np.float32), camera_stop=False
    ) == (0.3, 0.0, 0.0)


@pytest.mark.parametrize(
    ("velocity", "obstacle"),
    [
        ((0.0, 0.15, 0.0), (0.0, 0.09, 0.0)),
        ((0.0, -0.15, 0.0), (0.0, -0.09, 0.0)),
        ((-0.3, 0.0, 0.0), (-0.09, 0.0, 0.0)),
        ((0.2, 0.1, 0.0), (0.08, 0.04, 0.0)),
    ],
)
def test_hard_safety_rotates_sector_with_xy_translation(
    velocity: tuple[float, float, float], obstacle: tuple[float, float, float]
) -> None:
    assert apply_hard_safety(
        velocity, np.array([obstacle], dtype=np.float32), camera_stop=False
    ) == (0.0, 0.0, 0.0)


def test_lateral_motion_ignores_obstacle_outside_its_motion_sector() -> None:
    assert apply_hard_safety(
        (0.0, 0.15, 0.0),
        np.array([[0.0, -0.35, 0.0]], dtype=np.float32),
        camera_stop=False,
    ) == (0.0, 0.15, 0.0)


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


def test_world_path_tangent_preserves_lateral_translation() -> None:
    heading = navdp_planner.world_path_tangent_heading(
        np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]], dtype=np.float64),
        Pose2D(0.0, 0.0, 0.0),
    )

    assert heading == pytest.approx(np.pi / 4)


def test_arc_safety_velocity_is_expressed_from_measured_heading() -> None:
    velocity = navdp_planner.arc_target_to_body_velocity(
        speed=0.3,
        target_heading=0.6,
        actual_heading=0.4,
    )

    assert velocity == pytest.approx(
        (0.3 * np.cos(0.2), 0.3 * np.sin(0.2), 0.0)
    )


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


def test_mpc_twist_to_sonic_heading_uses_measured_yaw_and_preview_without_accumulation() -> None:
    first = navdp_planner.mpc_twist_to_sonic_target(
        linear_velocity=0.3,
        angular_velocity=0.4,
        odom_yaw=1.0,
        sonic_yaw_offset=-0.5,
        preview_s=0.5,
        max_heading_step_deg=10.0,
    )
    repeated = navdp_planner.mpc_twist_to_sonic_target(
        linear_velocity=0.3,
        angular_velocity=0.4,
        odom_yaw=1.0,
        sonic_yaw_offset=-0.5,
        preview_s=0.5,
        max_heading_step_deg=10.0,
    )

    assert repeated == pytest.approx(first)
    assert first == pytest.approx((0.3, 0.5 + np.deg2rad(10.0)))


def test_mpc_defaults_match_xnavdp_g1_with_sonic_timing() -> None:
    controller = navdp_planner.InternNavMpcController(
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    )

    assert controller.horizon_steps == 30
    assert controller.dt == pytest.approx(0.1)
    assert controller.desired_velocity == pytest.approx(0.15)
    assert controller.max_linear_velocity == pytest.approx(0.3)
    assert controller.max_angular_velocity == pytest.approx(0.5)
    assert controller.reference_gap == 3


def test_only_lidar_hard_stop_aborts_active_navigation() -> None:
    assert should_abort_nav_for_lidar(
        mode="nav_goal",
        before_safety=(0.3, 0.0, 0.0),
        after_safety=(0.0, 0.0, 0.0),
        camera_stop=False,
    )
    assert not should_abort_nav_for_lidar(
        mode="nav_goal",
        before_safety=(0.3, 0.0, 0.0),
        after_safety=(0.0, 0.0, 0.0),
        camera_stop=True,
    )
    assert not should_abort_nav_for_lidar(
        mode="manual_velocity",
        before_safety=(0.3, 0.0, 0.0),
        after_safety=(0.0, 0.0, 0.0),
        camera_stop=False,
    )


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


def test_sonic_message_uses_existing_planner_contract() -> None:
    message = build_planner_velocity_message((0.3, -0.1, 0.2), timestamp=8.0)
    assert message["type"] == "navila_reasan_velocity_command"
    assert message["velocity"] == {"vx": 0.3, "vy": -0.1, "wz": 0.2}
    assert message["duration_s"] == pytest.approx(0.05)
