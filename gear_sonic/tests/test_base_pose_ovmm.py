"""OVMM boundary checks. No Habitat, Torch, Gateway, or detector is required."""

from dataclasses import replace
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gear_sonic.utils.inference.base_pose.servo import (
    RawServoCalibration,
    ServoCommand,
    ServoPhase,
    VisualServoController,
    _observation,
)
from gear_sonic.utils.inference.base_pose.ovmm.actions import (
    ActionLimits,
    ResidualActionAdapter,
)
from gear_sonic.utils.inference.base_pose.ovmm.config import BasePoseConfig
from gear_sonic.utils.inference.base_pose.ovmm.controller import (
    AlignmentStatus,
    BasePoseSession,
)
from gear_sonic.utils.inference.base_pose.ovmm.observation import (
    adapt_observation,
    body_pose,
    camera_pose_in_start_frame,
)
from gear_sonic.utils.inference.base_pose.ovmm.targets import (
    MissingTarget,
    SemanticTargetProvider,
    _instances,
)
from gear_sonic.utils.inference.base_pose.ovmm.acquisition import (
    TargetMemory,
    tilt_sweep,
)


def observation(
    distance=0.65, instance_ids=(0, 1), gps=(0.0, 0.0), heading=0.0, edge=True
):
    rgb = np.zeros((160, 240, 3), dtype=np.uint8)
    depth = np.full((160, 240), distance, dtype=np.float32)
    semantic = np.zeros((160, 240), dtype=np.int32)
    semantic[95:150, 20:220] = 2
    semantic[40:90, 100:140] = 1
    if edge:
        rgb[95:150, 20:220] = 255
    instances = np.full(semantic.shape, -1, dtype=int)
    instances[semantic == 1], instances[semantic == 2] = instance_ids
    camera = np.eye(4)
    camera[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    return SimpleNamespace(
        rgb=rgb,
        depth=depth,
        semantic=semantic,
        instance=instances,
        camera_K=np.array([[200.0, 0.0, 120.0], [0.0, 200.0, 80.0], [0.0, 0.0, 1.0]]),
        camera_pose=body_pose(gps, np.array([heading])) @ camera,
        gps=np.array(gps),
        compass=np.array([heading]),
        joint=np.zeros(10),
        task_observations={"object_goal": 1, "start_recep_goal": 2},
    )


def targets(provider, obs):
    snapshot, calibration, start = adapt_observation(
        obs, timestamp=0.0, use_opencv_camera_pose=True
    )
    return provider.get_alignment_targets(obs, "pick", snapshot, calibration, start)


@pytest.mark.parametrize("reference_category", [2, 7, 42])
def test_every_reference_category_uses_raw_separate_instances(reference_category):
    obs = observation()
    target, reference = obs.semantic == 1, obs.semantic == 2
    other = np.zeros_like(reference)
    other[5:30, 5:30] = True
    obs.semantic[reference] = 0  # Entire reference removed by depth filtering.
    obs.instance[reference] = -1
    obs.task_observations.update(
        start_recep_goal=reference_category,
        instance_masks_raw=np.stack([target, reference, other]),
        instance_classes=np.array([1, reference_category, reference_category]),
        instance_scores=np.array([0.9, 0.8, 0.7]),
    )
    raw = _instances(obs, reference_category, raw=True)
    assert len(raw) == 2
    np.testing.assert_array_equal(raw[0].mask, reference)
    np.testing.assert_array_equal(raw[1].mask, other)
    selected = targets(SemanticTargetProvider(), obs)
    assert selected.reference_mask_source == "detic_raw"
    assert selected.yaw_reference is not None
    np.testing.assert_array_equal(selected.target.mask, target)


def test_malformed_raw_instances_are_not_silently_replaced_by_filtered_masks():
    obs = observation()
    obs.task_observations.update(
        instance_masks_raw=np.zeros((2, 10, 10)), instance_classes=[1, 2]
    )
    with pytest.raises(MissingTarget, match="current frame"):
        targets(SemanticTargetProvider(), obs)


def test_navigation_memory_uses_odometry_and_expires_without_fake_freshness():
    memory = TargetMemory(
        BasePoseConfig(target_memory_max_age_frames=20), use_opencv_camera_pose=True
    )
    obs = observation()
    memory.observe_navigation(obs, 10)
    assert memory.last_seen_frame == 10
    absent = observation()
    absent.semantic[absent.semantic == 1] = 0
    absent.camera_pose[2, 3] += 0.5  # A higher camera sees the old point lower.
    memory.observe_navigation(absent, 11)
    tilt, diagnostic = memory.guided_tilt(absent, 11)
    assert tilt < 0
    assert diagnostic["predicted_target_uv"][1] > absent.rgb.shape[0]
    assert memory.last_seen_frame == 10
    assert memory.guided_tilt(absent, 31)[0] is None
    memory.reset()
    assert not memory.available(31)


def test_full_tilt_sweep_samples_the_range_in_bounded_increments():
    angles = tilt_sweep(-30.0, -85.0, 45.0, 15.0)
    assert min(angles) == -85.0
    assert angles[-1] == 45.0
    assert np.all(np.abs(np.diff([-30.0, *angles])) <= 15.0 + 1e-12)
    assert len(angles) < 20
    with pytest.raises(ValueError):
        tilt_sweep(0.0, -85.0, 45.0, 0.0)


def test_head_reacquisition_resets_yaw_history_without_renewing_total_budget():
    session = BasePoseSession(
        BasePoseConfig(max_steps=10), ActionLimits(), use_opencv_camera_pose=True
    )
    session.reset(start_frame_id=100)
    session.step(observation(), frame_id=101)
    session.actions.residual[:] = 0.03
    session.restart_after_acquisition(108)
    assert session.start_frame_id == 100
    assert session.controller.phase is ServoPhase.FORWARD_APPROACH
    np.testing.assert_array_equal(session.actions.residual, 0.0)
    assert session.step(observation(), frame_id=110).status is AlignmentStatus.TIMEOUT


def test_metric_depth_filters_sentinels_without_integer_quantization():
    obs = observation(0.65321)
    obs.depth[0, :5] = [np.nan, np.inf, 10000, 10001, -1]
    snapshot, calibration, _ = adapt_observation(
        obs, timestamp=0.1, use_opencv_camera_pose=True
    )
    assert snapshot.depth_scale_m == 1
    assert np.all(snapshot.depth_raw[0, :5] == 0)
    assert snapshot.depth_raw[50, 120] == pytest.approx(0.65321, abs=1e-7)
    assert np.isnan(obs.depth[0, 0])  # Other official modules keep their input.
    np.testing.assert_allclose(
        calibration.camera_to_body([[0, 0, 1], [1, 0, 1], [0, 1, 1]]),
        [[1, 0, 0], [1, -1, 0], [1, 0, -1]],
    )


@pytest.mark.parametrize("heading", [0.0, 0.4, -2.0])
def test_legacy_camera_conversion_matches_habitat_sensor_axes(heading):
    obs = observation(gps=(1.2, -0.4), heading=heading)
    # Construct Habitat's raw CameraPoseSensor from its GL optical basis and
    # Stretch's Rx(-90 deg) base convention, then HomeRobot's ZXY permutation.
    flu_to_hab = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    raw_gl = flu_to_hab @ obs.camera_pose @ np.diag([1.0, -1.0, -1.0, 1.0])
    permutation = [2, 0, 1, 3]
    legacy = raw_gl[np.ix_(permutation, permutation)]
    np.testing.assert_allclose(
        camera_pose_in_start_frame(legacy, use_opencv_camera_pose=False),
        obs.camera_pose,
        atol=1e-12,
    )
    obs.camera_pose = legacy
    _, calibration, _ = adapt_observation(obs, timestamp=0.0)
    np.testing.assert_allclose(
        calibration.camera_to_body([[0.0, 0.0, 1.0]]), [[1.0, 0.0, 0.0]], atol=1e-12
    )


def test_waypoints_cross_deadzone_after_float32_environment_normalization():
    adapter = ResidualActionAdapter(ActionLimits())
    outputs = [
        adapter.step(ServoCommand(0.35, 0.0, 0.3), phase="yaw") for _ in range(12)
    ]
    assert np.all(outputs[0] == 0)
    assert sum(x[0] for x in outputs) == pytest.approx(0.21)
    for x in outputs:
        translation = np.asarray(x[:2] / 1.0, dtype=np.float32) * 1.0
        angle = np.float32(x[2] / math.pi) * math.pi
        if np.linalg.norm(translation):
            assert 0.1 <= np.linalg.norm(translation) <= 0.25
        if angle:
            assert math.radians(5) <= abs(angle) <= math.radians(30)


@pytest.mark.parametrize("change", ["reverse", "stop", "phase"])
def test_stale_residual_never_executes_after_direction_or_phase_change(change):
    adapter = ResidualActionAdapter(ActionLimits())
    for _ in range(5):
        adapter.step(ServoCommand(0.35, 0.0, 0.3), phase="a")
    assert np.any(adapter.residual)
    command = (
        ServoCommand(-0.35, 0.0, -0.3)
        if change == "reverse"
        else ServoCommand(0.0, 0.0, 0.0)
    )
    phase = "b" if change == "phase" else "a"
    if change == "phase":
        command = ServoCommand(0.35, 0.0, 0.3)
    np.testing.assert_array_equal(adapter.step(command, phase=phase), np.zeros(3))
    np.testing.assert_allclose(adapter.residual, np.asarray(command.velocity) * 0.05)


def test_waypoint_cap_and_standard_action_capabilities_are_checked():
    adapter = ResidualActionAdapter(ActionLimits(), dt_s=1.0)
    action = adapter.step(ServoCommand(2.0, 2.0, 4.0), phase="a")
    assert np.linalg.norm(action[:2]) == pytest.approx(0.25)
    assert action[2] == pytest.approx(math.radians(30))
    with pytest.raises(ValueError):
        ResidualActionAdapter(replace(ActionLimits(), allow_lateral_movement=False))


def test_target_ids_survive_detector_renumbering_and_robot_motion():
    provider = SemanticTargetProvider(match_distance_m=0.15)
    first = targets(provider, observation(0.8))
    moved = observation(0.6, instance_ids=(9, 4), gps=(0.2, 0.0))
    second = targets(provider, moved)
    assert first.target.track_id == second.target.track_id == 1
    assert first.yaw_reference.track_id == second.yaw_reference.track_id == 2
    np.testing.assert_allclose(
        first.target_position_start, second.target_position_start, atol=0.02
    )
    with pytest.raises(MissingTarget):
        targets(provider, observation(1.4, instance_ids=(9, 4), gps=(0.2, 0.0)))


def test_same_class_instances_are_not_merged_and_gt_coordinates_are_never_needed():
    obs = observation()
    obs.semantic[40:90, 10:50] = 1
    obs.instance[40:90, 10:50] = 7
    obs.depth[40:90, 10:50] = 1.2
    result = targets(SemanticTargetProvider(), obs)
    assert result.target.bbox_xyxy == (100.0, 40.0, 140.0, 90.0)
    assert np.count_nonzero(result.target.mask) == 2000


@pytest.mark.parametrize("value", [None, [], [1, 2], float("nan"), "invalid", 1.5])
def test_missing_task_category_is_a_tracking_failure(value):
    obs = observation()
    obs.task_observations["object_goal"] = value
    with pytest.raises(MissingTarget):
        targets(SemanticTargetProvider(), obs)


def test_raw_integer_and_metric_geometry_and_controller_replay_match():
    config = BasePoseConfig()
    native, adapted = VisualServoController(
        **config.controller_kwargs()
    ), VisualServoController(**config.controller_kwargs())
    native.reset(0.0, initial_phase=ServoPhase.FORWARD_APPROACH)
    adapted.reset(0.0, initial_phase=ServoPhase.FORWARD_APPROACH)
    provider = SemanticTargetProvider()
    for frame, distance in enumerate([1.2] * 8 + [0.95] * 8 + [0.65] * 80):
        obs = observation(distance)
        # Exact millimetre values avoid input quantization differences here.
        obs.depth = np.full(obs.depth.shape, distance, dtype=np.float64)
        snap, metric, start = adapt_observation(
            obs, timestamp=frame * 0.05, use_opencv_camera_pose=True
        )
        selection = provider.get_alignment_targets(obs, "pick", snap, metric, start)
        raw = replace(
            snap,
            depth_raw=np.rint(snap.depth_raw * 1000).astype(np.uint16),
            depth_scale_m=0.001,
        )
        raw_cal = RawServoCalibration(
            width=240,
            height=160,
            fx=200,
            fy=200,
            cx=120,
            cy=80,
            camera_pitch_deg=0.0,
            camera_forward_offset_m=0.0,
            camera_lateral_offset_m=0.0,
        )
        old = _observation(raw, selection.target, selection.yaw_reference, raw_cal)
        new = _observation(snap, selection.target, selection.yaw_reference, metric)
        assert old.target.forward_m == pytest.approx(new.target.forward_m, abs=1e-10)
        assert old.target.right_m == pytest.approx(new.target.right_m, abs=1e-10)
        assert (
            old.yaw_align_geometry.yaw_error_rad == new.yaw_align_geometry.yaw_error_rad
        )
        orientation = {
            "actual_heading_rad": 0.0,
            "heading_setpoint_rad": 0.0,
            "state_age_s": 0.0,
            "telemetry_age_s": 0.0,
        }
        a = native.update(old, now=frame * 0.05, orientation=orientation)
        b = adapted.update(new, now=frame * 0.05, orientation=orientation)
        assert a == b
        assert native.phase == adapted.phase
    assert adapted.terminal_reason == native.terminal_reason == "aligned"


def test_session_fresh_frames_reach_ready_and_duplicates_do_not_advance():
    session = BasePoseSession(
        BasePoseConfig(), ActionLimits(), use_opencv_camera_pose=True
    )
    obs = observation()
    first = session.step(obs, frame_id=0)
    duplicate = session.step(obs, frame_id=0)
    assert duplicate.diagnostics["duplicate_frame"]
    assert session.updates == 1
    np.testing.assert_array_equal(duplicate.xyt, np.zeros(3))
    for frame in range(1, 100):
        result = session.step(obs, frame_id=frame)
        if result.terminate:
            break
    assert result.status == AlignmentStatus.READY
    assert result.diagnostics["post_stop_valid_frames"] == 30


def test_no_edges_and_missing_target_never_become_ready():
    for missing in (False, True):
        session = BasePoseSession(
            BasePoseConfig(max_steps=40), ActionLimits(), use_opencv_camera_pose=True
        )
        obs = observation(edge=False)
        if missing:
            obs.semantic[:] = 0
        for frame in range(41):
            result = session.step(obs, frame_id=frame)
            if result.terminate:
                break
        assert result.status in (AlignmentStatus.TIMEOUT, AlignmentStatus.LOST_TARGET)
        assert not np.any(result.xyt)


def test_invalid_post_stop_samples_cannot_report_ready():
    session = BasePoseSession(
        BasePoseConfig(), ActionLimits(), use_opencv_camera_pose=True
    )
    obs = observation()
    for frame in range(100):
        if session.controller.phase == ServoPhase.POST_STOP_SAMPLING:
            obs.semantic[:] = 0
        result = session.step(obs, frame_id=frame)
        if result.terminate:
            break
    assert result.status == AlignmentStatus.LOST_TARGET


def test_collision_does_not_turn_command_distance_into_actual_travel():
    session = BasePoseSession(
        BasePoseConfig(max_steps=10), ActionLimits(), use_opencv_camera_pose=True
    )
    obs = observation(1.2)
    actions = [session.step(obs, frame_id=i) for i in range(11)]
    assert any(np.any(result.xyt) for result in actions)
    assert actions[-1].diagnostics["actual_travel_m"] == 0.0
    assert actions[-1].status == AlignmentStatus.TIMEOUT
