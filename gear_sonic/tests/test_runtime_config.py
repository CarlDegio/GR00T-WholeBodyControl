from __future__ import annotations

import json

import pytest

from gear_sonic.runtime.profile import (
    ENDPOINT_SCHEMES,
    default_runtime_profile_path,
    load_runtime_profile,
)
from gear_sonic.utils.inference.lavira.service import load_lavira_config
from gear_sonic.utils.inference.navdp.control import XNAVDP_G1_MPC_DEFAULTS
from gear_sonic.utils.inference.navdp.gateway import load_navdp_planner_config
from gear_sonic.utils.planner_control.executor_service import (
    load_planner_velocity_executor_config,
)
from gear_sonic.utils.inference.lavira.depth_service import load_depth_anything_config
from gear_sonic.utils.inference.vla.service import load_inference_config
from gear_sonic.utils.inference.base_pose.agent import load_base_pose_config


def test_default_profile_reproduces_current_topology_and_timing() -> None:
    profile = load_runtime_profile()

    assert profile.name == "agent_full_current"
    assert set(profile.endpoints) == set(ENDPOINT_SCHEMES)
    assert len({(address.host, address.port) for address in profile.endpoints.values()}) == len(
        profile.endpoints
    )
    assert profile.ros_topics == {
        "lidar": "/livox/lidar",
        "lidar_imu": "/livox/imu",
        "odometry": "/Odometry_loc",
        "registered_cloud": "/cloud_registered_1",
    }
    assert profile.component("vla") == {
        "embodiment_tag": "unitree_g1_sonic",
        "prompt": (
            "Move in front of the table, grasp the medicine bottle, and place it "
            "into the blue basket."
        ),
        "inference_hz": 2.0,
        "action_publish_rate": 50,
        "action_horizon": 50,
        "sensor_gateway_poll_hz": 50.0,
        "sensor_gateway_request_timeout_ms": 100,
        "sensor_gateway_max_age_ms": 1000.0,
        "sensor_gateway_max_skew_ms": 5.0,
    }
    assert profile.component("navdp")["control_hz"] == 20.0
    assert profile.component("navdp")["mpc_hz"] == 10.0
    assert profile.component("depth_anything")["inference_hz"] == 10.5
    assert profile.component("data_exporter") == {
        "frequency_hz": 50,
        "sensor_gateway_poll_hz": 50.0,
        "sensor_gateway_request_timeout_ms": 100,
        "sensor_gateway_max_age_ms": 1000.0,
        "sensor_gateway_max_skew_ms": 5.0,
        "record_chest_camera": False,
        "task_prompt": "",
        "dataset_name": "",
    }


def test_profile_parameters_match_current_process_defaults() -> None:
    profile = load_runtime_profile()
    lavira = load_lavira_config()
    navdp = load_navdp_planner_config()
    executor = load_planner_velocity_executor_config()
    vla = load_inference_config()
    base_pose = load_base_pose_config()
    depth_anything = load_depth_anything_config()

    vla_profile = profile.component("vla")
    assert vla_profile["sensor_gateway_poll_hz"] == vla.sensor_gateway_poll_hz
    assert (
        vla_profile["sensor_gateway_request_timeout_ms"]
        == vla.sensor_gateway_request_timeout_ms
    )
    assert vla_profile["sensor_gateway_max_age_ms"] == vla.sensor_gateway_max_age_ms
    assert (
        vla_profile["sensor_gateway_max_skew_ms"]
        == vla.sensor_gateway_max_skew_ms
    )

    assert profile.component("lavira")["camera_timeout_ms"] == lavira.camera_timeout_ms
    assert (
        profile.component("lavira")["qwenvl_timeout_seconds"]
        == lavira.qwenvl_timeout_seconds
    )
    assert profile.component("lavira")["min_confidence"] == lavira.min_confidence
    assert profile.component("base_pose")["task"] == base_pose.task
    assert (
        profile.component("base_pose")["raw_head_target_distance_m"]
        == base_pose.raw_head_target_distance_m
    )

    navdp_profile = profile.component("navdp")
    assert navdp_profile["control_hz"] == navdp.control_hz
    assert navdp_profile["mpc_hz"] == navdp.mpc_hz
    assert navdp_profile["mpc_result_timeout_s"] == navdp.mpc_result_timeout_s
    assert navdp_profile["heading_preview_s"] == navdp.heading_preview_s
    assert navdp_profile["goal_tolerance_m"] == navdp.goal_tolerance_m
    assert navdp_profile["stop_threshold"] == navdp.stop_threshold
    assert navdp_profile["request_timeout_s"] == navdp.request_timeout_s
    assert navdp_profile["sensor_gateway_poll_hz"] == navdp.sensor_gateway_poll_hz
    assert (
        navdp_profile["sensor_gateway_request_timeout_ms"]
        == navdp.sensor_gateway_request_timeout_ms
    )
    assert navdp_profile["sensor_gateway_max_age_ms"] == navdp.sensor_gateway_max_age_ms
    assert (
        navdp_profile["sensor_gateway_max_skew_ms"]
        == navdp.sensor_gateway_max_skew_ms
    )
    assert navdp_profile["odometry_timeout_s"] == navdp.odometry_timeout_s
    assert navdp_profile["trajectory_timeout_s"] == navdp.trajectory_timeout_s
    assert profile.component("xnavdp_mpc") == XNAVDP_G1_MPC_DEFAULTS

    executor_profile = profile.component("planner_executor")
    assert executor_profile["control_hz"] == executor.control_hz
    assert executor_profile["radar_timeout_s"] == executor.radar_timeout_s
    assert (
        executor_profile["manual_velocity_timeout_s"]
        == executor.manual_velocity_timeout_s
    )
    assert (
        executor_profile["navdp_velocity_timeout_s"]
        == executor.navdp_velocity_timeout_s
    )
    depth_profile = profile.component("depth_anything")
    assert depth_profile["root"] == depth_anything.root
    assert depth_profile["checkpoint"] == depth_anything.checkpoint
    assert depth_profile["encoder"] == depth_anything.encoder
    assert depth_profile["device"] == depth_anything.device
    assert depth_profile["inference_hz"] == depth_anything.inference_hz
    assert depth_profile["input_size"] == depth_anything.input_size
    assert depth_profile["model_max_depth_m"] == depth_anything.model_max_depth_m
    assert depth_profile["publish_max_depth_m"] == depth_anything.publish_max_depth_m
    assert depth_profile["use_amp"] is depth_anything.use_amp


def test_partial_overlay_changes_only_selected_values(tmp_path) -> None:
    overlay = tmp_path / "robot_lab.json"
    overlay.write_text(
        json.dumps(
            {
                "endpoints": {"camera_server": {"host": "192.168.123.164"}},
                "components": {
                    "vla": {"prompt": "pick up the paper ball"},
                    "lavira": {"min_confidence": 0.75},
                    "base_pose": {"raw_head_target_distance_m": 1.1},
                    "launcher": {"data_exporter": False},
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_runtime_profile(overlays=[overlay])

    assert profile.endpoint("camera_server").host == "192.168.123.164"
    assert profile.endpoint("camera_server").port == 5555
    assert profile.endpoint_uri("camera_server") == "tcp://192.168.123.164:5555"
    assert profile.endpoint_uri("xnavdp_http") == "http://127.0.0.1:19999"
    assert profile.component("vla")["prompt"] == "pick up the paper ball"
    assert profile.component("vla")["action_publish_rate"] == 50
    assert profile.component("launcher")["data_exporter"] is False
    assert load_inference_config(overlays=(str(overlay),)).prompt == "pick up the paper ball"
    assert load_lavira_config(overlays=(str(overlay),)).min_confidence == 0.75
    assert (
        load_base_pose_config(overlays=(str(overlay),)).raw_head_target_distance_m
        == 1.1
    )


def test_profile_rejects_unknown_endpoint_and_port_collisions(tmp_path) -> None:
    unknown = tmp_path / "unknown.json"
    unknown.write_text(
        json.dumps({"endpoints": {"mystery_socket": {"host": "localhost", "port": 6000}}}),
        encoding="utf-8",
    )
    collision = tmp_path / "collision.json"
    collision.write_text(
        json.dumps(
            {
                "endpoints": {
                    "camera_server": {
                        "host": "127.0.0.1",
                        "port": load_runtime_profile().endpoint("policy_server").port,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown endpoint"):
        load_runtime_profile(overlays=[unknown])
    with pytest.raises(ValueError, match="shared by"):
        load_runtime_profile(overlays=[collision])


def test_profile_rejects_unknown_top_level_fields(tmp_path) -> None:
    overlay = tmp_path / "typo.json"
    overlay.write_text(json.dumps({"component": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown runtime profile fields"):
        load_runtime_profile(overlays=[overlay])
