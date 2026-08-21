"""
Replay Sonic VLA datasets through the deploy-side `--input-type zmq` interface.

This script intentionally fixes the protocol at startup:
  - `--protocol v1` replays `action.wbc` as streamed joint chunks.
  - `--protocol v4` replays `action.motion_token` as token-by-token actions.

Run the C++ deploy with `--input-type zmq`, then manually press `ENTER` in the
deploy terminal to enable ZMQ streaming before starting playback.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty
from typing import Literal

import numpy as np
import pandas as pd
import zmq

from gear_sonic.utils.data_collection.state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    pack_pose_message,
    pack_pose_v1_message,
)


BODY_JOINT_DIM = 29
LEFT_HAND_DIM = 7
RIGHT_HAND_DIM = 7
DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

# Deploy-side Protocol v1 expects body joints in this IsaacLab order.
G1_BODY_JOINT_NAMES_ISAACLAB = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# Body joint order used by deploy-side outputs such as `body_q_target/body_q_measured`
# and by `action.wbc` body-actuated joints before converting to IsaacLab order.
G1_BODY_JOINT_NAMES_MUJOCO = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# `policy_parameters.hpp -> mujoco_to_isaaclab`
MUJOCO_TO_ISAACLAB_BODY_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]

# Deploy-side Dex3 hand order. This is the order consumed by
# InputInterface::GetHandPose() and sent to dex3_hands_.setAllJointsCommand().
G1_LEFT_HAND_JOINT_NAMES_DEPLOY = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
]

G1_RIGHT_HAND_JOINT_NAMES_DEPLOY = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

# Recorded `action.wbc` full-43DOF order from Sonic VLA dataset / RobotModel joint_names:
# [29D body in MuJoCo/body-actuated order, with left-hand 7D inserted before right arm,
#  then right-hand 7D at the tail].
ACTION_WBC_FALLBACK_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]


@dataclass
class ReplayConfig:
    dataset_path: str
    protocol: Literal["v1", "v4"] = "v1"
    episode_index: int = 0
    start_frame: int = 0
    end_frame: int | None = None
    fps: float | None = None
    loop_episode: bool = False
    host: str = "*"
    port: int = 5556
    topic: str = "pose"
    publisher_warmup_sec: float = 0.25
    chunk_size: int = 5
    stride: int = 1
    catch_up: bool = False
    state_feedback: bool = False
    state_zmq_host: str = "localhost"
    state_zmq_port: int = 5557
    state_zmq_topic: str = "g1_debug"
    state_feedback_timeout_sec: float = 0.02
    speed_scale: float = 1.0
    log_every: int = 25
    packet_debug: bool = False
    packet_debug_every: int = 1
    packet_debug_dims: int = 5
    packet_debug_dir: str = "logs/replay_debug"


def _coerce_vector(value, dtype: np.dtype | type = np.float64) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value.astype(dtype, copy=False)
    else:
        arr = np.asarray(value, dtype=dtype)
    return arr.reshape(-1)


def _stack_vector_column(
    frame_table: pd.DataFrame,
    column: str,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    if column not in frame_table.columns:
        raise KeyError(f"Dataset column '{column}' not found")
    stacked = [_coerce_vector(value, dtype=dtype) for value in frame_table[column]]
    return np.stack(stacked, axis=0)


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _get_parquet_path(dataset_path: Path, info: dict, episode_index: int) -> Path:
    data_path_pattern = info.get("data_path", DEFAULT_DATA_PATH)
    chunks_size = info.get("chunks_size", 1000)
    episode_chunk = episode_index // chunks_size
    return dataset_path / data_path_pattern.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )


def _get_feature_names(info: dict, feature_key: str) -> list[str] | None:
    feature = info.get("features", {}).get(feature_key, {})
    names = feature.get("names")
    if not isinstance(names, list):
        return None
    if not all(isinstance(name, str) for name in names):
        return None
    return names


def _indices_from_names(
    source_names: list[str],
    target_names: list[str],
    *,
    feature_key: str,
) -> list[int]:
    source_index = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_index]
    if missing:
        missing_preview = ", ".join(missing[:8])
        raise ValueError(
            f"{feature_key} metadata is missing required joints: {missing_preview}"
        )
    return [source_index[name] for name in target_names]


def _split_joint_configuration(
    joint_values: np.ndarray,
    joint_names: list[str] | None,
    *,
    feature_key: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    source_names = joint_names
    source_label = "metadata"
    if source_names is None and joint_values.shape[1] == len(ACTION_WBC_FALLBACK_NAMES):
        source_names = ACTION_WBC_FALLBACK_NAMES
        source_label = "built-in Sonic VLA 43DOF fallback"

    if source_names is not None:
        if len(source_names) != joint_values.shape[1]:
            raise ValueError(
                f"{feature_key} {source_label} has {len(source_names)} names but data has "
                f"{joint_values.shape[1]} dims"
            )
        body_mujoco_indices = _indices_from_names(
            source_names,
            G1_BODY_JOINT_NAMES_MUJOCO,
            feature_key=feature_key,
        )
        body_effective_indices = [body_mujoco_indices[index] for index in MUJOCO_TO_ISAACLAB_BODY_DOF]
        body_action_mujoco = joint_values[:, body_mujoco_indices]
        body_action_isaaclab = body_action_mujoco[:, MUJOCO_TO_ISAACLAB_BODY_DOF]
        left_hand_indices = _indices_from_names(
            source_names,
            G1_LEFT_HAND_JOINT_NAMES_DEPLOY,
            feature_key=feature_key,
        )
        right_hand_indices = _indices_from_names(
            source_names,
            G1_RIGHT_HAND_JOINT_NAMES_DEPLOY,
            feature_key=feature_key,
        )
        print(
            f"[Dataset] {feature_key} remap "
            f"({source_label}): body_mujoco_indices={body_mujoco_indices}, "
            "subtract_default_angles=False, "
            f"mujoco_to_isaaclab={MUJOCO_TO_ISAACLAB_BODY_DOF}, "
            f"body_effective_indices={body_effective_indices}, "
            f"left hand indices={left_hand_indices}, right hand indices={right_hand_indices}"
        )
        return (
            body_action_isaaclab,
            joint_values[:, left_hand_indices],
            joint_values[:, right_hand_indices],
        )

    if joint_values.shape[1] != BODY_JOINT_DIM:
        raise ValueError(
            f"{feature_key} has {joint_values.shape[1]} dims but no usable joint names. "
            f"Expected {BODY_JOINT_DIM} dims for an already-remapped body-only action "
            f"or {len(ACTION_WBC_FALLBACK_NAMES)} dims for Sonic VLA full-q fallback."
        )

    print(f"[Dataset] {feature_key} has no joint names; assuming body-only IsaacLab 29DOF.")
    return joint_values, None, None


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    velocities = np.zeros_like(values, dtype=np.float64)
    if len(values) <= 1:
        return velocities
    velocities[1:] = (values[1:] - values[:-1]) * fps
    velocities[0] = velocities[1]
    return velocities


@dataclass
class ReplayAction:
    protocol: Literal["v1", "v4"]
    payload: dict[str, np.ndarray]
    dataset_start: int
    dataset_stop: int
    session_frame: int
    stride: int


@dataclass
class EpisodeData:
    episode_index: int
    joint_source: np.ndarray
    body_action: np.ndarray
    left_hand_action: np.ndarray | None
    right_hand_action: np.ndarray | None
    body_velocity: np.ndarray
    root_orientation: np.ndarray
    motion_token: np.ndarray
    fps: float
    hand_action_source: str

    @property
    def length(self) -> int:
        return int(self.body_action.shape[0])


class DatasetEpisodeReader:
    def __init__(self, dataset_path: str, fps_override: float | None):
        self.dataset_path = Path(dataset_path)
        self.info = _read_json(self.dataset_path / "meta" / "info.json")
        self.episodes = _read_jsonl(self.dataset_path / "meta" / "episodes.jsonl")
        self.fps = float(fps_override if fps_override is not None else self.info.get("fps", 50))

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def load_episode(self, episode_index: int) -> EpisodeData:
        if not self.episodes:
            raise ValueError(f"No episodes found under {self.dataset_path}")
        episode_index = episode_index % len(self.episodes)
        parquet_path = _get_parquet_path(self.dataset_path, self.info, episode_index)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")

        frame_table = pd.read_parquet(parquet_path)
        joint_source = _stack_vector_column(frame_table, "observation.state", dtype=np.float64)
        root_orientation = _stack_vector_column(
            frame_table, "observation.root_orientation", dtype=np.float64
        )
        motion_token = _stack_vector_column(frame_table, "action.motion_token", dtype=np.float64)

        if root_orientation.shape[1] != 4:
            raise ValueError(
                f"observation.root_orientation has shape {root_orientation.shape}, expected [T, 4]"
            )

        source_names = _get_feature_names(self.info, "observation.state")
        body_action, left_hand_action_obs, right_hand_action_obs = _split_joint_configuration(
            joint_source,
            source_names,
            feature_key="observation.state",
        )
        if source_names is not None:
            print(
                "[Dataset] observation.state remapped by joint names: "
                f"{joint_source.shape[1]} dims -> body {body_action.shape[1]}, "
                f"left hand {0 if left_hand_action_obs is None else left_hand_action_obs.shape[1]}, "
                f"right hand {0 if right_hand_action_obs is None else right_hand_action_obs.shape[1]}."
            )

        left_hand_action = left_hand_action_obs
        right_hand_action = right_hand_action_obs
        hand_action_source = "observation.state"
        if "teleop.left_hand_joints" in frame_table.columns and "teleop.right_hand_joints" in frame_table.columns:
            left_hand_action = _stack_vector_column(
                frame_table, "teleop.left_hand_joints", dtype=np.float64
            )
            right_hand_action = _stack_vector_column(
                frame_table, "teleop.right_hand_joints", dtype=np.float64
            )
            hand_action_source = "teleop.left/right_hand_joints"
            print("[Dataset] Using `teleop.left_hand_joints/right_hand_joints` for replay hand commands.")
        else:
            print("[Dataset] `teleop.left/right_hand_joints` missing; falling back to hand data from `observation.state`.")

        body_velocity = _finite_difference(body_action, self.fps)

        return EpisodeData(
            episode_index=episode_index,
            joint_source=joint_source,
            body_action=body_action,
            left_hand_action=left_hand_action,
            right_hand_action=right_hand_action,
            body_velocity=body_velocity,
            root_orientation=root_orientation,
            motion_token=motion_token,
            fps=self.fps,
            hand_action_source=hand_action_source,
        )


class ReplayPolicyBase:
    protocol: Literal["v1", "v4"]

    def __init__(
        self,
        episode: EpisodeData,
        *,
        start_frame: int,
        end_frame: int | None,
        session_frame_start: int = 0,
    ):
        self.episode = episode
        self.start_frame = max(0, start_frame)
        self.end_frame = (
            min(episode.length, end_frame) if end_frame is not None else episode.length
        )
        self.cursor = self.start_frame
        self.session_frame = max(0, session_frame_start)
        if self.start_frame >= self.end_frame:
            raise ValueError(
                f"Invalid frame range [{self.start_frame}, {self.end_frame}) for episode length {episode.length}"
            )

    def reset(self, *, keep_session_frame: bool = True) -> None:
        self.cursor = self.start_frame
        if not keep_session_frame:
            self.session_frame = 0

    def is_finished(self) -> bool:
        return self.cursor >= self.end_frame

    def act(self, observation: dict | None = None) -> ReplayAction | None:
        raise NotImplementedError


class WBCReplayPolicyV1(ReplayPolicyBase):
    protocol: Literal["v1"] = "v1"

    def __init__(
        self,
        episode: EpisodeData,
        *,
        start_frame: int,
        end_frame: int | None,
        chunk_size: int,
        stride: int,
        catch_up: bool,
        session_frame_start: int = 0,
    ):
        super().__init__(
            episode,
            start_frame=start_frame,
            end_frame=end_frame,
            session_frame_start=session_frame_start,
        )
        self.chunk_size = max(1, chunk_size)
        self.stride = max(1, stride)
        self.catch_up = catch_up

    def act(self, observation: dict | None = None) -> ReplayAction | None:
        if self.is_finished():
            return None

        start = self.cursor
        stop = min(self.end_frame, start + self.chunk_size)
        chunk_len = stop - start
        frame_index = np.arange(
            self.session_frame,
            self.session_frame + chunk_len,
            dtype=np.int64,
        )
        payload = {
            "joint_pos": self.episode.body_action[start:stop].astype(np.float32, copy=False),
            "joint_vel": self.episode.body_velocity[start:stop].astype(np.float32, copy=False),
            "body_quat_w": self.episode.root_orientation[start:stop].astype(
                np.float32, copy=False
            ),
            "frame_index": frame_index,
            "catch_up": np.asarray([self.catch_up], dtype=np.uint8),
        }
        if self.episode.left_hand_action is not None:
            payload["left_hand_joints"] = self.episode.left_hand_action[stop - 1].astype(
                np.float32, copy=False
            )
        if self.episode.right_hand_action is not None:
            payload["right_hand_joints"] = self.episode.right_hand_action[stop - 1].astype(
                np.float32, copy=False
            )
        action = ReplayAction(
            protocol="v1",
            payload=payload,
            dataset_start=start,
            dataset_stop=stop,
            session_frame=self.session_frame,
            stride=self.stride,
        )
        self.cursor += self.stride
        self.session_frame += self.stride
        return action


class TokenReplayPolicyV4(ReplayPolicyBase):
    protocol: Literal["v4"] = "v4"

    def __init__(
        self,
        episode: EpisodeData,
        *,
        start_frame: int,
        end_frame: int | None,
        session_frame_start: int = 0,
    ):
        super().__init__(
            episode,
            start_frame=start_frame,
            end_frame=end_frame,
            session_frame_start=session_frame_start,
        )

    def act(self, observation: dict | None = None) -> ReplayAction | None:
        if self.is_finished():
            return None

        index = self.cursor
        payload = {
            "token_state": self.episode.motion_token[index].astype(np.float32, copy=False),
            "body_quat_w": self.episode.root_orientation[index].astype(
                np.float32, copy=False
            ),
            "frame_index": np.asarray([self.session_frame], dtype=np.int64),
        }
        if self.episode.left_hand_action is not None:
            payload["left_hand_joints"] = self.episode.left_hand_action[index].astype(
                np.float32, copy=False
            )
        if self.episode.right_hand_action is not None:
            payload["right_hand_joints"] = self.episode.right_hand_action[index].astype(
                np.float32, copy=False
            )

        action = ReplayAction(
            protocol="v4",
            payload=payload,
            dataset_start=index,
            dataset_stop=index + 1,
            session_frame=self.session_frame,
            stride=1,
        )
        self.cursor += 1
        self.session_frame += 1
        return action


class ZMQReplayEnv:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        topic: str,
        publisher_warmup_sec: float,
        state_feedback: bool,
        state_zmq_host: str,
        state_zmq_port: int,
        state_zmq_topic: str,
        state_feedback_timeout_sec: float,
    ):
        self.topic = topic
        self._state_feedback_timeout_sec = max(0.0, state_feedback_timeout_sec)
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.bind(f"tcp://{host}:{port}")
        self._state_subscriber = (
            ZMQStateSubscriber(
                host=state_zmq_host,
                port=state_zmq_port,
                topic=state_zmq_topic,
            )
            if state_feedback
            else None
        )
        print(f"[ReplayEnv] Publishing on tcp://{host}:{port} (topic: {topic})")
        time.sleep(max(0.0, publisher_warmup_sec))

    def step(self, action: ReplayAction) -> dict | None:
        if action.protocol == "v1":
            message = pack_pose_v1_message(
                action.payload["joint_pos"],
                action.payload["joint_vel"],
                action.payload["body_quat_w"],
                action.payload["frame_index"],
                catch_up=bool(action.payload["catch_up"][0]),
                left_hand_joints=action.payload.get("left_hand_joints"),
                right_hand_joints=action.payload.get("right_hand_joints"),
                topic=self.topic,
            )
        elif action.protocol == "v4":
            message = pack_pose_message(action.payload, topic=self.topic, version=4)
        else:
            raise ValueError(f"Unsupported protocol: {action.protocol}")

        self._socket.send(message)

        if self._state_subscriber is None:
            return None

        deadline = time.monotonic() + self._state_feedback_timeout_sec
        latest = self._state_subscriber.get_msg(clear=True)
        while latest is None and time.monotonic() < deadline:
            time.sleep(0.001)
            latest = self._state_subscriber.get_msg(clear=True)
        return latest

    def close(self) -> None:
        if self._state_subscriber is not None:
            self._state_subscriber.close()
        self._socket.close()
        self._ctx.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TerminalController:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_settings = None

    def __enter__(self):
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def poll_key(self) -> str | None:
        if self._fd is None:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return os.read(self._fd, 1).decode(errors="ignore")


def _build_policy(
    config: ReplayConfig,
    episode: EpisodeData,
    *,
    session_frame_start: int,
) -> ReplayPolicyBase:
    if config.protocol == "v1":
        return WBCReplayPolicyV1(
            episode,
            start_frame=config.start_frame,
            end_frame=config.end_frame,
            chunk_size=config.chunk_size,
            stride=config.stride,
            catch_up=config.catch_up,
            session_frame_start=session_frame_start,
        )
    if config.protocol == "v4":
        return TokenReplayPolicyV4(
            episode,
            start_frame=config.start_frame,
            end_frame=config.end_frame,
            session_frame_start=session_frame_start,
        )
    raise ValueError(f"Unsupported protocol: {config.protocol}")


def _print_controls() -> None:
    print("[Replay] Controls: space=pause/resume, r=restart, [=prev episode, ]=next episode, -=slower, +=faster, h=help, q=quit")


def _format_range(action: ReplayAction) -> str:
    return f"dataset[{action.dataset_start}:{action.dataset_stop}) session={action.session_frame}"


def _format_full_array(name: str, value: np.ndarray | None) -> str:
    if value is None:
        return f"{name}=None"
    arr = np.asarray(value)
    return (
        f"{name} shape={arr.shape}\n"
        f"{np.array2string(arr, precision=6, suppress_small=False, threshold=arr.size + 1)}"
    )


class PacketDebugLogger:
    def __init__(self, log_dir: str, protocol: str):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"replay_{protocol}_{timestamp}.log"
        self._file = open(self.path, "a", encoding="utf-8")

    def log(
        self,
        action: ReplayAction,
        observation: dict | None,
        *,
        sent_count: int,
        state_feedback_enabled: bool,
        state_feedback_timeout_sec: float,
    ) -> None:
        window_len = action.dataset_stop - action.dataset_start
        overlap = max(0, window_len - action.stride)
        self._file.write("=" * 100 + "\n")
        self._file.write(
            f"send#{sent_count} protocol={action.protocol} {_format_range(action)} "
            f"window={window_len} stride={action.stride} overlap={overlap}\n"
        )
        self._file.write(
            _format_full_array("joint_pos", action.payload.get("joint_pos")) + "\n"
        )
        self._file.write(
            _format_full_array("frame_index", action.payload.get("frame_index")) + "\n"
        )

        if observation is None:
            if state_feedback_enabled:
                self._file.write(
                    "g1_debug=None "
                    f"(no message within {state_feedback_timeout_sec:.3f}s)\n"
                )
            else:
                self._file.write("g1_debug=None (state feedback disabled)\n")
            self._file.flush()
            return

        self._file.write(f"g1_debug_keys={len(observation)}\n")
        for key in ("last_action", "body_q_target"):
            self._file.write(_format_full_array(key, observation.get(key)) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def main(config: ReplayConfig) -> None:
    if config.protocol == "v4":
        config.chunk_size = 1
        config.stride = 1

    dataset_reader = DatasetEpisodeReader(config.dataset_path, config.fps)
    current_episode = dataset_reader.load_episode(config.episode_index)
    policy = _build_policy(config, current_episode, session_frame_start=0)
    env = ZMQReplayEnv(
        host=config.host,
        port=config.port,
        topic=config.topic,
        publisher_warmup_sec=config.publisher_warmup_sec,
        state_feedback=config.state_feedback,
        state_zmq_host=config.state_zmq_host,
        state_zmq_port=config.state_zmq_port,
        state_zmq_topic=config.state_zmq_topic,
        state_feedback_timeout_sec=config.state_feedback_timeout_sec,
    )

    paused = False
    should_quit = False
    observation = None
    sent_count = 0
    packet_debug_logger: PacketDebugLogger | None = None
    speed_scale = max(1e-3, config.speed_scale)
    period_sec = 1.0 / current_episode.fps

    print(
        f"[Replay] Loaded episode {current_episode.episode_index}/{dataset_reader.num_episodes - 1}, "
        f"frames={current_episode.length}, protocol={config.protocol}, fps={current_episode.fps:.2f}"
    )
    if config.protocol == "v1":
        overlap = max(0, config.chunk_size - config.stride)
        print(
            f"[Replay] V1 window config: chunk_size={config.chunk_size}, "
            f"stride={config.stride}, overlap={overlap}"
        )
    if config.packet_debug:
        packet_debug_logger = PacketDebugLogger(config.packet_debug_dir, config.protocol)
        print(f"[ReplayDebug] Writing detailed packet logs to {packet_debug_logger.path}")
    _print_controls()
    print("[Replay] Start deploy with `--input-type zmq` and press ENTER there to enable streaming.")

    try:
        with TerminalController() as terminal:
            next_tick = time.monotonic()
            while not should_quit:
                key = terminal.poll_key()
                if key == " ":
                    paused = not paused
                    print(f"[Replay] {'Paused' if paused else 'Resumed'}")
                elif key == "r":
                    policy = _build_policy(
                        config,
                        current_episode,
                        session_frame_start=policy.session_frame,
                    )
                    paused = False
                    print(f"[Replay] Restarted episode {current_episode.episode_index}")
                elif key == "[":
                    new_index = (current_episode.episode_index - 1) % dataset_reader.num_episodes
                    current_episode = dataset_reader.load_episode(new_index)
                    policy = _build_policy(
                        config,
                        current_episode,
                        session_frame_start=policy.session_frame,
                    )
                    period_sec = 1.0 / current_episode.fps
                    paused = False
                    print(f"[Replay] Switched to episode {current_episode.episode_index}")
                elif key == "]":
                    new_index = (current_episode.episode_index + 1) % dataset_reader.num_episodes
                    current_episode = dataset_reader.load_episode(new_index)
                    policy = _build_policy(
                        config,
                        current_episode,
                        session_frame_start=policy.session_frame,
                    )
                    period_sec = 1.0 / current_episode.fps
                    paused = False
                    print(f"[Replay] Switched to episode {current_episode.episode_index}")
                elif key == "-":
                    speed_scale = max(0.1, speed_scale / 1.25)
                    print(f"[Replay] speed_scale={speed_scale:.2f}")
                elif key in ("+", "="):
                    speed_scale = min(8.0, speed_scale * 1.25)
                    print(f"[Replay] speed_scale={speed_scale:.2f}")
                elif key in ("h", "H"):
                    _print_controls()
                elif key in ("q", "Q"):
                    should_quit = True
                    continue

                if paused:
                    time.sleep(0.01)
                    continue

                action = policy.act(observation)
                if action is None:
                    if config.loop_episode:
                        policy = _build_policy(
                            config,
                            current_episode,
                            session_frame_start=policy.session_frame,
                        )
                        continue
                    paused = True
                    print(
                        f"[Replay] Episode {current_episode.episode_index} finished. "
                        "Press `r` to replay or `q` to quit."
                    )
                    continue

                observation = env.step(action)
                sent_count += 1

                if config.packet_debug and (
                    sent_count == 1
                    or sent_count % max(1, config.packet_debug_every) == 0
                ):
                    assert packet_debug_logger is not None
                    packet_debug_logger.log(
                        action,
                        observation,
                        sent_count=sent_count,
                        state_feedback_enabled=config.state_feedback,
                        state_feedback_timeout_sec=config.state_feedback_timeout_sec,
                    )

                if sent_count == 1 or sent_count % max(1, config.log_every) == 0:
                    obs_suffix = ""
                    if observation is not None:
                        obs_suffix = f", obs_keys={len(observation)}"
                    print(f"[Replay] Sent {action.protocol} {_format_range(action)}{obs_suffix}")

                next_tick += period_sec / speed_scale
                sleep_sec = next_tick - time.monotonic()
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                else:
                    next_tick = time.monotonic()
    finally:
        if packet_debug_logger is not None:
            packet_debug_logger.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay Sonic VLA datasets through the deploy-side ZMQ interface."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--protocol", choices=("v1", "v4"), default="v1")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--loop-episode", action="store_true")
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--topic", default="pose")
    parser.add_argument("--publisher-warmup-sec", type=float, default=0.25)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--catch-up", action="store_true")
    parser.add_argument("--state-feedback", action="store_true")
    parser.add_argument("--state-zmq-host", default="localhost")
    parser.add_argument("--state-zmq-port", type=int, default=5557)
    parser.add_argument("--state-zmq-topic", default="g1_debug")
    parser.add_argument("--state-feedback-timeout-sec", type=float, default=0.02)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--packet-debug", action="store_true")
    parser.add_argument("--packet-debug-every", type=int, default=1)
    parser.add_argument("--packet-debug-dims", type=int, default=5)
    parser.add_argument("--packet-debug-dir", default="logs/replay_debug")
    args = parser.parse_args()
    main(
        ReplayConfig(
            dataset_path=args.dataset_path,
            protocol=args.protocol,
            episode_index=args.episode_index,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            fps=args.fps,
            loop_episode=args.loop_episode,
            host=args.host,
            port=args.port,
            topic=args.topic,
            publisher_warmup_sec=args.publisher_warmup_sec,
            chunk_size=args.chunk_size,
            stride=args.stride,
            catch_up=args.catch_up,
            state_feedback=args.state_feedback,
            state_zmq_host=args.state_zmq_host,
            state_zmq_port=args.state_zmq_port,
            state_zmq_topic=args.state_zmq_topic,
            state_feedback_timeout_sec=args.state_feedback_timeout_sec,
            speed_scale=args.speed_scale,
            log_every=args.log_every,
            packet_debug=args.packet_debug,
            packet_debug_every=args.packet_debug_every,
            packet_debug_dims=args.packet_debug_dims,
            packet_debug_dir=args.packet_debug_dir,
        )
    )
