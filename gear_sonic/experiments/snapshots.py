"""Archive the exact first policy observation without blocking action publication."""

import json
import queue
import threading
import time

import numpy as np

from gear_sonic.runtime.gateway.sensor_client import SensorGatewayClient
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest, TimestampBasis

from .recording import Recorder


def record_pose_async(profile, recorder, generation, phase, skill_id=0):
    if recorder.path is None:
        return
    anchor = time.time_ns()

    def capture():
        try:
            with SensorGatewayClient(
                profile.endpoint_uri("sensor_gateway_metadata"), request_timeout_ms=100
            ) as client:
                snap = client.read_snapshot(
                    SnapshotRequest(
                        streams=("ros/odometry",),
                        max_age_ms=1000.0,
                        max_skew_ms=100.0,
                        anchor_timestamp_ns=anchor,
                        timestamp_basis=TimestampBasis.SOURCE,
                    )
                )
                vector = np.asarray(snap.arrays["ros/odometry"]).reshape(-1)
                if vector.size != 13 or not np.isfinite(vector).all():
                    raise ValueError("Invalid odometry vector")
                frame = snap.snapshot.frames["ros/odometry"]
                recorder.write(
                    "pose",
                    generation=generation,
                    phase=phase,
                    skill_id=skill_id,
                    anchor_time_ns=anchor,
                    timestamp_ns=frame.source_timestamp_ns,
                    position_xyz=vector[:3].tolist(),
                    quaternion_xyzw=vector[3:7].tolist(),
                    coordinate_system="FAST-LIO odometry",
                    attributes=dict(frame.attributes),
                )
        except Exception as exc:
            recorder.write(
                "pose",
                generation=generation,
                phase=phase,
                skill_id=skill_id,
                anchor_time_ns=anchor,
                missing_reason=str(exc),
            )

    threading.Thread(target=capture, daemon=True, name="experiment-pose-snapshot").start()


class FirstObservationRecorder:
    def __init__(self, profile):
        self.profile = profile
        self.recorder = Recorder(profile)
        self.seen = set()
        self.lock = threading.Lock()
        self.pending = queue.Queue()
        self.thread = None
        if self.recorder.path is not None:
            self.thread = threading.Thread(target=self._worker, daemon=True, name="experiment-first-observation")
            self.thread.start()

    def capture(self, identity, camera, state):
        if self.thread is None or identity[0] < 0:
            return
        with self.lock:
            if identity in self.seen:
                return
            self.seen.add(identity)
        # camera/state have already been copied out of shared memory by VLA.
        self.recorder.write(
            "vla_first_inference",
            generation=identity[0],
            skill_id=identity[1],
            camera_timestamps=camera["timestamps"],
        )
        self.pending.put((identity, camera, state))

    def _save(self, client, identity, camera, state):
        generation, skill = identity
        directory = self.recorder.path.parent / "snapshots" / self.recorder.trial_id(generation) / f"vla_{skill}"
        directory.mkdir(parents=True, exist_ok=True)
        result = {
            "generation": generation,
            "skill_id": skill,
            "trial_id": self.recorder.trial_id(generation),
            "cameras": {},
            "mask_status": "pending_offline",
            "robot_orientation": {},
            "odometry": None,
        }
        for key in ("base_quat", "cpp_rotation_offset", "init_base_quat"):
            if key in state:
                result["robot_orientation"][key] = np.asarray(state[key]).tolist()
        for name, jpeg in camera["images"].items():
            path = directory / f"{name}.jpg"
            path.write_bytes(jpeg)
            record = {
                "rgb": path.name,
                "timestamp_s": camera["timestamps"][name],
                "camera_info": camera.get("camera_info", {}).get(name, {}),
                "depth": None,
            }
            result["cameras"][name] = record
            if name not in {"ego_view", "chest_view"}:
                continue
            bp = self.profile.component("base_pose")
            stream = bp.get(
                "dual_head_depth_stream" if name == "ego_view" else "dual_chest_depth_stream",
                f"camera/{name}_depth",
            )
            record["depth_stream"] = stream
            anchor = int(camera["timestamps"][name] * 1e9)
            try:
                snap = client.read_snapshot(
                    SnapshotRequest(
                        streams=(stream,),
                        max_age_ms=1000.0,
                        max_skew_ms=5.0,
                        anchor_timestamp_ns=anchor,
                        timestamp_basis=TimestampBasis.SOURCE,
                    )
                )
                frame = snap.snapshot.frames[stream]
                delta = abs(frame.source_timestamp_ns - anchor) / 1e6
                if delta > 5.0:
                    raise ValueError(f"Depth does not match first RGB: {delta:.3f} ms")
                depth = np.asarray(snap.arrays[stream])
                if depth.dtype != np.uint16 or depth.ndim != 2:
                    raise ValueError("Depth must be a native uint16 image")
                np.save(directory / f"{name}_depth.npy", depth, allow_pickle=False)
                record.update(
                    depth=f"{name}_depth.npy",
                    depth_timestamp_ns=frame.source_timestamp_ns,
                    depth_skew_ms=delta,
                    depth_attributes=dict(frame.attributes),
                )
            except Exception as exc:
                record["depth_missing_reason"] = str(exc)
        try:
            anchor = int(camera["timestamps"]["ego_view"] * 1e9)
            snap = client.read_snapshot(
                SnapshotRequest(
                    streams=("ros/odometry",),
                    max_age_ms=1000.0,
                    max_skew_ms=100.0,
                    anchor_timestamp_ns=anchor,
                    timestamp_basis=TimestampBasis.SOURCE,
                )
            )
            frame = snap.snapshot.frames["ros/odometry"]
            values = np.asarray(snap.arrays["ros/odometry"]).reshape(-1)
            if values.size != 13 or not np.isfinite(values).all():
                raise ValueError("Invalid odometry vector")
            result["odometry"] = dict(
                position_xyz=values[:3].tolist(),
                quaternion_xyzw=values[3:7].tolist(),
                timestamp_ns=frame.source_timestamp_ns,
                frame_attributes=dict(frame.attributes),
                coordinate_system="FAST-LIO odometry",
            )
        except Exception as exc:
            result["odometry_missing_reason"] = str(exc)
        bp = dict(self.profile.component("base_pose"))
        result["mount_calibration"] = {
            k: v for k, v in bp.items() if k.startswith("camera_") or k.startswith("dual_chest_camera_")
        }
        metadata = directory / "snapshot.json"
        metadata.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        self.recorder.write(
            "vla_snapshot",
            generation=generation,
            skill_id=skill,
            path=str(metadata),
            rgb_count=len(result["cameras"]),
            depth_count=sum(bool(v["depth"]) for v in result["cameras"].values()),
            pose_valid=result["odometry"] is not None,
            mask_status="pending_offline",
        )

    def _worker(self):
        with SensorGatewayClient(
            self.profile.endpoint_uri("sensor_gateway_metadata"), request_timeout_ms=100
        ) as client:
            while True:
                item = self.pending.get()
                if item is None:
                    return
                try:
                    self._save(client, *item)
                except Exception as exc:
                    self.recorder.write(
                        "vla_snapshot_failed", generation=item[0][0], skill_id=item[0][1], reason=str(exc)
                    )

    def close(self):
        if self.thread:
            self.pending.put(None)
            self.thread.join(timeout=2.0)
