"""Semantic-role evaluation and post-trial mask generation; never commands robots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
from types import SimpleNamespace

import cv2
import numpy as np

from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.utils.inference.base_pose.agent import load_base_pose_config
from gear_sonic.utils.inference.base_pose.sensor import AlignedRGBDSnapshot
from gear_sonic.utils.inference.base_pose.servo import (
    RawServoCalibration,
    _observation,
    _prompts_match,
    _resolve_target,
)
from gear_sonic.utils.inference.lavira.agent import LaViRAClient

from .base_pose import tracker_for
from .config import ROOT
from .recording import Recorder, read_events


def visual_client(profile):
    # The tmux planner sources .env.local; standalone offline tools need the
    # same VA key without executing shell code or archiving credentials.
    local_keys = {}
    env_file = ROOT.parent / ".env.local"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            name, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and name.strip() in {"DASHSCOPE_API_KEY", "LAVIRA_VA_API_KEY"}:
                parts = shlex.split(value, comments=True)
                if len(parts) == 1:
                    local_keys[name.strip()] = parts[0]
    key = os.getenv("LAVIRA_VA_API_KEY") or local_keys.get("LAVIRA_VA_API_KEY")
    key = key or os.getenv("DASHSCOPE_API_KEY") or local_keys.get("DASHSCOPE_API_KEY")
    c = profile.component("lavira")
    client = LaViRAClient(
        la_client=SimpleNamespace(),  # Offline role evaluation never calls LA.
        va_api_key=key,
        **{
            k: c[k]
            for k in (
                "la_base_url",
                "va_base_url",
                "la_model",
                "va_model",
                "la_enable_thinking",
                "va_enable_thinking",
                "la_timeout_seconds",
                "va_timeout_seconds",
            )
        },
    )
    client.save_request_context = False
    return client


def load_samples(manifest):
    path = Path(manifest)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if row["sample_id"] in seen:
            raise ValueError("Duplicate semantic sample_id")
        seen.add(row["sample_id"])
        if row["task_id"] not in {"T1", "T2", "T3", "T4"} or not row["vla_prompt"].strip():
            raise ValueError("Samples require task_id and semantic vla_prompt")
        for key in ("rgb", "depth"):
            if row.get(key):
                row[key] = str((path.parent / row[key]).resolve())
    return rows


def segment_roles(config, image, roles, directory, *, depth=None, camera_info=None, mount=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target, yaw = roles["target"]["name"], roles["yaw_align_target"]["name"]
    tracker = tracker_for(config, target, yaw)
    instances = tracker.track(image)
    position, _, _ = _resolve_target(instances, -1)
    orientation = position
    if not _prompts_match(target, yaw):
        orientation, _, _ = _resolve_target(instances, -1, 1)
    out = {
        "roles": roles,
        "detection_geometry_available": None,
        "roles_detected": position is not None and orientation is not None,
    }
    for key, instance in (("target", position), ("yaw_align_target", orientation)):
        if instance is None:
            out[key] = {"mask": None, "missing_reason": "not_detected"}
        else:
            name = f"{key}_mask.png"
            if not cv2.imwrite(str(directory / name), instance.mask.astype(np.uint8) * 255):
                raise OSError("Cannot save role mask")
            out[key] = dict(
                mask=name,
                bbox_xyxy=list(instance.bbox_xyxy),
                track_id=instance.track_id,
                confidence=float(instance.confidence),
            )
    if position is None or orientation is None:
        out["detection_geometry_available"] = False
        return out
    if depth is None:
        out["geometry_missing_reason"] = "no_synchronized_depth"
        return out
    try:
        info, mount = dict(camera_info or {}), dict(mount or {})
        h, w = image.shape[:2]
        cal = RawServoCalibration(
            width=w,
            height=h,
            fx=info["fx"],
            fy=info["fy"],
            cx=info["cx"],
            cy=info["cy"],
            camera_pitch_deg=mount["camera_pitch_deg"],
            camera_roll_deg=mount["camera_roll_deg"],
            camera_yaw_deg=mount["camera_yaw_deg"],
            camera_forward_offset_m=mount["camera_forward_offset_m"],
            camera_lateral_offset_m=mount["camera_lateral_offset_m"],
        )
        snap = AlignedRGBDSnapshot(
            rgb=image,
            depth_raw=depth,
            fx=cal.fx,
            fy=cal.fy,
            cx=cal.cx,
            cy=cal.cy,
            depth_scale_m=info["depth_scale_m"],
            depth_aligned_to=info.get("depth_aligned_to"),
            depth_source=info.get("depth_source"),
            timestamp=0.0,
        )
        obs = _observation(
            snap, position, orientation, cal, exclude_target_from_yaw_align_edge=not _prompts_match(target, yaw)
        )
        out.update(
            detection_geometry_available=obs.yaw_align_geometry is not None,
            target_body_xyz_m=list(obs.target.body_xyz_m),
            yaw_geometry_error=obs.yaw_align_geometry_error,
        )
    except Exception as exc:
        out["geometry_missing_reason"] = str(exc)
    return out


def evaluate_manifest(profile_path, manifest):
    profile = load_runtime_profile(profile_path)
    recorder, client = Recorder(profile), visual_client(profile)
    config = load_base_pose_config(str(profile_path))
    selected = [row for row in load_samples(manifest) if row["task_id"] == recorder.config["task_id"]]
    if not selected:
        raise ValueError(f"No samples for {recorder.config['task_id']}")
    for index, row in enumerate(selected):
        try:
            image = cv2.imread(row["rgb"])
            if image is None:
                raise ValueError(f"Unreadable image: {row['rgb']}")
            roles = client.alignment_grounding(
                manipulation_prompt=row["vla_prompt"], direction="front", image_bgr=image
            )
            details = {"roles": roles, "detection_geometry_available": False}
            directory = recorder.path.parent / "semantic" / f"sample_{index:05d}"
            if roles["status"] == "FOUND":
                details = segment_roles(
                    config,
                    image,
                    roles,
                    directory,
                    depth=np.load(row["depth"], allow_pickle=False) if row.get("depth") else None,
                    camera_info=row.get("camera_info"),
                    mount=row.get("mount_calibration"),
                )
            recorder.write(
                "semantic_result",
                sample_id=row["sample_id"],
                task_id=row["task_id"],
                vla_prompt=row["vla_prompt"],
                input_rgb=row["rgb"],
                input_sha256=hashlib.sha256(Path(row["rgb"]).read_bytes()).hexdigest(),
                input_depth=row.get("depth"),
                camera_info=row.get("camera_info"),
                mount_calibration=row.get("mount_calibration"),
                artifact_dir=str(directory),
                **details,
            )
        except Exception as exc:
            recorder.write(
                "semantic_result",
                sample_id=row["sample_id"],
                task_id=row["task_id"],
                input_rgb=row.get("rgb"),
                vla_prompt=row["vla_prompt"],
                error=str(exc),
            )


def enrich_snapshots(profile_path):
    profile = load_runtime_profile(profile_path)
    recorder = Recorder(profile)
    events = read_events(recorder.path)
    starts = {e["trial_id"] for e in events if e["type"] == "trial_start"}
    ended = {e["trial_id"] for e in events if e["type"] == "trial_end"}
    if starts - ended:
        print(
            "Snapshot masks pending: an experiment trial has no terminal record. "
            "Finish/label the interrupted trial first."
        )
        return
    config = load_base_pose_config(str(profile_path))
    client = None
    completed = {(e["trial_id"], e["skill_id"]) for e in events if e["type"] == "mask_enrichment"}
    for event in events:
        if event["type"] != "vla_snapshot" or (event["trial_id"], event["skill_id"]) in completed:
            continue
        try:
            metadata = Path(event["path"])
            data = json.loads(metadata.read_text())
            first_inference = next(
                e["monotonic_ns"]
                for e in events
                if e["type"] == "vla_first_inference"
                and e.get("trial_id") == event["trial_id"]
                and e.get("skill_id") == event["skill_id"]
            )
            candidates = [
                e
                for e in events
                if e.get("trial_id") == event["trial_id"]
                and e["type"] == "check"
                and e.get("kind") == "alignment_grounding"
                and (e.get("result") or {}).get("status") == "FOUND"
                and e["monotonic_ns"] < first_inference
            ]
            roles = candidates[-1]["result"] if candidates else None
            source = "control_alignment" if roles else "offline_semantic_selection"
            if recorder.config["alignment"] == "geometric":
                target = recorder.config["geometric"]["target"]
                roles = dict(status="FOUND", target={"name": target}, yaw_align_target={"name": target})
                source = "control_geometric_centering"
            if roles is None:
                client = client or visual_client(profile)
                camera_name = "ego_view" if recorder.config["alignment"] == "head" else "chest_view"
                image = cv2.imread(str(metadata.parent / data["cameras"][camera_name]["rgb"]))
                roles = client.alignment_grounding(
                    manipulation_prompt=recorder.config["task"]["vla_prompt"], direction="front", image_bgr=image
                )
            results = {}
            for name in ("ego_view", "chest_view"):
                frame = data["cameras"][name]
                image = cv2.imread(str(metadata.parent / frame["rgb"]))
                info = dict(frame.get("depth_attributes", {}).get("camera_info", frame.get("camera_info", {})))
                mount = dict(data["mount_calibration"])
                if name == "chest_view":
                    for key in tuple(mount):
                        if key.startswith("dual_chest_camera_"):
                            mount[key.removeprefix("dual_chest_")] = mount[key]
                results[name] = (
                    segment_roles(
                        config,
                        image,
                        roles,
                        metadata.parent / f"{name}_roles",
                        depth=np.load(metadata.parent / frame["depth"], allow_pickle=False)
                        if frame.get("depth")
                        else None,
                        camera_info=info,
                        mount=mount,
                    )
                    if roles["status"] == "FOUND"
                    else {"roles": roles, "missing_reason": "roles_not_found"}
                )
            output = metadata.parent / "masks.json"
            output.write_text(json.dumps({"role_source": source, "views": results}, ensure_ascii=False, indent=2))
            recorder.write(
                "mask_enrichment",
                generation=event["generation"],
                skill_id=event["skill_id"],
                path=str(output),
                role_source=source,
            )
        except Exception as exc:
            recorder.write(
                "mask_enrichment_failed",
                generation=event["generation"],
                skill_id=event["skill_id"],
                reason=str(exc),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_profile", type=Path)
    args = parser.parse_args()
    enrich_snapshots(args.runtime_profile)
