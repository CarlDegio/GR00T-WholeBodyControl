"""Low-frequency, process-safe experimental evidence; never consume telemetry."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from .config import settings

EVENTS = {
    "TASK_ACCEPTED",
    "TASK_STARTED",
    "TASK_COMPLETED",
    "TASK_FAILED",
    "TASK_CANCELLED",
    "TASK_RESULT_PUBLISHED",
    "SKILL_STARTED",
    "SKILL_COMPLETED",
    "SKILL_FAILED",
    "NAV_HANDOFF_EVALUATED",
    "ALIGN_HANDOFF_EVALUATED",
    "CONTROLLER_STATUS",
    "VLA_TASK_ACKNOWLEDGED",
    "VLA_SAFETY_BLOCKED",
    "VLA_TASK_REJECTED",
    "VLA_INFERENCE_FAILED",
    "CAMERA_STALE",
    "CAMERA_RECOVERED",
    "GEOMETRIC_STARTED",
    "GEOMETRIC_PERCEPTION_READY",
    "GEOMETRIC_CAMERA_SELECTED",
    "GEOMETRIC_TARGET_LOST",
    "GEOMETRIC_MOTION_STARTED",
    "GEOMETRIC_FINISHED",
}
OMIT = {"skill_args", "expected_postcondition", "mission", "manipulation_prompt", "max_steps", "todo_list"}


def read_events(path):
    events = []
    with Path(path).open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        lines = stream.read().splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Incomplete/corrupt experiment log at line {number}: {path}") from exc
    return events


class Recorder:
    def __init__(self, profile=None, *, path=None):
        self.config = settings(profile) if profile is not None else {}
        self.path = (
            Path(path)
            if path
            else (Path(self.config["run_dir"]) / "events.jsonl" if self.config.get("run_dir") else None)
        )
        self.session_id = self.config.get("session_id", "")

    def trial_id(self, generation):
        return f"{self.session_id}-g{int(generation)}"

    def write(self, event_type, *, generation=None, **fields):
        if self.path is None:
            return None
        event = dict(
            type=event_type,
            event_id=uuid.uuid4().hex,
            wall_time_ns=time.time_ns(),
            monotonic_ns=time.monotonic_ns(),
            **fields,
        )
        if generation is not None:
            event.update(generation=int(generation), trial_id=self.trial_id(generation))
        payload = (json.dumps(event, ensure_ascii=False, allow_nan=False, default=str) + "\n").encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab", buffering=0) as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(payload)
            if event_type in {"trial_start", "trial_end", "annotation", "gate_label", "semantic_label"}:
                os.fsync(stream.fileno())
        return event

    def runtime(self, component, code, **fields):
        if code not in EVENTS:
            return
        generation = fields.pop("generation", None)
        if generation is None or int(generation) < 0:
            return
        self.write(
            "runtime",
            generation=generation,
            component=component,
            code=code,
            **{k: v for k, v in fields.items() if k not in OMIT},
        )
