"""Record vy pulses and target-relative motion from RGB-D back-projection."""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import queue
import threading
import time
from typing import Callable


def visual_displacement(start: dict, end: dict) -> dict:
    """A stationary target moves opposite to the camera/body translation."""

    for sample in (start, end):
        if "missing_reason" in sample:
            return {"missing_reason": sample["missing_reason"]}
    forward = start["forward_m"] - end["forward_m"]
    left = end["right_m"] - start["right_m"]
    return dict(planar_m=math.hypot(forward, left), forward_m=forward, left_m=left)


class VyPulseRecorder:
    """Observe sends and visual frames; write only one small row per pulse."""

    def __init__(
        self,
        path: str | Path,
        *,
        settle_s: float = 0.5,
        max_frame_gap_s: float = 0.4,
        logger: Callable[[str], None] = print,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.path = Path(path)
        self.settle_s = float(settle_s)
        self.max_frame_gap_s = float(max_frame_gap_s)
        self.monotonic = monotonic
        self.wall_time_ns = wall_time_ns
        self.logger = logger
        self.frames: deque[dict] = deque(maxlen=256)
        self.visual_epoch = 0
        self.visual_key = None
        self.active: dict | None = None
        self.completed: list[dict] = []
        self.sequence = 0
        self.closed = False
        self.pending = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._writer, name="base-pose-vy-recording", daemon=True)
        self.thread.start()

    def observe_frame(self, event, camera_stream: str | None) -> None:
        observation = event.observation
        if event.kind not in {"initialized", "observation"} or observation is None:
            self.visual_key = None
            return
        key = (
            event.generation,
            (event.details or {}).get("attempt_id"),
            camera_stream,
            observation.target_track_id,
        )
        if key != self.visual_key:
            self.visual_epoch += 1
            self.visual_key = key
        target = observation.target
        values = (observation.camera_timestamp, target.forward_m, target.right_m)
        if not all(math.isfinite(v) for v in values) or values[0] <= 0 or not camera_stream:
            self.visual_key = None
            return
        sample = dict(
            camera_timestamp_s=float(values[0]),
            forward_m=float(values[1]),
            right_m=float(values[2]),
            camera_stream=camera_stream,
            target_track_id=observation.target_track_id,
            epoch=self.visual_epoch,
        )
        if self.frames and sample["camera_timestamp_s"] <= self.frames[-1]["camera_timestamp_s"]:
            return
        self.frames.append(sample)

    def _sample(self, wall_ns: int, epoch: int | None) -> dict:
        timestamp = wall_ns / 1e9
        frames = list(self.frames)
        before = next((f for f in reversed(frames) if f["camera_timestamp_s"] <= timestamp), None)
        after = next((f for f in frames if f["camera_timestamp_s"] >= timestamp), None)
        if before is None or after is None:
            return {"missing_reason": "no visual frames bracketing boundary"}
        if before["epoch"] != epoch or after["epoch"] != epoch:
            return {"missing_reason": "camera/target changed or tracking interrupted"}
        lo, hi = before["camera_timestamp_s"], after["camera_timestamp_s"]
        if hi - lo > self.max_frame_gap_s:
            return {"missing_reason": "visual frames are too far apart"}
        fraction = 0.0 if hi == lo else (timestamp - lo) / (hi - lo)
        return dict(
            forward_m=before["forward_m"] + fraction * (after["forward_m"] - before["forward_m"]),
            right_m=before["right_m"] + fraction * (after["right_m"] - before["right_m"]),
            camera_stream=before["camera_stream"],
            target_track_id=before["target_track_id"],
            frame_timestamps_s=[lo, hi],
            interpolation_span_s=hi - lo,
        )

    def observe_send(
        self,
        velocity,
        *,
        pulse_token: float | None,
        identity: tuple[int, int, int],
        stop_reason: str | None = None,
    ) -> None:
        if self.closed:
            return
        now, wall_ns = self.monotonic(), self.wall_time_ns()
        self.tick(now=now)
        vy = float(velocity[1])
        moving = any(float(v) != 0.0 for v in velocity)
        is_pulse = pulse_token is not None and vy != 0.0
        if self.active is not None and "stop" not in self.active:
            if (
                is_pulse
                and self.active["pulse_token"] == pulse_token
                and self.active["identity"] == identity
                and self.active["vy_m_s"] == vy
            ):
                return
            self.active["stop_reason"] = stop_reason or "command_changed"
            self.active["stop"] = dict(monotonic_s=now, wall_time_ns=wall_ns)
        if self.active is not None and moving:
            self._finish(now, wall_ns, "interrupted_by_motion")
        if is_pulse:
            self.sequence += 1
            self.active = dict(
                pulse_index=self.sequence,
                identity=identity,
                pulse_token=pulse_token,
                visual_epoch=self.visual_epoch if self.visual_key is not None else None,
                vy_m_s=vy,
                generation=identity[0],
                skill_id=identity[1],
                segment_id=identity[2],
                start=dict(monotonic_s=now, wall_time_ns=wall_ns),
            )

    def tick(self, *, now: float | None = None, force: bool = False) -> None:
        now = self.monotonic() if now is None else now
        if self.active is not None and "stop" in self.active:
            stop = self.active["stop"]
            if now + 1e-9 >= stop["monotonic_s"] + self.settle_s:
                self._finish(
                    stop["monotonic_s"] + self.settle_s,
                    stop["wall_time_ns"] + round(self.settle_s * 1e9),
                    "settled",
                )
        for record in self.completed[:]:
            end = record["wait_end"]
            have_end_frame = self.frames and self.frames[-1]["camera_timestamp_s"] >= end["wall_time_ns"] / 1e9
            if not (force or have_end_frame or now >= end["monotonic_s"] + self.max_frame_gap_s):
                continue
            epoch = record.pop("visual_epoch")
            samples = {k: self._sample(record[k]["wall_time_ns"], epoch) for k in ("start", "stop", "wait_end")}
            record.update(
                measurement=(
                    "RGB-D target back-projection; stationary target, approximately fixed camera orientation"
                ),
                pulse_displacement=visual_displacement(samples["start"], samples["stop"]),
                wait_displacement=visual_displacement(samples["stop"], samples["wait_end"]),
                total_displacement=visual_displacement(samples["start"], samples["wait_end"]),
                visual_samples=samples,
            )
            self.pending.put(record)
            self.completed.remove(record)

    def _finish(self, now: float, wall_ns: int, reason: str) -> None:
        record = self.active
        self.active = None
        record.pop("identity")
        record.pop("pulse_token")
        record.update(
            type="vy_pulse",
            end_reason=reason,
            requested_wait_s=self.settle_s,
            wait_end=dict(monotonic_s=now, wall_time_ns=wall_ns),
            pulse_duration_s=record["stop"]["monotonic_s"] - record["start"]["monotonic_s"],
            wait_duration_s=now - record["stop"]["monotonic_s"],
        )
        self.completed.append(record)

    def cancel(self, reason: str) -> None:
        now, wall_ns = self.monotonic(), self.wall_time_ns()
        self.tick(now=now)
        if self.active is not None:
            if "stop" not in self.active:
                self.active["stop_reason"] = reason
                self.active["stop_time_is_cancellation"] = True
                self.active["stop"] = dict(monotonic_s=now, wall_time_ns=wall_ns)
            self._finish(now, wall_ns, reason)
        self.visual_key = None

    def close(self) -> None:
        if self.closed:
            return
        self.cancel("shutdown")
        self.tick(force=True)
        self.closed = True
        self.pending.put(None)
        self.thread.join(timeout=2.0)

    def _writer(self) -> None:
        while (record := self.pending.get()) is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            except Exception as exc:
                self.logger(f"[RawServo] WARNING vy pulse recording failed: {exc}")
