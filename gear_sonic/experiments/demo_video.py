"""Cache source JPEGs in RAM; recoverable archives and annotated videos at trial end.

No inference, image decoding, or video encoding runs in the capture loop. The
same compressed camera payloads used by data collection are kept once per new
frame. Overlay changes are small compressed records, applied only on export.
"""

from __future__ import annotations

import argparse
from array import array
import base64
from dataclasses import asdict
from fractions import Fraction
import json
import logging
import math
import os
from pathlib import Path
import queue
import signal
import threading
import time
import zlib

import cv2
import numpy as np

from gear_sonic.experiments.recording import Recorder
from gear_sonic.runtime.gateway.control_client import ControlGatewaySubscriber
from gear_sonic.runtime.gateway.sensor_client import SensorGatewayClient
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest
from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.utils.operator.cv_viewer import (
    HEAD_RGB_STREAM, CHEST_RGB_STREAM,
    NavigationViewerState, draw_base_pose_overlays,
)

LOGGER = logging.getLogger("sonic.demo")
STREAMS = {
    "head": "camera_encoded/ego_view",
    "chest": "camera_encoded/chest_view",
}
# Bound only the compressed payload cache, not recording duration. Crossing
# this emergency limit spills JPEG bytes, never raw RGB, and preserves the take.
DEFAULT_CACHE_BYTES = 2 * 1024**3
INDEX_FIELDS = ("monotonic_ns", "wall_time_ns", "source_timestamp_ns",
                "received_timestamp_ns", "sequence", "offset", "length")


class JpegCache:
    def __init__(self):
        self.data = bytearray()
        self.index = array("q")
        self.written = 0
        self.last_identity = None
        self.last_payload = None
        self.last_range = None
        self.source_clock = "unknown"

    def add(self, payload, frame, *, wall_ns, mono_ns):
        identity = (frame.metadata.source, frame.metadata.sequence, frame.metadata.timestamp_ns)
        if identity == self.last_identity:
            return
        if payload == self.last_payload:
            offset, length = self.last_range
        else:
            offset, length = self.written + len(self.data), len(payload)
            self.data.extend(payload)
            self.last_payload, self.last_range = payload, (offset, length)
        self.index.extend((mono_ns, wall_ns, frame.source_timestamp_ns,
                           frame.metadata.timestamp_ns, frame.metadata.sequence, offset, length))
        self.last_identity = identity
        self.source_clock = frame.source_clock

    def flush(self, path):
        if self.data:
            with path.open("ab") as stream:
                stream.write(self.data)
            self.written += len(self.data)
            self.data.clear()


class DemoTake:
    def __init__(self, run_dir, start, *, max_cache_bytes=DEFAULT_CACHE_BYTES):
        self.start = start
        self.root = Path(run_dir) / "demo" / start["trial_id"] / "video"
        self.root.mkdir(parents=True, exist_ok=True)
        self.caches = {name: JpegCache() for name in STREAMS}
        self.overlays = []
        self.last_overlay = None
        self.max_cache_bytes = max_cache_bytes
        self.spills = 0
        self.errors = {name: 0 for name in STREAMS}

    def overlay(self, state, mono_ns):
        # Preserve the actual viewer state, including mask row spans and lines.
        payload = json.dumps(asdict(state), separators=(",", ":"), sort_keys=True).encode()
        if payload != self.last_overlay:
            self.overlays.append((mono_ns, zlib.compress(payload)))
            self.last_overlay = payload

    def add(self, name, payload, frame, *, wall_ns, mono_ns):
        self.caches[name].add(payload, frame, wall_ns=wall_ns, mono_ns=mono_ns)
        if sum(len(cache.data) for cache in self.caches.values()) > self.max_cache_bytes:
            self.spills += 1
            LOGGER.warning("Demo compressed cache reached its limit; spilling JPEGs for %s",
                           self.start["trial_id"])
            for stream, cache in self.caches.items():
                cache.flush(self.root / f"{stream}.mjpeg")

    def finish(self, end):
        streams = {}
        for name, cache in self.caches.items():
            cache.flush(self.root / f"{name}.mjpeg")
            with (self.root / f"{name}.jsonl").open("w") as handle:
                for offset in range(0, len(cache.index), len(INDEX_FIELDS)):
                    row = dict(zip(INDEX_FIELDS, cache.index[offset:offset + len(INDEX_FIELDS)]))
                    handle.write(json.dumps(row) + "\n")
            streams[name] = dict(stream=STREAMS[name], frames=len(cache.index) // len(INDEX_FIELDS),
                                 jpeg_bytes=cache.written, source_clock=cache.source_clock,
                                 unavailable_polls=self.errors[name])
        with (self.root / "overlays.jsonl").open("w") as handle:
            for stamp, payload in self.overlays:
                handle.write(json.dumps(dict(monotonic_ns=stamp,
                                             state=json.loads(zlib.decompress(payload)))) + "\n")
        manifest = dict(schema="sonic.demo.video.v1", trial_id=self.start["trial_id"],
                        start=self.start, end=end, streams=streams, cache_spills=self.spills,
                        status="cached", fps=30, stale_after_seconds=1.5,
                        timing="Video zero is trial_start.monotonic_ns; source timestamps are preserved in indexes.",
                        overlay="Same NavigationViewerState and draw_base_pose_overlays as the live viewer.")
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        # No frames survive into the background encoder's queue.
        self.caches.clear()
        self.overlays.clear()
        self.last_overlay = None
        return manifest_path


def render_take(manifest_path):
    """Re-renderable, sequential decoding; preserve wall duration across gaps."""
    import av

    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    fps = manifest["fps"]
    start_ns = manifest["start"]["monotonic_ns"]
    end_ns = manifest["end"]["monotonic_ns"]
    count = max(1, math.ceil((end_ns - start_ns) / 1e9 * fps))
    overlays = [json.loads(line) for line in (root / "overlays.jsonl").read_text().splitlines()]
    for name, info in manifest["streams"].items():
        # Older manifests may contain navigation previews. Keep their history,
        # but export only the two camera streams, including on --render.
        if name not in STREAMS:
            continue
        if not info["frames"]:
            info.update(status="missing", reason="No fresh frames received")
            continue
        try:
            rows = [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines()]
            with (root / f"{name}.mjpeg").open("rb") as jpeg_file:
                def decode(row):
                    jpeg_file.seek(row["offset"])
                    payload = jpeg_file.read(row["length"])
                    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Invalid JPEG at {row['offset']}")
                    return image

                first = decode(rows[0])
                height, width = first.shape[:2]
                partial = root / f"{name}.partial.mp4"
                with av.open(str(partial), mode="w") as container:
                    stream = container.add_stream("libx264", rate=fps,
                                                  options={"preset": "veryfast", "crf": "23"})
                    stream.width, stream.height = width, height
                    stream.pix_fmt = "yuv420p"
                    stream.codec_context.thread_count = 2
                    row_index = overlay_index = -1
                    cached_image = first
                    state = NavigationViewerState()
                    for tick in range(count):
                        stamp = start_ns + tick * 1_000_000_000 // fps
                        previous = row_index
                        while row_index + 1 < len(rows) and rows[row_index + 1]["monotonic_ns"] <= stamp:
                            row_index += 1
                        if row_index != previous:
                            cached_image = decode(rows[row_index])
                        while overlay_index + 1 < len(overlays) and overlays[overlay_index + 1]["monotonic_ns"] <= stamp:
                            overlay_index += 1
                            state = NavigationViewerState(**overlays[overlay_index]["state"])
                        stale = row_index < 0 or stamp - rows[row_index]["monotonic_ns"] > manifest["stale_after_seconds"] * 1e9
                        if stale:
                            image = np.zeros_like(first)
                            cv2.putText(image, "NO FRESH FRAME", (12, min(35, height - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        else:
                            image = cached_image.copy()
                            draw_base_pose_overlays(image, HEAD_RGB_STREAM if name == "head" else CHEST_RGB_STREAM, state)
                        video_frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                        video_frame.pts, video_frame.time_base = tick, Fraction(1, fps)
                        for packet in stream.encode(video_frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                partial.replace(root / f"{name}.mp4")
                info.update(status="rendered", video=f"{name}.mp4", output_frames=count)
        except Exception as exc:
            LOGGER.exception("Demo export failed for %s", name)
            info.update(status="error", error=str(exc))
    required_ok = all(manifest["streams"][name]["status"] == "rendered" for name in ("head", "chest"))
    manifest["status"] = "rendered" if required_ok else "partial"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


class EventTail:
    """Incremental complete-line reader; a concurrent partial append is retried."""
    def __init__(self, path):
        self.path, self.offset = Path(path), 0

    def poll(self):
        if not self.path.exists():
            return []
        result = []
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                self.offset = stream.tell()
                event = json.loads(line)
                if event.get("type") in {"trial_start", "trial_end"}:
                    result.append(event)
        return result


class DemoSession:
    def __init__(self, profile):
        self.recorder = Recorder(profile)
        self.run_dir = Path(self.recorder.config["run_dir"])
        self.tail = EventTail(self.run_dir / "events.jsonl")
        self.take = None
        self.jobs = queue.Queue()
        self.worker = threading.Thread(target=self._export, name="demo-export")
        self.worker.start()
        self.recorder.write("demo_recorder_ready", pid=os.getpid())

    def _export(self):
        while (path := self.jobs.get()) is not None:
            try:
                result = render_take(path)
                self.recorder.write("demo_video_saved", trial_id=result["trial_id"],
                                    path=str(path), status=result["status"])
            except Exception:
                LOGGER.exception("Demo export failed; cached JPEGs remain at %s", path)

    def finish(self, end):
        if self.take is not None:
            path = self.take.finish(end)
            self.take = None
            self.jobs.put(path)

    def poll(self, state):
        events = self.tail.poll()
        # When attaching late, do not manufacture recordings for ended trials.
        ended = {e["trial_id"] for e in events if e["type"] == "trial_end"}
        for event in events:
            if event["type"] == "trial_start":
                if event["trial_id"] in ended:
                    continue
                if self.take is not None:
                    self.finish(dict(event, reason="restarted"))
                root = self.run_dir / "demo" / event["trial_id"] / "video"
                if root.exists():
                    LOGGER.error("Refusing to overwrite existing demo take %s", root)
                    continue
                self.take = DemoTake(self.run_dir, event)
                self.take.overlay(state if state.generation == event["generation"] else NavigationViewerState(),
                                  time.monotonic_ns())
                self.recorder.write("demo_video_started", generation=event["generation"], path=str(root))
            elif self.take is not None and event["trial_id"] == self.take.start["trial_id"]:
                self.finish(event)

    def close(self):
        self.finish(dict(type="recorder_shutdown", wall_time_ns=time.time_ns(),
                         monotonic_ns=time.monotonic_ns(), reason="recorder_shutdown"))
        self.jobs.put(None)
        self.worker.join()


def run(profile):
    if not profile.components.get("experiment", {}).get("demo"):
        raise ValueError("Demo recording requires launching the experiment with --demo")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_: stop.set())
    parent = os.getppid()
    client = SensorGatewayClient(profile.endpoint_uri("sensor_gateway_metadata"), request_timeout_ms=10)
    subscriber = ControlGatewaySubscriber(profile.endpoint_uri("control_gateway_dispatch"))
    state = NavigationViewerState()
    session = DemoSession(profile)
    try:
        while not stop.is_set() and os.getppid() == parent:
            began = time.monotonic()
            session.poll(state)
            # read_command returns None for filtered/expired messages too, so
            # the next loop will continue draining if such a message occurs.
            while (command := subscriber.read_command()) is not None:
                if state.accept_control(command) and session.take is not None:
                    session.take.overlay(state, time.monotonic_ns())
            if session.take is not None:
                for name, stream in STREAMS.items():
                    try:
                        snapshot = client.read_snapshot(SnapshotRequest(
                            streams=(stream,), max_age_ms=1500, max_skew_ms=0), retries=0)
                        frame = snapshot.snapshot.frames[stream]
                        values = snapshot.arrays[stream]
                        if values.ndim != 1 or values.dtype != np.uint8:
                            raise ValueError("Expected encoded JPEG stream")
                        payload = values.tobytes()
                        if frame.attributes.get("encoding") == "base64_jpeg":
                            payload = base64.b64decode(payload, validate=True)
                        session.take.add(name, payload, frame,
                                         wall_ns=time.time_ns(), mono_ns=time.monotonic_ns())
                    except Exception:
                        session.take.errors[name] += 1
            stop.wait(max(0, 1 / 30 - (time.monotonic() - began)))
    finally:
        subscriber.close()
        client.close()
        session.poll(state)
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile", type=Path)
    group.add_argument("--render", type=Path, help="Recover/export an existing video/manifest.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.render:
        render_take(args.render)
    else:
        run(load_runtime_profile(args.profile))


if __name__ == "__main__":
    main()
