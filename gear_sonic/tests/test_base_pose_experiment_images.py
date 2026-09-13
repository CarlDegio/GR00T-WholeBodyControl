"""Exercise experiment image recording without robot sockets or model loading."""

import queue
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gear_sonic.experiments import base_pose
from gear_sonic.utils.inference.base_pose.agent import BasePoseAgentConfig, run_base_pose_yolo_agent
from gear_sonic.utils.inference.base_pose.diagnostics import AsyncFrameDiagnosticsWriter
from gear_sonic.utils.inference.base_pose.servo import GenerationGate, TrackedInstance


def test_old_frozen_experiment_enables_images_before_selecting_worker(monkeypatch):
    config = SimpleNamespace(profile="old-runtime.yaml", overlay=(), raw_diagnostic_image_interval_frames=0)
    profile = SimpleNamespace(components={"experiment": {"alignment": "geometric"}})
    monkeypatch.setattr("gear_sonic.runtime.profile.load_runtime_profile", lambda *a, **k: profile)
    calls = []
    monkeypatch.setattr(base_pose, "run_geometric_service", lambda c, p: calls.append(c.raw_diagnostic_image_interval_frames))
    run_base_pose_yolo_agent(config)
    assert calls == [5]


@pytest.mark.parametrize("interval,expected", [(0, []), (5, [0, 5, 10])])
def test_sampler_saves_each_fifth_new_frame_and_skips_cached_frames(tmp_path, interval, expected):
    config = SimpleNamespace(output_root=str(tmp_path), raw_diagnostic_image_interval_frames=interval)
    diagnostics = AsyncFrameDiagnosticsWriter()
    images = base_pose.SampledCameraImages(config, diagnostics, 1, "geometric")
    try:
        for index in range(11):
            snapshot = SimpleNamespace(timestamp=float(index), rgb=np.full((20, 30, 3), index, dtype=np.uint8))
            images.record("ego_view", snapshot)
            images.record("ego_view", snapshot)
    finally:
        diagnostics.close()
    files = sorted(tmp_path.rglob("frame_*.jpg"))
    assert [p.stem for p in files] == [f"frame_{i:06d}" for i in expected]
    for path, index in zip(files, expected):
        assert int(cv2.imread(str(path))[10, 15, 0]) == index


def test_head_worker_saves_images_through_target_loss_and_invalid_geometry(tmp_path, monkeypatch):
    config = BasePoseAgentConfig(task="align", output_root=str(tmp_path))
    gate = GenerationGate()
    gate.activate(1)
    target = TrackedInstance(1, 0, .9, (20., 22., 40., 34.), np.zeros((40, 60), dtype=np.uint8))
    target.mask[22:34, 20:40] = 1

    class Camera:
        index = -1
        closed = False

        def begin_generation(self, generation):
            assert generation == 1

        def capture_stream(self, stream, **kwargs):
            self.index += 1
            if self.index == 11:
                gate.cancel(1)
                raise TimeoutError("end of offline frames")
            return SimpleNamespace(timestamp=float(self.index), rgb=np.full((40, 60, 3), self.index, dtype=np.uint8))

        def close(self):
            self.closed = True

    def observe(snapshot, *args, **kwargs):
        if snapshot.timestamp == 10:
            raise ValueError("missing depth")
        return SimpleNamespace(
            yaw_align_target_rgb_edges=None,
            yaw_align_geometry=SimpleNamespace(line_endpoints_px=((5., 12.), (54., 12.))),
        )

    camera = Camera()
    tracker = SimpleNamespace(track=lambda rgb: [] if int(rgb[0, 0, 0]) == 5 else [target])
    monkeypatch.setattr(base_pose, "tracker_for", lambda *a: tracker)
    monkeypatch.setattr(base_pose, "head_calibration", lambda *a: object())
    monkeypatch.setattr(base_pose, "_observation", observe)
    requests, events = queue.Queue(), queue.Queue()
    requests.put(1)
    requests.put(None)
    diagnostics = AsyncFrameDiagnosticsWriter()
    try:
        base_pose.run_head_worker(
            config, requests, events, gate, threading.Event(), diagnostics=diagnostics,
            camera_factory=lambda: camera,
        )
    finally:
        diagnostics.close()
    assert camera.closed
    assert all(event.kind != "error" for event in list(events.queue))
    files = sorted(tmp_path.rglob("frame_*.jpg"))
    assert [p.stem for p in files] == ["frame_000000", "frame_000005", "frame_000010"]
    image = cv2.imread(str(files[0]))
    assert int(image[12, 30, 1]) > int(image[12, 30, 2])
