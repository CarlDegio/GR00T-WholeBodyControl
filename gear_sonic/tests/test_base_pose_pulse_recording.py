from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gear_sonic.utils.inference.base_pose.pulse_recording import VyPulseRecorder


@pytest.fixture
def recording(tmp_path, monkeypatch):
    monkeypatch.setenv("SONIC_EXPERIMENT_MINIMAL_LOGGING", "1")
    clock = [10.0]
    recorder = VyPulseRecorder(
        tmp_path / "vy_pulses.jsonl",
        monotonic=lambda: clock[0],
        wall_time_ns=lambda: round((1000 + clock[0]) * 1e9),
    )
    yield recorder, clock
    recorder.close()


def frame(recorder, timestamp, *, forward=1.0, right=0.0, stream="chest_view", track=7, kind="observation"):
    observation = SimpleNamespace(
        camera_timestamp=1000 + timestamp,
        target_track_id=track,
        target=SimpleNamespace(forward_m=forward, right_m=right),
    )
    recorder.observe_frame(
        SimpleNamespace(generation=1, details={"attempt_id": 1}, kind=kind, observation=observation),
        stream,
    )


def send(recorder, clock, timestamp, vy=0.0, *, token=None, vx=0.0):
    clock[0] = timestamp
    recorder.observe_send((vx, vy, 0.0), pulse_token=token, identity=(1, 2, 9))


def read(recorder):
    recorder.close()
    return [json.loads(line) for line in recorder.path.read_text().splitlines()]


def test_signed_visual_motion_and_actual_send_interval_survive_minimal_logging(recording):
    recorder, clock = recording
    # The target moves 4 cm right and 1 cm closer during a +vy pulse,
    # then another 2 cm right while the outgoing command is zero.
    frame(recorder, 9.98, right=-0.002, forward=1.0005)
    send(recorder, clock, 10.0, 0.4, token=9.97)
    frame(recorder, 10.02, right=0.002, forward=0.9995)
    frame(recorder, 10.20, right=0.020, forward=0.995)
    send(recorder, clock, 10.20, 0.4, token=9.97)  # Renewal, not a second pulse.
    frame(recorder, 10.38, right=0.038, forward=0.9905)
    send(recorder, clock, 10.4)
    frame(recorder, 10.42, right=0.042, forward=0.9895)
    frame(recorder, 10.88, right=0.058, forward=0.99)
    clock[0] = 10.9
    recorder.tick()
    assert not recorder.path.exists()  # Wait for an image after the boundary.
    frame(recorder, 10.92, right=0.062, forward=0.99)
    recorder.tick()
    (row,) = read(recorder)
    assert row["vy_m_s"] == 0.4
    assert row["pulse_duration_s"] == pytest.approx(0.4)  # First actual send, not token time.
    assert row["wait_duration_s"] == pytest.approx(0.5)
    assert row["pulse_displacement"]["left_m"] == pytest.approx(0.04)
    assert row["pulse_displacement"]["forward_m"] == pytest.approx(0.01)
    assert row["pulse_displacement"]["planar_m"] == pytest.approx(0.041231056)
    assert row["wait_displacement"]["left_m"] == pytest.approx(0.02)
    assert row["total_displacement"]["left_m"] == pytest.approx(0.06)
    assert row["visual_samples"]["stop"]["interpolation_span_s"] == pytest.approx(0.04)


def test_reverse_pulses_have_opposite_signed_displacements(recording):
    recorder, clock = recording
    for start, sign in ((10.0, 1), (11.0, -1)):
        frame(recorder, start, right=0)
        send(recorder, clock, start, 0.4 * sign, token=start)
        frame(recorder, start + 0.2, right=sign * 0.02)
        send(recorder, clock, start + 0.2)
        frame(recorder, start + 0.7, right=sign * 0.03)
        clock[0] = start + 0.7
        recorder.tick()
    rows = read(recorder)
    assert [r["vy_m_s"] for r in rows] == [0.4, -0.4]
    assert [r["total_displacement"]["left_m"] for r in rows] == pytest.approx([0.03, -0.03])


@pytest.mark.parametrize("change", ["camera", "target", "lost", "stale"])
def test_unreliable_visual_boundaries_are_missing_not_zero(recording, change):
    recorder, clock = recording
    frame(recorder, 10.0)
    send(recorder, clock, 10.0, -0.4, token=10.0)
    send(recorder, clock, 10.2)
    kwargs = {"stream": "ego_view"} if change == "camera" else {"track": 8} if change == "target" else {}
    if change == "lost":
        frame(recorder, 10.1, kind="invalid")
    if change != "stale":
        frame(recorder, 10.2, right=0.1, **kwargs)
        frame(recorder, 10.7, right=0.2, **kwargs)
    clock[0] = 11.2
    recorder.tick()
    (row,) = read(recorder)
    assert "missing_reason" in row["pulse_displacement"]
    assert "planar_m" not in row["total_displacement"]


def test_new_motion_truncates_wait_instead_of_attributing_it_to_settling(recording):
    recorder, clock = recording
    frame(recorder, 10)
    send(recorder, clock, 10, 0.4, token=10)
    frame(recorder, 10.2, right=0.01)
    send(recorder, clock, 10.2)
    frame(recorder, 10.3, right=0.02)
    send(recorder, clock, 10.3, vx=0.4)
    (row,) = read(recorder)
    assert row["end_reason"] == "interrupted_by_motion"
    assert row["wait_duration_s"] == pytest.approx(0.1)


def test_cancel_records_partial_pulse_and_no_full_wait(recording):
    recorder, clock = recording
    frame(recorder, 10)
    send(recorder, clock, 10, 0.4, token=10)
    clock[0] = 10.1
    recorder.cancel("operator_stop")
    (row,) = read(recorder)
    assert row["pulse_duration_s"] == pytest.approx(0.1)
    assert row["wait_duration_s"] == 0
    assert row["stop_time_is_cancellation"]


def test_no_pulses_does_not_create_a_log(recording):
    recorder, clock = recording
    frame(recorder, 10)
    send(recorder, clock, 10, 0.4)  # Continuous lateral motion.
    send(recorder, clock, 11)
    recorder.close()
    assert not recorder.path.exists()


def test_real_backprojection_and_adapter_record_without_changing_commands(recording):
    import numpy as np

    from gear_sonic.utils.inference.base_pose.agent import BasePoseAgentConfig, GatewayRawServoAdapter
    from gear_sonic.utils.inference.base_pose.sensor import AlignedRGBDSnapshot
    from gear_sonic.utils.inference.base_pose.servo import (
        RawServoCalibration,
        RawServoEvent,
        RawServoObservation,
        ServoPhase,
        YawAlignGeometry,
        estimate_target_geometry,
    )

    recorder, clock = recording
    sent = []
    config = BasePoseAgentConfig(
        task="basket",
        raw_head_target_distance_m=1.0,
        raw_chest_target_distance_m=1.0,
        raw_lateral_tolerance_m=0.06,
        raw_lateral_pulse_max_s=0.2,
    )
    adapter = GatewayRawServoAdapter(
        config,
        submit_intent=lambda name, p: sent.append((name, p)),
        monotonic=lambda: clock[0],
        pulse_recorder=recorder,
    )
    calibration = RawServoCalibration(64, 64, 100.0, 100.0, 32.0, 32.0, camera_pitch_deg=0)

    def observe(timestamp, pixel_offset, kind="observation"):
        clock[0] = timestamp
        snapshot = AlignedRGBDSnapshot(
            rgb=np.zeros((64, 64, 3), dtype=np.uint8),
            depth_raw=np.full((64, 64), 1000, dtype=np.uint16),
            depth_scale_m=0.001,
            depth_aligned_to="chest_view",
            depth_source="raw",
            fx=100.0,
            fy=100.0,
            cx=32.0,
            cy=32.0,
            timestamp=1000 + timestamp,
        )
        bbox = (22.0 + pixel_offset, 20.0, 42.0 + pixel_offset, 44.0)
        geometry = estimate_target_geometry(
            snapshot, np.ones((64, 64), dtype=np.uint8), calibration, bbox_xyxy=bbox
        )
        observation = RawServoObservation(
            target=geometry,
            yaw_align_geometry=YawAlignGeometry(0.0, 30.0, 30, (32.0, 30.0)),
            camera_timestamp=snapshot.timestamp,
            target_track_id=7,
            yaw_align_target_track_id=None,
            target_bbox_xyxy=bbox,
            image_width=64,
            image_height=64,
        )
        assert adapter.runtime.accept_event(
            RawServoEvent(
                1, kind, observation=observation, details={"attempt_id": 1, "live_stream": "chest_view"}
            ),
            now=timestamp,
        )

    assert adapter.start(1)
    observe(10, 9, "initialized")
    adapter.runtime.controller.phase = ServoPhase.TRANSLATE_TARGET
    observe(10.05, 9)
    adapter.tick()
    assert sent[-1][1]["velocity"][1] == pytest.approx(-0.4)
    observe(10.25, 7)
    adapter.tick()
    assert sent[-1][1]["velocity"] == [0.0, 0.0, 0.0]
    observe(10.50, 6)
    observe(10.75, 5)
    adapter.tick()
    (row,) = read(recorder)
    assert row["pulse_duration_s"] == pytest.approx(0.2)
    assert row["pulse_displacement"]["left_m"] == pytest.approx(-0.02)
    assert row["wait_displacement"]["left_m"] == pytest.approx(-0.02)
    assert row["total_displacement"]["planar_m"] == pytest.approx(0.04)
    assert row["visual_samples"]["start"]["camera_stream"] == "chest_view"
    adapter.shutdown()
