"""Demo opt-in, model provenance, compressed retention and playable overlays."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import av
import cv2
import numpy as np
import pytest
import yaml

from gear_sonic.experiments.config import launch, resolve, CONFIGS
from gear_sonic.experiments.demo_evidence import DemoModelRecorder
from gear_sonic.experiments.demo_video import DemoTake, DemoSession, EventTail, render_take
from gear_sonic.experiments.recording import Recorder, read_events
from gear_sonic.utils.operator.cv_viewer import NavigationViewerState


def profile(tmp_path, demo=True):
    return SimpleNamespace(components={"experiment": dict(
        run_dir=str(tmp_path), session_id="demo-test", demo=demo)})


def jpeg(color=45):
    return cv2.imencode(".jpg", np.full((96, 128, 3), color, np.uint8))[1].tobytes()


def metadata(sequence):
    return SimpleNamespace(metadata=SimpleNamespace(source="camera", sequence=sequence,
                                                    timestamp_ns=1_000_000_000 + sequence),
                           source_timestamp_ns=2_000_000_000 + sequence, source_clock="camera")


def start_event():
    return dict(type="trial_start", trial_id="demo-test-g1", generation=1,
                monotonic_ns=1_000_000_000, wall_time_ns=2_000_000_000)


@pytest.mark.parametrize("enabled", [False, True])
def test_demo_cli_is_opt_in_and_background_recorder_does_not_depend_on_window(tmp_path, capsys, enabled):
    args = ["--dry-run", "--output", str(tmp_path)] + (["--demo"] if enabled else [])
    assert launch("full_vln", args) == 0
    result = yaml.safe_load(capsys.readouterr().out)
    assert result["experiment"].get("demo", False) is enabled
    from gear_sonic.scripts.launch_inference import load_inference_launch_config, build_sensor_gateway_command
    payload, _ = resolve(CONFIGS / "full_vln.yaml")
    payload["components"]["experiment"]["demo"] = enabled
    payload["components"]["launcher"]["opencv_viewer"] = False
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(payload))
    config = load_inference_launch_config(path)
    command = build_sensor_gateway_command(config, Path.cwd())
    assert ("gear_sonic.experiments.demo_video" in command) is enabled
    assert "cv_viewer" not in command


def test_demo_retains_todo_and_decisions_without_changing_normal_log(tmp_path):
    for enabled in (False, True):
        recorder = Recorder(profile(tmp_path / str(enabled), enabled))
        recorder.runtime("lavira", "TODO_UPDATED", generation=1, todo_list="- [x] MOVE_TO bag")
        recorder.runtime("lavira", "LA_DECISION", generation=1, skill_args={"target": "bin"})
        recorder.runtime("lavira", "SKILL_STARTED", generation=1, skill_args={"target": "bin"})
        events = read_events(recorder.path)
        assert len(events) == (3 if enabled else 1)
        assert ("skill_args" in events[-1]) is enabled
        if enabled:
            assert events[0]["todo_list"] == "- [x] MOVE_TO bag"


def test_exact_model_images_text_retries_and_full_response_are_recoverable(tmp_path, monkeypatch):
    from gear_sonic.utils.inference.lavira.agent import LaViRAClient, LaViRAAgentError
    recorder = Recorder(profile(tmp_path))
    demo = DemoModelRecorder(recorder, lambda: dict(generation=1, skill_id=7, stage="LA", step=4))
    payload = jpeg()
    request = dict(model="test", messages=[dict(role="user", content=[
        dict(type="text", text="move to bag; this is the second input"),
        dict(type="image_url", image_url={"url": "data:image/jpeg;base64," + base64.b64encode(payload).decode()}),
    ])])
    responses = iter(["invalid", '{"decision":"COMPLETE","updated_todo_list":"- [x] done"}'])
    cloud = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]))))
    client = LaViRAClient(la_base_url="http://unused", va_base_url="http://unused",
                          la_client=cloud, va_client=cloud)
    client.save_request_context = False
    client.demo_recorder = demo
    monkeypatch.setattr("gear_sonic.utils.inference.lavira.agent.time.sleep", lambda *_: None)

    def parse(completion):
        try:
            return json.loads(completion.choices[0].message.content)
        except ValueError:
            raise LaViRAAgentError("bad json")

    result = client._create(cloud, request_kind="la_decision", response_parser=parse, **request)
    assert result["decision"] == "COMPLETE"
    events = read_events(recorder.path)
    requests = [e for e in events if e["type"] == "demo_model_request"]
    replies = [e for e in events if e["type"] == "demo_model_response"]
    assert [e["attempt"] for e in requests] == [1, 2]
    assert replies[0]["error"] == "bad json" and replies[0]["content"] == "invalid"
    assert replies[1]["result"] == result and replies[1]["request_id"] == requests[1]["request_id"]
    artifact = Path(requests[0]["path"])
    saved = json.loads(artifact.read_text())
    content = saved["request"]["messages"][0]["content"]
    assert content[0]["text"] == request["messages"][0]["content"][0]["text"]
    assert (artifact.parent / content[1]["image_url"]["url"]["path"]).read_bytes() == payload
    assert len(list((artifact.parent / "model_images").glob("*.jpg"))) == 1


def test_actual_five_view_la_and_va_dual_inputs_keep_capture_provenance(tmp_path):
    from gear_sonic.experiments.agent import RecordedVisualClient
    from gear_sonic.utils.inference.lavira.agent import LaViRAClient, ScanView, MoveToView
    from gear_sonic.tests.test_lavira_agent import move, grounding
    responses = iter([move(), grounding()])
    cloud = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(responses))))]))))
    client = LaViRAClient(la_base_url="http://unused", va_base_url="http://unused",
                          la_client=cloud, va_client=cloud)
    client.save_request_context = False
    recorder = Recorder(profile(tmp_path))
    images = [np.full((32, 32, 3), i * 30, np.uint8) for i in range(6)]
    observations = {id(image): dict(captured_wall_ns=1000 + i, source_timestamp_ns=900 + i,
                                   stream="chest_view") for i, image in enumerate(images)}
    agent = SimpleNamespace(generation=1, _skill_id=3, recorder=recorder,
                            experiment=dict(demo=True, flexible_mission=True),
                            camera=SimpleNamespace(observations=observations))
    wrapper = RecordedVisualClient(client, agent)
    client.demo_recorder = DemoModelRecorder(recorder, wrapper.demo_context)
    views = [ScanView(1, direction, image, (0, 0, 0), 0) for direction, image in zip(
        ("front", "front_right", "right", "left", "front_left"), images)]
    wrapper.language_action(mission="bag then bin", navigation_mode="vln", global_target="basket",
                            current_step=2, todo_list="- [ ] next", scan_views=views,
                            move_to_views=[MoveToView(1, "door", images[-1], "reached", "SATISFIED")])
    wrapper.context.stage, wrapper.context.camera = "NAV", "chest"
    wrapper.grounding(mission="bag then bin", global_target="basket", strategic_goal="LA selected basket",
                      strategic_stop=False, target="basket", direction="front", image_bgr=images[0])
    events = read_events(recorder.path)
    requests = [e for e in events if e["type"] == "demo_model_request"]
    la = json.loads(Path(requests[0]["path"]).read_text())
    image_inputs = [c for c in la["request"]["messages"][1]["content"] if c["type"] == "image_url"]
    assert len(image_inputs) == 6
    assert [v["direction"] for v in la["views"][:5]] == [v.direction for v in views]
    assert la["views"][0]["observation"]["captured_wall_ns"] == 1000
    check = next(e for e in events if e["type"] == "check")
    assert check["inputs"]["target"] == "basket"
    assert check["inputs"]["strategic_goal"] == "LA selected basket"
    assert check["observation"]["source_timestamp_ns"] == 900
    assert requests[1]["stage"] == "NAV" and requests[1]["camera"] == "chest"


def test_compressed_cache_deduplicates_repeated_frame_and_spills_without_loss(tmp_path):
    take = DemoTake(tmp_path, start_event(), max_cache_bytes=1)
    first, second = jpeg(30), jpeg(80)
    for seq, data in [(0, first), (0, first), (1, first), (2, second)]:
        take.add("head", data, metadata(seq), wall_ns=2_000_000_000 + seq, mono_ns=1_000_000_000 + seq)
    manifest = take.finish(dict(monotonic_ns=3_000_000_000))
    value = json.loads(manifest.read_text())
    assert value["streams"]["head"]["frames"] == 3
    assert value["streams"]["head"]["jpeg_bytes"] == len(first) + len(second)
    assert value["cache_spills"] == 2
    assert (manifest.parent / "head.mjpeg").read_bytes() == first + second
    rows = [json.loads(line) for line in (manifest.parent / "head.jsonl").read_text().splitlines()]
    assert rows[0]["offset"] == rows[1]["offset"] == 0
    assert rows[2]["offset"] == len(first)
    assert not take.caches


def test_export_burns_mask_and_line_into_both_cameras_preserving_duration_and_gaps(tmp_path):
    take = DemoTake(tmp_path, start_event())
    state = NavigationViewerState(active=True, generation=1, camera_stream="camera/ego_view",
        target_camera_stream="camera/ego_view", yaw_align_geometry_camera_stream="camera/chest_view",
        target_bbox_xyxy=(5, 40, 45, 85), target_lateral_anchor_px=(25, 65),
        yaw_align_edge_endpoints_px=((55, 60), (115, 60)), overlay_image_size=(128, 96),
        completed_yaw_align_target_mask_row_spans=tuple((row, 60, 110) for row in range(65, 90)))
    take.overlay(state, 1_000_000_000)
    for name in ("head", "chest"):
        take.add(name, jpeg(), metadata(1), wall_ns=2_000_000_000, mono_ns=1_000_000_000)
    manifest = take.finish(dict(monotonic_ns=3_000_000_000))
    rendered = render_take(manifest)
    assert rendered["status"] == "rendered"
    assert set(rendered["streams"]) == {"head", "chest"}
    images = {}
    for name in ("head", "chest"):
        with av.open(str(manifest.parent / f"{name}.mp4")) as video:
            frames = list(video.decode(video=0))
            assert len(frames) == 60
            assert float(video.duration / av.time_base) == pytest.approx(2, abs=0.04)
            images[name] = frames[5].to_ndarray(format="bgr24")
            assert frames[-1].to_ndarray(format="bgr24")[75, 90].max() < 10
    assert images["head"][60, 5, 1] > 130  # Green bbox, not a plain camera recording.
    assert images["chest"][78, 85, 0] > 75  # Mask fill above the source image's 45.
    assert np.linalg.norm(images["chest"][60, 85].astype(float) - 45) > 80  # Alignment line.
    # Source cache is still unannotated and recoverable; export can be repeated.
    assert (manifest.parent / "head.mjpeg").read_bytes() == jpeg()
    assert render_take(manifest)["status"] == "rendered"


def test_old_manifest_does_not_reexport_navigation_previews(tmp_path):
    take = DemoTake(tmp_path, start_event())
    for name in ("head", "chest"):
        take.add(name, jpeg(), metadata(1), wall_ns=2_000_000_000, mono_ns=1_000_000_000)
    path = take.finish(dict(monotonic_ns=1_100_000_000))
    manifest = json.loads(path.read_text())
    for name in ("navdp_actor_ray", "navdp_slam_2d"):
        # No source files: a renderer that still tries to read them would fail.
        manifest["streams"][name] = dict(frames=3, status="error")
    path.write_text(json.dumps(manifest))
    rendered = render_take(path)
    assert rendered["status"] == "rendered"
    assert {p.name for p in path.parent.glob("*.mp4")} == {"head.mp4", "chest.mp4"}
    assert not list(path.parent.glob("navdp_*"))


def test_five_minute_take_keeps_only_compressed_frames_and_compact_indexes(tmp_path):
    take = DemoTake(tmp_path, start_event())
    payloads = [jpeg(30), jpeg(70)]
    for tick in range(30 * 300):
        for name in ("head", "chest"):
            take.add(name, payloads[tick % 2], metadata(tick), wall_ns=2_000_000_000 + tick * 33_333_333,
                     mono_ns=1_000_000_000 + tick * 33_333_333)
    assert take.spills == 0
    for cache in take.caches.values():
        assert len(cache.data) == sum(map(len, payloads)) * 4500
        assert cache.index.itemsize == 8 and len(cache.index) == 9000 * 7
    manifest = take.finish(dict(monotonic_ns=301_000_000_000))
    data = json.loads(manifest.read_text())
    assert all(stream["frames"] == 9000 for stream in data["streams"].values())


def test_trial_restart_exports_each_take_and_background_queue_owns_no_frames(tmp_path, monkeypatch):
    rendered = []
    def render(path):
        value = json.loads(path.read_text())
        rendered.append(value["trial_id"])
        return dict(trial_id=value["trial_id"], status="rendered")
    monkeypatch.setattr("gear_sonic.experiments.demo_video.render_take", render)
    recorder = Recorder(profile(tmp_path))
    session = DemoSession(profile(tmp_path))
    try:
        for generation in (1, 2):
            recorder.write("trial_start", generation=generation)
            session.poll(NavigationViewerState())
            assert session.take.start["generation"] == generation
            recorder.write("trial_end", generation=generation)
            session.poll(NavigationViewerState())
            assert session.take is None
    finally:
        session.close()
    assert rendered == ["demo-test-g1", "demo-test-g2"]
    # A restarted recorder must not overwrite or regenerate completed trials.
    session = DemoSession(profile(tmp_path))
    session.poll(NavigationViewerState())
    session.close()
    assert len(rendered) == 2


def test_event_tail_waits_for_complete_json_line(tmp_path):
    path = tmp_path / "events.jsonl"
    tail = EventTail(path)
    path.write_text('{"type":"trial_start",')
    assert tail.poll() == []
    with path.open("a") as handle:
        handle.write('"trial_id":"one"}\n')
    assert tail.poll() == [dict(type="trial_start", trial_id="one")]
    assert tail.poll() == []


def test_capture_service_uses_encoded_streams_and_saves_overlays_on_shutdown(tmp_path, monkeypatch):
    from gear_sonic.experiments import demo_video as module
    from gear_sonic.runtime.gateway.control import BASE_POSE_RUNTIME_STATUS_COMMAND
    p = profile(tmp_path)
    p.endpoint_uri = lambda name: "inproc://" + name
    recorder = Recorder(p)
    recorder.write("trial_start", generation=1)
    reads = []
    payload = jpeg()

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def read_snapshot(self, request, **kwargs):
            name = request.streams[0]
            reads.append(name)
            frame = metadata(len(reads))
            frame.attributes = {"encoding": "jpeg_bytes"}
            return SimpleNamespace(snapshot=SimpleNamespace(frames={name: frame}),
                                   arrays={name: np.frombuffer(payload, np.uint8)})

        def close(self):
            pass

    commands = iter([
        SimpleNamespace(name="start_base_pose", parameters={"generation": 1}),
        SimpleNamespace(name=BASE_POSE_RUNTIME_STATUS_COMMAND, parameters=dict(
            generation=1, state="motion", camera_stream="ego_view", viewer_overlay=dict(
                image_size=[128, 96], target_bbox_xyxy=[10, 10, 50, 50],
                yaw_align_edge_endpoints_px=[[20, 50], [80, 50]],
                completed_yaw_align_target_mask_row_spans=[[60, 20, 50]]))),
    ])
    class Subscriber:
        def __init__(self, *args, **kwargs):
            pass

        def read_command(self):
            return next(commands, None)

        def close(self):
            pass

    parents = iter([42, 42, 42, 43])
    monkeypatch.setattr(module.os, "getppid", lambda: next(parents))
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(module, "SensorGatewayClient", Client)
    monkeypatch.setattr(module, "ControlGatewaySubscriber", Subscriber)
    monkeypatch.setattr(module, "render_take", lambda path: dict(trial_id="demo-test-g1", status="rendered"))
    module.run(p)
    assert set(reads) == {"camera_encoded/ego_view", "camera_encoded/chest_view"}
    root = tmp_path / "demo/demo-test-g1/video"
    assert (root / "head.mjpeg").read_bytes() == payload
    overlays = [json.loads(line)["state"] for line in (root / "overlays.jsonl").read_text().splitlines()]
    assert overlays[-1]["completed_yaw_align_target_mask_row_spans"] == [[60, 20, 50]]
    assert overlays[-1]["yaw_align_edge_endpoints_px"] == [[20, 50], [80, 50]]
    assert json.loads((root / "manifest.json").read_text())["end"]["reason"] == "recorder_shutdown"
    with pytest.raises(ValueError, match="--demo"):
        module.run(profile(tmp_path, demo=False))
