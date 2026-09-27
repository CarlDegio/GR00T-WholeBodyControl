"""Exercise the shared agent_full LA/VA path through both baseline entrypoints."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from baselinelavira import DEFAULT_PROFILE, build_profile
from gear_sonic.experiments.config import CONFIGS, resolve
from gear_sonic.utils.inference.lavira import service
from gear_sonic.utils.inference.lavira.agent import LaViRAClient


@pytest.fixture(params=[False, True], ids=["standalone", "experiment"])
def agent_pair(request, tmp_path, monkeypatch):
    def init_client(self, **settings):
        self.__dict__.update(settings)
        self.la_client = self.va_client = object()

    monkeypatch.setattr(LaViRAClient, "__init__", init_client)
    monkeypatch.setattr(service, "SensorGatewayRGBDCamera", lambda *a, **kw: SimpleNamespace())
    runtime = SimpleNamespace(**{name: (lambda *a, **kw: None) for name in (
        "submit_intent", "wait_status", "is_cancelled", "pop_failure", "report_agent_event", "report_todo",
    )})
    if request.param:
        payloads = [resolve(CONFIGS / name)[0] for name in ("full_vln.yaml", "baselinelavira.yaml")]
    else:
        payloads = [yaml.safe_load(DEFAULT_PROFILE.read_text()), build_profile()]
    agents = []
    for index, payload in enumerate(payloads):
        path = tmp_path / f"runtime_{index}.yaml"
        path.write_text(yaml.safe_dump(payload))
        agents.append(service._agent(service.load_lavira_config(str(path)), "inproc://unused", runtime))
    return agents


def _client(agent):
    return getattr(agent.client, "client", agent.client)


def _responses(monkeypatch, client, values):
    replies = iter(values)
    requests = []

    def create(_client, *, response_parser, **request):
        requests.append(request)
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(next(replies))),
        )])
        return response_parser(completion)

    monkeypatch.setattr(client, "_create", create)
    return requests


def _decision(todo, target="white rubbish bin", direction="front"):
    return dict(
        global_target="table with the blue basket", progress_analysis="Follow the next waypoint",
        updated_todo_list=todo, reasoning="Approach the requested landmark",
        decision="EXECUTE", skill="MOVE_TO", skill_args=dict(view_direction=direction, target=target),
        expected_postcondition=f"Reach {target}",
    )


def _grounding(description="white rubbish bin", found=True):
    return dict(
        mode="GROUNDING", status="FOUND" if found else "NOT_FOUND",
        bbox_2d=[400, 400, 600, 800] if found else None, point_2d=None,
        target_description=description, confidence=0.9,
    )


def test_all_la_va_requests_match_full_vln_with_iplanner_enabled(agent_pair, monkeypatch):
    image = np.zeros((100, 100, 3), np.uint8)
    views = [SimpleNamespace(direction=direction, image_bgr=image) for direction in
             ("front", "front_right", "right", "left", "front_left")]
    todo = "- [ ] Approach the white rubbish bin\n- [ ] Approach the refrigerator"
    completed = (
        '- [x] Approach the white rubbish bin Result: Runtime VA confirmed MOVE_TO target '
        '"white rubbish bin" as SATISFIED.\n- [ ] Approach the refrigerator'
    )
    outputs, requests = [], []
    for agent in agent_pair:
        client = _client(agent)
        assert type(client) is LaViRAClient
        requests.append(_responses(monkeypatch, client, [
            _decision(todo), _decision(completed, "refrigerator", "left"), _grounding(),
            dict(mode="ALIGN_GROUNDING", status="FOUND",
                 target=dict(name="blue basket", visible=True, bbox_2d=[200, 200, 800, 800], confidence=0.9),
                 yaw_align_target=dict(name="desk", visible=True, confidence=0.9),
                 visual_evidence="Basket and desk edge visible"),
            dict(mode="POSTCHECK", status="SATISFIED", transition="TASK_COMPLETE",
                 visual_evidence="Bottle released inside basket", confidence=0.9),
        ]))
        la_context = dict(mission="Go to the white rubbish bin, turn left, then approach the refrigerator",
                          navigation_mode="vln", scan_views=views, manipulation_prompt="Collect the pill bottle")
        first = client.language_action(**la_context, global_target=None, current_step=1, todo_list="",
                                       move_to_views=[], transition_result=None)
        second = client.language_action(
            **la_context, global_target=first["global_target"], current_step=2, todo_list=completed,
            move_to_views=[SimpleNamespace(image_bgr=image)],
            transition_result=dict(executed_skill="MOVE_TO", status="SATISFIED",
                                   transition="CONTINUE_NAVIGATION", visual_evidence="White bin reached"),
        )
        va_context = dict(mission=la_context["mission"], global_target=first["global_target"],
                          strategic_goal="Approach the white rubbish bin", strategic_stop=False,
                          image_bgr=image)
        grounded = client.grounding(**va_context, target="white rubbish bin", direction="front")
        alignment = client.alignment_grounding(manipulation_prompt="Collect the pill bottle",
                                               direction="front", image_bgr=image)
        post = client.postcheck(**va_context, expected_postcondition="Bottle inside basket", skill="MANIPULATE")
        outputs.append((first, second, grounded, alignment, post))

    assert type(agent_pair[0]) is type(agent_pair[1])
    assert requests[0] == requests[1]
    assert outputs[0] == outputs[1]
    assert [r["request_kind"] for r in requests[1]] == [
        "la_decision", "la_decision", "va_grounding", "va_align_grounding", "va_postcheck",
    ]
    la_text = json.dumps(requests[1][0]["messages"])
    assert "Never mark a MOVE_TO item complete yourself" in la_text
    assert "never create a standalone turn" in la_text
    assert "There is no separate initial planning call" in la_text
    assert "Do not choose another target" in json.dumps(requests[1][2]["messages"])


@pytest.mark.parametrize("found", [True, False], ids=["landmark_found", "landmark_missing"])
def test_move_to_keeps_la_target_and_shared_depth_geometry(agent_pair, monkeypatch, found):
    image = np.zeros((100, 100, 3), np.uint8)
    snapshots = []
    for depth_m in (1.9, 2.0, 2.1, 2.2, 2.3):
        depth = np.full((100, 100), 9000.0)
        depth[56:63, 47:54] = depth_m * 1000  # Bbox center: (50, 59).
        depth[77:84, 47:54] = 1000.0  # Former bottom-edge projection would select this instead.
        snapshots.append(SimpleNamespace(rgb_bgr=image, depth_mm=depth, fx=100.0, cx=50.0))
    result_view = SimpleNamespace(rgb_bgr=image, depth_mm=np.full((100, 100), 2000.0), fx=100.0, cx=50.0)
    results, submitted, requests = [], [], []
    for agent in agent_pair:
        agent.global_target = "table with the blue basket"
        client = _client(agent)
        # A VA description cannot replace the LA-requested landmark, including
        # when VA reports that the landmark is missing and sees only open floor.
        requests.append(_responses(monkeypatch, client, [_grounding("open area", found)] * 3))
        monkeypatch.setattr(agent, "_face_scan_direction", lambda *a: None)
        monkeypatch.setattr(agent, "_face_move_to_goal", lambda *a: None)
        monkeypatch.setattr(agent, "_before_nav_handoff", lambda *a: None)
        monkeypatch.setattr(agent, "_capture_va_burst", lambda *a: snapshots)
        monkeypatch.setattr(agent, "_capture_va_snapshot", lambda *a: result_view)
        camera = getattr(agent.camera, "camera", agent.camera)
        monkeypatch.setattr(camera, "capture_camera_aligned_rgbd", lambda **kw: result_view, raising=False)
        monkeypatch.setattr(agent, "_wait", lambda *a: {"state": "reached"})
        intents = []
        monkeypatch.setattr(agent, "submit_intent", lambda name, args, sink=intents: sink.append((name, args)))
        results.append(agent._move_to(1, 1, dict(view_direction="front", target="white rubbish bin"),
                                      "Approach the white rubbish bin"))
        submitted.append(intents)
        assert not agent._global_target_navigation_ready  # An intermediate landmark cannot unlock ALIGN.

    assert results[0] == results[1]
    assert submitted[0] == submitted[1]
    assert results[1].target == "white rubbish bin"
    expected_calls = 3 if found else 1
    assert [r["request_kind"] for r in requests[1]] == ["va_grounding"] * expected_calls
    for request in requests[1]:
        assert '"white rubbish bin"' in request["messages"][1]["content"][-1]["text"]
    if found:
        assert results[1].va_result == "SATISFIED"
        assert submitted[1][0][0] == "navigation_goal"
        assert submitted[1][0][1]["target"] == "white rubbish bin"
        assert submitted[1][0][1]["goal_base"] == pytest.approx([2.1, 0.0])
    else:
        assert results[1].va_result == "NOT_SATISFIED"
        assert submitted[1] == []
