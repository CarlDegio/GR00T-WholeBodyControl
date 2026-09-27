"""Backends share one request lane and never share a ZMQ client across threads."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from gear_sonic.utils.inference.vla.backends import Backend, PolicyRouter


def test_switch_with_inflight_request_keeps_backend_and_drops_obsolete_queued_request():
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    calls = []
    epoch = [1]

    class Client:
        def __init__(self, host, port, **kw):
            self.host, self.thread = host, threading.get_ident()
        def get_action(self, obs):
            assert self.thread == threading.get_ident()
            calls.append((self.host, obs))
            if obs == "old_inflight":
                entered.set()
                assert release.wait(5)
            return self.host
        def ping(self, **kw):
            assert self.thread == threading.get_ident()
            return True

    router = PolicyRouter({"a": Backend("a", 1, "pick"), "b": Backend("b", 2, "place")}, "a", Client)
    def run(key, obs, generation):
        if obs == "old_queued":
            waiting.set()
        return router.infer(key, obs, current=lambda: epoch[0] == generation,
                            run=lambda client, value: client.get_action(value))
    with ThreadPoolExecutor(3) as pool:
        old = pool.submit(run, "a", "old_inflight", 1)
        assert entered.wait(5)
        queued = pool.submit(run, "a", "old_queued", 1)
        assert waiting.wait(5)
        epoch[0] = 2
        router.select("b")
        new = pool.submit(run, router.snapshot(), "new", 2)
        # No second backend request can pass the existing in-flight call.
        assert calls == [("a", "old_inflight")]
        release.set()
        assert old.result(5) == "a"
        assert queued.result(5) is None
        assert new.result(5) == "b"
    assert calls == [("a", "old_inflight"), ("b", "new")]
    assert router.ping()  # Uses a separate main-thread client.


def test_prewarm_each_model_uses_its_prompt_and_never_publishes_actions():
    calls = []
    class Client:
        def __init__(self, host, port, **kw): self.host = host
        def get_action(self, obs): calls.append((self.host, obs["language"]))
    router = PolicyRouter({"a": Backend("a", 1, "pick"), "b": Backend("b", 2, "place")},
                          "a", Client, require_warmup=True)
    with pytest.raises(ValueError, match="prewarmed"):
        router.select("a")
    observation = {"language": {"annotation.human.task_description": [["old"]]}}
    router.warmup(observation, stopped=lambda: False)
    assert calls == [("a", {"annotation.human.task_description": [["pick"]]}),
                     ("b", {"annotation.human.task_description": [["place"]]})]
    assert observation["language"]["annotation.human.task_description"] == [["old"]]
    assert router.select("b") == "b"
    with pytest.raises(ValueError, match="unknown"):
        router.select("typo")


def test_smoke_disables_all_backend_connections_even_without_prewarm():
    def forbidden_client(*args, **kwargs):
        pytest.fail("smoke must never create a policy client")

    router = PolicyRouter({"a": Backend("127.0.0.1", 29999, "pick")}, "a",
                          forbidden_client, disabled=True)
    assert not router.ready()
    assert not router.ping()
    router.warmup({"language": {}}, stopped=lambda: False)
    assert not router.warmed
    with pytest.raises(ValueError, match="disabled"):
        router.select("a")
    with pytest.raises(ValueError, match="disabled"):
        router.infer("a", {}, current=lambda: True, run=lambda client, obs: client.get_action(obs))
    router.close()
