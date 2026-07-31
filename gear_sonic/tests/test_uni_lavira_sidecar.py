"""Tests for the standalone Uni-LaViRA JSON planner sidecar."""
from __future__ import annotations

from unittest import mock
import unittest

from gear_sonic.scripts import uni_lavira_planner_thread_server as sidecar
from gear_sonic.tests.test_uni_lavira_bridge import FakeRepSocket
from gear_sonic.tests.test_uni_lavira_planner import payload
from gear_sonic.utils.inference.uni_lavira_planner import (
    PlannerModeTracker,
    UniLaviraJsonBridge,
    UniLaviraPlannerExecutor,
)


class FakeKeyboardSocket:
    def __init__(self, messages=()):
        self.messages = list(messages)

    def poll(self, _timeout=0):
        return bool(self.messages)

    def recv_string(self):
        return self.messages.pop(0)


class FakePublisher:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


class SidecarHelperTests(unittest.TestCase):
    def test_publish_output_uses_exact_planner_fields(self):
        output = UniLaviraPlannerExecutor().tick(0.0)
        publisher = FakePublisher()
        with mock.patch.object(
            sidecar, "build_planner_message", return_value=b"planner-bytes"
        ) as builder:
            sidecar.publish_output(publisher, output)

        builder.assert_called_once_with(
            output.mode,
            output.movement,
            output.facing,
            speed=output.speed,
            height=output.height,
        )
        self.assertEqual(publisher.sent, [b"planner-bytes"])

    def test_keyboard_pose_aborts_active_request(self):
        rep = FakeRepSocket([payload()])
        bridge = UniLaviraJsonBridge(rep, UniLaviraPlannerExecutor())
        tracker = PlannerModeTracker()
        tracker.apply("k")
        bridge.step(now=1.0, planner_ready=tracker.planner_ready)

        stopped = sidecar.drain_keyboard_commands(
            FakeKeyboardSocket(["i"]), tracker, bridge
        )

        self.assertFalse(tracker.planner_ready)
        self.assertIsNotNone(stopped)
        self.assertEqual(stopped.phase, "stopped")
        self.assertEqual(
            rep.sent[-1],
            {"status": "aborted", "reason": "pose_mode_requested"},
        )

    def test_drains_multiple_keyboard_commands_in_order(self):
        bridge = UniLaviraJsonBridge(FakeRepSocket(), UniLaviraPlannerExecutor())
        tracker = PlannerModeTracker()

        stopped = sidecar.drain_keyboard_commands(
            FakeKeyboardSocket(["k", "i", "o"]), tracker, bridge
        )

        self.assertIsNone(stopped)
        self.assertTrue(tracker.planner_ready)

    def test_completion_ack_follows_terminal_publish(self):
        rep = FakeRepSocket([payload(rotate_for=0.0, walk_for=0.0)])
        bridge = UniLaviraJsonBridge(rep, UniLaviraPlannerExecutor())
        tracker = PlannerModeTracker()
        tracker.apply("k")
        publisher = FakePublisher()

        bridge.step(now=0.0, planner_ready=tracker.planner_ready)
        stopped = bridge.step(now=0.5, planner_ready=tracker.planner_ready)
        self.assertEqual(rep.sent, [])

        with mock.patch.object(sidecar, "build_planner_message", return_value=b"stop"):
            sidecar.publish_output(publisher, stopped)
        bridge.acknowledge_output_published()

        self.assertEqual(publisher.sent, [b"stop"])
        self.assertEqual(rep.sent[-1]["status"], "completed")


if __name__ == "__main__":
    unittest.main()

