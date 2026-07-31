"""Lifecycle tests for the optional Uni-LaViRA JSON REP bridge."""
from __future__ import annotations

import unittest

from gear_sonic.tests.test_uni_lavira_planner import payload
from gear_sonic.utils.inference.uni_lavira_planner import (
    PlannerModeTracker,
    UniLaviraJsonBridge,
    UniLaviraPlannerExecutor,
)


class FakeRepSocket:
    def __init__(self, incoming=None):
        self.incoming = list(incoming or [])
        self.sent = []

    def poll(self, _timeout=0):
        return bool(self.incoming)

    def recv_json(self):
        value = self.incoming.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def send_json(self, value):
        self.sent.append(value)
class PlannerModeTrackerTests(unittest.TestCase):
    def test_tracks_operator_planner_lifecycle(self):
        tracker = PlannerModeTracker()
        self.assertFalse(tracker.running)
        self.assertEqual(tracker.mode, "OFF")
        self.assertFalse(tracker.planner_ready)

        self.assertIsNone(tracker.apply("k"))
        self.assertTrue(tracker.running)
        self.assertEqual(tracker.mode, "PLANNER")
        self.assertTrue(tracker.planner_ready)

        self.assertEqual(tracker.apply("i"), "pose_mode_requested")
        self.assertEqual(tracker.mode, "POSE")
        self.assertFalse(tracker.planner_ready)

        self.assertIsNone(tracker.apply("o"))
        self.assertTrue(tracker.planner_ready)
        self.assertEqual(tracker.apply(" K "), "control_stopped")
        self.assertFalse(tracker.running)
        self.assertEqual(tracker.mode, "OFF")

    def test_ignores_unrelated_commands_and_planner_while_off(self):
        tracker = PlannerModeTracker()
        for command in ("", "p", "prompt:new task", "o", "i"):
            with self.subTest(command=command):
                self.assertIsNone(tracker.apply(command))
                self.assertFalse(tracker.planner_ready)





class BridgeTests(unittest.TestCase):
    def test_rejects_request_outside_planner_without_output(self):
        socket = FakeRepSocket([payload()])
        bridge = UniLaviraJsonBridge(socket, UniLaviraPlannerExecutor())
        output = bridge.step(now=1.0, planner_ready=False)
        self.assertIsNone(output)
        self.assertEqual(
            socket.sent,
            [{"status": "rejected", "reason": "not_in_planner"}],
        )

    def test_replies_only_after_motion_completes(self):
        socket = FakeRepSocket([payload(rotate_for=0.1, walk_for=0.1)])
        bridge = UniLaviraJsonBridge(socket, UniLaviraPlannerExecutor())
        first = bridge.step(now=1.0, planner_ready=True)
        self.assertEqual(first.phase, "rotating")
        self.assertEqual(socket.sent, [])
        finished = bridge.step(now=1.7, planner_ready=True)
        self.assertEqual(finished.phase, "stopped")
        self.assertEqual(socket.sent, [])
        bridge.acknowledge_output_published()
        self.assertEqual(socket.sent[-1]["status"], "completed")
        self.assertAlmostEqual(socket.sent[-1]["heading_rad"], 0.04)

    def test_rejects_invalid_request_before_motion(self):
        socket = FakeRepSocket([{"commands": []}])
        bridge = UniLaviraJsonBridge(socket, UniLaviraPlannerExecutor())
        output = bridge.step(now=1.0, planner_ready=True)
        self.assertIsNone(output)
        self.assertEqual(socket.sent[-1]["status"], "rejected")
        self.assertEqual(socket.sent[-1]["reason"], "invalid_commands")

    def test_terminal_output_can_abort_before_publication_ack(self):
        socket = FakeRepSocket([payload(rotate_for=0.0, walk_for=0.0)])
        bridge = UniLaviraJsonBridge(socket, UniLaviraPlannerExecutor())
        bridge.step(now=1.0, planner_ready=True)
        stopped = bridge.step(now=1.5, planner_ready=True)
        self.assertEqual(stopped.phase, "stopped")
        self.assertEqual(socket.sent, [])

        bridge.abort("planner_publish_failed")
        self.assertEqual(
            socket.sent[-1],
            {"status": "aborted", "reason": "planner_publish_failed"},
        )

    def test_losing_planner_mode_aborts_pending_request(self):
        socket = FakeRepSocket([payload()])
        bridge = UniLaviraJsonBridge(socket, UniLaviraPlannerExecutor())
        bridge.step(now=1.0, planner_ready=True)
        stopped = bridge.step(now=1.1, planner_ready=False)
        self.assertEqual(stopped.phase, "stopped")
        self.assertEqual(stopped.speed, 0.0)
        self.assertEqual(
            socket.sent[-1],
            {"status": "aborted", "reason": "left_planner_mode"},
        )

    def test_reset_control_session_clears_accumulated_heading(self):
        socket = FakeRepSocket([payload(rotate_for=0.1, walk_for=0.1)])
        executor = UniLaviraPlannerExecutor()
        bridge = UniLaviraJsonBridge(socket, executor)
        bridge.step(now=0.0, planner_ready=True)
        bridge.step(now=0.7, planner_ready=True)
        self.assertAlmostEqual(executor.heading_rad, 0.04)
        bridge.reset_control_session()
        self.assertEqual(executor.heading_rad, 0.0)
        self.assertFalse(bridge.pending_reply)

    def test_stopped_output_is_retained_after_completion(self):
        socket = FakeRepSocket([payload(rotate_for=0.0, walk_for=0.0)])
        bridge = UniLaviraJsonBridge(socket, UniLaviraPlannerExecutor())
        bridge.step(now=0.0, planner_ready=True)
        bridge.step(now=0.5, planner_ready=True)
        held = bridge.step(now=1.0, planner_ready=True)
        self.assertEqual(held.phase, "stopped")
        self.assertEqual(held.movement, (0.0, 0.0, 0.0))
        self.assertEqual(held.speed, 0.0)


if __name__ == "__main__":
    unittest.main()
