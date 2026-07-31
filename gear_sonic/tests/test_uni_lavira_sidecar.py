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

    def poll(self, _timeout=0):
        return bool(self.sent)

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        return self.sent.pop(0)


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

    def test_action_subscriber_tracker_requires_command_and_planner(self):
        tracker = sidecar.ActionSubscriberTracker()
        self.assertFalse(tracker.ready)

        tracker.observe(b"\x01command")
        self.assertFalse(tracker.ready)
        tracker.observe(b"\x01planner")
        self.assertTrue(tracker.ready)

        tracker.observe(b"\x00planner")
        self.assertFalse(tracker.ready)

    def test_drains_xpub_subscription_events(self):
        tracker = sidecar.ActionSubscriberTracker()
        socket = FakePublisher()
        socket.sent.extend([b"\x01command", b"\x01planner"])

        count = sidecar.drain_action_subscriptions(socket, tracker)

        self.assertEqual(count, 2)
        self.assertTrue(tracker.ready)

    def test_keyboard_is_ignored_until_cpp_subscribers_are_ready(self):
        bridge = UniLaviraJsonBridge(FakeRepSocket(), UniLaviraPlannerExecutor())
        tracker = PlannerModeTracker()
        publisher = FakePublisher()

        sidecar.drain_keyboard_commands(
            FakeKeyboardSocket(["k"]), tracker, bridge, publisher,
            action_ready=False,
        )

        self.assertFalse(tracker.running)
        self.assertEqual(publisher.sent, [])

    def test_deploy_disconnect_aborts_and_requires_fresh_k(self):
        rep = FakeRepSocket([payload()])
        bridge = UniLaviraJsonBridge(rep, UniLaviraPlannerExecutor())
        tracker = PlannerModeTracker()
        tracker.apply("k")
        bridge.step(now=1.0, planner_ready=True)
        publisher = FakePublisher()

        with mock.patch.object(
            sidecar, "build_command_message", return_value=b"stop-command"
        ) as command_builder, mock.patch.object(
            sidecar, "build_planner_message", return_value=b"stopped-planner"
        ):
            sidecar.handle_deploy_disconnect(tracker, bridge, publisher)

        command_builder.assert_called_once_with(
            start=False, stop=True, planner=True
        )
        self.assertFalse(tracker.running)
        self.assertEqual(tracker.mode, "OFF")
        self.assertFalse(tracker.planner_ready)
        self.assertEqual(
            rep.sent[-1],
            {"status": "aborted", "reason": "deploy_disconnected"},
        )
        self.assertEqual(
            publisher.sent,
            [b"stop-command", b"stopped-planner"],
        )

    def test_config_targets_cpp_action_port_without_planner_relay(self):
        config = sidecar.UniLaviraPlannerConfig()

        self.assertEqual(config.action_host, "*")
        self.assertEqual(config.action_port, 5556)
        self.assertFalse(hasattr(config, "planner_port"))

    def test_keyboard_start_publishes_cpp_planner_command(self):
        bridge = UniLaviraJsonBridge(FakeRepSocket(), UniLaviraPlannerExecutor())
        tracker = PlannerModeTracker()
        publisher = FakePublisher()

        with mock.patch.object(
            sidecar, "build_command_message", return_value=b"command-bytes"
        ) as builder:
            stopped = sidecar.drain_keyboard_commands(
                FakeKeyboardSocket(["k"]), tracker, bridge, publisher
            )

        self.assertIsNone(stopped)
        self.assertTrue(tracker.planner_ready)
        builder.assert_called_once_with(start=True, stop=False, planner=True)
        self.assertEqual(publisher.sent, [b"command-bytes"])

    def test_keyboard_stop_preserves_planner_mode_bit(self):
        bridge = UniLaviraJsonBridge(FakeRepSocket(), UniLaviraPlannerExecutor())
        tracker = PlannerModeTracker()
        tracker.apply("k")
        publisher = FakePublisher()

        with mock.patch.object(
            sidecar, "build_command_message", return_value=b"stop-bytes"
        ) as builder:
            sidecar.drain_keyboard_commands(
                FakeKeyboardSocket(["k"]), tracker, bridge, publisher
            )

        builder.assert_called_once_with(start=False, stop=True, planner=True)
        self.assertFalse(tracker.running)
        self.assertEqual(tracker.mode, "OFF")

    def test_i_safely_stops_without_pose_producer(self):
        rep = FakeRepSocket([payload()])
        bridge = UniLaviraJsonBridge(rep, UniLaviraPlannerExecutor())
        tracker = PlannerModeTracker()
        tracker.apply("k")
        bridge.step(now=1.0, planner_ready=True)
        publisher = FakePublisher()

        with mock.patch.object(
            sidecar, "build_command_message", return_value=b"stop-bytes"
        ) as builder:
            stopped = sidecar.drain_keyboard_commands(
                FakeKeyboardSocket(["i"]), tracker, bridge, publisher
            )

        builder.assert_called_once_with(start=False, stop=True, planner=True)
        self.assertFalse(tracker.running)
        self.assertEqual(tracker.mode, "OFF")
        self.assertEqual(stopped.phase, "stopped")
        self.assertEqual(
            rep.sent[-1], {"status": "aborted", "reason": "control_stopped"}
        )

    def test_keyboard_i_aborts_active_request(self):
        rep = FakeRepSocket([payload()])
        bridge = UniLaviraJsonBridge(rep, UniLaviraPlannerExecutor())
        tracker = PlannerModeTracker()
        tracker.apply("k")
        bridge.step(now=1.0, planner_ready=tracker.planner_ready)

        stopped = sidecar.drain_keyboard_commands(
            FakeKeyboardSocket(["i"]), tracker, bridge, FakePublisher()
        )

        self.assertFalse(tracker.planner_ready)
        self.assertFalse(tracker.running)
        self.assertIsNotNone(stopped)
        self.assertEqual(stopped.phase, "stopped")
        self.assertEqual(
            rep.sent[-1],
            {"status": "aborted", "reason": "control_stopped"},
        )

    def test_drains_multiple_keyboard_commands_in_order(self):
        bridge = UniLaviraJsonBridge(FakeRepSocket(), UniLaviraPlannerExecutor())
        tracker = PlannerModeTracker()

        stopped = sidecar.drain_keyboard_commands(
            FakeKeyboardSocket(["k", "i", "o"]),
            tracker, bridge, FakePublisher()
        )

        self.assertIsNone(stopped)
        self.assertFalse(tracker.running)
        self.assertFalse(tracker.planner_ready)

    def test_abort_reply_failure_propagates_fail_closed(self):
        class FailingSendRep(FakeRepSocket):
            def send_json(self, value):
                raise RuntimeError("send failed")

        rep = FailingSendRep([payload()])
        bridge = UniLaviraJsonBridge(rep, UniLaviraPlannerExecutor())
        bridge.step(now=1.0, planner_ready=True)

        with self.assertRaisesRegex(RuntimeError, "send failed"):
            sidecar._best_effort_abort(bridge, "operator_stop")

        self.assertTrue(bridge.pending_reply)
        self.assertFalse(bridge.executor.active)

    def test_immediate_rejection_reply_failure_propagates_fatal(self):
        class FailingSendRep(FakeRepSocket):
            def send_json(self, value):
                raise RuntimeError("send failed")

        bridge = UniLaviraJsonBridge(
            FailingSendRep([{"commands": []}]),
            UniLaviraPlannerExecutor(),
        )

        with self.assertRaisesRegex(RuntimeError, "send failed"):
            sidecar.step_bridge_once(
                bridge, planner_ready=True, action_socket=FakePublisher(), now=1.0
            )

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

