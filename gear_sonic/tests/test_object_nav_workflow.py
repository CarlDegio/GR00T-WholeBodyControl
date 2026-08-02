"""Tests for ObjectNav perception and the N-triggered workflow."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.scripts import controlled_uni_lavira_planner_thread_server as controlled
from gear_sonic.scripts import launch_object_nav_vla as workflow
from gear_sonic.scripts import run_vla_inference as vla
from gear_sonic.utils.inference.object_nav import (
    ComposedRGBDCamera,
    ObjectNavConfig,
    ObjectNavResult,
    ObjectNavRunner,
    RGBDSnapshot,
)
from gear_sonic.utils.inference.object_nav_geometry import (
    build_object_nav_commands_from_frames,
)
from gear_sonic.utils.inference.uni_lavira_planner import (
    PlannerModeTracker,
    UniLaviraJsonBridge,
    UniLaviraPlannerExecutor,
)


def policy(*, action: str = "NAVIGATE", confidence: float = 0.9):
    boxable = action == "NAVIGATE"
    return {
        "visual_check": "target visible",
        "action": action,
        "bbox_2d": [450, 450, 550, 550] if boxable else None,
        "target": "chair",
        "target_type": "global_target",
        "estimated_distance_m": 2.5 if boxable else None,
        "target_center_normalized": [500.0, 500.0] if boxable else None,
        "target_center_pixel": [1.0, 1.0] if boxable else None,
        "horizontal_offset_pixel": 0.0 if boxable else None,
        "camera_bearing_deg": 0.0 if boxable else None,
        "rotation_direction": "CENTERED" if boxable else None,
        "rotation_angle_deg": 0.0 if boxable else None,
        "confidence": confidence,
        "distance_confidence": 0.7,
        "stop_reasoning": "reached" if action == "STOP" else "",
    }


def snapshot(index: int = 1, depth_mm: int = 2000) -> RGBDSnapshot:
    depth = np.full((5, 5), depth_mm, dtype=np.uint16)
    return RGBDSnapshot(
        rgb_bgr=np.full((5, 5, 3), index, dtype=np.uint8),
        depth_raw=depth,
        depth_mm=depth.astype(np.float32),
        fx=100.0,
        fy=100.0,
        cx=2.0,
        cy=2.0,
        depth_scale_m=0.001,
        depth_aligned_to="chest_view",
        timestamp=float(index),
    )


class CameraAndGeometryTests(unittest.TestCase):
    def test_decodes_repository_composed_rgbd_protocol(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        depth = np.array([[1000, 2000, 3000], [4000, 5000, 6000]], dtype=np.uint16)
        payload = ImageMessageSchema(
            timestamps={"chest_view": 12.0, "chest_view_depth": 12.0},
            images={"chest_view": rgb, "chest_view_depth": depth},
            camera_info={
                "chest_view": {
                    "fx": 100.0,
                    "fy": 101.0,
                    "cx": 1.0,
                    "cy": 0.5,
                    "width": 3,
                    "height": 2,
                    "depth_scale_m": 0.001,
                    "depth_aligned_to": "chest_view",
                }
            },
        ).serialize()

        restored = ComposedRGBDCamera.decode_payload(payload)

        np.testing.assert_array_equal(restored.depth_raw, depth)
        np.testing.assert_allclose(restored.depth_mm, depth.astype(np.float32))
        self.assertTrue(np.all(restored.rgb_bgr[:, :, 2] > 200))
        self.assertEqual(restored.timestamp, 12.0)

    def test_selected_zero_margin_and_eight_metre_limit(self):
        frames = [(np.full((5, 5), 2000, dtype=np.float32), 100.0, 2.0)] * 5
        commands, geometry = build_object_nav_commands_from_frames(policy(), frames)
        self.assertEqual(commands["commands"][1]["duration"], 6.667)
        self.assertEqual(geometry["travel"], 2.0)

        too_far = [(np.full((5, 5), 8100, dtype=np.float32), 100.0, 2.0)] * 5
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_object_nav_commands_from_frames(policy(), too_far)


class RunnerTests(unittest.TestCase):
    def _run(self, policy_value):
        camera = mock.Mock()
        camera.capture_aligned_rgbd.side_effect = [snapshot(index) for index in range(1, 6)]
        client = mock.Mock()
        client.locate.return_value = policy_value
        temporary = tempfile.TemporaryDirectory()
        runner = ObjectNavRunner(
            ObjectNavConfig(
                mission="find the chair",
                global_target="chair",
                output_root=temporary.name,
            ),
            camera=camera,
            policy_client=client,
        )
        result = runner.run_once()
        return result, camera, temporary

    def test_navigate_captures_five_frames_and_builds_commands(self):
        result, camera, temporary = self._run(policy())
        self.addCleanup(temporary.cleanup)
        self.assertEqual(result.outcome, "NAVIGATE")
        self.assertEqual(len(result.commands["commands"]), 2)
        self.assertEqual(camera.capture_aligned_rgbd.call_count, 5)

    def test_stop_is_distinct_from_low_confidence(self):
        stopped, _, temporary_stop = self._run(policy(action="STOP"))
        rejected, _, temporary_rejected = self._run(policy(confidence=0.2))
        self.addCleanup(temporary_stop.cleanup)
        self.addCleanup(temporary_rejected.cleanup)
        self.assertEqual(stopped.outcome, "STOP")
        self.assertEqual(rejected.outcome, "REJECTED")
        self.assertEqual(stopped.commands, {"commands": []})
        self.assertEqual(rejected.commands, {"commands": []})


class FakeControlSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    def poll(self, _timeout=0):
        return bool(self.messages)

    def recv_json(self):
        return self.messages.pop(0)

    def send_json(self, value):
        self.sent.append(value)


class FakePublisher:
    def __init__(self):
        self.sent = []

    def send(self, value):
        self.sent.append(value)


class ControlledPlannerTests(unittest.TestCase):
    def test_start_waits_for_subscribers_and_is_idempotent(self):
        tracker = PlannerModeTracker()
        bridge = UniLaviraJsonBridge(FakeControlSocket([]), UniLaviraPlannerExecutor())
        publisher = FakePublisher()

        not_ready = controlled.start_planner(
            tracker, bridge, publisher, action_ready=False
        )
        self.assertEqual(not_ready["status"], "not_ready")
        self.assertFalse(tracker.running)

        with mock.patch.object(controlled, "build_command_message", return_value=b"start"):
            ready = controlled.start_planner(
                tracker, bridge, publisher, action_ready=True
            )
            again = controlled.start_planner(
                tracker, bridge, publisher, action_ready=True
            )
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(again["status"], "ready")
        self.assertTrue(tracker.planner_ready)
        self.assertEqual(publisher.sent, [b"start"])

    def test_status_reports_busy_and_safe_shutdown_preserves_planner(self):
        tracker = PlannerModeTracker()
        bridge = UniLaviraJsonBridge(FakeControlSocket([]), UniLaviraPlannerExecutor())
        publisher = FakePublisher()
        status_socket = FakeControlSocket([{"op": "status"}])

        exit_mode = controlled.handle_control_request(
            status_socket,
            tracker,
            bridge,
            publisher,
            action_ready=True,
        )
        self.assertIsNone(exit_mode)
        self.assertTrue(status_socket.sent[-1]["action_ready"])

        shutdown_socket = FakeControlSocket([{"op": "shutdown"}])
        with mock.patch.object(
            controlled, "build_command_message", return_value=b"keep-planner"
        ) as build_command, mock.patch.object(controlled.base, "publish_output") as publish, mock.patch.object(
            controlled.time, "sleep"
        ):
            exit_mode = controlled.handle_control_request(
                shutdown_socket,
                tracker,
                bridge,
                publisher,
                action_ready=True,
            )
        self.assertEqual(exit_mode, "shutdown")
        self.assertEqual(
            shutdown_socket.sent[-1],
            {"status": "stopping", "planner_ready": True},
        )
        self.assertTrue(tracker.planner_ready)
        self.assertEqual(publisher.sent, [b"keep-planner"])
        build_command.assert_called_once_with(start=True, stop=False, planner=True)
        self.assertEqual(publish.call_count, 5)

    def test_handoff_aborts_busy_motion_and_keeps_planner(self):
        tracker = PlannerModeTracker()
        tracker.apply("k")
        tracker.apply("i")
        bridge_socket = FakeControlSocket([])
        bridge = UniLaviraJsonBridge(bridge_socket, UniLaviraPlannerExecutor())
        bridge.pending_reply = True
        bridge._has_output = True
        publisher = FakePublisher()
        handoff_socket = FakeControlSocket([{"op": "handoff"}])

        with mock.patch.object(
            controlled, "build_command_message", return_value=b"keep-planner"
        ) as build_command, mock.patch.object(controlled.base, "publish_output") as publish, mock.patch.object(
            controlled.time, "sleep"
        ):
            exit_mode = controlled.handle_control_request(
                handoff_socket,
                tracker,
                bridge,
                publisher,
                action_ready=True,
            )

        self.assertEqual(exit_mode, "handoff")
        self.assertEqual(
            handoff_socket.sent[-1],
            {"status": "handoff", "planner_ready": True},
        )
        self.assertEqual(
            bridge_socket.sent[-1],
            {"status": "aborted", "reason": "orchestrator_handoff"},
        )
        self.assertTrue(tracker.planner_ready)
        self.assertEqual(publisher.sent, [b"keep-planner"])
        build_command.assert_called_once_with(start=True, stop=False, planner=True)
        self.assertEqual(publish.call_count, 5)


class FakeKeyboardSocket:
    def __init__(self, messages):
        self.messages = list(messages)

    def setsockopt_string(self, *_args):
        pass

    def setsockopt(self, *_args):
        pass

    def connect(self, *_args):
        pass

    def poll(self, _timeout):
        return bool(self.messages)

    def recv_string(self):
        return self.messages.pop(0)

    def close(self):
        pass


class FakeKeyboardContext:
    def __init__(self, messages):
        self.socket_value = FakeKeyboardSocket(messages)

    def socket(self, _kind):
        return self.socket_value

    def term(self):
        pass


class WorkflowTests(unittest.TestCase):
    def config(self, mode="repeat-until-stop"):
        return workflow.ObjectNavVlaLaunchConfig(
            mission="find the chair",
            global_target="chair",
            navigation_mode=mode,
            attach=False,
        )

    def test_vla_command_inherits_running_planner(self):
        command = workflow._vla_command(self.config(), workflow.Path("/repo"))
        self.assertIn("--assume-cpp-planner-running", command)

    def test_vla_inherited_control_state_requires_only_i_then_p(self):
        self.assertEqual(vla.initial_cpp_control_state(True), (True, "PLANNER"))
        self.assertEqual(vla.initial_cpp_control_state(False), (False, "OFF"))

    def test_waits_for_case_insensitive_n(self):
        fake_context = FakeKeyboardContext(["x", "N"])
        with mock.patch.object(workflow.zmq, "Context", return_value=fake_context):
            workflow._wait_for_n(self.config())
        self.assertEqual(fake_context.socket_value.messages, [])

    def _run_with_results(self, results, mode="repeat-until-stop"):
        events = []
        fake_planner = mock.Mock(returncode=0)
        fake_runner = mock.Mock()
        fake_runner.run_once.side_effect = results

        with mock.patch.object(workflow.subprocess, "Popen", return_value=fake_planner), mock.patch.object(
            workflow, "_wait_for_n", side_effect=lambda _config: events.append("n")
        ), mock.patch.object(
            workflow, "_wait_for_action_subscribers"
        ), mock.patch.object(
            workflow, "_start_planner", side_effect=lambda _config: events.append("planner")
        ), mock.patch.object(
            workflow, "ObjectNavRunner", return_value=fake_runner
        ), mock.patch.object(
            workflow,
            "send_object_nav_commands",
            return_value={"status": "completed", "heading_rad": 0.0},
        ) as send, mock.patch.object(
            workflow,
            "_safe_release_planner",
            side_effect=lambda *_args: events.append("release") or True,
        ), mock.patch.object(
            workflow,
            "_start_vla_pane",
            side_effect=lambda *_args: events.append("vla"),
        ) as start_vla:
            code = workflow.run_workflow(self.config(mode))
        return code, events, send, start_vla, fake_runner

    def test_repeat_mode_runs_until_explicit_stop_then_hands_off(self):
        navigate = ObjectNavResult(
            "NAVIGATE", policy(), {"commands": [{}, {}]}, {}, "/tmp/one"
        )
        stopped = ObjectNavResult("STOP", policy(action="STOP"), {"commands": []}, {}, "/tmp/two")
        code, events, send, start_vla, fake_runner = self._run_with_results(
            [navigate, stopped]
        )
        self.assertEqual(code, 0)
        self.assertEqual(fake_runner.run_once.call_count, 2)
        send.assert_called_once()
        start_vla.assert_called_once()
        self.assertLess(events.index("n"), events.index("planner"))
        self.assertLess(events.index("release"), events.index("vla"))

    def test_once_mode_hands_off_after_first_completed_batch(self):
        navigate = ObjectNavResult(
            "NAVIGATE", policy(), {"commands": [{}, {}]}, {}, "/tmp/one"
        )
        code, _, send, start_vla, fake_runner = self._run_with_results(
            [navigate], mode="once"
        )
        self.assertEqual(code, 0)
        self.assertEqual(fake_runner.run_once.call_count, 1)
        send.assert_called_once()
        start_vla.assert_called_once()

    def test_failure_stops_planner_without_starting_vla(self):
        failed = ObjectNavResult(
            "FAILED", {}, {"commands": []}, {"status": "failed"}, "/tmp/fail", "camera"
        )
        code, events, _, start_vla, _ = self._run_with_results([failed])
        self.assertEqual(code, 1)
        self.assertIn("release", events)
        start_vla.assert_not_called()


if __name__ == "__main__":
    unittest.main()
