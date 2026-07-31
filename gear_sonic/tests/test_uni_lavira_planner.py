"""Tests for the Uni-LaViRA JSON-to-Sonic planner state machine."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest

from gear_sonic.utils.inference.uni_lavira_planner import (
    CommandValidationError,
    UniLaviraPlannerExecutor,
    validate_object_nav_batch,
)


def payload(*, wz=0.4, rotate_for=1.0, vx=0.3, vy=0.0, walk_for=2.0):
    return {"commands": [
        {"vx": 0.0, "vy": 0.0, "wz": wz, "duration": rotate_for},
        {"vx": vx, "vy": vy, "wz": 0.0, "duration": walk_for},
    ]}


class ValidationTests(unittest.TestCase):
    def test_accepts_rotate_then_translate_batch(self):
        batch = validate_object_nav_batch(payload())
        self.assertEqual(batch.rotation.wz, 0.4)
        self.assertEqual(batch.translation.vx, 0.3)

    def test_rejects_invalid_shapes_and_values(self):
        cases = [
            [],
            {},
            {"commands": []},
            {"commands": [
                {"vx": 0.0, "vy": 0.0, "wz": 0.4},
                {"vx": 0.3, "vy": 0.0, "wz": 0.0, "duration": 1.0},
            ]},
            payload(wz=True),
            payload(wz=float("nan")),
            payload(vx=float("inf")),
        ]
        for candidate in cases:
            with self.subTest(candidate=candidate):
                with self.assertRaises(CommandValidationError):
                    validate_object_nav_batch(candidate)

    def test_rejects_unsafe_or_mixed_commands(self):
        cases = [
            payload(rotate_for=-0.1),
            {"commands": [
                {"vx": 0.1, "vy": 0.0, "wz": 0.4, "duration": 1.0},
                {"vx": 0.3, "vy": 0.0, "wz": 0.0, "duration": 1.0},
            ]},
            {"commands": [
                {"vx": 0.0, "vy": 0.0, "wz": 0.4, "duration": 1.0},
                {"vx": 0.3, "vy": 0.0, "wz": 0.1, "duration": 1.0},
            ]},
            payload(vx=0.51),
            payload(walk_for=30.01),
            payload(wz=1.0, rotate_for=3.2),
        ]
        for candidate in cases:
            with self.subTest(candidate=candidate):
                with self.assertRaises(CommandValidationError):
                    validate_object_nav_batch(candidate)


class ExecutorTests(unittest.TestCase):
    def test_runs_rotation_pause_translation_and_stop(self):
        executor = UniLaviraPlannerExecutor(transition_pause=0.5)
        executor.start(payload(), now=10.0)

        rotating = executor.tick(10.2)
        self.assertEqual(rotating.phase, "rotating")
        self.assertEqual(rotating.mode, 1)
        self.assertEqual(rotating.movement, (0.0, 0.0, 0.0))
        self.assertEqual(rotating.speed, 0.0)

        paused = executor.tick(11.1)
        self.assertEqual(paused.phase, "transition_pause")
        self.assertEqual(paused.speed, 0.0)

        walking = executor.tick(11.6)
        self.assertEqual(walking.phase, "translating")
        self.assertAlmostEqual(walking.speed, 0.3)
        self.assert_sequence_almost_equal(walking.movement, walking.facing)

        stopped = executor.tick(13.6)
        self.assertEqual(stopped.phase, "stopped")
        self.assertEqual(stopped.speed, 0.0)
        self.assertTrue(executor.just_completed)
        self.assertFalse(executor.active)

    def test_accumulates_signed_relative_yaw(self):
        executor = UniLaviraPlannerExecutor()
        executor.start(payload(), now=0.0)
        executor.tick(3.6)
        self.assertAlmostEqual(executor.heading_rad, 0.4)

        executor.start(payload(wz=-0.2), now=4.0)
        output = executor.tick(4.1)
        self.assertAlmostEqual(executor.heading_rad, 0.2)
        self.assert_sequence_almost_equal(
            output.facing, (math.cos(0.2), math.sin(0.2), 0.0)
        )

    def test_rotates_local_lateral_velocity(self):
        executor = UniLaviraPlannerExecutor()
        executor.start(
            payload(wz=math.pi / 2, rotate_for=1.0, vx=0.0, vy=0.3),
            now=0.0,
        )
        walking = executor.tick(1.6)
        self.assertEqual(walking.phase, "translating")
        self.assert_sequence_almost_equal(walking.movement, (-1.0, 0.0, 0.0))
        self.assertAlmostEqual(walking.speed, 0.3)

    def test_zero_rotation_still_has_half_second_pause(self):
        executor = UniLaviraPlannerExecutor(transition_pause=0.5)
        executor.start(payload(wz=0.0, rotate_for=0.0), now=2.0)
        self.assertEqual(executor.tick(2.1).phase, "transition_pause")
        self.assertEqual(executor.tick(2.5).phase, "translating")

    def test_abort_and_reset_heading_are_stopped(self):
        executor = UniLaviraPlannerExecutor()
        executor.start(payload(), now=0.0)
        stopped = executor.abort("operator_stop")
        self.assertEqual(stopped.phase, "stopped")
        self.assertEqual(stopped.speed, 0.0)
        self.assertFalse(executor.active)
        self.assertAlmostEqual(executor.heading_rad, 0.4)
        executor.reset_heading()
        self.assertEqual(executor.heading_rad, 0.0)
        self.assertEqual(executor.tick(1.0).facing, (1.0, 0.0, 0.0))

    def test_forward_fields_match_keyboard_controller(self):
        keyboard_class = self._load_keyboard_controller()
        keyboard = keyboard_class(max_speed=0.3)
        keyboard.facing_angle = 0.4
        expected = keyboard.update_from_keys({"w"})

        executor = UniLaviraPlannerExecutor(transition_pause=0.5)
        executor.start(payload(), now=0.0)
        actual = executor.tick(1.5)

        self.assertEqual(actual.mode, expected[0])
        self.assert_sequence_almost_equal(actual.movement, expected[1])
        self.assert_sequence_almost_equal(actual.facing, expected[2])
        self.assertAlmostEqual(actual.speed, expected[3])
        self.assertEqual(actual.height, expected[4])

    def assert_sequence_almost_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value)

    @staticmethod
    def _load_keyboard_controller():
        sender_name = "gear_sonic.utils.teleop.zmq.zmq_planner_sender"
        fake_sender = types.ModuleType(sender_name)
        fake_sender.build_planner_message = lambda *args, **kwargs: b""
        replacements = {
            "tyro": types.ModuleType("tyro"),
            "zmq": types.ModuleType("zmq"),
            sender_name: fake_sender,
        }
        previous = {name: sys.modules.get(name) for name in replacements}
        sys.modules.update(replacements)
        script = Path(__file__).resolve().parents[1] / "scripts" / "keyboard_planner_thread_server.py"
        try:
            spec = importlib.util.spec_from_file_location("keyboard_equivalence", script)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module.PlannerKeyboardController
        finally:
            sys.modules.pop("keyboard_equivalence", None)
            for name, old_module in previous.items():
                if old_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module


if __name__ == "__main__":
    unittest.main()
