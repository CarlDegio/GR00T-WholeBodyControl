import json
import unittest

from gear_sonic.scripts.navila_planner import (
    NavilaPlannerConfig,
    NavilaPlannerControl,
    build_keyboard_message,
    build_policy_data,
)
from gear_sonic.scripts.keyboard_planner_thread_server import (
    KeyboardPlannerConfig,
    build_navila_message,
    key_commands,
)


class NavilaPlannerControlTest(unittest.TestCase):
    def test_toggle_starts_and_pauses_planner(self) -> None:
        control = NavilaPlannerControl()

        self.assertFalse(control.active)
        self.assertTrue(control.toggle())
        self.assertFalse(control.toggle())

    def test_remote_stop_pauses_planner(self) -> None:
        control = NavilaPlannerControl(active=True)

        changed = control.accept_action("stop", (0.0, 0.0, 0.0), 0.5)

        self.assertTrue(changed)
        self.assertFalse(control.active)

    def test_next_inference_waits_for_action_duration(self) -> None:
        control = NavilaPlannerControl(active=True)

        control.accept_action("forward", (0.5, 0.0, 0.0), 1.5, now=10.0)

        self.assertFalse(control.can_infer(now=11.49))
        self.assertTrue(control.can_infer(now=11.5))

    def test_toggle_back_to_navila_clears_old_action_wait(self) -> None:
        control = NavilaPlannerControl(active=True)
        control.accept_action("forward", (0.5, 0.0, 0.0), 10.0, now=1.0)

        self.assertFalse(control.toggle())
        self.assertTrue(control.toggle())

        self.assertTrue(control.can_infer(now=1.1))

    def test_repeated_action_is_not_reported_twice(self) -> None:
        control = NavilaPlannerControl(active=True)

        self.assertTrue(control.accept_action("forward", (0.5, 0.0, 0.0), 1.0))
        self.assertFalse(control.accept_action("forward", (0.5, 0.0, 0.0), 1.0))
        self.assertTrue(control.accept_action("turn_left", (0.0, 0.0, 0.5), 1.0))

    def test_keyboard_message_matches_existing_keyboard_planner(self) -> None:
        config = NavilaPlannerConfig()
        keyboard_config = KeyboardPlannerConfig()
        action, velocity = key_commands(keyboard_config)["q"]
        expected = build_navila_message(action, velocity, keyboard_config.duration)

        actual = build_keyboard_message("q", config)

        self.assertEqual(json.loads(actual), json.loads(expected))

    def test_unknown_keyboard_key_produces_no_message(self) -> None:
        config = NavilaPlannerConfig()

        self.assertIsNone(build_keyboard_message("z", config))

    def test_keyboard_message_is_available_during_old_navila_action_wait(self) -> None:
        config = NavilaPlannerConfig()
        control = NavilaPlannerControl(active=True)
        control.accept_action("forward", (0.5, 0.0, 0.0), 10.0, now=1.0)
        control.toggle()

        self.assertIsNotNone(build_keyboard_message("w", config))
        self.assertFalse(control.active)

    def test_policy_data_contains_image_and_instruction(self) -> None:
        data = build_policy_data(
            jpeg=b"jpeg-data",
            sequence=4,
            timestamp_ns=123,
            camera="chest_view",
            instruction="walk forward",
        )

        self.assertEqual(data["image_jpeg"], b"jpeg-data")
        self.assertEqual(data["instruction"], "walk forward")
        self.assertEqual(data["sequence"], 4)


if __name__ == "__main__":
    unittest.main()
