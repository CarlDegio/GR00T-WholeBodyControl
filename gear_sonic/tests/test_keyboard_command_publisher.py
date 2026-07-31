"""Tests for the standalone port-5580 keyboard publisher."""
from __future__ import annotations

import unittest

from gear_sonic.scripts.keyboard_command_publisher import encode_keyboard_input


class KeyboardCommandPublisherTests(unittest.TestCase):
    def test_plain_control_key_is_preserved(self):
        self.assertEqual(encode_keyboard_input("k"), "k")

    def test_prompt_shortcut_uses_existing_wire_prefix(self):
        self.assertEqual(
            encode_keyboard_input("t pick up the cup"),
            "prompt:pick up the cup",
        )


if __name__ == "__main__":
    unittest.main()
