"""Contract tests for the migrated Codex ObjectNav policy client."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from gear_sonic.utils.inference.object_nav import (
    CodexBBoxClient,
    RGBDSnapshot,
    get_object_nav_policy_prompt,
)


def snapshot() -> RGBDSnapshot:
    return RGBDSnapshot(
        rgb_bgr=np.zeros((10, 20, 3), dtype=np.uint8),
        depth_raw=np.ones((10, 20), dtype=np.uint16),
        depth_mm=np.ones((10, 20), dtype=np.float32),
        fx=123.5,
        fy=124.5,
        cx=9.5,
        cy=4.5,
        depth_scale_m=0.001,
        depth_aligned_to="chest_view",
        timestamp=1.0,
    )


def policy_json() -> str:
    return json.dumps(
        {
            "visual_check": "chair visible",
            "action": "NAVIGATE",
            "bbox_2d": [100, 200, 450, 800],
            "target": "red chair",
            "target_type": "global_target",
            "estimated_distance_m": 2.5,
            "target_center_normalized": [275.0, 500.0],
            "target_center_pixel": [5.5, 5.0],
            "horizontal_offset_pixel": -4.0,
            "camera_bearing_deg": -1.9,
            "rotation_direction": "CENTERED",
            "rotation_angle_deg": 0.0,
            "confidence": 0.91,
            "distance_confidence": 0.73,
            "stop_reasoning": "",
        }
    )


class PromptTests(unittest.TestCase):
    def test_prompt_uses_explicit_inputs_and_live_intrinsics(self):
        prompt = get_object_nav_policy_prompt(
            'find the "red chair"\nthen stop', "red chair", snapshot()
        )
        self.assertIn(r'find the \"red chair\"\nthen stop', prompt)
        self.assertIn("fx=123.5", prompt)
        self.assertIn("width=20, height=10", prompt)
        self.assertIn("Never return STOP merely", prompt)


class CodexClientTests(unittest.TestCase):
    def test_builds_read_only_image_command_with_proxy_environment(self):
        runner = mock.Mock(
            side_effect=[
                subprocess.CompletedProcess(
                    ["codex", "login", "status"],
                    0,
                    stdout="Logged in using ChatGPT\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    ["codex", "exec"], 0, stdout=policy_json(), stderr=""
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "input.png"
            image_path.write_bytes(b"image")
            schema_path = Path(temporary) / "schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            with mock.patch.dict(os.environ, {"MARKER": "kept"}, clear=True):
                result = CodexBBoxClient(
                    schema_path=schema_path,
                    runner=runner,
                    timeout_seconds=12,
                ).locate(
                    image_path=image_path,
                    mission="find chair",
                    global_target="chair",
                    snapshot=snapshot(),
                    cwd=temporary,
                )

        self.assertEqual(result["target"], "red chair")
        command = runner.call_args_list[1].args[0]
        self.assertIn("--ephemeral", command)
        self.assertIn("read-only", command)
        self.assertIn("--output-schema", command)
        self.assertFalse(runner.call_args_list[1].kwargs.get("shell", False))
        for call in runner.call_args_list:
            environment = call.kwargs["env"]
            self.assertEqual(environment["HTTP_PROXY"], "http://127.0.0.1:7897/")
            self.assertEqual(environment["ALL_PROXY"], "socks://127.0.0.1:7897/")
            self.assertEqual(environment["MARKER"], "kept")

    def test_empty_proxy_overrides_remove_inherited_values(self):
        inherited = {
            "HTTP_PROXY": "http://old",
            "ALL_PROXY": "socks://old",
            "OBJECT_NAV_CODEX_HTTP_PROXY": "",
            "OBJECT_NAV_CODEX_ALL_PROXY": "",
        }
        with mock.patch.dict(os.environ, inherited, clear=True):
            environment = CodexBBoxClient()._subprocess_env()
        for key in (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "all_proxy",
            "ALL_PROXY",
        ):
            self.assertNotIn(key, environment)

    def test_rejects_invalid_distance_and_stop_target_type(self):
        value = json.loads(policy_json())
        value["estimated_distance_m"] = 0.0
        with self.assertRaisesRegex(ValueError, "distance"):
            CodexBBoxClient.validate_policy(value)

        value = json.loads(policy_json())
        value.update(
            {
                "action": "STOP",
                "target_type": "intermediate_landmark",
                "stop_reasoning": "reached",
            }
        )
        with self.assertRaisesRegex(ValueError, "global target"):
            CodexBBoxClient.validate_policy(value)


if __name__ == "__main__":
    unittest.main()
