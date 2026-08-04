from __future__ import annotations

import sys
import types

import numpy as np
from pathlib import Path


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.scripts.run_lingbot_depth_viewer import (
    LingBotDepthViewerConfig,
    configure_lingbot_runtime_environment,
    conservative_depth_fusion,
    normalized_intrinsics,
    prepare_depth_meters,
    resolve_local_model_path,
    completed_depth_payload,
    mark_ready,
)
from gear_sonic.utils.inference.object_nav import ComposedRGBDCamera


def test_lingbot_viewer_uses_fixed_ten_meter_range_by_default() -> None:
    assert LingBotDepthViewerConfig().max_depth_m == 10.0


def test_prepare_depth_meters_marks_out_of_range_values_invalid() -> None:
    raw = np.array([[0, 500, 5000, 65535]], dtype=np.uint16)

    depth = prepare_depth_meters(raw, depth_scale_m=0.001, max_depth_m=5.0)

    np.testing.assert_allclose(depth, [[0.0, 0.5, 5.0, 0.0]])


def test_normalized_intrinsics_scales_rows_by_image_dimensions() -> None:
    matrix = normalized_intrinsics(
        {"fx": 600.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
        width=640,
        height=480,
    )

    np.testing.assert_allclose(
        matrix,
        [[600 / 640, 0.0, 0.5], [0.0, 500 / 480, 0.5], [0.0, 0.0, 1.0]],
    )


def test_conservative_fusion_keeps_nearest_valid_depth() -> None:
    raw = np.array([[0.0, 1.0, 3.0, 0.0]], dtype=np.float32)
    completed = np.array([[2.0, 2.0, 2.0, 0.0]], dtype=np.float32)

    fused = conservative_depth_fusion(raw, completed, max_depth_m=5.0)

    np.testing.assert_allclose(fused, [[2.0, 1.0, 2.0, 0.0]])


def test_model_resolution_never_contacts_huggingface_network() -> None:
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/cache/model.pt"

    result = resolve_local_model_path("org/model", download=fake_download)

    assert result == "/cache/model.pt"
    assert calls == [
        {
            "repo_id": "org/model",
            "repo_type": "model",
            "filename": "model.pt",
            "local_files_only": True,
        }
    ]


def test_lingbot_runtime_keeps_xformers_enabled(monkeypatch) -> None:
    monkeypatch.setenv("XFORMERS_DISABLED", "1")

    configure_lingbot_runtime_environment()

    assert "XFORMERS_DISABLED" not in __import__("os").environ


def test_mark_ready_replaces_stale_marker(tmp_path: Path) -> None:
    marker = tmp_path / "lingbot.ready"
    marker.write_text("stale")

    mark_ready(str(marker))

    assert marker.read_text() == "ready\n"


def test_completed_depth_payload_contains_only_lingbot_depth() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    completed_m = np.array(
        [[0.0, 1.25, 11.0], [2.5, np.inf, 0.1]], dtype=np.float32
    )
    info = {
        "fx": 500.0,
        "fy": 500.0,
        "cx": 1.0,
        "cy": 1.0,
        "width": 3,
        "height": 2,
        "depth_scale_m": 0.001,
        "depth_aligned_to": "chest_view",
    }

    payload = completed_depth_payload(
        rgb, completed_m, info, timestamp=12.5, max_depth_m=10.0
    )

    np.testing.assert_array_equal(
        payload.images["chest_view_depth"],
        [[0, 1250, 0], [2500, 0, 100]],
    )
    assert payload.timestamps == {"chest_view": 12.5, "chest_view_depth": 12.5}
    assert payload.camera_info["chest_view"]["depth_scale_m"] == 0.001

    decoded = ComposedRGBDCamera.decode_payload(payload.serialize())
    np.testing.assert_allclose(
        decoded.depth_mm,
        [[0.0, 1250.0, 0.0], [2500.0, 0.0, 100.0]],
    )
