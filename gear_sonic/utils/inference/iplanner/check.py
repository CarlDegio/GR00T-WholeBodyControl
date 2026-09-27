"""Check real iPlanner weights and HTTP inference on loopback, without robot services."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import requests
import torch
from werkzeug.serving import WSGIRequestHandler, make_server

from gear_sonic.experiments.config import CONFIGS, resolve
from gear_sonic.utils.inference.iplanner.client import request_trajectory, reset_iplanner
from gear_sonic.utils.inference.iplanner.config import IPlannerServerConfig
from gear_sonic.utils.inference.iplanner.control import IPlannerNavigationConfig, pure_pursuit


class QuietHandler(WSGIRequestHandler):
    def log_request(self, *args, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS / "baselinelavira.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--snapshot", type=Path, help="Existing VLA snapshot directory with chest RGB-D")
    parser.add_argument("--output", type=Path, required=True, help="JSON validation report")
    args = parser.parse_args()
    payload, _ = resolve(args.config)
    components = payload["components"]
    if components.get("navigation_policy", {}).get("backend") != "iplanner":
        parser.error("the experiment must select iPlanner")
    settings = IPlannerServerConfig(**components["iplanner"])
    control = IPlannerNavigationConfig(**components["iplanner_navigation"])
    missing = [f"{label}: {path}" for path, label in settings.required_assets() if not path.is_file()]
    if missing:
        parser.error("Missing " + ", ".join(missing))
    device = args.device or settings.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run where the NVIDIA GPU is accessible, or explicitly use --device cpu")
    torch.set_num_threads(4)
    root = Path(settings.root).expanduser().resolve()
    sys.path.insert(0, str(root))
    original_argv = sys.argv
    try:
        sys.argv = [str(root / "iplanner_server.py"), "--device", device,
                    "--checkpoint", str(settings.asset_path(settings.checkpoint)),
                    "--config", str(settings.asset_path(settings.model_config))]
        spec = importlib.util.spec_from_file_location("iplanner_check_server", root / "iplanner_server.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv

    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "device": device, "torch": torch.__version__, "cuda_build": torch.version.cuda,
              "gpu": torch.cuda.get_device_name() if device == "cuda" else None, "cases": []}
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full(rgb.shape[:2], 3.0, dtype=np.float32)
    info = dict(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    if args.snapshot:
        metadata = json.loads((args.snapshot / "snapshot.json").read_text())
        camera = metadata["cameras"]["chest_view"]
        info = camera["camera_info"]
        rgb = cv2.imread(str(args.snapshot / camera["rgb"]))
        raw = np.load(args.snapshot / camera["depth"], allow_pickle=False)
        depth = raw.astype(np.float32) * float(info["depth_scale_m"])
        report["snapshot"] = str(args.snapshot.resolve())

    server = make_server("127.0.0.1", 0, module.app, request_handler=QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"

    def infer(name, image, distance, goal):
        started = time.perf_counter()
        path = request_trajectory(endpoint, image, distance, goal)
        elapsed = (time.perf_counter() - started) * 1000
        velocity, _ = pure_pursuit(path, float(np.linalg.norm(goal)), control)
        assert np.isfinite(velocity).all()
        assert 0 <= velocity[0] <= control.max_forward_speed
        assert velocity[0] == 0 or velocity[0] >= control.min_forward_speed
        assert abs(velocity[2] - control.yaw_bias) <= control.max_yaw_speed
        report["cases"].append(dict(name=name, latency_ms=round(elapsed, 3), shape=list(path.shape),
                                    endpoint_xy=path[-1].tolist(), velocity=list(velocity)))
        return path

    try:
        reset_iplanner(endpoint, info)
        actual_device = str(next(module.iplanner_navigator.net.parameters()).device)
        assert actual_device.split(":")[0] == device
        report["model_device"] = actual_device
        base = infer("forward", rgb, depth, (3.0, 0.0))
        changed_rgb = infer("changed_rgb", 255 - rgb, depth, (3.0, 0.0))
        assert np.allclose(base, changed_rgb, atol=1e-6), "RGB unexpectedly changed the trajectory"
        left = infer("left_goal", rgb, depth, (3.0, 1.0))
        right = infer("right_goal", rgb, depth, (3.0, -1.0))
        assert not np.allclose(left, right), "trajectory is insensitive to goal changes"
        changed_depth = infer("changed_depth", rgb, depth * 0.5, (3.0, 0.0))
        assert not np.allclose(base, changed_depth), "trajectory is insensitive to depth changes"
        try:
            request_trajectory(endpoint, rgb, np.zeros_like(depth), (3.0, 0.0))
        except ValueError as exc:
            assert "valid positive depth" in str(exc)
            report["invalid_depth_rejected"] = True
        else:
            raise AssertionError("all-invalid depth was allowed to generate a trajectory")
        invalid_depth = depth.copy()
        invalid_depth[:32, :32] = np.nan
        infer("partially_invalid_depth", rgb, invalid_depth, (3.0, 0.0))
        for index in range(10):
            infer(f"timing_{index}", rgb, depth, (3.0, 0.0))
        report["http_median_ms"] = float(np.median([r["latency_ms"] for r in report["cases"][-10:]]))
        report["health"] = requests.get(endpoint + "/health", timeout=5).json()
        report["rgb_invariant"] = True
        report["goal_sensitive"] = True
        report["depth_sensitive"] = True
        checkpoint = settings.asset_path(settings.checkpoint)
        report["checkpoint"] = str(checkpoint)
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        report["checkpoint_sha256"] = digest.hexdigest()
        report["passed"] = True
        report["scope"] = "Real checkpoint, original LaViRA server, loopback HTTP, Groot client and pure pursuit; no robot commands"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


if __name__ == "__main__":
    main()
