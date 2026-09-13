#!/usr/bin/env python3
"""Launch Groot with LaViRA waypoint selection, iPlanner and pure pursuit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / "gear_sonic/config/launch_inference.yaml"
BASELINE_OVERLAY = ROOT / "gear_sonic/config/baselinelavira.yaml"


def build_profile(base_profile: Path = DEFAULT_PROFILE, **server_overrides) -> dict:
    from gear_sonic.runtime.profile import load_runtime_profile

    profile = load_runtime_profile(base_profile, overlays=(BASELINE_OVERLAY,))
    components = {name: dict(settings) for name, settings in profile.components.items()}
    components["iplanner"].update({k: v for k, v in server_overrides.items() if v is not None})
    components["iplanner"]["root"] = str(Path(components["iplanner"]["root"]).expanduser().resolve())
    launcher = components["launcher"]
    if not launcher["keyboard_planner"] or launcher["planner_input"] != "lavira":
        raise ValueError("baselinelavira requires the existing LaViRA navigation launcher")
    experiment = components.get("experiment", {})
    if (experiment.get("navigation_backend", "lavira") != "lavira"
            or experiment.get("entry_stage", "navigation") != "navigation"):
        raise ValueError("baselinelavira requires a LaViRA navigation entry stage")
    return {
        "schema": "sonic.runtime_profile",
        "version": 1,
        "profile": profile.name,
        "endpoints": {name: {"host": ep.host, "port": ep.port}
                      for name, ep in profile.endpoints.items()},
        "ros_topics": dict(profile.ros_topics),
        "components": components,
    }


def main(argv: list[str] | None = None) -> None:
    import yaml

    from gear_sonic.runtime.profile import load_runtime_profile
    from gear_sonic.scripts.launch_inference import (
        build_navdp_planner_command,
        build_navdp_server_command,
        load_inference_launch_config,
    )
    from gear_sonic.utils.inference.iplanner.config import load_iplanner_server_config
    from gear_sonic.utils.inference.iplanner.control import load_navigation_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE,
                        help="Base runtime profile; all non-iPlanner settings are inherited")
    parser.add_argument("--iplanner-root", help="LaViRA Unitree G1 iplanner directory")
    parser.add_argument("--checkpoint", help="iPlanner weights, absolute or relative to --iplanner-root")
    parser.add_argument("--model-config", help="iPlanner model YAML, absolute or relative to --iplanner-root")
    parser.add_argument("--conda-env", help="Environment containing torch, torchvision, Flask and OpenCV")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true", help="Write and inspect the profile without launching")
    args, launch_args = parser.parse_known_args(argv)
    payload = build_profile(
        args.config, root=args.iplanner_root, checkpoint=args.checkpoint,
        model_config=args.model_config, conda_env=args.conda_env, device=args.device,
    )
    output_dir = ROOT / "outputs/baselinelavira"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the snapshot after detaching: all tmux workers use this exact file.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".yaml",
                                     prefix="runtime_", dir=output_dir, delete=False) as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
        runtime_path = Path(stream.name)
    config = load_inference_launch_config(runtime_path)
    profile = load_runtime_profile(runtime_path)
    settings = load_iplanner_server_config(profile)
    load_navigation_config(profile)
    print(f"baselinelavira runtime: {runtime_path}", flush=True)
    if args.dry_run:
        print("Navigation: LaViRA LA strategy -> VA waypoint -> iPlanner -> Pure Pursuit -> SONIC")
        print(f"iPlanner server: {build_navdp_server_command(config)}")
        print(f"Navigation worker: {build_navdp_planner_command(config, ROOT)}")
        for path, label in settings.required_assets():
            print(f"{'OK' if path.is_file() else 'MISSING'}: {label}: {path}")
        return
    python = ROOT / ".venv_inference/bin/python"
    interpreter = str(python) if python.is_file() else sys.executable
    os.chdir(ROOT)
    os.execv(interpreter, [interpreter, "-m", "gear_sonic.scripts.launch_inference",
                          "--config", str(runtime_path), *launch_args])


if __name__ == "__main__":
    from gear_sonic.scripts.launcher import bootstrap_venv

    try:
        import tyro  # noqa: F401
        import yaml  # noqa: F401
        dependencies_available = True
    except ImportError:
        dependencies_available = False
    bootstrap_venv(
        dependencies_available, repo_root=ROOT, venv_name=".venv_inference",
        missing_message="Run bash install_scripts/install_inference.sh first.",
    )
    main()
