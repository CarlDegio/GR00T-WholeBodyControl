#!/usr/bin/env python3
"""Launch standalone UniLM positioning: K then B in the control pane; Space cancels.

The existing VLA service stays in PLANNER mode as the C++ command bridge.
Navigation models and manipulation inference are not needed for positioning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gear_sonic.scripts.launch_inference import (
    _signal_handler,
    default_launch_config_path,
    load_inference_launch_config,
    main,
)


def standalone_profile(source: Path, task: str | None = None) -> dict:
    # The launcher import above bootstraps the shared environment first.
    import yaml

    value = yaml.safe_load(source.read_text())
    components = value["components"]
    if "unilm_nav" not in components:
        raise ValueError("profile must define components.unilm_nav")
    components["unilm_nav"]["enabled"] = True
    components["launcher"].update(
        run_mode="unilm_only", base_pose_enabled=True, keyboard_planner=True, data_exporter=False
    )
    if task is not None:
        if not task.strip():
            raise ValueError("--task must not be empty")
        components["vla"]["vla_prompt"] = task.strip()
    value["profile"] = "unilm_nav_standalone"
    return value


def cli():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_launch_config_path())
    parser.add_argument(
        "--task", default=None,
        help="UniLM/agent task; defaults to components.vla.vla_prompt (does not change vla_trained_prompt)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print effective settings without starting services"
    )
    args = parser.parse_args()
    value = standalone_profile(args.config.resolve(), args.task)
    if args.dry_run:
        print(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
        return
    destination = Path(__file__).resolve().parents[2] / "outputs/runtime_profiles"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"unilm_{time.time_ns()}.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    main(load_inference_launch_config(path))


if __name__ == "__main__":
    cli()
