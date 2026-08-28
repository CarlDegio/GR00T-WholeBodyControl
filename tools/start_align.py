#!/usr/bin/env python3
"""Start one standalone BasePose ALIGN through the running ControlGateway."""

from __future__ import annotations

import argparse
import json
import time

from gear_sonic.runtime.gateway.control_client import ControlGatewayIntentClient
from gear_sonic.runtime.profile import load_runtime_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="gear_sonic/config/launch_inference.yaml")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--target", required=True)
    parser.add_argument("--yaw-align-target", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = str(args.target).strip()
    yaw_align_target = str(args.yaw_align_target).strip()
    if not target or not yaw_align_target:
        raise ValueError("target and yaw-align target must be non-empty")
    profile = load_runtime_profile(args.profile, overlays=tuple(args.overlay))
    parameters: dict[str, object] = {
        "key": "b",
        "target": target,
        "yaw_align_target": yaw_align_target,
    }

    client = ControlGatewayIntentClient(
        profile.endpoint_uri("control_gateway_intent"),
        source="standalone_align",
        ttl_ms=int(profile.component("control_gateway")["command_ttl_ms"]),
    )
    try:
        # Allow the PUSH connection to establish before the one-shot command.
        time.sleep(0.2)
        command = client.send("navigation_key", parameters)
        time.sleep(0.05)
    finally:
        client.close()
    print(json.dumps(command.parameters, ensure_ascii=False))


if __name__ == "__main__":
    main()
