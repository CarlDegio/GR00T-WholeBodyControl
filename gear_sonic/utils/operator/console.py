#!/usr/bin/env python3
"""Interactive CLI frontend for SonicControlGateway."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import sys
import termios
import time
import tty
from typing import Iterator, TextIO

import zmq

from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.runtime.gateway.control import ControlGatewayCore, OperatorConsoleRouter
from gear_sonic.runtime.zmq_sockets import connect_push


@contextmanager
def cbreak_terminal(stream: TextIO = sys.stdin) -> Iterator[None]:
    """Read deployed control keys immediately, matching the legacy keyboard pane."""
    if not stream.isatty():
        yield
        return
    descriptor = stream.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def read_operator_input(stream: TextIO = sys.stdin) -> str:
    """Return one immediate control key, or a line in explicit command mode."""
    if not stream.isatty():
        return stream.readline().rstrip("\n")
    key = stream.read(1)
    if key == "\x03":
        raise KeyboardInterrupt
    if key == ":":
        print("\ncommand> ", end="", flush=True)
        return stream.readline().rstrip("\n")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="")
    parser.add_argument("--overlay", action="append", default=[])
    args = parser.parse_args()
    profile = load_runtime_profile(args.profile or None, overlays=tuple(args.overlay))
    endpoint = profile.endpoint_uri("control_gateway_intent")
    ttl_ms = int(profile.component("control_gateway")["command_ttl_ms"])

    context = zmq.Context()
    sender = connect_push(context, endpoint, linger_ms=0)
    core = ControlGatewayCore(ttl_ms=ttl_ms)
    router = OperatorConsoleRouter()
    print(f"[OperatorCLI] ControlGateway: {endpoint}")
    print(
        "Keys: k=start/stop, i=pose, o=planner, p=pause, [/]=hands; "
        "g=agent success + stand, h=agent failure + stand; "
        "PLANNER: w/a/s/d/q/e manual, n=LaViRA, b=BasePose, space=cancel "
        "(single-key, no Enter); "
        ": enters a full command line"
    )
    time.sleep(0.2)
    try:
        with cbreak_terminal():
            while True:
                value = read_operator_input()
                if value in {"", "\n", "\r"}:
                    continue
                if value == "perturb" and "experiment" in profile.components:
                    event = core.accept_command("experiment_perturbation", parameters={})
                    sender.send_string(event.command.to_json())
                    print("\nRecorded prescribed perturbation.")
                    continue
                event = router.accept_line(value, core=core)
                sender.send_string(event.command.to_json())
                print(f"\rSent: {event.command.name:<28}", end="", flush=True)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        sender.close()
        context.term()


if __name__ == "__main__":
    main()
