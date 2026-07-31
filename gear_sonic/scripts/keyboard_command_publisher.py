"""Standalone terminal keyboard publisher for Sonic control commands."""
from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class KeyboardPublisherConfig:
    """CLI configuration for the port-5580 keyboard publisher."""

    host: str = "*"
    """ZMQ bind host."""

    port: int = 5580
    """ZMQ PUB port consumed by the direct Sonic controller."""


def encode_keyboard_input(line: str) -> str:
    """Encode one terminal line using the existing keyboard wire protocol."""
    command = str(line).strip()
    if command.startswith("t "):
        return "prompt:" + command[2:]
    return command


def main(config: KeyboardPublisherConfig) -> None:
    """Read terminal lines and publish them as ZMQ strings."""
    import zmq

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    endpoint = f"tcp://{config.host}:{config.port}"
    publisher.bind(endpoint)
    time.sleep(0.5)
    print(f"[KeyboardPublisher] PUB bound to {endpoint}")
    print("[KeyboardPublisher] k=start/stop, i=safe stop, o=PLANNER")
    try:
        while True:
            message = encode_keyboard_input(input("> "))
            if not message:
                continue
            publisher.send_string(message)
            print(f"[KeyboardPublisher] sent: {message}")
    except (EOFError, KeyboardInterrupt):
        print("\n[KeyboardPublisher] interrupted")
    finally:
        publisher.close()
        context.term()
        print("[KeyboardPublisher] shutdown complete")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(KeyboardPublisherConfig))
