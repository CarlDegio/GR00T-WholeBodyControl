"""Recording-command selection shared by typed and PICO control inputs."""

from __future__ import annotations


TYPED_RECORDING_KEYS = {
    "start_recording": "c",
    "stop_recording_success": "e",
    "stop_recording_failure": "x",
}


def select_recording_key(
    command_name: str | None,
    *,
    pico_abort: bool,
    pico_toggle: bool,
) -> str | None:
    """Preserve PICO priority while translating typed recording commands."""
    if pico_abort:
        return "x"
    if pico_toggle:
        return "c"
    return TYPED_RECORDING_KEYS.get(command_name)
