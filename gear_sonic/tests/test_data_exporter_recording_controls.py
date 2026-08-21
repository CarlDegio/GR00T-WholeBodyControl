from types import SimpleNamespace

import pytest

from gear_sonic.utils.data_collection.episode import EpisodeState
from gear_sonic.utils.data_collection.recording import select_recording_key
from gear_sonic.utils.data_collection.service import GrootDataCollector


YELLOW = "\033[93m"
RESET = "\033[0m"


class _ControlListener:
    def __init__(self, command_name: str | None):
        self._command = None if command_name is None else SimpleNamespace(name=command_name)

    def read_command(self):
        return self._command


def _recording_collector(*, command_name: str | None, pico_toggle: bool) -> GrootDataCollector:
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector.text_to_speech = None
    collector._control_listener = _ControlListener(command_name)
    collector._manager_toggle_da = False
    collector._manager_toggle_dc = pico_toggle
    collector._episode_state = EpisodeState()
    collector._episode_state.state = collector._episode_state.RECORDING
    collector._initial_yaw = 1.0
    return collector


def test_pico_recording_buttons_keep_their_original_data_exporter_actions() -> None:
    assert select_recording_key(None, pico_abort=False, pico_toggle=True) == "c"
    assert select_recording_key(None, pico_abort=True, pico_toggle=False) == "x"


def test_pico_buttons_keep_priority_over_typed_control_commands() -> None:
    assert (
        select_recording_key(
            "stop_recording_success",
            pico_abort=True,
            pico_toggle=False,
        )
        == "x"
    )
    assert (
        select_recording_key(
            "stop_recording_failure",
            pico_abort=False,
            pico_toggle=True,
        )
        == "c"
    )


def test_typed_recording_commands_match_episode_state_keys() -> None:
    assert select_recording_key("start_recording", pico_abort=False, pico_toggle=False) == "c"
    assert (
        select_recording_key(
            "stop_recording_success",
            pico_abort=False,
            pico_toggle=False,
        )
        == "e"
    )
    assert (
        select_recording_key(
            "stop_recording_failure",
            pico_abort=False,
            pico_toggle=False,
        )
        == "x"
    )


@pytest.mark.parametrize(
    ("command_name", "pico_toggle"),
    [
        (None, True),
        ("stop_recording_success", False),
    ],
)
def test_stopping_recording_prints_the_save_message_in_yellow(
    command_name: str | None,
    pico_toggle: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    collector = _recording_collector(command_name=command_name, pico_toggle=pico_toggle)

    collector._check_recording_commands()

    assert collector._episode_state.get_state() == collector._episode_state.NEED_TO_SAVE
    assert capsys.readouterr().out == (
        f"{YELLOW}Stopping recording, preparing to save{RESET}\n"
    )


def test_finished_saving_prints_the_completion_message_in_yellow(
    capsys: pytest.CaptureFixture[str],
) -> None:
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector.text_to_speech = None
    collector.frequency = 20
    collector._episode_state = EpisodeState()
    collector._episode_state.state = collector._episode_state.NEED_TO_SAVE
    collector.data_exporter = SimpleNamespace(
        episode_buffer={"size": 1},
        save_episode=lambda: None,
    )
    collector.sonic_timing_monitor = SimpleNamespace(reset=lambda: None)
    collector._initial_yaw = 1.0

    collector._finalize_frame(0.0)

    assert collector._episode_state.get_state() == collector._episode_state.IDLE
    assert capsys.readouterr().out.endswith(
        f"{YELLOW}Finished saving episode{RESET}\n"
    )
