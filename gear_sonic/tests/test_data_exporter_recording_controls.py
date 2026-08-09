from gear_sonic.utils.data_collection.recording_controls import select_recording_key


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
