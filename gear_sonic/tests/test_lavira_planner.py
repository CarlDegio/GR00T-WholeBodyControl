"""Behavioral tests for the single-cycle LaViRA-to-REASAN command source."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
import queue

import pytest

from gear_sonic.scripts.lavira_planner import (
    CommandValidationError,
    LaviraPlannerConfig,
    LaviraPlannerController,
    LaviraPlannerRuntime,
    ObjectNavBatch,
    PlannerBusyError,
    VelocityCommand,
    WorkerResult,
    build_reasan_velocity_message,
    run_inference_worker,
    run_planner_loop,
    validate_object_nav_batch,
)
from gear_sonic.utils.inference.object_nav import ObjectNavResult


def commands(
    *,
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.4, 1.0),
    translation: tuple[float, float, float, float] = (0.3, 0.0, 0.0, 2.0),
) -> dict[str, object]:
    return {
        "commands": [
            dict(zip(("vx", "vy", "wz", "duration"), rotation)),
            dict(zip(("vx", "vy", "wz", "duration"), translation)),
        ]
    }


def result(
    outcome: str = "NAVIGATE", payload: dict[str, object] | None = None
) -> ObjectNavResult:
    return ObjectNavResult(
        outcome=outcome,
        policy={},
        commands=commands() if payload is None else payload,
        geometry={},
        output_dir="/tmp/object-nav",
        error=None,
    )


def decoded_messages(messages: list[str]) -> list[dict[str, object]]:
    return [json.loads(message) for message in messages]


def assert_stop_messages(messages: list[str], count: int = 3) -> None:
    decoded = decoded_messages(messages[-count:])
    assert [message["action"] for message in decoded] == ["stop"] * count
    assert [message["velocity"] for message in decoded] == [
        {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    ] * count


def test_velocity_command_is_immutable_and_exposes_velocity_tuple() -> None:
    command = VelocityCommand(0.3, -0.1, 0.2, 0.5)

    assert command.velocity == (0.3, -0.1, 0.2)
    with pytest.raises(FrozenInstanceError):
        command.vx = 1.0  # type: ignore[misc]


def test_adapter_builds_existing_reasan_velocity_protocol() -> None:
    message = json.loads(
        build_reasan_velocity_message(
            VelocityCommand(0.0, 0.0, -0.4, 1.25), action="turn_right"
        )
    )

    assert message["type"] == "navila_reasan_velocity_command"
    assert message["version"] == 1
    assert message["status"] == "ok"
    assert message["action"] == "turn_right"
    assert message["duration_s"] == 1.25
    assert message["velocity"] == {"vx": 0.0, "vy": 0.0, "wz": -0.4}
    assert message["segments"] == [
        {"duration_s": 1.25, "vx": 0.0, "vy": 0.0, "wz": -0.4}
    ]


def test_validation_accepts_exact_rotation_then_translation_boundaries() -> None:
    batch = validate_object_nav_batch(
        commands(
            rotation=(0.0, 0.0, math.pi / 30.0, 30.0),
            translation=(0.3, 0.4, 0.0, 0.05),
        )
    )

    assert batch == ObjectNavBatch(
        rotation=VelocityCommand(0.0, 0.0, math.pi / 30.0, 30.0),
        translation=VelocityCommand(0.3, 0.4, 0.0, 0.05),
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "exactly two"),
        ({"commands": [{}, {}, {}]}, "exactly two"),
        (commands(rotation=(0.001, 0.0, 0.4, 1.0)), "pure rotation"),
        (commands(translation=(0.3, 0.0, 0.001, 2.0)), "pure translation"),
        (commands(rotation=(0.0, 0.0, float("nan"), 1.0)), "finite"),
        (commands(rotation=(0.0, 0.0, 0.4, 0.049)), "publish period"),
        (commands(translation=(0.3, 0.0, 0.0, 30.001)), "duration exceeds"),
        (commands(translation=(0.5, 0.001, 0.0, 2.0)), "speed exceeds"),
        (commands(rotation=(0.0, 0.0, 0.4, math.pi / 0.4 + 0.001)), "yaw exceeds"),
    ],
)
def test_validation_rejects_malformed_or_unsafe_batches(
    payload: object, message: str
) -> None:
    with pytest.raises(CommandValidationError, match=message):
        validate_object_nav_batch(payload)


@pytest.mark.parametrize(
    "payload",
    [
        commands(rotation=(0.0, 0.0, 0.4, -0.1)),
        commands(translation=(0.3, 0.0, 0.0, -0.1)),
        commands(rotation=(0.0, 0.0, 0.4, True)),
        {"commands": [
            {"vx": 0.0, "vy": 0.0, "wz": 0.4},
            {"vx": 0.3, "vy": 0.0, "wz": 0.0, "duration": 1.0},
        ]},
    ],
)
def test_validation_rejects_invalid_command_fields(payload: object) -> None:
    with pytest.raises(CommandValidationError):
        validate_object_nav_batch(payload)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_speed": 0.0}, "limits"),
        ({"max_duration": float("inf")}, "limits"),
        ({"max_abs_yaw": True}, "limits"),
        ({"min_positive_duration": -0.1}, "min_positive_duration"),
    ],
)
def test_validation_rejects_invalid_safety_limits(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_object_nav_batch(commands(), **kwargs)  # type: ignore[arg-type]


def test_controller_obeys_rotation_pause_translation_deadlines() -> None:
    controller = LaviraPlannerController(transition_pause=0.5)
    controller.start(result(), now=10.0)

    assert controller.phase == "rotating"
    assert controller.step(10.0).velocity == (0.0, 0.0, 0.4)
    assert controller.step(10.999).velocity == (0.0, 0.0, 0.4)
    assert controller.step(11.0).velocity == (0.0, 0.0, 0.0)
    assert controller.phase == "transition_pause"
    assert controller.step(11.499).velocity == (0.0, 0.0, 0.0)
    assert controller.step(11.5).velocity == (0.3, 0.0, 0.0)
    assert controller.phase == "translating"
    assert controller.step(13.499).velocity == (0.3, 0.0, 0.0)
    assert controller.step(13.5).velocity == (0.0, 0.0, 0.0)
    assert controller.phase == "final_stop"
    assert controller.failure_reason is None


def test_delayed_rotation_transition_starts_full_pause_from_observation() -> None:
    controller = LaviraPlannerController(transition_pause=0.5)
    controller.start(result(), now=10.0)

    assert controller.step(11.4).velocity == (0.0, 0.0, 0.0)
    assert controller.phase == "transition_pause"
    assert controller.step(11.899).velocity == (0.0, 0.0, 0.0)
    assert controller.step(11.9).velocity == (0.3, 0.0, 0.0)


def test_controller_publishes_three_final_stops_before_idle() -> None:
    controller = LaviraPlannerController(final_stop_count=3)
    controller.start(
        result(
            payload=commands(
                rotation=(0.0, 0.0, 0.0, 0.0),
                translation=(0.0, 0.0, 0.0, 0.0),
            )
        ),
        now=2.0,
    )

    assert [controller.step(2.5).velocity for _ in range(3)] == [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    assert controller.phase == "idle"


def test_controller_skips_zero_duration_motion_phases_but_keeps_pause() -> None:
    controller = LaviraPlannerController(transition_pause=0.5)
    controller.start(
        result(payload=commands(rotation=(0.0, 0.0, 0.0, 0.0))), now=4.0
    )

    assert controller.phase == "transition_pause"
    assert controller.step(4.0).velocity == (0.0, 0.0, 0.0)
    assert controller.step(4.5).velocity == (0.3, 0.0, 0.0)


@pytest.mark.parametrize("outcome", ["STOP", "FAILED", "REJECTED", "UNKNOWN"])
def test_non_navigation_results_fail_closed_into_final_stop(outcome: str) -> None:
    controller = LaviraPlannerController()
    controller.start(result(outcome), now=1.0)

    assert controller.phase == "final_stop"
    assert [controller.step(1.0).velocity for _ in range(3)] == [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    assert controller.phase == "idle"


def test_invalid_navigation_result_fails_closed_without_motion() -> None:
    controller = LaviraPlannerController()
    controller.start(
        result(payload=commands(translation=(0.3, 0.0, 0.2, 1.0))), now=1.0
    )

    assert controller.phase == "final_stop"
    assert controller.failure_reason is not None
    assert controller.step(1.0).velocity == (0.0, 0.0, 0.0)


def test_controller_rejects_busy_start_and_nonfinite_clock() -> None:
    controller = LaviraPlannerController()
    controller.start(result(), now=1.0)

    with pytest.raises(PlannerBusyError):
        controller.start(result(), now=2.0)
    with pytest.raises(ValueError, match="finite"):
        controller.step(float("nan"))


def test_controller_rejects_zero_stop_message_duration() -> None:
    with pytest.raises(ValueError, match="min_positive_duration"):
        LaviraPlannerController(min_positive_duration=0.0)


def test_config_preserves_agentnav_and_keyboard_defaults() -> None:
    config = LaviraPlannerConfig(mission="find chair", global_target="chair")

    assert config.port == 5558
    assert config.planner_hz == 20.0
    assert config.transition_pause == 0.5
    assert config.max_speed == 0.5
    assert config.max_duration == 30.0
    assert config.max_abs_yaw == math.pi
    assert config.camera_timeout_ms == 3000
    assert config.codex_timeout_seconds == 180.0
    assert config.min_confidence == 0.6
    assert config.rotation_speed == 0.4
    assert config.forward_speed == 0.3
    assert config.target_standoff_distance == 0.0
    assert config.max_direct_travel == 8.0


def test_runtime_uses_one_slot_queues_and_rejects_repeated_n() -> None:
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda _message: None,
        sleep=lambda _duration: None,
    )

    assert runtime.request_queue.maxsize == 1
    assert runtime.result_queue.maxsize == 1
    assert runtime.handle_key("n", now=1.0) == "started"
    assert runtime.phase == "inferencing"
    assert runtime.generation == 1
    assert runtime.handle_key("N", now=1.1) == "busy"
    assert runtime.generation == 1
    assert runtime.request_queue.get_nowait() == 1


def test_cancel_increments_generation_publishes_stops_and_discards_late_result() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"

    assert runtime.handle_key(" ", now=1.1) == "cancelled"
    assert runtime.generation == 2
    assert runtime.phase == "idle"
    assert_stop_messages(published)

    assert runtime.accept_worker_result(WorkerResult(1, result(), None), now=2.0) is False
    assert runtime.phase == "idle"
    assert runtime.publish_due(2.0) is None


@pytest.mark.parametrize("key", ["w", "s", "a", "d", "q", "e"])
def test_manual_keys_cancel_first_and_reuse_keyboard_defaults(key: str) -> None:
    published: list[str] = []
    config = LaviraPlannerConfig(mission="find chair", global_target="chair")
    runtime = LaviraPlannerRuntime(
        config, publish=published.append, sleep=lambda _duration: None
    )
    runtime.handle_key("n", now=1.0)

    assert runtime.handle_key(key.upper(), now=1.1) == "manual"
    assert runtime.generation == 2
    assert_stop_messages(published[:-1])
    expected_action, expected_velocity = {
        "w": ("forward", (0.5, 0.0, 0.0)),
        "s": ("backward", (-0.3, 0.0, 0.0)),
        "a": ("move_left", (0.0, 0.15, 0.0)),
        "d": ("move_right", (0.0, -0.15, 0.0)),
        "q": ("turn_left", (0.0, 0.0, 0.5)),
        "e": ("turn_right", (0.0, 0.0, -0.5)),
    }[key]
    manual = json.loads(published[-1])
    assert manual["action"] == expected_action
    assert manual["velocity"] == dict(zip(("vx", "vy", "wz"), expected_velocity))
    assert manual["duration_s"] == 0.5


def test_manual_override_spaces_stop_publications_before_nonzero() -> None:
    events: list[tuple[str, object]] = []

    def publish(message: str) -> None:
        events.append(("publish", json.loads(message)["action"]))

    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=publish,
        sleep=lambda duration: events.append(("sleep", duration)),
    )

    assert runtime.handle_key("w", now=1.0) == "manual"
    assert events == [
        ("publish", "stop"),
        ("sleep", 0.05),
        ("publish", "stop"),
        ("sleep", 0.05),
        ("publish", "stop"),
        ("sleep", 0.05),
        ("publish", "forward"),
    ]


def test_termination_during_manual_stop_interval_suppresses_nonzero() -> None:
    published: list[str] = []
    alive = True

    def request_termination(_duration: float) -> None:
        nonlocal alive
        alive = False

    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=request_termination,
    )

    run_planner_loop(
        runtime,
        read_key=iter(["w"]).__next__,
        monotonic=lambda: 1.0,
        sleep=lambda _duration: None,
        running=lambda: alive,
    )

    assert all(message["action"] == "stop" for message in decoded_messages(published))


def test_shutdown_spaces_all_repeated_stop_publications() -> None:
    events: list[tuple[str, object]] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda message: events.append(
            ("publish", json.loads(message)["action"])
        ),
        sleep=lambda duration: events.append(("sleep", duration)),
    )

    runtime.shutdown()

    assert events == [
        ("publish", "stop"),
        ("sleep", 0.05),
        ("publish", "stop"),
        ("sleep", 0.05),
        ("publish", "stop"),
        ("sleep", 0.05),
    ]


def test_exit_key_cancels_and_publishes_repeated_stop() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=1.0)

    assert runtime.handle_key("x", now=1.1) == "exit"
    assert runtime.generation == 2
    assert_stop_messages(published)


def test_current_worker_result_starts_motion_and_stale_or_error_results_do_not() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=10.0)
    assert runtime.request_queue.get_nowait() == 1

    assert runtime.accept_worker_result(WorkerResult(0, result(), None), now=10.0) is False
    assert runtime.phase == "inferencing"
    assert runtime.accept_worker_result(WorkerResult(1, result(), None), now=10.0) is True
    assert runtime.phase == "rotating"

    runtime.handle_key(" ", now=10.1)
    runtime.handle_key("n", now=10.2)
    assert runtime.accept_worker_result(
        WorkerResult(3, None, "camera timeout"), now=10.3
    ) is True
    assert runtime.phase == "final_stop"
    assert runtime.publish_due(10.3).velocity == (0.0, 0.0, 0.0)


def test_poll_worker_results_drains_queue_and_disregards_stale_generation() -> None:
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda _message: None,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=1.0)
    runtime.result_queue.put_nowait(WorkerResult(0, result(), None))

    assert runtime.poll_worker_results(now=2.0) == 0
    assert runtime.phase == "inferencing"
    assert runtime.result_queue.empty()


class RequestSequence:
    def __init__(self, *items: int | None):
        self.items = iter(items)

    def get(self) -> int | None:
        return next(self.items)


class FakeRunner:
    def __init__(self, *, error: Exception | None = None):
        self.error = error
        self.closed = False

    def run_once(self) -> ObjectNavResult:
        if self.error is not None:
            raise self.error
        return result()

    def close(self) -> None:
        self.closed = True


def test_worker_tags_result_replaces_full_slot_and_closes_runner() -> None:
    runner = FakeRunner()
    results: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)
    results.put_nowait(WorkerResult(2, result("FAILED"), None))

    run_inference_worker(
        lambda: runner,
        RequestSequence(7, None),  # type: ignore[arg-type]
        results,
    )

    item = results.get_nowait()
    assert item.generation == 7
    assert item.result == result()
    assert item.error is None
    assert runner.closed is True


def test_worker_converts_inference_exception_to_generation_tagged_error() -> None:
    runner = FakeRunner(error=RuntimeError("camera unavailable"))
    results: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)

    run_inference_worker(
        lambda: runner,
        RequestSequence(4, None),  # type: ignore[arg-type]
        results,
    )

    assert results.get_nowait() == WorkerResult(4, None, "camera unavailable")
    assert runner.closed is True


def test_publish_due_uses_20_hz_cadence_and_command_action_names() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=10.0)
    runtime.accept_worker_result(WorkerResult(1, result(), None), now=10.0)

    assert runtime.publish_due(10.0).velocity == (0.0, 0.0, 0.4)
    assert runtime.publish_due(10.049) is None
    assert runtime.publish_due(10.05).velocity == (0.0, 0.0, 0.4)
    assert [message["action"] for message in decoded_messages(published)] == [
        "turn_left",
        "turn_left",
    ]


def test_runtime_maps_negative_yaw_translation_and_stop_actions() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=0.0)
    runtime.accept_worker_result(
        WorkerResult(
            1,
            result(
                payload=commands(
                    rotation=(0.0, 0.0, -0.4, 0.05),
                    translation=(0.3, 0.0, 0.0, 0.05),
                )
            ),
            None,
        ),
        now=0.0,
    )

    for now in (0.0, 0.05, 0.55, 0.60, 0.65, 0.70):
        runtime.publish_due(now)

    assert [message["action"] for message in decoded_messages(published)] == [
        "turn_right",
        "stop",
        "move_forward",
        "stop",
        "stop",
        "stop",
    ]
    assert runtime.phase == "idle"


class RunningSequence:
    def __init__(self, *values: bool):
        self.values = iter(values)
        self.last = values[-1]

    def __call__(self) -> bool:
        self.last = next(self.values, self.last)
        return self.last


@pytest.mark.parametrize(
    ("running_values", "result_was_accepted"),
    [
        ((True, False), False),
        ((True, True, False), True),
        ((True, True, True, False), True),
    ],
)
def test_termination_gate_prevents_nonzero_after_request(
    running_values: tuple[bool, ...], result_was_accepted: bool
) -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=1.0)
    runtime.result_queue.put_nowait(WorkerResult(1, result(), None))

    run_planner_loop(
        runtime,
        read_key=lambda: None,
        monotonic=lambda: 1.0,
        sleep=lambda _duration: None,
        running=RunningSequence(*running_values),
    )

    assert all(message["action"] == "stop" for message in decoded_messages(published))
    assert runtime.result_queue.empty() is result_was_accepted


def test_keyboard_cancel_wins_when_current_worker_result_is_already_ready() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=1.0)
    runtime.result_queue.put_nowait(WorkerResult(1, result(), None))

    run_planner_loop(
        runtime,
        read_key=iter([" "]).__next__,
        monotonic=lambda: 1.0,
        sleep=lambda _duration: None,
        running=RunningSequence(True, True, True, False),
    )

    assert runtime.result_queue.empty()
    assert all(message["action"] == "stop" for message in decoded_messages(published))


@pytest.mark.parametrize("exit_mode", ["key", "signal", "exception"])
def test_event_loop_repeats_stop_on_exit_signal_and_exception(exit_mode: str) -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    if exit_mode == "key":
        keys = iter(["x"])
        read_key = lambda: next(keys)
        running = lambda: True
    elif exit_mode == "signal":
        read_key = lambda: None
        running = lambda: False
    else:
        def read_key() -> str:
            raise RuntimeError("terminal failed")

        running = lambda: True

    if exit_mode == "exception":
        with pytest.raises(RuntimeError, match="terminal failed"):
            run_planner_loop(
                runtime,
                read_key=read_key,
                monotonic=lambda: 1.0,
                sleep=lambda _duration: None,
                running=running,
            )
    else:
        run_planner_loop(
            runtime,
            read_key=read_key,
            monotonic=lambda: 1.0,
            sleep=lambda _duration: None,
            running=running,
        )

    assert_stop_messages(published)
