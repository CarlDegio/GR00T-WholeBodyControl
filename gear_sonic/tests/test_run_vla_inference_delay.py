import queue
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gear_sonic.scripts.run_vla_inference import (
    SIMULATED_INFERENCE_DELAY_SECONDS,
    _execute_cpp_control_toggle,
    _inference_worker_loop,
    _planner_message_has_frozen_targets,
    wait_for_planner_hold_ready,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message
from gear_sonic.utils.inference.vla_utils import calculate_latency_compensated_index


class InferenceWorkerDelayTest(unittest.TestCase):
    def setUp(self):
        self.inference_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.busy_event = threading.Event()
        self.inference_finished = threading.Event()
        self.inference_returned_at = None
        self.worker_errors = []

    def tearDown(self):
        self.stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)

    def start_worker(self, delay):
        def inference_fn(_observation):
            self.inference_returned_at = time.monotonic()
            self.inference_finished.set()
            return {"motion_token": "action"}

        def run_worker():
            try:
                _inference_worker_loop(
                    self.inference_queue,
                    self.result_queue,
                    self.stop_event,
                    self.busy_event,
                    lambda: {"observation": True},
                    inference_fn,
                    simulated_inference_delay_seconds=delay,
                )
            except BaseException as error:
                self.worker_errors.append(error)

        self.thread = threading.Thread(target=run_worker)
        self.thread.start()
        self.inference_queue.put_nowait(None)

    def test_successful_result_is_held_while_worker_remains_busy(self):
        delay = 0.08
        self.start_worker(delay)
        self.assertTrue(self.inference_finished.wait(timeout=0.5))
        self.assertTrue(self.busy_event.is_set())
        with self.assertRaises(queue.Empty):
            self.result_queue.get_nowait()

        action, inference_start_time = self.result_queue.get(timeout=0.5)

        self.assertGreaterEqual(time.monotonic() - self.inference_returned_at, delay)
        self.assertEqual(action, {"motion_token": "action"})
        self.assertLessEqual(inference_start_time, self.inference_returned_at)
        self.assertEqual(self.worker_errors, [])

    def test_shutdown_interrupts_hold_and_drops_pending_result(self):
        self.start_worker(delay=2.0)
        self.assertTrue(self.inference_finished.wait(timeout=0.5))
        self.assertTrue(self.busy_event.is_set())

        self.stop_event.set()
        self.thread.join(timeout=0.5)

        self.assertFalse(self.thread.is_alive())
        self.assertFalse(self.busy_event.is_set())
        with self.assertRaises(queue.Empty):
            self.result_queue.get_nowait()
        self.assertEqual(self.worker_errors, [])

    def test_production_delay_advances_50_hz_index_by_ten_points(self):
        self.assertEqual(SIMULATED_INFERENCE_DELAY_SECONDS, 0.2)
        without_delay = calculate_latency_compensated_index(0.1, 50, 70)
        with_delay = calculate_latency_compensated_index(
            0.1 + SIMULATED_INFERENCE_DELAY_SECONDS, 50, 70
        )
        self.assertEqual(with_delay - without_delay, 10)

    def test_base_pose_k_starts_before_request_and_second_k_stops(self):
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "hold.ready"
            request = Path(f"{marker}.request")
            events = []

            def send_control(start, planner):
                events.append((start, planner, request.exists()))
                return True

            self.assertEqual(
                _execute_cpp_control_toggle(
                    cpp_loop_running=False,
                    cpp_mode="OFF",
                    planner_hold_ready_file=str(marker),
                    send_control_command=send_control,
                ),
                "started",
            )
            self.assertEqual(events, [(True, True, False)])
            self.assertTrue(request.is_file())

            marker.write_text("ready\n", encoding="utf-8")
            self.assertEqual(
                _execute_cpp_control_toggle(
                    cpp_loop_running=True,
                    cpp_mode="PLANNER",
                    planner_hold_ready_file=str(marker),
                    send_control_command=send_control,
                ),
                "stopped",
            )
            self.assertEqual(events[-1], (False, True, True))
            self.assertFalse(marker.exists())
            self.assertFalse(request.exists())

    def test_planner_k_requests_a_fresh_hold_before_accepting_ready(self):
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "hold.ready"

            def acknowledge(_seconds):
                self.assertTrue(Path(f"{marker}.request").is_file())
                marker.write_text("ready\n")

            self.assertTrue(
                wait_for_planner_hold_ready(
                    str(marker),
                    5.0,
                    monotonic=lambda: 0.0,
                    sleep=acknowledge,
                )
            )

    def test_planner_k_timeout_removes_unhandled_latch_request(self):
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "hold.ready"
            now = [0.0]

            def monotonic():
                return now[0]

            def advance(seconds):
                now[0] += seconds

            self.assertFalse(
                wait_for_planner_hold_ready(
                    str(marker), 0.1, monotonic=monotonic, sleep=advance
                )
            )
            self.assertFalse(Path(f"{marker}.request").exists())

    def test_planner_k_accepts_only_a_relay_frame_with_body_and_both_hands(self):
        idle = build_planner_message(0, [0.0] * 3, [1.0, 0.0, 0.0])
        frozen = build_planner_message(
            0,
            [0.0] * 3,
            [1.0, 0.0, 0.0],
            upper_body_position=[0.0] * 17,
            upper_body_velocity=[0.0] * 17,
            left_hand_position=[0.0] * 7,
            right_hand_position=[0.0] * 7,
        )

        self.assertFalse(_planner_message_has_frozen_targets(idle))
        self.assertTrue(_planner_message_has_frozen_targets(frozen))


if __name__ == "__main__":
    unittest.main()
