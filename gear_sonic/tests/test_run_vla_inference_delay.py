from dataclasses import fields
import queue
import threading
import time
import unittest

from gear_sonic.scripts import run_vla_inference as inference_module
from gear_sonic.scripts.run_vla_inference import (
    SIMULATED_INFERENCE_DELAY_SECONDS,
    _inference_worker_loop,
)
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

    def test_base_pose_control_exposes_no_hold_ready_state(self):
        config_fields = {
            field.name for field in fields(inference_module.InferenceConfig)
        }
        self.assertNotIn("planner_hold_ready_file", config_fields)
        self.assertNotIn("planner_hold_ready_timeout_seconds", config_fields)
        for removed_helper in (
            "_clear_planner_hold",
            "_request_planner_hold",
            "_execute_cpp_control_toggle",
            "wait_for_planner_hold_ready",
            "_planner_message_has_frozen_targets",
            "_forward_frozen_planner_hold",
        ):
            self.assertFalse(
                hasattr(inference_module, removed_helper), removed_helper
            )


if __name__ == "__main__":
    unittest.main()
