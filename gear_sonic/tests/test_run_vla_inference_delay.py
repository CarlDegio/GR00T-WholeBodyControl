import queue
import threading
import time
import unittest
from unittest.mock import patch

import gear_sonic.scripts.run_vla_inference as run_vla_inference
from gear_sonic.scripts.run_vla_inference import (
    JPEG_VIDEO_MARKER,
    SIMULATED_INFERENCE_DELAY_SECONDS,
    _drain_queue,
    _inference_worker_loop,
    _pose_policy_is_active,
    prepare_observation_from_sensors,
    _should_schedule_vla_inference,
    _vla_inference_is_due,
)
from gear_sonic.utils.inference.vla_utils import calculate_latency_compensated_index


class InferenceWorkerDelayTest(unittest.TestCase):
    def test_camera_jpeg_wrapper_keeps_bytes_and_uses_existing_protocol(self):
        payload = b"already-encoded-camera-jpeg"

        self.assertEqual(
            run_vla_inference.wrap_camera_jpeg_for_video(payload, (24, 32, 3)),
            {
                JPEG_VIDEO_MARKER: True,
                "shape": (1, 1, 24, 32, 3),
                "dtype": "uint8",
                "data": payload,
            },
        )

    def test_prepare_observation_wraps_undecodable_camera_jpegs_without_codec(self):
        image_names = ("ego_view", "chest_view", "left_wrist", "right_wrist")
        payloads = {
            name: f"not-a-jpeg-{name}".encode() for name in image_names
        }
        image_shapes = {
            "ego_view": (24, 32, 3),
            "chest_view": (25, 33, 3),
            "left_wrist": (26, 34, 3),
            "right_wrist": (27, 35, 3),
        }

        class FakeGateway:
            def read_camera(self):
                return {
                    "images": payloads,
                    "image_shapes": image_shapes,
                    "timestamps": {name: [0.0] for name in image_names},
                }

            def read_state(self):
                return {
                    "body_q": [],
                    "left_hand_q": [],
                    "right_hand_q": [],
                    "base_quat": [1.0, 0.0, 0.0, 0.0],
                }

        class FakeRobotModel:
            def get_configuration_from_actuated_joints(self, **_kwargs):
                return [0.0]

        with patch.object(
            run_vla_inference,
            "prepare_observation_for_eval",
            side_effect=lambda _robot_model, observation: observation,
        ):
            observation = prepare_observation_from_sensors(
                FakeGateway(), FakeRobotModel(), "test prompt"
            )

        self.assertEqual(list(observation["video"]), list(image_names))
        for name in image_names:
            self.assertEqual(
                observation["video"][name],
                {
                    JPEG_VIDEO_MARKER: True,
                    "shape": (1, 1, *image_shapes[name]),
                    "dtype": "uint8",
                    "data": payloads[name],
                },
            )
        self.assertIn("jpeg_prepare", observation.timing_ms)
        self.assertNotIn("jpeg_encode", observation.timing_ms)

    def test_worker_emits_best_effort_timing_without_changing_result_contract(self):
        inference_queue = queue.Queue(maxsize=1)
        result_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        busy_event = threading.Event()
        samples = []
        inference_queue.put_nowait(7)

        worker = threading.Thread(
            target=_inference_worker_loop,
            args=(
                inference_queue,
                result_queue,
                stop_event,
                busy_event,
                lambda: {"observation": True},
                lambda _observation: {"action": True},
                0.0,
            ),
            kwargs={"timing_callback": samples.append},
            daemon=True,
        )
        worker.start()
        generation, action, _started = result_queue.get(timeout=1.0)
        stop_event.set()
        worker.join(timeout=1.0)

        self.assertEqual(generation, 7)
        self.assertEqual(action, {"action": True})
        self.assertEqual(len(samples), 1)
        self.assertGreaterEqual(samples[0]["worker_total"], 0.0)

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

        generation, action, inference_start_time = self.result_queue.get(timeout=0.5)

        self.assertEqual(generation, 0)
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

    def test_production_has_no_simulated_inference_delay(self):
        self.assertEqual(SIMULATED_INFERENCE_DELAY_SECONDS, 0.0)
        without_delay = calculate_latency_compensated_index(0.1, 50, 70)
        with_delay = calculate_latency_compensated_index(
            0.1 + SIMULATED_INFERENCE_DELAY_SECONDS, 50, 70
        )
        self.assertEqual(with_delay, without_delay)

    def test_worker_preserves_request_generation(self):
        def inference_fn(_observation):
            return {"motion_token": "action"}

        self.thread = threading.Thread(
            target=_inference_worker_loop,
            args=(
                self.inference_queue,
                self.result_queue,
                self.stop_event,
                self.busy_event,
                lambda: {"observation": True},
                inference_fn,
                0.0,
            ),
        )
        self.thread.start()
        self.inference_queue.put_nowait(7)

        generation, action, _inference_start_time = self.result_queue.get(timeout=0.5)
        self.assertEqual(generation, 7)
        self.assertEqual(action, {"motion_token": "action"})

    def test_pose_policy_requires_running_unpaused_pose_mode(self):
        self.assertTrue(_pose_policy_is_active(True, "POSE", False))
        self.assertFalse(_pose_policy_is_active(True, "POSE", True))
        self.assertFalse(_pose_policy_is_active(True, "PLANNER", False))
        self.assertFalse(_pose_policy_is_active(False, "POSE", False))

    def test_vla_inference_timing_predicate(self):
        self.assertTrue(
            _vla_inference_is_due(
                worker_is_busy=False,
                request_queue_is_empty=True,
                time_since_request=0.5,
                inference_interval=0.5,
            )
        )
        self.assertFalse(
            _vla_inference_is_due(
                worker_is_busy=True,
                request_queue_is_empty=True,
                time_since_request=10.0,
                inference_interval=0.5,
            )
        )

    def test_vla_observation_is_not_scheduled_outside_active_pose(self):
        common = dict(
            worker_is_busy=False,
            request_queue_is_empty=True,
            time_since_request=1.0,
            inference_interval=0.5,
        )
        self.assertTrue(
            _should_schedule_vla_inference(
                cpp_mode="POSE",
                **common,
            )
        )
        self.assertFalse(
            _should_schedule_vla_inference(
                cpp_mode="PLANNER",
                **common,
            )
        )
        self.assertFalse(
            _should_schedule_vla_inference(
                cpp_mode="OFF",
                **common,
            )
        )
        # POSE pause gates robot action publication, not observation requests.
        self.assertTrue(
            _should_schedule_vla_inference(
                cpp_mode="POSE",
                **common,
            )
        )
        self.assertFalse(
            _vla_inference_is_due(
                worker_is_busy=False,
                request_queue_is_empty=False,
                time_since_request=10.0,
                inference_interval=0.5,
            )
        )

    def test_drain_queue_removes_all_pending_items(self):
        self.inference_queue.put_nowait(1)
        self.result_queue.put_nowait(2)
        self.assertEqual(_drain_queue(self.inference_queue), 1)
        self.assertEqual(_drain_queue(self.result_queue), 1)
        self.assertTrue(self.inference_queue.empty())
        self.assertTrue(self.result_queue.empty())


if __name__ == "__main__":
    unittest.main()
