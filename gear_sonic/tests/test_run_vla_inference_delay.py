import queue
import threading
import unittest
from unittest.mock import patch

import gear_sonic.utils.inference.vla.service as run_vla_inference
from gear_sonic.utils.inference.vla.service import (
    JPEG_VIDEO_MARKER,
    _drain_queue,
    _inference_worker_loop,
    _pose_policy_is_active,
    _should_schedule_vla_inference,
    prepare_observation_from_sensors,
)


class InferenceWorkerTest(unittest.TestCase):
    def test_policy_safety_warning_is_promoted_to_runtime_event(self):
        class UnsafePolicy:
            last_timing_ms = {}

            def get_action(self, _observation):
                return {"motion_token": [2.0]}, {}

        events = []

        def report(*args, **kwargs):
            events.append((args, kwargs))
            return False

        result = run_vla_inference.run_policy_inference_and_process(
            UnsafePolicy(), {}, report
        )

        self.assertIsNone(result)
        self.assertEqual(events[0][0][1], "ACTION_REJECTED")

    def test_policy_exception_is_promoted_to_runtime_event(self):
        class BrokenPolicy:
            def get_action(self, _observation):
                raise RuntimeError("policy unavailable")

        events = []
        result = run_vla_inference.run_policy_inference_and_process(
            BrokenPolicy(), {},
            lambda *args, **kwargs: events.append((args, kwargs)) or False,
        )

        self.assertIsNone(result)
        self.assertEqual(events[0][0][1], "INFERENCE_FAILED")

    def test_camera_jpeg_wrapper_keeps_bytes_and_uses_existing_protocol(self):
        payload = b"already-encoded-camera-jpeg"

        self.assertEqual(JPEG_VIDEO_MARKER, "__opencv_jpeg_rgb__")
        self.assertEqual(
            run_vla_inference.wrap_camera_jpeg_for_video(payload, (24, 32, 3)),
            {
                "__opencv_jpeg_rgb__": True,
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
            observation, timing = prepare_observation_from_sensors(
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
        self.assertIn("jpeg_prepare", timing)
        self.assertNotIn("jpeg_encode", timing)

    def test_worker_returns_timing_with_its_result(self):
        inference_queue = queue.Queue(maxsize=1)
        result_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        busy_event = threading.Event()
        inference_queue.put_nowait(7)

        worker = threading.Thread(
            target=_inference_worker_loop,
            args=(
                inference_queue,
                result_queue,
                stop_event,
                busy_event,
                lambda: ({"observation": True}, {}),
                lambda _observation: ({"action": True}, {}),
                0.0,
            ),
            daemon=True,
        )
        worker.start()
        generation, action, _started, timing = result_queue.get(timeout=1.0)
        stop_event.set()
        worker.join(timeout=1.0)

        self.assertEqual(generation, 7)
        self.assertEqual(action, {"action": True})
        self.assertGreaterEqual(timing["worker_total"], 0.0)

    def setUp(self):
        self.inference_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.busy_event = threading.Event()
        self.worker_errors = []

    def tearDown(self):
        self.stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)

    def test_worker_preserves_request_generation(self):
        def inference_fn(_observation):
            return {"motion_token": "action"}, {}

        self.thread = threading.Thread(
            target=_inference_worker_loop,
            args=(
                self.inference_queue,
                self.result_queue,
                self.stop_event,
                self.busy_event,
                lambda: ({"observation": True}, {}),
                inference_fn,
                0.0,
            ),
        )
        self.thread.start()
        self.inference_queue.put_nowait(7)

        generation, action, _inference_start_time, _timing = self.result_queue.get(timeout=0.5)
        self.assertEqual(generation, 7)
        self.assertEqual(action, {"motion_token": "action"})

    def test_pose_policy_requires_running_unpaused_pose_mode(self):
        self.assertTrue(_pose_policy_is_active(True, "POSE", False))
        self.assertFalse(_pose_policy_is_active(True, "POSE", True))
        self.assertFalse(_pose_policy_is_active(True, "PLANNER", False))
        self.assertFalse(_pose_policy_is_active(False, "POSE", False))

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
            _should_schedule_vla_inference(
                cpp_mode="POSE",
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
