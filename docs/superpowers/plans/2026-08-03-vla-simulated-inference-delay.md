# VLA Simulated Inference Delay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hold each successful VLA inference result for an additional fixed 0.2 seconds while the main loop continues publishing the previous action chunk, then let the existing latency compensation select a correspondingly later trajectory point.

**Architecture:** Add a script-level production delay constant and keep the result inside `_inference_worker_loop` until an interruptible stop-event wait expires. Preserve the original inference start timestamp so the existing main-loop latency calculation automatically includes the simulated delay; expose only a worker argument with the production constant as its default so tests can run with a short delay without adding a CLI setting.

**Tech Stack:** Python 3.10+, standard-library `queue`, `threading`, `time`, and `unittest`; NumPy through the existing inference environment.

## Global Constraints

- The production simulated inference delay is fixed at exactly 0.2 seconds in `run_vla_inference.py`.
- Do not add a CLI option and do not modify `gear_sonic/scripts/launch_inference.py`.
- Apply the hold only to successful, non-`None` inference results.
- Keep `busy_event` set for the entire hold so no new inference is scheduled during it.
- Continue publishing the previously cached chunk from the main loop during the hold.
- Use `stop_event.wait(delay)` so shutdown interrupts the hold and drops the pending result.
- Preserve the original `inference_start_time`; do not add a second latency or index formula.
- Preserve all unrelated working-tree changes.

---

## File map

- `gear_sonic/scripts/run_vla_inference.py`: owns the fixed production delay and holds completed worker results before queue publication.
- `gear_sonic/tests/test_run_vla_inference_delay.py`: exercises result visibility, busy-state lifetime, shutdown interruption, timestamp preservation, and the existing 0.2-second/50-Hz index relationship.

---

### Task 1: Hold successful inference results inside the worker

**Files:**
- Create: `gear_sonic/tests/test_run_vla_inference_delay.py`
- Modify: `gear_sonic/scripts/run_vla_inference.py:68-70,354-399`

**Interfaces:**
- Produces: `SIMULATED_INFERENCE_DELAY_SECONDS: float = 0.2` at script scope.
- Extends: `_inference_worker_loop(..., simulated_inference_delay_seconds: float = SIMULATED_INFERENCE_DELAY_SECONDS) -> None`.
- Preserves: `result_queue` items as `(processed_action, inference_start_time)`.
- Consumes: existing `calculate_latency_compensated_index(inference_delay, control_freq, action_horizon) -> int` unchanged.

- [ ] **Step 1: Write failing worker timing tests**

Create `gear_sonic/tests/test_run_vla_inference_delay.py` with the following behavior-focused test fixture and cases:

```python
import queue
import threading
import time
import unittest

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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
.venv_inference/bin/python -m unittest gear_sonic.tests.test_run_vla_inference_delay -v
```

Expected: ERROR importing `SIMULATED_INFERENCE_DELAY_SECONDS`, proving the production delay feature is absent.

- [ ] **Step 3: Add the minimal interruptible result hold**

In `gear_sonic/scripts/run_vla_inference.py`, define the fixed constant near the other module constants:

```python
SIMULATED_INFERENCE_DELAY_SECONDS = 0.2
```

Extend the worker signature without changing existing callers:

```python
def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
    simulated_inference_delay_seconds: float = SIMULATED_INFERENCE_DELAY_SECONDS,
):
```

Immediately after `processed_action = inference_fn(observation)`, hold only a successful result and stop without queueing if shutdown interrupts the wait:

```python
if processed_action is not None:
    if stop_event.wait(simulated_inference_delay_seconds):
        continue
    try:
        result_queue.put_nowait((processed_action, inference_start_time))
```

Keep this code inside the existing `try` whose `finally` clears `busy_event`, so the event stays set throughout the hold. Do not change the queue replacement behavior or main-loop compensation code.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
.venv_inference/bin/python -m unittest gear_sonic.tests.test_run_vla_inference_delay -v
```

Expected: all 3 tests PASS, with no worker exceptions or pending threads.

- [ ] **Step 5: Run nearby utility tests and static checks**

Run:

```bash
.venv_inference/bin/python -m compileall -q gear_sonic/scripts/run_vla_inference.py gear_sonic/utils/inference/vla_utils.py gear_sonic/tests/test_run_vla_inference_delay.py
.venv_inference/bin/python -m unittest discover -s gear_sonic/tests -p 'test_run_vla*.py' -v
git diff --check -- gear_sonic/scripts/run_vla_inference.py gear_sonic/tests/test_run_vla_inference_delay.py
```

Expected: compilation succeeds, all discovered VLA tests pass, and `git diff --check` produces no output.

- [ ] **Step 6: Review the scoped diff and commit**

Run:

```bash
git diff -- gear_sonic/scripts/run_vla_inference.py gear_sonic/tests/test_run_vla_inference_delay.py
git status --short
git add gear_sonic/scripts/run_vla_inference.py gear_sonic/tests/test_run_vla_inference_delay.py
git commit -m "feat: simulate VLA inference result delay"
```

Expected: the commit contains only the inference script and its new test; pre-existing unrelated modifications remain unstaged.
