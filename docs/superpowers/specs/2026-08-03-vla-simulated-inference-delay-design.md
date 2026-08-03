# VLA Simulated Inference Delay Design

## Goal

Make `gear_sonic/scripts/run_vla_inference.py` behave as though every VLA
request takes an additional 0.2 seconds. During that interval, the controller
must continue publishing the previous action chunk. When the new chunk becomes
available, its latency-compensated starting index must include the simulated
delay.

## Scope

- Define a script-level fixed delay of 0.2 seconds. Do not add a CLI option or
  modify `launch_inference.py`.
- Apply the delay only to successful, non-`None` inference results before they
  enter the result queue.
- Preserve all existing action publishing, inference scheduling, and latency
  compensation behavior outside this delay.

## Design

The inference worker records `inference_start_time`, runs the existing inference
function, and then holds a successful result for 0.2 seconds before placing it
on `result_queue`. The worker's `busy_event` remains set throughout the hold.
Consequently, the main loop keeps publishing the previous cached chunk and does
not schedule another inference during the simulated delay.

The hold uses the worker's stop event rather than an unconditional sleep so
shutdown can interrupt it. If shutdown occurs during the hold, the worker drops
the pending result and exits without publishing it.

No new compensation formula is needed. The main loop already computes elapsed
time from `inference_start_time` until it consumes the result. Because queue
publication occurs after the hold, measured latency automatically includes the
additional 0.2 seconds. At the default 50 Hz action rate, this advances the
starting index by approximately 10 trajectory points, subject to the existing
rounding and horizon clipping.

## Testing

Add focused worker tests with a shorter injected test delay while production
continues to use the fixed 0.2-second constant. Verify that:

1. a completed inference result is not visible before the hold expires;
2. `busy_event` remains set while the result is held;
3. the result becomes available after the hold with its original inference
   start timestamp; and
4. stopping during the hold releases the worker promptly without publishing the
   pending result.

Retain or add a direct latency-index assertion showing that an additional 0.2
seconds at 50 Hz advances the selected point by 10, unless horizon clipping
applies.

## Success Criteria

- The main loop continues using the old chunk for about 0.2 seconds after the
  policy returns.
- A newly released chunk starts about 10 points later at 50 Hz than it would
  without the simulated delay.
- Shutdown remains responsive.
- Existing unrelated working-tree changes are preserved.
