# Base-Pose YOLOE Split Grounding Design

## Goal

Split raw BasePose YOLOE initialization into two independent, concurrent
grounding requests. One request identifies only the task target using the
current raw-servo prompt with all support-surface/table instructions removed.
The other identifies every visible table using exactly the single-target
grounding prompt and schema used by `tools/yoloe26m/auto_refer_detect.py`, with
the target set to `table`.

## Grounding contracts

The task-target request keeps the current task, attached-image, calibration,
target-selection, normalized-box, safety, and no-motion instructions. It no
longer requests or returns `support_surface`, refers to an operation platform or
table, or makes readiness depend on a second instance. Its strict structured
output retains the target status, `primary_target`, manipulation anchor,
selection reason, confidence, and limitations.

The table request imports and calls
`tools.yoloe26m.reference_pipeline.build_grounding_prompt("table")`, uses that
module's `GROUNDING_SCHEMA`, and validates the response with
`validate_grounding_response`. This preserves the exact prompt behavior of
`auto_refer_detect.py`, including returning every visible matching physical
table as a separate normalized box.

## Concurrency and failure behavior

Both requests receive the same saved initial RGB frame. They run concurrently
through separate configured grounding-client instances so that Codex processes
and request state are independent. The selected vision backend and its existing
model, timeout, reasoning, authentication, proxy, and Qwen settings remain
unchanged.

YOLOE initialization begins only after both requests finish and validate. A
failure, `UNSURE`, `UNSAFE`, `NOT_FOUND`, empty table-box list, invalid schema,
or cancellation in either branch aborts initialization before YOLOE tracking or
robot motion starts. A completed result is still discarded if its generation
has been cancelled.

## YOLOE handoff and tracking

The reference visual prompts contain the single task-target box followed by all
validated table boxes. The target box receives class ID `0`; every table box
receives class ID `1`. The class names remain the task-target visual name and
`table`.

Initial target selection continues to match class `0` against the target box.
Initial table selection considers class `1` detections against every returned
table reference box and selects the strongest reference-box match. Persistent
tracking then continues with the selected target and table IDs exactly as in
the current controller.

## Diagnostics

The run directory keeps the initial RGB/depth images and existing YOLOE and
tracking diagnostics. Target and table grounding each save their exact prompt,
schema, and raw response under distinct filenames. The YOLOE reference artifact
records all reference boxes and corresponding class IDs so multi-table
initialization can be audited.

## Verification

Focused software-only tests verify that:

- the task prompt and schema contain no table/support-surface request while
  retaining the existing task-target instructions;
- the table prompt is exactly `build_grounding_prompt("table")` and uses the
  shared single-target schema and validator;
- two independent clients enter their requests concurrently on the same image;
- all table boxes reach YOLOE as class `1` visual prompts and participate in
  initial table-track selection;
- either grounding failure prevents tracker initialization; and
- the existing visual-servo controller and launch contracts still pass.

Verification must not send live Codex/Qwen requests, start robot control, or
require a live camera.
