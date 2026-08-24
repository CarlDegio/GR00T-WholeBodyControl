# Uni-LaViRA Agent runtime

The runtime keeps robot motion on the existing safety path:

```text
head + chest views -> cloud LaViRA LA/VA -> ControlGateway
                   -> panorama / NavDP / BasePose / VLA -> SONIC
```

LaViRA emits skill intents, relative heading goals, and two-dimensional local
goals. It does not emit velocity. Press `N` to start the manipulation task
fixed in the runtime YAML; Space invalidates the generation and stops motion
immediately.

## Cloud model endpoints

The default profile uses Alibaba Cloud Model Studio's OpenAI-compatible
endpoint for two separate roles:

```yaml
components:
  lavira:
    la_model: qwen3.8-max
    la_base_url: https://ws-6yzgj1m087a053ip.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
    la_enable_thinking: false
    va_model: qwen3.5-27b
    va_base_url: https://ws-6yzgj1m087a053ip.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
    va_enable_thinking: false
    nav_handoff_min_depth_m: 0.3
    nav_handoff_max_depth_m: 3.0
    alignment_head_camera_stream: ego_view
    heading_settle_seconds: 1.0
    heading_settle_samples: 30
    heading_settle_bad_sample_threshold: 12
    heading_settle_tolerance_rad: 0.08726646259971647
    heading_correction_speed_rad_s: 0.2
    heading_correction_timeout_seconds: 10.0
```

Set `DASHSCOPE_API_KEY` for both roles. `LAVIRA_LA_API_KEY` and
`LAVIRA_VA_API_KEY` are optional role-specific overrides. Startup stops before
launching workers if neither the shared key nor a required role override is
available. The default profile disables thinking for both LA planning and VA's
bounded, schema-constrained grounding/postchecks. The two behaviors remain
independent YAML booleans: set `la_enable_thinking` or
`va_enable_thinking` to `true` or `false` without changing code.

## Task selection

Every task combines a navigation instruction with a subsequent manipulation
task. `components.lavira.navigation_mode` selects the navigation prompt and
must be `vln` or `object_nav`. `mission` contains the navigation instruction,
and `manipulation_prompt` is the exact static language prompt used by VLA after
the handoff. The first LA response derives `global_target` from those two task
texts; the runtime freezes it for that task generation and rejects any later LA
response that changes it. The default profile uses one YAML anchor so the
LaViRA and VLA copies cannot drift apart. The default limit is 20 Agent steps.

```yaml
components:
  lavira:
    navigation_mode: vln
    mission: Walk forward to the trash can, then turn right and walk to a position near the desk with the blue basket on it.
    global_target: ''  # Filled by the first LA response for each generation.
    manipulation_prompt: Move in front of the desk with the blue basket, grasp the medicine bottle, and place it into the blue basket.
```

## Cloud request context

Every panorama Agent step captures fresh `front`, `front_right`, `right`,
`left`, and `front_left` chest views using the sequence: capture front, turn
right 45 degrees and capture, turn right 45 degrees and capture, turn left 180
degrees and capture, turn right 45 degrees and capture, then turn right 45
degrees to return to front. The middle 180-degree transition carries a forced
left-turn constraint through NavDP so wraparound at the +/-180-degree boundary
cannot change it into a right turn. Every scan heading is anchored and corrected
with fresh measured SONIC yaw; Fast-LIO yaw is not used for scan turns. NavDP turns
at 0.4 rad/s, slows to 0.2 rad/s inside 20 degrees, and reports arrival inside
5 degrees. After each arrival it holds zero velocity while LaViRA samples 30
SONIC yaw values over one second. If at least 12 samples exceed 5 degrees,
LaViRA requests target-relative corrections capped at 0.2 rad/s. Fine
adjustment lasts at most 10 seconds; at the limit NavDP sends zero and LaViRA
captures the image before continuing the scan. The runtime then applies
fixed handoff contracts instead of accepting LA's free-form expected
postcondition as a stage boundary.
Navigation continuation (`CONTINUE_NAVIGATION`, `RETURN_TO_NAVIGATION`, or
navigation `UNKNOWN`) makes the next LA step capture another panorama.
Near-field transitions (`READY_TO_ALIGN`, `RETRY_ALIGN`,
`READY_TO_MANIPULATE`, or ALIGN `UNKNOWN`) make the next LA step capture only
one fresh fixed-front RGB, without a five-direction scan.

Each LA skill-decision request contains the navigation instruction, the static
manipulation task, the one-based current Agent step, the
complete current Markdown TODO (or an explicit no-plan-yet marker on step 1),
the current panorama or fixed-front observation, the latest VA transition
result, and at most five successful MOVE_TO result images. The
request content follows the G1 ordering: navigation-task/step header first,
`PLAN-N` history images second, current views third, and the decision prompt
last. There is no separate initial-TODO inference. The first LA request derives
and returns `global_target`, creates the working Markdown checklist, and selects
the first skill in one response. Later requests receive the frozen value and
must return it unchanged while revising the checklist and selecting the next
skill. MOVE_TO captures fresh chest and
head RGB-D observations after its issued NavDP goal reaches. Both views are
captured before either cloud check, and the view that makes the NAV handoff
ready is retained for LA history.
The LA system message also states the runtime contract explicitly: ALIGN is
forbidden until a successful MOVE_TO to the exact `global_target` has produced
`READY_TO_ALIGN`; an intermediate route landmark cannot authorize ALIGN.
Failed grounding, controller failure, invalid/out-of-range depth, and an
unsatisfied or unknown handoff do not create a MOVE_TO history image. Text
skill history and Fast-LIO exploration memory stay local and are not uploaded
as LA context.

Current-view labels use the G1 wording (`Image 1: ...`, through `Image 5: ...`)
and contain only the relative direction and Agent step, never Fast-LIO yaw.
MOVE_TO history images are labeled only `PLAN-N`; target names, skill IDs,
controller states, VA results, and evidence are not exposed in their labels.

LA chooses only `MOVE_TO`, `ALIGN`, `MANIPULATE`, or `FAIL`. MOVE_TO carries a
`view_direction` tied exclusively to the current panorama and a semantic
`target`; it has no target-role/type field. The runtime turns to the absolute
heading represented by that panorama direction, captures a new image, asks VA
to ground the target, and only then publishes one NavDP goal. Like the G1
prompt, every Agent step may choose only `front`, `front_right`, `right`,
`left`, or `front_left`. The two `front_*` directions represent the intermediate
45-degree views; MOVE_TO turns to that absolute SONIC yaw before fresh VA
grounding. No rear image is captured, and `behind` is rejected by the runtime
schema on every step.

ALIGN has empty skill arguments and always uses the front direction. Its VA
`ALIGN_GROUNDING` request asks VA to enumerate every object required by the
manipulation task, whether visible or not. Each entry contains its name,
visibility, bbox, confidence, and complete concrete supporting object. Surface
objects are not allowed as separate operation-object entries; navigation
landmarks and the frozen navigation `global_target` are also excluded unless
they independently appear as operated objects in the manipulation task.
Abstract surfaces such as `tabletop`, `desk surface`, or `plane` fail
validation and trigger the normal response retry. The runtime ignores model
ordering, filters visible candidates below one percent of the image, and
deterministically selects the remaining candidate with the largest bbox for
BasePose. After BasePose returns, ALIGN reuses the existing
VA `POSTCHECK` interface twice: once with a fresh chest RGB and once with a
fresh head RGB. Each request independently asks whether every operation object
required by the manipulation task is simultaneously visible in that single
camera view. Partial visibility cannot be combined across cameras. ALIGN is
ready only when BasePose reports `aligned` and either the chest or head result
is `SATISFIED`. MANIPULATE has empty arguments and requests VLA execution only
after that handoff. Sensor freshness and controller ownership remain
independent safety gates.

Each VA request still contains exactly one image and reuses the existing
`GROUNDING`, `ALIGN_GROUNDING`, or `POSTCHECK` schema. Navigation and ordinary
LA observations use chest RGB. MOVE_TO handoff sends two separate GROUNDING
requests with aligned depth: chest uses its leased metric Depth Anything frame,
while head uses `camera/ego_view_depth`. ALIGN handoff similarly sends two
separate POSTCHECK requests. Neither the API nor the saved request-context
format needs a new dual-image schema. Every request also carries the
relevant navigation or manipulation instruction, `global_target`, LA strategic
goal, and `strategic_stop`. VA
never selects a skill or exploration direction and is not told which skill just
ran.

The runtime owns the effective NAV/ALIGN transition:

- NAV calls fresh `GROUNDING` independently on the post-MOVE_TO chest and head
  RGB-D views. A view passes when the requested target is visible and the mean
  of finite, positive depth values in its bbox lies in
  `nav_handoff_min_depth_m..nav_handoff_max_depth_m`. `READY_TO_ALIGN` requires
  all three harness conditions: the navigation controller completed, the
  MOVE_TO target matches `global_target`, and either complete camera view
  passes. A passing intermediate landmark remains `CONTINUE_NAVIGATION` and
  clears ALIGN readiness. Values over 8 m are discarded. If neither view
  passes, missing/unavailable depth or a camera/check error produces `UNKNOWN`;
  two available but missing/out-of-range results produce `CONTINUE_NAVIGATION`.
- ALIGN combines the two one-image POSTCHECK statuses with OR. It produces
  `READY_TO_MANIPULATE` only when one complete camera view contains all
  operation objects and BasePose is aligned. If neither view is complete it
  produces `RETRY_ALIGN`, or `RETURN_TO_NAVIGATION` when the alignment target
  could not be grounded.
- MANIPULATE uses the visual POSTCHECK `status` as evidence and derives its
  transition deterministically: `NOT_SATISFIED` continues manipulation,
  `SATISFIED` completes the task, and `UNKNOWN` remains unknown. The request
  includes this restricted transition contract so a navigation/alignment
  transition cannot terminate manipulation.

The transition vocabulary remains:

- MOVE_TO: `CONTINUE_NAVIGATION`, `READY_TO_ALIGN`, or `UNKNOWN`.
- ALIGN: `RETRY_ALIGN`, `RETURN_TO_NAVIGATION`, `READY_TO_MANIPULATE`, or
  `UNKNOWN`.
- MANIPULATE: `CONTINUE_MANIPULATION`, `TASK_COMPLETE`, or `UNKNOWN`.

LA still issues the next skill call; readiness alone never switches a
ControlGateway mode. The harness gates only forward handoffs: a ready NAV may
either continue with another MOVE_TO or accept ALIGN, while MANIPULATE requires
`READY_TO_MANIPULATE`. MOVE_TO and ALIGN recovery remain repeatable. VA does not
directly switch ControlGateway modes. All image context and task-local state are
cleared when a new task generation starts. VLA receives exactly
`components.lavira.manipulation_prompt` at task start. During MANIPULATE, VA
postchecks run on the LaViRA decision thread without sending `hold_vla_task` or
`resume_vla_task`; the independent VLA service keeps its 50 Hz POSE action
stream continuous. Only completion, safety failure, timeout, or cancellation
stops VLA. VA incomplete evidence is not appended to the VLA language prompt.

LA emits only `EXECUTE` or `FAIL`; it does not make a final `COMPLETE`
decision. Once MANIPULATE returns the paired `SATISFIED` / `TASK_COMPLETE` VA
postcheck, the runtime emits the completed skill event and directly finalizes
the task without another LA request.

Every LA and VA request is archived before transmission under
`outputs/logs/inference/lavira_requests/`, alongside the inference component
logs. Each physical API attempt gets one JSON file recording the request kind,
attempt number, model parameters, exact message content, prompts, and embedded
data-URL images; API keys are not part of the request body and are not written.
Transport failures and responses that fail strict JSON or role-specific schema
validation are retried up to three total attempts inside the same inference
call. A format-validation retry adds an explicit strict-JSON correction to the
system message. No skill is returned to the runtime—and therefore no physical
action is issued—until a response passes validation; the task fails closed only
after all three attempts fail.
The filename contains a nanosecond timestamp, process ID, monotonic per-process
sequence, request kind, and attempt number.

The ControlGateway Events tmux pane reserves roughly its top quarter for the
current Markdown TODO and the remaining area for recent runtime events. The
TODO area shows generation, Agent step, completed count, completed items, the
first active item, and later pending items; long lists are clipped around the
active item with a hidden-line count. With no active task it shows a start hint,
and while the first combined LA plan/action request is running it shows a
waiting hint. TODO updates are
structured `TODO_UPDATED` runtime events that update the fixed area instead of
being duplicated in the scrolling event list. Complete event history remains
available in the component log files.

Production validation should progress from observation-only logging, to a
30-degree-limited turn, to a 0.5 m local goal, and finally a complete task.
Record the requested turn angle, accumulated SONIC global-yaw change, NavDP
segment status, and the final stopped state.

Prompt and orchestration portions are adapted from Uni-LaViRA commit `215e7aca`
under CC BY-NC-SA 4.0. See the repository third-party notice for attribution
and the non-commercial/share-alike obligations.
