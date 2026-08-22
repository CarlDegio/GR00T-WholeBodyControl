# Uni-LaViRA Agent runtime

The runtime keeps robot motion on the existing safety path:

```text
chest_view -> cloud LaViRA LA/VA -> ControlGateway
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
    la_base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    la_enable_thinking: true
    va_model: qwen3.5-27b
    va_base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    va_enable_thinking: false
    nav_handoff_min_depth_m: 0.3
    nav_handoff_max_depth_m: 3.0
    alignment_head_camera_stream: ego_view
```

Set `LAVIRA_LA_API_KEY` and `LAVIRA_VA_API_KEY`; the same Model Studio key may
be used for both. LA enables thinking for task decomposition and skill planning;
VA disables it for bounded, schema-constrained grounding and postchecks. Both
behaviors are independent YAML booleans: set `la_enable_thinking` or
`va_enable_thinking` to `true` or `false` without changing code.

## Task selection

Every task is a manipulation task. `components.lavira.navigation_mode` selects
only the navigation prompt and must be `vln` or `object_nav`. `mission` contains
the complete original task and `global_target` names the final navigation
target. The default limit is 20 Agent steps.

```yaml
components:
  lavira:
    navigation_mode: object_nav
    mission: Find the blue basket, then put the medicine bottle into it.
    global_target: blue basket
```

## Cloud request context

The first Agent step captures fresh `front, right, behind, left` chest views and
returns to its front heading. The runtime then applies fixed handoff contracts
instead of accepting LA's free-form expected postcondition as a stage boundary.
Navigation continuation (`CONTINUE_NAVIGATION`, `RETURN_TO_NAVIGATION`, or
navigation `UNKNOWN`) makes the next LA step capture another panorama.
Near-field transitions (`READY_TO_ALIGN`, `RETRY_ALIGN`,
`READY_TO_MANIPULATE`, or ALIGN `UNKNOWN`) make the next LA step capture only
one fresh fixed-front RGB, without a four-direction rotation.

Each LA skill-decision request contains the complete original mission,
`global_target`, the one-based current Agent step, the complete current
Markdown TODO, the current panorama or fixed-front observation, the latest VA
transition result, and at most five successful MOVE_TO result images. The
request content follows the G1 ordering: navigation-task/step header first,
`PLAN-N` history images second, current views third, and the decision prompt
last. The initial TODO request also keeps
the original G1 instruction preamble before its four starting views. The initial
TODO generator and first skill decision reuse the first panorama. A
MOVE_TO image is captured from a fresh leased RGB-D observation after its
issued NavDP goal reaches and is retained only when the NAV handoff is ready.
Failed grounding, controller failure, invalid/out-of-range depth, and an
unsatisfied or unknown handoff do not create a MOVE_TO history image. Text
skill history and Fast-LIO exploration memory stay local and are not uploaded
as LA context.

Current-view labels use the G1 wording (`Image 1: ...`, through `Image 4: ...`)
and contain only the relative direction and Agent step, never Fast-LIO yaw.
MOVE_TO history images are labeled only `PLAN-N`; target names, skill IDs,
controller states, VA results, and evidence are not exposed in their labels.

LA chooses only `MOVE_TO`, `ALIGN`, `MANIPULATE`, or `FAIL`. MOVE_TO carries a
`view_direction` tied exclusively to the current panorama and a semantic
`target`; it has no target-role/type field. The runtime turns to the absolute
heading represented by that panorama direction, captures a new image, asks VA
to ground the target, and only then publishes one NavDP goal. Like the G1
prompt, `behind` is available only on Agent step 1; later navigation steps may
choose only `front`, `left`, or `right`.

ALIGN has empty skill arguments and always uses the front direction. Its VA
`ALIGN_GROUNDING` request decomposes the complete original mission to derive
the dynamic BasePose `target` and `surface`, while also locating the target
bbox in one fresh front RGB. After BasePose returns, ALIGN reuses the existing
VA `POSTCHECK` interface twice: once with a fresh chest RGB and once with a
fresh head RGB. Each request independently asks whether every operation object
required by the original mission is simultaneously visible in that single
camera view. Partial visibility cannot be combined across cameras. ALIGN is
ready only when BasePose reports `aligned` and either the chest or head result
is `SATISFIED`. MANIPULATE has empty arguments and requests VLA execution only
after that handoff. No target-role/type matching gates BasePose or VLA; sensor
freshness and controller ownership remain independent safety gates.

Each VA request still contains exactly one image and reuses the existing
`GROUNDING`, `ALIGN_GROUNDING`, or `POSTCHECK` schema. Navigation and ordinary
LA observations use chest RGB. ALIGN handoff sends two separate POSTCHECK
requests, one chest and one head, so the API and saved request-context format do
not need a new dual-image schema. Every request also carries the complete
original mission, `global_target`, LA strategic goal, and `strategic_stop`. VA
never selects a skill or exploration direction and is not told which skill just
ran.

The runtime owns the effective NAV/ALIGN transition:

- NAV calls fresh `GROUNDING` after MOVE_TO. `READY_TO_ALIGN` requires the
  requested target visible and the mean of finite, positive depth values in its
  bbox to lie in `nav_handoff_min_depth_m..nav_handoff_max_depth_m`. Values over
  8 m are discarded. Missing valid depth produces `UNKNOWN`; a missing target
  or out-of-range depth produces `CONTINUE_NAVIGATION`.
- ALIGN combines the two one-image POSTCHECK statuses with OR. It produces
  `READY_TO_MANIPULATE` only when one complete camera view contains all
  operation objects and BasePose is aligned. If neither view is complete it
  produces `RETRY_ALIGN`, or `RETURN_TO_NAVIGATION` when the alignment target
  could not be grounded.
- MANIPULATE continues to use the original visual POSTCHECK result directly for
  `CONTINUE_MANIPULATION`, `TASK_COMPLETE`, or `UNKNOWN`.

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
cleared when a new task generation starts.

LA emits only `EXECUTE` or `FAIL`; it does not make a final `COMPLETE`
decision. Once MANIPULATE returns the paired `SATISFIED` / `TASK_COMPLETE` VA
postcheck, the runtime emits the completed skill event and directly finalizes
the task without another LA request.

Every LA and VA request is archived before transmission under
`outputs/logs/inference/lavira_requests/`, alongside the inference component
logs. Each physical API attempt gets one JSON file recording the request kind,
attempt number, model parameters, exact message content, prompts, and embedded
data-URL images; API keys are not part of the request body and are not written.
The filename contains a nanosecond timestamp, process ID, monotonic per-process
sequence, request kind, and attempt number.

The ControlGateway Events tmux pane reserves roughly its top quarter for the
current Markdown TODO and the remaining area for recent runtime events. The
TODO area shows generation, Agent step, completed count, completed items, the
first active item, and later pending items; long lists are clipped around the
active item with a hidden-line count. With no active task it shows a start hint,
and while LA creates the initial TODO it shows a waiting hint. TODO updates are
structured `TODO_UPDATED` runtime events that update the fixed area instead of
being duplicated in the scrolling event list. Complete event history remains
available in the component log files.

Production validation should progress from observation-only logging, to a
30-degree-limited turn, to a 0.5 m local goal, and finally a complete task.
Record Fast-LIO heading error, NavDP segment status, and the final stopped state.

Prompt and orchestration portions are adapted from Uni-LaViRA commit `215e7aca`
under CC BY-NC-SA 4.0. See the repository third-party notice for attribution
and the non-commercial/share-alike obligations.
