# BasePose Codex Sol/Max/600s Design

## Goal

Make the BasePose Codex path use explicit, stable defaults for the local
ChatGPT-authenticated Codex CLI:

- model: `gpt-5.6-sol`;
- reasoning effort: `max`;
- subprocess timeout: `600` seconds.

The manipulation task remains externally configurable through
`--base-pose-task`.

## Evidence and motivation

The 22:08 run used `gpt-5.6` and failed after 33.13 seconds. Codex 0.146.0
first failed to refresh its model metadata, then reported that `gpt-5.6` was
not supported for the ChatGPT account. The BasePose client passes its model
string directly to `codex exec`, so that run did not reach model inference.

The 22:12 run used explicit `gpt-5.6-sol` with `max` and was terminated by
the BasePose subprocess timeout after 180.30 seconds. Independent short,
read-only requests established that the same ChatGPT login and proxy
environment can run both `gpt-5.6-sol` at `low` and at `max`. The longer
timeout therefore addresses the complex vision/schema request's latency, not
an authentication or general proxy failure.

## Scope

Update every active BasePose default surface together:

1. `BasePoseConfig` and `CodexStructuredVisionClient`;
2. `BasePosePlannerConfig`;
3. `InferenceLaunchConfig`;
4. launch-contract tests and BasePose documentation.

The launcher must emit all three values explicitly in the pane-3 command:
`--model gpt-5.6-sol`, `--reasoning-effort max`, and
`--codex-timeout-seconds 600.0`.

## Non-goals

- Do not change the BasePose prompt, JSON schema, camera path, motion planner,
  proxy environment, or ChatGPT-subscription authentication.
- Do not add retries, fallback models, partial-output logging, or new timeout
  handling.
- Do not run a live robot command as part of verification.
- Preserve CLI overrides so an operator can select another model, effort, or
  timeout for an individual launch.

## Verification

Use a software-only command-builder contract to prove that a default BasePose
launch propagates the three new values. Run it once before implementation to
observe failure and again after implementation to observe success. Compile the
changed Python files and inspect the final diff so unrelated prompt and
stateless-planner behavior remains unchanged.
