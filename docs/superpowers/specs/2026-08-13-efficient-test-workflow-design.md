# Efficient Test Workflow

## Goal

Reduce the time spent writing and repeatedly running tests while preserving
meaningful regression coverage. The default feedback loop for Codex, local
development, and required CI checks must finish within 30 seconds.

The current `gear_sonic` suite is not itself the performance problem: 324 tests
pass in 3.23 seconds in the teleoperation environment. The main issues are
inconsistent entry points, the root pytest configuration selecting the
dependency-heavy `decoupled_wbc` suite, and development workflows that repeatedly
add or rerun broader tests than a change requires.

## Scope

This change will:

- establish one fast test entry point shared by Codex, developers, and CI;
- retain an explicit full-regression entry point for dependency-complete
  environments;
- prevent pytest from discovering virtual-environment, worktree, vendored, and
  third-party tests;
- define repository-level rules that limit redundant test creation and repeated
  execution by Codex;
- add a fast required CI check and keep heavy regression separate.

This change will not delete existing tests, infer source-to-test mappings, add a
test-selection plugin, or reduce the behavior currently covered by the test
suite.

## Test Commands

The root `Makefile` will provide these public targets:

- `make test` runs all of `gear_sonic/tests` with quiet output and a hard
  30-second wall-clock limit. It is the default completion check for ordinary
  changes.
- `make test-related TESTS="..."` runs only the explicitly supplied pytest
  paths or node IDs. It rejects an empty `TESTS` value with a usage message.
- `make test-full` explicitly runs both `gear_sonic/tests` and
  `decoupled_wbc/tests`. It has no 30-second limit and is expected to fail with
  normal pytest import diagnostics if invoked outside an environment containing
  the full ROS, simulation, model, and data dependencies.

All targets invoke `$(PYTHON) -m pytest`, with `PYTHON ?= python3`, so callers can
select an existing environment without repository-specific virtual-environment
paths. The fast timeout is configurable for diagnosis, but its committed default
remains 30 seconds. A timeout exits nonzero and prints guidance to inspect newly
added slow tests rather than silently increasing the budget.

## Pytest Collection

The root pytest configuration will make `gear_sonic/tests` the default
`testpaths`. Bare `pytest` and `python -m pytest` therefore run the same fast
suite as `make test`, subject only to the Make target's additional timeout.

Pytest recursion will exclude at least:

- `.venv*` and other local environment directories;
- `.worktrees`;
- `external_dependencies`;
- third-party and generated build directories;
- caches and logs.

`make test-full` bypasses the default `testpaths` selection by naming both owned
test directories explicitly. Vendored dependency tests remain outside both the
fast and full project suites.

## Codex Test Contract

A root `AGENTS.md` will make the following rules persistent for future Codex
work in this repository:

1. Add or change tests only for a new or changed observable behavior or for a
   reproduced defect. Documentation, formatting, comments, and pure renames do
   not receive new tests.
2. Modify the nearest existing test before creating a new file. Do not duplicate
   an already covered contract.
3. Cover one behavior with one regression test by default; use parametrization
   for equivalent input cases.
4. In the implementation loop, run the smallest related test once to establish
   the failure when a regression test is appropriate, run it once after the
   implementation, then run `make test` once at completion.
5. After a failure, rerun only failed or directly affected tests until they pass;
   do not repeatedly rerun the full fast suite.
6. Run `make test-full` only when the user requests it or when a change crosses
   both `gear_sonic` and `decoupled_wbc` and the required environment is
   available.
7. Do not test private implementation details, trivial assignments, or behavior
   owned by third-party packages merely to increase test count.
8. Report the commands, result counts, and elapsed time. If the heavy suite was
   not run, state that explicitly rather than implying full validation.

These are default limits, not a prohibition on risk-based verification. A
security, safety, concurrency, protocol, or data-integrity change may justify
additional targeted cases, but the agent must explain the risk being covered.

## Continuous Integration

A test workflow will contain two independent lanes:

- `fast-tests` runs `make test` for pull requests and relevant pushes in a
  lightweight Python environment. It is the required merge check and must obey
  the same 30-second execution budget as local development.
- `full-tests` runs `make test-full` only on a dependency-complete runner. Until
  the repository has a documented runner with ROS and the simulation/model/data
  stack, it is exposed only as an explicit/manual lane and is not represented as
  a successful skipped check.

The fast lane must not access the network, real hardware, model downloads, or a
GPU. Tests that require those resources belong to the heavy lane. CI will retain
pytest's normal collection diagnostics and the timeout exit status so dependency
leaks and newly slow tests are visible failures.

The implementation will not invent credentials or an unavailable full-runner
label. If no reproducible heavy environment is present, it will add the fast CI
lane and document `make test-full` as the manual/full entry point; connecting the
heavy lane becomes a follow-up deployment task.

## Failure Handling

- A fast-suite timeout is a test failure. The response is to identify the slow
  test or accidental external dependency, not to raise the default budget.
- An empty `TESTS` argument is a command-usage failure and must not fall back to
  the full suite.
- Missing dependencies in `test-full` remain explicit pytest failures; tests are
  not silently skipped merely because the active environment is incomplete.
- CI and local commands must never collect tests from local virtual environments,
  vendored code, or linked worktrees.

## Verification and Acceptance

Implementation is accepted when all of the following hold:

1. `make test` passes the current `gear_sonic` suite within 30 seconds and exits
   nonzero when that wall-clock limit is exceeded.
2. Bare pytest collection selects `gear_sonic/tests` and does not include local
   virtual environments, worktrees, external dependencies, or third-party code.
3. `make test-related` accepts explicit files and node IDs and refuses an empty
   selection.
4. `make test-full` explicitly includes both owned suites and exposes missing
   heavy dependencies instead of hiding them.
5. The repository-level Codex instructions encode the approved creation,
   execution, and reporting limits.
6. The fast CI job uses `make test`; any heavy lane is independent and truthful
   about runner availability.
7. Existing `gear_sonic` regression tests remain present and passing.
