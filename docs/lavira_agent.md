# Uni-LaViRA Agent runtime

The runtime keeps robot motion on the existing safety path:

```text
chest_view -> LaViRA LA/VA -> ControlGateway -> NavDP
           -> PlannerVelocityExecutor -> SONIC
```

LaViRA emits only relative heading goals and two-dimensional local goals. It
does not emit velocity. Press `N` to start the task fixed in the runtime YAML;
Space invalidates the generation and stops motion immediately.

## Local model endpoints

Start two OpenAI-compatible servers with the same configured alias (the model
files may be the same or independently deployed):

```bash
llama-server --model /path/to/Qwen3.5-27B-Q4_K_M.gguf \
  --alias Qwen3.5-27B-Q4_K_M --host 127.0.0.1 --port 8000
llama-server --model /path/to/Qwen3.5-27B-Q4_K_M.gguf \
  --alias Qwen3.5-27B-Q4_K_M --host 127.0.0.1 --port 8001
```

Set `LAVIRA_LA_API_KEY` and `LAVIRA_VA_API_KEY` when the servers require
authentication. Local unauthenticated servers use an internal `no-key`
placeholder.

## Task selection

`components.lavira.task_type` is one of `vln`, `object_nav`, or `eqa`.
`mission` and `global_target` are always required; `question` is additionally
required for EQA. The default limits are 20 Agent steps and five history items.
Examples:

```yaml
components:
  lavira:
    task_type: vln
    mission: walk through the doorway and stop beside the sofa
    global_target: sofa
    question: ''
```

```yaml
components:
  lavira:
    task_type: object_nav
    mission: find the blue basket
    global_target: blue basket
    question: ''
```

```yaml
components:
  lavira:
    task_type: eqa
    mission: go to the table
    global_target: table
    question: what color is the bottle on the table?
```

Production validation should progress from observation-only logging, to a
30-degree-limited turn, to a 0.5 m local goal, and finally a complete task.
Record Fast-LIO heading error, NavDP segment status, and the final stopped state.

Prompt and orchestration portions are adapted from Uni-LaViRA commit `215e7aca`
under CC BY-NC-SA 4.0. See the repository third-party notice for attribution
and the non-commercial/share-alike obligations.
