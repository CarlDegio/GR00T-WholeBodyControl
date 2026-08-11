# YOLOE Reference-Image Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build manual and Codex/Qwen-automatic reference-image example-box CLIs that feed validated visual prompts into local YOLOE-26M inference on a second image.

**Architecture:** Keep all validation, artifacts, provider selection, and YOLOE invocation in a dependency-injectable shared core, with two thin CLI adapters. Reuse the existing BasePose structured-vision clients without modifying their dirty source files, and keep cloud calls out of automated verification.

**Tech Stack:** Python 3.10, NumPy, OpenCV, Ultralytics YOLOE/YOLOEVPSegPredictor, existing Codex CLI client, existing OpenAI-compatible DashScope client, unittest/pytest.

## Global Constraints

- The operator must explicitly set the first-layer target with `--target`.
- Return and use every matching reference-image box, not only one.
- Automatically run YOLOE on the second image after grounding; do not pause.
- Codex uses `gpt-5.6-sol` with reasoning effort `xhigh` and the current ChatGPT CLI login.
- Qwen uses the same BasePose API URL, model default, and `DASHSCOPE_API_KEY` lookup.
- Do not modify existing BasePose files or persist credentials/model reasoning.
- The target output rendering must show bounding boxes and segmentation masks.

---

### Task 1: Pure grounding contracts and coordinate validation

**Files:**
- Create: `tools/yoloe26m/reference_pipeline.py`
- Create: `tools/yoloe26m/tests/test_reference_pipeline.py`

**Interfaces:**
- Produces: `GROUNDING_SCHEMA`, `build_grounding_prompt(target: str) -> str`, `validate_grounding_response(value: Mapping[str, Any]) -> list[NormalizedBox]`, and `normalized_boxes_to_pixels(boxes, width, height) -> list[PixelBox]`.

- [ ] **Step 1: Write failing tests** for empty target rejection, multi-box preservation, NOT_FOUND rejection, out-of-range/zero-area boxes, and the hand-derived `[100,200,800,900]` to `[64,96,512,432]` conversion on a 640x480 image.
- [ ] **Step 2: Run `./.venv_inference/bin/python -m pytest tools/yoloe26m/tests/test_reference_pipeline.py -v`** and verify imports fail because the core does not exist.
- [ ] **Step 3: Implement immutable normalized/pixel box data structures and the minimal pure validation/conversion functions.** Require finite numeric coordinates, `0 <= x1 < x2 <= 1000`, `0 <= y1 < y2 <= 1000`, READY status, and at least one box; convert with deterministic rounding and clip only the final integer edge to image bounds.
- [ ] **Step 4: Run the focused tests** and verify they pass.

### Task 2: Artifacts and YOLOE visual-prompt runner

**Files:**
- Modify: `tools/yoloe26m/reference_pipeline.py`
- Modify: `tools/yoloe26m/tests/test_reference_pipeline.py`

**Interfaces:**
- Produces: `prepare_run_dir(project, name, exist_ok) -> Path`, `annotate_reference(...) -> Path`, and `run_yoloe_visual_prompt(config, boxes) -> DetectionSummary`.

- [ ] **Step 1: Write failing behavior tests** that prove an existing directory is rejected without `exist_ok`, all example boxes use class ID 0, and the rendered output/JSON summary include all returned target detections.
- [ ] **Step 2: Run the focused tests** and verify failure at the missing artifact/runner interface.
- [ ] **Step 3: Implement directory safety and annotated-reference saving**, then implement the YOLOE call using `YOLOEVPSegPredictor`, `refer_image`, float32 `bboxes`, int32 zero `cls`, and save=False. Relabel class 0, save `target_annotated.jpg`, and serialize boxes/confidences/mask count to `detections.json`.
- [ ] **Step 4: Run the focused tests** and verify they pass.

### Task 3: Automatic grounding orchestration and provider reuse

**Files:**
- Modify: `tools/yoloe26m/reference_pipeline.py`
- Modify: `tools/yoloe26m/tests/test_reference_pipeline.py`

**Interfaces:**
- Produces: `create_grounding_client(backend: str, ...)`, and `run_auto_reference_pipeline(...) -> DetectionSummary` with injectable `client` and `yolo_runner` test seams.

- [ ] **Step 1: Write failing tests** using a complete fake structured response to prove every returned box reaches the runner in pixel coordinates, only the reference image is sent to the client, and invalid/empty grounding prevents runner invocation.
- [ ] **Step 2: Run the focused tests** and verify the orchestration functions are absent.
- [ ] **Step 3: Implement client creation** by importing the current `CodexStructuredVisionClient(model="gpt-5.6-sol", reasoning_effort="xhigh")` or `QwenVLStructuredVisionClient`; call its shared `run` contract and write `grounding_prompt.txt` plus `grounding_response.json` without reasoning content.
- [ ] **Step 4: Implement automatic handoff** from validated normalized boxes to pixel boxes, reference annotation, then YOLOE runner without a pause.
- [ ] **Step 5: Run the focused tests** and verify they pass.

### Task 4: Manual and automatic command-line entry points

**Files:**
- Create: `tools/yoloe26m/visual_prompt.py`
- Create: `tools/yoloe26m/auto_refer_detect.py`
- Modify: `tools/yoloe26m/tests/test_reference_pipeline.py`

**Interfaces:**
- Consumes: shared core functions from Tasks 1-3.
- Produces: executable manual, Codex, and Qwen command paths.

- [ ] **Step 1: Write failing parser tests** for repeated four-number `--bbox`, required `--target`, backend choices, and shared inference defaults.
- [ ] **Step 2: Run the focused tests** and verify the CLI modules do not exist.
- [ ] **Step 3: Implement `visual_prompt.py`** with required target image, `--refer-image`, `--target-name`, and repeated `--bbox`; validate original-pixel bounds before creating artifacts or loading YOLOE.
- [ ] **Step 4: Implement `auto_refer_detect.py`** with required target image, `--refer-image`, explicit `--target`, and `--backend {codex,qwenvl}` plus explicit Codex/Qwen tuning overrides whose defaults match BasePose.
- [ ] **Step 5: Run focused tests and both `--help` commands** and verify they pass without loading model weights or clients.

### Task 5: Documentation and real local verification

**Files:**
- Modify: `tools/yoloe26m/README.md`
- Verify: all files above.

**Interfaces:**
- Produces: three copy-paste startup commands and verified local output artifacts.

- [ ] **Step 1: Document the coordinate systems, all-match behavior, artifacts, authentication reuse, failure cases, and exactly three complete startup examples** (manual, Codex, Qwen).
- [ ] **Step 2: Run `python3 -m py_compile` on all new Python files** and the full focused test module.
- [ ] **Step 3: Run a real manual YOLOE smoke test** with the installed `yoloe-26m-seg.pt`, local `bus.jpg` as both reference and target, and one known bus box; verify a nonempty annotated target and JSON artifact.
- [ ] **Step 4: Run `git diff --check` and inspect status/diff** to ensure existing BasePose work is untouched and no credential/model output entered source control.
