# YOLOE Reference-Image Grounding Design

## Goal

Add a local command-line workflow in which a reference image and one or more
example boxes define a visual category for YOLOE-26M, then use that category to
detect and segment matching objects in a second image. Also add automatic
reference-box generation through either the existing ChatGPT-authenticated
Codex client (`gpt-5.6-sol`, `xhigh`) or the existing DashScope Qwen-VL Plus
client. The operator must explicitly supply the sought object with `--target`.

## Interfaces

The manual entry point accepts one target image, one reference image, a target
name, and one or more original-pixel `xyxy` boxes:

```bash
python tools/yoloe26m/visual_prompt.py TARGET.jpg \
  --refer-image REFERENCE.jpg \
  --target-name "blue basket" \
  --bbox 120 80 360 300 \
  --bbox 500 100 720 330
```

The automatic entry point accepts the same two images and an explicit target,
then selects one of two existing BasePose-compatible clients:

```bash
python tools/yoloe26m/auto_refer_detect.py TARGET.jpg \
  --refer-image REFERENCE.jpg --target "blue basket" --backend codex

python tools/yoloe26m/auto_refer_detect.py TARGET.jpg \
  --refer-image REFERENCE.jpg --target "blue basket" --backend qwenvl
```

Both commands accept the same YOLOE model, device, image-size, confidence,
IoU, output-root, run-name, overwrite, and FP16 controls.

## Architecture

`reference_pipeline.py` owns validation, normalized-to-pixel conversion,
artifact writing, VLM grounding, and YOLOE visual-prompt execution. The two
thin CLI files parse their different inputs and call this shared core. The
automatic path imports `CodexStructuredVisionClient` and
`QwenVLStructuredVisionClient` from the current BasePose implementation so it
uses the same authentication, API endpoint, key lookup, timeout behavior, and
proxy handling without copying secrets or configuration.

The grounding model sees only the reference image. Its prompt names the exact
operator-supplied target and requires every visible matching physical instance
as a separate tight box in normalized `[0,1000]` coordinates. A strict JSON
schema and runtime validation reject missing, uncertain, malformed,
out-of-range, zero-area, or empty results. All accepted boxes are converted to
the reference image's original pixels and assigned YOLOE visual-prompt class
ID 0. No confirmation pause occurs between grounding and YOLOE inference.

YOLOE uses `YOLOEVPSegPredictor`, `refer_image`, and `visual_prompts` with
`bboxes` and `cls`. It processes exactly one target image per invocation. The
saved target rendering contains both detection boxes and instance masks; class
0 is labeled with the supplied target name.

## Outputs and failure behavior

Each run writes one directory under `tools/yoloe26m/outputs/` (or
`--project`). Manual runs save the validated input description, annotated
reference image, machine-readable target detections, and annotated target
image. Automatic runs additionally save the exact grounding prompt, JSON
schema, and raw model response. Existing directories are rejected unless
`--exist-ok` is explicit.

Both CLIs exit nonzero with a direct error before YOLOE runs when an image,
model, login/key, or grounding result is invalid. Secrets and model reasoning
are never written. Qwen continues to read `DASHSCOPE_API_KEY` from the process
environment or `.venv_inference/.env`; Codex continues to use the current
ChatGPT CLI login.

## Verification

Use TDD for coordinate validation, all-box retention, output-directory safety,
client configuration, and automatic handoff to the YOLOE runner. Mock only the
external VLM request and heavyweight YOLOE call in unit tests. Then run syntax
checks, CLI help checks, all focused tests, and a real local YOLOE manual visual
prompt smoke test using the installed model and local sample image. Do not send
a paid/cloud VLM request during verification.
