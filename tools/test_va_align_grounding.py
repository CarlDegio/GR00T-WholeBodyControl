#!/usr/bin/env python3
"""Call the LaViRA VA ALIGN_GROUNDING prompt on one local image.

This is intentionally standalone: it starts from the production prompt but
keeps test-only prompt and deterministic-selection experiments isolated from
the running LaViRA agent.

The API key is read from LAVIRA_VA_API_KEY, falling back to DASHSCOPE_API_KEY,
first from the process environment and then from the repository's .env.local.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import math
import mimetypes
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / ".env.local"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-27b"
DEFAULT_MISSION = (
    "Move in front of the desk with the blue basket, grasp the medicine "
    "bottle, and place it into the blue basket."
)
DEFAULT_GLOBAL_TARGET = "desk with blue basket"
DEFAULT_STRATEGIC_GOAL = (
    "Prepare the visible manipulation-task objects for BasePose alignment."
)
DEFAULT_OUTPUT_ROOT = Path("outputs/va_align_grounding_tests")

ALIGN_GROUNDING_RESPONSE_KEYS = {
    "mode",
    "status",
    "objects",
    "visual_evidence",
}
ALIGN_OPERATION_OBJECT_KEYS = {
    "name",
    "surface",
    "visible",
    "bbox_2d",
    "confidence",
}
ABSTRACT_ALIGNMENT_SURFACES = {
    "surface",
    "tabletop",
    "desktop",
    "desktop surface",
    "table surface",
    "desk surface",
    "countertop",
    "top",
    "edge",
    "plane",
    "floor area",
}
FLOOR_ALIGNMENT_SURFACES = {
    "floor",
    "floor area",
    "floor plane",
    "floor surface",
    "flooring",
    "ground",
    "ground area",
    "ground plane",
    "ground surface",
}
MIN_ALIGNMENT_TARGET_BBOX_AREA = 10_000.0


class ResponseValidationError(ValueError):
    """The VA response does not satisfy the production ALIGN contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone LaViRA VA ALIGN_GROUNDING prompt on one image."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Local JPG/PNG/WebP/etc. image to send to VA.",
    )
    parser.add_argument(
        "--mission",
        default=DEFAULT_MISSION,
        help="Manipulation task used to enumerate operation objects.",
    )
    parser.add_argument(
        "--global-target",
        default=DEFAULT_GLOBAL_TARGET,
        help="Frozen navigation target supplied as ALIGN context.",
    )
    parser.add_argument(
        "--strategic-goal",
        default=DEFAULT_STRATEGIC_GOAL,
        help="Current LA strategy supplied as ALIGN context.",
    )
    parser.add_argument(
        "--strategic-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Strategic stop signal (production ALIGN uses false).",
    )
    parser.add_argument(
        "--direction",
        default="front",
        help="Fixed view label in the prompt (production ALIGN uses front).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LAVIRA_VA_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LAVIRA_VA_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Maximum API/validation attempts, matching production by default.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=10.0,
        help="Delay after an API error before retrying.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "New result directory. Defaults to a timestamped directory under "
            "outputs/va_align_grounding_tests."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and save the exact request without calling VA.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the rendered prompt; an image is not required.",
    )
    return parser.parse_args()


def _va_context(
    mission: str,
    global_target: str,
    strategic_goal: str,
    strategic_stop: bool,
) -> str:
    return f"""**MISSION**: {json.dumps(mission, ensure_ascii=False)}
**GLOBAL TARGET**: {json.dumps(global_target, ensure_ascii=False)}
**CURRENT STRATEGY**: {json.dumps(strategic_goal, ensure_ascii=False)}
**STRATEGIC STOP SIGNAL**: {json.dumps(strategic_stop)}"""


def alignment_grounding_prompt(
    *,
    mission: str,
    global_target: str,
    strategic_goal: str,
    strategic_stop: bool,
    direction: str,
) -> str:
    """Render the standalone experimental ALIGN_GROUNDING prompt."""

    return f"""**ROLE**: You are a humanoid robot agent's TACTICAL EYES in
ALIGN_GROUNDING mode.
{_va_context(mission, global_target, strategic_goal, strategic_stop)}

**INPUT**: You are looking at the CURRENT VIEW after turning to the fixed
{direction} panorama direction.

**TASK**:
1. **Manipulation-Task Objects**: Using only MISSION (the manipulation task in
   this request), enumerate every physical operation object required by that
   task: each object to grasp/move/place, each destination or container, and
   each physical control object the robot must operate. Include every such
   object even when it is not visible in the current image.
2. **Strict Exclusions**: Do not include any surface/support object as an item
   in `objects`. Also exclude the frozen navigation GLOBAL TARGET as such,
   navigation landmarks, rooms, people, robot body parts, and unrelated scene
   objects. For the task "grasp the medicine bottle and place it into the blue
   basket", `objects` must contain `medicine bottle` and `blue basket`; `desk`
   belongs only in their `surface` fields and must not be a separate item.
3. **Per-Object Grounding**: For every task object, return a short English
   detector-friendly `name`, its complete concrete supporting `surface`,
   whether it is visible, and its tight bbox and confidence. Invisible objects
   use `bbox_2d:null`. Use a whole-object surface name such as `desk`, `table`,
   or `shelf`; never use a part, region, material, or geometry such as
   `tabletop`, `desktop surface`, `top`, `edge`, or `plane`. If the object is
   supported directly by the floor or ground, never return `floor`, `ground`,
   `floor area`, `ground plane`, or any equivalent floor/ground term. Instead,
   set `surface` to exactly the same text as that object's `name`.
4. **Grounding Decision**: Return FOUND when at least one visible operation
   object occupies at least 1% of the full image. Otherwise return NOT_FOUND.
   Do not choose the final BasePose target yourself: the runtime compares all
   returned visible bboxes and deterministically selects the largest eligible
   operation object.

Use short English noun phrases. Coordinates are normalized [0,1000]. Return
exactly:
{{"mode":"ALIGN_GROUNDING","status":"FOUND|NOT_FOUND","objects":[
{{"name":"...","surface":"...","visible":true,
"bbox_2d":[0,0,1000,1000],"confidence":0.0}}],
"visual_evidence":"..."}}"""


def _normalized_label(value: str) -> str:
    return " ".join(value.casefold().split())


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_bbox(value: Any, *, required: bool) -> list[float] | None:
    if value is None and not required:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ResponseValidationError("VA bbox_2d is invalid")
    if not all(
        _finite_number(item) and 0.0 <= float(item) <= 1000.0
        for item in value
    ):
        raise ResponseValidationError("VA bbox_2d is invalid")
    x1, y1, x2, y2 = map(float, value)
    if x1 >= x2 or y1 >= y2:
        raise ResponseValidationError("VA bbox_2d has invalid corner ordering")
    return [x1, y1, x2, y2]


def validate_alignment_grounding(value: Any) -> dict[str, Any]:
    """Validate and select one target plus its supporting surface."""

    if not isinstance(value, dict) or set(value) != ALIGN_GROUNDING_RESPONSE_KEYS:
        raise ResponseValidationError(
            "VA ALIGN_GROUNDING output has an invalid object schema"
        )
    if (
        value["mode"] != "ALIGN_GROUNDING"
        or value["status"] not in {"FOUND", "NOT_FOUND"}
    ):
        raise ResponseValidationError("VA ALIGN_GROUNDING status is invalid")
    if not isinstance(value["visual_evidence"], str):
        raise ResponseValidationError(
            "VA ALIGN_GROUNDING visual_evidence must be a string"
        )
    raw_objects = value["objects"]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ResponseValidationError(
            "VA ALIGN_GROUNDING must list every manipulation-task object"
        )

    objects: list[dict[str, Any]] = []
    object_names: set[str] = set()
    external_surface_names: set[str] = set()
    eligible: list[tuple[float, float, dict[str, Any]]] = []
    visible_areas: list[float] = []
    for item in raw_objects:
        if not isinstance(item, dict) or set(item) != ALIGN_OPERATION_OBJECT_KEYS:
            raise ResponseValidationError(
                "VA ALIGN_GROUNDING operation object has an invalid schema"
            )
        name = str(item["name"]).strip() if isinstance(item["name"], str) else ""
        surface = (
            str(item["surface"]).strip()
            if isinstance(item["surface"], str)
            else ""
        )
        if not name:
            raise ResponseValidationError(
                "VA ALIGN_GROUNDING operation object name is required"
            )
        if not surface:
            raise ResponseValidationError(
                "VA ALIGN_GROUNDING operation object surface is required"
            )
        normalized_name = _normalized_label(name)
        normalized_surface = _normalized_label(surface)
        if normalized_name in object_names:
            raise ResponseValidationError(
                "VA ALIGN_GROUNDING operation object names must be unique"
            )
        self_surface = normalized_surface == normalized_name
        if normalized_surface in FLOOR_ALIGNMENT_SURFACES:
            raise ResponseValidationError(
                "VA ALIGN_GROUNDING floor-supported object must use its own "
                "name as surface, not a floor/ground term"
            )
        if not self_surface and (
            normalized_surface in ABSTRACT_ALIGNMENT_SURFACES
            or normalized_surface.endswith(" surface")
        ):
            raise ResponseValidationError(
                "VA ALIGN_GROUNDING surface must name a complete physical object"
            )
        visible = item["visible"]
        if not isinstance(visible, bool):
            raise ResponseValidationError(
                "VA ALIGN_GROUNDING operation object visible must be boolean"
            )
        confidence = item["confidence"]
        if (
            not _finite_number(confidence)
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ResponseValidationError("VA confidence is invalid")
        bbox = _validate_bbox(item["bbox_2d"], required=visible)
        if not visible and bbox is not None:
            raise ResponseValidationError(
                "VA ALIGN_GROUNDING invisible object cannot include a bbox"
            )
        canonical = {
            "name": name,
            "surface": surface,
            "visible": visible,
            "bbox_2d": bbox,
            "confidence": float(confidence),
        }
        objects.append(canonical)
        object_names.add(normalized_name)
        if not self_surface:
            external_surface_names.add(normalized_surface)
        if bbox is not None:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            visible_areas.append(area)
            if area >= MIN_ALIGNMENT_TARGET_BBOX_AREA:
                eligible.append((area, float(confidence), canonical))

    overlap = object_names & external_surface_names
    if overlap:
        raise ResponseValidationError(
            "VA ALIGN_GROUNDING surface objects must not appear in the "
            "manipulation-task object list: " + ", ".join(sorted(overlap))
        )

    found = value["status"] == "FOUND"
    if found and not eligible:
        largest_percent = max(visible_areas, default=0.0) / 10_000.0
        raise ResponseValidationError(
            "VA ALIGN_GROUNDING found no operation object large enough for "
            "stable BasePose alignment "
            f"(largest={largest_percent:.2f}% of image)"
        )
    if not found and eligible:
        raise ResponseValidationError(
            "VA ALIGN_GROUNDING status is NOT_FOUND despite an eligible "
            "visible manipulation-task object"
        )

    if eligible:
        _area, _confidence, selected = max(
            eligible,
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        target = str(selected["name"])
        selected_surface = str(selected["surface"])
        bbox = selected["bbox_2d"]
    else:
        selected = objects[0]
        target = str(selected["name"])
        selected_surface = str(selected["surface"])
        bbox = None

    floor_supported = _normalized_label(selected_surface) == _normalized_label(
        target
    )

    return {
        "target": {
            "text": target,
            "bbox_2d": bbox,
        },
        "surface": {
            "text": target if floor_supported else selected_surface,
            # ALIGN_GROUNDING does not ask VA for a separate surface bbox.
            # Reuse the target bbox only for the explicit floor fallback.
            "bbox_2d": bbox if floor_supported else None,
        },
    }


def image_data_url(image_path: Path) -> str:
    path = image_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"image does not exist: {path}")
    mime_type, _encoding = mimetypes.guess_type(path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"unsupported image type: {path}")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def build_messages(
    *, prompt: str, image_url: str, enable_thinking: bool
) -> list[dict[str, Any]]:
    system_prompt = (
        "Reason carefully and follow the requested output format exactly."
        if enable_thinking
        else "/no_think"
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def add_validation_retry_instruction(
    messages: list[dict[str, Any]], *, error: str
) -> list[dict[str, Any]]:
    cloned = [dict(message) for message in messages]
    cloned[0]["content"] = (
        f"{cloned[0]['content']}\n\nRETRY REQUIREMENT: The previous response "
        f"failed validation ({error}). Return one complete strict JSON object "
        "matching the requested schema. Close every string and do not use "
        "Markdown fences or commentary outside the object."
    )
    return cloned


def new_output_dir(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = DEFAULT_OUTPUT_ROOT / stamp
    else:
        path = requested.expanduser()
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def response_text(completion: Any) -> str:
    try:
        value = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ResponseValidationError("VA response is malformed") from exc
    if not isinstance(value, str) or not value.strip():
        raise ResponseValidationError("VA response content is empty")
    return value


def strict_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError("VA output is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ResponseValidationError("VA output must be a JSON object")
    return value


def _dotenv_value(path: Path, name: str) -> str | None:
    """Read one dotenv value without exporting or logging any secrets."""

    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, raw_value = line.partition("=")
        if separator and key.strip() == name:
            value = raw_value.strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]
            return value or None
    return None


def resolve_api_key() -> str | None:
    """Resolve the role-specific/shared VA key used by the normal launcher."""

    for name in ("LAVIRA_VA_API_KEY", "DASHSCOPE_API_KEY"):
        value = os.getenv(name, "").strip() or _dotenv_value(
            DEFAULT_ENV_FILE,
            name,
        )
        if value:
            return value
    return None


def run(args: argparse.Namespace) -> int:
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.timeout_seconds <= 0.0:
        raise ValueError("--timeout-seconds must be positive")
    if args.retry_delay_seconds < 0.0:
        raise ValueError("--retry-delay-seconds must be non-negative")

    prompt = alignment_grounding_prompt(
        mission=args.mission,
        global_target=args.global_target,
        strategic_goal=args.strategic_goal,
        strategic_stop=args.strategic_stop,
        direction=args.direction,
    )
    if args.print_prompt:
        print(prompt)
        if args.image is None:
            return 0
    if args.image is None:
        raise ValueError("IMAGE is required unless only --print-prompt is used")

    output_dir = new_output_dir(args.output_dir)
    image_path = args.image.expanduser().resolve()
    messages = build_messages(
        prompt=prompt,
        image_url=image_data_url(image_path),
        enable_thinking=args.enable_thinking,
    )
    context = {
        "image": str(image_path),
        "mission": args.mission,
        "global_target": args.global_target,
        "strategic_goal": args.strategic_goal,
        "strategic_stop": args.strategic_stop,
        "direction": args.direction,
        "base_url": args.base_url,
        "model": args.model,
        "enable_thinking": args.enable_thinking,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout_seconds,
    }
    (output_dir / "context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    if args.dry_run:
        summary = {
            "system": messages[0]["content"],
            "user_content_types": [
                item["type"] for item in messages[1]["content"]
            ],
            "image_data_url_bytes": len(
                messages[1]["content"][0]["image_url"]["url"]
            ),
            "output_dir": str(output_dir),
        }
        (output_dir / "dry_run.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    api_key = resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "set LAVIRA_VA_API_KEY or DASHSCOPE_API_KEY in the environment "
            f"or {DEFAULT_ENV_FILE} before calling VA"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "the standalone VA test requires the openai package; use "
            ".venv_inference/bin/python"
        ) from exc

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=float(args.timeout_seconds),
    )
    last_error: Exception | None = None
    for attempt in range(1, args.attempts + 1):
        request_metadata = {
            "attempt": attempt,
            "model": args.model,
            "system": messages[0]["content"],
            "prompt": prompt,
            "image": str(image_path),
        }
        (output_dir / f"request_attempt_{attempt}.json").write_text(
            json.dumps(request_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": args.enable_thinking},
            )
            raw_text = response_text(completion)
            (output_dir / f"raw_response_attempt_{attempt}.txt").write_text(
                raw_text,
                encoding="utf-8",
            )
            result = validate_alignment_grounding(
                strict_json_object(raw_text)
            )
        except Exception as exc:
            last_error = exc
            (output_dir / f"error_attempt_{attempt}.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            if attempt >= args.attempts:
                break
            if isinstance(exc, ResponseValidationError):
                messages = add_validation_retry_instruction(
                    messages,
                    error=str(exc),
                )
            elif args.retry_delay_seconds:
                time.sleep(args.retry_delay_seconds)
            continue

        (output_dir / "validated_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nSaved test artifacts to: {output_dir}", file=sys.stderr)
        return 0

    assert last_error is not None
    raise RuntimeError(
        f"VA ALIGN_GROUNDING failed after {args.attempts} attempts; "
        f"artifacts: {output_dir}; last error: {last_error}"
    ) from last_error


def main() -> int:
    try:
        return run(parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
