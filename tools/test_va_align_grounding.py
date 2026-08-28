#!/usr/bin/env python3
"""Call the production ALIGN_GROUNDING prompt on one local image.

The request contains only the alignment prompt, view direction, and image.
API keys are read from LAVIRA_VA_API_KEY or DASHSCOPE_API_KEY, with
``.env.local`` as a fallback.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import mimetypes
import os
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.inference.lavira.agent import (  # noqa: E402
    LaViRAAgentError,
    alignment_grounding_prompt,
    validate_alignment_grounding,
)


DEFAULT_ENV_FILE = REPO_ROOT / ".env.local"
DEFAULT_BASE_URL = (
    "https://ws-6yzgj1m087a053ip.cn-beijing.maas.aliyuncs.com/"
    "compatible-mode/v1"
)
DEFAULT_MODEL = "qwen3.5-27b"
DEFAULT_ALIGNMENT_PROMPT = (
    "Use the cardboard box as the distance-and-centering target.\n"
    "Use the cardboard box as the yaw-alignment target;\n"
    "align yaw to one visible straight edge of that same cardboard box."
)
DEFAULT_OUTPUT_ROOT = Path("outputs/va_align_grounding_tests")


class ResponseValidationError(ValueError):
    """The VA response or transport payload is malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Local JPG/PNG/WebP/etc. image to send to VA.",
    )
    parser.add_argument(
        "--alignment-prompt",
        default=DEFAULT_ALIGNMENT_PROMPT,
        help="The only task-semantic text supplied to ALIGN grounding.",
    )
    parser.add_argument(
        "--direction",
        default="front",
        help="Fixed view label (production ALIGN uses front).",
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
        help="Maximum API/validation attempts, matching production.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New result directory; defaults under outputs/va_align_grounding_tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Save the request without calling VA.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the rendered prompt; no image is required.",
    )
    return parser.parse_args()


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
    *, prompt: str, image_url: str, enable_thinking: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "Reason carefully and follow the requested output format exactly."
                if enable_thinking
                else "/no_think"
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def add_validation_retry_instruction(
    messages: list[dict[str, Any]], *, error: str,
) -> list[dict[str, Any]]:
    cloned = [dict(message) for message in messages]
    cloned[0]["content"] = (
        f"{cloned[0]['content']}\n\nRETRY REQUIREMENT: The previous response "
        f"failed validation ({error}). Return one complete strict JSON object "
        "matching the requested schema, with no Markdown or commentary."
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
        alignment_prompt=args.alignment_prompt,
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
        "alignment_prompt": args.alignment_prompt,
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
            result = validate_alignment_grounding(strict_json_object(raw_text))
        except Exception as exc:
            last_error = exc
            (output_dir / f"error_attempt_{attempt}.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            if attempt >= args.attempts:
                break
            if isinstance(exc, (ResponseValidationError, LaViRAAgentError)):
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
