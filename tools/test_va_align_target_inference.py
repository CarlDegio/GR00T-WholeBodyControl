#!/usr/bin/env python3
"""Test production BasePose target inference with one image and a VLA task."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, TOOLS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gear_sonic.utils.inference.lavira.agent import (  # noqa: E402
    LaViRAAgentError,
    alignment_grounding_prompt,
    validate_alignment_grounding,
)
from test_va_align_grounding import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ResponseValidationError,
    add_validation_retry_instruction,
    build_messages,
    image_data_url,
    resolve_api_key,
    response_text,
    strict_json_object,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/va_align_target_inference_tests")


def build_alignment_prompt(*, task_prompt: str, direction: str) -> str:
    """Build the same VA prompt used by production ALIGN."""

    return alignment_grounding_prompt(
        manipulation_prompt=task_prompt,
        direction=direction,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Local JPG/PNG/WebP/etc. image to send to VA.",
    )
    parser.add_argument(
        "--task-prompt",
        required=True,
        help="Overall manipulation task from which VA infers both targets.",
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
    parser.add_argument(
        "--response-format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request the API json_object response format when supported.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Maximum API/validation attempts, matching production.",
    )
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "New result directory; defaults under "
            "outputs/va_align_target_inference_tests."
        ),
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


def new_output_dir(requested: Path | None) -> Path:
    path = (
        DEFAULT_OUTPUT_ROOT
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if requested is None
        else requested.expanduser()
    ).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def run(args: argparse.Namespace) -> int:
    task_prompt = str(args.task_prompt).strip()
    if not task_prompt:
        raise ValueError("--task-prompt must be non-empty")
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.timeout_seconds <= 0.0:
        raise ValueError("--timeout-seconds must be positive")
    if args.retry_delay_seconds < 0.0:
        raise ValueError("--retry-delay-seconds must be non-negative")

    prompt = build_alignment_prompt(
        task_prompt=task_prompt,
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
        "task_prompt": task_prompt,
        "direction": args.direction,
        "base_url": args.base_url,
        "model": args.model,
        "enable_thinking": args.enable_thinking,
        "response_format": args.response_format,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "prompt_source": "production_alignment_grounding_prompt",
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
            "or .env.local before calling VA"
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
            completion_kwargs = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "extra_body": {"enable_thinking": args.enable_thinking},
            }
            if args.response_format:
                completion_kwargs["response_format"] = {
                    "type": "json_object",
                }
            completion = client.chat.completions.create(
                **completion_kwargs,
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
        f"VA target-inference test failed after {args.attempts} attempts; "
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
