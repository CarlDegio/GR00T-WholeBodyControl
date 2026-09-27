"""Exact, retry-aware model inputs and outputs for opt-in demo editing."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
import uuid


class DemoModelRecorder:
    def __init__(self, recorder, context):
        self.recorder, self.context = recorder, context

    def begin(self, kind, attempt, request):
        """Extract exact transmitted image bytes without duplicating base64 in JSON."""
        try:
            context = self.context()
            generation = context.pop("generation")
            root = self.recorder.path.parent / "demo" / self.recorder.trial_id(generation)
            images = root / "model_images"
            images.mkdir(parents=True, exist_ok=True)

            def externalize(value):
                if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
                    header, encoded = value.split(",", 1)
                    payload = base64.b64decode(encoded, validate=True)
                    digest = hashlib.sha256(payload).hexdigest()
                    suffix = ".png" if header.startswith("data:image/png;") else ".jpg"
                    path = images / (digest + suffix)
                    if not path.exists():
                        path.write_bytes(payload)
                    return {"path": str(path.relative_to(root)), "sha256": digest,
                            "mime_type": header[5:].split(";")[0]}
                if isinstance(value, dict):
                    return {k: externalize(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [externalize(v) for v in value]
                return value

            ticket = dict(request_id=uuid.uuid4().hex, generation=generation, kind=kind,
                          attempt=attempt, request_started_ns=time.time_ns(), **context)
            path = root / f"request_{ticket['request_id']}.json"
            path.write_text(json.dumps(dict(schema="sonic.demo.model_request.v1", **ticket,
                                           request=externalize(request)), ensure_ascii=False, indent=2))
            self.recorder.write("demo_model_request", **ticket, path=str(path))
            return ticket
        except Exception:
            logging.getLogger(__name__).exception("Could not save demo model request")
            return None

    def finish(self, ticket, *, completion=None, result=None, error=None):
        if ticket is None:
            return
        try:
            content = None
            if completion is not None:
                content = completion.choices[0].message.content
            self.recorder.write("demo_model_response", **ticket,
                                response_received_ns=time.time_ns(), content=content,
                                result=result, error=error)
        except Exception:
            logging.getLogger(__name__).exception("Could not save demo model response")
