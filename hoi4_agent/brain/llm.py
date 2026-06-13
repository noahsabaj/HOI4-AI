"""LLM backend protocol, image encoding, and a scripted fake for tests/offline."""

from __future__ import annotations

import base64
import io
from typing import Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class LLMBackend(Protocol):
    def chat(
        self,
        *,
        images: list[str],
        system: str,
        user: str,
        schema: dict,
        timeout: float | None = None,
    ) -> str:
        """Return the raw text content of the model's reply (expected to be JSON)."""
        ...


def encode_image(img: Image.Image, fmt: str = "JPEG", quality: int = 70) -> str:
    """PIL image -> base64 string (JPEG by default)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class ScriptedBackend:
    """Returns queued raw strings in order (last repeats); records every call."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._i = 0
        self.calls: list[dict] = []

    def chat(self, *, images, system, user, schema, timeout=None) -> str:
        self.calls.append({"system": system, "user": user, "schema": schema, "n_images": len(images)})
        if not self._responses:
            return "{}"
        raw = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return raw
