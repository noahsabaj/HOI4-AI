"""LLM backend protocol, image encoding, and a scripted fake for tests/offline."""

from __future__ import annotations

import base64
import io
from typing import Protocol, runtime_checkable

from PIL import Image


# Media type per encoding. Backends that wrap images in a data URL must label
# them with the format they were actually encoded in — reads are PNG on purpose
# (see encode_image), so a hardcoded "image/jpeg" label is a lie a strict server
# is entitled to reject.
IMAGE_MIME_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg"}


def mime_for(fmt: str) -> str:
    """Media type for an ``encode_image`` format."""
    try:
        return IMAGE_MIME_TYPES[fmt.upper()]
    except KeyError:
        raise ValueError(
            f"unsupported image format {fmt!r} (want one of {sorted(IMAGE_MIME_TYPES)})"
        ) from None


@runtime_checkable
class LLMBackend(Protocol):
    def chat(
        self,
        *,
        images: list[str],
        system: str,
        user: str,
        schema: dict,
        image_mime: str = "image/png",
        timeout: float | None = None,
    ) -> str:
        """Return the raw text content of the model's reply (expected to be JSON).

        ``image_mime`` is the media type ``images`` were encoded as; backends
        that must declare it (data URLs) use it instead of assuming.
        """
        ...


def encode_image(img: Image.Image, fmt: str = "PNG", quality: int = 85) -> str:
    """PIL image -> base64 string.

    PNG by default: numeric crops are tiny and JPEG artifacts on 20-px digits
    corrupt exactly the reads everything depends on. Pass ``fmt="JPEG"`` for
    full-frame judgment images where size matters (quality applies then).
    """
    fmt = fmt.upper()
    if fmt not in IMAGE_MIME_TYPES:
        raise ValueError(
            f"unsupported image format {fmt!r} (want one of {sorted(IMAGE_MIME_TYPES)})"
        )
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    if fmt == "JPEG":
        rgb.save(buf, format="JPEG", quality=quality)
    else:
        rgb.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class ScriptedBackend:
    """Returns queued raw strings in order (last repeats); records every call."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._i = 0
        self.calls: list[dict] = []

    def chat(self, *, images, system, user, schema, image_mime="image/png", timeout=None) -> str:
        self.calls.append({
            "system": system, "user": user, "schema": schema,
            "n_images": len(images), "image_mime": image_mime,
        })
        if not self._responses:
            return "{}"
        raw = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return raw
