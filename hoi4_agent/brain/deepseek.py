"""DeepSeek V4.1 Flash client (OpenAI-compatible chat): Tier 3 of the arena hierarchy.

Measured in this project: reads text and counts well, ~1.3 s median without thinking and
2-13 s with; an image costs ~250-330 prompt tokens. It is unreliable at pointing and cannot
read bar fill, so nothing here should be used to obtain coordinates or quantities.

Differences from ``OpenAICompatBackend`` (which this does not modify): bearer auth,
``response_format: json_object`` instead of ``json_schema``, the ``thinking`` switch, a
total deadline with bounded retries, and token usage in the reply. ``DeepSeekBackend``
adapts the client to the ``LLMBackend`` protocol so the older vision agent can use it.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import requests
from PIL import Image

from ..errors import BrainError, ParseError, SchemaError
from .jev import USER_AGENT, Post, post_json
from .llm import encode_image

DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_KEY_NAME = "DEEPSEEK_API_KEY"
RETRY_STATUSES = (429, 500, 503, 529)


@dataclass(frozen=True)
class DeepSeekReply:
    text: str
    reasoning: str  # thinking trace when thinking was enabled; never parsed for decisions
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    thinking: bool
    attempts: int = 1

    def json(self) -> dict[str, Any]:
        """The reply as a JSON object; ParseError/SchemaError otherwise."""
        try:
            value = json.loads(self.text)
        except ValueError:
            raise ParseError(self.text) from None
        if not isinstance(value, dict):
            raise SchemaError("expected a JSON object at the top level")
        return value


def data_url(image: Image.Image | str, mime: str = "image/png", scale: int = 1) -> str:
    """PIL image (encoded as PNG, optionally integer-upscaled) or ready base64 -> data URL."""
    if isinstance(image, str):
        return f"data:{mime};base64,{image}"
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    return f"data:image/png;base64,{encode_image(image, 'PNG')}"


@dataclass
class DeepSeekClient:
    api_key: str | None = field(default=None, repr=False)
    model: str = DEEPSEEK_MODEL
    endpoint: str = DEEPSEEK_ENDPOINT
    post: Post = field(default=requests.post, repr=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    default_timeout_s: float = 30.0

    def _key(self) -> str:
        if self.api_key is None:
            from ..arena.envfile import require
            self.api_key = require(DEEPSEEK_KEY_NAME)
        return self.api_key

    def chat(self, *, system: str, user: str, images: Sequence[Image.Image | str] = (),
             image_mime: str = "image/png", json_output: bool = True, thinking: bool = False,
             max_tokens: int | None = None, timeout: float | None = None) -> DeepSeekReply:
        """One chat turn. ``timeout`` is the total deadline in seconds, retries included.

        JSON mode needs the word "json" somewhere in the prompt (API rule); it is appended
        to the system message when missing. With thinking on, ``max_tokens`` also has to
        cover the reasoning, so the default is larger.
        """
        if json_output and "json" not in (system + user).lower():
            system = (system + " Reply with a single JSON object.").strip()
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": data_url(i, image_mime)}}
                                         for i in images]
        content.append({"type": "text", "text": user})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "thinking": {"type": "enabled" if thinking else "disabled"},
            "max_tokens": max_tokens if max_tokens is not None else (6000 if thinking else 800),
        }
        if json_output:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json",
                   "User-Agent": USER_AGENT}
        started = time.perf_counter()
        data, attempts = post_json(self.post, self.endpoint, headers, body,
                                   self.default_timeout_s if timeout is None else timeout,
                                   retry_statuses=RETRY_STATUSES, backoff_s=0.5, sleep=self.sleep)
        latency = time.perf_counter() - started
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise SchemaError(f"malformed chat response: {str(data)[:200]}") from exc
        if not text.strip():
            reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            raise BrainError(f"empty completion (finish_reason={reason!r}); with thinking on, raise max_tokens")
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return DeepSeekReply(text, str(message.get("reasoning_content") or ""),
                             int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
                             latency, thinking, attempts)


class DeepSeekBackend:
    """``LLMBackend`` adapter. DeepSeek has no ``json_schema`` mode, so the schema is stated in
    the system prompt and ``json_object`` guarantees only syntactic JSON; the existing
    ``brain.parse`` validation downstream stays authoritative."""

    def __init__(self, client: DeepSeekClient | None = None, *, thinking: bool = False,
                 timeout_s: float = 60.0) -> None:
        self.client = client if client is not None else DeepSeekClient()
        self.thinking, self.timeout_s = thinking, timeout_s
        self.last_reply: DeepSeekReply | None = None

    def chat(self, *, images: list[str], system: str, user: str, schema: dict,
             image_mime: str = "image/png", timeout: float | None = None) -> str:
        if schema:
            system = (f"{system}\n\nReply with one JSON object that validates against this JSON Schema, "
                      f"and nothing else:\n{json.dumps(schema, separators=(',', ':'))}")
        self.last_reply = self.client.chat(system=system, user=user, images=images, image_mime=image_mime,
                                           json_output=True, thinking=self.thinking,
                                           timeout=timeout or self.timeout_s)
        return self.last_reply.text
