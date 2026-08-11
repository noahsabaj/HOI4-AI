"""Seam backend: OpenAI-compatible /v1/chat/completions (LM Studio, llama.cpp).

This is the path for grounding models (e.g. Holo1.5) whose imported Qwen-VL vision
GGUFs Ollama currently mishandles. Images are sent as data-URLs; structured output
uses ``response_format: json_schema``.
"""

from __future__ import annotations

import requests

from ..errors import BackendTimeoutError, BackendUnavailableError, BrainError


class OpenAICompatBackend:
    def __init__(self, endpoint: str, model: str, timeout_s: float = 120.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat(self, *, images, system, user, schema, image_mime="image/png", timeout=None) -> str:
        # The data URL must declare the format the bytes were actually encoded
        # in: numeric reads are PNG on purpose (JPEG artifacts on 20-px digits
        # corrupt exactly those reads), so labelling everything "image/jpeg"
        # hands a strict server a PNG under a JPEG media type.
        content = [{"type": "text", "text": user}]
        for b in images:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{b}"}}
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            },
        }
        try:
            r = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=timeout or self.timeout_s,
            )
        except requests.exceptions.ConnectionError as e:
            raise BackendUnavailableError(f"cannot reach server at {self.endpoint}") from e
        except requests.exceptions.Timeout as e:
            raise BackendTimeoutError(f"call timed out after {timeout or self.timeout_s}s") from e
        except requests.exceptions.RequestException as e:  # pragma: no cover
            raise BrainError(f"request failed: {e}") from e
        if r.status_code != 200:
            raise BrainError(f"HTTP {r.status_code}: {r.text[:200]}")
        try:
            message = r.json()["choices"][0]["message"]
            text = message["content"]
        except (ValueError, KeyError, IndexError) as e:
            raise BrainError(f"malformed response: {r.text[:200]}") from e
        if not text.strip():
            # Thinking models (e.g. Holo 3.1 / qwen3.5): the chat template opens a
            # <think> block, the schema grammar forces pure JSON so </think> never
            # appears, and the server files the entire output under
            # reasoning_content, leaving content empty.
            text = message.get("reasoning_content") or ""
        if not text.strip():
            raise BrainError(f"empty completion (no content or reasoning_content): {r.text[:200]}")
        return text
