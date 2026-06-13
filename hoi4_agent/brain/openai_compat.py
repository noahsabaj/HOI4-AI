"""Seam backend: OpenAI-compatible /v1/chat/completions (LM Studio, llama.cpp).

This is the path for grounding models (e.g. Holo1.5) whose imported Qwen-VL vision
GGUFs Ollama currently mishandles. Images are sent as data-URLs; structured output
uses ``response_format: json_schema``.
"""

from __future__ import annotations

import requests

from ..errors import BrainError, OllamaTimeoutError, OllamaUnavailableError


class OpenAICompatBackend:
    def __init__(self, endpoint: str, model: str, timeout_s: float = 120.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat(self, *, images, system, user, schema, timeout=None) -> str:
        content = [{"type": "text", "text": user}]
        for b in images:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
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
            raise OllamaUnavailableError(f"cannot reach server at {self.endpoint}") from e
        except requests.exceptions.Timeout as e:
            raise OllamaTimeoutError(f"call timed out after {timeout or self.timeout_s}s") from e
        except requests.exceptions.RequestException as e:  # pragma: no cover
            raise BrainError(f"request failed: {e}") from e
        if r.status_code != 200:
            raise BrainError(f"HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as e:
            raise BrainError(f"malformed response: {r.text[:200]}") from e
