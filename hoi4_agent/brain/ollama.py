"""Default backend: Ollama /api/chat with native vision + JSON-schema `format`."""

from __future__ import annotations

import requests

from ..errors import BackendTimeoutError, BackendUnavailableError, BrainError


class OllamaBackend:
    def __init__(self, endpoint: str, model: str, timeout_s: float = 120.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat(self, *, images, system, user, schema, image_mime="image/png", timeout=None) -> str:
        # image_mime is unused here by design: Ollama takes bare base64 and
        # sniffs the format itself, so there is no media type to declare.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": images},
            ],
            "stream": False,
            "format": schema,  # JSON schema, not just "json" — constrains generation
            "options": {"temperature": 0},
        }
        try:
            r = requests.post(
                f"{self.endpoint}/api/chat",
                json=payload,
                timeout=timeout or self.timeout_s,
            )
        except requests.exceptions.ConnectionError as e:
            raise BackendUnavailableError(f"cannot reach Ollama at {self.endpoint}") from e
        except requests.exceptions.Timeout as e:
            raise BackendTimeoutError(f"Ollama call timed out after {timeout or self.timeout_s}s") from e
        except requests.exceptions.RequestException as e:  # pragma: no cover - rare
            raise BrainError(f"Ollama request failed: {e}") from e
        if r.status_code != 200:
            raise BrainError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()["message"]["content"]
        except (ValueError, KeyError) as e:
            raise BrainError(f"malformed Ollama response: {r.text[:200]}") from e
