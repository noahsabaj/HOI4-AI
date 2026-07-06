"""Brain: turns an LLMBackend into typed reads + decisions.

- Reads (``read_number`` / ``read_date``) return ``None`` on a parse/enum failure
  (treated as *uncertain*), but let infrastructure failures (endpoint down /
  timeout) propagate — we never silently mask a dead model endpoint.
- Decisions (``which_state`` / ``which_tech`` / ``yes_no``) raise a typed BrainError
  on a bad/invalid answer so the controller can treat the step as UNCERTAIN.
"""

from __future__ import annotations

from PIL import Image

from ..config import LLMConfig
from ..enums import BuildingType, GermanState, Tech
from ..errors import ConfigError, EnumError, ParseError, SchemaError
from ..schemas import GameDate
from . import prompts
from .llm import LLMBackend, encode_image
from .ollama import OllamaBackend
from .openai_compat import OpenAICompatBackend
from .parse import coerce_enum, coerce_int, extract_json


class Brain:
    def __init__(self, backend: LLMBackend, encode=encode_image) -> None:
        self.backend = backend
        self._encode = encode
        # Last exchange, for the trace (prompt + raw output of the most recent call).
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.last_raw: str | None = None

    def _ask(self, crop: Image.Image, system: str, user: str, schema: dict, fmt: str = "PNG") -> dict:
        raw = self.backend.chat(
            images=[self._encode(crop, fmt)], system=system, user=user, schema=schema
        )
        self.last_system, self.last_user, self.last_raw = system, user, raw
        return extract_json(raw)

    @property
    def last_prompt(self) -> str | None:
        if self.last_user is None:
            return None
        return f"[system] {self.last_system}\n[user] {self.last_user}"

    # --- Reader protocol (perception T1) ---
    def read_number(self, crop: Image.Image, field: str) -> int | None:
        system, user, schema = prompts.number_prompt(field)
        try:
            d = self._ask(crop, system, user, schema)
            return coerce_int(d.get("value"))
        except (ParseError, SchemaError, EnumError):
            return None

    def read_date(self, crop: Image.Image) -> GameDate | None:
        system, user, schema = prompts.date_prompt()
        try:
            d = self._ask(crop, system, user, schema)
            year = coerce_int(d.get("year"))
            if year < 1900:
                return None
            # GameDate range-checks month/day; a hallucinated month 13 -> ValueError.
            return GameDate(year, coerce_int(d.get("month")), coerce_int(d.get("day")))
        except (ParseError, SchemaError, EnumError, ValueError):
            return None

    # --- decisions (T2): full frames, JPEG keeps the payload sane ---
    def which_state(
        self, crop: Image.Image, options: list[GermanState], building: BuildingType | None = None
    ) -> GermanState:
        system, user, schema = prompts.which_state_prompt(options, building)
        d = self._ask(crop, system, user, schema, fmt="JPEG")
        return coerce_enum(d.get("state"), GermanState, options, "state")

    def which_tech(self, crop: Image.Image, options: list[Tech]) -> Tech:
        system, user, schema = prompts.which_tech_prompt(options)
        d = self._ask(crop, system, user, schema, fmt="JPEG")
        return coerce_enum(d.get("tech"), Tech, options, "tech")

    def yes_no(self, crop: Image.Image, question: str) -> bool:
        system, user, schema = prompts.yes_no_prompt(question)
        d = self._ask(crop, system, user, schema, fmt="JPEG")
        ans = d.get("answer")
        if ans not in ("yes", "no"):
            raise EnumError("answer", ans, ["yes", "no"])
        return ans == "yes"


def build_backend(cfg: LLMConfig) -> LLMBackend:
    if cfg.backend == "ollama":
        return OllamaBackend(cfg.endpoint, cfg.model, cfg.timeout_s)
    if cfg.backend == "openai_compat":
        return OpenAICompatBackend(cfg.endpoint, cfg.model, cfg.timeout_s)
    raise ConfigError(f"unknown llm backend {cfg.backend!r} (want 'ollama' or 'openai_compat')")
