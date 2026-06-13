"""Strict JSON extraction + coercion. Failures raise typed BrainErrors, never None."""

from __future__ import annotations

import json
import re
from enum import StrEnum

from ..errors import EnumError, ParseError, SchemaError


def _as_dict(obj) -> dict:
    if not isinstance(obj, dict):
        raise SchemaError(f"expected JSON object, got {type(obj).__name__}")
    return obj


def extract_json(raw: str) -> dict:
    """Parse a JSON object from model output: direct, then fenced, then first brace span."""
    if not raw or not raw.strip():
        raise ParseError(raw or "")
    s = raw.strip()
    try:
        return _as_dict(json.loads(s))
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", s, re.DOTALL)
    if m:
        try:
            return _as_dict(json.loads(m.group(1).strip()))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return _as_dict(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass
    raise ParseError(raw)


def coerce_int(v) -> int:
    if isinstance(v, bool):
        raise SchemaError("expected int, got bool")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            pass
    raise SchemaError(f"not an integer: {v!r}")


def coerce_enum(v, enum_cls: type[StrEnum], options, field: str):
    allowed = [o.value for o in options]
    if isinstance(v, str) and v in allowed:
        return enum_cls(v)
    raise EnumError(field, v, allowed)
