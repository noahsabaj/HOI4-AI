"""Prompt + JSON-schema builders, one per decision type.

Each returns ``(system, user, schema)``. Schemas constrain the model's structured
output; enum lists are generated from live enum members so schema and validator
cannot drift. The model is always fed a tight crop, never the full frame.
"""

from __future__ import annotations

from ..enums import GermanState, Tech

_SYSTEM_READ = (
    "You read a small cropped region of a Hearts of Iron IV screenshot and report "
    "exactly what it shows. Output JSON only, no commentary."
)
_SYSTEM_DECIDE = (
    "You are an expert Hearts of Iron IV player (Germany, 1936). You look at a "
    "cropped game panel and choose the single best option. Output JSON only."
)


def number_prompt(field: str) -> tuple[str, str, dict]:
    user = (
        f"This crop shows the '{field}' indicator. Reply with the integer value you "
        'see as {"value": N}. If you cannot read a number, reply {"value": -1}.'
    )
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    return _SYSTEM_READ, user, schema


def date_prompt() -> tuple[str, str, dict]:
    user = (
        "This crop shows the in-game date. Report it as "
        '{"year": YYYY, "month": M, "day": D}. If unreadable, set year to -1.'
    )
    schema = {
        "type": "object",
        "properties": {
            "year": {"type": "integer"},
            "month": {"type": "integer"},
            "day": {"type": "integer"},
        },
        "required": ["year", "month", "day"],
    }
    return _SYSTEM_READ, user, schema


def which_state_prompt(options: list[GermanState]) -> tuple[str, str, dict]:
    values = [o.value for o in options]
    user = (
        "Pick the single best German state to build a civilian factory in — prefer "
        "the state with the most free building slots. Choose exactly one of: "
        f"{values}. Reply {{\"state\": \"<one of the options>\"}}."
    )
    schema = {
        "type": "object",
        "properties": {"state": {"type": "string", "enum": values}},
        "required": ["state"],
    }
    return _SYSTEM_DECIDE, user, schema


def which_tech_prompt(options: list[Tech]) -> tuple[str, str, dict]:
    values = [o.value for o in options]
    user = (
        "Pick the single best available technology to start researching now. "
        f"Choose exactly one of: {values}. Reply {{\"tech\": \"<one of the options>\"}}."
    )
    schema = {
        "type": "object",
        "properties": {"tech": {"type": "string", "enum": values}},
        "required": ["tech"],
    }
    return _SYSTEM_DECIDE, user, schema


def yes_no_prompt(question: str) -> tuple[str, str, dict]:
    user = f'{question} Reply {{"answer": "yes"}} or {{"answer": "no"}}.'
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
        "required": ["answer"],
    }
    return _SYSTEM_DECIDE, user, schema
