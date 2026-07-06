"""Prompt + JSON-schema builders, one per decision type.

Each returns ``(system, user, schema)``. Schemas constrain the model's structured
output; enum lists are generated from live enum members so schema and validator
cannot drift. READ prompts (numbers/dates) see a tight ROI crop; DECISION
prompts (which state/tech) see the full frame — a map choice needs the map.
"""

from __future__ import annotations

from ..enums import BuildingType, GermanState, Tech

_SYSTEM_READ = (
    "You read a small cropped region of a Hearts of Iron IV screenshot and report "
    "exactly what it shows. Output JSON only, no commentary."
)
_SYSTEM_DECIDE = (
    "You are an expert Hearts of Iron IV player (Germany, 1936). You look at a "
    "cropped game panel and choose the single best option. Output JSON only."
)


# Fields not rendered as a printed digit anywhere in the UI — the VLM must derive
# the value from what the crop shows instead of transcribing a number.
_NUMBER_FIELD_PROMPTS = {
    "idle_research_slots": (
        "This crop shows the row of research slot cards. Count the slots that are "
        "EMPTY (no technology being researched in them). Reply with that count as "
        '{"value": N} — 0 if every slot is researching something. If the crop does '
        'not show the research slot row, reply {"value": -1}.'
    ),
    "free_civ_slots": (
        "This crop shows the construction header with civilian factory counts "
        "rendered like '38/38' (available/total), plus trade/owned breakdowns. "
        'Reply with the FIRST number of the X/Y pair (available factories) as '
        '{"value": N}. If no X/Y pair is visible, reply {"value": -1}.'
    ),
}


def number_prompt(field: str) -> tuple[str, str, dict]:
    user = _NUMBER_FIELD_PROMPTS.get(field) or (
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


def which_state_prompt(
    options: list[GermanState], building: BuildingType | None = None
) -> tuple[str, str, dict]:
    values = [o.value for o in options]
    building_label = (building or BuildingType.CIVILIAN_FACTORY).value.replace("_", " ")
    user = (
        f"Pick the single best German state to build a {building_label} in — prefer "
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


def locate_prompt(instruction: str) -> tuple[str, str, dict]:
    """Grounding: find one UI element on a (pre-cropped) image.

    Callers must crop-then-ground: hand the model the smallest region that can
    contain the target (map area, research panel, popup), never the full frame.
    Coordinates are 0-1000 normalized on the supplied image, matching
    ``geometry.NORM_SCALE`` so the executor can click them directly.
    """
    system = (
        "You precisely locate user-interface elements in screenshots of "
        "Hearts of Iron IV. Output JSON only, no commentary."
    )
    user = (
        f"Locate {instruction}. Reply with the CENTER of the target as "
        '{"x": X, "y": Y}, integers normalized to 0-1000 on this image '
        "(0,0 is the top-left corner; 1000,1000 the bottom-right)."
    )
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "minimum": 0, "maximum": 1000},
            "y": {"type": "integer", "minimum": 0, "maximum": 1000},
        },
        "required": ["x", "y"],
    }
    return system, user, schema


def recovery_prompt(options: list[str]) -> tuple[str, str, dict]:
    """One bounded hint when deterministic recovery failed (stuck-consultant)."""
    user = (
        "The automated player is stuck: pressing escape and the map-mode key did "
        "not return the game to the normal map view. Look at the screenshot and "
        "pick the ONE action most likely to unblock it — 'click_event_option' "
        "clicks an event window's option button; choose 'none' if nothing would "
        f"help. Choose exactly one of: {options}. Reply {{\"action\": \"<option>\"}}."
    )
    schema = {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": list(options)}},
        "required": ["action"],
    }
    return _SYSTEM_DECIDE, user, schema
