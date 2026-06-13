"""Load a playbook (ordered Goals) from TOML."""

from __future__ import annotations

import tomllib
from pathlib import Path

from ..enums import BuildingType, GermanState, PreconditionKind, Tech, ToolName
from ..errors import ConfigError
from ..schemas import GameDate, Goal, Precondition

_ENUM_FIELDS = {
    "tool": ToolName,
    "building": BuildingType,
    "state": GermanState,
    "tech": Tech,
}


def _coerce(field: str, value, enum_cls):
    try:
        return enum_cls(value)
    except ValueError as e:
        allowed = [m.value for m in enum_cls]
        raise ConfigError(f"goal {field}={value!r} not in {allowed}") from e


def _precondition(g: dict) -> Precondition:
    kind_s = g.get("precondition", "always")
    try:
        kind = PreconditionKind(kind_s)
    except ValueError as e:
        raise ConfigError(f"unknown precondition {kind_s!r}") from e
    date = None
    if "precondition_date" in g:
        date = GameDate.from_str(g["precondition_date"])
    if kind in (PreconditionKind.DATE_BEFORE, PreconditionKind.DATE_AFTER) and date is None:
        raise ConfigError(f"precondition {kind_s} requires precondition_date")
    return Precondition(kind=kind, date=date)


def parse_goals(raw: dict) -> list[Goal]:
    items = raw.get("goal", [])
    if not items:
        raise ConfigError("playbook has no [[goal]] entries")
    goals: list[Goal] = []
    seen_ids: set[str] = set()
    for i, g in enumerate(items):
        if "id" not in g or "tool" not in g:
            raise ConfigError(f"goal #{i} missing 'id' or 'tool'")
        if g["id"] in seen_ids:
            raise ConfigError(f"duplicate goal id {g['id']!r}")
        seen_ids.add(g["id"])
        goals.append(
            Goal(
                id=g["id"],
                tool=_coerce("tool", g["tool"], ToolName),
                building=_coerce("building", g["building"], BuildingType) if "building" in g else None,
                state=_coerce("state", g["state"], GermanState) if "state" in g else None,
                tech=_coerce("tech", g["tech"], Tech) if "tech" in g else None,
                speed=int(g["speed"]) if "speed" in g else None,
                paused=bool(g["paused"]) if "paused" in g else None,
                precondition=_precondition(g),
                repeatable=bool(g.get("repeatable", False)),
                needs_judgment=bool(g.get("needs_judgment", False)),
            )
        )
    return goals


def load_playbook(path: str | Path) -> list[Goal]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"playbook not found: {p}")
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    return parse_goals(raw)
