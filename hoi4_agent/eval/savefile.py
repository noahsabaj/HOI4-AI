"""Save-file ground truth: parse a (non-ironman, text) HOI4 autosave and
extract the facts the agent otherwise reads off pixels.

This is the strongest "verifier independent of the actor": the save file IS
the game state. Used offline to audit traces and to sanity-label the M0
corpus — the live actor still plays by sight.

Every fact is best-effort with several candidate schema paths (Paradox saves
drift between patches); what can't be found is reported in ``missing``, never
guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import clausewitz


@dataclass(frozen=True)
class SaveFacts:
    country: str
    date: str | None = None
    researching: tuple[str, ...] | None = None
    civilian_factories: int | None = None
    construction_lines: int | None = None
    missing: tuple[str, ...] = ()


def _dig(block, *path):
    cur = block
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _country_block(data: dict, country: str):
    countries = data.get("countries")
    if isinstance(countries, dict) and isinstance(countries.get(country), dict):
        return countries[country]
    if isinstance(data.get(country), dict):
        return data[country]
    return None


def _researching(country_block: dict) -> tuple[str, ...] | None:
    for path in (("technology", "slots"), ("research", "slots"), ("research",)):
        slots = _dig(country_block, *path)
        if isinstance(slots, dict) and slots:
            return tuple(str(k) for k in slots if k != "__items__")
    return None


def _civilian_factories(country_block: dict) -> int | None:
    for path in (("civilian_factories",), ("industry", "civilian_factories"),
                 ("num_of_civilian_factories",)):
        v = _dig(country_block, *path)
        if isinstance(v, int):
            return v
    return None


def _construction_lines(country_block: dict) -> int | None:
    lines = _dig(country_block, "construction", "line")
    if isinstance(lines, list):
        return len(lines)
    if isinstance(lines, dict):
        return 1
    return None


def read_save_facts(path: str | Path, country: str = "GER") -> SaveFacts:
    data = clausewitz.parse_file(path)
    if not isinstance(data, dict):
        data = {}
    missing: list[str] = []

    date = data.get("date")
    date = str(date) if isinstance(date, str) else None
    if date is None:
        missing.append("date")

    block = _country_block(data, country)
    if block is None:
        missing += [f"country block {country!r}", "researching",
                    "civilian_factories", "construction_lines"]
        return SaveFacts(country=country, date=date, missing=tuple(missing))

    researching = _researching(block)
    if researching is None:
        missing.append("researching")
    civ = _civilian_factories(block)
    if civ is None:
        missing.append("civilian_factories")
    lines = _construction_lines(block)
    if lines is None:
        missing.append("construction_lines")

    return SaveFacts(
        country=country, date=date, researching=researching,
        civilian_factories=civ, construction_lines=lines, missing=tuple(missing),
    )
