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
from ..schemas import GameDate


@dataclass(frozen=True)
class SaveFacts:
    country: str
    date: str | None = None
    researching: tuple[str, ...] | None = None
    civilian_factories: int | None = None
    construction_lines: int | None = None
    construction_states: tuple[str, ...] | None = None
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


def _construction_line_blocks(country_block: dict) -> list[dict] | None:
    lines = _dig(country_block, "construction", "line")
    if isinstance(lines, list):
        return [x for x in lines if isinstance(x, dict)]
    if isinstance(lines, dict):
        return [lines]
    return None


def _construction_lines(country_block: dict) -> int | None:
    blocks = _construction_line_blocks(country_block)
    return None if blocks is None else len(blocks)


# Candidate keys a construction line might carry its target state under. Paradox
# schemas drift between patches, so this is best-effort like everything else here
# — anything not found is reported missing, never guessed.
_STATE_KEYS = ("state", "province", "target", "location", "state_id")


def _construction_states(country_block: dict) -> tuple[str, ...] | None:
    """Where the queued projects are being built, if the save says so.

    This is the only independent answer to "did the click land on the state the
    playbook intended?" — the live postcondition (queue length +1) is a
    cardinality check that cannot tell Ruhr from Bavaria.
    """
    blocks = _construction_line_blocks(country_block)
    if not blocks:
        return None
    found = []
    for b in blocks:
        for key in _STATE_KEYS:
            v = b.get(key)
            if isinstance(v, (str, int)):
                found.append(str(v))
                break
    return tuple(found) if found else None


def dates_agree(traced: str | None, save: str | None) -> bool | None:
    """Do a trace date and a save-file date name the same in-game day?

    The two sides are formatted differently and must be normalized before they
    can be compared: a ``TraceRecord`` carries ``GameDate.to_str()``'s
    zero-padded ``"1936.01.01"``, while a Clausewitz save carries
    ``"1936.1.1.12"`` — year.month.day.HOUR. Comparing the raw strings reported
    MISMATCH for identical days, which made the independent-verifier check
    useless. ``None`` means undecidable (a side is absent or unparseable) —
    never a silent False.
    """
    if not traced or not save:
        return None
    a, b = GameDate.from_ui_text(traced), GameDate.from_ui_text(save)
    if a is None or b is None:
        return None
    return a == b


def intended_build_states(records) -> tuple[str, ...]:
    """States the trace says the agent MEANT to build in, oldest first.

    Read from each successful build record's resolved intent. Pairing this with
    ``SaveFacts.construction_states`` is the only way to check identity rather
    than cardinality: the live postcondition proves a project was queued, never
    that it was queued where the playbook asked.
    """
    out = []
    for r in records:
        intent = getattr(r, "parsed_intent", None) or {}
        if intent.get("tool") == "build_in_state" and getattr(r, "verdict", None) == "ok":
            state = intent.get("state")
            if state:
                out.append(str(state))
    return tuple(out)


def build_identity_check(records, facts: SaveFacts) -> tuple[str, list[str]]:
    """Compare intended build states against the save. Returns (status, notes).

    status is "match", "mismatch", or "undecidable" — undecidable when either
    side is unavailable, never a silent pass.
    """
    intended = intended_build_states(records)
    observed = facts.construction_states
    if not intended:
        return "undecidable", ["no successful build_in_state records in the trace"]
    if observed is None:
        return "undecidable", [
            f"trace intended {list(intended)}",
            "save exposes no per-line state (schema drift? inspect the save's "
            "construction block and extend _STATE_KEYS)",
        ]
    missing = [s for s in intended if s not in observed]
    notes = [f"trace intended {list(intended)}", f"save shows {list(observed)}"]
    if missing:
        notes.append(f"intended but NOT present in the save: {missing}")
        return "mismatch", notes
    return "match", notes


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
    states = _construction_states(block)
    if states is None:
        missing.append("construction_states")

    return SaveFacts(
        country=country, date=date, researching=researching,
        civilian_factories=civ, construction_lines=lines,
        construction_states=states, missing=tuple(missing),
    )
