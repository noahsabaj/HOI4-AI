"""Which vanilla files a two-country total conversion has to BLANK, derived from the install.

Launch 1 of the custom map (v1.19.3) loaded the map, then died with an access violation while the engine
validated vanilla script against databases the mod had unloaded (focus trees, decisions, events, country
tags): ~20k error lines, then a null dereference. "Harmless errors" was a wrong assumption.

Launches 2-3 (with this plan) cut error.log from 20151 to 3746 lines (1995 without the rescue rule) and showed
two things: blanking whole feature databases that OTHER kept files index (collections, faction rules and goals,
special projects, autonomy states, intelligence agencies) only trades tag errors for "empty database" errors,
so those folders are no longer candidates; and the access violation is NOT caused by these files: it persisted
unchanged (same faulting address) at every level of blanking. See FIRST_LAUNCH.md for the open crash.

The plan is computed, not hand-listed: inside a fixed set of CANDIDATE folders (feature content that is
written per country) a file is *bound* when it names an identifier that no longer exists: a vanilla
country tag, focus id, decision or decision-category id, or event id. Bound files are overridden by a
same-named comment-only file, unless a file we keep still calls something they define (scripted effects and
triggers, dynamic modifiers, ideas): those stay, to a fixpoint, and their remaining errors are accepted.
Idea files follow their own rule: only law and spirit files survive (``_*.txt`` and files without any
per-country category). Deterministic: sorted traversal, no timestamps. Reads the install, writes nothing.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from ...clausewitz import parse

BLANK_TEXT = "# Neutralised by the HOI4-AI arena generator: the vanilla file names tags/focuses/decisions/events\n" \
             "# that do not exist in this total conversion.\n"
# Folders (relative to the install) whose bound files are blanked. Not recursive.
CANDIDATE_FOLDERS = (
    "common/ideas", "common/scripted_effects", "common/scripted_triggers", "common/scripted_localisation",
    "common/dynamic_modifiers", "common/military_industrial_organization/organizations",
    "common/units/names_divisions", "common/units/names_ships", "common/units/names_railway_guns",
    "common/units/codenames_operatives", "common/peace_conference/ai_peace",
    "common/peace_conference/cost_modifiers", "common/ai_navy/goals", "common/ai_navy/taskforce",
    "common/ai_navy/fleet", "common/operations", "common/raids", "common/focus_inlay_windows", "common/technology_sharing", "common/bop",
    "common/unit_medals", "common/scripted_guis",
    "common/scripted_diplomatic_actions", "common/scorers/country",
    "common/ai_templates", "common/ai_equipment", "common/game_rules",
    "common/continuous_focus", "common/profile_pictures", "common/profile_backgrounds",
)
CANDIDATE_FILES = ("common/achievements.txt",)
PER_COUNTRY_IDEA_CATEGORIES = {
    "country", "hidden_ideas", "political_advisor", "army_chief", "navy_chief", "air_chief", "high_command",
    "theorist", "tank_manufacturer", "naval_manufacturer", "aircraft_manufacturer", "materiel_manufacturer",
    "industrial_concern"}
_TOKEN = re.compile(r"[A-Za-z_][\w.]*")
_COMMENT = re.compile(r"#[^\n]*")


def _read(path: Path) -> str:
    return _COMMENT.sub("", path.read_text(encoding="utf-8-sig", errors="replace"))


def _top_keys(text: str, depth: int = 0) -> set[str]:
    """Keys that open a block at brace depth ``depth`` (cheap scan, no full parse)."""
    out, level = set(), 0
    for match in re.finditer(r'"(?:\\.|[^"\\])*"|[{}]|([A-Za-z_][\w.\-]*)\s*=\s*\{', text):
        token = match.group(0)
        if token == "}":
            level -= 1
        elif token.endswith("{"):
            if level == depth and match.group(1):
                out.add(match.group(1))
            level += 1
    return out


def unloaded_identifiers(game: Path, replaced: tuple[str, ...]) -> set[str]:
    """Tags, focus ids, decision and category ids, event ids that vanish with the replaced folders."""
    out: set[str] = set()
    for path in sorted((game / "common" / "country_tags").glob("*.txt")):
        out |= set(re.findall(r"^\s*([A-Z][A-Z0-9]{2})\s*=", _read(path), re.M))
    if "common/national_focus" in replaced:
        for path in sorted((game / "common" / "national_focus").glob("*.txt")):
            out |= set(re.findall(r"\bid\s*=\s*([A-Za-z_][\w.]*)", _read(path)))
    if "common/decisions" in replaced:
        for path in sorted((game / "common" / "decisions").glob("*.txt")):
            text = _read(path)
            out |= _top_keys(text, 0) | _top_keys(text, 1)
    if "events" in replaced:
        for path in sorted((game / "events").glob("*.txt")):
            out |= set(re.findall(r"\bid\s*=\s*([A-Za-z_][\w]*\.\d+)", _read(path)))
    return out - {"id", "limit", "if", "modifier", "allowed", "available", "visible", "icon", "cost", "name"}


def _idea_file_is_generic(path: Path) -> bool:
    if path.name.startswith("_"):
        return True
    try:
        ideas = parse(_read(path)).get("ideas", {})
    except Exception:  # noqa: BLE001 - an unparsable vanilla file is left alone
        return True
    blocks = ideas if isinstance(ideas, list) else [ideas]
    categories = {key for block in blocks if isinstance(block, dict) for key in block}
    return bool(categories) and not categories & PER_COUNTRY_IDEA_CATEGORIES


@lru_cache(maxsize=4)
def _plan(game_text: str, replaced: tuple[str, ...], own_tokens: frozenset[str],
          rescue: bool) -> tuple[tuple[str, ...], tuple[str, ...]]:
    game = Path(game_text)
    gone = unloaded_identifiers(game, replaced)
    candidates: dict[str, Path] = {}
    for folder in CANDIDATE_FOLDERS:
        for path in sorted((game / folder).glob("*.txt")):
            candidates[f"{folder}/{path.name}"] = path
    for name in CANDIDATE_FILES:
        if (game / name).is_file():
            candidates[name] = game / name
    tokens: dict[str, set[str]] = {}
    corpus = [*sorted((game / "common").rglob("*.txt")), *sorted((game / "history" / "general").glob("*.txt")),
              *sorted((game / "music").rglob("*.txt"))]
    for path in corpus:
        relative = path.relative_to(game).as_posix()
        if relative.rsplit("/", 1)[0] in replaced:
            continue
        tokens[relative] = set(_TOKEN.findall(_read(path)))
    blank = set()
    for relative, path in candidates.items():
        if relative.startswith("common/ideas/"):
            if not _idea_file_is_generic(path):
                blank.add(relative)
        elif tokens.get(relative, set()) & gone:
            blank.add(relative)
    rescuable = ("common/scripted_effects/", "common/scripted_triggers/", "common/dynamic_modifiers/")
    defines = {relative: _top_keys(_read(candidates[relative]), 0) for relative in blank if relative.startswith(rescuable)}
    used = set(own_tokens)
    for relative, found in tokens.items():  # callers that certainly stay: everything outside the blank set
        if relative not in blank:
            used |= found
    rescued: set[str] = set()
    while rescue:  # a kept file that calls a scripted effect/trigger keeps its file alive, to a fixpoint
        back = {relative for relative, names in defines.items() if relative in blank and names & used}
        if not back:
            break
        blank -= back
        rescued |= back
        for relative in back:
            used |= tokens[relative]
    return tuple(sorted(blank)), tuple(sorted(rescued))


def blank_plan(game: Path, replaced: tuple[str, ...], own_text: str = "", rescue: bool = True) -> dict[str, Any]:
    """{"blank": files to override with ``BLANK_TEXT``, "rescued": bound files kept because kept script calls them}."""
    blank, rescued = _plan(str(game.resolve()), tuple(replaced), frozenset(_TOKEN.findall(own_text)), rescue)
    return {"blank": list(blank), "rescued_bound_files": list(rescued), "candidate_folders": list(CANDIDATE_FOLDERS),
            "rule": "bound = names an unloaded tag/focus/decision/event id; ideas: only law and spirit files survive"}
