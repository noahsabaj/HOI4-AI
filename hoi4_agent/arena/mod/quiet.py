"""Keep vanilla content from interrupting the vanilla-region arena, WITHOUT ``replace_path``.

Live matches on mod 485c0190 showed a "World News" event window over the map (Spanish Civil War, 25 Feb
1936) and war-declaration toasts: perception was blind for about 30% of frames. Removing vanilla folders
(``replace_path``) was rejected: ``neutralise.py`` documents ~20k error lines and an unresolved crash when
events, focus trees or decisions disappear under the vanilla script that references them. Instead, three
additive layers, none of which unloads an identifier:

1. ``hidden = yes`` is inserted into every vanilla ``news_event`` (same file names, same ids): the event
   still exists and still fires, it just has no window.
2. The generated profile ``settings.txt`` turns event/news popups and hints off (written in generate.py).
3. World freeze (``world="inert"``): a third, inert country (contract name NEU) is seeded with ONE vanilla
   state next to the arena and annexes every country except the two sides at ``on_startup``. No other
   country exists afterwards, so nobody declares war, runs focuses or fires tag events; vanilla tags, focus
   trees, decisions and events all stay loaded. The inert country may not declare war or join factions.

Honesty: layers 1 and 3 have NOT been loaded by the game yet. Known unknowns: whether ~80 annexations at
startup are slow or produce their own toasts; whether option effects of hidden news events still execute
(they are flavour in vanilla); toasts themselves are per-profile message settings
(``messagetypes_custom.txt`` in the user directory, format not documented in the install) and cannot be
generated here. ``world="vanilla"`` skips layer 3.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..contracts import ArenaError
from .generate import COLORS, Bare, Pairs, _flag, _write, emit, q, vanilla_tags
from .region import Region, VanillaState, read_states

NEUTRAL_TAGS = ("NEU", "ZNE", "XNE", "QNE")
NEUTRAL_COLOR = (96, 96, 96)
_NEWS = re.compile(r"^news_event\s*=\s*\{[^}\n]*$", re.M)
_NEWS_ID = re.compile(r"^news_event\s*=\s*\{.*?^\s*id\s*=\s*([\w.]+)", re.M | re.S)


def choose_neutral_tag(game: Path, taken: set[str]) -> str:
    used = vanilla_tags(game) | taken
    free = [tag for tag in NEUTRAL_TAGS if tag not in used]
    if not free:
        raise ArenaError("no free country tag for the inert country")
    return free[0]


def hidden_news_overrides(game: Path) -> dict[str, str]:
    """events file name -> vanilla text with ``hidden = yes`` as the first line of every news_event."""
    out = {}
    root = game / "events"
    for path in sorted(root.glob("*.txt")) if root.is_dir() else ():
        text = path.read_bytes().decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
        changed, count = _NEWS.subn(lambda m: m.group(0) + "\n\thidden = yes # HOI4-AI arena: no news window", text)
        if count:
            out[path.name] = changed
    return out


def seed_state(game: Path, region: Region) -> VanillaState:
    """The vanilla state the inert country starts with: the one holding the lowest-numbered land province
    that borders the arena from outside. Its provinces are NOT changed, only its owner."""
    arena = {state.id for state in region.states}
    states = read_states(game)
    for province in region.external_land_neighbors:
        for state in states.values():
            if province in state.provinces and state.id not in arena:
                return state
    raise ArenaError("the arena has no outside neighbour state to seed the inert country with")


def write_quiet(mod: Path, game: Path, region: Region, tags: dict[str, str], world: str) -> dict[str, Any]:
    """Write layers 1 and 3 into ``mod``; returns the manifest block describing what was done."""
    news = hidden_news_overrides(game)
    for name, text in news.items():
        _write(mod / "events" / name, text)
    report: dict[str, Any] = {
        "replace_path": [], "world": world,
        "overridden_vanilla_files": {
            "events/*.txt": {"files": sorted(news), "why": "hidden = yes on every news_event: no news window; all "
                                                           "event ids stay defined"}},
        "profile_settings": {"popup_news": False, "popup_events": False, "popup_minor_events": False,
                             "pause_on_popups": False, "hints": False},
        "not_scriptable": "toasts are per-profile message settings (messagetypes_custom.txt); set them once in the "
                          "UI of this profile - mod-build never deletes profile files",
    }
    if world != "inert":
        return report
    tag = choose_neutral_tag(game, set(tags.values()))
    seed = seed_state(game, region)
    source = (game / "history" / "states" / seed.file_name).read_bytes().decode("utf-8-sig", errors="replace")
    text, count = re.subn(r"\bowner\s*=\s*[A-Z][A-Z0-9]{2}\b", f"owner = {tag}\n\t\tadd_core_of = {tag}",
                          source.replace("\r\n", "\n"), count=1)
    if count != 1:
        raise ArenaError(f"cannot find the owner of seed state {seed.file_name}")
    _write(mod / "history" / "states" / seed.file_name, text)
    _write(mod / "common" / "country_tags" / "zz_arena_world.txt", f'{tag} = "countries/Arena Neutral.txt"\n')
    r, g, b = NEUTRAL_COLOR
    _write(mod / "common" / "countries" / "Arena Neutral.txt",
           "graphical_culture = western_european_gfx\ngraphical_culture_2d = western_european_2d\n"
           f"color = {{ {r} {g} {b} }}\n")
    for folder, size in (("", (82, 52)), ("medium", (41, 26)), ("small", (10, 7))):
        _flag(mod / "gfx" / "flags" / folder / f"{tag}.tga", NEUTRAL_COLOR, size)
    _write(mod / "history" / "countries" / f"{tag} - Arena Neutral.txt", emit([
        ("capital", seed.id), ("set_research_slots", 0), ("set_stability", 1.0), ("set_war_support", 0.0),
        ("set_politics", [("ruling_party", "neutrality"), ("elections_allowed", False)]),
        ("set_popularities", [("neutrality", 100)]), ("add_ideas", "arena_inert")]))
    _write(mod / "common" / "ideas" / "arena_world_ideas.txt", emit([("ideas", [("country", [("arena_inert", [
        ("picture", "generic_intel_bonus"), ("removal_cost", -1), ("allowed_civil_war", [("always", True)]),
        ("rule", [("can_not_declare_war", True), ("can_join_factions", False), ("can_create_factions", False),
                  ("can_send_volunteers", False)]),
        ("modifier", [("surrender_limit", 1.0), ("political_power_factor", -1.0)])])])])]))
    others: Pairs = [("not", [("tag", t)]) for t in (tags["BLU"], tags["RED"], tag)]
    _write(mod / "common" / "scripted_effects" / "arena_world_effects.txt", emit([("arena_freeze_world", [
        ("log", q("ARENA_WORLD freeze entered")),
        ("if", [("limit", [("not", [("has_global_flag", "arena_world_frozen")])]),
                ("set_global_flag", "arena_world_frozen"),
                ("every_country", [("limit", others),
                                   (tag, [("annex_country", [("target", "PREV"), ("transfer_troops", False)])])]),
                ("log", q(f"ARENA_WORLD frozen inert={tag} seed_state={seed.id}"))])])]))
    _write(mod / "common" / "on_actions" / "arena_world_on_actions.txt", emit([("on_actions", [
        ("on_startup", [("effect", [(tag, [("arena_freeze_world", True)])])])])]))
    people = [("male", [("names", Bare((q("Nemo"), q("Otto"))))]), ("surnames", Bare((q("Inert"), q("Null"))))]
    _write(mod / "common" / "names" / "zz_arena_world_names.txt", emit([(tag, people)]))
    rows = {f"{tag}{suffix}{part}": "Inert" for suffix in ("", "_fascism", "_democratic", "_neutrality", "_communism")
            for part in ("", "_DEF", "_ADJ")}
    rows |= {"arena_inert": "Inert world", "arena_inert_desc": "Owns everything outside the arena and does nothing."}
    _write(mod / "localisation" / "english" / "arena_world_l_english.yml",
           "﻿l_english:\n" + "".join(f' {key}:0 "{value}"\n' for key, value in sorted(rows.items())))
    report["inert_country"] = {"tag": tag, "color": list(NEUTRAL_COLOR), "seed_state": seed.id,
                               "seed_state_file": seed.file_name, "seed_state_vanilla_owner": seed.owner}
    report["overridden_vanilla_files"][f"history/states/{seed.file_name}"] = {
        "why": f"owner becomes {tag} so the inert country exists at startup; provinces unchanged"}
    report["log_lines"] = ["ARENA_WORLD freeze entered", "ARENA_WORLD frozen inert=<tag> seed_state=<id>"]
    return report


def colors_entry(tag: str) -> str:
    r, g, b = NEUTRAL_COLOR
    return f"{tag} = {{\n\tcolor = rgb {{ {r} {g} {b} }}\n\tcolor_ui = rgb {{ {r} {g} {b} }}\n}}\n"


def quiet_problems(game: Path, mod: Path, quiet: dict[str, Any], arena_tags: set[str]) -> list[str]:
    """Static guarantees: no identifier disappears. Every vanilla news event id is still defined by the
    override of its file and is hidden; the seed state keeps its vanilla provinces; the inert tag is free and
    far enough in colour from both sides; nothing uses replace_path."""
    problems: list[str] = []
    if "replace_path" in (mod / "descriptor.mod").read_text(encoding="utf-8"):
        problems.append("descriptor.mod uses replace_path; the quiet layers must stay additive")
    for path in sorted((mod / "events").glob("*.txt")) if (mod / "events").is_dir() else ():
        vanilla = game / "events" / path.name
        if not vanilla.is_file():
            problems.append(f"events/{path.name} does not override a vanilla file")
            continue
        ours = path.read_text(encoding="utf-8")
        theirs = vanilla.read_bytes().decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
        if ours.replace("\n\thidden = yes # HOI4-AI arena: no news window", "") != theirs:
            problems.append(f"events/{path.name} differs from vanilla by more than the hidden lines")
        if _NEWS_ID.findall(ours) != _NEWS_ID.findall(theirs):
            problems.append(f"events/{path.name} does not define the same news event ids as vanilla")
        if len(_NEWS.findall(ours)) != ours.count("hidden = yes # HOI4-AI arena"):
            problems.append(f"events/{path.name} has a news_event without the hidden line")
    missing = sorted(set(hidden_news_overrides(game)) - {p.name for p in (mod / "events").glob("*.txt")})
    if missing:
        problems.append(f"vanilla news event files without an override: {missing[:5]}")
    inert = quiet.get("inert_country")
    if quiet.get("world") == "inert" and not inert:
        problems.append("world is inert but no inert country is recorded")
    if inert:
        if inert["tag"] in vanilla_tags(game) | arena_tags:
            problems.append("the inert country's tag collides with another tag")
        states = read_states(game, (inert["seed_state"],))
        state_text = (mod / "history" / "states" / inert["seed_state_file"]).read_text(encoding="utf-8")
        listed = re.search(r"\bprovinces\s*=\s*\{([^}]*)\}", re.sub(r"#[^\n]*", "", state_text))
        if listed is None or sorted(int(t) for t in listed.group(1).split()) != sorted(
                states[inert["seed_state"]].provinces):
            problems.append("the seed state no longer lists exactly its vanilla provinces")
        if f"owner = {inert['tag']}" not in state_text:
            problems.append("the seed state is not owned by the inert country")
        for side, color in COLORS.items():
            if sum(abs(a - b) for a, b in zip(color, inert["color"], strict=True)) < 150:
                problems.append(f"the inert colour is too close to {side}")
    return problems
