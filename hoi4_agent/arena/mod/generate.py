"""Generate the v0.1 arena as a mod ON THE VANILLA MAP, plus an isolated launch profile.

Output (all under one directory, never the Steam folder or the user's Documents):
``mod/hoi4_arena/`` (the mod), ``mod/hoi4_arena.mod``, ``dlc_load.json``, ``settings.txt``,
``arena_layout.json`` (``ArenaLayout``), ``arena_region.json`` (pixel frame, rivers evidence) and
``arena_manifest.json`` (scenario ids and splits, tags, decisions, log grammar, mod hash).

The mod hands a block of vanilla states to two new countries at war, replaces their state history
so each state is one side x one sector, and drives episodes from script: ``on_startup`` and a
clickable RESET decision run the same effect (wipe divisions, restore control, respawn the order
of battle, log ``ARENA_RESET``); ``on_daily_<TAG>`` logs ``ARENA_TICK`` and decides ``ARENA_OUTCOME``.

Honesty: NOTHING here has been loaded by the game. ``validate_mod`` is a static check (files parse,
names exist in this build's documentation, IDs exist in vanilla). Output is deterministic: sorted,
``\\n`` newlines, no timestamps or absolute paths inside the hashed mod folder.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ...clausewitz import parse
from ..contracts import ArenaError, Country
from ..diagnostics import write_json
from ..fingerprint import tree_hash
from ..layout import ArenaLayout
from .region import (
    DEFAULT_GAME,
    DEFAULT_REGION,
    SECTORS,
    Region,
    RegionSpec,
    asymmetry,
    extract_region,
    parse_strategic_region,
    province_raster,
    read_definitions,
    read_states,
    strategic_region_files,
)

MOD_DIR = "hoi4_arena"
MOD_NAME = "HOI4-AI Arena v0.1"
LOG_PREFIXES = ("ARENA_STARTUP", "ARENA_RESET", "ARENA_TICK", "ARENA_VP", "ARENA_OUTCOME", "ARENA_SELECT")
INFANTRY, ARMOR = "Arena Infantry", "Arena Armor"
OPPONENTS = ("default", "passive", "aggressive", "push_north", "push_center", "push_south")
HANDICAPS = ("even", "weak", "strong")  # applied to AI-controlled sides only
FALLBACK_TAGS = {"BLU": ("ABL", "ZBL", "XBL", "QBL"), "RED": ("ARD", "ZRD", "XRD", "QRD")}
COLORS = {"BLU": (40, 120, 255), "RED": (230, 40, 40)}
SCOPE_WORDS = {"ROOT", "FROM", "PREV", "THIS", "OWNER", "CONTROLLER", "CAPITAL"}
STRUCTURAL = {"limit", "else", "else_if"}  # part of if/every_*; not listed as effects or triggers
HISTORY_KEYWORDS = {"capital"}  # country-history keyword, not an effect
# Every vanilla identifier the generated script relies on; mod-validate looks each one up in common/.
VANILLA_IDS = ("infantry_weapons", "infantry_weapons1", "tech_trucks", "motorised_infantry", "gwtank",
               "basic_light_tank", "gwtank_chassis", "basic_light_tank_chassis", "light_tank_chassis_1",
               "light_tank_equipment_1", "infantry_equipment_1", "motorized_equipment_1", "tank_small_cannon",
               "tank_light_two_man_tank_turret", "tank_bogie_suspension", "tank_riveted_armor",
               "tank_gasoline_engine", "infantry", "light_armor", "motorized", "despotism", "neutrality")


@dataclass(frozen=True)
class Variant:
    """One curriculum scenario. ``blue``/``red`` are placement rules (see ``placements``)."""
    id: str
    stage: int  # 1: 2-4 infantry, 2: 12 infantry, 3: 12 mixed infantry/armour
    divisions: int
    armor: int
    blue: str
    red: str
    split: str  # train | validation | held_out, fixed here before any training
    theme: str


VARIANTS = (
    Variant("inf2_line", 1, 2, 0, "line", "line", "train", "first contact"),
    Variant("inf2_reserve", 1, 2, 0, "reserve", "reserve", "validation", "meeting engagement"),
    Variant("inf4_line", 1, 4, 0, "line", "line", "train", "thin front"),
    Variant("inf4_reserve", 1, 4, 0, "reserve", "reserve", "train", "meeting engagement"),
    Variant("inf4_mass_south", 1, 4, 0, "mass_south", "mass_south", "train", "river crossing"),
    Variant("inf4_mass_north", 1, 4, 0, "mass_north", "mass_north", "validation", "breakthrough"),
    Variant("inf4_mass_center", 1, 4, 0, "mass_center", "mass_center", "held_out", "river crossing"),
    Variant("inf12_line", 2, 12, 0, "line", "line", "train", "full front"),
    Variant("inf12_mass_center", 2, 12, 0, "mass_center", "mass_center", "train", "river crossing"),
    Variant("inf12_blue_defends", 2, 12, 0, "line", "mass_center", "train", "defence (mirror pair)"),
    Variant("inf12_red_defends", 2, 12, 0, "mass_center", "line", "train", "defence (mirror pair)"),
    Variant("inf12_reserve", 2, 12, 0, "reserve", "reserve", "validation", "meeting engagement"),
    Variant("inf12_mass_north", 2, 12, 0, "mass_north", "mass_north", "held_out", "breakthrough"),
    Variant("mix12_line", 3, 12, 3, "line", "line", "train", "combined arms"),
    Variant("mix12_reserve", 3, 12, 3, "reserve", "reserve", "validation", "combined arms"),
    Variant("mix12_mass_south", 3, 12, 3, "mass_south", "mass_south", "held_out", "armoured crossing"),
)
NOT_BUILT = ("retreat", "encirclement", "supply disruption")  # architecture themes without a variant yet


@dataclass(frozen=True)
class ModConfig:
    width: int = 1920
    height: int = 1080
    default_scenario: str = "inf2_line"
    default_opponent: str = "default"
    default_handicap: str = "even"
    manpower_k: int = 50  # free manpower pool restored at every reset, thousands
    infantry_stockpile: int = 2000
    armor_stockpile: int = 200  # light tanks and trucks, only in variants that field armour
    infrastructure: int = 3
    state_manpower: int = 100000
    horizon_days: int = 90
    start_date: str = "1936.1.1.12"
    world: str = "inert"  # vanilla-region build only, see quiet.py: "inert" freezes the world, "vanilla" keeps it

    def __post_init__(self) -> None:
        if self.default_scenario not in {v.id for v in VARIANTS}:
            raise ArenaError(f"unknown default scenario {self.default_scenario}")
        if self.world not in ("inert", "vanilla"):
            raise ArenaError("world must be 'inert' or 'vanilla'")
        if self.default_opponent not in OPPONENTS or self.default_handicap not in HANDICAPS:
            raise ArenaError("unknown default opponent or handicap")
        if min(self.width, self.height, self.manpower_k, self.horizon_days, self.state_manpower) <= 0:
            raise ArenaError("resolution, manpower and horizon must be positive")
        if not 0 <= self.infrastructure <= 5 or min(self.infantry_stockpile, self.armor_stockpile) < 0:
            raise ArenaError("infrastructure must be 0-5 and stockpiles non-negative")


@dataclass(frozen=True)
class StatePlan:
    id: int
    file_name: str
    name: str
    owner: str  # Country value
    label: str  # "capital" or the sector(s) it covers
    land: tuple[int, ...]
    other: tuple[int, ...]


# ---------------------------------------------------------------------------------------------
# Clausewitz emission: nested (key, value) pairs so braces always balance.

class Bare(tuple[Any, ...]):
    """A brace block of bare items: ``{ 1 2 3 }``."""


Pairs = list[tuple[str, Any]]


def q(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return repr(round(value, 4)).rstrip("0").rstrip(".") if value % 1 else str(int(value))
    return str(value)


def emit(pairs: Pairs, depth: int = 0) -> str:
    pad, out = "\t" * depth, []
    for key, value in pairs:
        if isinstance(value, Bare):
            out.append(f"{pad}{key} = {{ {' '.join(_scalar(v) for v in value)} }}\n")
        elif isinstance(value, list):
            out.append(f"{pad}{key} = {{\n{emit(value, depth + 1)}{pad}}}\n")
        else:
            out.append(f"{pad}{key} = {_scalar(value)}\n")
    return "".join(out)


def _if(limit: Pairs, then: Pairs, key: str = "if") -> tuple[str, Pairs]:
    return key, [("limit", limit), *then]


def _var_is(name: str, value: int) -> tuple[str, Pairs]:
    return "check_variable", [("var", name), ("value", value), ("compare", "equals")]


def _switch(name: str, cases: list[Pairs]) -> Pairs:
    """if / else_if chain over an integer variable."""
    return [_if([_var_is(name, index)], body, "if" if index == 0 else "else_if")
            for index, body in enumerate(cases)]


# ---------------------------------------------------------------------------------------------
# Planning: tags, states, placements.

def vanilla_tags(game: Path) -> set[str]:
    tags: set[str] = set()
    root = game / "common" / "country_tags"
    for path in sorted(root.glob("*.txt")) if root.is_dir() else ():
        text = re.sub(r"#[^\n]*", "", path.read_text(encoding="utf-8-sig", errors="replace"))
        tags |= set(re.findall(r"^\s*([A-Z][A-Z0-9]{2})\s*=", text, re.M))
    return tags


def choose_tags(game: Path) -> dict[str, str]:
    """Contract tag -> tag used in the mod. The contract's own tags are kept when vanilla leaves them free."""
    used = vanilla_tags(game)
    out: dict[str, str] = {}
    for country in Country:
        free = [tag for tag in (country.value, *FALLBACK_TAGS[country.value])
                if tag not in used and tag not in out.values()]
        if not free:
            raise ArenaError(f"no free country tag for {country.value}")
        out[country.value] = free[0]
    return out


def plan_states(region: Region) -> tuple[StatePlan, ...]:
    """Re-partition the region's provinces over the SAME vanilla state IDs: one state per side and
    sector, the capital province as its own state when IDs are left, further halves after that."""
    ids = [state.id for state in region.states]
    if len(ids) < 2:
        raise ArenaError("the arena needs at least two vanilla states (one per side)")
    share = {Country.BLUE: len(ids) - len(ids) // 2, Country.RED: len(ids) // 2}
    groups: list[tuple[Country, str, list[int]]] = []
    for country in Country:
        capital = region.layout().capital(country).id
        mine = [(sector, [p.id for p in region.side(country) if p.sector == sector]) for sector in SECTORS]
        mine = [(label, members) for label, members in mine if members]
        if share[country] > len(mine) and any(capital in m and len(m) > 1 for _, m in mine):
            mine = [(label, [p for p in members if p != capital]) for label, members in mine]
            mine.insert(0, ("capital", [capital]))
        while len(mine) > share[country]:  # too few state IDs: fold the smallest band into a neighbour
            index = min(range(len(mine)), key=lambda i: (len(mine[i][1]), i))
            other = index - 1 if index > 0 else 1
            low, high = sorted((index, other))
            mine[low] = (f"{mine[low][0]}+{mine[high][0]}", sorted(mine[low][1] + mine[high][1]))
            del mine[high]
        while len(mine) < share[country]:  # spare state IDs: halve the largest group
            index = max(range(len(mine)), key=lambda i: (len(mine[i][1]), -i))
            label, members = mine[index]
            if len(members) < 2:
                raise ArenaError("more vanilla states than provinces to fill them")
            mine[index:index + 1] = [(f"{label}/a", members[:len(members) // 2]),
                                     (f"{label}/b", members[len(members) // 2:])]
        groups += [(country, label, members) for label, members in mine]
    vanilla = {state.id: set(state.provinces) for state in region.states}
    free: set[int] = set(ids)
    chosen: dict[int, int] = {}  # group index -> vanilla state ID
    while len(chosen) < len(groups):  # keep as much of each vanilla state as possible; ties by ID
        _, neg_state, neg_group = max((len(vanilla[s] & set(groups[g][2])), -s, -g)
                                      for s in sorted(free) for g in range(len(groups)) if g not in chosen)
        chosen[-neg_group] = -neg_state
        free.discard(-neg_state)
    by_province = {p: g for g, (_, _, members) in enumerate(groups) for p in members}
    others: dict[int, list[int]] = {g: [] for g in range(len(groups))}
    for province, state_id, adjacent in region.other_provinces:
        votes = [by_province[a] for a in adjacent]
        home = max(sorted(set(votes)), key=votes.count) if votes else next(
            g for g, s in chosen.items() if s == state_id)
        others[home].append(province)
    names = {state.id: state for state in region.states}
    # Own name keys: the vanilla names would be wrong after the re-partition (cosmetic only).
    return tuple(sorted((StatePlan(chosen[g], names[chosen[g]].file_name,
                                   "ARENA_STATE_" + re.sub(r"\W+", "_", f"{country.value}_{label}"), country.value,
                                   label, tuple(sorted(members)), tuple(sorted(others[g])))
                         for g, (country, label, members) in enumerate(groups)), key=lambda plan: plan.id))


def handicap_count(variant: Variant, level: str) -> tuple[int, int]:
    """(divisions, of which armour) for a handicap level."""
    total = {"even": variant.divisions, "weak": max(1, variant.divisions // 2),
             "strong": variant.divisions + (variant.divisions + 1) // 2}[level]
    return total, (variant.armor * total + variant.divisions // 2) // variant.divisions


def placements(region: Region, variant: Variant, country: Country, level: str = "even") -> tuple[tuple[int, str], ...]:
    """Deterministic (province, template) start list. Rules: ``line`` deals divisions round-robin to
    the front provinces of center, north, south; ``mass_<sector>`` stacks them on that sector's front;
    ``reserve`` starts them at the capital and its own neighbours. The last ones are the armour."""
    rule = variant.blue if country is Country.BLUE else variant.red
    total, armor = handicap_count(variant, level)
    # Front provinces that carry victory points come first. Third live load: with ID order, inf2_line left
    # 9690 and 6700 empty right next to RED's two start provinces and both were lost on day 1.
    prized = {p for p, _ in region.spec.victory_points}
    ordered = sorted(region.front(country), key=lambda p: (p.id not in prized, p.id))
    front = {sector: [p.id for p in ordered if p.sector == sector] for sector in SECTORS}
    everything = [p.id for p in ordered]
    own = {p.id for p in region.side(country)}
    capital = region.layout().capital(country).id
    if rule == "reserve":
        pools = [[capital, *[n for n in region.neighbors[capital] if n in own]]]
    elif rule.startswith("mass_") and rule[5:] in SECTORS:
        pools = [front[rule[5:]] or everything]
    elif rule == "line":
        pools = [front[sector] or everything for sector in ("center", "north", "south")]
    else:
        raise ArenaError(f"unknown placement rule {rule}")
    cursor = [0] * len(pools)
    spots = []
    for index in range(total):
        pool = pools[index % len(pools)]
        spots.append(pool[cursor[index % len(pools)] % len(pool)])
        cursor[index % len(pools)] += 1
    return tuple((spot, ARMOR if index >= total - armor else INFANTRY) for index, spot in enumerate(spots))


# ---------------------------------------------------------------------------------------------
# File contents.

def _descriptor(path: str | None = None) -> str:
    text = f'version="0.1"\ntags={{\n\t"Gameplay"\n}}\nname="{MOD_NAME}"\nsupported_version="1.19.*"\n'
    return text + (f'path="{path}"\n' if path else "")


def _state_file(plan: StatePlan, region: Region, tags: dict[str, str], config: ModConfig) -> str:
    points = dict(region.spec.victory_points)
    history: Pairs = [("owner", tags[plan.owner]), ("add_core_of", tags[plan.owner])]
    history += [("victory_points", Bare((p, points[p]))) for p in plan.land if p in points]
    history.append(("buildings", [("infrastructure", config.infrastructure)]))
    return emit([("state", [("id", plan.id), ("name", q(plan.name)), ("manpower", config.state_manpower),
                            ("state_category", "town"), ("history", history),
                            ("provinces", Bare(sorted(plan.land + plan.other))), ("local_supplies", 0.0)])])


def _templates() -> str:
    def grid(rows: list[tuple[str, int, int]]) -> Pairs:
        return [(unit, [("x", x), ("y", y)]) for unit, x, y in rows]
    infantry = grid([("infantry", x, y) for x in range(3) for y in range(3)])
    armor = grid([("light_armor", 0, y) for y in range(3)] + [("motorized", 1, y) for y in range(2)])
    return emit([("division_template", [("name", q(INFANTRY)), ("is_locked", True), ("regiments", infantry)]),
                 ("division_template", [("name", q(ARMOR)), ("is_locked", True), ("regiments", armor)])])


def _units_file(tag: str, spots: tuple[tuple[int, str], ...]) -> str:
    counters = {INFANTRY: 0, ARMOR: 0}
    divisions: Pairs = []
    for province, template in spots:
        counters[template] += 1
        label = "Infantry" if template == INFANTRY else "Armor"
        divisions.append(("division", [("name", q(f"{tag} {label} {counters[template]:02d}")),
                                       ("location", province), ("division_template", q(template)),
                                       ("start_experience_factor", 0.0), ("start_equipment_factor", 1.0)]))
    return emit([("units", divisions)])


def _country_history(tag: str, capital_state: int, leader: str) -> str:
    tank: Pairs = [("name", q("Arena Light Tank")), ("type", "light_tank_chassis_1"), ("parent_version", 0),
                   ("modules", [("main_armament_slot", "tank_small_cannon"),
                                ("turret_type_slot", "tank_light_two_man_tank_turret"),
                                ("suspension_type_slot", "tank_bogie_suspension"),
                                ("armor_type_slot", "tank_riveted_armor"),
                                ("engine_type_slot", "tank_gasoline_engine")])]
    return emit([
        ("capital", capital_state), ("set_oob", q(f"ARENA_{tag}_templates")), ("set_research_slots", 0),
        ("set_stability", 0.7), ("set_war_support", 0.7),
        ("set_technology", [("infantry_weapons", 1), ("infantry_weapons1", 1), ("tech_trucks", 1),
                            ("motorised_infantry", 1)]),
        _if([("has_dlc", q("No Step Back"))],
            [("set_technology", [("gwtank_chassis", 1), ("basic_light_tank_chassis", 1)]),
             ("create_equipment_variant", tank)]),
        _if([("not", [("has_dlc", q("No Step Back"))])],
            [("set_technology", [("gwtank", 1), ("basic_light_tank", 1)])]),
        ("set_politics", [("ruling_party", "neutrality"), ("elections_allowed", False)]),
        ("set_popularities", [("neutrality", 100)]),
        ("recruit_character", leader), ("add_ideas", "arena_rules"),
    ])


def _spawn_effect(tag: str, index: int, variant: Variant, config: ModConfig) -> Pairs:
    """Country scope: load this variant's order of battle, cap templates, restore the stockpile."""
    def load(level: str) -> Pairs:
        total, armor = handicap_count(variant, level)
        body: Pairs = [("load_oob", q(f"ARENA_{index:02d}_{tag}_{level}")),
                       ("set_division_template_cap", [("division_template", q(INFANTRY)),
                                                      ("division_cap", max(1, total - armor))]),
                       ("set_division_template_cap", [("division_template", q(ARMOR)),
                                                      ("division_cap", max(1, armor))])]
        return body
    body: Pairs = [
        _if([("is_ai", True), _var_is("global.arena_handicap", 1)], load("weak")),
        _if([("is_ai", True), _var_is("global.arena_handicap", 2)], load("strong"), "else_if"),
        ("else", load("even")),
    ]
    if variant.armor:
        amount = config.armor_stockpile
        body += [("add_equipment_to_stockpile", [("type", "motorized_equipment_1"), ("amount", amount)]),
                 _if([("has_dlc", q("No Step Back"))],
                     [("add_equipment_to_stockpile", [("type", "light_tank_chassis_1"), ("amount", amount)])]),
                 _if([("not", [("has_dlc", q("No Step Back"))])],
                     [("add_equipment_to_stockpile", [("type", "light_tank_equipment_1"), ("amount", amount)])])]
    return body


def _scripted_effects(region: Region, plans: tuple[StatePlan, ...], tags: dict[str, str],
                      variants: tuple[Variant, ...], config: ModConfig) -> str:
    blue, red = tags["BLU"], tags["RED"]
    points = sorted(region.spec.victory_points)
    capital = {c.value: region.layout().capital(c).id for c in Country}
    count: Pairs = [("set_variable", [("global.arena_vp_blu", 0)]), ("set_variable", [("global.arena_vp_red", 0)])]
    for province, value in points:
        count.append(("set_variable", [(f"global.arena_holder_{province}", 0)]))
        for tag, name, mark in ((blue, "global.arena_vp_blu", 1), (red, "global.arena_vp_red", 2)):
            count.append(_if([(tag, [("controls_province", province)])],
                             [("add_to_variable", [(name, value)]),
                              ("set_variable", [(f"global.arena_holder_{province}", mark)])]))
    holders = " ".join(f"{province}=[?global.arena_holder_{province}]" for province, _ in points)

    def outcome(winner: str, reason: str) -> Pairs:
        return [("set_global_flag", "arena_episode_over"),
                ("log", q(f"ARENA_OUTCOME episode=[?global.arena_episode] winner={winner} reason={reason} "
                          "day=[?global.arena_day] vp_blu=[?global.arena_vp_blu] vp_red=[?global.arena_vp_red]"))]

    def more(left: str, right: str) -> tuple[str, Pairs]:
        return "check_variable", [("var", left), ("value", right), ("compare", "greater_than")]

    blue_fell: Pairs = [(red, [("controls_province", capital["BLU"])])]
    red_fell: Pairs = [(blue, [("controls_province", capital["RED"])])]
    timeout = ("check_variable", [("var", "global.arena_day"), ("value", config.horizon_days),
                                  ("compare", "greater_than_or_equals")])
    daily: Pairs = [_if([("not", [("has_global_flag", "arena_episode_over")]), ("has_global_flag", "arena_started")], [
        ("set_variable", [("global.arena_day", "global.num_days")]),
        ("subtract_from_variable", [("global.arena_day", "global.arena_start_day")]),
        ("arena_count_vp", True),
        ("log", q("ARENA_TICK episode=[?global.arena_episode] day=[?global.arena_day] "
                  "vp_blu=[?global.arena_vp_blu] vp_red=[?global.arena_vp_red] "
                  f"div_blu=[?{blue}.num_divisions] div_red=[?{red}.num_divisions]")),
        ("log", q(f"ARENA_VP episode=[?global.arena_episode] day=[?global.arena_day] {holders}")),
        _if(blue_fell + red_fell, outcome("DRAW", "capital_both")),
        _if(blue_fell, outcome("RED", "capital"), "else_if"),
        _if(red_fell, outcome("BLU", "capital"), "else_if"),
        _if([timeout, more("global.arena_vp_blu", "global.arena_vp_red")], outcome("BLU", "timeout_vp"), "else_if"),
        _if([timeout, more("global.arena_vp_red", "global.arena_vp_blu")], outcome("RED", "timeout_vp"), "else_if"),
        _if([timeout], outcome("DRAW", "timeout_vp"), "else_if"),
    ])]
    capitulated: Pairs = [_if([("not", [("has_global_flag", "arena_episode_over")])], [
        _if([("tag", blue)], outcome("RED", "capitulation")),
        _if([("tag", red)], outcome("BLU", "capitulation"), "else_if")])]

    restore: Pairs = []
    for contract in ("BLU", "RED"):
        mine = [plan for plan in plans if plan.owner == contract]
        restore.append((tags[contract], [("transfer_state", plan.id) for plan in mine] +
                        [("set_province_controller", p) for plan in mine for p in plan.land]))
    country_reset: Pairs = [
        ("set_stability", 0.7), ("set_war_support", 0.7),
        ("set_temp_variable", [("arena_delta", config.manpower_k)]),
        ("subtract_from_temp_variable", [("arena_delta", "manpower_k")]),
        ("multiply_temp_variable", [("arena_delta", 1000)]),
        ("add_manpower", "arena_delta"),
        ("add_equipment_to_stockpile", [("type", "infantry_equipment_1"), ("amount", -1000000)]),
        ("add_equipment_to_stockpile", [("type", "infantry_equipment_1"), ("amount", config.infantry_stockpile)]),
        ("set_fuel_ratio", 1.0), ("country_lock_all_division_template", True),
    ]
    reset: Pairs = [
        ("add_to_variable", [("global.arena_episode", 1)]), ("clr_global_flag", "arena_episode_over"),
        ("set_variable", [("global.arena_start_day", "global.num_days")]), ("set_variable", [("global.arena_day", 0)]),
        (blue, [("delete_unit", [("disband", False)])]), (red, [("delete_unit", [("disband", False)])]),
        *[(str(plan.id), [("every_state_division", [("destroy_unit", True)])]) for plan in plans],
        *restore,
        (blue, [("arena_reset_country", True), ("arena_spawn_units", True)]),
        (red, [("arena_reset_country", True), ("arena_spawn_units", True)]),
        ("arena_count_vp", True),
        ("log", q("ARENA_RESET episode=[?global.arena_episode] variant=[?global.arena_variant] "
                  "opponent=[?global.arena_opponent] handicap=[?global.arena_handicap] "
                  "vp_blu=[?global.arena_vp_blu] vp_red=[?global.arena_vp_red]")),
    ]
    ids = [v.id for v in variants]
    startup: Pairs = [("log", q("ARENA_STARTUP effect entered")),
                      _if([("not", [("has_global_flag", "arena_started")])], [
        ("set_global_flag", "arena_started"), ("set_variable", [("global.arena_episode", 0)]),
        ("set_variable", [("global.arena_variant", ids.index(config.default_scenario))]),
        ("set_variable", [("global.arena_handicap", HANDICAPS.index(config.default_handicap))]),
        ("set_variable", [("global.arena_opponent", OPPONENTS.index(config.default_opponent))]),
        ("arena_apply_opponent", True),
        _if([(blue, [("not", [("has_war_with", red)])])],
            [(blue, [("declare_war_on", [("target", red), ("type", "annex_everything")])])]),
        ("log", q(f"ARENA_STARTUP region={region.spec.scenario_id} blue={blue} red={red} "
                  f"horizon_days={config.horizon_days}")),
        ("arena_reset_episode", True),
    ])]
    flags = [f"arena_ai_{name}" for name in OPPONENTS]
    opponent: Pairs = []
    for tag in (blue, red):
        opponent.append((tag, [("clr_country_flag", flag) for flag in flags] + _switch(
            "global.arena_opponent", [[("set_country_flag", flag)] for flag in flags])))
    effects: Pairs = [("arena_count_vp", count), ("arena_daily", daily), ("arena_on_capitulation", capitulated),
                      ("arena_reset_country", country_reset), ("arena_reset_episode", reset),
                      ("arena_startup", startup), ("arena_apply_opponent", opponent)]
    dispatch = []
    for index, variant in enumerate(variants):
        effects.append((f"arena_spawn_{index:02d}", [
            _if([("tag", tag)], [*_spawn_effect(tag, index, variant, config),
                                 ("log", q(f"ARENA_SELECT applied scenario={variant.id} index={index} side={side}"))])
            for side, tag in (("BLU", blue), ("RED", red))]))
        dispatch.append([(f"arena_spawn_{index:02d}", True)])
    effects.append(("arena_spawn_units", _switch("global.arena_variant", dispatch)))
    return emit(effects)


def _decisions(tags: dict[str, str], variants: tuple[Variant, ...]) -> tuple[str, str, dict[str, str]]:
    """(categories file, decisions file, decision id -> meaning). Human-only, free, repeatable."""
    ours: Pairs = [("or", [("original_tag", tags["BLU"]), ("original_tag", tags["RED"])])]

    def decision(name: str, effect: Pairs, selected: tuple[str, int] | None = None) -> tuple[str, Pairs]:
        available: Pairs = [("is_ai", False)]
        if selected:  # greyed out while it is the current choice, so the screen shows the selection
            available.append(("not", [_var_is(*selected)]))
        return name, [("allowed", [("always", True)]), ("visible", [("is_ai", False)]), ("available", available),
                      ("cost", 0), ("fire_only_once", False), ("ai_will_do", [("factor", 0)]),
                      ("complete_effect", effect)]

    meaning = {"arena_reset": "restore the episode: wipe divisions, restore control, respawn, log ARENA_RESET"}
    control = [decision("arena_reset", [("arena_reset_episode", True)])]
    scenario, opponent, handicap = [], [], []
    for index, variant in enumerate(variants):
        name = f"arena_scenario_{index:02d}_{variant.id}"
        meaning[name] = f"select scenario {variant.id} for the NEXT reset"
        scenario.append(decision(name, [("set_variable", [("global.arena_variant", index)]),
                                        ("log", q(f"ARENA_SELECT scenario={variant.id} index={index}"))],
                                 ("global.arena_variant", index)))
    for index, profile in enumerate(OPPONENTS):
        name = f"arena_opponent_{index}_{profile}"
        meaning[name] = f"AI-controlled sides use the {profile} ai_strategy profile (immediately)"
        opponent.append(decision(name, [("set_variable", [("global.arena_opponent", index)]),
                                        ("arena_apply_opponent", True),
                                        ("log", q(f"ARENA_SELECT opponent={profile} index={index}"))],
                                 ("global.arena_opponent", index)))
    for index, level in enumerate(HANDICAPS):
        name = f"arena_handicap_{index}_{level}"
        meaning[name] = f"AI-controlled sides get the {level} division count at the NEXT reset"
        handicap.append(decision(name, [("set_variable", [("global.arena_handicap", index)]),
                                        ("log", q(f"ARENA_SELECT handicap={level} index={index}"))],
                                 ("global.arena_handicap", index)))
    groups = [("arena_control", control), ("arena_scenario", scenario), ("arena_opponent", opponent),
              ("arena_handicap", handicap)]
    categories = emit([(name, [("allowed", ours)]) for name, _ in groups])
    return categories, emit([(name, list(body)) for name, body in groups]), meaning


def _ai_strategies(plans: tuple[StatePlan, ...], tags: dict[str, str]) -> str:
    """Opponent panel: flags pick one block. ``default`` adds nothing (stock AI)."""
    out: Pairs = []
    for contract in ("BLU", "RED"):
        me, enemy = tags[contract], tags[Country(contract).opponent.value]

        def strategy(profile: str, entries: list[Pairs], me: str = me) -> tuple[str, Pairs]:
            return f"arena_{profile}_{me}", [
                ("allowed", [("original_tag", me)]), ("enable", [("has_country_flag", f"arena_ai_{profile}")]),
                ("abort_when_not_enabled", True), *[("ai_strategy", entry) for entry in entries]]

        def control(priority: int, execute: bool, style: str, states: tuple[int, ...] = (), enemy: str = enemy) -> Pairs:
            target: Pairs = [("state", s) for s in states] if states else [("tag", enemy)]
            return [("type", "front_control"), *target, ("priority", priority), ("ordertype", "front"),
                    ("execution_type", style), ("execute_order", execute), ("manual_attack", execute)]

        out.append(strategy("passive", [control(100, False, "careful")]))
        out.append(strategy("aggressive", [control(100, True, "rush")]))
        for sector in SECTORS:
            states = tuple(plan.id for plan in plans if plan.owner != contract and sector in plan.label)
            entries = [control(50, False, "careful")]
            if states:
                entries += [control(100, True, "rush", states),
                            [("type", "front_unit_request"), *[("state", s) for s in states], ("value", 200)]]
            out.append(strategy(f"push_{sector}", entries))
    return emit(out)


def _localisation(region: Region, tags: dict[str, str], plans: tuple[StatePlan, ...],
                  meaning: dict[str, str]) -> str:
    rows = {"ARENA_BOOKMARK_NAME": "HOI4-AI Arena", "ARENA_BOOKMARK_DESC": "Two countries, one front, ninety days.",
            "arena_rules": "Arena rules", "arena_rules_desc": "No capitulation, no army experience.",
            "arena_focus_tree": "Arena", "arena_no_focus": "No focus", "arena_no_focus_desc": "The arena has no focuses.",
            "arena_control": "ARENA 1 CONTROL", "arena_scenario": "ARENA 2 SCENARIO",
            "arena_opponent": "ARENA 3 OPPONENT", "arena_handicap": "ARENA 4 HANDICAP"}
    for contract, label in (("BLU", "Blue"), ("RED", "Red")):
        tag = tags[contract]
        rows[f"{tag}_ARENA_DESC"] = f"{label} side of the arena."
        rows[f"{tag}_arena_leader"] = f"{label} Commander"
        for suffix in ("", "_fascism", "_democratic", "_neutrality", "_communism"):
            rows[f"{tag}{suffix}"], rows[f"{tag}{suffix}_DEF"], rows[f"{tag}{suffix}_ADJ"] = label, label, label
    for name in meaning:
        rows[name] = name.removeprefix("arena_").replace("_", " ").upper()
        rows[f"{name}_desc"] = meaning[name]
    for plan in plans:
        rows[plan.name] = f"{'Blue' if plan.owner == 'BLU' else 'Red'} {plan.label}"
    vanilla = {p for state in region.states for p, _ in state.victory_points}
    capitals = {region.spec.blue_capital: "Blue Capital", region.spec.red_capital: "Red Capital"}
    for province, _ in region.spec.victory_points:
        if province not in vanilla:  # vanilla names stay; overriding them would need replace/
            rows[f"VICTORY_POINTS_{province}"] = capitals.get(province, f"Arena {province}")
    return "\ufeffl_english:\n" + "".join(f' {key}:0 "{value}"\n' for key, value in sorted(rows.items()))


def strategic_region_overrides(game: Path, plans: tuple[StatePlan, ...]) -> dict[str, str]:
    """file name -> new text. The engine refuses a map in which a state spans strategic regions
    (MAP_ERROR seen on the first live load), so every arena province moves into ONE vanilla region:
    the one that already holds most of them. Only the ``provinces`` block of each touched file is
    rewritten; weather and naval terrain stay as they are."""
    files = strategic_region_files(game)
    arena = {p for plan in plans for p in plan.land + plan.other}
    inside = {name: len(arena & set(provinces)) for name, (_, provinces) in files.items()}
    touched = sorted(name for name, count in inside.items() if count)
    if len(touched) < 2:
        return {}
    target = max(touched, key=lambda name: (inside[name], -files[name][0]))
    out = {}
    for name in touched:
        keep = [p for p in files[name][1] if p not in arena] + (sorted(arena) if name == target else [])
        if not keep:
            raise ArenaError(f"strategic region file {name} would be left without provinces")
        raw = (game / "map" / "strategicregions" / name).read_bytes().decode("utf-8-sig", errors="replace")
        block = "provinces={\n\t\t" + " ".join(str(p) for p in keep) + "\n\t}"
        raw = raw.replace("\r\n", "\n")
        found = re.search(r"\bprovinces\s*=\s*\{[^}]*\}", raw)
        if found is None:
            raise ArenaError(f"cannot find the provinces block of {name}")
        out[name] = raw[:found.start()] + block + raw[found.end():]
    return out


def _building_pixel(row: list[str], height: int) -> tuple[int, int]:
    """buildings.txt is ``state;type;x;y;z;rotation;sea``: x is the bitmap column, z counts from the
    BOTTOM row. Measured on this install: all 326 vanilla rows of the default region's states land in
    a province of their own state with ``row = height - 1 - floor(z)`` (325 of 326 with ``height - z``)."""
    return int(float(row[2])), height - 1 - int(float(row[4]))


def buildings_override(game: Path, plans: tuple[StatePlan, ...]) -> str | None:
    """Vanilla ``map/buildings.txt`` with the state ID of every row inside an arena province rewritten to
    that province's NEW state, plus one row per (new state, building type) that would otherwise be missing."""
    path = game / "map" / "buildings.txt"
    if not path.is_file():
        return None
    state_of = {p: plan.id for plan in plans for p in plan.land + plan.other}
    ids, (x0, y0), height = province_raster(game / "map" / "provinces.bmp", read_definitions(game), set(state_of))
    lines: list[str] = []
    present: set[tuple[int, str]] = set()
    heights: list[float] = []
    kinds: set[str] = set()
    vanilla = path.read_bytes().decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    for line in vanilla.split("\n"):
        row = line.split(";")
        if len(row) >= 5 and row[0].strip().isdigit():
            try:
                x, y = _building_pixel(row, height)
            except ValueError:
                x, y = -1, -1
            if 0 <= x - x0 < ids.shape[1] and 0 <= y - y0 < ids.shape[0] and int(ids[y - y0, x - x0]) in state_of:
                row[0] = str(state_of[int(ids[y - y0, x - x0])])
                line = ";".join(row)
                present.add((int(row[0]), row[1]))
                heights.append(float(row[3]))
                kinds.add(row[1])
        lines.append(line)
    ending = ""
    while lines and not lines[-1].strip():  # re-attached below, so the file ends exactly like vanilla's
        ending = "\n" + lines.pop() + ending
    level = sum(heights) / len(heights) if heights else 10.0
    for plan in plans:  # e.g. the one-province capital states have no factory/air base positions of their own
        ys, xs = (ids == plan.land[0]).nonzero()
        px, py = min(zip(xs.tolist(), ys.tolist(), strict=True),
                     key=lambda xy: ((xy[0] - xs.mean()) ** 2 + (xy[1] - ys.mean()) ** 2, xy))
        for kind in sorted(kinds):
            if (plan.id, kind) not in present:
                lines.append(f"{plan.id};{kind};{px + x0:.2f};{level:.2f};{height - 1 - (py + y0):.2f};0.00;0")
    # The engine reports "invalid arguments count" for an empty last line (seen live): vanilla has no final
    # newline, so neither do we unless vanilla does.
    return "\n".join(lines) + ending


def _flag(path: Path, color: tuple[int, int, int], size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, (*color, 255)).save(path, format="TGA")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))  # bytes: no platform newline translation


# ---------------------------------------------------------------------------------------------
# Build.

def build_mod(game: Path, output: Path, spec: RegionSpec = DEFAULT_REGION, config: ModConfig | None = None,
              force: bool = False) -> dict[str, Any]:
    """Write the mod, launch profile, layout and manifest into ``output``. Never launches anything."""
    config = config or ModConfig()
    output = output.resolve()
    if output.is_relative_to(game.resolve()):
        raise ArenaError("refusing to generate inside the game installation")
    mod = output / "mod" / MOD_DIR
    if mod.exists():
        marker = mod / "descriptor.mod"
        if not force or not marker.is_file() or MOD_NAME not in marker.read_text(encoding="utf-8"):
            raise ArenaError(f"{mod} exists; pass force only to replace a previously generated arena mod")
        shutil.rmtree(mod)
    region = extract_region(game, spec)
    layout = region.layout()
    tags = choose_tags(game)
    plans = plan_states(region)
    variants = VARIANTS
    _write(mod / "descriptor.mod", _descriptor())
    vanilla_colors = game / "common" / "countries" / "colors.txt"
    # colors.txt is one file: a mod copy must carry every vanilla entry
    base = (vanilla_colors.read_bytes().decode("utf-8-sig", errors="replace").replace("\r\n", "\n").rstrip("\n")
            + "\n\n") if vanilla_colors.is_file() else None
    from . import quiet  # imported here: quiet.py uses this module's emitters
    if base is not None and config.world == "inert":
        base += quiet.colors_entry(quiet.choose_neutral_tag(game, set(tags.values())))
    meaning = write_content(mod, region, plans, tags, variants, config, base)
    quiet_report = quiet.write_quiet(mod, game, region, tags, config.world)
    for name, text in strategic_region_overrides(game, plans).items():
        _write(mod / "map" / "strategicregions" / name, text)
    buildings = buildings_override(game, plans)
    if buildings is not None:
        _write(mod / "map" / "buildings.txt", buildings)
    return _finish(output, mod, region, layout, spec, tags, plans, variants, config, meaning, quiet_report)


def write_content(mod: Path, region: Region, plans: tuple[StatePlan, ...], tags: dict[str, str],
                  variants: tuple[Variant, ...], config: ModConfig, base_colors: str | None) -> dict[str, str]:
    """Everything that does not depend on WHICH map the arena sits on: countries, flags, characters, state
    history, orders of battle, rules idea, focus stub, scripted effects, on_actions, decisions, AI profiles,
    bookmark and localisation. ``region`` may describe vanilla states or a generated map (``custommap.py``).
    ``base_colors`` is the text the two colour entries are appended to (None: no ``colors.txt``).
    Returns decision id -> meaning."""
    layout = region.layout()
    capital_state = {c.value: next(plan.id for plan in plans if layout.capital(c).id in plan.land) for c in Country}
    names = {"BLU": "Arena Blue", "RED": "Arena Red"}
    _write(mod / "common" / "country_tags" / "zz_arena_countries.txt",
           "".join(f'{tags[c]} = "countries/{names[c]}.txt"\n' for c in ("BLU", "RED")))
    colors = ""
    for contract in ("BLU", "RED"):
        r, g, b = COLORS[contract]
        _write(mod / "common" / "countries" / f"{names[contract]}.txt",
               "graphical_culture = western_european_gfx\ngraphical_culture_2d = western_european_2d\n"
               f"color = {{ {r} {g} {b} }}\n")
        colors += f"{tags[contract]} = {{\n\tcolor = rgb {{ {r} {g} {b} }}\n\tcolor_ui = rgb {{ {r} {g} {b} }}\n}}\n"
        for folder, size in (("", (82, 52)), ("medium", (41, 26)), ("small", (10, 7))):
            _flag(mod / "gfx" / "flags" / folder / f"{tags[contract]}.tga", COLORS[contract], size)
    if base_colors is not None:
        _write(mod / "common" / "countries" / "colors.txt", base_colors + colors)

    characters: Pairs = []
    for contract in ("BLU", "RED"):
        tag = tags[contract]
        characters.append((f"{tag}_arena_leader", [("name", f"{tag}_arena_leader"), ("country_leader", [
            ("ideology", "despotism"), ("expire", q("1999.1.1.1")), ("id", -1)])]))
        _write(mod / "history" / "countries" / f"{tag} - {names[contract]}.txt",
               _country_history(tag, capital_state[contract], f"{tag}_arena_leader"))
        _write(mod / "history" / "units" / f"ARENA_{tag}_templates.txt", _templates())
        for index, variant in enumerate(variants):
            for level in HANDICAPS:
                _write(mod / "history" / "units" / f"ARENA_{index:02d}_{tag}_{level}.txt",
                       _units_file(tag, placements(region, variant, Country(contract), level)))
    _write(mod / "common" / "characters" / "arena_characters.txt", emit([("characters", characters)]))
    for plan in plans:
        _write(mod / "history" / "states" / plan.file_name, _state_file(plan, region, tags, config))

    _write(mod / "common" / "ideas" / "arena_ideas.txt", emit([("ideas", [("country", [("arena_rules", [
        ("picture", "generic_intel_bonus"), ("removal_cost", -1), ("allowed_civil_war", [("always", True)]),
        ("modifier", [("surrender_limit", 1.0), ("experience_gain_army_factor", -1.0),
                      ("political_power_factor", -1.0)])])])])]))
    ours: Pairs = [("or", [("tag", tags["BLU"]), ("tag", tags["RED"])])]
    _write(mod / "common" / "national_focus" / "arena_focus.txt", emit([("focus_tree", [
        ("id", "arena_focus_tree"), ("country", [("factor", 0), ("modifier", [("add", 100), *ours])]),
        ("default", False),
        ("focus", [("id", "arena_no_focus"), ("icon", "GFX_goal_unknown"), ("x", 0), ("y", 0), ("cost", 10),
                   ("available", [("always", False)]), ("ai_will_do", [("factor", 0)]),
                   ("completion_reward", [("add_political_power", 1)])])])]))
    # Without a names entry the engine cannot name generated unit leaders ("Failed to generate a name", seen live).
    people = [("male", [("names", Bare(q(n) for n in ("Adam", "Bela", "Imre", "Janos", "Karl", "Laszlo", "Pal")))]),
              ("female", [("names", Bare(q(n) for n in ("Anna", "Eva", "Ilona", "Judit", "Klara", "Maria")))]),
              ("surnames", Bare(q(n) for n in ("Arany", "Balogh", "Feher", "Kovacs", "Molnar", "Nagy", "Szabo",
                                               "Toth", "Varga", "Voros"))),
              ("callsigns", Bare(q(n) for n in ("Anvil", "Falcon", "Hammer", "Lance", "Spur")))]
    _write(mod / "common" / "names" / "zz_arena_names.txt", emit([(tags[c], people) for c in ("BLU", "RED")]))
    _write(mod / "common" / "scripted_effects" / "arena_effects.txt",
           _scripted_effects(region, plans, tags, variants, config))
    _write(mod / "common" / "on_actions" / "arena_on_actions.txt", emit([("on_actions", [
        # on_startup runs in GLOBAL scope, where a scripted effect is an "Invalid Scope" (seen live):
        # every entry point enters a country scope first. The bare log answers "does log reach game.log".
        ("on_startup", [("effect", [("log", q("ARENA_STARTUP on_startup fired")),
                                    (tags["BLU"], [("arena_startup", True)])])]),
        (f"on_daily_{tags['BLU']}", [("effect", [(tags["BLU"], [("arena_daily", True)])])]),
        # ROOT is the capitulated country, FROM the winner (vanilla 00_on_actions.txt comment).
        ("on_capitulation_immediate", [("effect", [("ROOT", [("arena_on_capitulation", True)])])])])]))
    categories, decisions, meaning = _decisions(tags, variants)
    _write(mod / "common" / "decisions" / "categories" / "arena_categories.txt", categories)
    _write(mod / "common" / "decisions" / "arena_decisions.txt", decisions)
    _write(mod / "common" / "ai_strategy" / "arena_ai.txt", _ai_strategies(plans, tags))
    entries: Pairs = [("name", q("ARENA_BOOKMARK_NAME")), ("desc", q("ARENA_BOOKMARK_DESC")),
                      ("date", config.start_date), ("picture", q("GFX_select_date_1936")),
                      ("default_country", q(tags["BLU"])), ("default", True)]
    entries += [(q(tags[c]), [("history", q(f"{tags[c]}_ARENA_DESC")), ("ideology", "neutrality"),
                              ("ideas", []), ("focuses", [])]) for c in ("BLU", "RED")]
    entries.append(("effect", [("randomize_weather", 22345)]))
    _write(mod / "common" / "bookmarks" / "arena_bookmark.txt", emit([("bookmarks", [("bookmark", entries)])]))
    _write(mod / "localisation" / "english" / "arena_l_english.yml", _localisation(region, tags, plans, meaning))
    return meaning


def exposed_vps(region: Region, variant: Variant) -> dict[str, list[int]]:
    """side -> its victory-point provinces that start EMPTY and adjacent to an enemy start province: the enemy
    can walk in within about a day. Unavoidable when a side has fewer divisions than front VPs; recorded so
    nobody mistakes the resulting day-1 swing for a script bug again."""
    out = {}
    points = {p for p, _ in region.spec.victory_points}
    for country in Country:
        mine = {p for p, _ in placements(region, variant, country)}
        enemy = {p for p, _ in placements(region, variant, country.opponent)}
        own = {p.id for p in region.side(country)}
        out[country.value] = sorted(p for p in points & own if p not in mine and enemy & set(region.neighbors[p]))
    return out


LIVE_EVIDENCE = {
    "source": "integrator's live loads on v1.19.3 (mods 81d57adc and 9f83b59f); not re-verified for later hashes",
    "verified": [
        "map loads: 0 MAP_ERROR, no buildings.txt errors, error.log has no line caused by the mod",
        "-start_tag=BLU -start_speed=1 starts a single-player game as BLU without any menu click",
        "on_startup fires; the log effect writes to logs/game.log and [?global.var] placeholders expand",
        "ARENA_STARTUP, ARENA_SELECT applied (both sides), ARENA_RESET episode=1 (21:21) and daily ARENA_TICK appear",
        "load_oob spawns the divisions (own and enemy counters visible on the map)",
        "mod 485c0190: two full 90-day episodes played as BLU; 'ARENA_OUTCOME episode=1 winner=RED "
        "reason=timeout_vp day=90 vp_blu=18 vp_red=24' logged correctly: outcome line and timeout rule verified",
        "-start_speed=1 starts UNPAUSED at speed 1; numpad + raises the speed",
        "at speed 4 the vanilla map runs about 2.4 game days per second (before the world freeze)",
    ],
    "open": [
        "quiet layers (hidden news events, popup settings, inert world) were generated after 485c0190 and have "
        "not been loaded yet; on 485c0190 a World News window and war toasts blinded ~30% of frames",
        "inf2_line on 9f83b59f: vp went 21:21 -> 15:27 on day 1. The script audit found no wrong spawn, control "
        "or count; 6 = the two BLU front VPs (9690, 6700) that were empty and adjacent by land to RED's start "
        "provinces 716 and 684. ARENA_VP now logs the holder of every VP so the next load proves it.",
    ],
}


def shared_manifest(region: Region, scenario_id: str, variants: tuple[Variant, ...],
                    meaning: dict[str, str]) -> dict[str, Any]:
    """Manifest keys that are the same contract on every map: scenarios, selectors, log grammar."""
    return {
        "scenarios": [{**asdict(v), "scenario_id": f"{scenario_id}.{v.id}", "index": i,
                       "start": {c.value: [list(s) for s in placements(region, v, c)] for c in Country},
                       "exposed_vps": exposed_vps(region, v)}
                      for i, v in enumerate(variants)],
        "themes_not_built": list(NOT_BUILT), "opponents": list(OPPONENTS), "handicaps": list(HANDICAPS),
        "decisions": meaning, "log_prefixes": list(LOG_PREFIXES),
        "log_grammar": {
            "ARENA_OUTCOME": "episode=<n> winner=BLU|RED|DRAW reason=capital|capital_both|timeout_vp|capitulation "
                             "day=<n> vp_blu=<n> vp_red=<n>",
            "ARENA_RESET": "episode=<n> variant=<index> opponent=<index> handicap=<index> vp_blu=<n> vp_red=<n>",
            "ARENA_TICK": "episode=<n> day=<n> vp_blu=<n> vp_red=<n> div_blu=<n> div_red=<n>",
            "ARENA_VP": "episode=<n> day=<n> then <province>=<holder> per arena VP; holder 1=BLU 2=RED 0=neither",
            "line_format": "game.log wraps each line as '[time][effectbase.cpp:NNNN]: [time][effectbase.cpp:NNNN]: "
                           "ARENA_...': match the ARENA_ token anywhere in the line",
            "note": "winner uses CONTRACT tags (BLU/RED) even if the mod had to pick other country tags"},
    }


def _finish(output: Path, mod: Path, region: Region, layout: ArenaLayout, spec: RegionSpec, tags: dict[str, str],
            plans: tuple[StatePlan, ...], variants: tuple[Variant, ...], config: ModConfig,
            meaning: dict[str, str], quiet_report: dict[str, Any]) -> dict[str, Any]:
    """Launch profile, layout, region evidence and manifest of the vanilla-region build."""
    _write(output / "mod" / f"{MOD_DIR}.mod", _descriptor(mod.as_posix()))
    _write(output / "dlc_load.json", json.dumps({"enabled_mods": [f"mod/{MOD_DIR}.mod"], "disabled_dlcs": []}))
    # The game rewrites this file in full on exit but keeps these values. Popups off: events and news become
    # small icons under the top bar instead of a window over the map (seen live: 30% of frames were blind).
    _write(output / "settings.txt", 'language="l_english"\ngraphics={ size={ x=%d y=%d } fullScreen=no borderless=no }\n'
           "pause_on_popups=no\npopup_news=no\npopup_events=no\npopup_minor_events=no\nhints=no\n"
           % (config.width, config.height))
    layout.save(output / "arena_layout.json")
    write_json(output / "arena_region.json", region.to_json())
    manifest = {
        "schema_version": 1, "evidence_kind": "generated_static_never_loaded_by_the_game",
        "region": spec.scenario_id, "tags": tags, "colors": {k: list(v) for k, v in COLORS.items()},
        "config": asdict(config), "mod_sha256": tree_hash(mod), "mod_dir": f"mod/{MOD_DIR}",
        "bookmark": {"name": "ARENA_BOOKMARK_NAME", "date": config.start_date, "default_country": tags["BLU"]},
        "launch": {"cwd": "<game dir>", "args": ["-debug", f"-mod=mod/{MOD_DIR}.mod"],
                   "autostart_args": [f"-start_tag={tags['BLU']}", "-start_speed=1"],
                   "autostart_args_verified": "integrator live load of mod 81d57adc on v1.19.3: game.log shows "
                                              "'[[ Launching SINGLEPLAYER-game ]]' and the game runs as BLU",
                   "note": "Profile isolation needs gameDataPath in launcher-settings.json (see probe.py). The "
                           "log effect is documented to print to logs/game.log (effects_documentation.md)."},
        "states": [asdict(plan) for plan in plans], "asymmetry": asymmetry(region),
        "live_evidence": LIVE_EVIDENCE, "quiet": quiet_report,
        **shared_manifest(region, spec.scenario_id, variants, meaning),
    }
    write_json(output / "arena_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------------------------
# Static validation (no game launch).

def _doc_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {m.lower() for m in re.findall(r"^## (\S+)\s*$", path.read_text(encoding="utf-8-sig", errors="replace"), re.M)}


def _balanced(text: str) -> bool:
    depth = 0
    for token in re.findall(r'"(?:\\.|[^"\\])*"|#[^\n]*|[{}]', text):
        if token in "{}":
            depth += 1 if token == "{" else -1
            if depth < 0:
                return False
    return depth == 0


def _items(node: Any) -> Iterable[tuple[str, Any]]:
    """(key, value) of a parsed block with duplicate keys expanded again."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "__items__":
                continue
            # The parser folds duplicate keys into a list; a genuine bare list is a list of scalars.
            if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                for entry in value:
                    yield key, entry
            else:
                yield key, value


class _Names:
    """Walks effect/trigger blocks and records names that this build does not document."""

    def __init__(self, game: Path, scripted: set[str]) -> None:
        docs = game / "documentation"
        self.effects = _doc_names(docs / "effects_documentation.md")
        self.triggers = _doc_names(docs / "triggers_documentation.md")
        self.modifiers = _doc_names(docs / "modifiers_documentation.md")
        actions = game / "common" / "on_actions" / "_documentation.md"
        self.on_actions = set(re.findall(r"^- `(\w+)`", actions.read_text(encoding="utf-8-sig", errors="replace"),
                                         re.M)) if actions.is_file() else set()
        self.scripted = scripted
        self.problems: list[str] = []
        self.checked = 0

    @staticmethod
    def _scope(key: str) -> bool:
        return key in SCOPE_WORDS or key.isdigit() or re.fullmatch(r"[A-Z][A-Z0-9]{2}", key) is not None

    def effect(self, node: Any, where: str) -> None:
        for key, value in _items(node):
            low = key.lower()
            if low == "limit":
                self.trigger(value, where)
            elif self._scope(key) or low in ("else", "else_if"):
                self.effect(value, where)
            elif low in self.scripted or key in HISTORY_KEYWORDS:
                continue
            else:
                self.checked += 1
                if self.effects and low not in self.effects:
                    self.problems.append(f"{where}: effect '{key}' is not in effects_documentation.md")
                if low in ("if", "hidden_effect") or low.startswith(("every_", "random_")):
                    self.effect(value, where)

    def trigger(self, node: Any, where: str) -> None:
        for key, value in _items(node):
            low = key.lower()
            if self._scope(key):
                self.trigger(value, where)
                continue
            self.checked += 1
            if self.triggers and low not in self.triggers:
                self.problems.append(f"{where}: trigger '{key}' is not in triggers_documentation.md")
            if low in ("and", "or", "not") or low.startswith(("any_", "all_")):
                self.trigger(value, where)


def check_script(parsed: dict[str, Any], game: Path) -> tuple[list[str], int]:
    """Names used by the generated script against THIS install: effects, triggers, modifiers, on_actions
    and the vanilla technology/equipment/unit ids. Map-independent (also used by ``custommap.py``)."""
    problems: list[str] = []
    effect_files = [data for name, data in sorted(parsed.items()) if name.startswith("common/scripted_effects/")]
    scripted = {key.lower() for data in effect_files for key in data}
    names = _Names(game, scripted)
    for data in effect_files:
        for key, body in _items(data):
            names.effect(body, f"scripted effect {key}")
    actions = [pair for name, data in sorted(parsed.items()) if name.startswith("common/on_actions/")
               for pair in _items(data.get("on_actions", {}))]
    for key, body in actions:
        generic = re.sub(r"^(on_(?:daily|weekly|monthly))_[A-Z][A-Z0-9]{2}$", r"\1_TAG", key)
        if names.on_actions and generic not in names.on_actions:
            problems.append(f"on_action '{key}' is not in common/on_actions/_documentation.md")
        names.effect(body.get("effect", {}), f"on_action {key}")
    for _, category in _items(parsed.get("common/decisions/arena_decisions.txt", {})):
        for key, body in _items(category):
            for part in ("allowed", "visible", "available"):
                names.trigger(body.get(part, {}), f"decision {key}")
            names.effect(body.get("complete_effect", {}), f"decision {key}")
    for key, body in _items(parsed.get("common/decisions/categories/arena_categories.txt", {})):
        names.trigger(body.get("allowed", {}), f"decision category {key}")
    for key, body in _items(parsed.get("common/ai_strategy/arena_ai.txt", {})):
        for part in ("allowed", "enable"):
            names.trigger(body.get(part, {}), f"ai_strategy {key}")
    for relative, data in parsed.items():
        if relative.startswith("history/countries/"):
            names.effect(data, relative)
    bookmark = parsed.get("common/bookmarks/arena_bookmark.txt", {}).get("bookmarks", {}).get("bookmark", {})
    names.effect(bookmark.get("effect", {}), "bookmark effect")
    # The custom map blanks vanilla idea files; an empty file does not parse to a mapping.
    ideas = [idea for name, data in sorted(parsed.items())
             if name.startswith("common/ideas/") and isinstance(data, dict) and isinstance(data.get("ideas"), dict)
             for _, idea in _items(data["ideas"].get("country", {})) if isinstance(idea, dict)]
    for idea in ideas:
        for modifier in idea.get("modifier", {}):
            names.checked += 1
            if names.modifiers and modifier.lower() not in names.modifiers:
                problems.append(f"modifier '{modifier}' is not in modifiers_documentation.md")
    problems = [*problems, *names.problems]
    vanilla_text = ""
    for folder in ("common/technologies", "common/units/equipment", "common/units/equipment/modules",
                   "common/units", "common/ideologies"):
        for path in sorted((game / folder).glob("*.txt")) if (game / folder).is_dir() else ():
            vanilla_text += path.read_text(encoding="utf-8-sig", errors="replace")
    if vanilla_text:
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", vanilla_text))
        problems += [f"'{token}' (technology/equipment/unit id) was not found in vanilla common/"
                     for token in VANILLA_IDS if token not in tokens]
    return problems, names.checked


def entry_scope_problems(parsed: dict[str, Any]) -> list[str]:
    """on_startup (and most on_actions) run in GLOBAL scope, where calling a scripted effect fails with
    "Invalid Scope" (seen on the second live load). Rule: at the top of every on_action effect only ``log``
    or an explicit country scope (a tag, ROOT, FROM) may appear; arena effects are called inside one.
    Decisions are country scope by definition and need no such wrapper."""
    problems = []
    actions = [pair for name, data in sorted(parsed.items()) if name.startswith("common/on_actions/")
               for pair in _items(data.get("on_actions", {}))]
    for key, body in actions:
        effect = body.get("effect", {}) if isinstance(body, dict) else {}
        for name, inner in _items(effect):
            scoped = name in ("ROOT", "FROM") or re.fullmatch(r"[A-Z][A-Z0-9]{2}", name) is not None
            if name != "log" and not (scoped and isinstance(inner, dict)):
                problems.append(f"on_action {key}: '{name}' is called outside an explicit country scope")
    return problems


def map_problems(game: Path, mod: Path, state_of: dict[int, int]) -> list[str]:
    """The two map rules the first live load tripped over, checked statically.

    1. After the mod's overrides every province is still in exactly one strategic region and no
       generated state spans two. 2. Every ``map/buildings.txt`` row sits in a province of its state."""
    problems: list[str] = []
    files = strategic_region_files(game)
    vanilla_all = sorted(p for _, provinces in files.values() for p in provinces)
    folder = mod / "map" / "strategicregions"
    for path in sorted(folder.glob("*.txt")) if folder.is_dir() else ():
        parsed = parse_strategic_region(path.read_text(encoding="utf-8"))
        if path.name not in files or parsed is None or parsed[0] != files[path.name][0]:
            problems.append(f"strategic region override {path.name} does not match a vanilla file and ID")
        else:
            files[path.name] = parsed
    if sorted(p for _, provinces in files.values() for p in provinces) != vanilla_all:
        problems.append("strategic region overrides lose or duplicate provinces")
    region_of = {p: rid for rid, provinces in files.values() for p in provinces}
    for state in sorted(set(state_of.values())) if files else ():
        regions = sorted({region_of.get(p, -1) for p, s in state_of.items() if s == state})
        if len(regions) != 1 or regions[0] < 0:
            problems.append(f"state {state} spans strategic regions {regions}")
    source = mod / "map" / "buildings.txt"
    if (game / "map" / "buildings.txt").is_file() and not source.is_file():
        problems.append("map/buildings.txt override is missing")
    if source.is_file():
        ids, (x0, y0), height = province_raster(game / "map" / "provinces.bmp", read_definitions(game), set(state_of))
        seen: set[tuple[int, str]] = set()
        kinds: set[str] = set()
        wrong = 0
        ours = source.read_text(encoding="utf-8")
        vanilla = (game / "map" / "buildings.txt").read_bytes().decode("utf-8-sig", errors="replace")
        blank = sum(not line.strip() for line in ours.split("\n"))
        if blank != sum(not line.strip() for line in vanilla.replace("\r\n", "\n").split("\n")):
            problems.append(f"buildings.txt has {blank} blank lines, not the same as vanilla (engine: arguments count)")
        malformed = [n for n, line in enumerate(ours.split("\n"), 1) if line.strip() and line.count(";") != 6]
        if malformed:
            problems.append(f"buildings.txt rows without exactly 7 fields at lines {malformed[:5]}")
        for line in ours.split("\n"):
            row = line.split(";")
            if len(row) != 7 or not row[0].strip().isdigit():
                continue
            x, y = _building_pixel(row, height)
            inside = 0 <= x - x0 < ids.shape[1] and 0 <= y - y0 < ids.shape[0]
            expected = state_of.get(int(ids[y - y0, x - x0])) if inside else None
            if expected is not None:
                seen.add((expected, row[1]))
                kinds.add(row[1])
            if (expected is not None or int(row[0]) in state_of.values()) and expected != int(row[0]):
                wrong += 1
                if wrong <= 5:
                    problems.append(f"buildings.txt: '{line}' is not inside a province of state {row[0]}")
        if wrong > 5:
            problems.append(f"buildings.txt: {wrong} rows in total are outside their state")
        missing = sorted((s, k) for s in set(state_of.values()) for k in kinds if (s, k) not in seen)
        if missing:
            problems.append(f"buildings.txt lacks positions for {missing[:5]} ({len(missing)} in total)")
    return problems


def validate_mod(output: Path, game: Path | None = None) -> dict[str, Any]:
    """Static checks only. ``game=None`` skips every check that needs the install."""
    problems: list[str] = []
    mod = output / "mod" / MOD_DIR
    manifest = json.loads((output / "arena_manifest.json").read_text(encoding="utf-8"))
    tags = manifest["tags"]
    parsed: dict[str, Any] = {}
    for path in sorted(mod.rglob("*.txt")):
        relative = path.relative_to(mod).as_posix()
        text = path.read_text(encoding="utf-8")
        if relative in ("common/countries/colors.txt", "map/buildings.txt"):
            continue  # vanilla bytes plus our rows; not key = value script (buildings are checked below)
        if relative.startswith("events/"):  # 3.6 MB of vanilla text plus hidden lines: compared in quiet_problems
            continue
        try:
            parsed[relative] = parse(text)
        except Exception as exc:  # noqa: BLE001 - every parser failure is a finding
            problems.append(f"{relative}: does not parse: {exc}")
        if not _balanced(text):
            problems.append(f"{relative}: unbalanced braces")
    if tree_hash(mod) != manifest["mod_sha256"]:
        problems.append("mod folder does not match the manifest hash")

    # Layout: the loader enforces symmetric adjacency and one capital per side.
    layout = ArenaLayout.load(output / "arena_layout.json")
    ids = {p.id for p in layout.provinces}
    seen, stack = {layout.provinces[0].id}, [layout.provinces[0].id]
    while stack:
        for neighbor in layout.province(stack.pop()).neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    if seen != ids:
        problems.append("layout adjacency is not connected")
    if layout.source != "vanilla_region" or set(layout.sectors) != set(SECTORS):
        problems.append(f"layout source/sectors are wrong: {layout.source} {layout.sectors}")
    if any(not p.sector or p.initial_controller not in ("BLU", "RED") for p in layout.provinces):
        problems.append("every layout province needs a sector and an initial controller")
    if any(not set(p.river_neighbors) <= set(p.neighbors) for p in layout.provinces):
        problems.append("river crossings must be adjacencies")

    # States: every layout province in exactly one state of its own side; VPs inside their state.
    owner_of: dict[int, str] = {}
    state_of: dict[int, int] = {}
    quiet_report = manifest.get("quiet", {})
    seed_file = "history/states/" + quiet_report.get("inert_country", {}).get("seed_state_file", "")
    contract = {value: key for key, value in tags.items()}
    for relative, data in parsed.items():
        if relative.startswith("history/states/") and relative != seed_file:
            state = data["state"]
            history = state["history"]
            for province in state["provinces"]:
                if province in owner_of:
                    problems.append(f"province {province} is in two states")
                owner_of[province] = contract.get(history["owner"], "?")
                state_of[province] = state["id"]
            points = history.get("victory_points", [])
            for block in points if points and isinstance(points[0], list) else [points] if points else []:
                if block[0] not in state["provinces"]:
                    problems.append(f"{relative}: victory point {block[0]} is outside the state")
                if layout.province(block[0]).victory_points != float(block[1]):
                    problems.append(f"{relative}: victory point {block[0]} disagrees with the layout")
    for province in layout.provinces:
        if owner_of.get(province.id) != province.initial_controller:
            problems.append(f"province {province.id}: state owner {owner_of.get(province.id)} != layout")

    # Orders of battle: equal counts per side, units start on their own land.
    enemy_capital = {c.value: layout.capital(c.opponent).id for c in Country}
    for scenario in manifest["scenarios"]:
        counts = {}
        for side in ("BLU", "RED"):
            for level in manifest["handicaps"]:
                name = f"history/units/ARENA_{scenario['index']:02d}_{tags[side]}_{level}.txt"
                units = parsed.get(name, {}).get("units", {}).get("division", [])
                units = units if isinstance(units, list) else [units]
                counts[side, level] = len(units)
                for unit in units:
                    if owner_of.get(unit["location"]) != side or unit["location"] not in ids:
                        problems.append(f"{name}: division starts outside its own land at {unit['location']}")
                    elif layout.province(unit["location"]).initial_controller != side:
                        problems.append(f"{name}: start province {unit['location']} is not controlled by {side}")
                    elif enemy_capital[side] in layout.province(unit["location"]).neighbors:
                        problems.append(f"{name}: division starts next to the enemy capital at {unit['location']}")
                    if unit["division_template"] not in (INFANTRY, ARMOR):
                        problems.append(f"{name}: unknown template {unit['division_template']}")
        for level in manifest["handicaps"]:
            if counts["BLU", level] != counts["RED", level] or counts["BLU", level] < 1:
                problems.append(f"{scenario['id']}: unequal or empty division counts at handicap {level}")
        if counts["BLU", "even"] != scenario["divisions"]:
            problems.append(f"{scenario['id']}: order of battle does not match the declared division count")
    problems += entry_scope_problems(parsed)
    splits = {scenario["split"] for scenario in manifest["scenarios"]}
    if splits != {"train", "validation", "held_out"}:
        problems.append(f"scenario splits are incomplete: {sorted(splits)}")

    checked = 0
    if game is not None:
        definitions = read_definitions(game)
        states = read_states(game)
        for province in owner_of:
            if province not in definitions:
                problems.append(f"province {province} does not exist in vanilla")
        for relative, data in parsed.items():
            if relative.startswith("history/states/") and relative != seed_file and data["state"]["id"] not in states:
                problems.append(f"{relative}: state {data['state']['id']} does not exist in vanilla")
        vanilla_members = {p for s in manifest["states"] for p in states[s["id"]].provinces if s["id"] in states}
        if vanilla_members != set(owner_of):
            problems.append("the rewritten states do not cover exactly the vanilla states' provinces")
        problems += map_problems(game, mod, state_of)
        if quiet_report:
            from . import quiet
            problems += quiet.quiet_problems(game, mod, quiet_report, set(tags.values()))
        if set(tags.values()) & vanilla_tags(game):
            problems.append("an arena country tag collides with a vanilla tag")
        found, checked = check_script(parsed, game)
        problems += found
    return {"ok": not problems, "problems": problems, "files_parsed": len(parsed), "names_checked": checked,
            "install_checks": game is not None, "mod_sha256": manifest["mod_sha256"],
            "evidence_kind": "static_validation_not_a_game_load"}


# ---------------------------------------------------------------------------------------------
# CLI.

def add_commands(commands: Any) -> dict[str, Callable[[argparse.Namespace], int]]:
    build = commands.add_parser("mod-build", help="generate the arena mod, layout and launch profile (no launch)")
    build.add_argument("--game", type=Path, default=DEFAULT_GAME)
    build.add_argument("--output", type=Path, default=Path("artifacts/arena_mod/default"))
    build.add_argument("--width", type=int, default=1920)
    build.add_argument("--height", type=int, default=1080)
    build.add_argument("--default-scenario", default="inf2_line", choices=[v.id for v in VARIANTS])
    build.add_argument("--default-opponent", default="default", choices=OPPONENTS)
    build.add_argument("--default-handicap", default="even", choices=HANDICAPS)
    build.add_argument("--world", default="inert", choices=("inert", "vanilla"),
                       help="inert: one do-nothing country annexes everyone else at startup (see quiet.py)")
    build.add_argument("--force", action="store_true", help="replace a previously generated arena mod")
    check = commands.add_parser("mod-validate", help="static checks of a generated arena mod (no launch)")
    check.add_argument("--game", type=Path, default=DEFAULT_GAME)
    check.add_argument("--output", type=Path, default=Path("artifacts/arena_mod/default"))
    check.add_argument("--no-install", action="store_true", help="skip checks that read the game install")

    def run_build(args: argparse.Namespace) -> int:
        config = ModConfig(width=args.width, height=args.height, default_scenario=args.default_scenario,
                           default_opponent=args.default_opponent, default_handicap=args.default_handicap,
                           world=args.world)
        manifest = build_mod(args.game, args.output, DEFAULT_REGION, config, args.force)
        report = validate_mod(args.output, args.game)
        print(json.dumps({"output": str(args.output.resolve()), "mod_sha256": manifest["mod_sha256"],
                          "tags": manifest["tags"], "scenarios": len(manifest["scenarios"]),
                          "asymmetry": manifest["asymmetry"],
                          "validation": report}, indent=2))
        return 0 if report["ok"] else 1

    def run_validate(args: argparse.Namespace) -> int:
        report = validate_mod(args.output, None if args.no_install else args.game)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    return {"mod-build": run_build, "mod-validate": run_validate}
