"""FROM-SCRATCH total-conversion arena maps: every ``map/`` file the game needs, generated from a
``MapDesign`` (see ``mapdesign.py``), plus the arena content of ``generate.py`` retargeted to the new ids.

Output per design, under ``<output>/<design>/`` (never the Steam folder, never the user's Documents):
``mod/hoi4_arena_<design>/`` + ``.mod`` descriptor, ``dlc_load.json``, ``settings.txt``, ``arena_layout.json``
(``ArenaLayout``, source ``custom_map``, province ids == game province ids), ``arena_map.json`` (everything
perception may need: centres, boxes, partners, frame, repairs), ``arena_manifest.json``, ``FIRST_LAUNCH.md``
and the preview PNGs of ``preview.py``.

Choices that matter (details next to the code):
- Blocked infill is LAND inside ``impassable = yes`` states (default). Vanilla does exactly this for the
  Sahara or the Himalaya, it draws the hatched impassable border, and unlike lakes or sea it adds no coast,
  no naval zone and no ports inside the arena. ``blocked_kind="lake"`` is kept as the documented fallback.
- The look is schematic on purpose: flat heightmap (land 100 = 10.0, water 85), one terrain texture for all
  playable land, no trees, no cities, flat normal map, flat colour maps.
- Only id-bound vanilla files are overridden or unloaded (``REPLACE_PATHS``); everything else (terrain
  textures, seasons, cities.txt, technologies, units, ideas...) deliberately falls through to vanilla.

Honesty: NOTHING here has been loaded by the game. ``validate_custom_map`` is a static check of our own
files against the documented rules; whether a two-country, ~400-province world loads at all is unknown
(``FIRST_LAUNCH.md`` lists what to expect). Output is deterministic: two builds are byte-identical.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ...clausewitz import parse
from ..contracts import ArenaError, Country
from ..diagnostics import write_json
from ..fingerprint import tree_hash
from ..layout import ArenaLayout
from . import bmpio
from .generate import (
    ARMOR,
    COLORS,
    INFANTRY,
    VARIANTS,
    Bare,
    ModConfig,
    Pairs,
    StatePlan,
    _balanced,
    _write,
    check_script,
    emit,
    q,
    shared_manifest,
    write_content,
)
from .neutralise import BLANK_TEXT, blank_plan
from .mapdesign import DEFAULT_DESIGN, MIN_PIXELS, PRESETS, TERRAIN_INDEX, CustomMap, MapDesign, border_lengths
from .mapdesign import rasterize, summary, x_crossings
from .region import DEFAULT_GAME, SECTORS, asymmetry

MOD_PREFIX = "hoi4_arena_"
MOD_NAME = "HOI4-AI Arena Map"
DEFAULT_OUTPUT = Path("artifacts/arena_custom")
LAND_HEIGHT, WATER_HEIGHT = 100, 85  # heightmap values; 95 is sea level, y = value / 10
RAIL_LEVEL = 3
DYNAMIC_TAGS = 10
ADJACENCY_HEADER = "From;To;Type;Through;start_x;start_y;stop_x;stop_y;adjacency_rule_name;Comment\n"
CLOSING_LINE = "-1;-1;-1;-1;-1;-1;-1;-1;-1"
CONTINENTS = ("europe", "north_america", "south_america", "australia", "africa", "asia", "middle_east")
MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
WATER_RGBA = (57, 100, 104, 4)  # typical open-ocean block of vanilla colormap_water_0.dds
TINT = {"plains": (70, 96, 48), "forest": (44, 74, 40), "hills": (104, 100, 60), "mountain": (112, 106, 98),
        "marsh": (60, 96, 84), "lakes": (40, 70, 96), "ocean": (12, 28, 6)}
STATE_SINGLES = (("air_base", 1), ("anti_air_building", 3), ("arms_factory", 6), ("fuel_silo", 1),
                 ("industrial_complex", 6), ("nuclear_reactor_spawn", 1), ("radar_station", 1),
                 ("rocket_site_spawn", 1), ("stronghold_network", 1), ("synthetic_refinery", 1))
PROVINCE_BUILDINGS = ("bunker", "supply_node", "special_project_facility_spawn")
COASTAL_BUILDINGS = ("coastal_bunker", "naval_base_spawn", "naval_headquarters", "naval_supply_hub")

# folder -> why vanilla's content must not load. Derived from a scan of the 1.19.3 install for files that
# name state / province / strategic-region ids as DATA (see ``scan_id_references``), not from a recipe.
REPLACE_PATHS: dict[str, str] = {
    "history/states": "1081 state files list vanilla province ids; ours are the only valid states",
    "history/countries": "364 countries whose capitals are vanilla state ids and whose tags no longer exist",
    "history/units": "609 orders of battle place divisions and fleets on vanilla province ids",
    "map/strategicregions": "304 regions list vanilla province ids; a province in two regions is a MAP_ERROR",
    "map/supplyareas": "legacy file naming vanilla state ids",
    "common/country_tags": "every vanilla tag would need a history file and a capital state",
    "common/bookmarks": "vanilla bookmarks offer countries that do not exist (crash on selection)",
    "common/national_focus": "15619 state/province references in completion rewards; trees are per vanilla tag",
    "common/decisions": "20415 id references; hundreds of targeted decisions would evaluate against missing states",
    "common/decisions/categories": "categories gate on vanilla tags/states; ours are the only ones wanted",
    "common/on_actions": "fires vanilla events and effects on vanilla states/tags at startup and daily",
    "events": "9592 id references; nothing may fire into the arena and its logs",
    "common/ai_strategy": "726 references to vanilla states/regions/tags in AI orders",
    "common/ai_strategy_plans": "plans reference vanilla focus ids and tags",
    "common/ai_areas": "areas list vanilla strategic-region ids (replaced by one arena area)",
    "common/ai_faction_theaters": "theaters list vanilla strategic-region ids",
    "common/characters": "673 tag-bound character files; ours define the two leaders",
}
LEFT_ALONE: dict[str, str] = {
    "common/scripted_effects, scripted_triggers, scripted_localisation, ideas, dynamic_modifiers": "referenced "
    "by un-replaced vanilla script (technologies, laws, units); their id references sit inside triggers that "
    "are simply false here. Expect harmless 'invalid state/tag' lines in error.log.",
    "common/units/names_*, common/names, portraits, common/countries": "keyed by tag, never looked up.",
    "common/operations, raids, factions, peace_conference, military_industrial_organization, "
    "special_projects, game_rules, difficulty_settings": "feature data gated by triggers/tags; not reachable "
    "with two scripted countries. First candidates to add to replace_path if error.log noise matters.",
    "map/terrain/*.dds (except colour maps), map/seasons.txt, cities.txt, colors.txt, terrain atlas": "not "
    "id-bound; vanilla's files are valid for any map size.",
}


def mod_dir(design: MapDesign) -> str:
    return MOD_PREFIX + design.name


# ---------------------------------------------------------------------------------------------
# Planning: colours, states, strategic regions.

def province_colors(arena: CustomMap) -> dict[int, tuple[int, int, int]]:
    """Unique, deterministic RGB per province; hue family by kind so the raw bitmap is readable."""
    bases = {"play": (70, 120, 30), "blocked": (110, 100, 100), "box": (150, 150, 40), "sea": (10, 30, 120)}
    spans = {"play": (150, 130, 90), "blocked": (40, 30, 30), "box": (60, 60, 40), "sea": (40, 70, 130)}
    used: set[tuple[int, int, int]] = {(0, 0, 0)}
    out = {}
    for province in arena.provinces:
        base, span = bases[province.kind], spans[province.kind]
        step = province.id
        while True:
            color = (base[0] + step * 53 % span[0], base[1] + step * 29 % span[1], base[2] + step * 17 % span[2])
            if color not in used:
                break
            step += 101
        used.add(color)
        out[province.id] = color
    return out


def plan_custom_states(arena: CustomMap) -> tuple[StatePlan, ...]:
    """Per side: the capital province as its own state, one state per sector, then blocked infill and the
    optional player box. Sequential ids from 1; BLU first so state n and n + k are counterparts."""
    groups: list[tuple[str, str, list[int]]] = []
    for side in ("BLU", "RED"):
        mine = [p for p in arena.of_kind("play") if p.side == side]
        capital = arena.capitals[side]
        groups.append((side, "capital", [capital]))
        for sector in SECTORS:
            members = [p.id for p in mine if p.sector == sector and p.id != capital]
            if members:
                groups.append((side, sector, members))
    for kind, label in (("blocked", "infill"), ("box", "box")):
        for side in ("BLU", "RED"):
            members = [p.id for p in arena.of_kind(kind) if p.side == side]
            if members:
                groups.append((side, label, members))
    return tuple(StatePlan(index, f"{index}-ARENA_{side}_{label}.txt", f"ARENA_STATE_{side}_{label}", side, label,
                           tuple(sorted(members)), ())
                 for index, (side, label, members) in enumerate(groups, start=1))


def plan_regions(arena: CustomMap) -> tuple[tuple[int, str, tuple[int, ...]], ...]:
    """(id, name, provinces): one land region per side, the ocean as four quadrants."""
    height, width = arena.ids.shape
    buckets: dict[str, list[int]] = {}
    for p in arena.provinces:
        if p.kind == "sea":
            name = ("North" if p.cy < height // 2 else "South") + (" West" if p.cx < width // 2 else " East") + " Sea"
        else:
            name = "Blue Land" if p.side == "BLU" else "Red Land"
        buckets.setdefault(name, []).append(p.id)
    order = ["Blue Land", "Red Land", "North West Sea", "North East Sea", "South West Sea", "South East Sea"]
    return tuple((index, name, tuple(sorted(buckets[name])))
                 for index, name in enumerate([n for n in order if n in buckets], start=1))


# ---------------------------------------------------------------------------------------------
# map/ files.

def _palette(game: Path | None, name: str) -> list[tuple[int, int, int]]:
    path = None if game is None else game / "map" / f"{name}.bmp"
    return bmpio.vanilla_palette(path) if path is not None and path.is_file() else bmpio.fallback_palette(name)


def _fmt(*values: float) -> str:
    return ";".join(f"{value:.2f}" for value in values)


def _positions(arena: CustomMap, plans: tuple[StatePlan, ...]) -> tuple[str, str]:
    """(buildings.txt, unitstacks.txt). Geometric: interior pixels of the right province, y from the flat
    heightmap, z counted from the BOTTOM of the bitmap as vanilla does."""
    ids = arena.ids
    height, width = ids.shape
    interior = np.ones(ids.shape, dtype=bool)
    for _ in range(3):
        core = interior.copy()
        core[1:] &= interior[:-1] & (ids[1:] == ids[:-1])
        core[:-1] &= interior[1:] & (ids[:-1] == ids[1:])
        core[:, 1:] &= interior[:, :-1] & (ids[:, 1:] == ids[:, :-1])
        core[:, :-1] &= interior[:, 1:] & (ids[:, :-1] == ids[:, 1:])
        core[0] = core[-1] = False
        core[:, 0] = core[:, -1] = False
        interior = core
    flat = ids.ravel()
    order = np.argsort(flat, kind="stable")
    bounds = np.concatenate(([0], np.cumsum(np.bincount(flat, minlength=len(arena.provinces) + 1))))
    inner = interior.ravel()

    def pixels(province_id: int) -> Any:
        mine = order[bounds[province_id]:bounds[province_id + 1]]
        good = mine[inner[mine]]
        return good if len(good) else mine

    def spot(province_id: int, k: int) -> tuple[float, float]:
        mine = pixels(province_id)
        index = int(mine[(k * 7919 + 13) % len(mine)])
        return index % width + 0.5, index // width + 0.5

    sea = {p.id for p in arena.of_kind("sea")}
    level = {p.id: (9.5 if p.kind == "sea" or p.terrain == "lakes" else LAND_HEIGHT / 10) for p in arena.provinces}
    coast: dict[int, tuple[int, tuple[int, int], tuple[int, int]]] = {}  # land -> sea id, land px, sea px
    is_sea = np.zeros(len(arena.provinces) + 1, dtype=bool)
    is_sea[sorted(sea)] = True
    for dy, dx in ((0, 1), (1, 0)):
        a, b = ids[:height - dy, :width - dx], ids[dy:, dx:]
        for y, x in np.argwhere((a != b) & (is_sea[a] != is_sea[b])).tolist():
            first, second = (x, y), (x + dx, y + dy)
            land_px, sea_px = (second, first) if is_sea[ids[y, x]] else (first, second)
            coast.setdefault(int(ids[land_px[1], land_px[0]]), (int(ids[sea_px[1], sea_px[0]]), land_px, sea_px))

    def row(owner: int, kind: str, x: float, y_level: float, y: float, rotation: float, extra: str) -> str:
        return f"{owner};{kind};{_fmt(x, y_level, height - y, rotation)};{extra}\n"

    buildings = []
    for plan in plans:
        k = 0
        for kind, count in STATE_SINGLES:
            for _ in range(count):
                province = plan.land[k % len(plan.land)]
                bx, by = spot(province, k)
                buildings.append(row(plan.id, kind, bx, level[province], by, 0, "0"))
                k += 1
        for province in plan.land:
            for kind in PROVINCE_BUILDINGS:
                k += 1
                bx, by = spot(province, k)
                buildings.append(row(plan.id, kind, bx, level[province], by, 0, "0"))
            if province in coast:
                sea_id, (lx, ly), (sx, sy) = coast[province]
                rotation = math.atan2(-(sy - ly), sx - lx)
                for kind in COASTAL_BUILDINGS:
                    extra = str(sea_id) if kind == "naval_base_spawn" else "0"
                    buildings.append(row(plan.id, kind, lx + 0.5, level[province], ly + 0.5, rotation, extra))
                buildings.append(row(plan.id, "floating_harbor", sx + 0.5, 9.5, sy + 0.5, rotation, str(sea_id)))
        first_coast = next((p for p in plan.land if p in coast), None)
        if first_coast is not None:
            _, (lx, ly), _ = coast[first_coast]
            buildings.append(row(plan.id, "dockyard", lx + 0.5, level[first_coast], ly + 0.5, 0, "0"))

    around: dict[int, list[int]] = {p.id: [] for p in arena.provinces}
    for a, b in sorted(arena.borders):
        around[a].append(b)
        around[b].append(a)
    stacks = []
    for p in arena.provinces:
        cx, cy, radius = p.cx + 0.5, p.cy + 0.5, max(2.0, min(12.0, math.sqrt(p.pixels) / 5))

        def stack(kind: int, x: float, y: float, rotation: float = 0.0, p: Any = p) -> None:
            x, y = round(x, 2), round(y, 2)  # what the file will say
            if not (0 <= x < width and 0 <= y < height) or int(ids[int(y), int(x)]) != p.id:
                x, y = p.cx + 0.5, p.cy + 0.5
            stacks.append(f"{p.id};{kind};{_fmt(x, level[p.id], height - y, rotation)};0.00\n")

        def toward(index: int, pool: list[int]) -> tuple[float, float, float]:
            if pool:
                other = arena.province(pool[index % len(pool)])
                angle = math.atan2(-(other.cy - p.cy), other.cx - p.cx)
            else:
                angle = index * math.pi / 4
            return cx + math.cos(angle) * radius * 1.5, cy - math.sin(angle) * radius * 1.5, angle

        same = [n for n in around[p.id] if (n in sea) == (p.id in sea)]
        other = [n for n in around[p.id] if (n in sea) != (p.id in sea)]
        stack(0, cx, cy)
        for index in range(min(8, len(same))):
            stack(1 + index, *toward(index, same))
        stack(9, cx - radius, cy, 1.57)
        stack(10, cx + radius, cy, -1.57)
        if other:
            for index in range(8):
                stack(11 + index, *toward(index, other))
            if p.id in coast:
                _, (lx, ly), _ = coast[p.id]
                stack(19, lx + 0.5, ly + 0.5)
                stack(20, lx + 0.5, ly + 0.5)
        stack(21, cx - radius * 0.6, cy)
        for index in range(min(8, len(same))):
            stack(22 + index, *toward(index, same))
        if other:
            for index in range(8):
                stack(30 + index, *toward(index, other))
        stack(38, cx, cy + radius)
    return "".join(buildings), "".join(stacks)


def _region_file(region_id: int, name: str, provinces: tuple[int, ...], sea: bool) -> str:
    periods = []
    for month, days in enumerate(MONTH_DAYS):
        periods.append(
            f"\t\tperiod={{\n\t\t\tbetween={{ 0.{month} {days - 1}.{month} }}\n\t\t\ttemperature={{ 12.0 20.0 }}\n"
            "\t\t\tno_phenomenon=1.000\n\t\t\train_light=0.000\n\t\t\train_heavy=0.000\n\t\t\tsnow=0.000\n"
            "\t\t\tblizzard=0.000\n\t\t\tarctic_water=0.000\n\t\t\tmud=0.000\n\t\t\tsandstorm=0.000\n"
            "\t\t\tmin_snow_level=0.000\n\t\t}\n")
    terrain = "\tnaval_terrain=water_deep_ocean\n" if sea else ""
    return (f"\nstrategic_region={{\n\tid={region_id}\n\tname=\"STRATEGICREGION_{region_id}\"\n\tprovinces={{\n\t\t"
            + " ".join(str(p) for p in provinces) + f" \n\t}}\n{terrain}\tweather={{\n" + "".join(periods) + "\t}\n}\n")


def write_map(mod: Path, arena: CustomMap, plans: tuple[StatePlan, ...], game: Path | None,
              uniform_texture: bool = True) -> None:
    ids = arena.ids
    height, width = ids.shape
    folder = mod / "map"
    colors = province_colors(arena)
    table = np.zeros((len(arena.provinces) + 1, 3), dtype=np.uint8)
    for province_id, color in colors.items():
        table[province_id] = color
    bmpio.write_bmp24(folder / "provinces.bmp", table[ids], ppm=3780)

    kind_lookup = np.array([""] + [p.terrain for p in arena.provinces])
    water = np.isin(kind_lookup, ("ocean", "lakes"))[ids]
    bmpio.write_bmp8(folder / "heightmap.bmp", np.where(water, WATER_HEIGHT, LAND_HEIGHT), bmpio.grey_palette())
    texture = np.zeros(len(arena.provinces) + 1, dtype=np.uint8)
    for p in arena.provinces:
        texture[p.id] = (15 if p.terrain == "ocean" else 14 if p.terrain == "lakes" else 6 if p.kind == "blocked"
                         else 0 if uniform_texture else TERRAIN_INDEX[p.terrain])
    bmpio.write_bmp8(folder / "terrain.bmp", texture[ids], _palette(game, "terrain"))
    bmpio.write_bmp8(folder / "rivers.bmp", arena.rivers, _palette(game, "rivers"))
    bmpio.write_bmp8(folder / "trees.bmp", np.zeros((height * 75 // 256, width * 75 // 256), dtype=np.uint8),
                     _palette(game, "trees"))
    bmpio.write_bmp8(folder / "cities.bmp", np.full((height, width), 15, dtype=np.uint8), _palette(game, "cities"))
    normal = np.empty((height // 2, width // 2, 3), dtype=np.uint8)
    normal[:] = (128, 128, 255)
    bmpio.write_bmp24(folder / "world_normal.bmp", normal)

    tint = np.zeros((len(arena.provinces) + 1, 4), dtype=np.uint8)
    for p in arena.provinces:
        tint[p.id] = (*TINT["mountain" if p.kind == "blocked" and p.terrain != "lakes" else p.terrain
                            if not uniform_texture or p.kind != "play" else "plains"], 0)
    bmpio.write_dds_argb(folder / "terrain" / "colormap_rgb_cityemissivemask_a.dds", tint[ids[::2, ::2]])
    for level in range(3):
        bmpio.write_dds_dxt5_flat(folder / "terrain" / f"colormap_water_{level}.dds", width >> (level + 1),
                                  height >> (level + 1), WATER_RGBA)

    rows = ["0;0;0;0;land;false;unknown;0"]
    for p in arena.provinces:
        r, g, b = colors[p.id]
        kind = "sea" if p.kind == "sea" else "lake" if p.terrain == "lakes" else "land"
        rows.append(f"{p.id};{r};{g};{b};{kind};{'true' if p.coastal else 'false'};{p.terrain};"
                    f"{0 if p.kind == 'sea' else 1}")
    (folder / "definition.csv").write_bytes("".join(row + "\r\n" for row in rows).encode("ascii"))  # CRLF: wiki

    kind_of = {p.id: p.kind for p in arena.provinces}
    links = [pair for pair in arena.impassable_links if "sea" not in (kind_of[pair[0]], kind_of[pair[1]])]
    _write(folder / "adjacencies.csv",
           ADJACENCY_HEADER
           + "".join(f"{a};{b};impassable;-1;-1;-1;-1;-1;;centre-line X-crossing repair\n" for a, b in links)
           + CLOSING_LINE + "\n")
    _write(folder / "adjacency_rules.txt", "# The arena has no straits or canals.\n")
    _write(folder / "continent.txt", "continents = {\n" + "".join(f"\t{name}\n" for name in CONTINENTS) + "}\n")
    _write(folder / "default.map",
           'definitions = "definition.csv"\nprovinces = "provinces.bmp"\npositions = "positions.txt"\n'
           'terrain = "terrain.bmp"\nrivers = "rivers.bmp"\nheightmap = "heightmap.bmp"\n'
           'tree_definition = "trees.bmp"\ncontinent = "continent.txt"\nadjacency_rules = "adjacency_rules.txt"\n'
           'adjacencies = "adjacencies.csv"\nambient_object = "ambient_object.txt"\nseasons = "seasons.txt"\n\n'
           "tree = { 3 4 7 10 }\n")
    _write(folder / "positions.txt", "")
    # Vanilla's ambient objects (map frame, water, the Chinese wall) sit at vanilla coordinates far outside
    # this bitmap. Keep only the wind entity, which lives at the origin on any map.
    _write(folder / "ambient_object.txt",
           'type={\n\ttype="ambient_wind_entity"\n\tuse_animation=no\n\talways_visible=yes\n\tobject={\n'
           '\t\tname="ambient_wind"\n\t\tposition={\n\t\t\t0 0 0 \n\t\t}\n\t\trotation={\n\t\t\t0 0 0 \n\t\t}\n\t}\n}\n')
    _write(folder / "supply_nodes.txt", "".join(f"1 {hub} \n" for hub in arena.hubs))
    _write(folder / "railways.txt",
           "".join(f"{RAIL_LEVEL} {len(rail)} {' '.join(str(p) for p in rail)} \n" for rail in arena.railways))
    buildings, stacks = _positions(arena, plans)
    _write(folder / "buildings.txt", buildings)
    _write(folder / "unitstacks.txt", stacks)
    weather = []
    for region_id, name, members in plan_regions(arena):
        sea = arena.province(members[0]).kind == "sea"
        _write(folder / "strategicregions" / f"{region_id}-{name}.txt", _region_file(region_id, name, members, sea))
        anchor = arena.province(members[len(members) // 2])
        weather.append(f"{region_id};{_fmt(anchor.cx + 0.5, 9.5 if sea else 10.0, height - anchor.cy - 0.5)};small\n")
    _write(folder / "weatherpositions.txt", "".join(weather))


def _descriptor(design: MapDesign, path: str | None = None, keep: tuple[str, ...] = ()) -> str:
    text = (f'version="0.1"\ntags={{\n	"Gameplay"\n	"Map"\n}}\nname="{MOD_NAME}: {design.name}"\n'
            'supported_version="1.19.*"\n'
            + "".join(f'replace_path="{folder}"\n' for folder in REPLACE_PATHS if folder not in keep))
    return text + (f'path="{path}"\n' if path else "")


# Live bisection aid (``--bisect``): each name removes or empties one generated piece so a crash can be pinned
# down. A build made with any of them is NOT a valid arena (the validators will say so) and says so in its manifest.
BISECT_FILES = {"rails": "map/railways.txt", "hubs": "map/supply_nodes.txt", "buildings": "map/buildings.txt",
                "stacks": "map/unitstacks.txt", "weather": "map/weatherpositions.txt"}
BISECT_REMOVE = {"ambient": "map/ambient_object.txt", "dds": "map/terrain", "normal": "map/world_normal.bmp",
                 "areas": "common/ai_areas/arena_ai_areas.txt",
                 "dynamic_tags": "common/country_tags/zz_arena_dynamic_countries.txt"}


def _apply_bisect(mod: Path, bisect: tuple[str, ...]) -> None:
    for name in bisect:
        if name in BISECT_FILES:
            _write(mod / BISECT_FILES[name], "")
        elif name in BISECT_REMOVE:
            target = mod / BISECT_REMOVE[name]
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        elif name == "adjacencies":
            _write(mod / "map" / "adjacencies.csv", ADJACENCY_HEADER + CLOSING_LINE + "\n")
        elif not name.startswith("keep:") and name not in ("no_blank", "no_rescue"):
            raise ArenaError(f"unknown bisect switch {name}")


# ---------------------------------------------------------------------------------------------
# Build.

def build_custom_map(design: MapDesign, output: Path, game: Path | None = DEFAULT_GAME,
                     config: ModConfig | None = None, force: bool = False, previews: bool = True,
                     bisect: tuple[str, ...] = ()) -> dict[str, Any]:
    """Write one design's mod, launch profile, layout, manifest, previews and FIRST_LAUNCH.md into
    ``output``. ``game`` is only READ (palettes; ``None`` uses the built-in palette entries)."""
    config = config or ModConfig()
    output = output.resolve()
    if game is not None and output.is_relative_to(game.resolve()):
        raise ArenaError("refusing to generate inside the game installation")
    mod = output / "mod" / mod_dir(design)
    if mod.exists():
        marker = mod / "descriptor.mod"
        if not force or not marker.is_file() or MOD_NAME not in marker.read_text(encoding="utf-8"):
            raise ArenaError(f"{mod} exists; pass force only to replace a previously generated arena map")
        shutil.rmtree(mod)
    arena = rasterize(design)
    region, layout, plans = arena.region(), arena.layout(), plan_custom_states(arena)
    tags = {"BLU": "BLU", "RED": "RED"}  # country_tags is replaced, so the contract tags are always free
    meaning = write_content(mod, region, plans, tags, VARIANTS, config, "")
    keep = tuple(name[5:] for name in bisect if name.startswith("keep:"))
    _write(mod / "descriptor.mod", _descriptor(design, keep=keep))
    points = dict(arena.victory_points)
    for plan in plans:  # state history again: ours need impassable / wasteland and no manpower in the infill
        blocked = plan.label == "infill"
        history: Pairs = [("owner", plan.owner), ("add_core_of", plan.owner)]
        history += [("victory_points", Bare((p, points[p]))) for p in plan.land if p in points]
        history.append(("buildings", [("infrastructure", 0 if blocked else config.infrastructure)]))
        body: Pairs = [("id", plan.id), ("name", q(plan.name)), ("manpower", 1 if blocked else config.state_manpower),
                       ("state_category", "wasteland" if blocked else "town")]
        if blocked and design.blocked_kind == "impassable":
            body.append(("impassable", True))
        body += [("history", history), ("provinces", Bare(plan.land)), ("local_supplies", 0.0)]
        _write(mod / "history" / "states" / plan.file_name, emit([("state", body)]))
    regions = plan_regions(arena)
    _write(mod / "common" / "ai_areas" / "arena_ai_areas.txt",
           emit([("areas", [("arena", [("continents", Bare(("europe",))),
                                       ("strategic_regions", Bare(tuple(r[0] for r in regions)))])])]))
    capital_name = {arena.capitals["BLU"]: "Blue Capital", arena.capitals["RED"]: "Red Capital"}
    # replace/: vanilla localisation still loads and already owns STRATEGICREGION_<n> / VICTORY_POINTS_<n> keys.
    _write(mod / "localisation" / "english" / "replace" / "arena_map_l_english.yml", "\ufeffl_english:\n" + "".join(
        [f' STRATEGICREGION_{region_id}:0 "{name}"\n' for region_id, name, _ in regions]
        + [f' VICTORY_POINTS_{p}:0 "{capital_name.get(p, f"Arena {p}")}"\n' for p, _ in arena.victory_points]))
    # Dynamic tags (civil wars, releasables): vanilla's D01.. point at common/countries/D01.txt, which we do not
    # replace. The tag list itself is replaced, so it has to be declared again.
    _write(mod / "common" / "country_tags" / "zz_arena_dynamic_countries.txt", "dynamic_tags = yes\n" + "".join(
        f'D{n:02d} = "countries/D{n:02d}.txt"\n' for n in range(1, DYNAMIC_TAGS + 1)))
    write_map(mod, arena, plans, game)
    _apply_bisect(mod, bisect)
    neutralised: dict[str, Any] = {"blank": [], "note": "built without an install: no vanilla file was neutralised"}
    if game is not None:
        neutralised = blank_plan(game, tuple(REPLACE_PATHS), rescue="no_rescue" not in bisect)
        if "no_blank" in bisect:
            neutralised["blank"] = []
        for relative in neutralised["blank"]:
            _write(mod / relative, BLANK_TEXT)

    _write(output / "mod" / f"{mod_dir(design)}.mod", _descriptor(design, mod.as_posix(), keep))
    _write(output / "dlc_load.json", json.dumps({"enabled_mods": [f"mod/{mod_dir(design)}.mod"], "disabled_dlcs": []}))
    _write(output / "settings.txt", 'language="l_english"\ngraphics={ size={ x=%d y=%d } fullScreen=no borderless=no }\n'
           % (config.width, config.height))
    layout.save(output / "arena_layout.json")
    write_json(output / "arena_map.json", {
        "schema_version": 1, "source": "custom_map", **summary(arena), "design_spec": asdict(design),
        "frame": {"origin": list(arena.frame[:2]), "scale": list(arena.frame[2:]), "offset": list(arena.frame_offset),
                  "note": "normalized = (map_pixel - origin) * scale + offset; flat fit, not the camera"},
        "provinces": [asdict(p) for p in arena.provinces], "repairs": [list(p) for p in arena.repairs],
        "capitals": arena.capitals, "victory_points": [list(v) for v in arena.victory_points],
        "hubs": list(arena.hubs), "railways": [list(r) for r in arena.railways],
        "strategic_regions": [[r, name, list(members)] for r, name, members in regions],
        "river_crossings": {"method": "4-connected line between province centres touches a river pixel (unverified)",
                            "links": sorted([a, b] for a, bs in arena.river_neighbors.items() for b in bs if a < b)}})
    manifest = {
        "schema_version": 1, "evidence_kind": "generated_static_never_loaded_by_the_game", "map": "custom_total_conversion",
        "design": design.name, "region": design.scenario_id, "description": design.description, "tuned": design.tuned,
        "symmetry": design.symmetry, "expected_routes": design.expected_routes, "seed_used": arena.seed_used,
        "dataset_role": {"suggestion": design.dataset_role,
                         "note": "train_validation: use the per-variant splits below. held_out_map: keep EVERY variant "
                                 "of this map out of training so generalization to an unseen map can be measured. "
                                 "A suggestion; the integrator may override it."},
        "tags": tags, "colors": {k: list(v) for k, v in COLORS.items()}, "config": asdict(config),
        "mod_sha256": tree_hash(mod), "mod_dir": f"mod/{mod_dir(design)}",
        "bookmark": {"name": "ARENA_BOOKMARK_NAME", "date": config.start_date, "default_country": "BLU"},
        "launch": {"cwd": "<game dir>", "args": ["-debug", f"-mod=mod/{mod_dir(design)}.mod"],
                   "note": "Profile isolation needs gameDataPath in launcher-settings.json (see probe.py). "
                           "Read FIRST_LAUNCH.md before the first start."},
        "states": [asdict(plan) for plan in plans], "asymmetry": asymmetry(region),
        "debug_bisect": list(bisect), "replace_path": REPLACE_PATHS, "left_to_vanilla": LEFT_ALONE, "neutralised_vanilla_files": neutralised,
        "blocked_infill": design.blocked_kind, "provinces": summary(arena)["provinces"],
        "known_limitations": ["load_oob cannot create armies: the executor forms one army after every reset"],
        **shared_manifest(region, design.scenario_id, VARIANTS, meaning),
    }
    write_json(output / "arena_manifest.json", manifest)
    _write(output / "FIRST_LAUNCH.md", first_launch_text(design, manifest))
    if previews:
        from .preview import write_previews
        write_previews(arena, plans, region, output)
    return manifest


# ---------------------------------------------------------------------------------------------
# Static validation: re-reads OUR files from disk. Each check returns a list of problems.

def check_bitmaps(folder: Path) -> list[str]:
    problems: list[str] = []
    try:
        provinces = bmpio.read_bmp(folder / "provinces.bmp")
    except (ArenaError, OSError) as exc:
        return [f"provinces.bmp: {exc}"]
    width, height = provinces.width, provinces.height
    if provinces.bits != 24 or provinces.header_size != 40 or provinces.compression:
        problems.append(f"provinces.bmp must be 24-bit uncompressed BITMAPINFOHEADER, found {provinces.bits} bit "
                        f"(header {provinces.header_size}): the game refuses other bit depths")
    if width % 256 or height % 256:
        problems.append(f"provinces.bmp is {width}x{height}: both must be multiples of 256")
    expected = {"heightmap": (width, height), "terrain": (width, height), "rivers": (width, height),
                "cities": (width, height), "trees": (width * 75 // 256, height * 75 // 256)}
    for name, size in expected.items():
        try:
            image = bmpio.read_bmp(folder / f"{name}.bmp")
        except (ArenaError, OSError) as exc:
            problems.append(f"{name}.bmp: {exc}")
            continue
        if image.bits != 8 or image.compression or len(image.palette) != 256:
            problems.append(f"{name}.bmp must be 8-bit indexed with 256 palette entries")
        if (image.width, image.height) != size:
            problems.append(f"{name}.bmp is {image.width}x{image.height}, expected {size[0]}x{size[1]}")
        if name == "heightmap" and image.bits == 8 and list(image.palette) != bmpio.grey_palette():
            problems.append("heightmap.bmp palette is not the greyscale ramp")
        if name in bmpio.FALLBACK_PALETTES and image.bits == 8:
            wrong = [i for i, rgb in bmpio.FALLBACK_PALETTES[name].items()
                     if i < len(image.palette) and image.palette[i] != rgb and i != 14]
            if wrong:
                problems.append(f"{name}.bmp palette differs from vanilla at indices {wrong}")
    try:
        normal = bmpio.read_bmp(folder / "world_normal.bmp")
        if normal.bits != 24 or (normal.width, normal.height) != (width // 2, height // 2):
            problems.append("world_normal.bmp must be 24-bit and half the province bitmap in each dimension")
    except (ArenaError, OSError) as exc:
        problems.append(f"world_normal.bmp: {exc}")
    for name, size, fourcc in (("colormap_rgb_cityemissivemask_a", (width // 2, height // 2), ""),
                               ("colormap_water_0", (width // 2, height // 2), "DXT5"),
                               ("colormap_water_1", (width // 4, height // 4), "DXT5"),
                               ("colormap_water_2", (width // 8, height // 8), "DXT5")):
        try:
            head = bmpio.read_dds_header(folder / "terrain" / f"{name}.dds")
        except (ArenaError, OSError) as exc:
            problems.append(f"{name}.dds: {exc}")
            continue
        want = size[0] * size[1] * (1 if fourcc else 4)
        if (head["width"], head["height"]) != size or head["fourcc"] != fourcc or head["payload"] != want:
            problems.append(f"{name}.dds has the wrong size, format or payload: {head}")
    return problems


def read_definitions(folder: Path) -> tuple[list[tuple[int, tuple[int, int, int], str, bool, str, int]], list[str]]:
    problems: list[str] = []
    raw = (folder / "definition.csv").read_bytes()
    if raw.count(b"\n") != raw.count(b"\r\n") or not raw.endswith(b"\r\n"):
        problems.append("definition.csv must use CRLF line endings on every line")
    rows = []
    for number, line in enumerate(raw.decode("ascii", errors="replace").splitlines()):
        parts = line.split(";")
        if len(parts) != 8 or not parts[0].isdigit():
            problems.append(f"definition.csv line {number + 1} is malformed: {line!r}")
            continue
        rows.append((int(parts[0]), (int(parts[1]), int(parts[2]), int(parts[3])), parts[4], parts[5] == "true",
                     parts[6], int(parts[7])))
    if [row[0] for row in rows] != list(range(len(rows))):
        problems.append("definition.csv ids must be sequential from 0 without gaps or duplicates")
    if len({row[1] for row in rows}) != len(rows):
        problems.append("definition.csv has duplicate province colours")
    for province_id, _, kind, _, terrain, continent in rows[1:]:
        if kind not in ("land", "sea", "lake") or (kind == "land") != (continent > 0 and kind == "land"):
            problems.append(f"province {province_id}: land needs a continent and the type must be land|sea|lake")
        if kind == "sea" and (continent != 0 or terrain != "ocean"):
            problems.append(f"province {province_id}: sea must be ocean on continent 0")
        if continent > len(CONTINENTS):
            problems.append(f"province {province_id}: continent {continent} is not in continent.txt")
    return rows, problems


def province_raster(folder: Path) -> tuple[Any, list[str]]:
    """provinces.bmp as an id raster through definition.csv (0 where a colour is undefined)."""
    rows, problems = read_definitions(folder)
    image = bmpio.read_bmp(folder / "provinces.bmp")
    if image.bits != 24:
        return None, [*problems, "provinces.bmp is not 24-bit"]
    rgb = image.pixels.astype(np.uint32)
    keys = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    table = np.array(sorted(((r << 16) | (g << 8) | b, province_id) for province_id, (r, g, b), *_ in rows),
                     dtype=np.int64)
    index = np.clip(np.searchsorted(table[:, 0], keys), 0, len(table) - 1)
    ids = np.where(table[index, 0] == keys, table[index, 1], -1).astype(np.int32)
    if (ids < 0).any():
        problems.append(f"provinces.bmp has {int((ids < 0).sum())} pixels whose colour is not in definition.csv")
    if (ids == 0).any():
        problems.append("provinces.bmp uses the colour of the reserved province 0")
    missing = sorted(set(range(1, len(rows))) - set(np.unique(ids).tolist()))
    if missing:
        problems.append(f"provinces without a single pixel: {missing[:10]}")
    return ids, problems


def check_provinces(ids: Any) -> list[str]:
    problems: list[str] = []
    height, width = ids.shape
    crossings = x_crossings(np.concatenate([ids, ids[:, :1]], axis=1))  # the map wraps horizontally
    if len(crossings):
        problems.append(f"{len(crossings)} X crossings (four provinces meet), first at x={int(crossings[0][1])} "
                        f"y={int(crossings[0][0])}")
    counts = np.bincount(ids[ids >= 0].ravel())
    small = [i for i, n in enumerate(counts.tolist()) if 0 < n < MIN_PIXELS]
    if small:
        problems.append(f"provinces smaller than {MIN_PIXELS} pixels: {small[:10]}")
    ys, xs = np.indices(ids.shape)
    flat = ids.ravel()
    size = int(flat.max()) + 1
    for name, coords, limit in (("wide", xs.ravel(), width // 8), ("tall", ys.ravel(), height // 8)):
        low = np.full(size, 1 << 30)
        high = np.full(size, -1)
        np.minimum.at(low, flat, coords)
        np.maximum.at(high, flat, coords)
        big = [i for i in range(1, size) if high[i] - low[i] + 1 >= limit]
        if big:
            problems.append(f"provinces too {name} for the 1/8 bounding-box rule (limit {limit}): {big[:10]}")
    return problems


def _blocks(value: Any) -> list[Any]:
    return value if isinstance(value, list) and value and isinstance(value[0], (dict, list)) else [value]


def check_membership(mod: Path, definitions: list[tuple[int, Any, str, bool, str, int]]) -> list[str]:
    """Every land province in exactly one state, every province in exactly one strategic region."""
    problems: list[str] = []
    kind = {row[0]: row[2] for row in definitions[1:]}
    in_state: dict[int, int] = {}
    state_ids = []
    for path in sorted((mod / "history" / "states").glob("*.txt")):
        state = parse(path.read_text(encoding="utf-8")).get("state", {})
        state_ids.append(state.get("id"))
        for province in state.get("provinces", []):
            if province in in_state:
                problems.append(f"province {province} is in states {in_state[province]} and {state.get('id')}")
            in_state[province] = state.get("id")
            if kind.get(province) == "sea" or province not in kind:
                problems.append(f"state {state.get('id')} lists province {province}, which is sea or undefined")
    if sorted(state_ids) != list(range(1, len(state_ids) + 1)):
        problems.append(f"state ids must be sequential from 1: {sorted(state_ids)}")
    homeless = sorted(p for p, k in kind.items() if k != "sea" and p not in in_state)
    if homeless:
        problems.append(f"land provinces without a state: {homeless[:10]}")
    in_region: dict[int, int] = {}
    for path in sorted((mod / "map" / "strategicregions").glob("*.txt")):
        region = parse(path.read_text(encoding="utf-8")).get("strategic_region", {})
        periods = _blocks(region.get("weather", {}).get("period", []))
        if len(periods) != 12 or any(len(p.get("between", [])) != 2 for p in periods if isinstance(p, dict)):
            problems.append(f"{path.name}: needs twelve weather periods with a 'between' pair")
        for province in region.get("provinces", []):
            if province in in_region:
                problems.append(f"province {province} is in strategic regions {in_region[province]} and {region['id']}")
            in_region[province] = region.get("id")
    lost = sorted(set(kind) - set(in_region))
    if lost:
        problems.append(f"provinces without a strategic region (crash risk): {lost[:10]}")
    if set(in_region) - set(kind):
        problems.append("a strategic region lists an undefined province")
    return problems


def check_text_files(folder: Path, definitions: list[tuple[int, Any, str, bool, str, int]], ids: Any) -> list[str]:
    """adjacencies closing line, supply/railway graph, buildings and unitstacks ids and positions."""
    problems: list[str] = []
    kind = {row[0]: row[2] for row in definitions[1:]}
    height, width = ids.shape
    lines = (folder / "adjacencies.csv").read_text(encoding="utf-8").splitlines()
    data = [line for line in lines if line.strip() and not line.startswith("#")]
    if not data or data[-1].replace(" ", "").split(";")[:2] != ["-1", "-1"] or data[-1].count(";") != 8:
        problems.append(f"adjacencies.csv must end with the closing line {CLOSING_LINE}")
    for line in data[1:-1]:
        parts = line.split(";")
        if len(parts) != 10 or not all(p.isdigit() and int(p) in kind for p in parts[:2]):
            problems.append(f"adjacencies.csv row is malformed or names an unknown province: {line}")
    touching = set(border_lengths(ids))
    hubs = [int(line.split()[1]) for line in (folder / "supply_nodes.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not hubs or any(kind.get(hub) != "land" for hub in hubs) or len(set(hubs)) != len(hubs):
        problems.append("supply_nodes.txt needs unique hubs on land provinces")
    for line in (folder / "railways.txt").read_text(encoding="utf-8").splitlines():
        numbers = [int(token) for token in line.split()]
        if len(numbers) < 4 or numbers[1] != len(numbers) - 2 or not 1 <= numbers[0] <= 5:
            problems.append(f"railways.txt row is malformed: {line}")
            continue
        chain = numbers[2:]
        if any(kind.get(p) != "land" for p in chain):
            problems.append(f"railway runs over a non-land province: {line}")
        problems += [f"railway provinces {a} and {b} do not touch" for a, b in zip(chain, chain[1:], strict=False)
                     if (min(a, b), max(a, b)) not in touching]
    buildings = (folder / "buildings.txt").read_text(encoding="utf-8").splitlines()
    if not buildings:
        problems.append("buildings.txt is empty")
    state_of = province_states(folder.parent)
    states = set(state_of.values())
    bad = 0
    for line in buildings:
        parts = line.split(";")
        x, z = float(parts[2]), float(parts[4])
        inside = 0 <= x < width and 0 < z <= height
        here = int(ids[int(height - z), int(x)]) if inside else -1
        sea_ok = parts[6] == "0" or kind.get(int(parts[6])) == "sea"
        where_ok = kind.get(here) == "sea" if parts[1] == "floating_harbor" else state_of.get(here) == int(parts[0])
        bad += not (len(parts) == 7 and int(parts[0]) in states and inside and sea_ok and where_ok)
    if bad:
        problems.append(f"buildings.txt: {bad} rows with an unknown state, a position outside it or a bad sea province")
    seen: set[int] = set()
    bad = 0
    for line in (folder / "unitstacks.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split(";")
        x, z = float(parts[2]), float(parts[4])
        inside = 0 <= x < width and 0 < z <= height
        bad += not (len(parts) == 7 and 0 <= int(parts[1]) <= 38 and inside
                    and int(ids[int(height - z), int(x)]) == int(parts[0]))
        if parts[1] == "0":
            seen.add(int(parts[0]))
    if bad:
        problems.append(f"unitstacks.txt: {bad} rows with a bad type or a position outside their province")
    if seen != set(kind):
        problems.append("unitstacks.txt needs a standstill position (type 0) for every province")
    return problems


def province_states(mod: Path) -> dict[int, int]:
    out = {}
    for path in sorted((mod / "history" / "states").glob("*.txt")):
        state = parse(path.read_text(encoding="utf-8")).get("state", {})
        out.update({province: state.get("id") for province in state.get("provinces", [])})
    return out


def check_rivers(folder: Path, definitions: list[tuple[int, Any, str, bool, str, int]], ids: Any) -> list[str]:
    """Palette indices, one-pixel-thick orthogonal paths, exactly one green source per river, on land."""
    problems: list[str] = []
    rivers = bmpio.read_bmp(folder / "rivers.bmp").pixels
    if rivers.shape != ids.shape:
        return ["rivers.bmp size differs from provinces.bmp"]
    if np.isin(rivers, list(range(12, 254))).any():
        problems.append("rivers.bmp uses palette indices outside 0-11 and 254/255")
    wet = rivers <= 11
    sea = np.array([False] + [row[2] == "sea" for row in definitions[1:]])
    if (wet & sea[ids]).any():
        problems.append("a river runs over sea pixels")
    if ((rivers == 255) & sea[ids]).any() or ((rivers == 254) & ~sea[ids]).any():
        problems.append("rivers.bmp land (255) / sea (254) background does not match the provinces")
    if (wet[:-1, :-1] & wet[:-1, 1:] & wet[1:, :-1] & wet[1:, 1:]).any():
        problems.append("a river is more than one pixel thick (2x2 block)")
    comp_wet = np.where(wet, 1, 0).astype(np.int32)
    from .mapdesign import components
    comp, labels, _ = components(comp_wet)
    for index, label in enumerate(labels):
        if label:
            mask = comp == index
            sources = int((rivers[mask] == 0).sum())
            if sources != 1:
                problems.append(f"a river has {sources} source pixels (needs exactly one green source)")
            degree = np.zeros(ids.shape, dtype=np.int8)
            degree[1:] += mask[:-1]
            degree[:-1] += mask[1:]
            degree[:, 1:] += mask[:, :-1]
            degree[:, :-1] += mask[:, 1:]
            if int(((degree == 1) & mask).sum()) != 2 or ((degree > 2) & mask).any():
                problems.append("a river is not a simple orthogonal path (it branches or loops without markers)")
    diagonal = (wet[:-1, :-1] & wet[1:, 1:] & ~wet[:-1, 1:] & ~wet[1:, :-1]) | \
               (wet[:-1, 1:] & wet[1:, :-1] & ~wet[:-1, :-1] & ~wet[1:, 1:])
    if diagonal.any():
        problems.append("river pixels touch only diagonally (rivers must be orthogonally connected)")
    return problems


def check_symmetry(output: Path, ids: Any, layout: ArenaLayout) -> list[str]:
    """Pixel symmetry up to the declared repair pixels, and EXACT symmetry of the gameplay graph."""
    problems: list[str] = []
    meta = json.loads((output / "arena_map.json").read_text(encoding="utf-8"))
    mirror = meta["symmetry"] == "mirror"
    partner = np.zeros(len(meta["provinces"]) + 1, dtype=np.int32)
    for row in meta["provinces"]:
        partner[row["id"]] = row["partner"]
    image = ids[:, ::-1] if mirror else ids[::-1, ::-1]
    differ = np.argwhere(partner[np.clip(image, 0, len(partner) - 1)] != ids)
    height, width = ids.shape
    allowed = set()
    for x, y in meta["repairs"]:
        allowed |= {(x, y), (width - 1 - x, y) if mirror else (width - 1 - x, height - 1 - y)}
    stray = [(int(x), int(y)) for y, x in differ.tolist() if (int(x), int(y)) not in allowed]
    if stray:
        problems.append(f"{len(stray)} asymmetric pixels outside the declared repairs, first {stray[:5]}")
    by_id = {p.id: p for p in layout.provinces}
    flip = {"north": "south", "south": "north", "center": "center"}
    for p in layout.provinces:
        other = by_id.get(int(partner[p.id]))
        if other is None:
            problems.append(f"province {p.id} has no playable counterpart")
            continue
        facts = [
            ("adjacency", sorted(int(partner[n]) for n in p.neighbors) == sorted(other.neighbors)),
            ("river crossings", sorted(int(partner[n]) for n in p.river_neighbors) == sorted(other.river_neighbors)),
            ("terrain", p.terrain == other.terrain), ("victory points", p.victory_points == other.victory_points),
            ("sector", other.sector == (p.sector if mirror else flip[p.sector])),
            ("owner", p.initial_controller != other.initial_controller),
            ("capital", bool(p.capital_of) == bool(other.capital_of)),
            ("position", abs(p.x + other.x - 1) < 1e-5 and abs((p.y - other.y) if mirror else (p.y + other.y - 1)) < 1e-5),
        ]
        problems += [f"province {p.id} and its counterpart {other.id} differ in {name}" for name, ok in facts if not ok]
    return problems


def check_routes(layout: ArenaLayout, expected: int | None) -> list[str]:
    """Connectivity, and the per-design number of separate approach routes: every route alone must still
    join the capitals, and without any route the two sides must be disconnected."""
    problems: list[str] = []
    side = {p.id: p.initial_controller for p in layout.provinces}
    neighbors = {p.id: p.neighbors for p in layout.provinces}
    front = {p for p in side if any(side[n] != side[p] for n in neighbors[p])}
    groups: list[set[int]] = []
    for start in sorted(front):
        if any(start in group for group in groups):
            continue
        group, queue = {start}, deque([start])
        while queue:
            for nxt in neighbors[queue.popleft()]:
                if nxt in front and nxt not in group:
                    group.add(nxt)
                    queue.append(nxt)
        groups.append(group)

    def joined(open_group: set[int] | None) -> bool:
        start, goal = layout.capital(Country.BLUE).id, layout.capital(Country.RED).id
        seen, queue = {start}, deque([start])
        while queue:
            here = queue.popleft()
            for nxt in neighbors[here]:
                crossing = side[nxt] != side[here]
                if nxt not in seen and (not crossing or (open_group is not None and {here, nxt} <= open_group)):
                    seen.add(nxt)
                    queue.append(nxt)
        return goal in seen

    if joined(None):
        problems.append("the sides are connected without crossing the front (ownership is wrong)")
    problems += [f"route {sorted(group)} alone does not connect the capitals" for group in groups if not joined(group)]
    if expected is not None and len(groups) != expected:
        problems.append(f"expected {expected} separate routes, found {len(groups)}: {[sorted(g) for g in groups]}")
    sectors = {layout.province(p).sector for p in front}
    if expected == 3 and sectors != set(SECTORS):
        problems.append(f"three routes must be one per sector, the front only covers {sorted(sectors)}")
    if {p.sector for p in layout.provinces} != set(SECTORS):
        problems.append("the layout must use all of north/center/south")
    return problems


def check_scripts(output: Path, mod: Path, layout: ArenaLayout, manifest: dict[str, Any],
                  game: Path | None) -> tuple[list[str], int, int]:
    problems: list[str] = []
    parsed: dict[str, Any] = {}
    for path in sorted(mod.rglob("*.txt")):
        relative = path.relative_to(mod).as_posix()
        if relative == "common/countries/colors.txt" or relative.startswith("map/") and "strategicregions" not in relative:
            continue
        text = path.read_text(encoding="utf-8")
        try:
            parsed[relative] = parse(text)
        except Exception as exc:  # noqa: BLE001 - every parser failure is a finding
            problems.append(f"{relative}: does not parse: {exc}")
        if not _balanced(text):
            problems.append(f"{relative}: unbalanced braces")
    owner = {p.id: p.initial_controller for p in layout.provinces}
    for scenario in manifest["scenarios"]:
        counts = {}
        for side in ("BLU", "RED"):
            for level in manifest["handicaps"]:
                name = f"history/units/ARENA_{scenario['index']:02d}_{side}_{level}.txt"
                units = parsed.get(name, {}).get("units", {}).get("division", [])
                units = units if isinstance(units, list) else [units]
                counts[side, level] = len(units)
                problems += [f"{name}: division starts outside its own land at {unit['location']}"
                             for unit in units if owner.get(unit["location"]) != side]
                problems += [f"{name}: unknown template" for unit in units
                             if unit["division_template"] not in (INFANTRY, ARMOR)]
        problems += [f"{scenario['id']}: unequal or empty division counts at handicap {level}"
                     for level in manifest["handicaps"] if counts["BLU", level] != counts["RED", level]
                     or counts["BLU", level] < 1]
    descriptor = (mod / "descriptor.mod").read_text(encoding="utf-8")
    missing = [folder for folder in REPLACE_PATHS if f'replace_path="{folder}"' not in descriptor]
    if missing:
        problems.append(f"descriptor.mod lacks replace_path for {missing}")
    if manifest.get("debug_bisect"):
        problems.append(f"this is a DEBUG bisect build, not a valid arena: {manifest['debug_bisect']}")
    blanks = manifest.get("neutralised_vanilla_files", {}).get("blank", [])
    problems += [f"neutralised vanilla file is missing or not blank: {relative}" for relative in blanks
                 if not (mod / relative).is_file() or (mod / relative).read_text(encoding="utf-8") != BLANK_TEXT]
    checked = 0
    if game is not None:
        found, checked = check_script(parsed, game)
        problems += found
        if not (game / "map" / "seasons.txt").is_file() or not (game / "map" / "cities.txt").is_file():
            problems.append("the install lacks map/seasons.txt or map/cities.txt, which this mod leaves to vanilla")
    return problems, len(parsed), checked


def validate_custom_map(output: Path, game: Path | None = None) -> dict[str, Any]:
    """All static checks for one built design. ``game=None`` skips the checks that read the install."""
    output = output.resolve()
    manifest = json.loads((output / "arena_manifest.json").read_text(encoding="utf-8"))
    mod = output / manifest["mod_dir"]
    folder = mod / "map"
    checks: dict[str, list[str]] = {"bitmaps": check_bitmaps(folder)}
    ids, found = province_raster(folder)
    definitions, _ = read_definitions(folder)
    checks["definitions"] = found
    layout = ArenaLayout.load(output / "arena_layout.json")
    if ids is not None and not found:
        checks["provinces"] = check_provinces(ids)
        checks["membership"] = check_membership(mod, definitions)
        checks["text_files"] = check_text_files(folder, definitions, ids)
        checks["rivers"] = check_rivers(folder, definitions, ids)
        checks["symmetry"] = check_symmetry(output, ids, layout)
        game_ids = {row[0] for row in definitions[1:] if row[2] == "land"}
        touching = set(border_lengths(ids))
        checks["layout"] = [f"layout province {p.id} is not a land province of definition.csv"
                            for p in layout.provinces if p.id not in game_ids]
        checks["layout"] += [f"layout link {p.id}-{n} is not a pixel adjacency" for p in layout.provinces
                             for n in p.neighbors if (min(p.id, n), max(p.id, n)) not in touching]
        if layout.source != "custom_map" or any(not set(p.river_neighbors) <= set(p.neighbors) for p in layout.provinces):
            checks["layout"].append("layout source must be custom_map and river crossings must be adjacencies")
    checks["routes"] = check_routes(layout, manifest["expected_routes"])
    checks["scripts"], files_parsed, names_checked = check_scripts(output, mod, layout, manifest, game)
    checks["hash"] = [] if tree_hash(mod) == manifest["mod_sha256"] else ["mod folder does not match the manifest hash"]
    problems = [f"{name}: {problem}" for name, found in checks.items() for problem in found]
    return {"ok": not problems, "design": manifest["design"], "problems": problems,
            "checks": {name: len(found) for name, found in checks.items()}, "files_parsed": files_parsed,
            "names_checked": names_checked, "install_checks": game is not None, "mod_sha256": manifest["mod_sha256"],
            "evidence_kind": "static_validation_not_a_game_load"}


# ---------------------------------------------------------------------------------------------
# Install scan behind REPLACE_PATHS, and FIRST_LAUNCH.md.

_ID_REFERENCE = re.compile(
    r"\b(state|province|owns_state|controls_state|has_full_control_of_state|transfer_state|add_state_core|capital|"
    r"controls_province|strategic_region|region|area|location|provinces|states|set_province_controller|"
    r"strategic_regions)\s*=\s*\{?\s*\d{1,5}\b|^\s*\d{1,5}\s*=\s*\{", re.M)


def scan_id_references(game: Path) -> dict[str, dict[str, int]]:
    """Per folder of the install: files and numeric state/province/region references. The evidence the
    ``REPLACE_PATHS`` table was derived from; slow (reads ~5000 files), so it is a separate command."""
    out: dict[str, dict[str, int]] = {}
    for top in ("common", "events", "history", "map"):
        for path in sorted((game / top).rglob("*.txt")):
            parts = path.relative_to(game).parts
            key = "/".join(parts[:3] if len(parts) > 3 else parts[:2] if len(parts) > 2 else parts[:1])
            text = re.sub(r"#[^\n]*", "", path.read_text(encoding="utf-8-sig", errors="replace"))
            row = out.setdefault(key, {"files": 0, "id_references": 0})
            row["files"] += 1
            row["id_references"] += len(_ID_REFERENCE.findall(text))
    return {key: row for key, row in sorted(out.items()) if row["id_references"]}


def first_launch_text(design: MapDesign, manifest: dict[str, Any]) -> str:
    name = mod_dir(design)
    paths = "\n".join(f"- `{folder}`: {why}" for folder, why in REPLACE_PATHS.items())
    left = "\n".join(f"- {what}: {why}" for what, why in LEFT_ALONE.items())
    return f"""# First launch of the generated arena map `{design.name}`

Status: GENERATED AND STATICALLY VALIDATED ONLY. The game has never loaded this mod. Expect to iterate.
Mod hash at build time: `{manifest['mod_sha256']}` ({manifest['provinces']} provinces, symmetry `{design.symmetry}`).

## What live launches have shown so far (three_lanes, v1.19.3, 8 launches on 2026-09-19, all crashed)
REAL game evidence, from `logs/` and `crashes/` of the three_lanes profile:
- The map itself loads: `game.log` "Loaded 397 provinces", `setup.log` "Calculated 1 land masses", our on_actions
  are registered, and error.log never contained a `MAP_ERROR`. definition.csv with CRLF was accepted.
- Found and fixed in the generator: the map WRAPS horizontally, so the mirrored ocean grid met itself at
  x = width - 1 ("Map invalid X crossing. Please fix pixels at coords: 2047,<y>", 19 lines). The seam is now
  repaired like the centre line and the validator scans the wrap column; launch 3 onwards logged none.
- OPEN: `EXCEPTION_ACCESS_VIOLATION` about one second after the on_actions are registered, always at the same
  address (hoi4.exe + 0x2D73BD), on a shallow worker thread, before `system.log` reaches its "Version:" line and
  before `game.log` says "Resetting game" (so before any history executes). It is independent of: the amount of
  vanilla script errors (20151 lines with nothing blanked, 3746 and 1995 with the blank plan), railways, supply
  nodes, adjacencies, ai_areas, buildings, unitstacks, weatherpositions, ambient objects, our colour-map DDS
  files, dynamic country tags, and `-start_tag`.
- Best remaining lead (untested: the launch was refused by the permission system): vanilla databases keyed by
  country tag stop parsing at the first unknown tag, e.g. `gfx/interface/equipmentdesigner/graphic_db/*.txt`
  ("Expected 'default', a continent name or a country tag: GER"), `gfx/train_gfx_database`, `common/units/names`.
  Stack fragments of the crashing thread read "...pment" and "Tag". Next experiment: keep vanilla's tag list so
  these files parse: `custom-map-build --design three_lanes --force --bisect keep:common/country_tags,dynamic_tags`.
  If that loads, make it the default (drop `common/country_tags` from REPLACE_PATHS; vanilla countries own no
  state and therefore do not exist) and shrink the blank plan accordingly. After that, bisect `history/states`
  (one state per side), the bookmark, and the remaining replace_path entries one by one with `keep:<folder>`.
- Not reached yet, so still unknown: main menu, `-start_tag=BLU`, ARENA_* log lines, supply, the impassable infill,
  rivers, building and unit-stack positions in game.

## Procedure
1. Use the isolated profile, not your Documents folder: point `gameDataPath` of a COPY of
   `launcher-settings.json` at this directory (see `hoi4_agent/arena/probe.py`); this directory already
   holds `dlc_load.json`, `settings.txt` and `mod/{name}.mod`. Do not enable any other mod.
2. Start `hoi4.exe -debug -nolauncher -mod=mod/{name}.mod` from the game directory. `-debug` turns
   MAP_ERRORs from a silent refusal into log lines and enables the console and the nudger.
3. Read, in this order, under `<profile>/logs/`: `error.log` (search `MAP_ERROR`, then `Error`),
   `game.log` (last lines before a crash name the loader stage), `setup.log`, `system.log`; crash dumps land
   in `<profile>/crashes/`. Some MAP_ERRORs only appear after a country was selected and the map loaded.
4. In the menu pick the only bookmark, country BLU. After load, check the log for `ARENA_STARTUP`,
   `ARENA_RESET` and one `ARENA_TICK` per day (grammar: `arena_manifest.json`).
5. Console checks (`-debug`): `tdebug` (hover shows province and state ids: compare with
   `preview_political.png`), `nudge` (opens the nudger; see the last section), `observe`.

## Most likely failures, in the order I would expect them
The quoted error texts come from the wiki and from memory of other mods' logs, NOT from a run of this mod:
match them loosely (search for the key words), and trust the log over this list.
1. Crash or hang while "Loading map" before any log line: a bitmap header the engine rejects. Texts: `We do
   not support bitdepth at 32`, `Bitmap size ... not multiple of`. Check `custom-map-validate`; if it is clean,
   re-save ONE bitmap at a time from an image editor (8-bit indexed, keep the palette) to find the file.
2. `MAP_ERROR: Province X has TOO LARGE BOX. Perhaps pixels are spread around the world in provinces.bmp`:
   the 1/8 rule is stricter than assumed. Lower `MapDesign.scale` or raise `provinces_per_side`; sea cells
   are `width // 20` by `height // 20` in `mapdesign._half_labels`.
3. `MAP_ERROR: Province X has only N pixels` / `Province X has no pixels`: raise the cleanup minimums in
   `mapdesign.rasterize`.
4. `MAP_ERROR: ... invalid X crossing` (the text names a pixel): the centre-line repair missed a case;
   the validator's X-crossing scan uses the 2x2 rule only.
5. `Palette in rivers.bmp is probably not correct`: harmless per the wiki. `River ... has no source` or rivers
   drawn nowhere: check the source pixel (index 0) and that the river is a single orthogonal path.
6. `Province X is not in any strategic region` / crash when hovering a province: membership; the validator
   covers it, so suspect `replace_path="map/strategicregions"` not taking effect (path typo, launcher cache).
7. `Continent ... not defined` / `province has no continent`: definition.csv line endings. We write CRLF as the
   wiki demands, while vanilla 1.19.3 ships LF; flip the single `\\r\\n` in `custommap.write_map` if needed.
8. Crash right after the bookmark screen or when picking BLU: a vanilla database that lost its ids. Look
   at the LAST file named in `game.log`, add its folder to `REPLACE_PATHS` or supply a stub. Candidates:
   `common/ai_areas` (we replace it), `common/operations`, `common/raids`, `common/factions`, `common/game_rules`.
9. Hundreds of `Invalid state/tag/event` lines in `error.log` from vanilla `common/scripted_effects`,
   `scripted_triggers`, `ideas`: expected and harmless (see "left to vanilla"); they are why the arena's own
   lines should be grepped by the `ARENA_` prefix.
10. Supply is zero everywhere or `Supply node X is not connected`: hubs need a railway to the capital; we draw
    level-{RAIL_LEVEL} lines from each capital. Check `preview_supply.png`, then the state infrastructure in `ModConfig`.
11. Buildings/ports errors (`naval base ... not adjacent to sea`, `Building ... has no position`): positions are
    geometric guesses; regenerate them with the nudger (below).
12. Units spawn but counters overlap: increase `MapDesign.scale` (bigger provinces) and rebuild.
13. The infill is walkable or units path through it: `impassable = yes` did not apply (state file parse) or the
    one-pixel repair links are open; check `map/adjacencies.csv`. Fallback: `blocked_kind="lake"`.

## Honest unknowns
- Whether a world with TWO countries, {manifest['provinces']['play']} playable provinces and no other land
  loads at all. No hard engine minimum is documented; total conversions with few countries exist, none this small
  that I can cite.
- Whether the engine requires specific tags or databases (a default country, dynamic `D01`-style tags for civil
  wars, `common/countries/cosmetic.txt` entries). We ship only BLU and RED and leave `common/countries` to vanilla.
- Minimum sea: the arena is surrounded by ocean in four strategic regions with `naval_terrain`; neither country
  has ports, ships or convoys, so naval AI should idle. Untested.
- Coastal provinces exist on the outer rim (the ocean touches the land). With no navies nothing can land, but the
  AI may still garrison the coast. If that distorts play, surround the land with an impassable rim instead.
- River crossings in `arena_layout.json` are computed, not read from the game. Compare a few borders in game.
- The one-pixel centre-line repairs are declared `impassable` adjacencies. Whether the front line renderer or
  the AI's pathing shows artefacts there is untested.
- Multiplayer: both clients need byte-identical mod folders. The build is deterministic and `mod_sha256` in
  the manifest must match on both machines, but the in-game checksum also covers vanilla files that this mod
  leaves loaded, and whether the checksum is stable with `replace_path` on this beta build is unverified.
- The camera: the arena is about {int(design.scale * 0.6 * design.width)} map pixels wide. Whether the farthest
  zoom shows it on one screen with readable counters must be measured; `scale` is the knob.
- The nudger: open it with `nudge` in the console. Buildings tab -> "Validate All" / regenerate writes
  `map/buildings.txt`; Units tab writes `map/unitstacks.txt`; Supply tab writes `supply_nodes.txt`/`railways.txt`;
  Weather tab writes `weatherpositions.txt`. Output goes to the PROFILE's `map/` folder; copy it over the mod's
  files only deliberately (the mod hash changes) and prefer fixing the generator.

## replace_path (every entry unloads vanilla files that name vanilla ids)
{paths}

## Deliberately left to vanilla
{left}

## Neutralised vanilla files ({len(manifest['neutralised_vanilla_files']['blank'])} same-named comment-only overrides)
Derived by `neutralise.blank_plan` from the install (rule and full list: `neutralised_vanilla_files` in
`arena_manifest.json`). Launch 1 crashed in vanilla script validation without them.
"""


# ---------------------------------------------------------------------------------------------
# CLI.

def _designs(name: str) -> list[MapDesign]:
    if name == "all":
        return list(PRESETS.values())
    if name not in PRESETS:
        raise ArenaError(f"unknown design {name}; choose from {sorted(PRESETS)} or all")
    return [PRESETS[name]]


def add_commands(commands: Any) -> dict[str, Callable[[argparse.Namespace], int]]:
    def common(parser: Any) -> None:
        parser.add_argument("--design", default=DEFAULT_DESIGN, help=f"{', '.join(PRESETS)} or all")
        parser.add_argument("--game", type=Path, default=DEFAULT_GAME)
        parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="one sub-directory per design")
        parser.add_argument("--no-install", action="store_true", help="do not read the game install at all")

    build = commands.add_parser("custom-map-build", help="generate total-conversion arena map mods (no launch)")
    common(build)
    build.add_argument("--width", type=int, default=0, help="bitmap width, multiple of 256 (default: the design's)")
    build.add_argument("--height", type=int, default=0)
    build.add_argument("--provinces-per-side", type=int, default=0)
    build.add_argument("--seed", type=int, default=-1)
    build.add_argument("--infill", choices=("impassable", "lake"), default=None)
    build.add_argument("--player-box", action="store_true", help="add a detached staging island per side")
    build.add_argument("--force", action="store_true", help="replace previously generated arena maps")
    build.add_argument("--bisect", default="", help="DEBUG, comma separated: " + ", ".join(
        [*BISECT_FILES, *BISECT_REMOVE, "adjacencies", "no_blank", "no_rescue", "keep:<replace_path folder>"]))
    check = commands.add_parser("custom-map-validate", help="static checks of generated arena maps (no launch)")
    common(check)
    show = commands.add_parser("custom-map-preview", help="re-render the preview PNGs and the contact sheet")
    common(show)
    scan = commands.add_parser("custom-map-scan", help="count vanilla id references per folder (replace_path evidence)")
    scan.add_argument("--game", type=Path, default=DEFAULT_GAME)

    def tuned(design: MapDesign, args: argparse.Namespace) -> MapDesign:
        from dataclasses import replace
        changes: dict[str, Any] = {}
        if args.width or args.height:
            changes |= {"width": args.width or design.width, "height": args.height or design.height}
        if args.provinces_per_side:
            changes["provinces_per_side"] = args.provinces_per_side
        if args.seed >= 0:
            changes["seed"] = args.seed
        if args.infill:
            changes["blocked_kind"] = args.infill
        if args.player_box:
            changes["player_box"] = True
        return replace(design, **changes) if changes else design

    def run_build(args: argparse.Namespace) -> int:
        from .preview import contact_sheet
        game = None if args.no_install else args.game
        reports = []
        for design in _designs(args.design):
            manifest = build_custom_map(tuned(design, args), args.output / design.name, game, force=args.force,
                                        bisect=tuple(name for name in args.bisect.split(",") if name))
            report = validate_custom_map(args.output / design.name, game)
            reports.append({"design": design.name, "output": str((args.output / design.name).resolve()),
                            "mod_sha256": manifest["mod_sha256"], "provinces": manifest["provinces"],
                            "dataset_role": design.dataset_role, "tuned": design.tuned, "validation": report})
        sheet = contact_sheet(args.output)
        print(json.dumps({"builds": reports, "contact_sheet": str(sheet) if sheet else None}, indent=2))
        return 0 if all(r["validation"]["ok"] for r in reports) else 1

    def run_validate(args: argparse.Namespace) -> int:
        reports = [validate_custom_map(args.output / design.name, None if args.no_install else args.game)
                   for design in _designs(args.design)]
        print(json.dumps(reports, indent=2))
        return 0 if all(r["ok"] for r in reports) else 1

    def run_preview(args: argparse.Namespace) -> int:
        from .preview import contact_sheet, write_previews
        written = []
        for design in _designs(args.design):
            meta = json.loads((args.output / design.name / "arena_map.json").read_text(encoding="utf-8"))
            built = MapDesign(**{**meta["design_spec"], **_tuples(meta["design_spec"])})
            arena = rasterize(built)
            written += write_previews(arena, plan_custom_states(arena), arena.region(), args.output / design.name)
        sheet = contact_sheet(args.output)
        print(json.dumps({"written": [str(p) for p in written], "contact_sheet": str(sheet) if sheet else None}, indent=2))
        return 0

    def run_scan(args: argparse.Namespace) -> int:
        print(json.dumps({"id_references_by_folder": scan_id_references(args.game), "replace_path": REPLACE_PATHS},
                         indent=2))
        return 0

    return {"custom-map-build": run_build, "custom-map-validate": run_validate, "custom-map-preview": run_preview,
            "custom-map-scan": run_scan}


def _tuples(raw: dict[str, Any]) -> dict[str, Any]:
    """JSON lists back into the tuples and shapes a ``MapDesign`` holds."""
    from .mapdesign import River, Shape

    def shapes(rows: list[dict[str, Any]]) -> tuple[Shape, ...]:
        return tuple(Shape(**row) for row in rows)

    return {"land": shapes(raw["land"]), "cut": shapes(raw["cut"]), "blocked": shapes(raw["blocked"]),
            "capital": tuple(raw["capital"]), "victory_points": tuple(tuple(v) for v in raw["victory_points"]),
            "sector_cuts": tuple(raw["sector_cuts"]) if raw["sector_cuts"] else None,
            "rivers": tuple(River(tuple(tuple(p) for p in r["points"]), r["width"], r["center"]) for r in raw["rivers"]),
            "terrain_zones": tuple((name, Shape(**shape)) for name, shape in raw["terrain_zones"]),
            "supply_hubs": tuple(tuple(h) for h in raw["supply_hubs"]),
            "rails": tuple(tuple(tuple(p) for p in rail) for rail in raw["rails"])}
