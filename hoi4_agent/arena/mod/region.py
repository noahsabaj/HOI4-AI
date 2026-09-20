"""Extract a compact land region of the VANILLA map as the arena's static layout.

v0.1 does not ship a custom map: the arena is a set of vanilla states handed to two new
countries. This module only READS the install (``map/definition.csv``, ``map/provinces.bmp``,
``map/adjacencies.csv``, ``map/rivers.bmp``, ``map/strategicregions``, ``history/states``)
and derives, for the chosen states: land provinces, pixel centroids, terrain, true province
adjacency, river crossings, victory points, sides, sectors and a normalized ``ArenaLayout``.

Honesty caveats:
- Adjacency is a 4-neighbour pixel scan inside the region's bounding box plus the explicit rows
  of ``adjacencies.csv`` (``impassable`` removes a link, every other type adds one). Provinces that
  only touch diagonally are NOT adjacent here; the engine's own rule has not been compared live.
- River crossings are a HEURISTIC over ``rivers.bmp`` (a link is a crossing when enough of its
  border pixels touch a river pixel). It has not been checked against the game; treat
  ``river_neighbors`` as unverified until the integrator compares a few borders in game.
- Normalized coordinates are a flat, aspect-true fit of the region's pixel box into the view frame.
  The real camera is a tilted perspective; perception owns the mapping to screen pixels.
"""
from __future__ import annotations

import csv
import re
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ...clausewitz import parse
from ..contracts import ArenaError, Country
from ..layout import ArenaLayout, LayoutProvince

DEFAULT_GAME = Path("C:/Program Files (x86)/Steam/steamapps/common/Hearts of Iron IV")
SECTORS = ("north", "center", "south")
RIVER_MAX_INDEX = 11  # rivers.bmp palette: 0-11 are river pixels (source/flow markers + widths)
RIVER_MIN_TOUCHES = 2  # a river merely passing through one border pixel pair is not a crossing
RIVER_MIN_FRACTION = 0.25  # share of a border's pixel pairs that must touch the river


@dataclass(frozen=True)
class ProvinceDef:
    id: int
    rgb: tuple[int, int, int]
    kind: str  # land | sea | lake
    coastal: bool
    terrain: str
    continent: int


@dataclass(frozen=True)
class VanillaState:
    id: int
    file_name: str
    name: str
    owner: str
    provinces: tuple[int, ...]
    victory_points: tuple[tuple[int, float], ...]
    manpower: int
    category: str


@dataclass(frozen=True)
class RegionSpec:
    """Which vanilla states form the arena and how they are split. Everything is explicit."""
    scenario_id: str
    state_ids: tuple[int, ...]
    blue_provinces: tuple[int, ...]  # every other land province of the region is RED
    blue_capital: int
    red_capital: int
    victory_points: tuple[tuple[int, float], ...]  # arena VPs, capitals included
    margin: float = 0.06
    view_aspect: float = 16 / 9
    rationale: str = ""


# Default: the Hungarian plain. 50 land provinces in 8 vanilla states, no coast: 33 plains,
# 13 forest, 3 hills, 1 urban. The Danube and the Tisza both cross it north-south.
# BLU holds Transdanubia plus the strip between the rivers in the south; RED holds Budapest, the
# north-eastern forest and everything beyond the Tisza: 25 land provinces each. The resulting front
# gives three different approaches: NORTH a forested land border without a river, CENTER the
# Danube crossing into urban Budapest (the only vanilla supply hub), SOUTH a crossing of the Tisza
# on open plains. Capitals: Gyor (6720) and Debrecen (11659), each two provinces behind the front. It is not mirror
# symmetric (RED owns the hub, BLU has the deeper rear): fairness comes from side-swapped evaluation.
DEFAULT_REGION = RegionSpec(
    scenario_id="hungary_plain_v1",
    state_ids=(43, 83, 154, 155, 664, 973, 974, 975),
    blue_provinces=(682, 686, 701, 3670, 3680, 3683, 3700, 3716, 6561, 6670, 6685, 6700, 6703, 6720,
                    9624, 9643, 9661, 9676, 9690, 11610, 11625, 11627, 11630, 11663, 11679),
    blue_capital=6720, red_capital=11659,
    # Capitals 10 each; one contested front VP per side and sector, equal totals (21 : 21) at the start:
    # north 9690 | 3731 (land border), center 6703 | 9660 Budapest (Danube), south 6700 | 679 (Tisza).
    victory_points=((679, 3.0), (3731, 3.0), (6700, 3.0), (6703, 5.0), (6720, 10.0), (9660, 5.0), (9690, 3.0),
                    (11659, 10.0)),
    rationale="Hungarian plain: landlocked, flat, two river lines, 25 land provinces per side.",
)


@dataclass(frozen=True)
class RegionProvince:
    id: int
    vanilla_state: int
    terrain: str
    coastal: bool
    pixels: int
    map_x: float  # centroid in provinces.bmp pixels (origin top-left)
    map_y: float
    x: float  # normalized to the arena view frame
    y: float
    side: str
    sector: str


@dataclass(frozen=True)
class Region:
    spec: RegionSpec
    provinces: tuple[RegionProvince, ...]
    neighbors: dict[int, tuple[int, ...]]
    river_neighbors: dict[int, tuple[int, ...]]
    # (province, vanilla state, adjacent region land provinces) for lakes/sea listed in the states
    other_provinces: tuple[tuple[int, int, tuple[int, ...]], ...]
    states: tuple[VanillaState, ...]
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive) of the region's pixels
    frame: tuple[float, float, float, float]  # origin x, origin y, scale x, scale y (per map pixel)
    frame_offset: tuple[float, float]
    strategic_regions: tuple[tuple[int, tuple[int, ...]], ...]
    external_land_neighbors: tuple[int, ...]
    river_evidence: tuple[tuple[int, int, int, int], ...] = field(default=())  # a, b, border, touches

    def province(self, province_id: int) -> RegionProvince:
        for province in self.provinces:
            if province.id == province_id:
                return province
        raise ArenaError(f"province {province_id} is not in the region")

    def side(self, country: Country) -> tuple[RegionProvince, ...]:
        return tuple(p for p in self.provinces if p.side == country.value)

    def front(self, country: Country) -> tuple[RegionProvince, ...]:
        """Provinces of ``country`` that touch the other side, ordered by ID."""
        enemy = {p.id for p in self.side(country.opponent)}
        return tuple(p for p in self.side(country) if enemy & set(self.neighbors[p.id]))

    def layout(self) -> ArenaLayout:
        points = dict(self.spec.victory_points)
        capitals = {self.spec.blue_capital: Country.BLUE.value, self.spec.red_capital: Country.RED.value}
        return ArenaLayout(self.spec.scenario_id, tuple(
            LayoutProvince(p.id, p.x, p.y, p.terrain, self.neighbors[p.id], self.river_neighbors[p.id],
                           p.sector, float(points.get(p.id, 0.0)), p.side, capitals.get(p.id, ""))
            for p in self.provinces), source="vanilla_region")

    def to_json(self) -> dict[str, Any]:
        """Everything perception/the simulator may need beyond the layout. Deterministic."""
        return {
            "schema_version": 1, "scenario_id": self.spec.scenario_id, "source": "vanilla_region",
            "spec": asdict(self.spec), "bbox": list(self.bbox),
            "frame": {"origin": list(self.frame[:2]), "scale": list(self.frame[2:]), "offset": list(self.frame_offset),
                      "note": "normalized = (map_pixel - origin) * scale + offset; flat fit, not the camera"},
            "provinces": [asdict(p) for p in self.provinces],
            "other_provinces": [[p, s, list(adjacent)] for p, s, adjacent in self.other_provinces],
            "states": [asdict(state) for state in self.states],
            "strategic_regions": [[rid, list(provs)] for rid, provs in self.strategic_regions],
            "external_land_neighbors": list(self.external_land_neighbors), "asymmetry": asymmetry(self),
            "river_crossings": {"method": "heuristic_border_pixels_unverified",
                                "min_touches": RIVER_MIN_TOUCHES, "min_fraction": RIVER_MIN_FRACTION,
                                "evidence_a_b_border_touches": [list(row) for row in self.river_evidence]},
        }


def read_definitions(game: Path) -> dict[int, ProvinceDef]:
    path = game / "map" / "definition.csv"
    if not path.is_file():
        raise ArenaError(f"map definition is missing: {path}")
    out: dict[int, ProvinceDef] = {}
    with path.open(encoding="utf-8", errors="replace", newline="") as stream:
        for row in csv.reader(stream, delimiter=";"):
            if len(row) < 8 or not row[0].strip().isdigit():
                continue
            out[int(row[0])] = ProvinceDef(int(row[0]), (int(row[1]), int(row[2]), int(row[3])), row[4].strip(),
                                           row[5].strip().lower() == "true", row[6].strip(), int(row[7]))
    return out


def _pairs(value: Any) -> tuple[tuple[int, float], ...]:
    """``victory_points = { 9660 20 }``, possibly repeated, as (province, value) pairs."""
    if not value:
        return ()
    blocks = value if isinstance(value[0], list) else [value]
    return tuple(sorted((int(block[0]), float(block[1])) for block in blocks if len(block) >= 2))


def read_states(game: Path, state_ids: tuple[int, ...] | None = None) -> dict[int, VanillaState]:
    root = game / "history" / "states"
    if not root.is_dir():
        raise ArenaError(f"state history is missing: {root}")
    wanted = None if state_ids is None else set(state_ids)
    out: dict[int, VanillaState] = {}
    for path in sorted(root.glob("*.txt")):
        leading = re.match(r"\s*(\d+)", path.name)
        if wanted is not None and leading and int(leading.group(1)) not in wanted:
            continue  # vanilla names start with the state ID; files without one are always parsed
        data = parse(path.read_text(encoding="utf-8-sig", errors="replace")).get("state")
        if not isinstance(data, dict) or "id" not in data or (wanted is not None and data["id"] not in wanted):
            continue
        raw = data.get("history")
        history: dict[str, Any] = raw if isinstance(raw, dict) else {}
        provinces = data.get("provinces") or []
        out[int(data["id"])] = VanillaState(
            int(data["id"]), path.name, str(data.get("name", f"STATE_{data['id']}")),
            str(history.get("owner", "")), tuple(int(p) for p in provinces),
            _pairs(history.get("victory_points")), int(data.get("manpower", 0) or 0),
            str(data.get("state_category", "rural")))
    if wanted is not None and wanted - set(out):
        raise ArenaError(f"states missing from the install: {sorted(wanted - set(out))}")
    return out


def read_adjacency_rules(game: Path) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """(added links, removed links) from ``adjacencies.csv``; pairs are (low, high)."""
    added: set[tuple[int, int]] = set()
    removed: set[tuple[int, int]] = set()
    path = game / "map" / "adjacencies.csv"
    if not path.is_file():
        return added, removed
    with path.open(encoding="utf-8", errors="replace", newline="") as stream:
        for row in csv.reader(stream, delimiter=";"):
            if len(row) < 3 or not row[0].strip().lstrip("-").isdigit() or not row[1].strip().isdigit():
                continue
            a, b = int(row[0]), int(row[1])
            if a <= 0 or b <= 0 or a == b:
                continue
            (removed if row[2].strip().lower() == "impassable" else added).add((min(a, b), max(a, b)))
    return added - removed, removed


def strategic_region_files(game: Path) -> dict[str, tuple[int, tuple[int, ...]]]:
    """file name -> (region ID, provinces) for every vanilla ``map/strategicregions`` file."""
    out: dict[str, tuple[int, tuple[int, ...]]] = {}
    root = game / "map" / "strategicregions"
    for path in sorted(root.glob("*.txt")) if root.is_dir() else ():
        parsed = parse_strategic_region(path.read_text(encoding="utf-8-sig", errors="replace"))
        if parsed:
            out[path.name] = parsed
    return out


def parse_strategic_region(text: str) -> tuple[int, tuple[int, ...]] | None:
    clean = re.sub(r"#[^\n]*", "", text)
    rid = re.search(r"\bid\s*=\s*(\d+)", clean)
    block = re.search(r"\bprovinces\s*=\s*\{([^}]*)\}", clean)
    if not rid or not block:
        return None
    return int(rid.group(1)), tuple(int(tok) for tok in block.group(1).split() if tok.isdigit())


def read_strategic_regions(game: Path, provinces: set[int]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    root = game / "map" / "strategicregions"
    out = []
    for path in sorted(root.glob("*.txt")) if root.is_dir() else ():
        text = re.sub(r"#[^\n]*", "", path.read_text(encoding="utf-8-sig", errors="replace"))
        rid = re.search(r"\bid\s*=\s*(\d+)", text)
        block = re.search(r"\bprovinces\s*=\s*\{([^}]*)\}", text)
        if rid and block:
            inside = tuple(sorted(provinces & {int(tok) for tok in block.group(1).split() if tok.isdigit()}))
            if inside:
                out.append((int(rid.group(1)), inside))
    return tuple(sorted(out))


def province_raster(image: Path, definitions: dict[int, ProvinceDef],
                    wanted: set[int]) -> tuple[Any, tuple[int, int], int]:
    """(province-ID raster cropped to the wanted provinces' box plus one pixel, (x0, y0), map height)."""
    Image.MAX_IMAGE_PIXELS = None  # the vanilla map is 5632x2048; it is a trusted local file
    with Image.open(image) as handle:
        rgb = np.asarray(handle.convert("RGB"), dtype=np.uint32)
    keys = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    del rgb
    wanted_keys = np.array(sorted((definitions[p].rgb[0] << 16) | (definitions[p].rgb[1] << 8) |
                                  definitions[p].rgb[2] for p in wanted), dtype=np.uint32)
    mask = np.isin(keys, wanted_keys)
    if not mask.any():
        raise ArenaError("none of the region's provinces appear in provinces.bmp")
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    y0, y1 = max(int(rows[0]) - 1, 0), min(int(rows[-1]) + 2, keys.shape[0])
    x0, x1 = max(int(cols[0]) - 1, 0), min(int(cols[-1]) + 2, keys.shape[1])
    crop = keys[y0:y1, x0:x1]
    table = np.array(sorted(((d.rgb[0] << 16) | (d.rgb[1] << 8) | d.rgb[2], d.id) for d in definitions.values()),
                     dtype=np.int64)
    index = np.clip(np.searchsorted(table[:, 0], crop), 0, len(table) - 1)
    ids = np.where(table[index, 0] == crop, table[index, 1], 0).astype(np.int64)
    return ids, (x0, y0), int(keys.shape[0])


def _border_pairs(ids: Any, rivers: Any | None) -> dict[tuple[int, int], list[int]]:
    """(low, high) -> [border pixel pairs, pairs touching a river pixel]; 4-neighbour scan."""
    out: dict[tuple[int, int], list[int]] = {}
    base = int(ids.max()) + 1
    wet_map = np.zeros(ids.shape, dtype=bool) if rivers is None else rivers
    for a, b, wa, wb in ((ids[:, :-1], ids[:, 1:], wet_map[:, :-1], wet_map[:, 1:]),
                         (ids[:-1], ids[1:], wet_map[:-1], wet_map[1:])):
        differ = a != b
        code = np.minimum(a, b)[differ] * base + np.maximum(a, b)[differ]
        wet = (wa | wb)[differ]
        for slot, values in ((0, code), (1, code[wet])):
            unique, counts = np.unique(values, return_counts=True)
            for key, count in zip(unique.tolist(), counts.tolist(), strict=True):
                out.setdefault((key // base, key % base), [0, 0])[slot] += count
    return out


def assign_sectors(points: dict[int, tuple[float, float]], blue_capital: int, red_capital: int) -> dict[int, str]:
    """Three equal-count bands across the capital-to-capital axis, labelled north/center/south.

    Pure geometry: project every centroid on the perpendicular of the line between the capitals
    (oriented towards increasing y, i.e. south on the map) and cut the ordering into thirds.
    """
    ax, ay = points[blue_capital]
    bx, by = points[red_capital]
    nx, ny = -(by - ay), bx - ax
    if ny < 0 or (ny == 0 and nx < 0):
        nx, ny = -nx, -ny
    if nx == 0 and ny == 0:
        raise ArenaError("capitals must be different provinces")
    order = sorted(points, key=lambda p: (round(points[p][0] * nx + points[p][1] * ny, 9), p))
    out = {}
    for rank, province in enumerate(order):
        out[province] = SECTORS[min(2, rank * 3 // len(order))]
    return out


def _connected(nodes: set[int], neighbors: dict[int, tuple[int, ...]]) -> bool:
    if not nodes:
        return False
    start = min(nodes)
    seen, queue = {start}, deque([start])
    while queue:
        for nxt in neighbors[queue.popleft()]:
            if nxt in nodes and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen == nodes


def extract_region(game: Path, spec: RegionSpec) -> Region:
    definitions = read_definitions(game)
    states = read_states(game, spec.state_ids)
    members: dict[int, int] = {}
    for state in states.values():
        for province in state.provinces:
            if province not in definitions:
                raise ArenaError(f"state {state.id} lists unknown province {province}")
            if province in members:
                raise ArenaError(f"province {province} is in two states")
            members[province] = state.id
    land = {p for p in members if definitions[p].kind == "land"}
    blue = set(spec.blue_provinces)
    if not blue <= land or not land - blue:
        raise ArenaError("blue provinces must be a proper subset of the region's land provinces")
    if spec.blue_capital not in blue or spec.red_capital not in land - blue:
        raise ArenaError("each capital must lie on its own side")
    if not {p for p, _ in spec.victory_points} <= land or any(v <= 0 for _, v in spec.victory_points):
        raise ArenaError("arena victory points must be positive and on region land provinces")
    if not 0 <= spec.margin < 0.5 or spec.view_aspect <= 0:
        raise ArenaError("margin must be in [0, 0.5) and the view aspect positive")

    ids, (x0, y0), _ = province_raster(game / "map" / "provinces.bmp", definitions, set(members))
    rivers = None
    river_path = game / "map" / "rivers.bmp"
    if river_path.is_file():
        with Image.open(river_path) as handle:
            if handle.mode in ("P", "L") and handle.size[0] >= x0 + ids.shape[1] and handle.size[1] >= y0 + ids.shape[0]:
                rivers = np.asarray(handle.crop((x0, y0, x0 + ids.shape[1], y0 + ids.shape[0]))) <= RIVER_MAX_INDEX
    borders = _border_pairs(ids, rivers)
    added, removed = read_adjacency_rules(game)
    links = {pair for pair in borders if pair[0] in land and pair[1] in land and pair not in removed}
    links |= {pair for pair in added if pair[0] in land and pair[1] in land}
    wet = {pair for pair in links if pair in borders and borders[pair][1] >= RIVER_MIN_TOUCHES
           and borders[pair][1] >= RIVER_MIN_FRACTION * borders[pair][0]}
    neighbors = {p: tuple(sorted({b if a == p else a for a, b in links if p in (a, b)})) for p in land}
    crossings = {p: tuple(sorted({b if a == p else a for a, b in wet if p in (a, b)})) for p in land}
    if not _connected(land, neighbors):
        raise ArenaError("the region's land provinces are not one connected area")
    for name, group in (("blue", blue), ("red", land - blue)):
        if not _connected(group, neighbors):
            raise ArenaError(f"the {name} side is not one connected area")

    ys, xs = np.indices(ids.shape)
    flat = ids.ravel()
    size = int(flat.max()) + 1
    count = np.bincount(flat, minlength=size)
    sum_x, sum_y = np.bincount(flat, xs.ravel(), size), np.bincount(flat, ys.ravel(), size)
    inside = np.isin(ids, np.array(sorted(land)))
    rows, cols = np.flatnonzero(inside.any(axis=1)), np.flatnonzero(inside.any(axis=0))
    bx0, by0, bx1, by1 = int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
    # Aspect-true fit of the land's pixel box into the unit view frame, centred, with a margin.
    usable = 1.0 - 2 * spec.margin
    scale = min(usable * spec.view_aspect / (bx1 - bx0), usable / (by1 - by0))  # in view-height units
    sx, sy = scale / spec.view_aspect, scale
    off_x = (1.0 - (bx1 - bx0) * sx) / 2
    off_y = (1.0 - (by1 - by0) * sy) / 2
    centroid = {p: (float(sum_x[p] / count[p]) + 0.5, float(sum_y[p] / count[p]) + 0.5) for p in land}
    normal = {p: (round((cx - bx0) * sx + off_x, 6), round((cy - by0) * sy + off_y, 6))
              for p, (cx, cy) in centroid.items()}
    # Sectors use aspect-true distances (x scaled back to view-height units).
    sectors = assign_sectors({p: (x * spec.view_aspect, y) for p, (x, y) in normal.items()},
                             spec.blue_capital, spec.red_capital)
    provinces = tuple(RegionProvince(
        p, members[p], definitions[p].terrain, definitions[p].coastal, int(count[p]),
        round(centroid[p][0] + x0, 3), round(centroid[p][1] + y0, 3), normal[p][0], normal[p][1],
        Country.BLUE.value if p in blue else Country.RED.value, sectors[p]) for p in sorted(land))
    external = sorted({b if a in land else a for a, b in borders if (a in land) != (b in land)
                       and (b if a in land else a) in definitions
                       and definitions[b if a in land else a].kind == "land"} - set(members))
    evidence = tuple(sorted((a, b, borders[(a, b)][0], borders[(a, b)][1]) for a, b in links
                            if (a, b) in borders and borders[(a, b)][1]))
    return Region(
        spec, provinces, neighbors, crossings,
        tuple(sorted((p, s, tuple(sorted({b if a == p else a for a, b in borders if p in (a, b)} & land)))
                     for p, s in members.items() if p not in land)),
        tuple(states[s] for s in sorted(states)), (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0),
        (float(bx0 + x0), float(by0 + y0), round(sx, 9), round(sy, 9)), (round(off_x, 6), round(off_y, 6)),
        read_strategic_regions(game, land), tuple(external), evidence)


def asymmetry(region: Region) -> dict[str, Any]:
    """How unequal the two halves are: 0 = identical on every measure, 1 = maximally different.

    Each component is |blue - red| / (blue + red) (terrain: half the L1 distance between the two
    terrain mixes). ``score`` is their plain mean. It describes the static map only: supply hubs,
    railways and the engine's real river flags are not included.
    """
    def ratio(blue: float, red: float) -> float:
        return round(abs(blue - red) / (blue + red), 4) if blue + red else 0.0

    points = dict(region.spec.victory_points)
    capitals = {Country.BLUE: region.spec.blue_capital, Country.RED: region.spec.red_capital}
    raw: dict[str, dict[Country, float]] = {name: {} for name in (
        "provinces", "victory_points", "front_provinces", "river_links_inside", "capital_to_front_steps")}
    mix: dict[Country, Counter[str]] = {}
    for country in Country:
        mine = {p.id for p in region.side(country)}
        front = {p.id for p in region.front(country)}
        steps, queue = {capitals[country]: 0}, deque([capitals[country]])
        while queue:
            here = queue.popleft()
            for nxt in region.neighbors[here]:
                if nxt in mine and nxt not in steps:
                    steps[nxt] = steps[here] + 1
                    queue.append(nxt)
        raw["provinces"][country] = len(mine)
        raw["victory_points"][country] = sum(points.get(p, 0.0) for p in mine)
        raw["front_provinces"][country] = len(front)
        raw["river_links_inside"][country] = sum(len(set(region.river_neighbors[p]) & mine) for p in mine) / 2
        raw["capital_to_front_steps"][country] = min(steps[p] for p in front) if front else 0
        mix[country] = Counter(p.terrain for p in region.side(country))
    components = {name: ratio(values[Country.BLUE], values[Country.RED]) for name, values in raw.items()}
    blue_n, red_n = raw["provinces"][Country.BLUE], raw["provinces"][Country.RED]
    components["terrain_mix"] = round(sum(abs(mix[Country.BLUE][t] / blue_n - mix[Country.RED][t] / red_n)
                                          for t in set(mix[Country.BLUE]) | set(mix[Country.RED])) / 2, 4)
    return {"score": round(sum(components.values()) / len(components), 4), "components": components,
            "per_side": {name: {c.value: v for c, v in values.items()} for name, values in raw.items()},
            "terrain": {c.value: dict(sorted(mix[c].items())) for c in Country}}


def terrain_counts(region: Region) -> dict[str, int]:
    return dict(sorted(Counter(p.terrain for p in region.provinces).items()))
