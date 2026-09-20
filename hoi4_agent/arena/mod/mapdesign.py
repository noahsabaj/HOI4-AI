"""Design-agnostic arena maps: a ``MapDesign`` describes ONE HALF of a small world, this module turns it
into a symmetric province raster plus everything derived from it (adjacency, rivers, sectors, supply).

Pipeline (every stage after ``_half_labels`` knows nothing about the particular design):
masks of the west half -> anisotropic Voronoi + Lloyd relaxation for the playable provinces, a coarse
grid for blocked infill and ocean -> cleanup (stray fragments, undersized provinces, four-province
"X crossings") -> symmetry (``mirror`` or ``rotate180``) -> centre-line X repair -> sequential ids ->
adjacency from a 4-neighbour scan -> rivers -> sectors, capitals, victory points, hubs, railways.

Two facts worth knowing before trusting the output:
- EXACT pixel mirror symmetry and "no X crossings" contradict each other on the centre line: wherever a
  west border reaches it, its mirror image arrives at the same point and four provinces meet. Each such
  point is repaired by moving ONE pixel (``CustomMap.repairs``); the one-pixel adjacency this creates is
  declared ``impassable`` in ``adjacencies.csv`` so the GAMEPLAY graph stays exactly symmetric. The
  validators allow precisely these pixels and nothing else.
- River crossings are computed as the task brief describes the engine's rule (the straight path between
  two province centres touches a river pixel). Not compared with the live game yet.

Coordinates in a design are fractions of the whole bitmap (x right, y DOWN); only x < 0.5 is evaluated.
Nothing here is HOI4 data; the layout source is ``custom_map``.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..contracts import ArenaError, Country
from ..layout import ArenaLayout, LayoutProvince
from .region import SECTORS, Region, RegionProvince, RegionSpec, VanillaState

TERRAIN_INDEX = {"plains": 0, "forest": 1, "hills": 17, "mountain": 6, "marsh": 9}  # terrain.bmp palette
KINDS = ("play", "blocked", "box", "sea")
MIN_PIXELS = 8  # the engine's documented minimum province size


@dataclass(frozen=True)
class Shape:
    kind: str  # rect | ellipse, both given by their bounding box
    x0: float
    y0: float
    x1: float
    y1: float

    def mask(self, xs: Any, ys: Any) -> Any:
        if self.kind == "rect":
            return (xs >= self.x0) & (xs < self.x1) & (ys >= self.y0) & (ys < self.y1)
        if self.kind != "ellipse":
            raise ArenaError(f"unknown shape kind {self.kind}")
        rx, ry = (self.x1 - self.x0) / 2, (self.y1 - self.y0) / 2
        return ((xs - self.x0 - rx) / rx) ** 2 + ((ys - self.y0 - ry) / ry) ** 2 <= 1.0


def rect(x0: float, y0: float, x1: float, y1: float) -> Shape:
    return Shape("rect", x0, y0, x1, y1)


def ellipse(x0: float, y0: float, x1: float, y1: float) -> Shape:
    return Shape("ellipse", x0, y0, x1, y1)


@dataclass(frozen=True)
class River:
    points: tuple[tuple[float, float], ...]  # source first; drawn as an orthogonal staircase
    width: int = 6  # rivers.bmp palette index 3 (narrow) .. 11 (wide)
    center: bool = False  # drawn once on the centre line instead of once per half


@dataclass(frozen=True)
class MapDesign:
    name: str
    description: str
    seed: int
    land: tuple[Shape, ...]
    capital: tuple[float, float]
    victory_points: tuple[tuple[float, float, float], ...]  # x, y, value; the capital is added on top
    symmetry: str = "mirror"  # mirror | rotate180
    width: int = 2048
    height: int = 1024
    provinces_per_side: int = 24
    cut: tuple[Shape, ...] = ()  # removed from the land (ocean)
    blocked: tuple[Shape, ...] = ()  # land nobody can enter: the infill between routes
    blocked_kind: str = "impassable"  # impassable (land in an `impassable = yes` state) | lake
    sector_cuts: tuple[float, float] | None = None  # y fractions; None: terciles of the front, symmetrized
    rivers: tuple[River, ...] = ()
    terrain_zones: tuple[tuple[str, Shape], ...] = ()  # first match wins, default plains
    supply_hubs: tuple[tuple[float, float], ...] = ()  # besides the capital
    rails: tuple[tuple[tuple[float, float], ...], ...] = ()  # waypoints after the capital
    rails_cross: bool = True  # link each rail's last province to its counterpart when they touch
    expected_routes: int | None = None  # separate groups of front provinces; None: not checked
    player_box: bool = False  # detached one-province staging island per side, outside the layout
    capital_vp: float = 10.0
    scale: float = 0.78  # every coordinate below is shrunk about the bitmap centre by this factor
    cell_aspect: float = 1.7  # playable provinces are wider than tall (the box rule is tighter in y)
    dataset_role: str = "train_validation"  # suggestion only: train_validation | held_out_map
    tuned: bool = True  # False: builds and validates, but nobody has looked hard at its balance

    def __post_init__(self) -> None:
        if self.symmetry not in ("mirror", "rotate180") or self.blocked_kind not in ("impassable", "lake"):
            raise ArenaError("symmetry must be mirror|rotate180 and blocked_kind impassable|lake")
        if self.width % 256 or self.height % 256 or min(self.width, self.height) < 256:
            raise ArenaError("bitmap width and height must be multiples of 256")
        if self.provinces_per_side < 3 or not self.land:
            raise ArenaError("a design needs land and at least three provinces per side")
        if any(name not in TERRAIN_INDEX for name, _ in self.terrain_zones):
            raise ArenaError(f"terrain zones must use {sorted(TERRAIN_INDEX)}")
        if self.sector_cuts and not 0 < self.sector_cuts[0] < self.sector_cuts[1] < 1:
            raise ArenaError("sector cuts must be increasing fractions")

    def to_map(self, x: float, y: float) -> tuple[float, float]:
        """Design fraction -> bitmap fraction."""
        return 0.5 + (x - 0.5) * self.scale, 0.5 + (y - 0.5) * self.scale

    def to_design(self, x: Any, y: Any) -> tuple[Any, Any]:
        return 0.5 + (x - 0.5) / self.scale, 0.5 + (y - 0.5) / self.scale

    @property
    def scenario_id(self) -> str:
        return f"custom_{self.name}_v1"

    def sized(self, width: int, height: int, provinces_per_side: int | None = None) -> MapDesign:
        return replace(self, width=width, height=height,
                       provinces_per_side=provinces_per_side or self.provinces_per_side)


@dataclass(frozen=True)
class MapProvince:
    id: int
    kind: str  # play | blocked | box | sea
    side: str  # Country value of the half it lies in
    partner: int  # its image under the design's symmetry
    pixels: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 exclusive
    cx: int  # an interior pixel near the centroid: where counters stand
    cy: int
    terrain: str
    coastal: bool
    sector: str


@dataclass(frozen=True)
class CustomMap:
    design: MapDesign
    seed_used: int
    ids: Any = field(repr=False)  # (h, w) int32 province ids
    rivers: Any = field(repr=False)  # (h, w) uint8 rivers.bmp indices
    provinces: tuple[MapProvince, ...] = ()
    borders: dict[tuple[int, int], int] = field(default_factory=dict)  # 4-neighbour border length
    neighbors: dict[int, tuple[int, ...]] = field(default_factory=dict)  # playable graph
    river_neighbors: dict[int, tuple[int, ...]] = field(default_factory=dict)
    impassable_links: tuple[tuple[int, int], ...] = ()
    repairs: tuple[tuple[int, int], ...] = ()  # (x, y) of every centre-line repair pixel
    capitals: dict[str, int] = field(default_factory=dict)
    victory_points: tuple[tuple[int, float], ...] = ()
    hubs: tuple[int, ...] = ()
    railways: tuple[tuple[int, ...], ...] = ()
    sector_cuts: tuple[float, float] = (1 / 3, 2 / 3)
    frame: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # origin x, y, scale x, y
    frame_offset: tuple[float, float] = (0.0, 0.0)

    def province(self, province_id: int) -> MapProvince:
        return self.provinces[province_id - 1]

    def of_kind(self, kind: str) -> tuple[MapProvince, ...]:
        return tuple(p for p in self.provinces if p.kind == kind)

    def image(self, x: int, y: int) -> tuple[int, int]:
        """Where the design's symmetry sends pixel (x, y)."""
        height, width = self.ids.shape
        return (width - 1 - x, y) if self.design.symmetry == "mirror" else (width - 1 - x, height - 1 - y)

    def normalized(self, x: float, y: float) -> tuple[float, float]:
        ox, oy, sx, sy = self.frame
        return round((x - ox) * sx + self.frame_offset[0], 6), round((y - oy) * sy + self.frame_offset[1], 6)

    def layout(self) -> ArenaLayout:
        points = dict(self.victory_points)
        capital_of = {province: side for side, province in self.capitals.items()}
        rows = []
        for p in self.of_kind("play"):
            x, y = self.normalized(p.cx + 0.5, p.cy + 0.5)
            rows.append(LayoutProvince(p.id, x, y, p.terrain, self.neighbors[p.id], self.river_neighbors[p.id],
                                       p.sector, float(points.get(p.id, 0.0)), p.side, capital_of.get(p.id, "")))
        return ArenaLayout(self.design.scenario_id, tuple(rows), source="custom_map")

    def region(self) -> Region:
        """The generated map in the shape ``generate.py`` expects of a vanilla region."""
        play = self.of_kind("play")
        spec = RegionSpec(self.design.scenario_id, (), tuple(p.id for p in play if p.side == "BLU"),
                          self.capitals["BLU"], self.capitals["RED"], self.victory_points,
                          rationale=self.design.description)
        provinces = []
        for p in play:
            x, y = self.normalized(p.cx + 0.5, p.cy + 0.5)
            provinces.append(RegionProvince(p.id, 0, p.terrain, p.coastal, p.pixels, p.cx + 0.5, p.cy + 0.5,
                                            x, y, p.side, p.sector))
        xs = [p.bbox[0] for p in play] + [p.bbox[2] for p in play]
        ys = [p.bbox[1] for p in play] + [p.bbox[3] for p in play]
        states: tuple[VanillaState, ...] = ()
        return Region(spec, tuple(provinces), dict(self.neighbors), dict(self.river_neighbors), (), states,
                      (min(xs), min(ys), max(xs), max(ys)), self.frame, self.frame_offset, (), ())

    def front(self, side: str) -> tuple[int, ...]:
        mine = {p.id for p in self.of_kind("play") if p.side == side}
        return tuple(sorted(p for p in mine if any(n not in mine for n in self.neighbors[p])))

    def routes(self) -> tuple[tuple[int, ...], ...]:
        """Separate groups of front provinces (both sides): one group = one approach route."""
        front = set(self.front("BLU")) | set(self.front("RED"))
        groups, seen = [], set()
        for start in sorted(front):
            if start in seen:
                continue
            group, queue = {start}, deque([start])
            while queue:
                for nxt in self.neighbors[queue.popleft()]:
                    if nxt in front and nxt not in group:
                        group.add(nxt)
                        queue.append(nxt)
            seen |= group
            groups.append(tuple(sorted(group)))
        return tuple(groups)


# ---------------------------------------------------------------------------------------------
# Raster helpers (no scipy in this environment).

def components(lab: Any) -> tuple[Any, list[int], list[int]]:
    """4-connected components of equal labels, by run-length union-find. Returns the component raster
    and, per component, its label and its pixel count."""
    height, width = lab.shape
    parent: list[int] = []
    label: list[int] = []
    size: list[int] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rows: list[tuple[list[int], list[int]]] = []
    previous: list[tuple[int, int, int]] = []
    for y in range(height):
        row = lab[y]
        cuts = (np.flatnonzero(row[1:] != row[:-1]) + 1).tolist()
        starts, ends = [0, *cuts], [*cuts, width]
        current, j = [], 0
        for start, end in zip(starts, ends, strict=True):
            index = len(parent)
            parent.append(index)
            label.append(int(row[start]))
            size.append(end - start)
            while j < len(previous) and previous[j][1] <= start:
                j += 1
            k = j
            while k < len(previous) and previous[k][0] < end:
                if label[previous[k][2]] == label[index]:
                    a, b = find(index), find(previous[k][2])
                    if a != b:
                        parent[max(a, b)] = min(a, b)
                k += 1
            current.append((start, end, index))
        rows.append(([c[2] for c in current], [c[1] - c[0] for c in current]))
        previous = current
    roots = [find(i) for i in range(len(parent))]
    order = {root: n for n, root in enumerate(sorted(set(roots)))}
    comp_label = [0] * len(order)
    comp_size = [0] * len(order)
    for i, root in enumerate(roots):
        comp_label[order[root]] = label[root]
        comp_size[order[root]] += size[i]
    comp = np.empty((height, width), dtype=np.int32)
    for y, (indices, lengths) in enumerate(rows):
        comp[y] = np.repeat(np.array([order[roots[i]] for i in indices], dtype=np.int32), lengths)
    return comp, comp_label, comp_size


def x_crossings(lab: Any) -> Any:
    """(y, x) of the top-left pixel of every 2x2 block in which four different provinces meet."""
    a, b, c, d = lab[:-1, :-1], lab[:-1, 1:], lab[1:, :-1], lab[1:, 1:]
    return np.argwhere((a != b) & (a != c) & (a != d) & (b != c) & (b != d) & (c != d))


def border_lengths(ids: Any) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    base = int(ids.max()) + 1
    for a, b in ((ids[:, :-1], ids[:, 1:]), (ids[:-1], ids[1:])):
        differ = a != b
        low, high = np.minimum(a, b)[differ].astype(np.int64), np.maximum(a, b)[differ].astype(np.int64)
        unique, counts = np.unique(low * base + high, return_counts=True)
        for key, count in zip(unique.tolist(), counts.tolist(), strict=True):
            pair = (key // base, key % base)
            out[pair] = out.get(pair, 0) + count
    return out


def line_pixels(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """4-connected line: every step moves in x OR y, so it cannot slip through a staircase river."""
    points = [(x0, y0)]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
    x, y, error = x0, y0, dx - dy
    while (x, y) != (x1, y1):
        if x != x1 and (y == y1 or error * 2 > -dy):  # horizontal step
            error -= dy
            x += sx
        else:
            error += dx
            y += sy
        points.append((x, y))
    return points


def _bbox(mask: Any) -> tuple[int, int, int, int]:
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _voronoi(mask: Any, count: int, seed: int, aspect: float) -> Any:
    """Labels 0..count-1 over ``mask``: Lloyd-relaxed seeds, distance stretched in y by ``aspect``."""
    height, width = mask.shape
    step = max(1, width // 256)
    ys, xs = np.nonzero(mask[::step, ::step])
    points = np.stack([xs * step, ys * step * aspect], axis=1).astype(np.float64)
    if len(points) < count * 4:
        raise ArenaError("the playable area is too small for the requested number of provinces")
    rng = random.Random(seed)
    seeds = points[sorted(rng.sample(range(len(points)), count))].copy()
    for _ in range(48):
        nearest = ((points[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        for k in range(count):
            mine = points[nearest == k]
            if len(mine):
                seeds[k] = np.round(mine.mean(axis=0), 3)
    out = np.full(mask.shape, -1, dtype=np.int32)
    columns = np.arange(width, dtype=np.float64)
    for y0 in range(0, height, 64):
        band = np.arange(y0, min(y0 + 64, height), dtype=np.float64) * aspect
        distance = ((columns[None, :, None] - seeds[None, None, :, 0]) ** 2
                    + (band[:, None, None] - seeds[None, None, :, 1]) ** 2)
        out[y0:y0 + 64] = distance.argmin(axis=2)
    out[~mask] = -1
    return out


def _half_labels(design: MapDesign, seed: int) -> tuple[Any, dict[int, str]]:
    """Label raster of the WEST half and the kind of every label. The only design-specific stage."""
    height, half = design.height, design.width // 2
    ys, xs = np.mgrid[0:height, 0:half]
    fx, fy = design.to_design((xs + 0.5) / design.width, (ys + 0.5) / design.height)

    def union(shapes: tuple[Shape, ...]) -> Any:
        out = np.zeros((height, half), dtype=bool)
        for shape in shapes:
            out |= shape.mask(fx, fy)
        return out

    land = union(design.land) & ~union(design.cut)
    blocked = land & union(design.blocked)
    play = land & ~blocked
    box = np.zeros_like(land)
    if design.player_box:
        box = rect(0.04, 0.44, 0.04 + 0.06, 0.56).mask(fx, fy) & ~land
    if not play[:, -1].any():
        raise ArenaError("the playable land must reach the centre line (there would be no front)")
    cell_w, cell_h = max(design.width // 20, 4), max(design.height // 20, 4)
    grid = (ys // cell_h) * (half // cell_w + 1) + xs // cell_w
    cells = int(grid.max()) + 1
    lab = np.where(play, _voronoi(play, design.provinces_per_side, seed, design.cell_aspect), 0)
    offset = design.provinces_per_side
    lab = np.where(blocked, offset + grid, lab)
    lab = np.where(box, offset + cells, lab)
    lab = np.where(~land & ~box, offset + cells + 1 + grid, lab).astype(np.int32)
    kinds = {}
    for value in np.unique(lab).tolist():
        kinds[value] = ("play" if value < offset else "blocked" if value < offset + cells
                        else "box" if value == offset + cells else "sea")
    return lab, kinds


def _cleanup(lab: Any, kinds: dict[int, str], limits: tuple[int, int], minimum: dict[str, int]) -> Any:
    """Merge stray fragments and undersized provinces into a neighbour, then break X crossings."""
    lab = lab.copy()
    for _ in range(8):
        comp, comp_label, comp_size = components(lab)
        largest: dict[int, int] = {}
        total: dict[int, int] = {}
        for index, (value, size) in enumerate(zip(comp_label, comp_size, strict=True)):
            total[value] = total.get(value, 0) + size
            if value not in largest or size > comp_size[largest[value]]:
                largest[value] = index
        victims = [i for i, value in enumerate(comp_label)
                   if largest[value] != i or total[value] < minimum[kinds[value]]]
        boxes: dict[int, tuple[int, int, int, int]] = {}
        for index in victims:
            mask = comp == index
            x0, y0, x1, y1 = _bbox(mask)
            px0, py0, px1, py1 = max(x0 - 1, 0), max(y0 - 1, 0), x1 + 1, y1 + 1
            window = mask[py0:py1, px0:px1]
            ring = np.zeros_like(window)
            ring[1:] |= window[:-1]
            ring[:-1] |= window[1:]
            ring[:, 1:] |= window[:, :-1]
            ring[:, :-1] |= window[:, 1:]
            ring &= ~window
            around = lab[py0:py1, px0:px1][ring]
            around = around[around != comp_label[index]]
            if not around.size:
                continue
            values, counts = np.unique(around, return_counts=True)
            ranked = []
            for value, count in zip(values.tolist(), counts.tolist(), strict=True):
                if value not in boxes:
                    boxes[value] = _bbox(lab == value)
                bx0, by0, bx1, by1 = boxes[value]
                fits = max(bx1, x1) - min(bx0, x0) < limits[0] and max(by1, y1) - min(by0, y0) < limits[1]
                ranked.append((kinds[value] == kinds[comp_label[index]], fits, count, -value))
            _, _, _, negative = max(ranked)
            lab[mask] = -negative
            boxes.pop(-negative, None)
        crossings = x_crossings(lab)
        for y, x in crossings.tolist():
            if len({int(lab[y, x]), int(lab[y, x + 1]), int(lab[y + 1, x]), int(lab[y + 1, x + 1])}) == 4:
                lab[y + 1, x] = lab[y, x]
        if not victims and not len(crossings):
            return lab
    raise ArenaError("province cleanup did not converge (fragments or X crossings remain)")


def _nearest(provinces: list[MapProvince], x: float, y: float, design: MapDesign) -> MapProvince:
    mx, my = design.to_map(x, y)
    px, py = mx * design.width, my * design.height
    return min(provinces, key=lambda p: ((p.cx - px) ** 2 + (p.cy - py) ** 2, p.id))


def _path(start: int, goal: int, allowed: set[int], neighbors: dict[int, tuple[int, ...]]) -> list[int]:
    previous: dict[int, int] = {start: start}
    queue = deque([start])
    while queue:
        here = queue.popleft()
        if here == goal:
            break
        for nxt in neighbors[here]:
            if nxt in allowed and nxt not in previous:
                previous[nxt] = here
                queue.append(nxt)
    if goal not in previous:
        raise ArenaError(f"no railway path from province {start} to {goal}")
    out = [goal]
    while out[-1] != start:
        out.append(previous[out[-1]])
    return out[::-1]


def _river_raster(design: MapDesign, ids: Any, land: Any, image: Any) -> Any:
    height, width = ids.shape
    out = np.where(land, 255, 254).astype(np.uint8)

    def draw(pixels: list[tuple[int, int]], index: int) -> None:
        runs: list[list[tuple[int, int]]] = [[]]
        for x, y in pixels:  # keep the longest stretch that stays on land and off other rivers
            free = 0 <= x < width and 0 <= y < height and land[y, x] and out[y, x] == 255
            touching = free and any(out[y + dy, x + dx] <= 11 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                                    if 0 <= x + dx < width and 0 <= y + dy < height)
            if free and not touching:
                runs[-1].append((x, y))
            elif runs[-1]:
                runs.append([])
        best = max(runs, key=len)
        if len(best) < 3:
            raise ArenaError("a river does not fit on the land of this design")
        for n, (x, y) in enumerate(best):
            out[y, x] = 0 if n == 0 else index

    for river in design.rivers:
        if not 3 <= river.width <= 11 or len(river.points) < 2:
            raise ArenaError("river width must be a palette index 3-11 and a river needs two points")
        pixels: list[tuple[int, int]] = []
        mapped = [design.to_map(x, y) for x, y in river.points]
        for (ax, ay), (bx, by) in zip(mapped, mapped[1:], strict=False):
            x0 = width // 2 if river.center else min(int(ax * width), width // 2 - 3)
            x1 = width // 2 if river.center else min(int(bx * width), width // 2 - 3)
            segment = line_pixels(x0, int(ay * height), x1, int(by * height))
            pixels += segment[1:] if pixels else segment
        unique = list(dict.fromkeys(pixels))
        draw(unique, river.width)
        if not river.center:
            draw([image(x, y) for x, y in unique], river.width)
    return out


def rasterize(design: MapDesign) -> CustomMap:
    """Run the whole pipeline. Tries ``seed``, ``seed + 1``, ... until the engine's size rules hold."""
    limits = (design.width // 8, design.height // 8)
    problem = ""
    for attempt in range(12):
        lab, kinds = _half_labels(design, design.seed + attempt)
        play_pixels = int(np.isin(lab, [v for v, k in kinds.items() if k == "play"]).sum())
        minimum = {"play": max(MIN_PIXELS, play_pixels // design.provinces_per_side // 5), "box": MIN_PIXELS,
                   "blocked": max(MIN_PIXELS, limits[0] * limits[1] // 64),
                   "sea": max(MIN_PIXELS, limits[0] * limits[1] // 64)}
        try:
            half = _cleanup(lab, kinds, limits, minimum)
        except ArenaError as exc:
            problem = str(exc)
            continue
        sizes = [(_bbox(half == value), value) for value in np.unique(half).tolist()]
        too_big = [value for (x0, y0, x1, y1), value in sizes if x1 - x0 >= limits[0] or y1 - y0 >= limits[1]]
        if not too_big:
            return _assemble(design, design.seed + attempt, half, kinds)
        problem = f"{len(too_big)} provinces exceed the 1/8 bounding-box rule"
    raise ArenaError(f"design {design.name}: no seed produced a legal raster ({problem})")


def _assemble(design: MapDesign, seed_used: int, half: Any, kinds: dict[int, str]) -> CustomMap:
    height, width = design.height, design.width
    mirror = design.symmetry == "mirror"
    span = int(half.max()) + 1
    east = (half[:, ::-1] if mirror else half[::-1, ::-1]) + span
    full = np.concatenate([half, east], axis=1).astype(np.int32)
    middle = width // 2
    repairs: list[tuple[int, int]] = []
    jog_links: list[tuple[int, int]] = []
    # Two seams where a half meets its image: the centre line, and the east/west edge (the map WRAPS
    # horizontally: launch 2 logged "Map invalid X crossing" at x = width - 1 for every sea row boundary).
    for left, right in ((middle - 1, middle), (width - 1, 0)):
        for _ in range(4 * height):  # one pixel per crossing (see the module docstring)
            found = x_crossings(np.stack([full[:, left], full[:, right]], axis=1))
            if not len(found):
                break
            y = int(found[0][0])
            jog_links.append((int(full[y, left]), int(full[y + 1, right])))
            full[y + 1, left] = full[y, left]
            repairs.append((left, y + 1))
            if not mirror:  # the same repair seen through the rotation keeps the raster rotation-symmetric
                jog_links.append((int(full[height - 1 - y, right]), int(full[height - 2 - y, left])))
                full[height - 2 - y, right] = full[height - 1 - y, right]
                repairs.append((right, height - 2 - y))
        else:
            raise ArenaError("seam X crossing repair did not converge")
    if len(x_crossings(np.concatenate([full, full[:, :1]], axis=1))):
        raise ArenaError("X crossings remain after the seam repairs")

    # Sequential ids: west playable (top to bottom), their partners, then blocked, box and sea the same way.
    west_values = np.unique(half).tolist()
    centroid = {}
    for value in west_values:
        ys, xs = np.nonzero(half == value)
        cy, cx = float(ys.mean()), float(xs.mean())
        nearest = int(((ys - cy) ** 2 + (xs - cx) ** 2).argmin())
        centroid[value] = (int(xs[nearest]), int(ys[nearest]))
    new_id: dict[int, int] = {}
    for kind in KINDS:
        members = sorted((v for v in west_values if kinds[v] == kind), key=lambda v: (centroid[v][1], centroid[v][0]))
        for value in members:
            new_id[value] = len(new_id) + 1
        for value in members:
            new_id[value + span] = len(new_id) + 1
    table = np.zeros(2 * span, dtype=np.int32)
    for value, province_id in new_id.items():
        table[value] = province_id
    ids = table[full]
    jog = sorted({(min(new_id[a], new_id[b]), max(new_id[a], new_id[b])) for a, b in jog_links})

    land = np.isin(ids, [new_id[v + s] for v in west_values for s in (0, span) if kinds[v] != "sea"])
    borders = border_lengths(ids)
    kind_of = {new_id[v + s]: kinds[v] for v in west_values for s in (0, span)}
    play_ids = {p for p, kind in kind_of.items() if kind == "play"}
    links = {pair for pair in borders if pair[0] in play_ids and pair[1] in play_ids and pair not in jog}
    neighbors = {p: tuple(sorted({b if a == p else a for a, b in links if p in (a, b)})) for p in sorted(play_ids)}
    coastal = {p for pair in borders for p in pair
               if (kind_of[pair[0]] == "sea") != (kind_of[pair[1]] == "sea")}

    def image(x: int, y: int) -> tuple[int, int]:
        return (width - 1 - x, y) if mirror else (width - 1 - x, height - 1 - y)

    centers: dict[int, tuple[int, int]] = {}
    for value in west_values:
        centers[new_id[value]] = centroid[value]
        centers[new_id[value + span]] = image(*centroid[value])
    for province_id, (cx, cy) in centers.items():
        if int(ids[cy, cx]) != province_id:
            raise ArenaError(f"province {province_id}: its centre pixel was lost to a repair")

    rivers = _river_raster(design, ids, land, image)
    wet = rivers <= 11
    crossings: dict[int, set[int]] = {p: set() for p in play_ids}
    for a, b in sorted(links):
        if any(wet[y, x] for x, y in line_pixels(*centers[a], *centers[b])):
            crossings[a].add(b)
            crossings[b].add(a)

    # Sectors from y cuts; without explicit cuts, the terciles of the west front, symmetrized about 0.5.
    west_front = sorted((centers[p][1] + 0.5) / height for p in play_ids
                        if centers[p][0] < middle and any(centers[n][0] >= middle for n in neighbors[p]))
    if design.sector_cuts:
        cuts = (design.to_map(0.5, design.sector_cuts[0])[1], design.to_map(0.5, design.sector_cuts[1])[1])
    elif len(west_front) >= 3:
        low = (west_front[len(west_front) // 3 - 1] + west_front[len(west_front) // 3]) / 2
        high = (west_front[-(len(west_front) // 3) - 1] + west_front[-(len(west_front) // 3)]) / 2
        shift = (low + (1 - high)) / 2
        cuts = (round(shift, 6), round(1 - shift, 6))
    else:
        cuts = (1 / 3, 2 / 3)

    def terrain_of(province_id: int) -> str:
        if kind_of[province_id] == "sea":
            return "ocean"
        if kind_of[province_id] == "blocked":
            return "lakes" if design.blocked_kind == "lake" else "mountain"
        cx, cy = centers[province_id]
        if cx >= middle:
            cx, cy = image(cx, cy)
        fx, fy = design.to_design((cx + 0.5) / width, (cy + 0.5) / height)
        return next((name for name, shape in design.terrain_zones if bool(shape.mask(fx, fy))), "plains")

    size = np.bincount(ids.ravel(), minlength=len(new_id) + 1)
    partner = {new_id[v]: new_id[v + span] for v in west_values} | {new_id[v + span]: new_id[v] for v in west_values}
    provinces = []
    for province_id in range(1, len(new_id) + 1):
        cx, cy = centers[province_id]
        fy = (cy + 0.5) / height
        sector = SECTORS[0] if fy < cuts[0] else SECTORS[2] if fy > cuts[1] else SECTORS[1]
        provinces.append(MapProvince(
            province_id, kind_of[province_id], Country.BLUE.value if cx < middle else Country.RED.value,
            partner[province_id], int(size[province_id]), _bbox(ids == province_id), cx, cy, terrain_of(province_id),
            province_id in coastal, sector if kind_of[province_id] == "play" else ""))

    west_play = [p for p in provinces if p.kind == "play" and p.side == "BLU"]
    if not _connected({p.id for p in west_play}, neighbors) or not _connected(play_ids, neighbors):
        raise ArenaError(f"design {design.name}: the playable land is not one connected area")
    capital = _nearest(west_play, *design.capital, design)
    points: dict[int, float] = {capital.id: design.capital_vp, capital.partner: design.capital_vp}
    for vx, vy, value in design.victory_points:
        target = _nearest(west_play, vx, vy, design)
        if target.id not in points:
            points[target.id] = points[target.partner] = float(value)
    hubs = {capital.id, capital.partner}
    for hx, hy in design.supply_hubs:
        hub = _nearest(west_play, hx, hy, design)
        hubs |= {hub.id, hub.partner}
    railways: list[tuple[int, ...]] = []
    west_ids = {p.id for p in west_play}
    for rail in design.rails:
        chain = [capital.id]
        for rx, ry in rail:
            chain += _path(chain[-1], _nearest(west_play, rx, ry, design).id, west_ids, neighbors)[1:]
        if len(chain) >= 2:
            railways += [tuple(chain), tuple(partner[p] for p in chain)]
            if design.rails_cross and partner[chain[-1]] in neighbors[chain[-1]]:
                railways.append((chain[-1], partner[chain[-1]]))
    railways = sorted(set(railways))

    play_list = [p for p in provinces if p.kind != "sea"]
    bx0, by0 = min(p.bbox[0] for p in play_list), min(p.bbox[1] for p in play_list)
    bx1, by1 = max(p.bbox[2] for p in play_list), max(p.bbox[3] for p in play_list)
    margin, aspect = 0.06, 16 / 9
    usable = 1.0 - 2 * margin
    scale = min(usable * aspect / (bx1 - bx0), usable / (by1 - by0))
    sx, sy = scale / aspect, scale
    return CustomMap(
        design, seed_used, ids, rivers, tuple(provinces), borders, neighbors,
        {p: tuple(sorted(c)) for p, c in sorted(crossings.items())}, tuple(jog), tuple(repairs),
        {"BLU": capital.id, "RED": capital.partner}, tuple(sorted(points.items())), tuple(sorted(hubs)),
        tuple(railways), (float(cuts[0]), float(cuts[1])), (float(bx0), float(by0), round(sx, 9), round(sy, 9)),
        (round((1.0 - (bx1 - bx0) * sx) / 2, 6), round((1.0 - (by1 - by0) * sy) / 2, 6)))


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


# ---------------------------------------------------------------------------------------------
# Presets. x < 0.5 is BLUE's half; the arena occupies roughly the middle 56% x 64% of the bitmap.

_RAILS3 = (((0.47, 0.27),), ((0.47, 0.50),), ((0.47, 0.73),))

PRESETS: dict[str, MapDesign] = {design.name: design for design in (
    MapDesign(
        "three_lanes", "Dumbbell: two home lobes joined by a forest lane, a river lane and a hill lane; the "
        "infill between the lanes is blocked, so there are exactly three approach routes.", seed=11,
        land=(ellipse(0.20, 0.18, 0.46, 0.82), rect(0.38, 0.20, 0.50, 0.80)), capital=(0.29, 0.50),
        victory_points=((0.47, 0.27, 3.0), (0.47, 0.50, 5.0), (0.47, 0.73, 3.0)),
        blocked=(rect(0.385, 0.34, 0.50, 0.44), rect(0.385, 0.56, 0.50, 0.66)), sector_cuts=(0.39, 0.61),
        rivers=(River(((0.5, 0.445), (0.5, 0.555)), width=8, center=True),),
        terrain_zones=(("forest", rect(0.38, 0.20, 0.50, 0.34)), ("hills", rect(0.38, 0.66, 0.50, 0.80))),
        rails=_RAILS3, expected_routes=3),
    MapDesign(
        "open_field", "One wide continuous front on open ground with a few woods and hills, no chokepoints, "
        "three central victory points: front management and encirclement.", seed=23,
        land=(ellipse(0.19, 0.20, 0.35, 0.80), rect(0.27, 0.20, 0.50, 0.80)), capital=(0.26, 0.50),
        victory_points=((0.47, 0.32, 3.0), (0.47, 0.50, 5.0), (0.47, 0.68, 3.0)),
        terrain_zones=(("forest", ellipse(0.33, 0.24, 0.41, 0.38)), ("hills", ellipse(0.35, 0.62, 0.43, 0.76))),
        rails=_RAILS3, expected_routes=1),
    MapDesign(
        "river_line", "Plains split by one major river along the whole centre border: every attack is a "
        "river crossing, the front is only a few provinces tall.", seed=37, provinces_per_side=20,
        land=(ellipse(0.22, 0.24, 0.36, 0.76), rect(0.29, 0.24, 0.50, 0.76)), capital=(0.29, 0.50),
        victory_points=((0.46, 0.34, 4.0), (0.46, 0.66, 4.0)),
        rivers=(River(((0.5, 0.20), (0.5, 0.80)), width=9, center=True),),
        rails=(((0.46, 0.34),), ((0.46, 0.66),)), expected_routes=1, tuned=False),
    MapDesign(
        "highland_pass", "Hills with two blocked massifs on the border: a narrow northern pass, a mountain pass "
        "in the centre and a long southern flanking route around the larger massif.", seed=41,
        land=(rect(0.24, 0.16, 0.50, 0.86),), capital=(0.29, 0.48),
        victory_points=((0.47, 0.21, 3.0), (0.47, 0.49, 5.0), (0.47, 0.81, 3.0)),
        blocked=(rect(0.41, 0.26, 0.50, 0.44), rect(0.34, 0.54, 0.50, 0.76)), sector_cuts=(0.35, 0.65),
        terrain_zones=(("mountain", rect(0.41, 0.44, 0.50, 0.54)), ("hills", rect(0.33, 0.16, 0.50, 0.86))),
        rails=(((0.47, 0.49),), ((0.30, 0.81), (0.47, 0.81))), expected_routes=3,
        dataset_role="held_out_map", tuned=False),
    MapDesign(
        "king_of_the_hill", "180-degree rotational symmetry: capitals in opposite corners, far apart, and a "
        "high-value cluster of hill provinces in the middle that both sides must contest.", seed=53,
        symmetry="rotate180", land=(rect(0.23, 0.18, 0.50, 0.82),), capital=(0.27, 0.26),
        victory_points=((0.475, 0.50, 8.0), (0.45, 0.38, 4.0), (0.45, 0.62, 4.0)),
        terrain_zones=(("hills", ellipse(0.40, 0.34, 0.60, 0.66)), ("forest", ellipse(0.26, 0.58, 0.36, 0.78))),
        supply_hubs=((0.40, 0.50),), rails=(((0.40, 0.50), (0.47, 0.50)),), expected_routes=1,
        dataset_role="held_out_map", tuned=False),
)}
DEFAULT_DESIGN = "three_lanes"


def summary(arena: CustomMap) -> dict[str, Any]:
    counts = {kind: len(arena.of_kind(kind)) for kind in KINDS}
    play = [p.pixels for p in arena.of_kind("play")]
    return {"design": arena.design.name, "symmetry": arena.design.symmetry, "seed_used": arena.seed_used,
            "size": [arena.design.width, arena.design.height], "provinces": counts,
            "playable_pixels_min_max": [min(play), max(play)], "routes": [list(r) for r in arena.routes()],
            "centre_repairs": len(arena.repairs), "impassable_links": [list(p) for p in arena.impassable_links],
            "river_crossings": sum(len(v) for v in arena.river_neighbors.values()) // 2,
            "sector_cuts": list(arena.sector_cuts)}
