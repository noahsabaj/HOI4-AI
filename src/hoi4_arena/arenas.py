"""Named arena designs, and the painting that turns one into the map's pictures.

A design is written on the 24 by 8 province grid of the playable arena: x runs from
Blue's west coast (0) across the seam (12) to Red's east coast (24), y from the north
coast (0) to the south (8). Everything in it is mirrored by a half turn about the
middle, so a forest drawn at (3, 2) also grows at (21, 6) and a river flowing east in
the north has a twin flowing west in the south. The two sides fight over the same
ground, turned round.

Everything painted is vanilla: the stock terrain types, the stock river palette, the
stock tree and city models. Terrain changes how a fight goes only through the game's own
rules for it (attack, movement, width, attrition and supply by terrain; river crossings).

The facts each constant rests on were measured on the stock 1.19 map unless said
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.special import ndtr

# The grid presets are written on: 24 province columns (12 a side) by 8 rows.
DESIGN_COLUMNS, DESIGN_ROWS = 24, 8

# Graphical terrain: terrain.bmp palette index -> province terrain type, from the stock
# common/terrain/00_terrain.txt. Only indices without side effects: snow_16 (16) and
# plains_17 (19) carry perm_snow. forest_13 (13) is type urban with spawn_city, so it is
# painted only where a city stands.
GRAPHICAL_TYPE = {
    0: "plains",
    5: "plains",
    1: "forest",
    4: "forest",
    17: "hills",
    2: "hills",
    20: "mountain",
    11: "mountain",
    9: "marsh",
    13: "urban",
    15: "ocean",
}
WATER_INDEX = 15
# The indices each type is painted with. Stock plains provinces are 81% grass (0) and 8%
# farmland (5); forest 63% dark (1) and 29% light (4); hills 50% ridged (2) and 33%
# rolling (17); mountains 33% rock (11) and 24% grassy slopes (20); marsh 73% index 9;
# urban 76% index 13.
PLAINS, FARMLAND, DARK_FOREST, LIGHT_FOREST = 0, 5, 1, 4
ROLLING_HILLS, RIDGED_HILLS, GREEN_MOUNTAIN, ROCK, MARSH, CITY = 17, 2, 20, 11, 9, 13
LAND_TYPES = ("plains", "forest", "hills", "mountain", "marsh", "urban")

# Land relief per type, as a heightmap byte level and the spread around it. Stock medians
# (90th percentiles): plains 102 (122), forest 105 (121), hills 115 (140), mountains 129
# (172), marsh 98, urban 101. Sea level is byte 95; nothing on land goes below 96 away
# from the coast, so no land floods.
RELIEF = {
    "plains": (102.0, 3.0),
    "forest": (105.0, 4.0),
    "urban": (101.0, 1.0),
    "marsh": (97.0, 0.6),
    "hills": (119.0, 9.0),
    "mountain": (150.0, 32.0),
}
SEA_LEVEL, DRY_LAND, WATER_FLOOR = 95, 96, 89
# Above this byte a mountain is painted as bare rock, below it as green slopes.
ROCK_LINE = 150
# A coast falls to the water over this many pixels: steeper ground needs the room.
COAST_PIXELS = 12

# rivers.bmp: index 0 marks a river's source and 1 the last pixel of a tributary where it
# joins another river; 3 to 11 are river pixels from narrowest to widest; 254 is water
# and 255 land. The stock defines NMilitary.RIVER_SMALL_STOP_INDEX = 6 and
# RIVER_LARGE_STOP_INDEX = 11: a crossing over indices 7-11 is a large river (-60% attack,
# -50% speed), over 0-6 a small one (-30%, -25%). 86% of stock river pixels lie on a
# province border (borders are 16.5% of the land), because only a river along a border
# is ever crossed.
RIVER_SOURCE, RIVER_JOIN, RIVER_WATER, RIVER_LAND = 0, 1, 254, 255
RIVER_WIDTHS = {
    False: ((0.5, 3), (1.01, 4)),
    True: ((0.05, 3), (0.1, 5), (0.2, 7), (0.45, 9), (1.01, 11)),
}

# trees.bmp: the indices stock European forests use (43% and 28% of forest pixels; 28 and
# 29 are the tropical ones), none of them in default.map's tree list, which only the
# automatic terrain assignment reads. Cover is the share of pixels with a tree: stock
# forest pixels are 85% covered, plains 2.6%.
TREE_INDICES = (6, 5)
TREE_COVER = {"forest": 0.85, "marsh": 0.08, "plains": 0.02, "hills": 0.05, "mountain": 0.03}

# cities.bmp picks the city group from the stock cities.txt: index 1 the dense French city
# set, 15 the western houses. A city province is painted urban over CITY_SHARE of its
# pixels, so its majority reads urban, as a stock city province does (the median one is
# 87% index 13); its middle CITY_CORE of them is dense city.
CITY_CORE_INDEX, CITY_SUBURB_INDEX = 1, 15
CITY_SHARE, CITY_CORE = 0.6, 0.25
# City lights live in the colour map's alpha. Stock European cities average 168.
CITY_LIGHTS = (200, 110)

# Colour-map ground per graphical index. Kept within the brightness and tint of the plain
# arena's (78, 86, 62) grass and (58, 68, 48) forest: at full zoom-out the colour map shows
# through the political view, and vision.country_pixels tells Blue's land from Red's by
# tint (blue minus red above 10, red minus blue above 15, sum above 250), calibrated on
# those two. Stock Europe is greener and darker: plains (61, 69, 35), forest (43, 57, 20).
GROUND = {
    PLAINS: (80, 88, 62),
    FARMLAND: (88, 88, 60),
    DARK_FOREST: (56, 68, 46),
    LIGHT_FOREST: (64, 76, 52),
    ROLLING_HILLS: (84, 86, 64),
    RIDGED_HILLS: (88, 88, 66),
    GREEN_MOUNTAIN: (88, 88, 72),
    ROCK: (104, 102, 94),
    MARSH: (62, 74, 58),
    CITY: (86, 86, 78),
}


@dataclass(frozen=True)
class Patch:
    """Terrain painted over a shape, with a ragged edge.

    `shape` is "blob" (`at` is the centre, `size` the radius), "line" (`at` a polyline,
    `size` the half width) or "rect" (`at` is x0, y0, x1, y1). Sizes are in provinces.
    `rough` is how far the edge wanders, as a share of the size.
    """

    terrain: str
    shape: str
    at: tuple
    size: float = 1.0
    rough: float = 0.3


@dataclass(frozen=True)
class River:
    """A river from its source (the course's first point) to the sea, a lake, or another.

    `joins` is the index of an earlier river in the preset's list that this one flows
    into; otherwise it runs to the nearest water past the course's end. `large` rivers
    are crossed at -60% attack instead of -30%. `seam` keeps it exactly on the border
    between the two countries.
    """

    course: tuple
    large: bool = False
    joins: int | None = None
    seam: bool = False


@dataclass(frozen=True)
class Preset:
    """A named arena. The eastern half is always the western half turned round."""

    title: str
    summary: str
    seed: int
    patches: tuple = ()
    rivers: tuple = ()
    lakes: tuple = ()
    bays: tuple = ()
    # Cells on Red's side of the seam that Blue holds instead, each given back by its
    # twin on Blue's side: the border bends round them.
    trades: tuple = ()
    # Blue's cities, capital first: the capital holds 20 victory points and each of the
    # others 5, the 35 a side the plain arena has. The capital stands mid-country, as the
    # plain arena's does: the country picker opens centred on Blue's capital, and the
    # recorder picks Red by clicking Red's land there. With it at x = 2.6 Red was off the
    # screen and the pick failed.
    cities: tuple = ()
    base: str = "plains"
    # How far province borders wander (pixels) and seeds stray (share of a province).
    warp: float = 26.0
    jitter: float = 0.22
    land_columns: int = 12
    land_rows: int = 8
    state_columns: int = 4
    state_rows: int = 2


PRESETS = {
    "plains": Preset(
        title="Open Plains",
        summary="Farmland from coast to coast, a few woods and low hills, rivers running "
        "toward the enemy. Nothing to hide behind: a war of movement.",
        seed=101,
        patches=(
            Patch("forest", "blob", (3.4, 1.6), 1.1),
            Patch("forest", "blob", (8.2, 6.2), 0.9),
            Patch("forest", "blob", (5.0, 3.4), 0.6),
            Patch("hills", "blob", (1.2, 5.8), 1.2),
            Patch("hills", "blob", (10.2, 2.4), 0.6),
            Patch("marsh", "blob", (5.0, 7.6), 0.7),
        ),
        rivers=(
            River(((8.8, 1.0), (11.0, 1.7), (14.0, 1.4), (16.5, 0.0))),
            River(((3.8, 3.2), (4.6, 5.6), (4.2, 8.0))),
        ),
        cities=((6.4, 4.4), (2.6, 2.4), (3.0, 6.2), (9.6, 2.0)),
    ),
    "river": Preset(
        title="River Line",
        summary="A great river runs the length of each country, four provinces behind the "
        "border: an open front, then a line to fall back to that costs 60% of an "
        "attack to cross.",
        seed=202,
        patches=(
            Patch("hills", "blob", (1.4, 1.8), 1.2),
            Patch("forest", "blob", (10.4, 1.4), 1.0),
            Patch("forest", "blob", (3.8, 6.6), 1.2),
            Patch("marsh", "line", ((7.6, 6.4), (8.2, 8.2)), 0.55),
            Patch("forest", "blob", (10.8, 6.2), 0.6),
        ),
        rivers=(
            # From the north coast to the south: no way round it.
            River(((7.2, -0.4), (7.9, 2.5), (7.4, 4.6), (8.0, 6.6), (8.2, 8.0)), large=True),
            River(((2.6, 2.8), (5.0, 3.3), (7.3, 3.4)), joins=0),
            River(((11.0, 5.2), (9.6, 5.6), (8.0, 5.4)), joins=0),
        ),
        cities=((6.3, 4.2), (7.0, 1.6), (3.0, 5.6), (10.4, 5.6)),
    ),
    "passes": Preset(
        title="Mountain Passes",
        summary="A mountain range runs the length of the border, two provinces deep on each "
        "side, crossed by two passes. Attacks into the mountains lose half their "
        "strength, so the war is fought for the passes.",
        seed=303,
        patches=(
            Patch("hills", "line", ((12.0, -1.0), (12.0, 9.0)), 3.2, 0.12),
            Patch("mountain", "line", ((12.0, -1.0), (12.0, 9.0)), 2.0, 0.1),
            # One valley through the range on row 2, and its twin on row 5.
            Patch("plains", "rect", (8.9, 2.0, 15.1, 3.0)),
            Patch("forest", "blob", (7.8, 4.0), 0.8),
            Patch("hills", "blob", (1.0, 6.6), 1.0),
            Patch("forest", "blob", (4.6, 1.0), 0.9),
            Patch("marsh", "blob", (0.6, 3.6), 0.5),
        ),
        rivers=(River(((9.3, 4.0), (6.0, 4.3), (2.5, 3.8), (0.0, 4.1)), large=True),),
        cities=((6.4, 3.8), (8.3, 2.5), (8.3, 5.5), (3.0, 6.0)),
    ),
    "marsh": Preset(
        title="Forest and Marsh",
        summary="A marsh around a lake fills the middle of the front and forests cover "
        "the wings: slow going, attrition, and two separate fronts north and south.",
        seed=404,
        patches=(
            Patch("forest", "blob", (8.8, 0.8), 1.9, 0.35),
            Patch("forest", "blob", (5.8, 7.0), 2.0, 0.35),
            Patch("forest", "blob", (2.6, 2.0), 1.4, 0.35),
            Patch("marsh", "blob", (12.0, 4.0), 2.3, 0.4),
            Patch("marsh", "blob", (7.0, 4.2), 0.9, 0.35),
            Patch("hills", "blob", (0.9, 5.8), 0.9),
        ),
        rivers=(River(((4.4, 3.6), (7.0, 4.3), (9.8, 4.0)), large=True),),
        lakes=((11.4, 4.0),),
        cities=((6.6, 5.0), (7.2, 1.6), (8.4, 6.6), (2.6, 3.0)),
    ),
    "bay": Preset(
        title="Two Bays",
        summary="The sea cuts in from the north and the south where the countries meet, "
        "leaving a four-province isthmus: one narrow front, and coasts to hold.",
        seed=505,
        bays=((11.5, 0.5), (10.5, 0.5), (11.5, 1.5), (9.5, 0.5), (10.5, 1.5)),
        patches=(
            Patch("hills", "blob", (11.2, 3.8), 0.9),
            Patch("forest", "blob", (8.2, 2.6), 0.9),
            Patch("forest", "blob", (3.0, 1.4), 1.2),
            Patch("marsh", "blob", (8.6, 0.6), 0.6),
            Patch("hills", "blob", (1.4, 6.2), 1.0),
        ),
        rivers=(River(((4.0, 3.4), (6.4, 2.4), (8.0, 1.4))),),
        cities=((6.4, 4.2), (9.0, 1.8), (10.0, 5.2), (3.0, 6.0)),
    ),
    "salient": Preset(
        title="Two Salients",
        summary="The border bends round two bulges: Blue's pushes into Red in the north, "
        "Red's into Blue in the south. Each can be cut off at its base, and each is a "
        "springboard.",
        seed=606,
        trades=((12.5, 1.5), (12.5, 2.5), (13.5, 2.0)),
        patches=(
            Patch("hills", "blob", (10.6, 2.2), 0.8),
            Patch("forest", "blob", (8.8, 5.6), 1.0),
            Patch("forest", "blob", (4.0, 1.4), 1.1),
            Patch("hills", "blob", (1.2, 5.4), 1.1),
            Patch("marsh", "blob", (6.4, 7.6), 0.6),
            Patch("forest", "blob", (14.6, 4.6), 0.7),
        ),
        rivers=(
            River(((9.6, 3.4), (8.2, 4.4), (6.0, 4.0), (3.0, 4.6), (0.0, 4.2))),
            River(((8.6, 0.8), (7.2, 2.2), (6.1, 3.9)), joins=0),
        ),
        cities=((6.2, 3.0), (7.6, 1.2), (5.0, 6.4), (10.4, 4.6)),
    ),
}


# ---------------------------------------------------------------------------------------
# Fields and symmetry.


def smooth(rng, shape, scale, mirrored=True):
    """Smooth noise with unit spread and features about `scale` pixels across.

    A B-spline through random samples `scale` apart, evaluated on a grid a sixth of that
    spacing and stretched linearly to full size: the same field to the eye, at a tenth of
    the cost of evaluating the spline at every pixel. `mirrored` makes it the same turned
    round, so a picture painted from it needs no seam where its halves meet: copying one
    half's turn onto the other left a 42-byte cliff down the middle of the heightmap.
    """
    import cv2

    rows, columns = shape
    step = max(1, scale // 6)
    coarse = rng.standard_normal((rows // scale + 4, columns // scale + 4)).astype(np.float32)
    middle = ndimage.zoom(coarse, scale / step, order=3, prefilter=False)
    size = (middle.shape[1] * step, middle.shape[0] * step)
    field_ = cv2.resize(middle, size, interpolation=cv2.INTER_LINEAR)[:rows, :columns]
    if mirrored:
        field_ = field_ + field_[::-1, ::-1]
    return (field_ - field_.mean()) / (field_.std() + 1e-6)


def blur(pixels, sigma):
    """A Gaussian blur, taken at a quarter of the resolution when it is wide."""
    import cv2

    pixels = pixels.astype(np.float32)
    if sigma < 8:
        return cv2.GaussianBlur(pixels, (0, 0), sigma)
    rows, columns = pixels.shape
    small = cv2.resize(pixels, (columns // 4, rows // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4)
    return cv2.resize(small, (columns, rows), interpolation=cv2.INTER_LINEAR)


def distance(mask):
    """Each pixel's distance to the nearest pixel outside `mask` (0 outside it)."""
    import cv2

    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)


def symmetric(pixels):
    """The western half's half turn copied onto the eastern half, exactly."""
    half = pixels.shape[1] // 2
    out = pixels.copy()
    out[:, half:] = pixels[::-1, :half][:, ::-1]
    return out


def mirror_ids(ids, half_count):
    """Province ids turned round: the western half's i is the eastern half's i + half."""
    return (ids + half_count - 1) % (2 * half_count) + 1


def warp_field(rng, shape, amplitude):
    """A displacement that turns with the map, w(half turn of p) = -w(p).

    A Voronoi of half-turn-symmetric seeds, taken in coordinates displaced this way, stays
    half-turn symmetric, and its borders wander like the stock map's instead of running
    ruler-straight. The field is scaled down wherever it would fold space over (the
    Jacobian of p + w(p) kept above 0.3), since a fold is what pinches a province apart.
    """
    # None at the map's left and right edges, where it wraps: the fix for four-way
    # corners never visits that seam, so it keeps the plain arena's regular lattice.
    columns = shape[1]
    edge = np.minimum(np.arange(columns), columns - 1 - np.arange(columns))
    fade = np.clip((edge - 150) / 300, 0, 1).astype(np.float32)
    fade = fade * fade * (3 - 2 * fade)
    fields = []
    for _ in range(2):
        noise = 0.8 * smooth(rng, shape, 180, False) + 0.2 * smooth(rng, shape, 60, False)
        fields.append(amplitude * fade * (noise - noise[::-1, ::-1]) / 2)
    wx, wy = fields
    jacobian = (1 + np.gradient(wx, axis=1)) * (1 + np.gradient(wy, axis=0)) - np.gradient(
        wx, axis=0
    ) * np.gradient(wy, axis=1)
    if jacobian.min() < 0.3:
        scale = 0.7 / (1 - jacobian.min())
        wx, wy = wx * scale, wy * scale
    return wx, wy


def voronoi(points, shape, warp, half_count):
    """Province ids for every pixel: the nearest seed in warped coordinates.

    Only the western half of the bitmap is computed; the eastern half is its half turn,
    so the map is symmetric exactly rather than up to ties.
    """
    rows, columns = shape
    half = columns // 2
    yy, xx = np.mgrid[:rows, :half]
    wx, wy = warp
    query = np.stack([(xx + wx[:, :half]).ravel(), (yy + wy[:, :half]).ravel()], 1)
    left = cKDTree(points).query(query, workers=4)[1].reshape(rows, half) + 1
    ids = np.empty(shape, np.int64)
    ids[:, :half] = left
    ids[:, half:] = mirror_ids(left[::-1, ::-1], half_count)
    return ids


def merge_fragments(ids):
    """Give each detached piece of a province to the neighbour it borders most.

    A warped Voronoi can pinch a cell, and a province in two pieces is two places with
    one name. Returns the number of pixels moved.
    """
    moved = 0
    rows, columns = ids.shape
    for i, box in enumerate(ndimage.find_objects(ids), 1):
        if box is None:
            continue
        y0, y1 = max(box[0].start - 1, 0), min(box[0].stop + 1, rows)
        x0, x1 = max(box[1].start - 1, 0), min(box[1].stop + 1, columns)
        window = ids[y0:y1, x0:x1]
        pieces, count = ndimage.label(window == i)
        if count < 2:
            continue
        sizes = np.bincount(pieces.ravel())
        sizes[0] = 0
        keep = int(sizes.argmax())
        for piece in range(1, count + 1):
            if piece == keep:
                continue
            mask = pieces == piece
            ring = ndimage.binary_dilation(mask) & ~mask
            around = window[ring]
            around = around[around != i]
            if len(around):
                window[mask] = np.bincount(around).argmax()
                moved += int(mask.sum())
    return moved


def anchors(ids, count, half_count):
    """Each province's point furthest inside it, as (x, y): where its counters, its
    buildings and its weather stand. The eastern half's are the western half's turned."""
    edge = np.zeros(ids.shape, bool)
    vertical, horizontal = ids[:-1] != ids[1:], ids[:, :-1] != ids[:, 1:]
    edge[:-1] |= vertical
    edge[1:] |= vertical
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    inside = distance(~edge)
    found = ndimage.maximum_position(inside, ids, np.arange(1, count + 1))
    points = np.array([(x, y) for y, x in found], dtype=np.int64)
    rows, columns = ids.shape
    points[half_count:] = [columns - 1, rows - 1] - points[:half_count]
    return points


# ---------------------------------------------------------------------------------------
# The design, province by province.


def _polyline_distance(point, line):
    """Distance from `point` to the polyline through `line`'s points."""
    p = np.asarray(point, float)
    best = np.inf
    for a, b in zip(line, line[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * (b - a)))))
    return best


def _covers(patch, point, wobble):
    x, y = point
    if patch.shape == "rect":
        x0, y0, x1, y1 = patch.at
        return x0 <= x < x1 and y0 <= y < y1
    if patch.shape == "blob":
        distance = float(np.hypot(x - patch.at[0], y - patch.at[1]))
    elif patch.shape == "line":
        distance = _polyline_distance(point, patch.at)
    else:
        raise ValueError(f"unknown patch shape {patch.shape}")
    return distance / patch.size + patch.rough * wobble < 1


def design_terrain(preset, cells, rng):
    """Terrain for each western land cell, from its position in design units.

    A patch covers a cell if it covers the cell's position or the position's half turn,
    so whatever is drawn anywhere on the grid is drawn for both sides.
    """
    wobble = rng.uniform(-0.5, 0.5, len(cells))
    terrain = []
    for (x, y), shake in zip(cells, wobble):
        twin = (DESIGN_COLUMNS - x, DESIGN_ROWS - y)
        chosen = preset.base
        for patch in preset.patches:
            if _covers(patch, (x, y), shake) or _covers(patch, twin, shake):
                chosen = patch.terrain
        terrain.append(chosen)
    return terrain


# ---------------------------------------------------------------------------------------
# Rivers.


def _neighbours4(mask):
    padded = np.pad(mask, 1)
    return (
        padded[:-2, 1:-1].astype(np.int8) + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    )


def _touches(mask, eight=False):
    """Pixels next to `mask`: sharing an edge, or a corner too with `eight`."""
    structure = np.ones((3, 3), bool) if eight else ndimage.generate_binary_structure(2, 1)
    return ndimage.binary_dilation(mask, structure) & ~mask


def cheapest_path(allowed, cost, start, goal):
    """The cheapest 4-connected path over `allowed` from `start` (y, x) to any `goal` pixel.

    Each step costs the cost of the pixel stepped onto. A cheapest path never touches
    itself, so it is one pixel wide wherever it goes.
    """
    ys, xs = np.nonzero(allowed)
    index = np.full(allowed.shape, -1, np.int64)
    index[ys, xs] = np.arange(len(ys))
    sources, targets = [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = allowed[: allowed.shape[0] - dy, : allowed.shape[1] - dx]
        b = allowed[dy:, dx:]
        both = a & b
        ay, ax = np.nonzero(both)
        first, second = index[ay, ax], index[ay + dy, ax + dx]
        sources += [first, second]
        targets += [second, first]
    sources, targets = np.concatenate(sources), np.concatenate(targets)
    weights = cost[ys[targets], xs[targets]]
    graph = csr_matrix((weights, (sources, targets)), shape=(len(ys), len(ys)))
    origin = index[start]
    if origin < 0:
        raise ValueError("a river's source is not on an allowed pixel")
    distance, previous = dijkstra(graph, indices=origin, return_predecessors=True)
    ends = index[goal & allowed]
    ends = ends[ends >= 0]
    if not len(ends) or not np.isfinite(distance[ends]).any():
        raise ValueError("a river cannot reach its mouth along province borders")
    node = int(ends[np.argmin(distance[ends])])
    path = []
    while node >= 0:
        path.append((int(ys[node]), int(xs[node])))
        node = int(previous[node])
    return path[::-1]


def draw_rivers(specs, ids, kind, owner, to_pixel):
    """Every river and its half turn as pixel paths, source first.

    `kind` is a pixel map (0 sea, 1 land, 2 lake) and `owner` one of which country's
    land a pixel is (1 Blue, 2 Red, 0 neither). Rivers run only along borders between two
    land provinces, so every one of them is crossed somewhere, and keep a pixel clear of
    each other except where a tributary joins. Returns (paths, joins): joins[k] is true
    when paths[k] ends on another river rather than in water.
    """
    rows, columns = ids.shape
    land = kind == 1
    water = ~land
    border = np.zeros_like(land)
    for a, b, sl_a, sl_b in (
        (ids[:-1], ids[1:], np.s_[:-1], np.s_[1:]),
        (ids[:, :-1], ids[:, 1:], np.s_[:, :-1], np.s_[:, 1:]),
    ):
        differ = (a != b) & land[sl_a] & land[sl_b]
        border[sl_a] |= differ
        border[sl_b] |= differ
    seam = np.zeros_like(land)
    for sl_a, sl_b in ((np.s_[:-1], np.s_[1:]), (np.s_[:, :-1], np.s_[:, 1:])):
        across = (owner[sl_a] > 0) & (owner[sl_b] > 0) & (owner[sl_a] != owner[sl_b])
        seam[sl_a] |= across
        seam[sl_b] |= across
    near_water = ndimage.binary_dilation(water, np.ones((3, 3), bool), iterations=3)
    mouth = border & _touches(water) & land
    drawn = np.zeros_like(land)
    paths, joins = [], []
    heads = {}
    for number, spec in enumerate(specs):
        course = [to_pixel(x, y) for x, y in spec.course]
        xs = [p[0] for p in course]
        ys = [p[1] for p in course]
        margin = 260
        y0, y1 = max(int(min(ys)) - margin, 0), min(int(max(ys)) + margin, rows)
        x0, x1 = max(int(min(xs)) - margin, 0), min(int(max(xs)) + margin, columns)
        window = np.s_[y0:y1, x0:x1]
        line = np.zeros((y1 - y0, x1 - x0), bool)
        for (ax, ay), (bx, by) in zip(course, course[1:]):
            steps = int(max(abs(bx - ax), abs(by - ay))) + 1
            for t in np.linspace(0, 1, steps):
                py, px = int(round(ay + t * (by - ay))) - y0, int(round(ax + t * (bx - ax))) - x0
                if 0 <= py < line.shape[0] and 0 <= px < line.shape[1]:
                    line[py, px] = True
        off_course = ndimage.distance_transform_edt(~line)
        cost = 1 + 6 * (off_course / 30.0) ** 2 + 40 * near_water[window]
        main = np.zeros_like(land)
        if spec.joins is not None:
            for path in heads[spec.joins]:
                main[tuple(np.array(path).T)] = True
        # Two pixels clear of every other river, so no two ever run side by side.
        others = ndimage.binary_dilation(drawn & ~main, np.ones((3, 3), bool), iterations=2)
        clear = ~others[window]
        allowed = border[window] & clear
        if spec.seam:
            allowed &= seam[window]
        if spec.joins is None:
            end_x, end_y = course[-1]
            grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
            reach = np.hypot(grid_x - end_x, grid_y - end_y)
            goal = mouth[window] & clear & (reach < 200)
        else:
            main = main[window]
            # The join is the only pixel allowed next to the main river, and it touches
            # exactly one of the main river's pixels, as every stock join does.
            goal = allowed & (_neighbours4(main) == 1) & ~main
            allowed = (allowed & ~main & ~_touches(main, eight=True)) | goal
        allowed = allowed | goal
        start_x, start_y = course[0]
        candidates = np.argwhere(allowed & ~goal & (near_water[window] == 0))
        if not len(candidates):
            raise ValueError(f"river {number} has nowhere to start")
        nearest = candidates[
            np.argmin(np.hypot(candidates[:, 0] + y0 - start_y, candidates[:, 1] + x0 - start_x))
        ]
        path = cheapest_path(allowed, cost, tuple(nearest), goal)
        path = [(py + y0, px + x0) for py, px in path]
        twin = [(rows - 1 - py, columns - 1 - px) for py, px in path]
        mask = np.zeros_like(land)
        mask[tuple(np.array(path).T)] = True
        twin_mask = np.zeros_like(land)
        twin_mask[tuple(np.array(twin).T)] = True
        if (ndimage.binary_dilation(mask, np.ones((3, 3), bool), iterations=2) & twin_mask).any():
            raise ValueError(f"river {number} runs into its own half turn")
        drawn |= mask | twin_mask
        heads[number] = (path, twin)
        for course_path in (path, twin):
            paths.append((course_path, spec.large))
            joins.append(spec.joins is not None)
    return paths, joins


def river_pixels(paths, joins, shape):
    """rivers.bmp indices for the drawn rivers: -1 where there is none."""
    out = np.full(shape, -1, np.int16)
    for (path, large), joined in zip(paths, joins):
        widths = RIVER_WIDTHS[large]
        count = len(path)
        for k, (y, x) in enumerate(path):
            fraction = k / max(count - 1, 1)
            out[y, x] = next(index for limit, index in widths if fraction < limit)
        out[path[0]] = RIVER_SOURCE
        if joined:
            out[path[-1]] = RIVER_JOIN
    return out


# ---------------------------------------------------------------------------------------
# Pictures.


def relief(rng, types, kind, rivers):
    """The heightmap: relief by terrain type, coasts ramped down to the water, rivers in
    shallow valleys. `types` holds a LAND_TYPES index per pixel (-1 for water)."""
    shape = types.shape
    base = np.full(shape, float(WATER_FLOOR), np.float32)
    spread = np.zeros(shape, np.float32)
    for n, name in enumerate(LAND_TYPES):
        level, rough = RELIEF[name]
        base[types == n] = level
        spread[types == n] = rough
    land = kind == 1
    # Blur the levels so the ground climbs across province borders instead of stepping.
    base = blur(np.where(land, base, 100.0), 18)
    spread = blur(spread, 14)
    detail = (
        smooth(rng, shape, 150) + 0.55 * smooth(rng, shape, 64) + 0.3 * smooth(rng, shape, 26)
    ) / 1.2
    ridges = np.clip(1 - np.abs(smooth(rng, shape, 110)), 0, 1)
    mountain = blur(types == LAND_TYPES.index("mountain"), 10)
    shape_ = (1 - mountain) * detail + mountain * (1.8 * ridges**2 - 0.2 + 0.4 * detail)
    ground = base + spread * shape_
    if rivers is not None and (rivers >= 0).any():
        # Rivers lie in shallow valleys, five bytes deep at the water.
        valley = blur(rivers >= 0, 4)
        ground = ground - 5 * valley / valley.max()
    ground = np.maximum(ground, DRY_LAND)
    # Coasts: the land falls to sea level at the water's edge and the bed to the floor.
    to_water = distance(land)
    to_land = distance(~land)
    rise = np.clip(to_water / COAST_PIXELS, 0, 1)
    rise = rise * rise * (3 - 2 * rise)
    fall = np.clip(to_land / 6, 0, 1)
    heights = np.where(
        land,
        SEA_LEVEL + (ground - SEA_LEVEL) * rise,
        SEA_LEVEL - (SEA_LEVEL - WATER_FLOOR) * fall,
    )
    return symmetric(np.clip(np.round(heights), 0, 255).astype(np.uint8))


def normals(heights):
    """world_normal.bmp from the heightmap, at half its resolution.

    Stock: red falls as the ground rises eastward, green rises as it rises southward,
    blue is near 255, at about 0.05 per byte of height a half-resolution pixel, which is
    the physical slope (a byte is 0.1 world units, a half-resolution pixel two).
    """
    rows, columns = heights.shape
    half = heights.astype(np.float32).reshape(rows // 2, 2, columns // 2, 2).mean(axis=(1, 3))
    gx = np.gradient(half, axis=1) * 0.05
    gy = np.gradient(half, axis=0) * 0.05
    vector = np.stack([-gx, gy, np.ones_like(gx)], -1)
    vector /= np.linalg.norm(vector, axis=-1, keepdims=True)
    return symmetric(np.clip(np.round(127.5 + 127.5 * vector), 0, 255).astype(np.uint8))


def terrain_pixels(rng, types, heights, city_mask):
    """terrain.bmp indices: each type painted with its stock variants in patches, bare rock
    above the rock line, and the city where one stands. `types` holds a LAND_TYPES index
    per pixel (-1 for water)."""
    shape = types.shape
    out = np.full(shape, WATER_INDEX, np.uint8)
    patches, clumps = smooth(rng, shape, 44), smooth(rng, shape, 30)
    plains = np.where(patches > 0.45, FARMLAND, PLAINS)
    variants = {
        "plains": plains,
        "forest": np.where(clumps > 0.25, LIGHT_FOREST, DARK_FOREST),
        "hills": np.where(patches > 0.1, RIDGED_HILLS, ROLLING_HILLS),
        "mountain": np.where(heights >= ROCK_LINE, ROCK, GREEN_MOUNTAIN),
        "marsh": np.full(shape, MARSH),
        "urban": np.where(city_mask, CITY, plains),
    }
    for n, name in enumerate(LAND_TYPES):
        where = types == n
        out[where] = variants[name][where]
    return symmetric(out)


def city_layers(rng, ids, cities, anchors):
    """Where each city is built: (urban mask, cities.bmp indices, city lights).

    A city covers CITY_SHARE of its province around the province's inmost point, with a
    ragged edge; the middle CITY_CORE of it is dense city, the rest houses.
    """
    shape = ids.shape
    urban = np.zeros(shape, bool)
    style = np.full(shape, -1, np.int16)
    lights = np.zeros(shape, np.uint8)
    ragged = smooth(rng, shape, 18)
    for province in cities:
        ys, xs = np.nonzero(ids == province)
        x0, y0 = anchors[province - 1]
        score = np.hypot(xs - x0, ys - y0) * (1 + 0.35 * ragged[ys, xs])
        built = score <= np.quantile(score, CITY_SHARE)
        core = score <= np.quantile(score, CITY_CORE)
        urban[ys[built], xs[built]] = True
        style[ys[built], xs[built]] = CITY_SUBURB_INDEX
        style[ys[core], xs[core]] = CITY_CORE_INDEX
        lights[ys[built], xs[built]] = CITY_LIGHTS[1]
        lights[ys[core], xs[core]] = CITY_LIGHTS[0]
    return symmetric(urban), symmetric(style), symmetric(lights)


def tree_pixels(rng, types, size):
    """trees.bmp at its own resolution: woods in the forests, a few trees elsewhere."""
    columns, rows = size
    height, width = types.shape
    ty = ((np.arange(rows) + 0.5) * height / rows).astype(int)
    tx = ((np.arange(columns) + 0.5) * width / columns).astype(int)
    sampled = types[ty][:, tx]
    cover = np.zeros(sampled.shape, np.float32)
    for n, name in enumerate(LAND_TYPES):
        cover[sampled == n] = TREE_COVER.get(name, 0.0)
    # Clumped rather than scattered: a smooth field, thresholded at the cover share.
    clumps = ndimage.gaussian_filter(rng.standard_normal(sampled.shape).astype(np.float32), 1.2)
    clumps = clumps + clumps[::-1, ::-1]
    rank = ndtr((clumps - clumps.mean()) / (clumps.std() + 1e-6))
    kind = smooth(rng, sampled.shape, 12)
    trees = np.where(rank < cover, np.where(kind > -0.3, TREE_INDICES[0], TREE_INDICES[1]), 0)
    return symmetric(trees.astype(np.uint8))


def colour_map(rng, graphical, lights, ocean):
    """The colour map at half resolution: ground colour by painted terrain, gently mottled,
    with the city lights in its alpha."""
    half = graphical[::2, ::2]
    colour = np.empty((*half.shape, 4), np.float32)
    colour[...] = ocean
    for index, rgb in GROUND.items():
        colour[half == index, :3] = rgb
    land = half != WATER_INDEX
    colour[..., :3] = ndimage.gaussian_filter(colour[..., :3], (1.5, 1.5, 0))
    mottle = 1 + 0.045 * smooth(rng, half.shape, 50)
    colour[..., :3] = np.where(land[..., None], colour[..., :3] * mottle[..., None], ocean[:3])
    colour[..., 3] = np.where(land, lights[::2, ::2], ocean[3])
    return symmetric(np.clip(np.round(colour), 0, 255).astype(np.uint8))
