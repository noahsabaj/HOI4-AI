"""Generate an arena for three or four nations, each holding the same share turned.

The two-country arena (mapgen.generate) copies one half a half turn onto the other. Here
one nation's share is turned N times about the map's centre:

- **Four nations** stand on a square lattice of 23 by 23 cells filling the middle
  2048 by 2048 of the map. A quarter turn maps the square's pixels onto themselves, so the
  province bitmap and every picture are exact quarter-turn copies, as the two-country
  arena's are half-turn copies. Nation 0 holds the cells right of and below the centre
  (a 9 by 10 block), and the others are its turns: a pinwheel round one lake cell at the
  centre, where no nation stands (four provinces meeting at a pixel is a corner the
  engine cannot trace, and a turn fixes the centre).
- **Three nations** stand on a hexagonal lattice whose triangle's centre is the map's, so
  a third of a turn maps it onto itself and the three meet at a point. A third of a turn
  does not map pixels onto pixels, so here the symmetry is exact in everything the game
  plays on (which provinces there are, who holds them, their terrain, their neighbours,
  coasts, states, victory points, hubs, railways and divisions) and holds to about a pixel
  in the pictures. `check_turn` checks the province graph and the generator redraws with
  a gentler warp if the pixels ever broke it.

Everything outside the turned core (the square, or a disk of radius 1000 px) is open sea
on a regular lattice, as the two-country arena's is. The land is drawn as the presets'
is (arenas.py): stock terrain in regions, relief, cities with the victory points, trunk
railways, trees; there are no rivers yet.

Province ids are laid out so the turn is arithmetic: the core's provinces come in orbits
of N, id o*N + k + 1 being orbit o's copy in nation k, then the provinces the turn fixes
(the four-nation centre), then the outer sea. The nation-k copy of anything is the
nation-0 one turned k times.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from . import arenas, multination
from .mapgen import (
    ARMY_SPEED_FACTOR,
    CITY_INDEX,
    COASTAL_BUILDINGS,
    COMMANDER_PORTRAIT,
    FOG_LAND,
    FOG_SEA,
    GENERALS_PER_COUNTRY,
    LAKE,
    LAND,
    LAND_STACKS,
    LAYOUT,
    MAP_SIZE,
    MINIMAP_LAND,
    MINIMAP_SEA,
    MINIMAP_SIZES,
    MONTH_LAST_DAY,
    OCEAN_COLOUR,
    PORT_BUILDINGS,
    PORT_STACKS,
    PROVINCE_BUILDINGS,
    SEA,
    SEA_STACKS,
    STATE_BUILDINGS,
    TREES_DENOMINATOR,
    TREES_NUMERATOR,
    VICTORY_POINT_NAMES,
    WATER_COLOUR,
    adjacency,
    write_dds,
)
from .state_channel import daily_effect, startup_effect

WIDTH, HEIGHT = MAP_SIZE
CENTRE = ((WIDTH - 1) / 2, (HEIGHT - 1) / 2)
# The four-nation core: the middle square of the map, 23 cells a side, the cell at the
# centre a lake. Land reaches 9 cells out from the centre and two rings of sea surround it.
SQUARE, SQUARE_X0 = HEIGHT, (WIDTH - HEIGHT) // 2
QUAD_CELLS, QUAD_LAND = 23, 9
QUAD_STEP = SQUARE / QUAD_CELLS
# The three-nation core: hexagonal cells 88 px apart (the two-country arena's province
# size), land out to 800 px from the centre (about 100 provinces a nation) and sea to
# 1000 px, two and a half rings.
HEX_STEP, HEX_LAND, CORE_RADIUS = 88.0, 800.0, 1000.0
# The open sea around the core, a regular lattice as on the two-country arena: rows of
# 85.3 px, alternate columns offset a quarter row.
OUTER_ROWS = 24
# Merged into a neighbour: an outer sea province this small, cut off by the core's rim.
SLIVER = 0.25


def _smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


class Turn:
    """The arena's symmetry: N turns about the map's centre, in any picture's own pixels.

    Nation k's share is nation 0's turned k times by `rotate`, which for four nations is
    numpy's rot90 (the pixel (y, x) of the square goes to (S - 1 - x, y)). `field` makes a
    scalar field the same turned; `copy` makes a picture an exact copy of its first quarter
    turned (four nations), and leaves it alone for three, whose turn is not a pixel map.
    """

    def __init__(self, n):
        if n not in (3, 4):
            raise ValueError("a multi-nation arena has 3 or 4 nations")
        self.n = n
        self.exact = n == 4
        self.angle = -2 * math.pi / n

    def rotate(self, xy, k=1):
        """Points (x, y) of the full-size map turned k times about the centre."""
        xy = np.asarray(xy, dtype=np.float64)
        phi = self.angle * k
        c, s = math.cos(phi), math.sin(phi)
        x, y = xy[..., 0] - CENTRE[0], xy[..., 1] - CENTRE[1]
        return np.stack([c * x - s * y + CENTRE[0], s * x + c * y + CENTRE[1]], -1)

    def turn_vector(self, v, k=1):
        v = np.asarray(v, dtype=np.float64)
        phi = self.angle * k
        c, s = math.cos(phi), math.sin(phi)
        return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)

    @staticmethod
    def _square(shape):
        rows, columns = shape[:2]
        x0 = (columns - rows) // 2
        return x0, rows

    def field(self, f):
        """`f` made the same turned, within the core; outside it, left as it was."""
        f = np.array(f, dtype=np.float32)
        if self.exact:
            x0, side = self._square(f.shape)
            sub = f[:, x0 : x0 + side]
            f[:, x0 : x0 + side] = sum(np.rot90(sub, k) for k in range(4)) / 4
            return f
        rows, columns = f.shape
        radius = CORE_RADIUS * rows / HEIGHT
        cx, cy = (columns - 1) / 2, (rows - 1) / 2
        y0, y1 = max(0, int(cy - radius)), min(rows, int(cy + radius) + 2)
        x0, x1 = max(0, int(cx - radius)), min(columns, int(cx + radius) + 2)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        inside = dx * dx + dy * dy <= radius * radius
        total = np.zeros_like(dx)
        for k in range(self.n):
            c, s = math.cos(self.angle * k), math.sin(self.angle * k)
            total += map_coordinates(
                f, [s * dx + c * dy + cy, c * dx - s * dy + cx], order=1, mode="nearest"
            )
        window = f[y0:y1, x0:x1]
        window[inside] = (total / self.n)[inside]
        return f

    def vector_field(self, gx, gy):
        """A displacement w with w(turned p) = w(p) turned, within the core: the average
        of the field (gx, gy) read at each turn of p and turned back. Zero outside."""
        wx, wy = np.zeros_like(gx, np.float32), np.zeros_like(gy, np.float32)
        if self.exact:
            x0, side = self._square(gx.shape)
            sx, sy = gx[:, x0 : x0 + side], gy[:, x0 : x0 + side]
            tx, ty = wx[:, x0 : x0 + side], wy[:, x0 : x0 + side]
            for k in range(4):
                # rot90(a, -k) reads a at the k-th turn of each pixel; a quarter turn back
                # takes a vector (x, y) to (-y, x).
                ax, ay = np.rot90(sx, -k), np.rot90(sy, -k)
                for _ in range(k):
                    ax, ay = -ay, ax
                tx += ax
                ty += ay
            return wx / 4, wy / 4
        rows, columns = gx.shape
        cx, cy = (columns - 1) / 2, (rows - 1) / 2
        radius = CORE_RADIUS
        y0, y1 = max(0, int(cy - radius)), min(rows, int(cy + radius) + 2)
        x0, x1 = max(0, int(cx - radius)), min(columns, int(cx + radius) + 2)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        inside = dx * dx + dy * dy <= radius * radius
        totals = [np.zeros_like(dx), np.zeros_like(dx)]
        for k in range(self.n):
            c, s = math.cos(self.angle * k), math.sin(self.angle * k)
            at = [s * dx + c * dy + cy, c * dx - s * dy + cx]
            ax = map_coordinates(gx, at, order=1, mode="nearest")
            ay = map_coordinates(gy, at, order=1, mode="nearest")
            # Turned back: by -k turns.
            totals[0] += c * ax + s * ay
            totals[1] += -s * ax + c * ay
        wx[y0:y1, x0:x1][inside] = (totals[0] / self.n)[inside]
        wy[y0:y1, x0:x1][inside] = (totals[1] / self.n)[inside]
        return wx, wy

    def _quarters(self, side):
        first = np.zeros((side, side), bool)
        first[: side // 2, : side // 2] = True
        return [np.rot90(first, k) for k in range(4)]

    def copy(self, picture):
        """The square's first quarter (top left) turned onto the other three, exactly."""
        if not self.exact:
            return picture
        out = picture.copy()
        x0, side = self._square(picture.shape)
        sub = picture[:, x0 : x0 + side]
        target = out[:, x0 : x0 + side]
        for k, mask in enumerate(self._quarters(side)):
            if k:
                target[mask] = np.rot90(sub, k)[mask]
        return out

    def copy_ids(self, ids, sigma):
        """Province ids made exact: each quarter the first turned, ids moved to the turned
        provinces' (`sigma[i]` is the id of province i turned once)."""
        x0, side = self._square(ids.shape)
        sub = ids[:, x0 : x0 + side].copy()
        target = ids[:, x0 : x0 + side]
        turned = np.arange(len(sigma))
        for k, mask in enumerate(self._quarters(side)):
            if k:
                turned = sigma[turned]
                target[mask] = turned[np.rot90(sub, k)][mask]
        return ids

    def pixel(self, y, x, k=1):
        """A pixel of the square turned k times (four nations only)."""
        for _ in range(k % 4):
            y, x = SQUARE - 1 - (x - SQUARE_X0), y + SQUARE_X0
        return y, x


class Core:
    """The turned core's cells: canonical ones (nation 0's share and its sea) and their
    turns, with their seeds, kinds and nations. Built for a number of nations and a
    preset; everything random comes from `rng`."""

    def __init__(self, turn, design, rng, jitter_scale=1.0):
        self.turn = turn
        n = turn.n
        self.design = design
        if n == 4:
            self.step = QUAD_STEP
            canonical, fixed = [], []
            for u in range(-11, 12):
                for v in range(-11, 12):
                    if u > 0 and v >= 0:
                        canonical.append((u, v))
            canonical.sort(key=lambda p: (max(abs(p[0]), abs(p[1])), p[0], p[1]))
            positions = np.array(
                [(CENTRE[0] + u * self.step, CENTRE[1] + v * self.step) for u, v in canonical]
            )
            fixed = [(CENTRE[0], CENTRE[1])]
            reach = np.array([max(abs(u), abs(v)) for u, v in canonical], float)
            land = reach <= QUAD_LAND
            jittered = reach <= QUAD_LAND + 1
            self.border_angle = math.pi / 4
            self.frame_angle = math.pi / 4
        else:
            self.step = HEX_STEP
            a = self.step
            origin = np.array(CENTRE) - [a / 2, a * math.sqrt(3) / 6]
            span = int(CORE_RADIUS / a) + 3
            found = []
            for j in range(-span, span + 1):
                for i in range(-2 * span, 2 * span + 1):
                    p = origin + [a * (i + j / 2), a * math.sqrt(3) / 2 * j]
                    if np.hypot(*(p - CENTRE)) <= CORE_RADIUS - a / 2:
                        found.append(p)
            found = np.array(found)
            beta = self._ray_angle(found)
            angles = np.mod(np.arctan2(found[:, 1] - CENTRE[1], found[:, 0] - CENTRE[0]) - beta,
                            2 * math.pi)  # fmt: skip
            keep = angles < 2 * math.pi / 3 - 1e-9
            positions = found[keep]
            radius = np.hypot(*(positions - CENTRE).T)
            order = np.lexsort((angles[keep], np.round(radius, 3)))
            positions = positions[order]
            radius = radius[order]
            fixed = []
            land = radius <= HEX_LAND
            jittered = radius <= HEX_LAND + a
            self.border_angle = beta
            self.frame_angle = beta + math.pi / 3
        self.orbits = len(positions)
        self.canonical = positions
        self.kind = np.where(land, LAND, SEA).astype(np.int8)
        # Seeds stray, nation 0's drawn and turned for the others.
        stray = rng.uniform(-design.jitter, design.jitter, positions.shape) * self.step
        stray *= jitter_scale
        stray[~jittered] = 0
        seeds = np.empty((self.orbits * n, 2))
        for k in range(n):
            seeds[k::n] = turn.rotate(positions + 0, k) + turn.turn_vector(stray, k)
        self.seeds = np.concatenate([seeds, np.array(fixed).reshape(-1, 2)])
        self.fixed = len(fixed)
        self.count = len(self.seeds)
        # Where each canonical cell stands in the nation frame, in cells from the centre.
        self.frame = self.to_frame(positions)
        for point in design.lakes:
            cell = self.nearest(point)
            self.kind[cell] = LAKE

    def to_frame(self, pixels):
        """Map pixels to the nation frame: cells from the centre, +x through nation 0."""
        d = (np.asarray(pixels, float) - CENTRE) / self.step
        c, s = math.cos(self.frame_angle), math.sin(self.frame_angle)
        return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)

    def from_frame(self, point):
        c, s = math.cos(self.frame_angle), math.sin(self.frame_angle)
        x, y = point
        return np.array(CENTRE) + self.step * np.array([c * x - s * y, s * x + c * y])

    def nearest(self, point):
        """The canonical cell nearest a nation-frame point, or nearest any of its turns."""
        best, cell = np.inf, 0
        for k in range(self.turn.n):
            at = self.turn.rotate(self.from_frame(point), -k)
            d = np.hypot(*(self.canonical - at).T)
            if d.min() < best:
                best, cell = d.min(), int(d.argmin())
        return cell

    def _ray_angle(self, points):
        """Where the three-nation borders run out from the centre: the angle whose three
        rays pass furthest from any land cell, so no cell sits on a border."""
        offsets = points - CENTRE
        radius = np.hypot(*offsets.T)
        near = radius <= HEX_LAND + self.step
        theta = np.arctan2(offsets[near, 1], offsets[near, 0])
        r = radius[near]
        best, chosen = -1.0, 0.0
        for beta in np.radians(np.arange(0.0, 120.0, 0.25)):
            margin = np.inf
            for k in range(3):
                d = theta - (beta + 2 * math.pi * k / 3)
                ahead = np.cos(d) > 0
                margin = min(margin, float((r[ahead] * np.abs(np.sin(d[ahead]))).min()))
            if margin > best + 1e-6:
                best, chosen = margin, beta
        return chosen

    def province(self, orbit, k):
        return orbit * self.turn.n + k + 1

    def sigma(self, total):
        """sigma[i]: the id of province i turned once (the identity off the orbits)."""
        n = self.turn.n
        sigma = np.arange(total + 1)
        ids = np.arange(1, self.orbits * n + 1)
        sigma[ids] = (ids - 1) // n * n + ((ids - 1) % n + 1) % n + 1
        return sigma


def _warp(rng, turn, core, amplitude):
    """A displacement that turns with the arena (Turn.vector_field) from the noise the
    two-country warp uses, as strong as that one, faded out before the core's rim so the
    rim keeps the lattice. Scaled down where it would fold space over, as
    arenas.warp_field is."""
    shape = (HEIGHT, WIDTH)
    fields = [
        0.8 * arenas.smooth(rng, shape, 180, False) + 0.2 * arenas.smooth(rng, shape, 60, False)
        for _ in range(2)
    ]
    gx, gy = turn.vector_field(*fields)
    del fields
    yy, xx = np.mgrid[:HEIGHT, :WIDTH].astype(np.float32)
    dx, dy = xx - CENTRE[0], yy - CENTRE[1]
    if turn.n == 4:
        reach = np.maximum(np.abs(dx), np.abs(dy))
        full, zero = (QUAD_LAND + 0.5) * core.step, (QUAD_LAND + 1.5) * core.step
    else:
        reach = np.hypot(dx, dy)
        full, zero = HEX_LAND + 0.5 * core.step, HEX_LAND + 1.5 * core.step
    fade = _smoothstep((zero - reach) / (zero - full)).astype(np.float32)
    del yy, xx, dx, dy, reach
    # The two-country warp's components have a spread of about 0.58 of its amplitude; an
    # average of N independent readings has 1/sqrt(N) of one's, made up here.
    strength = amplitude * 0.7 * math.sqrt(turn.n)
    wx, wy = strength * fade * gx, strength * fade * gy
    jacobian = (1 + np.gradient(wx, axis=1)) * (1 + np.gradient(wy, axis=0)) - np.gradient(
        wx, axis=0
    ) * np.gradient(wy, axis=1)
    if jacobian.min() < 0.3:
        scale = 0.7 / (1 - jacobian.min())
        wx, wy = wx * scale, wy * scale
    return wx, wy


def _outer_seeds(turn):
    """The open sea's lattice, outside the core, as the two-country arena's."""
    step_y = HEIGHT / OUTER_ROWS
    if turn.n == 4:
        columns = [(x0 + (c + 0.5) * (SQUARE_X0 / 20), c) for x0 in (0, SQUARE_X0 + SQUARE)
                   for c in range(20)]  # fmt: skip
    else:
        columns = [((c + 0.5) * WIDTH / 64, c) for c in range(64)]
    seeds = []
    for x, c in columns:
        for r in range(OUTER_ROWS):
            y = r * step_y + step_y / 2 + (c % 2) * step_y / 4
            if turn.n == 3 and math.hypot(x - CENTRE[0], y - CENTRE[1]) < CORE_RADIUS + 20:
                continue
            seeds.append((x, y))
    return np.array(seeds)


def _core_mask(turn):
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    if turn.n == 4:
        return (xx >= SQUARE_X0) & (xx < SQUARE_X0 + SQUARE)
    return (xx - CENTRE[0]) ** 2 + (yy - CENTRE[1]) ** 2 <= CORE_RADIUS**2


def _fix_crossings(ids, core_px, turn):
    """Break pixel-only four-way contacts, as mapgen does, keeping the turn: in the square
    each fix is made to its whole orbit, and at the core's rim only outer pixels move."""
    rows, columns = ids.shape
    for _ in range(5):
        wrapped = np.concatenate([ids, ids[:, :1]], axis=1)
        a, b = wrapped[:-1, :-1], wrapped[:-1, 1:]
        c, d = wrapped[1:, :-1], wrapped[1:, 1:]
        crossing = (a != b) & (a != c) & (a != d) & (b != c) & (b != d) & (c != d)
        found = np.argwhere(crossing)
        if not len(found):
            return
        for y, x in found:
            x1 = (x + 1) % columns
            block = [(y, x), (y, x1), (y + 1, x), (y + 1, x1)]
            if len({int(ids[p]) for p in block}) < 4:
                continue  # Fixed already, by an earlier fix of its orbit.
            outside = [p for p in block if not core_px[p]]
            if outside:
                sides = {0: (1, 2), 1: (0, 3), 2: (0, 3), 3: (1, 2)}
                p = block.index(outside[0])
                q = min(sides[p], key=lambda k: core_px[block[k]])
                ids[block[p]] = ids[block[q]]
            elif turn.exact:
                for k in range(4):
                    ty, tx = turn.pixel(y, x1, k)
                    sy, sx = turn.pixel(y, x, k)
                    ids[ty, tx] = ids[sy, sx]
            else:
                ids[y, x1] = ids[y, x]


def _merge_slivers(ids, first_outer):
    """Outer sea provinces the core's rim cut down to a sliver join their largest outer
    neighbour; then the outer ids are renumbered to follow the core's."""
    nominal = (WIDTH / 64) * (HEIGHT / OUTER_ROWS)
    areas = np.bincount(ids.ravel())
    for i in np.flatnonzero((areas > 0) & (areas < SLIVER * nominal)):
        if i < first_outer:
            continue
        mask = ids == i
        if not mask.any():
            continue
        ring = ndimage.binary_dilation(mask) & ~mask
        around = ids[ring]
        around = around[around >= first_outer]
        if len(around):
            ids[mask] = np.bincount(around).argmax()
    present = np.unique(ids[ids >= first_outer])
    lookup = np.arange(ids.max() + 1)
    lookup[present] = first_outer + np.arange(len(present))
    ids[...] = lookup[ids]
    return first_outer + len(present) - 1


def check_turn(neighbours, kinds, core, total):
    """Where the province graph breaks the turn: each land or lake province's neighbours,
    turned once, must be its turn's neighbours, and no land may touch the outer sea."""
    sigma = core.sigma(total)
    problems = []
    last_core = core.orbits * core.turn.n + core.fixed
    for i in range(1, last_core + 1):
        if kinds[i - 1] == SEA:
            continue
        for j in neighbours[i]:
            if j > last_core:
                problems.append(f"land province {i} touches the outer sea {j}")
            elif sigma[j] not in neighbours[sigma[i]]:
                problems.append(f"provinces {i} and {j} touch, but their turns do not")
    return problems


def _anchors(ids, count, turn, core):
    """Each province's point furthest inside it, as (x, y); on four nations the turns'
    are nation 0's turned, exactly."""
    edge = np.zeros(ids.shape, bool)
    vertical, horizontal = ids[:-1] != ids[1:], ids[:, :-1] != ids[:, 1:]
    edge[:-1] |= vertical
    edge[1:] |= vertical
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    inside = arenas.distance(~edge)
    found = ndimage.maximum_position(inside, ids, np.arange(1, count + 1))
    points = np.array([(x, y) for y, x in found], dtype=np.int64)
    if turn.exact:
        n = turn.n
        for orbit in range(core.orbits):
            x, y = points[orbit * n]
            for k in range(1, n):
                ty, tx = turn.pixel(y, x, k)
                points[orbit * n + k] = (tx, ty)
    return points


def _bisect(cells, positions, parts):
    """Split `cells` into `parts` groups of equal size (to one), cutting across the longer
    side of their bounding box each time: compact, balanced states."""
    if parts == 1:
        return [list(cells)]
    points = positions[cells]
    axis = int(np.ptp(points[:, 1]) > np.ptp(points[:, 0]))
    order = [cells[k] for k in np.lexsort((points[:, 1 - axis], points[:, axis]))]
    left_parts = parts // 2
    cut = round(len(cells) * left_parts / parts)
    return _bisect(order[:cut], positions, left_parts) + _bisect(
        order[cut:], positions, parts - left_parts
    )


@dataclass
class Drawn:
    """A multi-nation arena's provinces: the bitmap `ids`, the kind of each (`kinds[i - 1]`),
    their neighbours, how many there are, the core they were drawn from, and the attempt
    and random generator that drew them (the rest of the arena goes on drawing from it)."""

    turn: Turn
    core: Core
    ids: np.ndarray
    kinds: np.ndarray
    neighbours: dict
    total: int
    attempt: int
    rng: np.random.Generator


def draw_provinces(design, seed):
    """The province bitmap of a multi-nation arena, from its design and seed alone.

    Three nations' symmetry is exact only in the province graph, and a warp that folds a
    border into a near-tie can break it, so the bitmap is drawn again with a gentler warp
    and seeds that stray less, from the same seed, until check_turn passes.
    """
    turn = Turn(design.nations)
    n = turn.n
    core_px = _core_mask(turn)
    for attempt in range(5):
        rng = np.random.default_rng(seed if attempt == 0 else [seed, attempt])
        gentler = 0.7**attempt
        core = Core(turn, design, rng, jitter_scale=gentler)
        wx, wy = _warp(rng, turn, core, design.warp * gentler)
        ids = np.zeros((HEIGHT, WIDTH), np.int64)
        ys, xs = np.nonzero(core_px)
        query = np.stack([xs + wx[ys, xs], ys + wy[ys, xs]], 1)
        ids[ys, xs] = cKDTree(core.seeds).query(query, workers=4)[1] + 1
        del query, wx, wy
        outer = _outer_seeds(turn)
        ys, xs = np.nonzero(~core_px)
        ids[ys, xs] = cKDTree(outer).query(np.stack([xs, ys], 1), workers=4)[1] + core.count + 1
        del ys, xs
        sigma = core.sigma(core.count + len(outer))
        if turn.exact:
            turn.copy_ids(ids, sigma)
        arenas.merge_fragments(ids)
        if turn.exact:
            turn.copy_ids(ids, sigma)
        _fix_crossings(ids, core_px, turn)
        total = _merge_slivers(ids, core.count + 1)
        kinds = np.concatenate([np.repeat(core.kind, n), np.full(core.fixed, LAKE),
                                np.full(total - core.count, SEA)]).astype(np.int8)  # fmt: skip
        neighbours = adjacency(ids, total)
        broken = check_turn(neighbours, kinds, core, total)
        # A province the warp left no pixel is a province the engine cannot find.
        missing = np.flatnonzero(np.bincount(ids.ravel(), minlength=total + 1)[1:] == 0)
        broken += [f"province {i + 1} has no pixels" for i in missing]
        if not broken:
            return Drawn(turn, core, ids, kinds, neighbours, total, attempt, rng)
    raise RuntimeError(f"the province graph would not keep the turn: {broken[:3]}")


def generate_multi(game, output, *, preset, seed=None, wars=None, lone_bonus=0.0):
    """Write a multi-nation arena: `preset` names a design in multination.NATION_PRESETS,
    `seed` redraws its noise and province shapes, `wars` is a setup (multination.wars;
    the preset's default without one), and `lone_bonus` (0.25 is +25%) raises the attack
    and defence of every nation on a side smaller than the largest, off by default."""
    if preset not in multination.NATION_PRESETS:
        names = ", ".join(multination.NATION_PRESETS)
        raise ValueError(f"unknown multi-nation preset {preset!r}: choose from {names}")
    design = multination.NATION_PRESETS[preset]
    n = design.nations
    tags = multination.NATIONS[:n]
    plan = multination.wars(wars or design.wars, tags)
    if lone_bonus < 0:
        raise ValueError("the lone side's bonus is a fraction, 0 or more")
    game, root = Path(game), Path(output).resolve()
    if not (game / "map/provinces.bmp").exists():
        raise ValueError("Point --game at the installed HOI4 directory")
    seed = design.seed if seed is None else int(seed)
    root.mkdir(parents=True, exist_ok=False)
    drawn = draw_provinces(design, seed)
    turn, core, ids, kinds = drawn.turn, drawn.core, drawn.ids, drawn.kinds
    neighbours, total, attempt, rng = drawn.neighbours, drawn.total, drawn.attempt, drawn.rng
    sigma = core.sigma(total)

    def turned(province, k):
        for _ in range(k % n):
            province = int(sigma[province])
        return province

    land = kinds == LAND
    sea = kinds == SEA
    colors = np.array(
        [[(i * 2654435761 >> shift) & 255 for shift in (16, 8, 0)] for i in range(total + 1)],
        dtype=np.uint8,
    )
    (root / "map").mkdir()
    Image.fromarray(colors[ids]).save(root / "map/provinces.bmp")

    def nation_of(province):
        """0 to n - 1 for a land province, -1 for anything else."""
        if province > core.orbits * n or not land[province - 1]:
            return -1
        return (province - 1) % n

    nation = np.array([nation_of(i) for i in range(1, total + 1)])
    home = [i for i in range(1, total + 1) if nation[i - 1] == 0]
    coastal = {
        i: bool(
            (land[i - 1] and any(sea[j - 1] for j in sides))
            or (sea[i - 1] and any(land[j - 1] for j in sides))
        )
        for i, sides in neighbours.items()
    }
    front = [i for i in home if any(nation[j - 1] not in (-1, 0) for j in neighbours[i])]

    # States: nation 0's land cut into balanced, compact pieces, then made connected.
    frame = {core.province(o, 0): core.frame[o] for o in range(core.orbits)}
    positions = np.zeros((total + 1, 2))
    for province, at in frame.items():
        positions[province] = at
    pieces = _bisect(home, positions, design.states)
    state_of = {p: s for s, piece in enumerate(pieces) for p in piece}
    for _ in range(4):
        moved = False
        for s in range(design.states):
            members = {p for p, t in state_of.items() if t == s}
            parts = []
            left = set(members)
            while left:
                start = left.pop()
                seen, todo = {start}, [start]
                while todo:
                    for q in neighbours[todo.pop()]:
                        if q in left:
                            left.discard(q)
                            seen.add(q)
                            todo.append(q)
                parts.append(seen)
            for part in sorted(parts, key=len)[:-1]:
                votes = [state_of[q] for p in part for q in neighbours[p]
                         if q in state_of and state_of[q] != s]  # fmt: skip
                if votes:
                    target = max(set(votes), key=votes.count)
                    for p in part:
                        state_of[p] = target
                    moved = True
        if not moved:
            break
    listed = [sorted(p for p, t in state_of.items() if t == s) for s in range(design.states)]
    if any(not piece for piece in listed):
        raise RuntimeError("a state came out empty")
    if max(len(piece) for piece in listed) > 21:
        # The state channel logs a state's provinces as bits of one variable, which holds
        # 21 of them.
        raise RuntimeError("a state holds more than 21 provinces")

    anchor = _anchors(ids, total, turn, core)
    step = core.step

    # Terrain: the design's for each of nation 0's land cells (and its turns), then the
    # cities: the capital nearest the middle of the nation, three more spread out.
    wobble = rng.uniform(-0.5, 0.5, core.orbits)
    terrain_of = {}
    for o in range(core.orbits):
        if core.kind[o] != LAND:
            continue
        spots = [core.to_frame(turn.rotate(core.canonical[o], k)) for k in range(n)]
        chosen = design.base
        for patch in design.patches:
            if any(arenas._covers(patch, tuple(spot), wobble[o]) for spot in spots):
                chosen = patch.terrain
        terrain_of[core.province(o, 0)] = chosen
    # How many provinces each of nation 0's lies from its coast or its front: the cities
    # stand two or more in, as the two-country presets' do.
    depth = {p: 0 for p in home if coastal[p] or p in front}
    todo = list(depth)
    while todo:
        p = todo.pop(0)
        for q in neighbours[p]:
            if nation[q - 1] == 0 and q not in depth:
                depth[q] = depth[p] + 1
                todo.append(q)
    middle = positions[home].mean(axis=0)
    inland = [p for p in home if depth.get(p, 0) >= 2] or [p for p in home if depth.get(p, 0)]
    capital = min(inland, key=lambda p: np.linalg.norm(positions[p] - middle))
    cities = [capital]
    for _ in range(3):
        taken = {state_of[c] for c in cities}
        spots = [p for p in inland if state_of[p] not in taken] or [
            p for p in home if depth.get(p, 0) and state_of[p] not in taken
        ]
        cities.append(max(spots, key=lambda p: min(np.linalg.norm(positions[p] - positions[c])
                                                   for c in cities)))  # fmt: skip
    for city in cities:
        terrain_of[city] = "urban"
    terrain_types = ["ocean"] * total
    for province, name in terrain_of.items():
        for k in range(n):
            terrain_types[turned(province, k) - 1] = name
    for i in range(total):
        if kinds[i] == LAKE:
            terrain_types[i] = "lakes"
    all_cities = [turned(c, k) for k in range(n) for c in cities]

    def indexed(name, pixels):
        original = Image.open(game / "map" / name)
        image = Image.fromarray(pixels.astype(np.uint8)).convert("P")
        palette = original.getpalette()
        if palette is not None:
            image.putpalette(palette)
        image.save(root / "map" / name)

    trees = (
        WIDTH * TREES_NUMERATOR // TREES_DENOMINATOR,
        HEIGHT * TREES_NUMERATOR // TREES_DENOMINATOR,
    )
    lookup = np.zeros(total + 1, np.int8)
    lookup[1:] = kinds
    kind_px = lookup[ids]
    lookup = np.full(total + 1, -1, np.int8)
    for i, name in enumerate(terrain_types, 1):
        if kinds[i - 1] == LAND:
            lookup[i] = arenas.LAND_TYPES.index(name)
    types_px = lookup[ids]
    heights = arenas.relief(rng, types_px, kind_px, None, turn=turn)
    urban_px, style_px, lights_px = arenas.city_layers(rng, ids, all_cities, anchor, turn=turn)
    graphical = arenas.terrain_pixels(rng, types_px, heights, urban_px, turn=turn)
    ground = kind_px == LAND
    indexed("terrain.bmp", graphical)
    indexed("rivers.bmp", np.where(ground, arenas.RIVER_LAND, arenas.RIVER_WATER))
    Image.fromarray(heights).save(root / "map/heightmap.bmp")
    Image.fromarray(arenas.normals(heights, turn=turn)).save(root / "map/world_normal.bmp")
    indexed("trees.bmp", arenas.tree_pixels(rng, types_px, trees, turn=turn))
    indexed("cities.bmp", np.where(style_px >= 0, style_px, CITY_INDEX))
    colour = arenas.colour_map(rng, graphical, lights_px, np.array(OCEAN_COLOUR), turn=turn)
    land_half = ground[::2, ::2]
    del kind_px, types_px, urban_px, style_px, lights_px
    write_dds(root / "map/terrain/colormap_rgb_cityemissivemask_a.dds", colour)
    fog = np.where(land_half[..., None], np.array(FOG_LAND), np.array(FOG_SEA)).astype(np.uint8)
    write_dds(root / "map/terrain/fow_rgb_waterspec_a.dds", fog)
    for level in range(3):
        shrink = 2 ** (level + 1)
        water = np.empty((HEIGHT // shrink, WIDTH // shrink, 4), dtype=np.uint8)
        water[...] = WATER_COLOUR
        write_dds(root / f"map/terrain/colormap_water_{level}.dds", water)
    for name, (widget_columns, widget_rows) in MINIMAP_SIZES.items():
        shrunk = np.array(
            Image.fromarray(ground.astype(np.uint8) * 255).resize(
                (widget_columns, widget_rows), Image.BILINEAR
            )
        )
        picture = np.where(shrunk[..., None] > 127, np.array(MINIMAP_LAND), np.array(MINIMAP_SEA))
        write_dds(root / name, picture.astype(np.uint8))

    def write(name, text):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            text, encoding="utf-8-sig" if name.endswith(".yml") else "utf8", newline="\r\n"
        )

    definitions = ["0;0;0;0;land;false;unknown;0"]
    for i in range(1, total + 1):
        r, g, b = colors[i]
        category, terrain = {
            LAND: ("land", terrain_types[i - 1]),
            SEA: ("sea", "ocean"),
            LAKE: ("lake", "lakes"),
        }[kinds[i - 1]]
        definitions.append(
            f"{i};{r};{g};{b};{category};{str(coastal[i]).lower()};{terrain};{1 if land[i - 1] else 0}"
        )
    write("map/definition.csv", "\n".join(definitions) + "\n")

    def ground_y(province, x, y):
        if kinds[province - 1] != LAND:
            return "9.50"
        return f"{heights[int(y), int(x)] / 10:.2f}"

    write(
        "map/default.map",
        "\n".join(
            f'{key} = "{value}"'
            for key, value in {
                "definitions": "definition.csv",
                "provinces": "provinces.bmp",
                "positions": "positions.txt",
                "terrain": "terrain.bmp",
                "rivers": "rivers.bmp",
                "heightmap": "heightmap.bmp",
                "tree_definition": "trees.bmp",
                "continent": "continent.txt",
                "adjacency_rules": "adjacency_rules.txt",
                "adjacencies": "adjacencies.csv",
                "ambient_object": "ambient_object.txt",
                "seasons": "seasons.txt",
            }.items()
        )
        + "\ntree = { 3 4 7 10 }\n",
    )
    write(
        "map/continent.txt",
        "continents = { europe north_america south_america australia africa asia middle_east }\n",
    )
    for name in ["positions.txt", "adjacency_rules.txt", "ambient_object.txt"]:
        write(f"map/{name}", "# Generated arena: no custom entries.\n")
    write(
        "tutorial/tutorial.txt",
        'tutorial = {\n\twindow = "tutorial_screen_1"\n\tuse_mil_fac = { textbox = "obj_1" }\n}\n',
    )
    write(
        "map/adjacencies.csv",
        "From;To;Type;Through;start_x;start_y;stop_x;stop_y;adjacency_rule_name;Comment\n"
        "-1;-1;;-1;-1;-1;-1;-1;-1\n",
    )
    stacks = []
    for i, (x, y) in enumerate(anchor, 1):
        if kinds[i - 1] == LAKE:
            continue
        slots = LAND_STACKS if land[i - 1] else SEA_STACKS
        if land[i - 1] and coastal[i]:
            slots += PORT_STACKS
        for slot in slots:
            stacks.append(f"{i};{slot};{x}.00;{ground_y(i, x, y)};{HEIGHT - y}.00;0.00;0.30")
    write("map/unitstacks.txt", "\n".join(stacks) + "\n")
    weather = " ".join(
        f"period = {{ between = {{ 0.{month} {last}.{month} }} "
        f"temperature = {{ 15.0 20.0 }} no_phenomenon = 1.0 }}"
        for month, last in enumerate(MONTH_LAST_DAY)
    )
    regions = [(1, kinds != SEA), (2, kinds == SEA)]
    for region, mask in regions:
        members = (np.flatnonzero(mask) + 1).tolist()
        naval = "" if region == 1 else "naval_terrain = water_shallow_sea "
        write(
            f"map/strategicregions/{region}-arena.txt",
            f'strategic_region = {{ id = {region} name = "ARENA_REGION_{region}" '
            f"provinces = {{ {' '.join(map(str, members))} }} {naval}"
            f"weather = {{ {weather} }} }}",
        )
    placed = []
    for region, mask in regions:
        members = (np.flatnonzero(mask) + 1).tolist()
        chosen = [members[len(members) // 4], members[3 * len(members) // 4]]
        for size in ["small", "big"]:
            for province in chosen:
                x, y = anchor[province - 1]
                placed.append(f"{region};{x}.00;{ground_y(province, x, y)};{HEIGHT - y}.00;{size}")
    write("map/weatherpositions.txt", "\n".join(placed) + "\n")

    # States, owners, victory points, and each nation's files.
    states_per_country = design.states
    states, state_owner = {}, {}
    for k, tag in enumerate(tags):
        for s, piece in enumerate(listed):
            state = k * states_per_country + s + 1
            states[state] = sorted(turned(p, k) for p in piece)
            state_owner[state] = tag
    capital_state = state_of[capital] + 1
    deployed = sorted(front, key=lambda p: (positions[p][1] > 0, np.hypot(*positions[p])))[::2]
    victory_points = {}
    for k, tag in enumerate(tags):
        victory_points[tag] = {turned(capital, k): 20, **{turned(c, k): 5 for c in cities[1:]}}
        write(
            f"common/countries/{tag}.txt",
            f"graphical_culture = western_european_gfx\ngraphical_culture_2d = western_european_2d\n"
            f"color = rgb {{ {' '.join(map(str, multination.COLOUR[tag]))} }}",
        )
        bonus = "add_ideas = arena_underdog\n" if lone_bonus and tag in plan.underdogs() else ""
        write(
            f"history/countries/{tag} - Arena.txt",
            f'capital = {k * states_per_country + capital_state}\noob = "{tag}_1936"\n'
            f"recruit_character = {tag}_commander\nrecruit_character = {tag}_marshal\n"
            + "".join(
                f"recruit_character = {tag}_general_{g}\n"
                for g in range(1, GENERALS_PER_COUNTRY + 1)
            )
            + "set_politics = { ruling_party = neutrality elections_allowed = no }\n"
            "set_popularities = { neutrality = 100 }\nset_stability = 1\nset_war_support = 1\n"
            f"add_ideas = arena_march_speed\n{bonus}"
            "set_technology = { infantry_weapons = 1 infantry_weapons1 = 1 basic_train = 1 }\n"
            f"add_equipment_to_stockpile = {{ type = infantry_equipment_1 amount = 50000 producer = {tag} }}\n"
            f"add_equipment_to_stockpile = {{ type = train_equipment_1 amount = 50 producer = {tag} }}\n"
            + multination.faction_history(plan, tag),
        )
        regiments = " ".join(
            f"infantry = {{ x = {x} y = {y} }}" for x in range(2) for y in range(3)
        )
        divisions = "\n".join(
            f'division = {{ name = "Infantry {d}" location = {turned(p, k)} division_template = "Arena Infantry" start_experience_factor = 0.3 start_equipment_factor = 1 }}'
            for d, p in enumerate(deployed, 1)
        )
        write(
            f"history/units/{tag}_1936.txt",
            f'division_template = {{ name = "Arena Infantry" regiments = {{ {regiments} }} }}\nunits = {{ {divisions} }}',
        )
        for sub, size in [("", (82, 52)), ("medium/", (41, 26)), ("small/", (10, 7))]:
            flag = root / f"gfx/flags/{sub}{tag}.tga"
            flag.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", size, (*multination.COLOUR[tag], 255)).save(flag)
    for state, members in sorted(states.items()):
        tag = state_owner[state]
        points_block = " ".join(
            f"victory_points = {{ {province} {value} }}"
            for province, value in victory_points[tag].items()
            if province in set(members)
        )
        write(
            f"history/states/{state}-arena.txt",
            f'state = {{ id = {state} name = "ARENA_STATE_{state}" manpower = {1000000 // states_per_country} state_category = rural history = {{ owner = {tag} add_core_of = {tag} {points_block} buildings = {{ infrastructure = 4 }} }} provinces = {{ {" ".join(map(str, members))} }} }}',
        )
    ideas = (
        "ideas = {\n\tcountry = {\n\t\tarena_march_speed = {\n"
        "\t\t\tallowed = { always = no }\n\t\t\tremoval_cost = -1\n"
        f"\t\t\tmodifier = {{ army_speed_factor = {ARMY_SPEED_FACTOR} }}\n"
        "\t\t}\n" + (multination.underdog_idea(lone_bonus) if lone_bonus else "") + "\t}\n}\n"
    )
    write("common/ideas/arena.txt", ideas)
    from .mapgen import _NAME_LISTS  # The two-country arena's, for Blue and Red.

    name_lists = {**_NAME_LISTS, **multination.NAME_LISTS}
    write(
        "common/names/01_arena_names.txt",
        "\n".join(
            f"{tag} = {{\n\tmale = {{ names = {{ {name_lists[tag][0]} }} }}\n"
            f"\tfemale = {{ names = {{ {name_lists[tag][1]} }} }}\n"
            f"\tsurnames = {{ {name_lists[tag][2]} }}\n\tcallsigns = {{ }}\n}}"
            for tag in tags
        )
        + "\n",
    )

    def commander(key, role):
        return (
            f'\t{key} = {{\n\t\tname = "{key}"\n'
            f'\t\tportraits = {{ army = {{ small = "{COMMANDER_PORTRAIT}_small" }} '
            f'army = {{ large = "{COMMANDER_PORTRAIT}" }} }}\n'
            f"\t\t{role} = {{ traits = {{ }} skill = 3 attack_skill = 3 defense_skill = 3 "
            f"planning_skill = 3 logistics_skill = 3 }}\n\t}}\n"
        )

    write(
        "common/characters/arena.txt",
        "characters = {\n"
        + "".join(
            f'\t{tag}_commander = {{\n\t\tname = "{multination.NAMES[tag]} Command"\n'
            f'\t\tcountry_leader = {{ ideology = despotism expire = "1965.1.1.1" id = -1 }}\n\t}}\n'
            + commander(f"{tag}_marshal", "field_marshal")
            + "".join(
                commander(f"{tag}_general_{g}", "corps_commander")
                for g in range(1, GENERALS_PER_COUNTRY + 1)
            )
            for tag in tags
        )
        + "}\n",
    )
    write("common/ai_strategy/arena.txt", multination.strategies(plan))
    write("common/factions/templates/arena.txt", multination.faction_template())

    # Hubs: each state's city, or the easiest ground near its middle; nation 0's turned.
    centres = {}
    for s, piece in enumerate(listed):
        towns = [p for p in piece if terrain_types[p - 1] == "urban"]
        mid = anchor[np.array(piece) - 1].mean(axis=0)
        centres[s] = (
            towns[0]
            if towns
            else min(
                piece,
                key=lambda p: (
                    np.linalg.norm(anchor[p - 1] - mid) / step
                    + 3 * (arenas.RAIL_COST[terrain_types[p - 1]] - 1)
                ),
            )
        )
    hubs = sorted({turned(h, k) for h in centres.values() for k in range(n)})
    write("map/supply_nodes.txt", "\n".join(f"1 {hub}" for hub in hubs) + "\n")
    # Railways: a trunk through nation 0 joining its capital, cities and hubs, and two
    # lines across its border with nation 1, a third and two thirds of the way out; all
    # turned, so every border between neighbours is crossed twice.
    pairs = [(a, b) for a in home for b in sorted(neighbours[a]) if nation[b - 1] == 1]
    reach = {pair: np.hypot(*(positions[pair[0]])) for pair in pairs}
    far = max(reach.values())
    crossings = []
    for share in (0.35, 0.7):
        crossings.append(
            min(
                (p for p in pairs if p not in crossings),
                key=lambda ab: (
                    abs(reach[ab] - share * far)
                    + 2 * (arenas.RAIL_COST[terrain_types[ab[0] - 1]] - 1)
                    + 2 * (arenas.RAIL_COST[terrain_types[ab[1] - 1]] - 1)
                ),
            )
        )
    links = arenas.trunk_rails(
        set(home),
        neighbours,
        {i: arenas.RAIL_COST[terrain_types[i - 1]] for i in home},
        list(centres.values()) + cities,
        crossings,
        lambda p: turned(p, n - 1),
    )
    links = [(a, b) for a, b in links if nation[a - 1] == 0 and nation[b - 1] == 0] + crossings
    rails = sorted(
        {tuple(sorted((turned(a, k), turned(b, k)))) for a, b in links for k in range(n)}
    )
    write("map/railways.txt", "\n".join(f"1 2 {a} {b}" for a, b in rails) + "\n")
    # Ports face the sea province nation 0's does, turned.
    ports = {}
    for p in home:
        if coastal[p]:
            port = min(j for j in neighbours[p] if sea[j - 1])
            for k in range(n):
                ports[turned(p, k)] = turned(port, k)
    state_centre = {}
    for k in range(n):
        for s in range(states_per_country):
            state_centre[k * states_per_country + s + 1] = turned(centres[s], k)
    buildings = []
    for state, members in sorted(states.items()):
        cx, cy = anchor[state_centre[state] - 1]
        cz = ground_y(state_centre[state], cx, cy)
        for building, slots in STATE_BUILDINGS.items():
            for slot in range(slots):
                buildings.append(
                    f"{state};{building};{cx + slot * 2}.00;{cz};{HEIGHT - cy + slot * 2}.00;0.00;0"
                )
        if any(coastal[i] for i in members):
            buildings.append(f"{state};dockyard;{cx}.00;{cz};{HEIGHT - cy}.00;0.00;0")
        for i in members:
            x, y = anchor[i - 1]
            z = ground_y(i, x, y)
            for building in PROVINCE_BUILDINGS:
                buildings.append(f"{state};{building};{x}.00;{z};{HEIGHT - y}.00;0.00;0")
            for building in COASTAL_BUILDINGS if coastal[i] else ():
                port = ports[i] if building in PORT_BUILDINGS else 0
                buildings.append(f"{state};{building};{x}.00;{z};{HEIGHT - y}.00;0.00;{port}")
    write("map/buildings.txt", "\n".join(buildings) + "\n")
    write(
        "common/national_focus/arena.txt",
        "focus_tree = { id = arena_focus default = yes country = { factor = 1 } "
        "focus = { id = arena_training icon = GFX_goal_generic_army_doctrines "
        "x = 0 y = 0 cost = 1000 completion_reward = { } } }",
    )
    write(
        "common/country_tags/00_arena.txt",
        "".join(f'{tag} = "countries/{tag}.txt"\n' for tag in tags),
    )
    write(
        "common/countries/colors.txt",
        "#reload countrycolors\n"
        + "".join(
            f"{tag} = {{\n"
            f"\tcolor = rgb {{ {' '.join(map(str, multination.COLOUR[tag]))} }}\n"
            f"\tcolor_ui = rgb {{ {' '.join(map(str, multination.COLOUR_UI[tag]))} }}\n"
            f"}}\n"
            for tag in tags
        ),
    )
    write(
        "common/bookmarks/arena.txt",
        'bookmarks = { bookmark = { name = "ARENA_BOOKMARK" desc = "ARENA_DESC" '
        f'date = 1936.1.1.12 picture = "GFX_select_date_1936" default_country = "{tags[0]}" '
        "default = yes effect = { randomize_weather = 22 } "
        + " ".join(
            f'{tag} = {{ history = "ARENA_{tag}_HISTORY" ideology = neutrality }}' for tag in tags
        )  # fmt: skip
        + " } }",
    )
    write("events/arena.txt", multination.events(plan))
    state_ids = sorted(states)
    daily = (
        "".join(
            f" set_temp_variable = {{ arena_d{s} = num_armies_in_state@{s} }}" for s in state_ids
        )
        + " set_temp_variable = { arena_rifles = num_equipment_in_armies_k@infantry_equipment }"
        " set_temp_variable = { arena_needed = num_target_equipment_in_armies_k@infantry_equipment }"
        ' log = "ARENA day [GetDateText] [ROOT.GetTag] states [?num_controlled_states] owned'
        " [?num_owned_controlled_states] divisions [?num_divisions] surrender"
        " [?surrender_progress] strength [?enemies_strength_ratio] casualties [?casualties_k]"
        " manpower [?manpower_k] deployed [?deployed_army_manpower_k] rifles [?arena_rifles]"
        " needed [?arena_needed] at" + "".join(f" {s}=[?arena_d{s}]" for s in state_ids) + '"'
    )
    daily += daily_effect(states)
    write(
        "common/on_actions/arena.txt",
        "on_actions = {\n"
        "\ton_startup = { effect = {"
        ' log = "ARENA start [GetDateText]"'
        ' every_country = { limit = { is_ai = no } log = "ARENA player [THIS.GetTag]" } '
        + multination.startup_log(plan).strip()
        + " "
        + startup_effect(states).strip()
        + " } }\n"
        '\ton_weekly = { effect = { log = "ARENA week [GetDateText] [ROOT.GetTag] states'
        " [?num_controlled_states] owned [?num_owned_controlled_states] divisions"
        ' [?num_divisions] surrender [?surrender_progress]" } }\n'
        + "".join(
            f"\ton_daily_{tag} = {{ effect = {{{multination.fallback(plan) if k == 0 else ''}"
            f"{daily} }} }}\n"
            for k, tag in enumerate(tags)
        )
        + '\ton_capitulation = { effect = { log = "ARENA capitulated [ROOT.GetTag] winner'
        ' [FROM.GetTag] [GetDateText]" } }\n'
        '\ton_state_control_changed = { effect = { log = "ARENA control [ROOT.GetTag] from'
        ' [FROM.GetTag] [FROM.FROM.GetName] [GetDateText]" } }\n'
        '\ton_peaceconference_ended = { effect = { log = "ARENA peace [ROOT.GetTag]'
        ' [FROM.GetTag] [GetDateText]" } }\n'
        "}",
    )
    names = multination.NAMES
    title = f"Arena: {design.title} ({plan.name})"
    localisation = [
        "l_english:",
        f' ARENA_BOOKMARK:0 "{title}"',
        f' ARENA_DESC:0 "{design.summary} Wars: {plan.name}, {plan.note()}."',
        ' ARENA_REGION_1:0 "Arena"',
        ' ARENA_REGION_2:0 "Ocean"',
        ' ARENA_FACTION:0 "Arena Pact"',
        ' arena_focus:0 "Arena"',
        ' arena_training:0 "Army Training"',
        ' arena_training_desc:0 ""',
        ' arena_march_speed:0 "Arena March Rate"',
        ' arena_march_speed_desc:0 "Provinces here are far larger than a stock one, so'
        ' armies march proportionally faster."',
        ' arena_underdog:0 "Stand Alone"',
        ' arena_underdog_desc:0 "Outnumbered by design, and given an edge to make up for it."',
    ]
    for number, side in enumerate(plan.sides, 1):
        pact = "-".join(names[t] for t in side)
        localisation.append(f' ARENA_FACTION_{number}:0 "{pact} Pact"')
    for tag in tags:
        localisation.append(f' ARENA_{tag}_HISTORY:0 "{names[tag]} holds one of the arena\'s '
                            f'{n} equal shares."')  # fmt: skip
    for state in sorted(states):
        localisation.append(
            f' ARENA_STATE_{state}:0 "{names[state_owner[state]]} {(state - 1) % states_per_country + 1}"'
        )
    for tag in tags:
        name = names[tag]
        localisation += [
            f' {tag}:0 "{name}"',
            f' {tag}_DEF:0 "{name}"',
            f' {tag}_ADJ:0 "{name}"',
            f' {tag}_neutrality:0 "{name}"',
            f' {tag}_neutrality_DEF:0 "{name}"',
            f' {tag}_neutrality_ADJ:0 "{name}"',
            f' {tag}_commander:0 "{name} Command"',
            f' {tag}_marshal:0 "{name} Marshal"',
            *(
                f' {tag}_general_{g}:0 "{name} General {g}"'
                for g in range(1, GENERALS_PER_COUNTRY + 1)
            ),
        ]
    write("localisation/english/arena_l_english.yml", "\n".join(localisation) + "\n")
    labels = ["l_english:"]
    for tag, points in victory_points.items():
        for order, province in enumerate(points):
            label = f"{names[tag]} Capital" if order == 0 else f"{names[tag]} City {order}"
            labels.append(f' VICTORY_POINTS_{province}:0 "{label}"')
    write(VICTORY_POINT_NAMES, "\n".join(labels) + "\n")
    replacements = [
        "tutorial",
        "history/countries",
        "history/states",
        "history/units",
        "common/bookmarks",
        "common/national_focus",
        "common/ai_focuses",
        "common/ai_strategy_plans",
        "common/decisions",
        "common/decisions/categories",
        "common/strategic_locations",
        "events",
        "common/on_actions",
        "map/strategicregions",
        "map/supplyareas",
    ]
    descriptor = (
        'name = "HOI4 Visual Infantry Arena"\nversion = "0.1"\nsupported_version = "1.19.*"\n'
        + "".join(f'replace_path = "{p}"\n' for p in replacements)
    )
    for directory in replacements:
        (root / directory).mkdir(parents=True, exist_ok=True)
    write("descriptor.mod", descriptor)
    root.with_suffix(".mod").write_text(
        descriptor + f'path = "{root.as_posix()}"\n', newline="\r\n"
    )

    lookup = np.zeros(total + 1, np.int16)
    for state, members in states.items():
        lookup[members] = state
    ys, xs = np.nonzero(ground)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    across = (box[0] + (np.arange(LAYOUT[0]) + 0.5) * (box[2] - box[0]) / LAYOUT[0]).astype(int)
    down = (box[1] + (np.arange(LAYOUT[1]) + 0.5) * (box[3] - box[1]) / LAYOUT[1]).astype(int)
    grid = lookup[ids[down][:, across]]
    fronts = {}
    for k in range(1, n):
        if any(nation[j - 1] == k for p in home for j in neighbours[p]):
            pair_owner = np.where(np.isin(nation, [0, k]), np.where(nation == 0, 1, 2), 0)
            fronts[f"{tags[0]}-{tags[k]}"] = arenas.front_report(neighbours, pair_owner,
                                                                  terrain_types, {})  # fmt: skip
    report = {
        "width": WIDTH,
        "height": HEIGHT,
        "countries": n,
        "nations": n,
        "tags": list(tags),
        "provinces": total,
        "land_provinces_per_country": len(home),
        "states_per_country": states_per_country,
        "divisions_per_country": len(deployed),
        "army_speed_factor": ARMY_SPEED_FACTOR,
        "coastal_land_provinces": sum(1 for i in neighbours if land[i - 1] and coastal[i]),
        "victory_points_per_country": len(victory_points[tags[0]]),
        "rotational_mirror": True,
        "symmetry": {
            "order": n,
            "orbits": core.orbits,
            "fixed": list(range(core.orbits * n + 1, core.orbits * n + core.fixed + 1)),
            "pixel_exact": turn.exact,
            "centre": list(CENTRE),
            "square": [SQUARE_X0, 0, SQUARE] if turn.exact else None,
            "core_radius": None if turn.exact else CORE_RADIUS,
            "attempts": attempt + 1,
        },
        "preset": preset,
        "wars": {
            "setup": plan.name,
            "sides": [list(side) for side in plan.sides],
            "neutral": list(plan.neutral),
            "fair": plan.fair,
            "note": plan.note(),
            "lone_bonus": lone_bonus,
            "underdogs": list(plan.underdogs()) if lone_bonus else [],
        },
        "layout": {"box": box, "states": [" ".join(map(str, row)) for row in grid.tolist()]},
        "design": {
            "title": design.title,
            "seed": seed,
            "warp": round(design.warp * 0.7**attempt, 2),
            "front": fronts,
            "terrain": {
                name: sum(1 for i in home if terrain_types[i - 1] == name)
                for name in arenas.LAND_TYPES
            },  # fmt: skip
            "lakes": int((kinds == LAKE).sum()),
            "cities": cities,
            "capital": capital,
        },
        "gameplay_verified": False,
        "engine_load_verified": False,
    }
    write("generation.json", json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------------------
# The audit's multi-nation checks (mapgen.audit calls this when a map has 3 or more).


def audit_nations(root, report, rows, ids, graphical, heights):
    """What only a multi-nation arena can get wrong, read from the written files:

    - the turn: every province's turn has the same kind, terrain and coast, the province
      graph maps onto itself, and (four nations) the pixels of the square are exact turns;
    - each nation holds the same: land provinces, terrain, states, victory points, hubs,
      divisions and railways, turned;
    - the arena lessons for N sides: railways cross every border between neighbours at
      least twice, each capital stands mid-country, no lake or bay splits a front, no
      state lies in two strategic regions, every state is in one piece;
    - the wars: each nation has an AI strategy against every other, and the events declare
      exactly the wars generation.json names.
    """
    import re

    problems = []
    n = report["nations"]
    tags = report["tags"]
    symmetry = report["symmetry"]
    count = len(rows) - 1
    orbits = symmetry["orbits"]
    kinds = ["land"] + [r[4] for r in rows[1:]]
    terrains = ["unknown"] + [r[6] for r in rows[1:]]
    coasts = [False] + [r[5] == "true" for r in rows[1:]]

    def sigma(i, k=1):
        for _ in range(k % n):
            if i <= orbits * n:
                i = (i - 1) // n * n + ((i - 1) % n + 1) % n + 1
        return i

    core = range(1, orbits * n + 1)
    unlike = [
        i
        for i in core
        if (kinds[i], terrains[i], coasts[i])
        != (kinds[sigma(i)], terrains[sigma(i)], coasts[sigma(i)])
    ]
    if unlike:
        problems.append(f"{len(unlike)} provinces differ from their turn, first {unlike[0]}")
    neighbours = adjacency(ids, count)
    for i in core:
        if kinds[i] == "sea":
            continue
        for j in neighbours[i]:
            if j > orbits * n + len(symmetry["fixed"]):
                problems.append(f"land province {i} touches the open sea outside the core")
                break
            if sigma(j) not in neighbours[sigma(i)]:
                problems.append(f"provinces {i} and {j} touch, but their turns do not")
                break
    if symmetry["pixel_exact"]:
        x0, _, side = symmetry["square"]
        sub = ids[:, x0 : x0 + side]
        lookup = np.array([sigma(i) for i in range(count + 1)])
        broke = int((lookup[np.rot90(sub)] != sub).sum())
        if broke:
            problems.append(f"{broke} province pixels break the quarter turn")
        for name, picture in (("terrain.bmp", graphical), ("heightmap.bmp", heights)):
            square = picture[:, x0 : x0 + side]
            if (np.rot90(square) != square).any():
                problems.append(f"{name} is not the same turned a quarter")
        for name in ("trees.bmp", "cities.bmp"):
            picture = np.array(Image.open(root / "map" / name))
            px0 = (picture.shape[1] - picture.shape[0]) // 2
            square = picture[:, px0 : px0 + picture.shape[0]]
            if (np.rot90(square) != square).any():
                problems.append(f"{name} is not the same turned a quarter")
        # The copy leaves no seam along the lines its quarters meet on.
        middle = side // 2
        across = np.abs(np.diff(heights[:, x0 + middle - 1 : x0 + middle + 1], axis=1)).mean()
        beside = np.abs(np.diff(heights[:, x0 + middle - 4 : x0 + middle + 4], axis=1)).mean()
        if across > 2 * beside + 0.5:
            problems.append(f"the heightmap has a seam where its quarters meet ({across:.1f})")

    # Who holds what.
    states, holder, state_owner, points = {}, {}, {}, {}
    for path in sorted((root / "history/states").glob("*.txt")):
        text = path.read_text()
        state = int(re.search(r"id\s*=\s*(\d+)", text).group(1))
        tag = re.search(r"owner\s*=\s*(\w+)", text).group(1)
        members = [int(p) for p in re.search(r"provinces\s*=\s*\{([^}]*)\}", text).group(1).split()]
        states[state], state_owner[state] = members, tag
        holder.update(dict.fromkeys(members, tag))
        for province, value in re.findall(r"victory_points\s*=\s*\{\s*(\d+)\s+(\d+)", text):
            points.setdefault(tag, {})[int(province)] = int(value)
    for k, tag in enumerate(tags):
        mine = sorted(p for p, t in holder.items() if t == tag)
        turned = sorted(sigma(p, k) for p, t in holder.items() if t == tags[0])
        if mine != turned:
            problems.append(f"{tag}'s land is not {tags[0]}'s turned {k} times")
        if sorted(sigma(p, k) for p in points.get(tags[0], {})) != sorted(points.get(tag, {})):
            problems.append(f"{tag}'s victory points are not {tags[0]}'s turned")
        if sorted(points.get(tag, {}).values()) != sorted(points.get(tags[0], {}).values()):
            problems.append(f"{tag}'s victory points are not worth {tags[0]}'s")
    for state, members in states.items():
        seen, todo = {members[0]}, [members[0]]
        while todo:
            for q in neighbours[todo.pop()]:
                if q in members and q not in seen:
                    seen.add(q)
                    todo.append(q)
        if len(seen) != len(members):
            problems.append(f"state {state} is in more than one piece")
    regions = {}
    for path in sorted((root / "map/strategicregions").glob("*.txt")):
        text = path.read_text()
        region = int(re.search(r"id\s*=\s*(\d+)", text).group(1))
        for p in re.search(r"provinces\s*=\s*\{([^}]*)\}", text).group(1).split():
            regions[int(p)] = region
    for state, members in states.items():
        if len({regions.get(p) for p in members}) > 1:
            problems.append(f"state {state} lies in two strategic regions")
    units = {}
    for path in sorted((root / "history/units").glob("*.txt")):
        tag = path.name.split("_")[0]
        units[tag] = sorted(int(p) for p in re.findall(r"location\s*=\s*(\d+)", path.read_text()))
    hubs = [int(c) for c in (root / "map/supply_nodes.txt").read_text().split()[1::2]]
    links = set()
    for line in (root / "map/railways.txt").read_text().splitlines():
        cells = [int(c) for c in line.split()]
        links |= {tuple(sorted(pair)) for pair in zip(cells[2:-1], cells[3:])}
    for k, tag in enumerate(tags[1:], 1):
        if sorted(sigma(p, k) for p in units.get(tags[0], [])) != units.get(tag):
            problems.append(f"{tag}'s divisions do not stand where {tags[0]}'s do, turned")
    if sorted(sigma(h) for h in hubs) != sorted(hubs):
        problems.append("the supply hubs are not the same turned")
    if {tuple(sorted((sigma(a), sigma(b)))) for a, b in links} != links:
        problems.append("the railways are not the same turned")
    # Lines across every border between neighbours.
    for a_tag in tags:
        for b_tag in tags:
            if a_tag >= b_tag:
                continue
            touching = any(
                holder.get(q) == b_tag
                for p, t in holder.items()
                if t == a_tag
                for q in neighbours[p]
            )
            if not touching:
                continue
            across = sum(1 for a, b in links if {holder.get(a), holder.get(b)} == {a_tag, b_tag})
            if across < 2:
                problems.append(f"{across} railways cross between {a_tag} and {b_tag}")
            # One front, not two: the border's pixels are one stretch.
            owner_px = np.zeros(count + 1, np.int8)
            for p, t in holder.items():
                owner_px[p] = 1 if t == a_tag else 2 if t == b_tag else 0
            px = owner_px[ids]
            line = np.zeros(px.shape, bool)
            meet = ((px[:, :-1] == 1) & (px[:, 1:] == 2)) | ((px[:, :-1] == 2) & (px[:, 1:] == 1))
            line[:, :-1] |= meet
            meet = ((px[:-1] == 1) & (px[1:] == 2)) | ((px[:-1] == 2) & (px[1:] == 1))
            line[:-1] |= meet
            _, stretches = ndimage.label(ndimage.binary_dilation(line, iterations=3))
            if stretches > 1:
                problems.append(f"the front between {a_tag} and {b_tag} is split in {stretches}")
    # Capitals mid-country: within a third of the nation's reach of its middle.
    centres = np.array(
        [(0.0, 0.0)]
        + ndimage.center_of_mass(np.ones(ids.shape, np.int8), ids, np.arange(1, count + 1))
    )
    capitals = {}
    for tag in tags:
        text = (root / f"history/countries/{tag} - Arena.txt").read_text()
        state = int(re.search(r"capital\s*=\s*(\d+)", text).group(1))
        capital = max(
            (p for p in states[state] if p in points.get(tag, {})),
            key=lambda p: points[tag][p],
            default=None,
        )
        capitals[tag] = capital
        if capital is None:
            problems.append(f"{tag}'s capital state holds no victory point")
            continue
        mine = [p for p, t in holder.items() if t == tag]
        where, spread = centres[capital], centres[mine]
        middle = spread.mean(axis=0)
        reach = np.hypot(*(spread - middle).T).max()
        if np.hypot(*(where - middle)) > reach / 3:
            problems.append(f"{tag}'s capital is not mid-country")
    # The wars.
    strategy = (root / "common/ai_strategy/arena.txt").read_text()
    for tag in tags:
        for other in tags:
            if other != tag and f"{tag}_arena_offensive_{other}" not in strategy:
                problems.append(f"{tag} has no front_control strategy against {other}")
    wars = report["wars"]
    declared = (root / "events/arena.txt").read_text()
    for number in range(1, n + 1):
        block = re.search(rf"id = arena\.{number} .*", declared)
        if block is None:
            problems.append(f"events/arena.txt has no arena.{number}")
            continue
        found = set(
            re.findall(r"(\w{3}) = \{ declare_war_on = \{ target = (\w{3})", block.group(0))
        )
        sides = wars["sides"]
        expected = {frozenset((a[0], b[0])) for i, a in enumerate(sides) for b in sides[i + 1 :]}
        if {frozenset(pair) for pair in found} != expected:
            problems.append(f"arena.{number} does not declare the wars generation.json names")
    return problems
