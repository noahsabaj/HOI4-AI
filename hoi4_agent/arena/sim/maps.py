"""Deterministic SIMULATOR scenarios: mirrored grid maps, deployments and dataset splits.

Every map is a west/east mirror image (BLUE west, RED east), so a symmetric pair of
players draws and side advantage can only come from play. Sizes: ``tiny`` (4x2, 8
provinces, two routes), ``small`` (6x2, 12 provinces, two routes) and ``full`` (8x6, 48
provinces, three routes that are only connected to each other behind the front).
"Contested central victory points" are the front-column objectives either side of the
border: there are no neutral provinces, because a two-country HOI4 mod has none either.

A scenario ID fully determines the scenario: ``sim-<size>-<variant>-d<divisions>a<armor>-s<seed>``.
Nothing here is derived from real HOI4 map data; the layout source is "synthetic".
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from ..contracts import ArenaError, Country
from ..layout import ArenaLayout, LayoutProvince

SIZES: dict[str, tuple[int, int]] = {"tiny": (4, 2), "small": (6, 2), "full": (8, 6)}  # columns, rows
DEFAULT_DIVISIONS = {"tiny": 2, "small": 3, "full": 12}
VARIANTS = ("open", "crossing", "breakthrough", "defence", "retreat", "encirclement", "supply")
SPLITS = ("train", "validation", "held_out")
_TERRAIN_WEIGHTS = (("plains", 8), ("forest", 3), ("hills", 3), ("marsh", 1), ("mountain", 1), ("desert", 1),
                    ("urban", 1))
_ID = re.compile(r"^sim-(tiny|small|full)-([a-z]+)-d(\d+)a(\d+)-s(\d+)$")
CAPITAL_VP, FRONT_VP = 5.0, 3.0


@dataclass(frozen=True)
class Deployment:
    country: str
    province_id: int
    kind: str = "infantry"
    organization: float = 1.0
    strength: float = 1.0


@dataclass(frozen=True)
class Scenario:
    """A layout plus the starting armies and supply reach the simulator needs."""
    layout: ArenaLayout
    deployments: tuple[Deployment, ...]
    size: str
    variant: str
    seed: int
    supply_range: int  # hops from the capital with full supply
    supply_falloff: int  # further hops over which supply decays to zero
    columns: int
    rows: int

    def mirror(self, province_id: int) -> int:
        row, column = divmod(province_id - 1, self.columns)
        return row * self.columns + (self.columns - 1 - column) + 1


def scenario_id(size: str, variant: str, seed: int, divisions: int | None = None, armor: int = 0) -> str:
    count = DEFAULT_DIVISIONS[size] if divisions is None else divisions
    return f"sim-{size}-{variant}-d{count}a{armor}-s{seed}"


def _sector(rows: int, row: int) -> str:
    if rows == 2:
        return ("north", "south")[row]
    return ("north", "center", "south")[row * 3 // rows]


def _capital_row(rows: int) -> int:
    return 0 if rows == 2 else 2


def _front_vp_rows(rows: int) -> tuple[int, ...]:
    return (1,) if rows == 2 else (0, 3, 5)


def scenario(identifier: str) -> Scenario:
    """Build the scenario named by ``identifier``; the same ID always gives the same scenario."""
    match = _ID.match(identifier)
    if match is None or match.group(2) not in VARIANTS:
        raise ArenaError(f"unknown simulator scenario {identifier!r}")
    size, variant = match.group(1), match.group(2)
    divisions, armor, seed = int(match.group(3)), int(match.group(4)), int(match.group(5))
    if not 1 <= divisions <= 24 or armor > divisions:
        raise ArenaError("scenario needs 1-24 divisions per side and armor <= divisions")
    columns, rows = SIZES[size]
    half = columns // 2
    rng = random.Random(f"{size}:{variant}:{seed}")

    def pid(row: int, column: int) -> int:
        return row * columns + column + 1

    names, weights = zip(*_TERRAIN_WEIGHTS)
    terrain: dict[tuple[int, int], str] = {}
    for row in range(rows):
        for column in range(half):
            kind = rng.choices(names, weights)[0]
            if column == half - 1 and variant == "breakthrough":
                kind = rng.choice(("forest", "hills"))
            if variant == "defence" and column == half - 1 and row in _front_vp_rows(rows):
                kind = "hills"
            if column == 0 and row == _capital_row(rows):
                kind = "urban"
            terrain[row, column] = terrain[row, columns - 1 - column] = kind
    # Rivers lie on west-east edges. edge (row, c) joins columns c and c + 1.
    rivers: set[tuple[int, int]] = set()
    for row in range(rows):
        for column in range(half - 1):
            if rng.random() < 0.15:
                rivers |= {(row, column), (row, columns - 2 - column)}
        if variant == "crossing" or rng.random() < 0.25:
            rivers.add((row, half - 1))
    # Encirclement: the two central provinces of the pocket row swap owners, so each side
    # starts with one division cut off inside enemy territory.
    pocket_row = rng.randrange(rows) if variant == "encirclement" else None

    def owner(row: int, column: int) -> Country:
        west = column < half
        if row == pocket_row and column in (half - 1, half):
            west = not west
        return Country.BLUE if west else Country.RED

    provinces = []
    for row in range(rows):
        for column in range(columns):
            neighbors, crossings = [], []
            for d_row, d_column in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                n_row, n_column = row + d_row, column + d_column
                if not (0 <= n_row < rows and 0 <= n_column < columns):
                    continue
                # On the full map the three routes only join in the two rear columns of each side.
                if d_row and rows > 2 and _sector(rows, row) != _sector(rows, n_row) and 2 <= column < columns - 2:
                    continue
                neighbors.append(pid(n_row, n_column))
                if d_column and (row, min(column, n_column)) in rivers:
                    crossings.append(pid(n_row, n_column))
            capital = row == _capital_row(rows) and column in (0, columns - 1)
            front_vp = row in _front_vp_rows(rows) and column in (half - 1, half)
            provinces.append(LayoutProvince(
                pid(row, column), round((column + 0.5) / columns, 4), round((row + 0.5) / rows, 4),
                terrain[row, column], tuple(sorted(neighbors)), tuple(sorted(crossings)), _sector(rows, row),
                CAPITAL_VP if capital else FRONT_VP if front_vp else 0.0, owner(row, column).value,
                (Country.BLUE.value if column == 0 else Country.RED.value) if capital else ""))
    layout = ArenaLayout(identifier, tuple(provinces), "synthetic")

    # West-side deployment slots as (row, column); RED gets the mirror image.
    start_column = 0 if rows == 2 else 1
    if variant in ("breakthrough", "retreat", "supply"):
        slots = [(row, half - 1) for row in range(rows)]
    elif variant == "defence":
        slots = [(_capital_row(rows), 0)] + [(row, half - 1) for row in _front_vp_rows(rows)]
    else:
        slots = [(_capital_row(rows), 0)] + [(row, start_column) for row in range(rows)
                                              if (row, start_column) != (_capital_row(rows), 0)]
    if pocket_row is not None:
        slots = [(pocket_row, half)] + [slot for slot in slots if slot != (pocket_row, half - 1)]
    organization = 0.35 if variant == "retreat" else 1.0
    strength = 0.8 if variant == "retreat" else 1.0
    deployments = []
    for index in range(divisions):
        row, column = slots[index % len(slots)]
        kind = "armor" if index >= divisions - armor else "infantry"
        deployments.append(Deployment(Country.BLUE.value, pid(row, column), kind, organization, strength))
        deployments.append(Deployment(Country.RED.value, pid(row, columns - 1 - column), kind, organization,
                                      strength))
    # Supply disruption: the front column is already at half supply and enemy ground has none.
    reach = (max(0, half - 2), 2) if variant == "supply" else (max(3, half), 4)
    return Scenario(layout, tuple(deployments), size, variant, seed, reach[0], reach[1], columns, rows)


def scenario_split(identifier: str) -> str:
    """Deterministic 70/15/15 assignment from the scenario ID alone; decided before any training."""
    bucket = int.from_bytes(hashlib.sha256(identifier.encode("utf-8")).digest()[:4], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "held_out"


def scenario_ids(split: str, size: str, count: int, variants: tuple[str, ...] = VARIANTS,
                 divisions: int | None = None, armor: int = 0) -> tuple[str, ...]:
    """The first ``count`` scenario IDs of a split, cycling variants over increasing seeds."""
    if split not in SPLITS or size not in SIZES or not variants or count <= 0:
        raise ArenaError("unknown split/size, no variants, or non-positive count")
    found: list[str] = []
    seed = 0
    while len(found) < count:
        for variant in variants:
            identifier = scenario_id(size, variant, seed, divisions, armor)
            if scenario_split(identifier) == split and len(found) < count:
                found.append(identifier)
        seed += 1
    return tuple(found)
