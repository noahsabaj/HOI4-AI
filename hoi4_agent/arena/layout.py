"""Static arena map shared by the mod generator, the simulator and perception.

Coordinates are normalized to the arena map view (0,0 top-left; 1,1 bottom-right), the
same frame the fixed camera shows. Perception maps them to screen pixels through its own
calibration; nothing here depends on a resolution.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import ArenaError, Country, Province
from .diagnostics import write_json


@dataclass(frozen=True)
class LayoutProvince:
    id: int
    x: float
    y: float
    terrain: str
    neighbors: tuple[int, ...]
    river_neighbors: tuple[int, ...] = ()
    sector: str = ""
    victory_points: float = 0.0
    initial_controller: str = ""  # Country value, or "" for neutral/impassable-adjacent
    capital_of: str = ""  # Country value when this province is that side's capital

    def view(self, controller: Country | None, supply: float | None = None) -> Province:
        return Province(self.id, self.x, self.y, self.terrain, self.neighbors, controller,
                        self.victory_points, supply, self.river_neighbors, self.sector)


@dataclass(frozen=True)
class ArenaLayout:
    scenario_id: str
    provinces: tuple[LayoutProvince, ...]
    source: str = "synthetic"  # "synthetic" or "vanilla_region"

    def __post_init__(self) -> None:
        ids = {province.id for province in self.provinces}
        if not self.scenario_id or len(ids) != len(self.provinces) or len(ids) < 2:
            raise ArenaError("layout needs a scenario ID and unique provinces")
        by_id = {province.id: province for province in self.provinces}
        for province in self.provinces:
            if not set(province.neighbors) <= ids or province.id in province.neighbors:
                raise ArenaError(f"province {province.id} has invalid adjacency")
            if any(province.id not in by_id[n].neighbors for n in province.neighbors):
                raise ArenaError(f"province {province.id} adjacency is not symmetric")
            if province.initial_controller not in ("", *(c.value for c in Country)):
                raise ArenaError("unknown initial controller")
        for country in Country:
            if sum(province.capital_of == country.value for province in self.provinces) != 1:
                raise ArenaError(f"layout needs exactly one capital for {country.value}")

    def province(self, province_id: int) -> LayoutProvince:
        for province in self.provinces:
            if province.id == province_id:
                return province
        raise ArenaError(f"unknown province {province_id}")

    def capital(self, country: Country) -> LayoutProvince:
        return next(p for p in self.provinces if p.capital_of == country.value)

    @property
    def sectors(self) -> tuple[str, ...]:
        return tuple(sorted({province.sector for province in self.provinces if province.sector}))

    def save(self, path: Path) -> None:
        write_json(path, {"schema_version": 1, **asdict(self)})

    @classmethod
    def load(cls, path: Path) -> ArenaLayout:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.pop("schema_version", None) != 1:
            raise ArenaError("unsupported layout schema")
        try:
            provinces = tuple(LayoutProvince(**{**row, "neighbors": tuple(row["neighbors"]),
                                                "river_neighbors": tuple(row.get("river_neighbors", ()))})
                              for row in data["provinces"])
            return cls(data["scenario_id"], provinces, data.get("source", "synthetic"))
        except (KeyError, TypeError) as exc:
            raise ArenaError(f"invalid layout: {exc}") from exc
