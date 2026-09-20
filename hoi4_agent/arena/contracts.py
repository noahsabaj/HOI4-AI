"""Versioned player-facing contracts. Full engine state never belongs here."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 4 * 1024 * 1024


class ArenaError(RuntimeError):
    """An actionable configuration, transport, or engine failure."""


class CapabilityError(ArenaError):
    """A required, verified engine capability is unavailable."""


class Country(StrEnum):
    BLUE = "BLU"
    RED = "RED"

    @property
    def opponent(self) -> Country:
        return Country.RED if self is Country.BLUE else Country.BLUE


class Verb(StrEnum):
    NOOP = "noop"
    MOVE = "move"  # HOI4 has no separate attack order: moving into a held province is the attack.
    SUPPORT_ATTACK = "support_attack"  # join an adjacent battle without advancing
    CANCEL = "cancel"
    SET_SPEED = "set_speed"  # game speed 1-5; there are no human-like rate rules
    PAUSE = "pause"  # toggles pause


UNIT_KINDS = ("infantry", "armor", "unknown")
STANCES = ("attack", "hold", "retreat")


def finite(value: float, name: str, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ArenaError(f"{name} must be finite and >= {minimum}")


def fraction(value: float | None, name: str) -> None:
    if value is not None:
        finite(value, name)
        if value > 1:
            raise ArenaError(f"{name} must be <= 1")


def integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ArenaError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class BuildFingerprint:
    executable_sha256: str
    game_version: str
    dlc_sha256: str
    mod_sha256: str

    def __post_init__(self) -> None:
        for name in ("executable_sha256", "dlc_sha256", "mod_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ArenaError(f"{name} must be a lowercase SHA-256 digest")
        if not isinstance(self.game_version, str) or not self.game_version:
            raise ArenaError("game_version is required")


@dataclass(frozen=True)
class ArenaSpec:
    scenario_id: str
    fingerprint: BuildFingerprint
    scenario_seed: int = 0
    horizon_hours: int = 90 * 24
    decisions_per_second: float = 2.0  # pacing hint only: there is no action cap
    orders_per_second: float = 1.0
    max_observation_age_ms: int = 500  # screen perception alone takes 100-140 ms

    def __post_init__(self) -> None:
        integer(self.scenario_seed, "scenario_seed")
        integer(self.horizon_hours, "horizon_hours", 1)
        integer(self.max_observation_age_ms, "max_observation_age_ms", 1)
        if not self.scenario_id or self.horizon_hours <= 0 or self.max_observation_age_ms <= 0:
            raise ArenaError("scenario, positive horizon and observation deadline are required")
        for name in ("decisions_per_second", "orders_per_second"):
            finite(getattr(self, name), name, 0.001)


@dataclass(frozen=True)
class Province:
    id: int
    x: float
    y: float
    terrain: str
    neighbors: tuple[int, ...]
    controller: Country | None = None  # None means unknown, not neutral.
    victory_points: float = 0.0
    supply: float | None = None
    river_neighbors: tuple[int, ...] = ()  # static map data: adjacencies that cross a river
    sector: str = ""  # static approach route this province belongs to; "" when none

    def __post_init__(self) -> None:
        integer(self.id, "province ID", 1)
        for neighbor in self.neighbors:
            integer(neighbor, "neighbor ID", 1)
        if not set(self.river_neighbors) <= set(self.neighbors) or len(set(self.river_neighbors)) != len(self.river_neighbors):
            raise ArenaError("river crossings must be unique adjacent provinces")
        if self.controller is not None and not isinstance(self.controller, Country):
            raise ArenaError("invalid province controller")
        if self.id <= 0 or self.id in self.neighbors or len(set(self.neighbors)) != len(self.neighbors):
            raise ArenaError("invalid province identity/adjacency")
        finite(self.x, "x")
        finite(self.y, "y")
        fraction(self.x, "x")
        fraction(self.y, "y")
        fraction(self.supply, "supply")
        finite(self.victory_points, "victory_points")


@dataclass(frozen=True)
class UnitView:
    id: int
    country: Country
    province_id: int
    organization: float | None
    strength: float | None
    supply: float | None
    in_combat: bool | None = None
    kind: str = "unknown"
    # Own units: current movement/attack destination, None when idle.
    # Enemy contacts: only what the player can see; None means unknown.
    order_target_province_id: int | None = None
    entrenchment: float | None = None
    # The map shows a stack as one counter: division count plus AVERAGE bars.
    # Own divisions read from the army panel are count == 1.
    count: int = 1
    confidence: float = 1.0  # perception confidence in this view; 1.0 for exact sources
    # Ephemeral player-visible identity for enemy contacts; no hidden engine IDs.

    def __post_init__(self) -> None:
        integer(self.id, "unit ID")
        integer(self.province_id, "unit province ID", 1)
        if self.kind not in UNIT_KINDS:
            raise ArenaError("unknown unit kind")
        if self.order_target_province_id is not None:
            integer(self.order_target_province_id, "unit order target", 1)
        fraction(self.entrenchment, "entrenchment")
        integer(self.count, "stack division count", 1)
        fraction(self.confidence, "confidence")
        if not isinstance(self.country, Country) or (self.in_combat is not None and type(self.in_combat) is not bool):
            raise ArenaError("invalid unit country/combat state")
        if self.id < 0 or self.province_id <= 0:
            raise ArenaError("invalid unit identity/location")
        for name in ("organization", "strength", "supply"):
            fraction(getattr(self, name), name)


@dataclass(frozen=True)
class PlayerObservation:
    episode_id: str
    country: Country
    sequence: int
    game_hour: int
    captured_monotonic_ns: int
    provinces: tuple[Province, ...]
    units: tuple[UnitView, ...]
    terminal: bool = False
    winner: Country | None = None
    game_speed: int = 1
    paused: bool = False

    def __post_init__(self) -> None:
        for name in ("sequence", "game_hour", "captured_monotonic_ns"):
            integer(getattr(self, name), name)
        if type(self.game_speed) is not int or not 1 <= self.game_speed <= 5 or type(self.paused) is not bool:
            raise ArenaError("game speed must be 1-5 and paused a boolean")
        if (not isinstance(self.country, Country) or type(self.terminal) is not bool or
            (self.winner is not None and not isinstance(self.winner, Country))):
            raise ArenaError("invalid observation player/outcome")
        if not self.episode_id or min(self.sequence, self.game_hour, self.captured_monotonic_ns) < 0:
            raise ArenaError("invalid observation epoch or timestamp")
        ids = {p.id for p in self.provinces}
        if not ids or len(ids) != len(self.provinces):
            raise ArenaError("province IDs must be present and unique")
        if len({u.id for u in self.units}) != len(self.units):
            raise ArenaError("unit IDs must be unique within the player view")
        if any(not set(p.neighbors) <= ids for p in self.provinces):
            raise ArenaError("adjacency references unknown provinces")
        if any(u.province_id not in ids or (u.order_target_province_id is not None and
                                             u.order_target_province_id not in ids) for u in self.units):
            raise ArenaError("unit references an unknown province")
        if self.winner is not None and not self.terminal:
            raise ArenaError("nonterminal observation cannot have a winner")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerObservation:
        try:
            values = dict(data)
            values["country"] = Country(values["country"])
            values["winner"] = Country(values["winner"]) if values.get("winner") else None
            values["provinces"] = tuple(
                Province(**{**p, "controller": Country(p["controller"]) if p["controller"] else None,
                            "neighbors": tuple(p["neighbors"]),
                            "river_neighbors": tuple(p.get("river_neighbors", ()))}) for p in values["provinces"]
            )
            values["units"] = tuple(
                UnitView(**{**u, "country": Country(u["country"])}) for u in values["units"]
            )
            return cls(**values)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArenaError(f"invalid player observation: {exc}") from exc


@dataclass(frozen=True)
class Order:
    id: str
    episode_id: str
    observation_sequence: int
    country: Country
    verb: Verb
    unit_ids: tuple[int, ...] = ()
    target_province_id: int | None = None
    speed: int | None = None

    def __post_init__(self) -> None:
        integer(self.observation_sequence, "observation_sequence")
        if (self.verb is Verb.SET_SPEED) != (self.speed is not None):
            raise ArenaError("exactly set_speed orders carry a speed")
        if self.speed is not None and (type(self.speed) is not int or not 1 <= self.speed <= 5):
            raise ArenaError("speed must be 1-5")
        for unit_id in self.unit_ids:
            integer(unit_id, "selected unit ID")
        if self.target_province_id is not None:
            integer(self.target_province_id, "target province ID", 1)
        if not isinstance(self.country, Country) or not isinstance(self.verb, Verb):
            raise ArenaError("order country and verb must be valid enum values")
        if not self.id or not self.episode_id or self.observation_sequence < 0:
            raise ArenaError("order identity, episode and observation are required")
        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise ArenaError("duplicate units in order")
        if self.verb in (Verb.NOOP, Verb.SET_SPEED, Verb.PAUSE):
            if self.unit_ids or self.target_province_id is not None:
                raise ArenaError("noop/speed/pause must not carry units or a target")
        elif not self.unit_ids:
            raise ArenaError("non-noop order requires units")
        if self.verb in (Verb.MOVE, Verb.SUPPORT_ATTACK) and self.target_province_id is None:
            raise ArenaError("move/support attack requires a target")
        if self.verb is Verb.CANCEL and self.target_province_id is not None:
            raise ArenaError("cancel must not carry a target")


@dataclass(frozen=True)
class OrderReceipt:
    order_id: str
    episode_id: str
    accepted: bool
    reason: str
    applied_game_hour: int | None
    # accepted means enqueued as a normal engine order, not successful combat.

    def __post_init__(self) -> None:
        if not self.order_id or not self.episode_id or type(self.accepted) is not bool or not self.reason:
            raise ArenaError("receipt identity, acceptance and reason are required")
        if self.accepted and self.applied_game_hour is None:
            raise ArenaError("accepted receipt must identify the game hour of engine dispatch")
        if self.applied_game_hour is not None and self.applied_game_hour < 0:
            raise ArenaError("negative receipt game hour")
        if self.applied_game_hour is not None:
            integer(self.applied_game_hour, "receipt game hour")


@dataclass(frozen=True)
class Intent:
    """Tier-2 guidance: one stance per sector, stamped with when it was produced.

    Consumers must look at ``age_hours``: at speed 5 a cloud call can span days.
    """
    stances: tuple[tuple[str, str], ...]  # (sector, stance), sorted by sector
    produced_game_hour: int
    source: str = "scripted"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        integer(self.produced_game_hour, "intent game hour")
        fraction(self.confidence, "intent confidence")
        sectors = [sector for sector, _ in self.stances]
        if sectors != sorted(set(sectors)) or any(not sector or stance not in STANCES for sector, stance in self.stances):
            raise ArenaError("intent needs unique sorted sectors with known stances")
        if not self.source:
            raise ArenaError("intent source is required")

    @classmethod
    def of(cls, stances: dict[str, str], produced_game_hour: int, source: str = "scripted",
           confidence: float = 1.0) -> Intent:
        return cls(tuple(sorted(stances.items())), produced_game_hour, source, confidence)

    def stance(self, sector: str) -> str | None:
        return dict(self.stances).get(sector)

    def age_hours(self, game_hour: int) -> int:
        return max(0, game_hour - self.produced_game_hour)


class ArenaSession(Protocol):
    def reset(self, spec: ArenaSpec, model_countries: tuple[Country, ...]) -> str: ...
    def observe(self, country: Country) -> PlayerObservation: ...
    def submit(self, order: Order) -> OrderReceipt: ...
    def close(self) -> None: ...


class Policy(Protocol):
    def act(self, observation: PlayerObservation, memory: Any,
            intent: Intent | None = None) -> tuple[Order, Any]: ...
