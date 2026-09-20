"""``ArenaSession`` over the SIMULATOR: fogged, optionally noisy player views and order receipts.

Loop: every model country calls ``observe`` then ``submit`` (any number of orders, there is
no action cap), then the driver calls ``step()`` once, which advances game time by what the
speed model says one decision costs. Observations are cached per step, so ``observe`` is
stable until ``step`` and trajectories are continuous.

Fog of war. A side sees its own divisions one by one (``count == 1``). Enemy divisions are
seen only in provinces adjacent to (or equal to) a province the side controls or occupies,
and only as ONE stack per province: division count, average organization and strength,
majority kind. Stack IDs are ephemeral (>= 1001), handed out per observer in the order stacks
become visible, kept only while that province keeps showing a stack; they never derive from
engine IDs. Enemy orders, supply and entrenchment are never shown.
Province controller is reported for EVERY province, visible or not: HOI4's map recolours
control changes everywhere in real time, fog only hides units. So a hidden enemy capturing a
province is (faithfully) visible as a control change; hidden unit positions, numbers and
condition never are.

Time. Game speed (1-5) and pause are shared session state that orders change; by default any
model country may change them, ``speed_authority`` restricts that to one (the LAN host).
Both the hours-per-second table and the decision latency are PLACEHOLDERS to be calibrated
against the real game. While paused no game time passes, so zero-hour transitions are legal.
``max_decisions`` ends a pause loop as a truncated draw.

Perception noise (``NoiseConfig``) imitates the screen reader: bars quantized to about 1/30,
enemy stacks dropped for a frame, division counts misread by one, and a ``confidence`` that
is lower, on average, for the misread frames. Own units stay exact apart from quantization.
The rates are placeholders until the real reader is audited against save files.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass

from ..contracts import (ArenaError, ArenaSpec, BuildFingerprint, Country, Order, OrderReceipt, PlayerObservation,
                         Province, UnitView, Verb)
from .engine import Engine, Rules
from .maps import Scenario, scenario

SIMULATOR_VERSION = "simulator-0.1"
ENEMY_STACK_ID_BASE = 1001


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def simulator_fingerprint(rules: Rules | None = None) -> BuildFingerprint:
    """Synthetic fingerprint: NOT a game build. It changes whenever the simulator rules change."""
    return BuildFingerprint(_digest("SIMULATOR: not a HOI4 executable"), SIMULATOR_VERSION,
                            _digest("SIMULATOR: no DLC"), _digest(f"SIMULATOR rules {rules or Rules()!r}"))


SIM_FINGERPRINT = simulator_fingerprint()


@dataclass(frozen=True)
class SpeedModel:
    """Game hours between decisions = max(interval, hours_per_second[speed] * latency).

    PLACEHOLDER numbers, to be calibrated against the real game and the real executor.
    ``decision_interval_hours`` is the policy's game-time pacing: it is not asked more often.
    """
    hours_per_second: tuple[float, float, float, float, float] = (1.0, 2.0, 4.0, 8.0, 24.0)
    decision_latency_ms: float = 500.0
    decision_interval_hours: float = 1.0

    def __post_init__(self) -> None:
        if (len(self.hours_per_second) != 5 or min(self.hours_per_second) <= 0 or self.decision_latency_ms <= 0
                or self.decision_interval_hours < 0):
            raise ArenaError("speed model needs five positive rates, positive latency, non-negative interval")

    def hours(self, speed: int) -> float:
        return max(self.decision_interval_hours, self.hours_per_second[speed - 1] * self.decision_latency_ms / 1000)


@dataclass(frozen=True)
class NoiseConfig:
    bar_levels: int = 30
    stack_dropout: float = 0.05
    count_misread: float = 0.05
    min_confidence: float = 0.6

    def __post_init__(self) -> None:
        if self.bar_levels < 1 or not all(0 <= x <= 1 for x in (self.stack_dropout, self.count_misread)) \
                or not 0 <= self.min_confidence < 1:
            raise ArenaError("invalid noise configuration")


@dataclass(frozen=True)
class SimConfig:
    speed_model: SpeedModel = SpeedModel()
    noise: NoiseConfig | None = NoiseConfig()  # on by default: train against imperfect perception
    initial_speed: int = 3
    start_paused: bool = False
    max_decisions: int = 5000
    speed_authority: Country | None = None
    rules: Rules | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.initial_speed <= 5 or self.max_decisions <= 0:
            raise ArenaError("initial speed must be 1-5 and the decision cap positive")


class SimSession:
    """One simulated match with one or two model-controlled countries (the other side idles)."""

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()
        self.fingerprint = simulator_fingerprint(self.config.rules)
        self.engine: Engine | None = None
        self.episodes = 0
        self.episode_id = ""

    # ----- ArenaSession --------------------------------------------------------------------
    def reset(self, spec: ArenaSpec, model_countries: tuple[Country, ...]) -> str:
        if not model_countries or len(set(model_countries)) != len(model_countries):
            raise ArenaError("one or two distinct model countries are required")
        if spec.fingerprint != self.fingerprint:
            raise ArenaError("spec fingerprint is not this simulator's synthetic fingerprint")
        self.scenario: Scenario = scenario(spec.scenario_id)
        self.engine = Engine(self.scenario, spec.scenario_seed, spec.horizon_hours, self.config.rules)
        self.spec, self.model_countries = spec, tuple(model_countries)
        self.episodes += 1
        self.episode_id = f"sim:{spec.scenario_id}:{spec.scenario_seed}:{self.episodes}"
        self.sequence = self.decisions = 0
        self.wall_ns = 0
        self.speed, self.paused = self.config.initial_speed, self.config.start_paused
        self.truncated = False
        self._carry = 0.0
        self._cache: dict[Country, PlayerObservation] = {}
        self._stack_ids: dict[Country, dict[int, int]] = {country: {} for country in Country}
        self._next_stack_id = dict.fromkeys(Country, ENEMY_STACK_ID_BASE)
        self._noise_rng = {country: random.Random(f"noise:{spec.scenario_id}:{spec.scenario_seed}:{country.value}")
                           for country in Country}
        self._province_cache: dict[tuple[int, Country | None, float | None], Province] = {}
        self.rejections: Counter[str] = Counter()
        self.accepted = 0
        return self.episode_id

    def observe(self, country: Country) -> PlayerObservation:
        self._require(country)
        if country not in self._cache:
            self._cache[country] = self._build(country)
        return self._cache[country]

    def submit(self, order: Order) -> OrderReceipt:
        engine = self._require(order.country)
        reason = self._apply(engine, order)
        if reason is None:
            self.accepted += 1
            return OrderReceipt(order.id, self.episode_id, True, "accepted", engine.hour)
        self.rejections[reason] += 1
        return OrderReceipt(order.id, self.episode_id, False, reason, None)

    def close(self) -> None:
        self.engine = None
        self._cache = {}

    # ----- pacing --------------------------------------------------------------------------
    def step(self) -> int:
        """Advance by one decision's worth of game time; returns the whole hours that passed."""
        if self.engine is None:
            raise ArenaError("session is not reset")
        engine = self.engine
        if engine.terminal:
            return 0
        self.decisions += 1
        self.wall_ns += int(self.config.speed_model.decision_latency_ms * 1_000_000)
        hours = 0
        if not self.paused:
            self._carry += self.config.speed_model.hours(self.speed)
            hours = int(self._carry)
            self._carry -= hours
            before = engine.hour
            engine.advance(hours)
            hours = engine.hour - before
        if not engine.terminal and self.decisions >= self.config.max_decisions:
            engine.truncate()
            self.truncated = True
        self.sequence += 1
        self._cache = {}
        return hours

    @property
    def terminal(self) -> bool:
        return self.engine is not None and self.engine.terminal

    # ----- internals -----------------------------------------------------------------------
    def _require(self, country: Country) -> Engine:
        if self.engine is None:
            raise ArenaError("session is not reset")
        if country not in self.model_countries:
            raise ArenaError(f"{country.value} is not model-controlled in this session")
        return self.engine

    def _apply(self, engine: Engine, order: Order) -> str | None:
        if order.episode_id != self.episode_id:
            return "wrong episode"
        if engine.terminal:
            return "episode is over"
        if order.observation_sequence != self.sequence:
            return "stale observation"
        if order.verb is Verb.NOOP:
            return None
        if order.verb in (Verb.SET_SPEED, Verb.PAUSE):
            if self.config.speed_authority not in (None, order.country):
                return "country does not control speed and pause"
            if order.verb is Verb.PAUSE:
                self.paused = not self.paused
            else:
                assert order.speed is not None
                self.speed = order.speed
            return None
        for unit_id in order.unit_ids:  # validate everything before changing anything
            if unit_id >= ENEMY_STACK_ID_BASE:
                return "not your unit"
            if order.verb is Verb.CANCEL:
                reason = None if engine.division(order.country, unit_id) else "unknown unit"
            else:
                assert order.target_province_id is not None
                reason = engine.order(order.country, unit_id, order.target_province_id,
                                      order.verb is Verb.SUPPORT_ATTACK, dry_run=True)
            if reason is not None:
                return reason
        for unit_id in order.unit_ids:
            if order.verb is Verb.CANCEL:
                engine.cancel(order.country, unit_id)
            else:
                assert order.target_province_id is not None
                engine.order(order.country, unit_id, order.target_province_id, order.verb is Verb.SUPPORT_ATTACK)
        return None

    def visible_provinces(self, country: Country) -> set[int]:
        assert self.engine is not None
        engine = self.engine
        held = {province for province, controller in engine.control.items() if controller == country}
        held |= {division.province for division in engine.army(country)}
        return held | {neighbor for province in held for neighbor in engine.neighbors[province]}

    def _province(self, province_id: int, controller: Country | None, supply: float | None) -> Province:
        key = (province_id, controller, supply)
        view = self._province_cache.get(key)
        if view is None:
            view = self._province_cache[key] = self.scenario.layout.province(province_id).view(controller, supply)
        return view

    def _build(self, country: Country) -> PlayerObservation:
        assert self.engine is not None
        engine, noise, rng = self.engine, self.config.noise, self._noise_rng[country]
        levels = noise.bar_levels if noise else 0

        def bar(value: float) -> float:
            value = min(1.0, max(0.0, value))
            return round(value * levels) / levels if levels else value

        provinces = tuple(self._province(
            province, controller,
            round(engine.supply[country].get(province, 0.0), 2) if controller == country else None)
            for province, controller in engine.control.items())
        units = [UnitView(d.id, country, d.province, bar(d.organization), bar(d.strength), bar(d.supply),
                          d.in_combat, d.kind, d.target, bar(d.entrenchment))
                 for d in engine.army(country)]
        visible = self.visible_provinces(country)
        stacks: dict[int, list] = {}
        for division in engine.army(country.opponent):
            if division.province in visible:
                stacks.setdefault(division.province, []).append(division)
        ids = self._stack_ids[country]
        for province in [p for p in ids if p not in stacks]:
            del ids[province]
        for province in sorted(stacks):
            members = stacks[province]
            if province not in ids:
                ids[province] = self._next_stack_id[country]
                self._next_stack_id[country] += 1
            count, confidence = len(members), 1.0
            if noise:
                dropped = rng.random() < noise.stack_dropout
                confidence = 1.0 - rng.random() * (1.0 - noise.min_confidence)
                # Misreads are likelier on low-confidence frames; the mean rate is count_misread.
                misread = rng.random() < noise.count_misread * (1.0 - confidence) / ((1.0 - noise.min_confidence) / 2)
                direction = 1 if rng.random() < 0.5 else -1
                if dropped:
                    continue
                if misread:
                    count = max(1, count + direction)
            kinds = Counter(d.kind for d in members).most_common()
            kind = kinds[0][0] if len(kinds) == 1 or kinds[0][1] > kinds[1][1] else "unknown"
            units.append(UnitView(ids[province], country.opponent, province,
                                  bar(sum(d.organization for d in members) / len(members)),
                                  bar(sum(d.strength for d in members) / len(members)), None,
                                  any(d.in_combat for d in members), kind, None, None, count,
                                  round(confidence, 3)))
        return PlayerObservation(self.episode_id, country, self.sequence, engine.hour, self.wall_ns, provinces,
                                 tuple(units), engine.terminal, engine.winner, self.speed, self.paused)
