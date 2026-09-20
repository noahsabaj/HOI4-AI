"""Deterministic players over the player view: intent execution and scripted opponents.

Everything here follows the Policy protocol (observation, memory, intent) -> (Order, memory)
and sees only a PlayerObservation, so it runs unchanged in the simulator and the real game.
The intent executor is what turns Tier-2 stances into orders when no learned policy is used.
"""
from __future__ import annotations

import random
import uuid
from collections import deque
from typing import Any

from .contracts import Intent, Order, PlayerObservation, Province, UnitView, Verb


def next_step(provinces: dict[int, Province], start: int, goal: int) -> int | None:
    """First province on a shortest path from start to goal; None if equal or unreachable."""
    if start == goal:
        return None
    previous: dict[int, int] = {start: start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(provinces[current].neighbors):
            if neighbor in previous:
                continue
            previous[neighbor] = current
            if neighbor == goal:
                while previous[neighbor] != start:
                    neighbor = previous[neighbor]
                return neighbor
            queue.append(neighbor)
    return None


def distance(provinces: dict[int, Province], start: int, goal: int) -> int:
    seen, queue = {start: 0}, deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            return seen[current]
        for neighbor in provinces[current].neighbors:
            if neighbor not in seen:
                seen[neighbor] = seen[current] + 1
                queue.append(neighbor)
    return 10**6


def objective(observation: PlayerObservation, unit: UnitView, stance: str) -> int | None:
    """Where a unit should head: forward to the best objective we do not hold, or back to ours."""
    provinces = {province.id: province for province in observation.provinces}
    mine = observation.country
    if stance == "attack":
        candidates = [p for p in observation.provinces if p.victory_points > 0 and p.controller != mine]
    elif stance == "retreat":
        candidates = [p for p in observation.provinces if p.victory_points > 0 and p.controller == mine]
    else:
        return None
    if not candidates:
        return None
    same_sector = [p for p in candidates if p.sector and p.sector == provinces[unit.province_id].sector]
    pool = same_sector if stance == "attack" and same_sector else candidates
    best = min(pool, key=lambda p: (distance(provinces, unit.province_id, p.id) - p.victory_points, p.id))
    return best.id


def order_for(observation: PlayerObservation, unit: UnitView, stance: str) -> Order | None:
    """The single order that moves this unit toward its stance, or None if it already complies."""
    provinces = {province.id: province for province in observation.provinces}
    goal = objective(observation, unit, stance)
    if goal is None:  # hold, or nothing to aim at
        if unit.order_target_province_id is not None:
            return _order(observation, Verb.CANCEL, unit)
        return None
    step = next_step(provinces, unit.province_id, goal)
    if step is None or unit.order_target_province_id == step:
        return None
    return _order(observation, Verb.MOVE, unit, step)


def _order(observation: PlayerObservation, verb: Verb, unit: UnitView | None = None,
           target: int | None = None) -> Order:
    return Order(uuid.uuid4().hex, observation.episode_id, observation.sequence, observation.country,
                 verb, () if unit is None else (unit.id,), target)


class IntentExecutor:
    """One order per decision, cycling through own units, following the given intent."""

    def __init__(self, default_stance: str = "hold") -> None:
        self.default_stance = default_stance

    def act(self, observation: PlayerObservation, memory: Any,
            intent: Intent | None = None) -> tuple[Order, Any]:
        cursor = memory if isinstance(memory, int) else 0
        own = sorted((u for u in observation.units if u.country == observation.country), key=lambda u: u.id)
        if observation.terminal or not own:
            return _order(observation, Verb.NOOP), cursor
        sectors = {province.id: province.sector for province in observation.provinces}
        for offset in range(len(own)):
            unit = own[(cursor + offset) % len(own)]
            stance = (intent.stance(sectors[unit.province_id]) if intent is not None else None) or self.default_stance
            order = order_for(observation, unit, stance)
            if order is not None:
                return order, (cursor + offset + 1) % len(own)
        return _order(observation, Verb.NOOP), cursor


class ScriptedPolicy:
    """Fixed-style opponents for panels and demonstrations: hold, advance, flank, random."""

    STYLES = ("hold", "advance", "flank", "random")

    def __init__(self, style: str, seed: int = 0) -> None:
        if style not in self.STYLES:
            raise ValueError(f"unknown scripted style {style!r}")
        self.style, self.rng = style, random.Random(seed)
        self.executor = IntentExecutor("hold")
        self.flank_sector: str | None = None

    @property
    def id(self) -> str:
        return f"{self.style}-v1"

    def act(self, observation: PlayerObservation, memory: Any,
            intent: Intent | None = None) -> tuple[Order, Any]:
        if self.style == "random":
            from .actions import choices
            unit_choices = [c for c in choices(observation) if c.verb not in (Verb.PAUSE, Verb.SET_SPEED)]
            return self.rng.choice(unit_choices).order(observation, uuid.uuid4().hex), memory
        sectors = sorted({p.sector for p in observation.provinces if p.sector})
        if self.style == "hold":
            stances = dict.fromkeys(sectors, "hold")
        elif self.style == "advance":
            stances = dict.fromkeys(sectors, "attack")
        else:
            if self.flank_sector not in sectors:
                self.flank_sector = self.rng.choice(sectors) if sectors else None
            stances = {sector: "attack" if sector == self.flank_sector else "hold" for sector in sectors}
        scripted = Intent.of(stances, observation.game_hour, source=self.id) if sectors else None
        if scripted is None and self.style == "advance":
            self.executor.default_stance = "attack"
        return self.executor.act(observation, memory, scripted)
