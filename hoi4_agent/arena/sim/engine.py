"""Full-state SIMULATOR of two-country land combat, advanced in game hours.

THIS IS NOT HEARTS OF IRON IV. It is a small, fast stand-in with the same shape of problem,
so a policy can do reinforcement learning at thousands of games per hour. Its results are
never strength evidence for the real game, and NO COEFFICIENT BELOW IS CALIBRATED against
HOI4; they are round placeholders chosen so that games resolve within the 90-day horizon.

HOI4 mechanics that are approximated (and how):

* Movement: a division walks to an adjacent province in a terrain-dependent number of hours
  (rivers add time, armor is faster on open ground). Control flips when a division enters.
* Attack = move into a province holding enemy divisions. Attackers stay in their origin
  province while the battle runs and walk in once the defenders are gone (movement progress
  keeps accruing during the battle, as in HOI4).
* Support attack: join/start a battle against an adjacent province without advancing.
* Combat: each hour both sides deal organization damage proportional to summed attack or
  defence, spread over the enemy front line; strength falls as a fixed share of organization
  damage. Attackers suffer terrain and river-crossing penalties (larger for armor); defenders
  gain from entrenchment; low supply weakens both; a small dice factor varies each battle hour.
  HOI4's soft/hard attack, armor/piercing, tactics, leaders, air and equipment are absent.
* Combat width: only the ``base_width`` highest-organization divisions per side fight (more
  when attacking from several provinces); the rest are reserves. Divisions in one province
  beyond ``stacking_limit`` also give every fighter there a stacking penalty.
* Retreat: a defender at zero organization retreats at once to an adjacent friendly province
  with no enemy in it (HOI4 retreats take time; here it is instantaneous). With nowhere to
  go it is destroyed: that is the encirclement rule. An attacker at zero organization just stops.
* Entrenchment builds while a division has no order; organization recovers out of combat.
* Supply: a province is supplied by hop distance from the capital through friendly-controlled
  provinces; a division's supply drifts toward its province's. Low supply reduces combat
  power and recovery; very low supply drains organization and strength. HOI4's supply hubs,
  railways and capacity are absent.
* Victory: capture the enemy capital, eliminate the enemy army, or at ``horizon_hours`` hold
  more victory points. Equal points, or simultaneous elimination, is a draw.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from ..contracts import ArenaError, Country
from .maps import Scenario


@dataclass(frozen=True)
class Rules:
    """Uncalibrated placeholder coefficients (see module docstring)."""
    move_hours: tuple[tuple[str, float], ...] = (
        ("plains", 24.0), ("desert", 28.0), ("urban", 30.0), ("forest", 36.0), ("hills", 36.0),
        ("marsh", 48.0), ("mountain", 48.0), ("jungle", 48.0))
    river_move_hours: float = 12.0
    armor_open_speed: float = 0.6  # multiplier on move hours for armor on plains/desert
    attack_penalty: tuple[tuple[str, float], ...] = (
        ("forest", 0.15), ("hills", 0.25), ("urban", 0.3), ("marsh", 0.3), ("mountain", 0.5), ("jungle", 0.3))
    river_penalty: float = 0.3
    armor_terrain_factor: float = 1.5  # armor suffers terrain/river penalties this much more
    attack: tuple[tuple[str, float], ...] = (("infantry", 1.0), ("armor", 1.6))
    defence: tuple[tuple[str, float], ...] = (("infantry", 1.1), ("armor", 0.9))
    armor_breakthrough: float = 0.7  # damage multiplier on attacking armor
    org_damage: float = 0.012  # organization lost per hour per unit of enemy power, before spreading
    strength_share: float = 0.15  # strength lost per unit of organization lost
    dice: float = 0.1  # +/- relative variation per battle hour
    base_width: int = 4
    flank_width: int = 2  # extra front-line slots per additional attack direction
    stacking_limit: int = 8
    stacking_penalty: float = 0.04  # per division above the limit
    entrench_hours: float = 120.0
    entrench_bonus: float = 0.3
    recovery: float = 0.01  # organization per idle hour at full supply
    moving_recovery: float = 0.25
    retreat_org: float = 0.05
    destroyed_strength: float = 0.05
    supply_drift: float = 1 / 48
    starving_below: float = 0.25
    starving_org: float = 0.004
    starving_strength: float = 0.001


@dataclass
class Division:
    id: int  # unique within its country only
    country: Country
    kind: str
    province: int
    organization: float = 1.0
    strength: float = 1.0
    supply: float = 1.0
    entrenchment: float = 0.0
    target: int | None = None
    support: bool = False
    progress: float = 0.0
    in_combat: bool = False
    alive: bool = True


class Engine:
    """Authoritative hidden state. Only ``SimSession`` turns it into fogged player views."""

    def __init__(self, scenario: Scenario, seed: int = 0, horizon_hours: int = 90 * 24,
                 rules: Rules | None = None) -> None:
        self.scenario, self.rules, self.horizon_hours = scenario, rules or Rules(), horizon_hours
        self.rng = random.Random(f"engine:{scenario.layout.scenario_id}:{seed}")
        layout = scenario.layout
        self.neighbors = {p.id: p.neighbors for p in layout.provinces}
        self.rivers = {p.id: frozenset(p.river_neighbors) for p in layout.provinces}
        self.terrain = {p.id: p.terrain for p in layout.provinces}
        self.victory_points = {p.id: p.victory_points for p in layout.provinces}
        self.capital = {country: layout.capital(country).id for country in Country}
        self.control: dict[int, Country | None] = {
            p.id: Country(p.initial_controller) if p.initial_controller else None for p in layout.provinces}
        self._move = dict(self.rules.move_hours)
        self._penalty = dict(self.rules.attack_penalty)
        self._attack, self._defence = dict(self.rules.attack), dict(self.rules.defence)
        self.home_distance = {country: self._distances(self.capital[country], None) for country in Country}
        self.divisions: list[Division] = []
        counters = dict.fromkeys(Country, 0)
        for item in scenario.deployments:
            country = Country(item.country)
            counters[country] += 1
            self.divisions.append(Division(counters[country], country, item.kind, item.province_id,
                                           item.organization, item.strength))
        self.hour = 0
        self.terminal, self.reason = False, ""
        self.winner: Country | None = None
        self.supply: dict[Country, dict[int, float]] = {}
        self._refresh_supply()
        for division in self.divisions:
            division.supply = self.supply[division.country].get(division.province, 0.0)

    # ----- queries -------------------------------------------------------------------------
    def army(self, country: Country) -> list[Division]:
        return [d for d in self.divisions if d.alive and d.country == country]

    def division(self, country: Country, unit_id: int) -> Division | None:
        return next((d for d in self.divisions if d.alive and d.country == country and d.id == unit_id), None)

    def occupants(self, province: int, country: Country) -> list[Division]:
        return [d for d in self.divisions if d.alive and d.country == country and d.province == province]

    def points(self, country: Country) -> float:
        return sum(value for province, value in self.victory_points.items() if self.control[province] == country)

    def move_hours(self, division: Division, target: int) -> float:
        terrain = self.terrain[target]
        hours = self._move.get(terrain, 36.0)
        if division.kind == "armor" and terrain in ("plains", "desert"):
            hours *= self.rules.armor_open_speed
        return hours + (self.rules.river_move_hours if target in self.rivers[division.province] else 0.0)

    # ----- orders --------------------------------------------------------------------------
    def order(self, country: Country, unit_id: int, target: int, support: bool = False,
              dry_run: bool = False) -> str | None:
        """Give a move/attack or support-attack order. Returns a rejection reason, or None."""
        division = self.division(country, unit_id)
        if division is None:
            return "unknown unit"
        if target not in self.neighbors:
            return "unknown province"
        if target not in self.neighbors[division.province]:
            return "target not adjacent"
        if support and self.control[target] == country:
            return "support attack target is friendly"
        if not dry_run and (division.target != target or division.support != support):
            division.target, division.support, division.progress = target, support, 0.0
            division.entrenchment = 0.0
        return None

    def cancel(self, country: Country, unit_id: int) -> str | None:
        division = self.division(country, unit_id)
        if division is None:
            return "unknown unit"
        division.target, division.support, division.progress = None, False, 0.0
        return None

    # ----- time ----------------------------------------------------------------------------
    def advance(self, hours: int) -> None:
        if hours < 0:
            raise ArenaError("cannot advance negative hours")
        for _ in range(hours):
            if self.terminal:
                return
            self._tick()

    def _tick(self) -> None:
        rules = self.rules
        self.hour += 1
        alive = [d for d in self.divisions if d.alive]
        here: dict[tuple[int, Country], list[Division]] = {}
        for division in alive:
            division.in_combat = False
            here.setdefault((division.province, division.country), []).append(division)
        # 1. Battles: (target province, attacking country) -> attackers
        battles: dict[tuple[int, Country], list[Division]] = {}
        for division in alive:
            if division.target is not None and (division.target, division.country.opponent) in here:
                battles.setdefault((division.target, division.country), []).append(division)
        damage: dict[int, float] = {}
        defenders_hit: set[int] = set()
        for (province, country), attackers in sorted(battles.items(), key=lambda item: (item[0][0], item[0][1].value)):
            defenders = here[province, country.opponent]
            directions = len({d.province for d in attackers})
            width = rules.base_width + rules.flank_width * (directions - 1)
            front_a = sorted(attackers, key=lambda d: (-d.organization, d.id))[:width]
            front_d = sorted(defenders, key=lambda d: (-d.organization, d.id))[:rules.base_width]
            roll = 1.0 + rules.dice * (2 * self.rng.random() - 1)
            power_a = sum(self._attack_power(d, province, len(here[d.province, country])) for d in front_a) * roll
            power_d = sum(self._defence_power(d, len(defenders)) for d in front_d) * (2 - roll)
            for division in front_d:
                damage[id(division)] = damage.get(id(division), 0.0) + rules.org_damage * power_a / len(front_d)
                defenders_hit.add(id(division))
            for division in front_a:
                taken = rules.org_damage * power_d / len(front_a)
                if division.kind == "armor":
                    taken *= rules.armor_breakthrough
                damage[id(division)] = damage.get(id(division), 0.0) + taken
            for division in (*attackers, *defenders):
                division.in_combat = True
        # 2. Apply damage, then break: defenders retreat or die, attackers stop.
        for division in alive:
            if id(division) not in damage:
                continue
            lost = damage[id(division)]
            division.organization = max(0.0, division.organization - lost)
            division.strength = max(0.0, division.strength - lost * rules.strength_share)
            if division.strength <= rules.destroyed_strength:
                division.alive = False
            elif division.organization <= 0.0:
                if id(division) in defenders_hit:
                    self._retreat(division)
                else:
                    division.target, division.support, division.progress = None, False, 0.0
        # 3. Movement. Arrivals into a province the enemy still occupies wait (that is a battle).
        arrivals: list[Division] = []
        for division in alive:
            if not division.alive or division.target is None:
                continue
            enemy_present = bool(self.occupants(division.target, division.country.opponent))
            if division.support:
                if not enemy_present and not division.in_combat:
                    division.target, division.support = None, False
                continue
            division.progress = min(division.progress + 1.0, self.move_hours(division, division.target))
            if not enemy_present and division.progress >= self.move_hours(division, division.target):
                arrivals.append(division)
        if len({d.country for d in arrivals}) > 1:
            self.rng.shuffle(arrivals)  # no fixed side wins a simultaneous race for a province
        changed = False
        for division in arrivals:
            assert division.target is not None
            if self.occupants(division.target, division.country.opponent):
                continue  # the other side got there first this hour; next hour this is a battle
            division.province, division.target, division.progress = division.target, None, 0.0
            division.entrenchment = 0.0
            if self.control[division.province] != division.country:
                self.control[division.province] = division.country
                changed = True
        if changed:
            self._refresh_supply()
        # 4. Supply, recovery, entrenchment.
        for division in alive:
            if not division.alive:
                continue
            local = self.supply[division.country].get(division.province, 0.0)
            division.supply += max(-rules.supply_drift, min(rules.supply_drift, local - division.supply))
            if division.supply < rules.starving_below:
                division.organization = max(0.0, division.organization - rules.starving_org)
                division.strength -= rules.starving_strength
                if division.strength <= rules.destroyed_strength:
                    division.alive = False
                    continue
            if not division.in_combat:
                rate = rules.recovery * division.supply * (rules.moving_recovery if division.target else 1.0)
                division.organization = min(1.0, division.organization + rate)
                if division.target is None:
                    division.entrenchment = min(1.0, division.entrenchment + 1.0 / rules.entrench_hours)
        self._check_victory()

    def _stacking(self, count: int) -> float:
        return max(0.2, 1.0 - self.rules.stacking_penalty * max(0, count - self.rules.stacking_limit))

    def _attack_power(self, division: Division, target: int, stacked: int) -> float:
        penalty = self._penalty.get(self.terrain[target], 0.0)
        if target in self.rivers[division.province]:
            penalty += self.rules.river_penalty
        if division.kind == "armor":
            penalty *= self.rules.armor_terrain_factor
        return (self._attack[division.kind] * division.strength * max(0.1, 1.0 - penalty)
                * (0.5 + 0.5 * division.supply) * self._stacking(stacked))

    def _defence_power(self, division: Division, stacked: int) -> float:
        return (self._defence[division.kind] * division.strength
                * (1.0 + self.rules.entrench_bonus * division.entrenchment)
                * (0.5 + 0.5 * division.supply) * self._stacking(stacked))

    def _retreat(self, division: Division) -> None:
        options = [p for p in self.neighbors[division.province] if self.control[p] == division.country
                   and not self.occupants(p, division.country.opponent)]
        if not options:
            division.alive = False  # encircled: nowhere to retreat
            return
        distances = self.home_distance[division.country]
        division.province = min(options, key=lambda p: (distances[p], p))
        division.organization = self.rules.retreat_org
        division.target, division.support, division.progress, division.entrenchment = None, False, 0.0, 0.0

    def _distances(self, start: int, country: Country | None) -> dict[int, int]:
        """Hop distances from start; restricted to provinces a country controls when given."""
        if country is not None and self.control[start] != country:
            return {}
        seen, queue = {start: 0}, deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.neighbors[current]:
                if neighbor not in seen and (country is None or self.control[neighbor] == country):
                    seen[neighbor] = seen[current] + 1
                    queue.append(neighbor)
        return seen

    def _refresh_supply(self) -> None:
        reach, falloff = self.scenario.supply_range, self.scenario.supply_falloff
        for country in Country:
            distances = self._distances(self.capital[country], country)
            self.supply[country] = {province: max(0.0, min(1.0, 1.0 - max(0, hops - reach) / falloff))
                                    for province, hops in distances.items()}

    def _check_victory(self) -> None:
        captured = [c for c in Country if self.control[self.capital[c.opponent]] == c]
        standing = [c for c in Country if self.army(c)]
        if captured:
            self._finish(captured[0] if len(captured) == 1 else None, "capital captured")
        elif len(standing) < 2:
            self._finish(standing[0] if standing else None, "army eliminated")
        elif self.hour >= self.horizon_hours:
            blue, red = self.points(Country.BLUE), self.points(Country.RED)
            self._finish(Country.BLUE if blue > red else Country.RED if red > blue else None, "victory points")

    def _finish(self, winner: Country | None, reason: str) -> None:
        self.terminal, self.winner, self.reason = True, winner, reason

    def truncate(self, reason: str = "decision cap") -> None:
        """End the episode as a draw (for example a pause loop that hit the decision cap)."""
        if not self.terminal:
            self._finish(None, reason)
