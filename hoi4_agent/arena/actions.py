"""Player-view-only structural masks. Engine validation remains authoritative."""
from __future__ import annotations

from dataclasses import dataclass

from .contracts import ArenaError, Order, PlayerObservation, Verb


@dataclass(frozen=True)
class Choice:
    verb: Verb
    unit_id: int | None = None
    target_id: int | None = None
    speed: int | None = None

    def order(self, observation: PlayerObservation, order_id: str) -> Order:
        return Order(order_id, observation.episode_id, observation.sequence, observation.country,
                     self.verb, () if self.unit_id is None else (self.unit_id,), self.target_id, self.speed)


def choices(observation: PlayerObservation) -> tuple[Choice, ...]:
    """Mask ownership/topology, never query hidden engine validity.

    Move (which is also how HOI4 attacks) and support attack remain candidates
    on adjacent terrain even when ownership is unknown. A hidden defender must
    not change this mask. The engine can reject a normal order; that rejection
    is recorded as an interaction.
    """
    result = [Choice(Verb.NOOP)]
    if observation.terminal:
        return tuple(result)
    result.append(Choice(Verb.PAUSE))
    result.extend(Choice(Verb.SET_SPEED, speed=speed) for speed in range(1, 6))
    provinces = {province.id: province for province in observation.provinces}
    for unit in sorted(observation.units, key=lambda unit: unit.id):
        if unit.country != observation.country:
            continue
        result.append(Choice(Verb.CANCEL, unit.id))
        for destination in sorted(provinces[unit.province_id].neighbors):
            result.extend((Choice(Verb.MOVE, unit.id, destination),
                           Choice(Verb.SUPPORT_ATTACK, unit.id, destination)))
    return tuple(result)


def choice_index(observation: PlayerObservation, order: Order) -> int:
    if order.country != observation.country or order.episode_id != observation.episode_id:
        raise ArenaError("demonstration order crosses player/episode boundaries")
    if order.observation_sequence != observation.sequence:
        raise ArenaError("demonstration order is not aligned with its observation")
    if len(order.unit_ids) > 1:
        raise ArenaError("initial policy supports one division per decision; group demonstration is unsupported")
    candidate = Choice(order.verb, order.unit_ids[0] if order.unit_ids else None, order.target_province_id,
                       order.speed)
    try:
        return choices(observation).index(candidate)
    except ValueError as exc:
        raise ArenaError("order is outside this observation's action vocabulary") from exc
