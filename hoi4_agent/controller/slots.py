"""Cheap, deterministic capacity signals for the mostly-asleep poll."""

from __future__ import annotations

from ..schemas import WorldState


def has_capacity(world: WorldState) -> bool:
    """True if there's a free build or research slot to act on right now."""
    return (world.free_civ_slots or 0) >= 1 or (world.idle_research_slots or 0) >= 1


def slot_freed(prev: WorldState, world: WorldState) -> bool:
    """True if a build or research slot count increased since the previous read."""
    return (world.free_civ_slots or 0) > (prev.free_civ_slots or 0) or (
        world.idle_research_slots or 0
    ) > (prev.idle_research_slots or 0)
