"""Goal selection: strict-order, precondition-gated.

Returns the first not-yet-completed goal *iff* its precondition is satisfied by the
current world. If that goal's precondition is unsatisfied or uncertain, returns
None — the controller then advances time and re-checks, which is exactly the
event-driven "wake when a slot frees" behaviour (the gate is the precondition).
"""

from __future__ import annotations

from ..schemas import Goal, PlaybookState, WorldState


def next_pending_goal(
    goals: list[Goal], state: PlaybookState, world: WorldState
) -> Goal | None:
    for g in goals:
        if not g.repeatable and g.id in state.completed_goal_ids:
            continue
        return g if g.precondition.satisfied(world) is True else None
    return None


def all_done(goals: list[Goal], state: PlaybookState) -> bool:
    return all(g.repeatable or g.id in state.completed_goal_ids for g in goals)
