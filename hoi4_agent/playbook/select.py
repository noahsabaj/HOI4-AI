"""Goal selection: strict-order for one-shots, step-past for repeatables.

One-shot goals gate strictly: the first pending one whose precondition is False
(or uncertain and not handler-enforceable) blocks everything after it — that
ordering is the strategy. Two refinements:

- Slot preconditions are only readable while their panel is open, so an
  *uncertain* slot precondition returns the goal anyway and the self-contained
  tool decides (``NotReadyError`` if the slot isn't actually free).
- A **repeatable** goal never blocks: when it isn't ready it is stepped past so
  later goals still run, and it is re-offered every cycle (e.g. "refill any idle
  research slot" fires whenever a tech completes). The controller passes ``skip``
  to step past repeatables that already reported NotReady this cycle.

``expired_goal`` finds a one-shot whose DATE_BEFORE window has definitively
passed — the date only moves forward, so waiting can never satisfy it; the
controller skips it instead of advancing time forever.
"""

from __future__ import annotations

from collections.abc import Set

from ..enums import PreconditionKind
from ..schemas import Goal, PlaybookState, WorldState


def next_pending_goal(
    goals: list[Goal],
    state: PlaybookState,
    world: WorldState,
    skip: Set[str] = frozenset(),
) -> Goal | None:
    for g in goals:
        if not g.repeatable and g.id in state.completed_goal_ids:
            continue
        if g.id in skip:  # reported NotReady earlier this cycle
            continue
        sat = g.precondition.satisfied(world)
        if sat is True:
            return g
        if sat is None and g.precondition.handler_enforced:
            return g  # attempt it: the tool re-checks with the panel open
        if g.repeatable:
            continue  # a not-ready repeatable never blocks its successors
        return None
    return None


def expired_goal(
    goals: list[Goal], state: PlaybookState, world: WorldState
) -> Goal | None:
    """First pending one-shot whose ``date_before`` gate is permanently missed."""
    if world.date is None:
        return None
    for g in goals:
        if g.repeatable or g.id in state.completed_goal_ids:
            continue
        pre = g.precondition
        if pre.kind is PreconditionKind.DATE_BEFORE and pre.date is not None and world.date >= pre.date:
            return g
        return None  # strict order: only the head of the queue can expire
    return None


def all_done(goals: list[Goal], state: PlaybookState) -> bool:
    return all(g.repeatable or g.id in state.completed_goal_ids for g in goals)
