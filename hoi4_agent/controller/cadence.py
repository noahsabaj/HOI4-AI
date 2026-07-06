"""Observed-not-assumed pause + date-driven time advance.

Pause is toggled only when the observed state differs from the desired one (the
tool ``ensure_paused`` enforces this). Time advance unpauses, sets the run speed,
and polls the *screen-read* in-game date until it reaches the target — never a
blind fixed sleep that can drift. Every poll reads only the date (one field, one
deterministic read on the glyph path), not the full fact set.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

from ..enums import ToolName
from ..errors import ControllerError
from ..schemas import GameDate, Intent, ToolResult
from ..tools.executor import execute

if TYPE_CHECKING:
    from ..context import AgentContext

POLL_WAIT_S = 2.0       # real seconds between date reads while time advances
BLIND_POLLS = 3         # with no target, advance only this many polls then re-pause


def compute_advance_target(
    date: "GameDate | None",
    pending_etas: tuple[tuple[str, str], ...],
    cycle_days: int,
    max_advance_days: int,
    streak: int,
) -> tuple["GameDate | None", str]:
    """Pick where to fast-forward to when nothing is actionable.

    Baseline: exponential backoff from ``cycle_days`` up to ``max_advance_days``
    (a quiet world costs ever fewer polls). A future ETA (predicted research
    completion) may only SHORTEN the hop, never extend it — other wake-ups
    (freed civilian capacity, events) are not ETA-modeled, so sleeping past the
    backoff target on a prediction would be trusting a hint over perception.
    Returns (target, reason); target None means the date is unreadable.
    """
    if date is None:
        return None, "blind (date unreadable)"
    days = min(cycle_days * (2 ** max(streak, 0)), max_advance_days)
    backoff_target = date.plus_days(days)
    future = sorted(
        eta for _key, s in pending_etas if (eta := GameDate.from_str(s)) > date
    )
    if future and future[0] < backoff_target:
        return future[0], f"eta {future[0].to_str()}"
    return backoff_target, f"backoff {days}d"


def ensure_paused(ctx: "AgentContext", desired: bool) -> ToolResult:
    return execute(Intent(ToolName.ENSURE_PAUSED, paused=desired), ctx)


def run_to_date(
    ctx: "AgentContext",
    target: GameDate | None,
    *,
    max_polls: int = 240,
    wait_s: float = POLL_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> GameDate | None:
    """Advance in-game time until the read date >= target, then re-pause.

    If ``target`` is None (date unreadable at the call site), advance exactly
    ``BLIND_POLLS`` polls so *some* time passes, then re-pause — regardless of
    whether the date becomes readable mid-advance. Returns the last date read.

    Raises ControllerError if the game cannot be re-paused afterward (the loop's
    failure path recovers/halts rather than acting with time running).
    """
    unpaused = execute(Intent(ToolName.ENSURE_PAUSED, paused=False), ctx)
    if not unpaused.ok:
        # Could not unpause: no time can pass, so don't poll for it. The game is
        # still paused, which is the safe state — report nothing advanced.
        return None
    execute(Intent(ToolName.SET_SPEED, speed=ctx.config.timing.run_speed), ctx)  # best-effort
    last: GameDate | None = None
    polls = 0
    while polls < max_polls:
        world = ctx.perceive(fields={"date"})
        if world.date is not None:
            last = world.date
            if target is not None and world.date >= target:
                break
        if target is None and polls >= BLIND_POLLS:
            break
        polls += 1
        sleep(wait_s)
    repaused = execute(Intent(ToolName.ENSURE_PAUSED, paused=True), ctx)
    if not repaused.ok:
        raise ControllerError(f"failed to re-pause after time advance: {repaused.assertion}")
    return last
