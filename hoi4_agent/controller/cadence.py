"""Observed-not-assumed pause + date-driven time advance.

Pause is toggled only when the observed state differs from the desired one (the
tool ``ensure_paused`` enforces this). Time advance unpauses, sets the run speed,
and polls the *screen-read* in-game date until it reaches the target — never a
blind fixed sleep that can drift.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

from ..enums import ToolName
from ..schemas import GameDate, Intent, ToolResult
from ..tools.executor import execute

if TYPE_CHECKING:
    from ..context import AgentContext

POLL_WAIT_S = 2.0       # real seconds between date reads while time advances
BLIND_POLLS = 3         # if the date is unreadable, advance this many polls then re-pause


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

    If ``target`` is None (date currently unreadable), advance a few blind polls so
    *some* time passes, then re-pause. Returns the last date read (or None).
    """
    execute(Intent(ToolName.ENSURE_PAUSED, paused=False), ctx)
    execute(Intent(ToolName.SET_SPEED, speed=ctx.config.timing.run_speed), ctx)
    last: GameDate | None = None
    polls = 0
    while polls < max_polls:
        world = ctx.perceive()
        if world.date is not None:
            last = world.date
            if target is not None and world.date >= target:
                break
        elif target is None and polls >= BLIND_POLLS:
            break
        polls += 1
        sleep(wait_s)
    execute(Intent(ToolName.ENSURE_PAUSED, paused=True), ctx)
    return last
