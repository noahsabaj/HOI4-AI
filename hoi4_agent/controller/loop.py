"""The closed control loop.

Each cycle: ensure paused -> perceive -> pick the next actionable goal. If one is
actionable, act-with-retry and verify; on OK mark progress, on failure recover or
halt-and-flag (bounded — never a silent loop). If nothing is actionable, advance
in-game time (date-driven) and re-check — that wait is the event-driven "wake when
a slot frees" behaviour. The VLM is touched only inside tools (T1 reads / T2
judgment); sequential actionable goals run back-to-back while paused.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

from ..enums import GermanState, Tech, ToolName, Verdict
from ..errors import HaltAndFlag
from ..playbook.select import all_done, next_pending_goal
from ..playbook.state import save_state
from ..schemas import Goal, Intent, PlaybookState, WorldState
from ..trace.record import build_record
from . import cadence, recovery

if TYPE_CHECKING:
    from ..context import AgentContext
    from ..trace.writer import JsonlTraceWriter


def _resolve_intent(ctx: "AgentContext", goal: Goal, world: WorldState) -> tuple[Intent, bool]:
    """Fill a model-chosen arg if the goal needs judgment. Returns (intent, vlm_used)."""
    intent = goal.to_intent()
    if not goal.needs_judgment:
        return intent, False
    crop = ctx.capture.grab(ctx.geometry)
    if goal.tool is ToolName.BUILD_IN_STATE and intent.state is None:
        options = [s for s in GermanState if s.value in ctx.calibration.state_points]
        return replace(intent, state=ctx.brain.which_state(crop, options)), True
    if goal.tool is ToolName.ASSIGN_RESEARCH and intent.tech is None:
        options = [t for t in Tech if t.value in ctx.calibration.tech_points]
        return replace(intent, tech=ctx.brain.which_tech(crop, options)), True
    return intent, False


def run(
    ctx: "AgentContext",
    goals: list[Goal],
    state: PlaybookState,
    *,
    writer: "JsonlTraceWriter | None" = None,
    state_path: str | None = None,
    max_cycles: int = 1000,
    max_failures: int = 5,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> PlaybookState:
    """Run the loop until the playbook is done (or a safety cap / halt fires)."""
    ctx.geometry.assert_resolution(ctx.config.display.width, ctx.config.display.height)
    failures = 0
    cycles = 0

    while not all_done(goals, state) and cycles < max_cycles:
        cycles += 1
        cadence.ensure_paused(ctx, True)
        world = ctx.perceive()
        goal = next_pending_goal(goals, state, world)

        if goal is None:
            # Nothing actionable now -> let in-game time pass, then re-check.
            target = world.date.plus_days(ctx.config.timing.cycle_days) if world.date else None
            cadence.run_to_date(ctx, target, sleep=sleep)
            state = state.with_date(world.date).advance_cycle()
            if state_path:
                save_state(state_path, state)
            continue

        intent, vlm_used = _resolve_intent(ctx, goal, world)
        result = recovery.act_with_retry(ctx, intent, ctx.config.timing.max_retries)

        if writer is not None:
            writer.append(
                build_record(
                    cycle=state.cycle_count,
                    ts=clock(),
                    result=result,
                    goal=goal,
                    mode=ctx.mode.value,
                    vlm_used=vlm_used,
                )
            )

        if result.verdict is Verdict.OK:
            failures = 0
            state = state.with_completed(goal.id)
        else:
            failures += 1
            recovered = recovery.recover(ctx)
            if not recovered or failures >= max_failures:
                raise HaltAndFlag(
                    f"halting after {failures} consecutive failures at goal "
                    f"{goal.id!r}: {result.assertion}",
                    trace_ref=str(writer.path) if writer else None,
                )

        state = state.with_date(world.date).advance_cycle()
        if state_path:
            save_state(state_path, state)

    if cycles >= max_cycles and not all_done(goals, state):
        raise HaltAndFlag(
            f"hit max_cycles={max_cycles} before completing playbook",
            trace_ref=str(writer.path) if writer else None,
        )
    return state
