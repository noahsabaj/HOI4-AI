"""The closed control loop.

Each cycle: ensure paused -> perceive -> pick the next actionable goal. If one is
actionable, act-with-retry and verify; on OK mark progress, on NOT-READY advance
in-game time (a wait-state, not a failure), on failure recover or halt-and-flag
(bounded — never a silent loop). The whole cycle body runs under an ``AgentError``
catch, so an infra failure (model endpoint down mid-perceive) becomes a counted,
recovered, traced failure instead of a raw crash — ``HaltAndFlag`` is the only
way out.

Observability: every cycle appends a TraceRecord — actions with pre/post frames,
the resolved intent, the judgment prompt/raw output when the VLM was consulted,
and the exact input events injected; time-advance and error cycles get their own
record kinds. That is what makes the JSONL trace actually replayable.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

from ..enums import GermanState, Tech, ToolName, Verdict
from ..errors import AgentError, ControllerError, HaltAndFlag, NotReadyError
from ..playbook.select import all_done, expired_goal, next_pending_goal
from ..playbook.state import save_state
from ..schemas import GameDate, Goal, Intent, PlaybookState, ToolResult, WorldState
from ..trace.record import build_record
from . import cadence, recovery

if TYPE_CHECKING:
    from ..context import AgentContext
    from ..trace.writer import JsonlTraceWriter


def _resolve_intent(
    ctx: "AgentContext", goal: Goal, world: WorldState
) -> tuple[Intent, dict | None]:
    """Fill a model-chosen arg if the goal needs judgment.

    Returns (intent, judgment) where judgment is None when no model was consulted,
    else {"prompt", "raw"} CAPTURED AT THE CALL. Reading ``brain.last_prompt``
    later does not work: the Brain keeps only its most recent exchange and is
    also the terminal tier of the perception ChainReader, so the tool's own
    reads overwrite the judgment before the record is built — the trace then
    attributes a slot-counting prompt to a tech decision.
    """
    intent = goal.to_intent()
    if not goal.needs_judgment:
        return intent, None
    if ctx.brain is None:
        raise ControllerError(f"goal {goal.id!r} needs judgment but no brain is wired")
    crop = ctx.capture.grab(ctx.geometry)
    resolved = intent
    if goal.tool is ToolName.BUILD_IN_STATE and intent.state is None:
        state_opts = [s for s in GermanState if s.value in ctx.calibration.state_points]
        resolved = replace(intent, state=ctx.brain.which_state(crop, state_opts, goal.building))
    elif goal.tool is ToolName.ASSIGN_RESEARCH and intent.tech is None:
        tech_opts = [t for t in Tech if t.value in ctx.calibration.tech_points]
        resolved = replace(intent, tech=ctx.brain.which_tech(crop, tech_opts))
    else:
        return intent, None
    return resolved, {
        "prompt": getattr(ctx.brain, "last_prompt", None),
        "raw": getattr(ctx.brain, "last_raw", None),
    }


def run(
    ctx: "AgentContext",
    goals: list[Goal],
    state: PlaybookState,
    *,
    writer: "JsonlTraceWriter | None" = None,
    state_path: str | None = None,
    max_cycles: int = 1000,
    max_failures: int = 5,
    research_days: dict[Tech, int] | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> PlaybookState:
    """Run the loop until the playbook is done (or a safety cap / halt fires)."""
    ctx.geometry.assert_resolution(ctx.config.display.width, ctx.config.display.height)
    failures = 0
    cycles = 0
    streak = 0  # consecutive time-advances with no successful action (backoff input)

    def _drain_actions() -> tuple[dict, ...]:
        drain = getattr(ctx.input, "drain", None)
        return tuple(drain()) if callable(drain) else ()

    def _save_frame(name: str) -> str | None:
        if writer is None:
            return None
        return writer.save_frame(ctx.capture.grab(ctx.geometry), name)

    def _append(record) -> None:
        if writer is not None:
            writer.append(record)

    def _trace_ref() -> str | None:
        return str(writer.path) if writer else None

    def _model_calls() -> int:
        """Model round trips so far — judgment AND perception-chain reads."""
        return int(getattr(ctx.brain, "call_count", 0) or 0)

    def _advance_time(world: WorldState) -> GameDate | None:
        nonlocal streak
        target, reason = cadence.compute_advance_target(
            world.date, state.pending_etas, ctx.config.timing.cycle_days,
            ctx.config.timing.max_advance_days, streak,
        )
        streak += 1
        before = _model_calls()
        last = cadence.run_to_date(ctx, target, sleep=sleep)
        res = ToolResult(
            ToolName.OBSERVE, Verdict.OK, pre=world,
            assertion=(
                f"advanced time toward {target.to_str() if target else 'blind polls'}"
                f" [{reason}] (last read {last.to_str() if last else 'none'})"
            ),
        )
        _append(build_record(
            cycle=state.cycle_count, ts=clock(), result=res, mode=ctx.mode.value,
            kind="advance", actions=_drain_actions(),
            # An advance polls the date repeatedly; with no glyph/OCR tier every
            # one of those polls is a model call, which a bare vlm_used=False hides.
            vlm_calls=_model_calls() - before,
        ))
        return last

    def _recover_and_trace(world: WorldState | None) -> bool:
        """Deterministic recovery, escalating once to a traced VLM consult."""
        before = _model_calls()
        recovered, consult = recovery.recover_with_consult(ctx)
        if consult is not None:
            res = ToolResult(
                ToolName.OBSERVE, Verdict.OK if recovered else Verdict.FAILED,
                pre=world, assertion=f"recovery consult -> {consult['hint']}",
            )
            _append(build_record(
                cycle=state.cycle_count, ts=clock(), result=res, mode=ctx.mode.value,
                kind="consult", vlm_used=True, prompt=consult["prompt"],
                raw=consult["raw"], actions=_drain_actions(),
                vlm_calls=_model_calls() - before,
            ))
        return recovered

    _drain_actions()  # discard any pre-loop input noise

    while not all_done(goals, state) and cycles < max_cycles:
        cycles += 1
        world: WorldState | None = None
        goal: Goal | None = None
        try:
            pause_res = cadence.ensure_paused(ctx, True)
            if not pause_res.ok:
                raise ControllerError(f"cannot pause the game: {pause_res.assertion}")
            world = ctx.perceive()
            state = state.drop_etas_through(world.date)  # arrived wake-hints are spent

            # Inner selection round: a repeatable goal that reports NotReady is
            # stepped past (cycle_skip) and the NEXT goal tried in the same
            # cycle — "refill research when possible" must never starve builds.
            cycle_skip: set[str] = set()
            # Frame names must key on (cycle, step), not cycle alone: several
            # goals can act in ONE cycle, and a cycle-only name made the second
            # goal's frames overwrite the first's — leaving the first record
            # pointing at pixels from a different action.
            cycle_step = 0
            while True:
                goal = next_pending_goal(goals, state, world, skip=cycle_skip)

                if goal is None:
                    expired = expired_goal(goals, state, world)
                    if expired is not None:
                        # A missed date_before window can never re-open; waiting
                        # on it would advance time forever. Skip it, loudly.
                        state = state.with_completed(expired.id)
                        gate = expired.precondition.date
                        res = ToolResult(
                            expired.tool, Verdict.FAILED, pre=world,
                            assertion=f"date_before {gate.to_str() if gate else '?'} window "
                                      f"expired at {world.date.to_str() if world.date else '?'} — skipped",
                        )
                        _append(build_record(
                            cycle=state.cycle_count, ts=clock(), result=res, goal=expired,
                            mode=ctx.mode.value, kind="skipped", actions=_drain_actions(),
                        ))
                        break
                    # Nothing actionable now -> let in-game time pass, re-check next cycle.
                    last = _advance_time(world)
                    if last is not None:  # the tail records world.date — keep it fresh
                        world = replace(world, date=last)
                    break

                stem = f"c{state.cycle_count:05d}_s{cycle_step}"
                cycle_step += 1
                pre_path = _save_frame(f"{stem}_pre.jpg")
                before = _model_calls()
                intent, judgment = _resolve_intent(ctx, goal, world)
                result = recovery.act_with_retry(ctx, intent, ctx.config.timing.max_retries)
                post_path = _save_frame(f"{stem}_post.jpg")

                _append(build_record(
                    cycle=state.cycle_count, ts=clock(), result=result, goal=goal,
                    intent=intent, mode=ctx.mode.value,
                    vlm_used=judgment is not None,
                    vlm_calls=_model_calls() - before,
                    pre_path=pre_path, post_path=post_path,
                    prompt=judgment["prompt"] if judgment else None,
                    raw=judgment["raw"] if judgment else None,
                    actions=_drain_actions(),
                ))

                if result.verdict is Verdict.OK:
                    failures = 0
                    streak = 0  # the world moved; backoff starts over
                    if (
                        goal.tool is ToolName.ASSIGN_RESEARCH
                        and research_days
                        and intent.tech is not None
                        and world.date is not None
                        and intent.tech in research_days
                    ):
                        eta = world.date.plus_days(research_days[intent.tech])
                        state = state.with_eta(intent.tech.value, eta)
                    if not goal.repeatable:  # repeatables are re-offered forever
                        state = state.with_completed(goal.id)
                    break
                if isinstance(result.error, NotReadyError):
                    if goal.repeatable:
                        cycle_skip.add(goal.id)
                        continue  # same cycle: give the next goal its chance
                    # A one-shot gates strictly: wait for the game.
                    _advance_time(world)
                    break
                if result.verdict is Verdict.UNCERTAIN and not result.retry_safe:
                    # The handler clicked and then could not read the effect, so
                    # it is unknown whether the mutation landed. act_with_retry
                    # honors retry_safe within the cycle, but the goal is not
                    # marked complete, so falling through to the failure path
                    # would re-offer it next cycle and repeat a mutation that may
                    # already have happened — the one silent-wrong path in a
                    # system built on loud-stop. Both alternatives (re-run, skip)
                    # are guesses about game state, so stop and hand it over.
                    raise HaltAndFlag(
                        f"unverifiable mutation at goal {goal.id!r}: {result.assertion} "
                        "— cannot tell whether the action landed, so neither retrying "
                        "nor skipping is safe; inspect the post frame and resume with "
                        "an amended plan state",
                        trace_ref=_trace_ref(),
                    )
                failures += 1
                recovered = _recover_and_trace(world)
                if not recovered or failures >= max_failures:
                    raise HaltAndFlag(
                        f"halting after {failures} consecutive failures at goal "
                        f"{goal.id!r}: {result.assertion}",
                        trace_ref=_trace_ref(),
                    )
                break
        except HaltAndFlag:
            raise
        except AgentError as e:
            # Infra/controller error mid-cycle: counted, traced, recovered — loud
            # but bounded. This is the catch the error hierarchy promises.
            failures += 1
            err_result = ToolResult(
                goal.tool if goal else ToolName.OBSERVE, Verdict.FAILED,
                pre=world, assertion=str(e), error=e,
            )
            _append(build_record(
                cycle=state.cycle_count, ts=clock(), result=err_result, goal=goal,
                mode=ctx.mode.value, kind="error", actions=_drain_actions(),
            ))
            recovered = _recover_and_trace(world)
            if not recovered or failures >= max_failures:
                raise HaltAndFlag(
                    f"halting after {failures} consecutive failures: {e}",
                    trace_ref=_trace_ref(),
                ) from e

        state = state.with_date(world.date if world else None).advance_cycle()
        if state_path:
            save_state(state_path, state)

    if cycles >= max_cycles and not all_done(goals, state):
        raise HaltAndFlag(
            f"hit max_cycles={max_cycles} before completing playbook",
            trace_ref=_trace_ref(),
        )
    return state
