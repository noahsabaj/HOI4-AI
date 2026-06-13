"""Tool handlers + ``execute``. Each handler returns a ToolResult (never None).

The contract: precondition (checked against a fresh perceive) -> deterministic
steps (hotkeys + the few calibrated clicks) -> post-condition assertion via
deterministic perception. ``build_in_state`` / ``assign_research`` are the
load-bearing ones: they assert the construction queue grew / idle slots dropped,
proving the click had effect — not by re-asking the model.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

from ..enums import PanelId, ToolName, Verdict
from ..errors import (
    AgentError,
    AssignResearchError,
    BuildInStateError,
    IntentValidationError,
    PanelOpenError,
    PauseToggleError,
    PreconditionError,
    ResetFailedError,
    SpeedSetError,
    TemplateMissingError,
)
from ..schemas import Intent, ToolResult, validate_intent
from . import macros

if TYPE_CHECKING:
    from ..context import AgentContext
    from ..schemas import WorldState

MAX_SPEED_STEPS = 6


def _ok(tool, pre, post, assertion) -> ToolResult:
    return ToolResult(tool, Verdict.OK, pre, post, assertion)


def _uncertain(tool, pre, post, assertion) -> ToolResult:
    return ToolResult(tool, Verdict.UNCERTAIN, pre, post, assertion)


def _failed(tool, pre, post, err: Exception) -> ToolResult:
    return ToolResult(tool, Verdict.FAILED, pre, post, str(err), err)


# --- handlers ---------------------------------------------------------------
def _observe(intent: Intent, ctx: "AgentContext") -> ToolResult:
    w = ctx.perceive()
    return _ok(ToolName.OBSERVE, w, w, "observed world state")


def _ensure_paused(intent: Intent, ctx: "AgentContext") -> ToolResult:
    ctx.input.focus(ctx.geometry)
    desired = bool(intent.paused)
    pre = ctx.perceive(read_numbers=False)
    if pre.paused is None:
        return _uncertain(ToolName.ENSURE_PAUSED, pre, pre, "pause state unreadable")
    if pre.paused == desired:
        return _ok(ToolName.ENSURE_PAUSED, pre, pre, f"already paused={desired}")
    ctx.input.key(macros.HOTKEYS["pause"])
    post = ctx.perceive(read_numbers=False)
    if post.paused == desired:
        return _ok(ToolName.ENSURE_PAUSED, pre, post, f"toggled to paused={desired}")
    return _failed(ToolName.ENSURE_PAUSED, pre, post, PauseToggleError(post.paused, desired))


def _set_speed(intent: Intent, ctx: "AgentContext") -> ToolResult:
    ctx.input.focus(ctx.geometry)
    target = intent.speed
    pre = ctx.perceive()
    cur = pre.speed
    if cur is None:
        return _uncertain(ToolName.SET_SPEED, pre, pre, "speed unreadable")
    post = pre
    steps = 0
    while cur != target and steps < MAX_SPEED_STEPS:
        key = macros.HOTKEYS["speed_up"] if cur < target else macros.HOTKEYS["speed_down"]
        ctx.input.key(key)
        post = ctx.perceive()
        cur = post.speed
        steps += 1
        if cur is None:
            return _uncertain(ToolName.SET_SPEED, pre, post, "speed unreadable mid-adjust")
    if post.speed == target:
        return _ok(ToolName.SET_SPEED, pre, post, f"speed -> {target}")
    return _failed(ToolName.SET_SPEED, pre, post, SpeedSetError(f"speed {post.speed} != {target}", observed=post.speed))


def _open(intent, ctx, target_panel, hotkey, tool) -> ToolResult:
    ctx.input.focus(ctx.geometry)
    pre = ctx.perceive(read_numbers=False)
    if pre.open_panel is target_panel:
        return _ok(tool, pre, pre, f"{target_panel.value} already open")
    ctx.input.key(hotkey)
    post = ctx.perceive(read_numbers=False)
    if post.open_panel is target_panel:
        return _ok(tool, pre, post, f"opened {target_panel.value}")
    conf = post.confidence.get("panel", 0.0)
    if conf < ctx.config.timing.ncc_threshold:
        return _uncertain(tool, pre, post, f"panel uncertain (conf {conf:.2f})")
    return _failed(tool, pre, post, PanelOpenError(target_panel.value, post.open_panel.value))


def _open_construction(intent: Intent, ctx: "AgentContext") -> ToolResult:
    return _open(intent, ctx, PanelId.CONSTRUCTION, macros.HOTKEYS["construction"], ToolName.OPEN_CONSTRUCTION)


def _open_research(intent: Intent, ctx: "AgentContext") -> ToolResult:
    return _open(intent, ctx, PanelId.RESEARCH, macros.HOTKEYS["research"], ToolName.OPEN_RESEARCH)


def _ensure_panel(ctx, target_panel, hotkey):
    """Make ``target_panel`` the open panel via its hotkey. Returns (ok, world)."""
    world = ctx.perceive(read_numbers=False)
    if world.open_panel is target_panel:
        return True, world
    ctx.input.key(hotkey)
    world = ctx.perceive(read_numbers=False)
    return (world.open_panel is target_panel), world


def _select_building(intent: Intent, ctx: "AgentContext") -> ToolResult:
    ctx.input.focus(ctx.geometry)
    pre = ctx.perceive(read_numbers=False)
    if pre.open_panel is not PanelId.CONSTRUCTION:
        return _failed(
            ToolName.SELECT_BUILDING, pre, pre,
            PreconditionError("construction panel not open", observed=pre.open_panel.value),
        )
    try:
        nx, ny = ctx.calibration.building_point(intent.building)
    except TemplateMissingError as e:
        return _failed(ToolName.SELECT_BUILDING, pre, pre, e)
    ctx.input.click(ctx.geometry, ctx.geometry.full_crop(), nx, ny)
    post = ctx.perceive(read_numbers=False)
    # M1: selection isn't independently verified; build_in_state's queue-growth is.
    return _ok(ToolName.SELECT_BUILDING, pre, post, f"clicked {intent.building.value} (verified downstream)")


def _build_in_state(intent: Intent, ctx: "AgentContext") -> ToolResult:
    """Self-contained: ensure construction panel open + building selected, then queue.

    Verified by the construction queue growing by exactly 1 (deterministic), not by
    re-asking the model. Being self-contained removes ordered-setup-goal coupling.
    """
    ctx.input.focus(ctx.geometry)
    ok, w = _ensure_panel(ctx, PanelId.CONSTRUCTION, macros.HOTKEYS["construction"])
    if not ok:
        if w.confidence.get("panel", 0.0) < ctx.config.timing.ncc_threshold:
            return _uncertain(ToolName.BUILD_IN_STATE, w, w, "construction panel uncertain")
        return _failed(ToolName.BUILD_IN_STATE, w, w, PanelOpenError("construction", w.open_panel.value))
    if intent.building is not None:
        try:
            bx, by = ctx.calibration.building_point(intent.building)
        except TemplateMissingError as e:
            return _failed(ToolName.BUILD_IN_STATE, w, w, e)
        ctx.input.click(ctx.geometry, ctx.geometry.full_crop(), bx, by)
    try:
        nx, ny = ctx.calibration.state_point(intent.state)
    except TemplateMissingError as e:
        return _failed(ToolName.BUILD_IN_STATE, w, w, e)
    pre = ctx.perceive()
    pre_q = pre.construction_queue_len
    ctx.input.click(ctx.geometry, ctx.geometry.full_crop(), nx, ny)
    post = ctx.perceive()
    post_q = post.construction_queue_len
    if pre_q is None or post_q is None:
        return _uncertain(ToolName.BUILD_IN_STATE, pre, post, "queue length unreadable")
    if post_q == pre_q + 1:
        return _ok(ToolName.BUILD_IN_STATE, pre, post, f"queue {pre_q}->{post_q}")
    return _failed(
        ToolName.BUILD_IN_STATE, pre, post,
        BuildInStateError(f"queue {pre_q}->{post_q}, expected +1", observed=post_q),
    )


def _assign_research(intent: Intent, ctx: "AgentContext") -> ToolResult:
    """Self-contained: ensure research panel open, then assign; assert idle slots -1."""
    ctx.input.focus(ctx.geometry)
    ok, w = _ensure_panel(ctx, PanelId.RESEARCH, macros.HOTKEYS["research"])
    if not ok:
        if w.confidence.get("panel", 0.0) < ctx.config.timing.ncc_threshold:
            return _uncertain(ToolName.ASSIGN_RESEARCH, w, w, "research panel uncertain")
        return _failed(ToolName.ASSIGN_RESEARCH, w, w, PanelOpenError("research", w.open_panel.value))
    pre = ctx.perceive()
    if pre.idle_research_slots is None:
        return _uncertain(ToolName.ASSIGN_RESEARCH, pre, pre, "idle research slots unreadable")
    if pre.idle_research_slots < 1:
        return _failed(
            ToolName.ASSIGN_RESEARCH, pre, pre,
            PreconditionError("no idle research slot", observed=pre.idle_research_slots),
        )
    try:
        nx, ny = ctx.calibration.tech_point(intent.tech)
    except TemplateMissingError as e:
        return _failed(ToolName.ASSIGN_RESEARCH, pre, pre, e)
    pre_idle = pre.idle_research_slots
    ctx.input.click(ctx.geometry, ctx.geometry.full_crop(), nx, ny)
    post = ctx.perceive()
    if post.idle_research_slots is None:
        return _uncertain(ToolName.ASSIGN_RESEARCH, pre, post, "idle slots unreadable post-click")
    if post.idle_research_slots == pre_idle - 1:
        return _ok(ToolName.ASSIGN_RESEARCH, pre, post, f"idle {pre_idle}->{post.idle_research_slots}")
    return _failed(
        ToolName.ASSIGN_RESEARCH, pre, post,
        AssignResearchError(f"idle {pre_idle}->{post.idle_research_slots}, expected -1", observed=post.idle_research_slots),
    )


def _close_panels(intent: Intent, ctx: "AgentContext") -> ToolResult:
    pre = ctx.perceive(read_numbers=False)
    try:
        post = macros.reset_to_home(ctx)
    except ResetFailedError as e:
        return _failed(ToolName.CLOSE_PANELS, pre, ctx.perceive(read_numbers=False), e)
    return _ok(ToolName.CLOSE_PANELS, pre, post, "reset to home")


HANDLERS: dict[ToolName, Callable[[Intent, "AgentContext"], ToolResult]] = {
    ToolName.OBSERVE: _observe,
    ToolName.ENSURE_PAUSED: _ensure_paused,
    ToolName.SET_SPEED: _set_speed,
    ToolName.OPEN_CONSTRUCTION: _open_construction,
    ToolName.SELECT_BUILDING: _select_building,
    ToolName.BUILD_IN_STATE: _build_in_state,
    ToolName.OPEN_RESEARCH: _open_research,
    ToolName.ASSIGN_RESEARCH: _assign_research,
    ToolName.CLOSE_PANELS: _close_panels,
}


def execute(intent: Intent, ctx: "AgentContext") -> ToolResult:
    """Validate then dispatch. Result is always a ToolResult with a real verdict."""
    try:
        validate_intent(intent)
    except IntentValidationError as e:
        tool = intent.tool if isinstance(intent.tool, ToolName) else ToolName.OBSERVE
        return ToolResult(tool, Verdict.FAILED, assertion=str(e), error=e)

    handler = HANDLERS.get(intent.tool)
    if handler is None:  # pragma: no cover - validate_intent already guards
        return ToolResult(intent.tool, Verdict.FAILED, assertion="no handler", error=PreconditionError("no handler"))

    start = time.time()
    try:
        result = handler(intent, ctx)
    except AgentError as e:
        result = ToolResult(intent.tool, Verdict.FAILED, assertion=str(e), error=e)
    return replace(result, latency_s=time.time() - start)
