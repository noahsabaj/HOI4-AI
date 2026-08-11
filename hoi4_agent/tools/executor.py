"""Tool handlers + ``execute``. Each handler returns a ToolResult (never None).

The contract: precondition (checked against a fresh perceive) -> deterministic
steps (hotkeys + the few calibrated clicks) -> post-condition assertion via
deterministic perception. ``build_in_state`` / ``assign_research`` are the
load-bearing ones: they assert the construction queue grew / idle slots dropped,
proving the click had effect — not by re-asking the model.

Retry-safety rules (enforced here, honored by ``recovery.act_with_retry``):
- A mutation is only performed against a *readable* baseline; if the baseline is
  unreadable the handler returns UNCERTAIN **without acting** (safe to re-run).
- After any injected input the handler waits ``timing.settle_ms`` and, when a
  verify metric is unreadable, re-READS up to ``timing.verify_read_retries``
  times. It never re-acts. If the metric stays unreadable after an action, the
  result is UNCERTAIN with ``retry_safe=False`` so nothing re-runs the handler.
- "Not ready" (no free slot right now) is ``NotReadyError`` — a wait-state the
  controller answers by advancing time, not a failure.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

from ..enums import TECH_TABS, PanelId, ToolName, Verdict
from ..errors import (
    AgentError,
    AssignResearchError,
    BuildInStateError,
    IntentValidationError,
    NotReadyError,
    PanelOpenError,
    PauseToggleError,
    PreconditionError,
    ResetFailedError,
    SpeedSetError,
    TemplateMissingError,
)
from ..geometry import CropRect
from ..schemas import Intent, ToolResult, validate_intent
from . import grounding, macros

if TYPE_CHECKING:
    from ..context import AgentContext
    from ..schemas import WorldState

MAX_SPEED_STEPS = 6


def _ok(tool, pre, post, assertion) -> ToolResult:
    return ToolResult(tool, Verdict.OK, pre, post, assertion)


def _uncertain(tool, pre, post, assertion, *, retry_safe: bool = True) -> ToolResult:
    return ToolResult(tool, Verdict.UNCERTAIN, pre, post, assertion, retry_safe=retry_safe)


def _failed(tool, pre, post, err: Exception) -> ToolResult:
    return ToolResult(tool, Verdict.FAILED, pre, post, str(err), err)


# --- input-then-settle primitives --------------------------------------------
def _settle(ctx: "AgentContext") -> None:
    ctx.sleep(ctx.config.timing.settle_ms / 1000.0)


def _unfocused(tool: ToolName, ctx: "AgentContext") -> ToolResult | None:
    """FAILED result if the game window can't take focus — no input is sent to
    whatever window IS focused (v3 sprayed keystrokes into editors this way)."""
    if ctx.input.focus(ctx.geometry):
        return None
    return _failed(tool, None, None, PreconditionError("game window not focused"))


def _press(ctx: "AgentContext", key: str) -> None:
    ctx.input.key(key)
    _settle(ctx)


def _click_pt(ctx: "AgentContext", nx: int, ny: int, crop: CropRect | None = None) -> None:
    ctx.input.click(ctx.geometry, crop or ctx.geometry.full_crop(), nx, ny)
    _settle(ctx)


def _read_until(
    ctx: "AgentContext",
    extract: Callable[["WorldState"], object],
    *,
    fields=None,
    read_numbers: bool = True,
):
    """Perceive until ``extract(world)`` is readable — bounded re-READS, never re-acts."""
    world = ctx.perceive(read_numbers=read_numbers, fields=fields)
    value = extract(world)
    tries = 0
    while value is None and tries < ctx.config.timing.verify_read_retries:
        tries += 1
        _settle(ctx)
        world = ctx.perceive(read_numbers=read_numbers, fields=fields)
        value = extract(world)
    return value, world


# --- handlers ---------------------------------------------------------------
def _observe(intent: Intent, ctx: "AgentContext") -> ToolResult:
    w = ctx.perceive()
    return _ok(ToolName.OBSERVE, w, w, "observed world state")


def _ensure_paused(intent: Intent, ctx: "AgentContext") -> ToolResult:
    if (unfocused := _unfocused(ToolName.ENSURE_PAUSED, ctx)) is not None:
        return unfocused
    desired = bool(intent.paused)
    pre = ctx.perceive(read_numbers=False)
    if pre.paused is None:
        return _uncertain(ToolName.ENSURE_PAUSED, pre, pre, "pause state unreadable")
    if pre.paused == desired:
        return _ok(ToolName.ENSURE_PAUSED, pre, pre, f"already paused={desired}")
    _press(ctx, macros.HOTKEYS["pause"])
    paused, post = _read_until(ctx, lambda w: w.paused, read_numbers=False)
    if paused is None:
        # Toggling again on re-run is safe: the handler re-checks state first.
        return _uncertain(ToolName.ENSURE_PAUSED, pre, post, "pause unreadable after toggle")
    if paused == desired:
        return _ok(ToolName.ENSURE_PAUSED, pre, post, f"toggled to paused={desired}")
    return _failed(ToolName.ENSURE_PAUSED, pre, post, PauseToggleError(post.paused, desired))


def _set_speed(intent: Intent, ctx: "AgentContext") -> ToolResult:
    if (unfocused := _unfocused(ToolName.SET_SPEED, ctx)) is not None:
        return unfocused
    target = intent.speed
    if target is None:  # pragma: no cover - validate_intent guarantees it
        return _failed(ToolName.SET_SPEED, None, None,
                       IntentValidationError("set_speed", "missing speed"))
    pre = ctx.perceive(fields={"speed"})
    cur = pre.speed
    if cur is None:
        return _uncertain(ToolName.SET_SPEED, pre, pre, "speed unreadable")
    post = pre
    steps = 0
    while cur != target and steps < MAX_SPEED_STEPS:
        key = macros.HOTKEYS["speed_up"] if cur < target else macros.HOTKEYS["speed_down"]
        _press(ctx, key)
        cur, post = _read_until(ctx, lambda w: w.speed, fields={"speed"})
        steps += 1
        if cur is None:
            # set_speed is convergent: a re-run re-reads and continues safely.
            return _uncertain(ToolName.SET_SPEED, pre, post, "speed unreadable mid-adjust")
    if post.speed == target:
        return _ok(ToolName.SET_SPEED, pre, post, f"speed -> {target}")
    return _failed(ToolName.SET_SPEED, pre, post, SpeedSetError(f"speed {post.speed} != {target}", observed=post.speed))


def _open(intent, ctx, target_panel, hotkey, tool) -> ToolResult:
    if (unfocused := _unfocused(tool, ctx)) is not None:
        return unfocused
    pre = ctx.perceive(read_numbers=False)
    if pre.open_panel is target_panel:
        return _ok(tool, pre, pre, f"{target_panel.value} already open")
    _press(ctx, hotkey)
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
    _press(ctx, hotkey)
    world = ctx.perceive(read_numbers=False)
    return (world.open_panel is target_panel), world


def _build_in_state(intent: Intent, ctx: "AgentContext") -> ToolResult:
    """Self-contained: ensure construction panel open + building selected, then queue.

    Verified by the construction queue growing by exactly 1, not by re-asking the
    model to confirm. The queue baseline must be readable BEFORE the state click —
    otherwise the handler declines to act. Slot pacing (free civ factory) is
    enforced here, where the panel showing it is open.

    KNOWN LIMIT — the postcondition is CARDINALITY, not IDENTITY. ``queue+1`` is
    satisfied identically whether the click landed on Ruhr or on Bavaria, and
    which state gets the factory is the entire content of the playbook. Nothing
    perceived here distinguishes a correct state click from a wrong one; the
    calibrated point is simply trusted, and ``Calibration`` carries no camera or
    zoom state that could invalidate that trust. Closing this needs either a
    per-state read or the offline save cross-check (``save-audit --trace``), and
    until it is closed no claim about WHERE the agent built is evidence-backed.
    """
    if (unfocused := _unfocused(ToolName.BUILD_IN_STATE, ctx)) is not None:
        return unfocused
    ok, w = _ensure_panel(ctx, PanelId.CONSTRUCTION, macros.HOTKEYS["construction"])
    if w.event_popup:
        # A popup swallows clicks; recovery (reset_to_home) knows how to dismiss it.
        return _failed(
            ToolName.BUILD_IN_STATE, w, w,
            PreconditionError("event popup blocking input", observed="event_popup"),
        )
    if not ok:
        if w.confidence.get("panel", 0.0) < ctx.config.timing.ncc_threshold:
            return _uncertain(ToolName.BUILD_IN_STATE, w, w, "construction panel uncertain")
        return _failed(ToolName.BUILD_IN_STATE, w, w, PanelOpenError("construction", w.open_panel.value))
    if intent.building is not None:
        try:
            bx, by = ctx.calibration.building_point(intent.building)
        except TemplateMissingError as e:
            return _failed(ToolName.BUILD_IN_STATE, w, w, e)
        _click_pt(ctx, bx, by)
    if intent.state is None:  # pragma: no cover - validate_intent guarantees it
        return _failed(ToolName.BUILD_IN_STATE, w, w,
                       IntentValidationError("build_in_state", "missing state"))
    point: tuple[int, int] | None = None
    point_crop: CropRect | None = None
    try:
        point = ctx.calibration.state_point(intent.state)
    except TemplateMissingError as e:
        if ctx.locator is None:
            return _failed(ToolName.BUILD_IN_STATE, w, w, e)
        # no calibrated point: ground on the map crop AFTER the camera anchor

    pre_q, pre = _read_until(
        ctx, lambda w: w.construction_queue_len, fields={"construction_queue", "free_civ_slots"}
    )
    if pre_q is None:
        return _uncertain(ToolName.BUILD_IN_STATE, pre, pre, "queue baseline unreadable — not clicking")
    # Pacing gate (playbook precondition, enforced where it's visible). A None
    # free-civ read does NOT block: queueing without a free factory is legal in
    # HOI4, and the +1 assertion below verifies the click regardless.
    if pre.free_civ_slots is not None and pre.free_civ_slots < 1:
        return _failed(
            ToolName.BUILD_IN_STATE, pre, pre,
            NotReadyError("no free civilian factory", observed=pre.free_civ_slots),
        )
    # Re-anchor the camera so the state point is valid regardless of pan drift
    # (minimap clicks pan without selecting). Zoom is assumed unchanged — the
    # agent never scrolls; calibrate and run at the same zoom.
    anchor = ctx.calibration.ui_point("minimap_anchor")
    if anchor is not None:
        _click_pt(ctx, *anchor)
    if point is None:
        point_crop = grounding.map_crop(ctx)
        state_name = intent.state.value.replace("_", " ").title()
        point = grounding.locate_point(ctx, f"the German state of {state_name} on the map", point_crop)
        if point is None:
            return _failed(
                ToolName.BUILD_IN_STATE, pre, pre,
                BuildInStateError(f"no calibrated point and grounding failed for {intent.state.value}"),
            )
    _click_pt(ctx, *point, crop=point_crop)
    post_q, post = _read_until(ctx, lambda w: w.construction_queue_len, fields={"construction_queue"})
    if post_q is None:
        return _uncertain(
            ToolName.BUILD_IN_STATE, pre, post,
            "queue unreadable after click — may have landed", retry_safe=False,
        )
    if post_q == pre_q + 1:
        return _ok(ToolName.BUILD_IN_STATE, pre, post, f"queue {pre_q}->{post_q}")
    return _failed(
        ToolName.BUILD_IN_STATE, pre, post,
        BuildInStateError(f"queue {pre_q}->{post_q}, expected +1", observed=post_q),
    )


def _assign_research(intent: Intent, ctx: "AgentContext") -> ToolResult:
    """Self-contained: ensure research panel open, then assign; assert idle slots -1."""
    if (unfocused := _unfocused(ToolName.ASSIGN_RESEARCH, ctx)) is not None:
        return unfocused
    ok, w = _ensure_panel(ctx, PanelId.RESEARCH, macros.HOTKEYS["research"])
    if w.event_popup:
        return _failed(
            ToolName.ASSIGN_RESEARCH, w, w,
            PreconditionError("event popup blocking input", observed="event_popup"),
        )
    if not ok:
        if w.confidence.get("panel", 0.0) < ctx.config.timing.ncc_threshold:
            return _uncertain(ToolName.ASSIGN_RESEARCH, w, w, "research panel uncertain")
        return _failed(ToolName.ASSIGN_RESEARCH, w, w, PanelOpenError("research", w.open_panel.value))
    pre_idle, pre = _read_until(ctx, lambda w: w.idle_research_slots, fields={"idle_research_slots"})
    if pre_idle is None:
        return _uncertain(ToolName.ASSIGN_RESEARCH, pre, pre, "idle research slots unreadable")
    if pre_idle < 1:
        return _failed(
            ToolName.ASSIGN_RESEARCH, pre, pre,
            NotReadyError("no idle research slot", observed=pre_idle),
        )
    if intent.tech is None:  # pragma: no cover - validate_intent guarantees it
        return _failed(ToolName.ASSIGN_RESEARCH, pre, pre,
                       IntentValidationError("assign_research", "missing tech"))
    # Select the tech's research tab BEFORE resolving/clicking its point: tech
    # points (and the grounding crop) are relative to the open tab, which HOI4
    # remembers across panel opens. Preflight guarantees the tab is calibrated.
    tab = TECH_TABS.get(intent.tech)
    if tab is None:  # pragma: no cover - TECH_TABS covers all Tech (asserted in tests)
        return _failed(ToolName.ASSIGN_RESEARCH, pre, pre,
                       AssignResearchError(f"no research-tab mapping for {intent.tech.value}"))
    tab_point = ctx.calibration.research_tab_point(tab)
    if tab_point is None:
        return _failed(ToolName.ASSIGN_RESEARCH, pre, pre,
                       AssignResearchError(f"research tab {tab.value!r} has no calibrated point"))
    _click_pt(ctx, tab_point[0], tab_point[1])
    point_crop: CropRect | None = None
    try:
        point: tuple[int, int] | None = ctx.calibration.tech_point(intent.tech)
    except TemplateMissingError as e:
        if ctx.locator is None:
            return _failed(ToolName.ASSIGN_RESEARCH, pre, pre, e)
        point_crop = grounding.roi_crop(ctx, "research_panel")
        tech_name = intent.tech.value.replace("_", " ")
        point = grounding.locate_point(
            ctx, f"the '{tech_name}' technology entry in the research panel", point_crop
        )
    if point is None:
        return _failed(
            ToolName.ASSIGN_RESEARCH, pre, pre,
            AssignResearchError(f"no calibrated point and grounding failed for {intent.tech.value}"),
        )
    _click_pt(ctx, point[0], point[1], crop=point_crop)
    post_idle, post = _read_until(ctx, lambda w: w.idle_research_slots, fields={"idle_research_slots"})
    if post_idle is None:
        return _uncertain(
            ToolName.ASSIGN_RESEARCH, pre, post,
            "idle slots unreadable post-click — may have landed", retry_safe=False,
        )
    if post_idle == pre_idle - 1:
        return _ok(ToolName.ASSIGN_RESEARCH, pre, post, f"idle {pre_idle}->{post_idle}")
    return _failed(
        ToolName.ASSIGN_RESEARCH, pre, post,
        AssignResearchError(f"idle {pre_idle}->{post_idle}, expected -1", observed=post_idle),
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
