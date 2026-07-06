"""Bounded retry + best-effort recovery (never an unbounded silent loop)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..enums import ToolName, Verdict
from ..errors import AgentError
from ..schemas import Intent, ToolResult
from ..tools import macros
from ..tools.executor import execute

if TYPE_CHECKING:
    from ..context import AgentContext


def act_with_retry(ctx: "AgentContext", intent: Intent, max_retries: int) -> ToolResult:
    """Run a tool; retry only retry-SAFE UNCERTAIN results, up to ``max_retries``.

    Handlers mark an UNCERTAIN result ``retry_safe=False`` when it was produced
    *after* a non-idempotent mutation (clicked, couldn't read the effect) —
    re-executing could repeat the mutation, so such results return as-is. A
    FAILED result also returns immediately for the controller to recover/halt.
    """
    result = execute(intent, ctx)
    attempt = 0
    while result.verdict is Verdict.UNCERTAIN and result.retry_safe and attempt < max_retries:
        attempt += 1
        result = execute(intent, ctx)
    return replace(result, retries=attempt)


def recover(ctx: "AgentContext") -> bool:
    """Try to return the game to a known state. True if it worked.

    Pause first (stop the clock before anything else), then reset the UI to a
    home/no-panel state. Any AgentError here means recovery failed — the caller
    decides whether to halt.
    """
    try:
        execute(Intent(ToolName.ENSURE_PAUSED, paused=True), ctx)  # best-effort
        macros.reset_to_home(ctx)
        return True
    except AgentError:
        return False


def recover_with_consult(ctx: "AgentContext") -> tuple[bool, dict | None]:
    """``recover()``, escalating ONCE to a VLM hint before giving up.

    When deterministic recovery fails and consulting is enabled, the brain looks
    at the screen and picks a single unblock action (escape / map-mode key /
    event-option click / none); the action is applied and recovery retried once.
    The hint never bypasses verification — ``reset_to_home`` must still succeed.

    Returns (recovered, consult_info) — consult_info is None when no consult
    happened, else {"hint", "prompt", "raw"} for the trace.
    """
    if recover(ctx):
        return True, None
    if not (ctx.config.recovery_vlm_consult and ctx.brain is not None):
        return False, None

    options = ["press_escape", "press_map_mode"]
    if ctx.calibration.ui_point("event_option") is not None:
        options.append("click_event_option")
    options.append("none")
    try:
        hint = ctx.brain.consult_recovery(ctx.capture.grab(ctx.geometry), options)
    except AgentError as e:
        return False, {"hint": f"consult failed: {e}", "prompt": None, "raw": None}
    info = {
        "hint": hint,
        "prompt": getattr(ctx.brain, "last_prompt", None),
        "raw": getattr(ctx.brain, "last_raw", None),
    }
    settle = ctx.config.timing.settle_ms / 1000.0
    if hint == "press_escape":
        ctx.input.key(macros.HOTKEYS["back"])
        ctx.sleep(settle)
    elif hint == "press_map_mode":
        key = macros.MAP_MODE_KEYS.get(ctx.calibration.home_map_mode)
        if key is None:
            return False, info
        ctx.input.key(key)
        ctx.sleep(settle)
    elif hint == "click_event_option":
        option = ctx.calibration.ui_point("event_option")
        assert option is not None  # only offered when calibrated
        ctx.input.click(ctx.geometry, ctx.geometry.full_crop(), *option)
        ctx.sleep(settle)
    else:  # "none": the model sees nothing recoverable — trust it and halt
        return False, info
    return recover(ctx), info
