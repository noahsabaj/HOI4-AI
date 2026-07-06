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
