"""Bounded retry + best-effort recovery (never an unbounded silent loop)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..enums import Verdict
from ..errors import ResetFailedError
from ..schemas import Intent, ToolResult
from ..tools import macros
from ..tools.executor import execute

if TYPE_CHECKING:
    from ..context import AgentContext


def act_with_retry(ctx: "AgentContext", intent: Intent, max_retries: int) -> ToolResult:
    """Run a tool; retry only on UNCERTAIN (transient read) up to ``max_retries``.

    A FAILED result returns immediately for the controller to recover/halt — there's
    no point hammering a deterministic failure.
    """
    result = execute(intent, ctx)
    attempt = 0
    while result.verdict is Verdict.UNCERTAIN and attempt < max_retries:
        attempt += 1
        result = execute(intent, ctx)
    return replace(result, retries=attempt)


def recover(ctx: "AgentContext") -> bool:
    """Try to return the UI to a known home state. True if it worked."""
    try:
        macros.reset_to_home(ctx)
        return True
    except ResetFailedError:
        return False
