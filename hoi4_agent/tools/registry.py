"""Introspection over the tool table (handlers live in ``executor``)."""

from __future__ import annotations

from ..enums import ToolName
from .executor import HANDLERS, execute  # re-export for convenience


def supported_tools() -> list[ToolName]:
    return sorted(HANDLERS, key=lambda t: t.value)


__all__ = ["execute", "supported_tools", "HANDLERS"]
