"""Pure step-wizard engine for interactive operator flows.

Models a linear sequence of steps the operator walks with single-key
commands: capture (Enter), back (B / left arrow), keep a previously
recorded value when revisiting (K), skip where allowed (S), quit (Q).
Going back never discards work — earlier answers stay recorded, and on
the way forward each revisited step offers keep-or-recapture, so fixing
one mistake costs one recapture instead of a restart.

No I/O here: console key reading and screen/cursor capture live in the
CLI layer. Everything in this module is unit-testable off-Windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Step:
    """One operator action. ``id`` is "<section>:<name>[:<part>]"."""

    id: str
    prompt: str
    skippable: bool = False
    textual: bool = False  # value comes from typed text, not a hover/capture


class Wizard:
    """Linear cursor over steps with revisit-aware navigation."""

    def __init__(self, steps: list[Step]) -> None:
        if not steps:
            raise ValueError("wizard needs at least one step")
        ids = [s.id for s in steps]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate step ids")
        self.steps = list(steps)
        self.results: dict[str, Any] = {}  # step.id -> value; None means skipped
        self._i = 0

    @property
    def done(self) -> bool:
        return self._i >= len(self.steps)

    @property
    def current(self) -> Step | None:
        return None if self.done else self.steps[self._i]

    @property
    def position(self) -> tuple[int, int]:
        """1-based (current, total) for progress display."""
        return (min(self._i + 1, len(self.steps)), len(self.steps))

    def has_value(self) -> bool:
        """True when the current step was answered before (revisit)."""
        s = self.current
        return s is not None and s.id in self.results

    def record(self, value: Any) -> None:
        """Answer the current step (overwrites any earlier answer) and advance."""
        s = self.current
        if s is None:
            raise ValueError("wizard is done")
        self.results[s.id] = value
        self._i += 1

    def keep(self) -> bool:
        """Advance past a revisited step, keeping its earlier answer."""
        if not self.has_value():
            return False
        self._i += 1
        return True

    def skip(self) -> bool:
        """Record a skip (None) where the step allows it, and advance."""
        s = self.current
        if s is None or not s.skippable:
            return False
        self.results[s.id] = None
        self._i += 1
        return True

    def back(self) -> bool:
        """Step to the previous prompt (its answer stays until re-recorded)."""
        if self._i == 0:
            return False
        self._i -= 1
        return True
