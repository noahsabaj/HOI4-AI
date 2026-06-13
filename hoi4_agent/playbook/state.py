"""Persist PlaybookState to JSON so progress survives restarts."""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas import PlaybookState


def load_state(path: str | Path) -> PlaybookState:
    p = Path(path)
    if not p.is_file():
        return PlaybookState()
    return PlaybookState.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_state(path: str | Path, state: PlaybookState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
