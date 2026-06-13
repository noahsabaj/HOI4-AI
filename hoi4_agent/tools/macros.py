"""Hotkeys + the known-good reset macro.

HOTKEYS are best-effort defaults. HOI4 hotkeys can shift between patches, so the
live bring-up / calibration step MUST confirm these in-game before trusting the
agent. They are centralized here so confirming them is a one-line change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..enums import PanelId
from ..errors import ResetFailedError

if TYPE_CHECKING:
    from ..context import AgentContext
    from ..schemas import WorldState

# VERIFY IN-GAME at bring-up. (Plan's research note: construction=T, research=W,
# E=Diplomacy. Confirm before a real run.)
HOTKEYS = {
    "construction": "t",
    "research": "w",
    "map_mode": "f1",     # default map mode = the calibrated "home view"
    "pause": "space",
    "speed_up": "+",
    "speed_down": "-",
    "back": "escape",
}

MAX_ESCAPES = 3


def reset_to_home(ctx: "AgentContext") -> "WorldState":
    """Esc up to 3x to close menus, then map-mode hotkey; assert we're at home.

    Raises ResetFailedError if the UI is not at a no-panel home state afterward.
    """
    ctx.input.focus(ctx.geometry)
    world = ctx.perceive(read_numbers=False)
    for _ in range(MAX_ESCAPES):
        if world.open_panel is PanelId.NONE and not world.event_popup:
            break
        ctx.input.key(HOTKEYS["back"])
        world = ctx.perceive(read_numbers=False)
    ctx.input.key(HOTKEYS["map_mode"])
    world = ctx.perceive(read_numbers=False)
    if world.open_panel is not PanelId.NONE or world.event_popup:
        raise ResetFailedError(
            "UI not at home after reset", observed=world.open_panel.value
        )
    return world
