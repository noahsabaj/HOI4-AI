"""Hotkeys + the known-good reset macro.

HOTKEYS are best-effort defaults. HOI4 hotkeys can shift between patches, so the
live bring-up MUST confirm these in-game before trusting the agent — the live
smoke test (`smoke-test`, with calibration + templates present) does exactly
that by opening each panel and asserting perception sees it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..enums import MapMode, PanelId
from ..errors import ResetFailedError

if TYPE_CHECKING:
    from ..context import AgentContext
    from ..schemas import WorldState

# VERIFY IN-GAME at bring-up (see live smoke). (Plan's research note:
# construction=T, research=W, E=Diplomacy.)
HOTKEYS = {
    "construction": "t",
    "research": "w",
    "pause": "space",
    "speed_up": "+",
    "speed_down": "-",
    "back": "escape",
}

# The calibrated home view's map mode -> the key that selects it.
MAP_MODE_KEYS = {MapMode.DEFAULT.value: "f1"}

MAX_ESCAPES = 3


def reset_to_home(ctx: "AgentContext") -> "WorldState":
    """Clear blocking UI, then re-anchor to the calibrated home view.

    Bounded loop handles the three blocking states in priority order: an event
    popup (click its calibrated option button if we have one, else escape), the
    pause/game menu (escape), an open panel (escape). Then the calibrated map
    mode is selected and — if calibrated — the minimap anchor is clicked so the
    camera pans back to the position every state click-point was recorded at.

    Raises ResetFailedError if focus fails or the UI still shows a panel, popup,
    or pause menu afterward.
    """

    def _press(key: str) -> None:
        ctx.input.key(key)
        ctx.sleep(ctx.config.timing.settle_ms / 1000.0)

    def _click(nx: int, ny: int) -> None:
        ctx.input.click(ctx.geometry, ctx.geometry.full_crop(), nx, ny)
        ctx.sleep(ctx.config.timing.settle_ms / 1000.0)

    if not ctx.input.focus(ctx.geometry):
        raise ResetFailedError("cannot focus the game window")
    world = ctx.perceive(read_numbers=False)
    for _ in range(MAX_ESCAPES):
        if world.event_popup:
            option = ctx.calibration.ui_point("event_option")
            if option is not None:
                _click(*option)  # popups usually need an option click, not escape
            else:
                _press(HOTKEYS["back"])
        elif world.pause_menu or world.open_panel is not PanelId.NONE:
            _press(HOTKEYS["back"])
        else:
            break
        world = ctx.perceive(read_numbers=False)

    map_mode_key = MAP_MODE_KEYS.get(ctx.calibration.home_map_mode)
    if map_mode_key is None:
        raise ResetFailedError(
            f"calibrated home map mode {ctx.calibration.home_map_mode!r} has no known key"
        )
    _press(map_mode_key)
    anchor = ctx.calibration.ui_point("minimap_anchor")
    if anchor is not None:
        _click(*anchor)  # minimap click: pans the camera, selects nothing
    world = ctx.perceive(read_numbers=False)
    if world.open_panel is not PanelId.NONE or world.event_popup or world.pause_menu:
        raise ResetFailedError(
            "UI not at home after reset", observed=world.open_panel.value
        )
    return world
