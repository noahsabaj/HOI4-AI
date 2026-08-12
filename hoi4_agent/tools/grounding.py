"""Runtime grounding fallback: when a click-point isn't calibrated, ask the
locator model (a GUI-grounding specialist wired via ``[grounding] profile``)
where to click — crop-then-ground, never the full frame.

A grounded click goes through exactly the same postcondition verification as a
calibrated one (queue +1, idle -1, popup gone); grounding only supplies the
coordinates, never the trust.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import AgentError
from ..geometry import CropRect

if TYPE_CHECKING:
    from ..context import AgentContext


def locate_point(ctx: "AgentContext", instruction: str, crop: CropRect) -> tuple[int, int] | None:
    """Coordinates (0-1000 normalized on ``crop``) for the described element,
    or None when no locator is wired or the model can't answer."""
    if ctx.locator is None:
        return None
    try:
        img = ctx.capture.grab(ctx.geometry, crop)
        return ctx.locator.locate(img, instruction)
    except AgentError:
        return None


# Every calibrated ROI that lies INSIDE the construction panel. Their right
# edges bound where the panel ends; "construction_panel" alone does not, because
# the wizard has the operator box "a small static region (its header)" — its x1
# sits well inside the panel (0.09 of the client on a real 4K calibration, while
# the queue list reaches 0.16), so a crop starting there still contains most of
# the panel it is supposed to exclude.
_CONSTRUCTION_AREA_ROIS = ("construction_panel", "construction_queue", "free_civ_slots")


def map_crop(ctx: "AgentContext") -> CropRect:
    """The map area: everything right of the construction panel (the panel
    occludes the west of Germany, and grounding must see only map pixels).

    The panel's right edge is not calibrated directly, so this uses the
    rightmost edge of every ROI known to sit inside it — the best available
    bound. Uncalibrated entries simply don't contribute.
    """
    geo = ctx.geometry
    edges = [
        ctx.calibration.rois[name][2]
        for name in _CONSTRUCTION_AREA_ROIS
        if name in ctx.calibration.rois
    ]
    if not edges:
        return geo.full_crop()
    x0 = min(int(round(max(edges) * geo.client_w)), geo.client_w - 1)
    return CropRect(x0, 0, max(1, geo.client_w - x0), geo.client_h)


def roi_crop(ctx: "AgentContext", roi_name: str) -> CropRect:
    """A calibrated ROI as a capture crop (full client when uncalibrated)."""
    frac = ctx.calibration.rois.get(roi_name)
    if frac is None:
        return ctx.geometry.full_crop()
    return ctx.geometry.panel_crop(frac)
