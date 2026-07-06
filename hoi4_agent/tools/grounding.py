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


def map_crop(ctx: "AgentContext") -> CropRect:
    """The map area: everything right of the construction panel (the panel
    occludes the west of Germany, and grounding must see only map pixels)."""
    geo = ctx.geometry
    frac = ctx.calibration.rois.get("construction_panel")
    if frac is None:
        return geo.full_crop()
    x0 = int(round(frac[2] * geo.client_w))
    return CropRect(x0, 0, max(1, geo.client_w - x0), geo.client_h)


def roi_crop(ctx: "AgentContext", roi_name: str) -> CropRect:
    """A calibrated ROI as a capture crop (full client when uncalibrated)."""
    frac = ctx.calibration.rois.get(roi_name)
    if frac is None:
        return ctx.geometry.full_crop()
    return ctx.geometry.panel_crop(frac)
