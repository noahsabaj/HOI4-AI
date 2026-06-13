"""Perception tiers.

T0 (deterministic, no model): is a panel open? paused? event popup? — via NCC of a
calibrated ROI crop against a stored template, with a confidence floor. A
below-threshold match is *uncertain*, never reported as success.

T1/T2 reads are delegated to the injected ``reader`` (see ``perceive``).
"""

from __future__ import annotations

from PIL import Image

from ..calibration import Calibration
from ..enums import PanelId
from ..geometry import WindowGeometry
from .ncc import match_resized
from .templates import TemplateStore


def crop_roi(full_img: Image.Image, geo: WindowGeometry, frac) -> Image.Image:
    crop = geo.panel_crop(frac)
    box = (
        crop.client_x0,
        crop.client_y0,
        crop.client_x0 + crop.crop_w,
        crop.client_y0 + crop.crop_h,
    )
    return full_img.crop(box)


def roi_score(
    full_img: Image.Image,
    geo: WindowGeometry,
    calib: Calibration,
    templates: TemplateStore,
    roi_name: str,
    template_name: str | None = None,
) -> float:
    """NCC of the named ROI crop vs its stored template (0.0 if either is absent)."""
    template_name = template_name or roi_name
    if roi_name not in calib.rois or not templates.has(template_name):
        return 0.0
    roi_img = crop_roi(full_img, geo, calib.roi(roi_name))
    return match_resized(roi_img, templates.get(template_name))


def classify_panel(
    full_img: Image.Image,
    geo: WindowGeometry,
    calib: Calibration,
    templates: TemplateStore,
    threshold: float,
) -> tuple[PanelId, float]:
    candidates = {
        PanelId.EVENT_POPUP: "event_popup",
        PanelId.CONSTRUCTION: "construction_panel",
        PanelId.RESEARCH: "research_panel",
    }
    best_panel, best_score = PanelId.NONE, 0.0
    for panel, roi_name in candidates.items():
        score = roi_score(full_img, geo, calib, templates, roi_name)
        if score > best_score:
            best_panel, best_score = panel, score
    if best_score >= threshold:
        return best_panel, best_score
    return PanelId.NONE, best_score


def read_pause(
    full_img: Image.Image,
    geo: WindowGeometry,
    calib: Calibration,
    templates: TemplateStore,
    threshold: float,
) -> tuple[bool | None, float]:
    """True if paused, False if running, None if uncertain.

    Uses ``pause_on`` / ``pause_off`` templates over the ``pause`` ROI.
    """
    on = roi_score(full_img, geo, calib, templates, "pause", "pause_on")
    off = roi_score(full_img, geo, calib, templates, "pause", "pause_off")
    if max(on, off) < threshold:
        return None, max(on, off)
    return (on >= off), max(on, off)
