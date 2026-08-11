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
    """Argmax over the mutually-exclusive MAIN panels only.

    Blocking overlays (event popup, pause menu) are detected independently by
    ``detect_popup`` / ``detect_pause_menu`` — inside an argmax a popup could be
    masked by a stronger panel score and become invisible while blocking input.
    """
    candidates = {
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


def _roi_present(
    full_img: Image.Image,
    geo: WindowGeometry,
    calib: Calibration,
    templates: TemplateStore,
    roi_name: str,
    threshold: float,
) -> tuple[bool, float]:
    score = roi_score(full_img, geo, calib, templates, roi_name)
    return score >= threshold, score


def detect_popup(
    full_img: Image.Image,
    geo: WindowGeometry,
    calib: Calibration,
    templates: TemplateStore,
    threshold: float,
) -> tuple[bool, float]:
    """Independent event-popup check — never competes with panel scores."""
    return _roi_present(full_img, geo, calib, templates, "event_popup", threshold)


def detect_pause_menu(
    full_img: Image.Image,
    geo: WindowGeometry,
    calib: Calibration,
    templates: TemplateStore,
    threshold: float,
) -> tuple[bool, float]:
    """The escape/game menu — otherwise undetectable and it swallows hotkeys."""
    return _roi_present(full_img, geo, calib, templates, "pause_menu", threshold)


# Minimum score separation between pause_on/pause_off before we trust the argmax
# — the two templates share most of their pixels, so a near-tie is a guess.
PAUSE_MARGIN = 0.05

# Same rule for the speed indicator, and it needs it more: speed_1..speed_5 are
# the SAME widget with a different number of chevrons lit, so they share even
# more pixels than pause_on/pause_off. Without a margin an argmax turns a
# coin-flip between adjacent speeds into a confident fact.
SPEED_MARGIN = 0.05


def read_pause(
    full_img: Image.Image,
    geo: WindowGeometry,
    calib: Calibration,
    templates: TemplateStore,
    threshold: float,
) -> tuple[bool | None, float]:
    """True if paused, False if running, None if uncertain.

    Uses ``pause_on`` / ``pause_off`` templates over the ``pause`` ROI. Uncertain
    when both score below ``threshold`` OR when they score within PAUSE_MARGIN
    of each other (a coin-flip must never become a fact).
    """
    on = roi_score(full_img, geo, calib, templates, "pause", "pause_on")
    off = roi_score(full_img, geo, calib, templates, "pause", "pause_off")
    best = max(on, off)
    if best < threshold or abs(on - off) < PAUSE_MARGIN:
        return None, best
    return (on > off), best


def read_speed(
    full_img: Image.Image,
    geo: WindowGeometry,
    calib: Calibration,
    templates: TemplateStore,
    threshold: float,
) -> tuple[int | None, float]:
    """Game speed 1..5 via whole-ROI template argmax (``speed_1``..``speed_5``).

    Deterministic T0: the indicator renders as chevrons, not digits, so it is a
    5-way template classification, not a numeric read.

    Uncertain (None) when the best score is below ``threshold`` OR within
    SPEED_MARGIN of the runner-up — see SPEED_MARGIN. Templates that are absent
    score 0.0 and so can't be the runner-up that blocks a clean single match.
    """
    scored = [
        (roi_score(full_img, geo, calib, templates, "speed", f"speed_{s}"), s)
        for s in range(1, 6)
    ]
    ranked = sorted(scored, reverse=True)
    best_score, best_speed = ranked[0]
    runner_up = ranked[1][0]
    if best_score < threshold or (best_score - runner_up) < SPEED_MARGIN:
        return None, best_score
    return best_speed, best_score
