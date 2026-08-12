"""Perception tiers.

T0 (deterministic, no model): is a panel open? paused? event popup? — via NCC of a
calibrated ROI crop against a stored template, with a confidence floor. A
below-threshold match is *uncertain*, never reported as success.

T1/T2 reads are delegated to the injected ``reader`` (see ``perceive``).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..calibration import Calibration
from ..enums import PanelId
from ..geometry import WindowGeometry
from .ncc import match_resized, ncc, to_gray_f32
from .templates import TemplateStore


def _resized(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    """Grayscale array onto a common grid, so residuals can be compared."""
    if gray.shape == (height, width):
        return gray
    img = Image.fromarray(gray.astype(np.float32), mode="F")
    return np.asarray(img.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


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

# Same rule for the speed indicator, but applied to a DIFFERENT score — see
# read_speed. speed_1..speed_5 are the same widget with a different number of
# chevrons lit, so whole-ROI NCC between them is dominated by the shared widget:
# measured on real captures, speed_4 vs speed_5 scores 0.9592. A margin over
# whole-ROI scores therefore rejects every reading at the high speeds instead of
# only the ambiguous ones, so the discrimination happens on mean-centered
# residuals where the shared structure has been removed (same captures: worst
# margin 0.0408 -> 0.3324).
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
    """Game speed 1..5 by template classification (``speed_1``..``speed_5``).

    Deterministic T0: the indicator renders as chevrons, not digits, so it is a
    5-way template classification, not a numeric read.

    Two scores, answering two different questions:

    - *Is this the speed widget at all?* Whole-ROI NCC against the best template,
      gated on ``threshold``. This is the confidence returned to callers.
    - *Which speed?* NCC on MEAN-CENTERED residuals, gated on SPEED_MARGIN.
      Subtracting the per-pixel mean of the template set removes the widget
      structure every speed shares and leaves only what distinguishes them, which
      is the one chevron that differs. Without this the shared structure swamps
      the signal and adjacent speeds are indistinguishable (see SPEED_MARGIN).

    Uncertain (None) if the ROI doesn't look like the widget, or if the residuals
    cannot separate the top two candidates. With fewer than two templates loaded
    there is nothing to center against and nothing to confuse, so the whole-ROI
    argmax stands on its own.
    """
    available = [(s, templates.get(f"speed_{s}")) for s in range(1, 6) if templates.has(f"speed_{s}")]
    if not available or "speed" not in calib.rois:
        return None, 0.0

    whole = [(roi_score(full_img, geo, calib, templates, "speed", f"speed_{s}"), s) for s, _ in available]
    best_score, best_speed = max(whole)
    if best_score < threshold:
        return None, best_score
    if len(available) == 1:
        return best_speed, best_score

    # Put every template and the ROI on one grid, then compare residuals.
    ref_h, ref_w = available[0][1].shape
    grid = {s: _resized(t, ref_w, ref_h) for s, t in available}
    mean = np.mean(list(grid.values()), axis=0)
    roi = _resized(to_gray_f32(crop_roi(full_img, geo, calib.roi("speed"))), ref_w, ref_h)

    residual = sorted(((ncc(roi - mean, t - mean), s) for s, t in grid.items()), reverse=True)
    if (residual[0][0] - residual[1][0]) < SPEED_MARGIN:
        return None, best_score
    return residual[0][1], best_score
