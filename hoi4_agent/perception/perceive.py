"""Build a WorldState from one captured frame.

Deterministic T0 (panels, pause, speed) + numeric T1 reads through the injected
``reader``. The wired reader is glyph-templates-first with a VLM fallback (see
``digits.FallbackReader``) — the deterministic seam the design called for is now
the default path. ``None`` fields mean uncertain.

Two guards keep reads honest:
- ``fields``: callers name exactly the numeric facts they need (None = all), so
  a queue verification is one read, not five.
- context-gating: slot/queue counters only render while their panel is open, so
  those fields are read only when the T0 panel classification matches — never
  asked of map pixels.
"""

from __future__ import annotations

import time
from typing import Iterable, Protocol

from PIL import Image

from ..calibration import Calibration
from ..enums import PanelId
from ..geometry import WindowGeometry
from ..schemas import GameDate, WorldState
from .templates import TemplateStore
from .tiers import classify_panel, crop_roi, detect_pause_menu, detect_popup, read_pause, read_speed


class Reader(Protocol):
    def read_number(self, crop: Image.Image, field: str) -> int | None: ...
    def read_date(self, crop: Image.Image) -> GameDate | None: ...


class CaptureLike(Protocol):
    def grab(self, geo: WindowGeometry, crop=None) -> Image.Image: ...


NUMERIC_FIELDS = ("date", "free_civ_slots", "idle_research_slots", "construction_queue", "speed")

# A field listed here is only readable while its panel is open; reading it in any
# other UI state returns None (uncertain) without touching the reader.
FIELD_PANEL_CONTEXT: dict[str, PanelId] = {
    "free_civ_slots": PanelId.CONSTRUCTION,
    "construction_queue": PanelId.CONSTRUCTION,
    "idle_research_slots": PanelId.RESEARCH,
}


def _read_num(full, geo, calib, reader, roi_name, lo=0, hi=999) -> int | None:
    if roi_name not in calib.rois:
        return None
    crop = crop_roi(full, geo, calib.roi(roi_name))
    v = reader.read_number(crop, roi_name)
    if v is None:
        return None
    return v if lo <= v <= hi else None


def perceive(
    *,
    capture: CaptureLike,
    geo: WindowGeometry,
    calib: Calibration,
    templates: TemplateStore,
    threshold: float,
    reader: Reader,
    read_numbers: bool = True,
    fields: Iterable[str] | None = None,
) -> WorldState:
    """Capture and interpret the current screen.

    ``read_numbers=False`` skips all numeric reads (T0-only poll). With
    ``read_numbers=True``, ``fields`` selects which numeric facts to read
    (None = all of ``NUMERIC_FIELDS``).
    """
    full = capture.grab(geo)
    conf: dict[str, float] = {}

    panel, pconf = classify_panel(full, geo, calib, templates, threshold)
    conf["panel"] = pconf
    paused, pause_conf = read_pause(full, geo, calib, templates, threshold)
    conf["pause"] = pause_conf
    popup, popup_conf = detect_popup(full, geo, calib, templates, threshold)
    conf["popup"] = popup_conf
    pause_menu, menu_conf = detect_pause_menu(full, geo, calib, templates, threshold)
    conf["pause_menu"] = menu_conf

    if not read_numbers:
        want: set[str] = set()
    elif fields is None:
        want = set(NUMERIC_FIELDS)
    else:
        want = set(fields)

    def in_context(field: str) -> bool:
        need = FIELD_PANEL_CONTEXT.get(field)
        return need is None or panel is need

    date = None
    free_civ = idle_research = queue_len = speed = None
    if "date" in want and "date" in calib.rois:
        date = reader.read_date(crop_roi(full, geo, calib.roi("date")))
    if "free_civ_slots" in want and in_context("free_civ_slots"):
        free_civ = _read_num(full, geo, calib, reader, "free_civ_slots", 0, 99)
    if "idle_research_slots" in want and in_context("idle_research_slots"):
        idle_research = _read_num(full, geo, calib, reader, "idle_research_slots", 0, 9)
    if "construction_queue" in want and in_context("construction_queue"):
        queue_len = _read_num(full, geo, calib, reader, "construction_queue", 0, 99)
    if "speed" in want:
        speed, sconf = read_speed(full, geo, calib, templates, threshold)
        conf["speed"] = sconf

    return WorldState(
        date=date,
        paused=paused,
        speed=speed,
        open_panel=panel,
        free_civ_slots=free_civ,
        idle_research_slots=idle_research,
        construction_queue_len=queue_len,
        event_popup=popup,
        pause_menu=pause_menu,
        confidence=conf,
        captured_at=time.time(),
    )
