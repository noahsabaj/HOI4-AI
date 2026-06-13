"""Build a WorldState from one captured frame.

Pure-ish: deterministic T0 + cropping, with T1/T2 reads delegated to an injected
``reader`` (the brain). No dependency on the LLM layer — the controller wires the
reader in. ``None`` fields mean uncertain.

Note: in this first build, numeric reads (date, slot counts) go through the VLM
reader. The seam to swap in a deterministic numpy digit-template reader is exactly
``Reader.read_number`` / ``Reader.read_date`` — replace the implementation, not
this call site.
"""

from __future__ import annotations

import time
from typing import Protocol

from PIL import Image

from ..calibration import Calibration
from ..geometry import WindowGeometry
from ..schemas import GameDate, WorldState
from .templates import TemplateStore
from .tiers import classify_panel, crop_roi, read_pause


class Reader(Protocol):
    def read_number(self, crop: Image.Image, field: str) -> int | None: ...
    def read_date(self, crop: Image.Image) -> GameDate | None: ...


class CaptureLike(Protocol):
    def grab(self, geo: WindowGeometry, crop=None) -> Image.Image: ...


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
) -> WorldState:
    """Capture and interpret the current screen.

    ``read_numbers=False`` skips the (model-backed) T1 reads — used by the
    mostly-asleep poll where only T0 panel/pause state is needed.
    """
    full = capture.grab(geo)
    conf: dict[str, float] = {}

    panel, pconf = classify_panel(full, geo, calib, templates, threshold)
    conf["panel"] = pconf
    paused, pause_conf = read_pause(full, geo, calib, templates, threshold)
    conf["pause"] = pause_conf

    date = None
    free_civ = idle_research = queue_len = speed = None
    if read_numbers:
        if "date" in calib.rois:
            date = reader.read_date(crop_roi(full, geo, calib.roi("date")))
        free_civ = _read_num(full, geo, calib, reader, "free_civ_slots", 0, 99)
        idle_research = _read_num(full, geo, calib, reader, "idle_research_slots", 0, 9)
        queue_len = _read_num(full, geo, calib, reader, "construction_queue", 0, 99)
        speed = _read_num(full, geo, calib, reader, "speed", 1, 5)

    return WorldState(
        date=date,
        paused=paused,
        speed=speed,
        open_panel=panel,
        free_civ_slots=free_civ,
        idle_research_slots=idle_research,
        construction_queue_len=queue_len,
        event_popup=(panel.name == "EVENT_POPUP"),
        confidence=conf,
        captured_at=time.time(),
    )
