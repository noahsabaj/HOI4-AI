"""EXPERIMENTAL: derive a draft calibration from HOI4's own .gui interface files.

The game's UI layout is data (``interface/*.gui`` in the install folder), so ROI
rectangles can be computed instead of hand-hovered. This tool parses .gui files,
lists every positioned element, and emits a DRAFT calibration for the elements a
curated map recognizes. The draft MUST be hand-verified with ``calibrate`` —
anchoring/orientation rules in the .gui format are only partially handled
(right/bottom-anchored elements with negative coordinates are skipped and
reported).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import clausewitz
from .calibration import ROI_NAMES, Calibration
from .enums import MapMode

# Curated: HOI4 .gui element name -> our ROI name. Best-guess seed — VERIFY and
# extend against the actual interface files of the installed game version.
ELEMENT_MAP: dict[str, str] = {
    "DateText": "date",
    "topbar_date": "date",
    "date_text": "date",
    "speed_indicator": "speed",
    "topbar_speed": "speed",
    "pause_button": "pause",
    "topbar_pause": "pause",
}


@dataclass(frozen=True)
class GuiElement:
    name: str
    x: int
    y: int
    width: int | None
    height: int | None
    source: str  # file it came from


def _as_int(v) -> int | None:
    return v if isinstance(v, int) else None


def collect_elements(data, source: str) -> list[GuiElement]:
    """Walk parsed .gui data; every dict with a name and an x/y position is an element."""
    out: list[GuiElement] = []

    def _walk(node) -> None:
        if isinstance(node, dict):
            name = node.get("name")
            pos = node.get("position")
            if isinstance(name, str) and isinstance(pos, dict):
                x, y = _as_int(pos.get("x")), _as_int(pos.get("y"))
                raw_size = node.get("size")
                size: dict = raw_size if isinstance(raw_size, dict) else {}
                w = _as_int(size.get("width")) or _as_int(size.get("x"))
                h = _as_int(size.get("height")) or _as_int(size.get("y"))
                if x is not None and y is not None:
                    out.append(GuiElement(name, x, y, w, h, source))
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(data)
    return out


def load_elements(path: str | Path) -> list[GuiElement]:
    """Elements from one .gui file or every ``*.gui`` under a directory."""
    p = Path(path)
    files = sorted(p.glob("**/*.gui")) if p.is_dir() else [p]
    elements: list[GuiElement] = []
    for f in files:
        elements.extend(collect_elements(clausewitz.parse_file(f), f.name))
    return elements


def draft_calibration(
    elements: list[GuiElement], width: int, height: int
) -> tuple[Calibration, dict]:
    """Map recognized elements to ROI fractions; report everything else.

    Report keys: ``mapped`` (roi -> element), ``skipped`` (recognized but
    unusable: negative/anchored coords or missing size), ``unmapped_rois``
    (ROI names still needing manual calibration).
    """
    rois: dict[str, tuple[float, float, float, float]] = {}
    mapped: dict[str, str] = {}
    skipped: list[str] = []
    for el in elements:
        roi = ELEMENT_MAP.get(el.name)
        if roi is None or roi in rois:
            continue
        if el.x < 0 or el.y < 0 or not el.width or not el.height:
            skipped.append(f"{el.name} ({el.source}): anchored/negative coords or no size")
            continue
        rois[roi] = (
            round(el.x / width, 4),
            round(el.y / height, 4),
            round((el.x + el.width) / width, 4),
            round((el.y + el.height) / height, 4),
        )
        mapped[roi] = f"{el.name} ({el.source})"
    calib = Calibration(width=width, height=height, rois=rois,
                        home_map_mode=MapMode.DEFAULT.value)
    report = {
        "mapped": mapped,
        "skipped": skipped,
        "unmapped_rois": [n for n in ROI_NAMES if n not in rois],
    }
    return calib, report
