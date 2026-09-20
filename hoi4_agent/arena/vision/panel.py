"""Read the army panel's division list: one row per own division at a fixed pitch.

UNVERIFIED. Designed for the vanilla unit list but never checked against it, which is why
every coordinate lives in ``PanelCalibration`` and the reader is off until calibrated
(``enabled = false``). Known open points for the live game: the list only shows while the
army is selected, so reading it costs a click on ``army_select``; whether row bars are lit
the same way as counter bars; and what a selected row looks like (``selected_probe`` is a
brightness probe at the row's left edge, a guess).

Rows carry no province, so the panel alone cannot place a division on the map; ``observe``
joins rows to stacks through its own-division tracker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .calibration import PanelCalibration, Rect


@dataclass(frozen=True)
class PanelRow:
    index: int  # 0-based row; division identity is its list position
    organization: float | None
    strength: float | None
    selected: bool
    confidence: float


def row_rect(panel: PanelCalibration, index: int) -> Rect:
    x0, y0, x1, y1 = panel.first_row
    return x0, y0 + index * panel.row_pitch, x1, y1 + index * panel.row_pitch


def row_point(panel: PanelCalibration, index: int) -> tuple[int, int]:
    """Click point (0..1000 over the client) at the middle of a row."""
    x0, y0, x1, y1 = row_rect(panel, index)
    return int(round((x0 + x1) / 2 * 1000)), int(round((y0 + y1) / 2 * 1000))


def _sub(row: np.ndarray, box: Rect) -> np.ndarray:
    height, width = row.shape[:2]
    x0, y0 = int(round(box[0] * width)), int(round(box[1] * height))
    x1, y1 = max(x0 + 1, int(round(box[2] * width))), max(y0 + 1, int(round(box[3] * height)))
    return row[y0:y1, x0:x1]


def _fill(region: np.ndarray, lit_value: float) -> float | None:
    if region.size == 0:
        return None
    lit = region.max(axis=2).mean(axis=0) >= lit_value
    dark = np.nonzero(~lit)[0]
    return float((int(dark[0]) if dark.size else lit.size) / lit.size)


def read_panel(frame: Image.Image, panel: PanelCalibration) -> tuple[PanelRow, ...]:
    """Rows from the top until the first empty one. Empty tuple when disabled or unreadable."""
    if not panel.enabled:
        return ()
    rgb = np.asarray(frame.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    rows: list[PanelRow] = []
    for index in range(panel.max_rows):
        x0, y0, x1, y1 = row_rect(panel, index)
        px0, py0 = int(x0 * width), int(y0 * height)
        px1, py1 = int(np.ceil(x1 * width)), int(np.ceil(y1 * height))
        if py1 > height or px1 > width or px1 - px0 < 4 or py1 - py0 < 3:
            break
        row = rgb[py0:py1, px0:px1]
        spread = float(row.std())
        if spread < panel.row_present_std:
            break  # flat background: the list ended
        selected = float(_sub(row, panel.selected_probe).max(axis=2).mean()) >= panel.selected_value
        rows.append(PanelRow(index, _fill(_sub(row, panel.org_bar), panel.lit_value),
                             _fill(_sub(row, panel.strength_bar), panel.lit_value), selected,
                             round(min(1.0, spread / (4 * panel.row_present_std)), 3)))
    return tuple(rows)
