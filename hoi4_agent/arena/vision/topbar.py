"""Read pause state and game speed from the top bar, with no templates.

Measured on live 1920x1080 frames of this build; every constant is a fraction of the client
width/height, so the same numbers serve other resolutions until proven otherwise.

  play/pause glyph   a green triangle when running, two yellow bars when paused, told apart
                     by green minus red: about +68 running, about -30 paused
  speed bar          five segments left of the date; the lit ones are bright green

Both reads return None when the evidence is weak rather than guessing, because the caller
uses them to decide whether to press space, and a wrong answer there stalls a whole match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# Fractions of the client area, measured from live frames (1920x1080 -> the pixel boxes in comments).
GLYPH_BOX = (0.8474, 0.0120, 0.8552, 0.0250)   # (1627, 13) - (1642, 27)
SPEED_BOX = (0.8526, 0.0305, 0.9182, 0.0380)   # (1637, 33) - (1763, 41)
SEGMENTS = 5
LIT_MIN_GREEN = 90.0     # a lit segment's mean green channel; unlit segments sit near 60
LIT_MIN_MARGIN = 18.0    # lit minus unlit must separate this clearly, or the read is uncertain


@dataclass(frozen=True)
class TopBar:
    paused: bool | None
    speed: int | None
    glyph_green: float
    glyph_yellow: float
    segment_green: tuple[float, ...]


def _box(frame: Image.Image, box: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = frame.crop((int(x0 * frame.width), int(y0 * frame.height),
                       int(x1 * frame.width), int(y1 * frame.height)))
    return np.asarray(crop.convert("RGB"), dtype=np.float32)


def read_top_bar(frame: Image.Image) -> TopBar:
    glyph = _box(frame, GLYPH_BOX)
    # The play triangle is green (R well below G); the pause bars are yellow (R close to G).
    bright = glyph[glyph.max(axis=2) > 110]
    green = float((bright[:, 1] - bright[:, 0]).mean()) if len(bright) else 0.0
    yellow = float((bright[:, 1] - bright[:, 2]).mean()) if len(bright) else 0.0
    paused: bool | None = None
    if len(bright) >= 8 and yellow > 15:  # both glyphs are far brighter in green than in blue
        if green > 30:
            paused = False
        elif green < -10:
            paused = True
    bar = _box(frame, SPEED_BOX)
    width = bar.shape[1] // SEGMENTS
    greens = tuple(float(bar[:, i * width:(i + 1) * width, 1].mean()) for i in range(SEGMENTS))
    lit = [g >= LIT_MIN_GREEN for g in greens]
    speed: int | None = None
    if lit[0] and all(lit[i] >= lit[i + 1] for i in range(SEGMENTS - 1)):  # a prefix of lit segments
        count = sum(lit)
        rest = greens[count:]
        if count == SEGMENTS or min(greens[:count]) - max(rest) >= LIT_MIN_MARGIN:
            speed = count
    return TopBar(paused, speed, green, yellow, greens)
