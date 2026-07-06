"""Deterministic glyph reader: the digit-template tier the design promised.

HOI4 renders UI numerals in a fixed font at the locked resolution, so reading a
number is: segment the crop into glyphs by column projection, NCC each glyph
against stored ``glyph_*`` templates (captured by ``calibrate``), and assemble.
Any unknown glyph makes the whole read ``None`` (uncertain, never a guess).

``FallbackReader`` composes this with the VLM reader per call, so live bring-up
works before glyph templates exist — preflight warns while the fallback is active.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..schemas import GameDate
from .ncc import match_resized, to_gray_f32
from .templates import TemplateStore

GLYPH_PREFIX = "glyph_"
# template file names use words for characters that can't appear in filenames
_GLYPH_CHAR_NAMES = {"dot": "."}
_FG_DELTA = 40.0  # gray-level deviation from background median that counts as ink


def _foreground_mask(gray: np.ndarray, delta: float = _FG_DELTA) -> np.ndarray:
    """Boolean ink mask: pixels deviating from the (background) median."""
    bg = float(np.median(gray))
    return np.abs(gray - bg) > delta


def glyph_boxes(gray: np.ndarray, delta: float = _FG_DELTA) -> list[tuple[int, int, int, int]]:
    """Segment a text strip into per-glyph boxes (x0, y0, x1, y1).

    Column projection: runs of columns containing ink are glyphs; each run is
    then trimmed to its own row extent so NCC compares tight crops.
    """
    mask = _foreground_mask(gray, delta)
    cols: list[bool] = np.asarray(mask.any(axis=0)).tolist()
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for x, on in enumerate(cols):
        if on and start is None:
            start = x
        elif not on and start is not None:
            spans.append((start, x))
            start = None
    if start is not None:
        spans.append((start, len(cols)))
    out: list[tuple[int, int, int, int]] = []
    for x0, x1 in spans:
        rows = mask[:, x0:x1].any(axis=1)
        ys = np.nonzero(rows)[0]
        if ys.size:
            out.append((x0, int(ys[0]), x1, int(ys[-1]) + 1))
    return out


def _char_for(template_name: str) -> str:
    suffix = template_name[len(GLYPH_PREFIX):]
    return _GLYPH_CHAR_NAMES.get(suffix, suffix)


class GlyphReader:
    """Implements the perception ``Reader`` protocol deterministically."""

    def __init__(self, templates: TemplateStore, threshold: float = 0.75) -> None:
        self.templates = templates
        self.threshold = threshold

    def _glyph_names(self) -> list[str]:
        return [n for n in self.templates.names() if n.startswith(GLYPH_PREFIX)]

    def available(self) -> bool:
        return bool(self._glyph_names())

    def _classify(self, glyph_img: Image.Image) -> str | None:
        best_char, best_score = None, -1.0
        for name in self._glyph_names():
            score = match_resized(glyph_img, self.templates.get(name))
            if score > best_score:
                best_char, best_score = _char_for(name), score
        return best_char if best_score >= self.threshold else None

    def read_text(self, crop: Image.Image) -> str | None:
        """All glyphs left-to-right, or None if any glyph is unknown."""
        gray = to_gray_f32(crop)
        boxes = glyph_boxes(gray)
        if not boxes:
            return None
        chars = []
        for x0, y0, x1, y1 in boxes:
            c = self._classify(crop.crop((x0, y0, x1, y1)))
            if c is None:
                return None
            chars.append(c)
        return "".join(chars)

    # --- Reader protocol ---
    def read_number(self, crop: Image.Image, field: str) -> int | None:
        s = self.read_text(crop)
        if not s or not s.isdigit():
            return None
        return int(s)

    def read_date(self, crop: Image.Image) -> GameDate | None:
        s = self.read_text(crop)
        if s is None:
            return None
        parts = [p for p in s.split(".") if p]
        if len(parts) != 3:
            return None
        try:
            y, m, d = (int(p) for p in parts)
            return GameDate(y, m, d)  # range-checked; bad month/day -> ValueError
        except ValueError:
            return None


class FallbackReader:
    """Glyphs first, VLM second — each read falls back independently."""

    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback

    def read_number(self, crop: Image.Image, field: str) -> int | None:
        if self.primary is not None:
            v = self.primary.read_number(crop, field)
            if v is not None:
                return v
        if self.fallback is not None:
            return self.fallback.read_number(crop, field)
        return None

    def read_date(self, crop: Image.Image) -> GameDate | None:
        if self.primary is not None:
            v = self.primary.read_date(crop)
            if v is not None:
                return v
        if self.fallback is not None:
            return self.fallback.read_date(crop)
        return None
