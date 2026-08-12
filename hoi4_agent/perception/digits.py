"""Deterministic glyph reader: the character-template tier the design promised.

HOI4 renders UI text in a fixed font at the locked resolution, so reading a
strip is: segment it into glyphs by column projection, NCC each glyph against
stored ``glyph_*`` templates (captured by ``calibrate``), and assemble. Any
unknown glyph makes the whole read ``None`` (uncertain, never a guess).

Templates cover EVERY character the read strips contain, not just digits: the
top-bar date renders as "12:00, 1 Jan, 1936", so a digits-only set could never
classify the colon, commas, or the month NAME — and under the all-or-nothing
rule that made the whole deterministic date read impossible. ``calibrate``
already knows the exact string the operator typed, so it labels letters and
punctuation the same way it labels digits.

``FallbackReader`` composes this with the VLM reader per call, so live bring-up
works before glyph templates exist — preflight warns while the fallback is active.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..schemas import MONTH_ABBREVS, GameDate
from .ncc import match_resized, to_gray_f32
from .templates import TemplateStore

GLYPH_PREFIX = "glyph_"
# Template file names use words for characters that can't appear in filenames.
# Digits keep their bare name so glyph sets captured before letters were
# supported still load unchanged.
_CHAR_SUFFIXES = {".": "dot", ":": "colon", ",": "comma", "/": "slash", "-": "dash"}
_SUFFIX_CHARS = {v: k for k, v in _CHAR_SUFFIXES.items()}
# Letters carry an explicit case prefix: template files must stay unambiguous on
# a case-INSENSITIVE filesystem, where glyph_J.png and glyph_j.png are one file.
_UPPER, _LOWER = "upper_", "lower_"
_FG_DELTA = 40.0  # gray-level deviation from background median that counts as ink
# Anti-aliased text (4K UI) leaves faint ink bridging glyph gaps: at a low delta
# neighbours merge into one box. Readers walk this ladder until every segmented
# glyph classifies; calibrate uses it to find the count that matches the typed
# string. (Verified live 2026-07-06: HOI4 top-bar date needed delta≈115.)
DELTA_LADDER = (40.0, 70.0, 100.0, 130.0)


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


def template_name_for(ch: str) -> str | None:
    """Template file name a character is stored under, or None if unstorable.

    ``calibrate`` uses this to auto-label the glyphs it segments out of a typed
    string; the reader uses ``_char_for`` to invert it.
    """
    if len(ch) != 1 or not ch.isascii():
        return None
    if ch.isdigit():
        return f"{GLYPH_PREFIX}{ch}"
    if ch in _CHAR_SUFFIXES:
        return f"{GLYPH_PREFIX}{_CHAR_SUFFIXES[ch]}"
    if ch.isalpha():
        return f"{GLYPH_PREFIX}{_UPPER if ch.isupper() else _LOWER}{ch.lower()}"
    return None


def _char_for(template_name: str) -> str | None:
    """Character a glyph template stands for, or None if the name is unrecognized.

    An unknown name must NOT fall through to its raw suffix: a stray
    ``glyph_foo.png`` would otherwise inject the three characters "foo" into an
    otherwise-clean read.
    """
    suffix = template_name[len(GLYPH_PREFIX):]
    if len(suffix) == 1 and suffix.isascii() and suffix.isdigit():
        return suffix
    if suffix in _SUFFIX_CHARS:
        return _SUFFIX_CHARS[suffix]
    for prefix, cased in ((_UPPER, str.upper), (_LOWER, str.lower)):
        if suffix.startswith(prefix):
            letter = suffix[len(prefix):]
            if len(letter) == 1 and letter.isascii() and letter.isalpha():
                return cased(letter)
    return None


# Every character the standard top-bar date strip can contain — "12:00, 1 Jan,
# 1936" across all twelve months. Spaces leave no ink, so they never segment out
# and need no template.
DATE_STRIP_CHARS = tuple(sorted(
    set("0123456789:,") | {ch for name in MONTH_ABBREVS for ch in name}
))


def missing_date_glyphs(templates: TemplateStore) -> list[str]:
    """Characters a date read needs that this glyph set cannot classify.

    Advisory, and it assumes the standard top-bar rendering. The reader is
    all-or-nothing, so ANY missing character means the deterministic date read
    can never complete and every poll falls through to OCR or the VLM — which
    is worth saying out loud at startup rather than discovering per poll.
    """
    have = {_char_for(n) for n in templates.names() if n.startswith(GLYPH_PREFIX)}
    return [ch for ch in DATE_STRIP_CHARS if ch not in have]


class GlyphReader:
    """Implements the perception ``Reader`` protocol deterministically."""

    def __init__(self, templates: TemplateStore, threshold: float = 0.75) -> None:
        self.templates = templates
        self.threshold = threshold

    def _glyph_names(self) -> list[str]:
        """Loaded glyph templates whose file name decodes to a character."""
        return [
            n for n in self.templates.names()
            if n.startswith(GLYPH_PREFIX) and _char_for(n) is not None
        ]

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
        """All glyphs left-to-right, or None if any glyph is unknown.

        Walks DELTA_LADDER: a segmentation only counts when EVERY box
        classifies above threshold — merged boxes fail classification, so the
        next (stricter) delta gets a chance instead of a guess.
        """
        gray = to_gray_f32(crop)
        for delta in DELTA_LADDER:
            boxes = glyph_boxes(gray, delta)
            if not boxes:
                continue
            chars = [self._classify(crop.crop(box)) for box in boxes]
            if all(c is not None for c in chars):
                return "".join(c for c in chars if c is not None)
        return None

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
        return GameDate.from_ui_text(s)


class ChainReader:
    """Try each reader in order per call: glyphs -> OCR -> VLM.

    None entries are skipped, so callers can wire optional tiers directly
    (``ChainReader([glyphs_or_none, ocr_or_none, brain_or_none])``).
    """

    def __init__(self, readers) -> None:
        self.readers = [r for r in readers if r is not None]

    def read_number(self, crop: Image.Image, field: str) -> int | None:
        for r in self.readers:
            v = r.read_number(crop, field)
            if v is not None:
                return v
        return None

    def read_date(self, crop: Image.Image) -> GameDate | None:
        for r in self.readers:
            v = r.read_date(crop)
            if v is not None:
                return v
        return None


class FallbackReader(ChainReader):
    """Two-tier chain (primary -> fallback), kept for its self-describing name."""

    def __init__(self, primary, fallback) -> None:
        super().__init__([primary, fallback])
