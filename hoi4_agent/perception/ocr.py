"""Optional OCR middle tier for numeric reads: glyphs -> OCR -> VLM.

Uses rapidocr-onnxruntime when installed (``pip install -e ".[ocr]"``); the core
package deliberately does not depend on it. The engine is injected/testable and
lazily constructed. Anything ambiguous reads as ``None`` (uncertain) — this tier
must never guess, exactly like the glyph reader above it.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

from ..schemas import GameDate

_DATE_RE = re.compile(r"(\d{4})\D{1,3}(\d{1,2})\D{1,3}(\d{1,2})")
_DIGITS_RE = re.compile(r"\d+")


def _load_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    return RapidOCR()


class OcrReader:
    """Implements the perception ``Reader`` protocol via general OCR.

    ``engine`` is a callable ``engine(np.ndarray) -> (results, elapse)`` where
    ``results`` is a list of ``(box, text, score)`` or None — rapidocr's shape.
    """

    def __init__(self, engine=None) -> None:
        self._engine = engine
        self._resolved = engine is not None

    def _get_engine(self):
        if not self._resolved:
            self._resolved = True
            self._engine = _load_engine()
        return self._engine

    def available(self) -> bool:
        return self._get_engine() is not None

    def _read_text(self, crop: Image.Image) -> str | None:
        engine = self._get_engine()
        if engine is None:
            return None
        results, _elapse = engine(np.asarray(crop.convert("RGB")))
        if not results:
            return None
        return " ".join(str(text) for _box, text, _score in results)

    # --- Reader protocol ---
    def read_number(self, crop: Image.Image, field: str) -> int | None:
        text = self._read_text(crop)
        if text is None:
            return None
        groups = _DIGITS_RE.findall(text)
        if len(groups) != 1:  # zero or several digit runs: ambiguous, stay uncertain
            return None
        return int(groups[0])

    def read_date(self, crop: Image.Image) -> GameDate | None:
        text = self._read_text(crop)
        if text is None:
            return None
        m = _DATE_RE.search(text)
        if m is None:
            return None
        try:
            return GameDate(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
