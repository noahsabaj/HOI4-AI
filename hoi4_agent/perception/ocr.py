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

_DIGITS_RE = re.compile(r"\d+")

# Field-specific shapes. free_civ_slots renders as "38/38  From trade: 2  Owned: 36"
# in the construction header — the wanted value is X of the first X/Y (factories
# currently available / total pool); the generic one-digit-run rule would always
# call this crop ambiguous.
_FIELD_RES = {"free_civ_slots": re.compile(r"(\d+)\s*/\s*\d+")}


def _load_engine():
    """rapidocr v2+/v3 preferred (py3.13+); legacy rapidocr-onnxruntime accepted.

    Construction downloads the PP-OCR models (~20 MB) on first use.
    """
    try:
        from rapidocr import RapidOCR
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            return None
    return RapidOCR()


def _texts_of(raw) -> list[str]:
    """Normalize both rapidocr output generations to a list of strings."""
    txts = getattr(raw, "txts", None)  # v2+/v3: RapidOCROutput.txts
    if txts is not None:
        return [str(t) for t in txts]
    if isinstance(raw, tuple):  # legacy: (results, elapse), results = [(box, text, score)]
        results = raw[0]
        if not results:
            return []
        return [str(text) for _box, text, _score in results]
    return []


class OcrReader:
    """Implements the perception ``Reader`` protocol via general OCR.

    ``engine`` is a callable taking an ``np.ndarray`` image and returning either
    a rapidocr v2+ output object (``.txts``) or the legacy ``(results, elapse)``
    tuple — both are normalized.
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
        # PP-OCR's detector misses text flush against the crop edge (tight ROIs);
        # pad with the crop's own border tone before detection.
        arr = np.asarray(crop.convert("RGB"))
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
        fill = np.median(border, axis=0).astype(arr.dtype)
        pad = 16
        padded = np.full((arr.shape[0] + 2 * pad, arr.shape[1] + 2 * pad, 3), fill, dtype=arr.dtype)
        padded[pad:-pad, pad:-pad] = arr
        texts = _texts_of(engine(padded))
        if not texts:
            return None
        return " ".join(texts)

    # --- Reader protocol ---
    def read_number(self, crop: Image.Image, field: str) -> int | None:
        text = self._read_text(crop)
        if text is None:
            return None
        field_re = _FIELD_RES.get(field)
        if field_re is not None:
            m = field_re.search(text)
            return int(m.group(1)) if m else None
        groups = _DIGITS_RE.findall(text)
        if len(groups) != 1:  # zero or several digit runs: ambiguous, stay uncertain
            return None
        return int(groups[0])

    def read_date(self, crop: Image.Image) -> GameDate | None:
        text = self._read_text(crop)
        if text is None:
            return None
        return GameDate.from_ui_text(text)
