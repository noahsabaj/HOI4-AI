"""Cache of ROI template images (grayscale float32), loaded from PNGs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..errors import TemplateMissingError
from .ncc import to_gray_f32


class TemplateStore:
    def __init__(self, templates: dict[str, np.ndarray] | None = None) -> None:
        self._t: dict[str, np.ndarray] = dict(templates or {})

    def add(self, name: str, img) -> None:
        self._t[name] = to_gray_f32(img)

    def get(self, name: str) -> np.ndarray:
        if name not in self._t:
            raise TemplateMissingError(name)
        return self._t[name]

    def has(self, name: str) -> bool:
        return name in self._t

    def names(self) -> list[str]:
        return sorted(self._t)

    @classmethod
    def load_dir(cls, path: str | Path) -> "TemplateStore":
        store = cls()
        p = Path(path)
        if p.is_dir():
            for f in sorted(p.glob("*.png")):
                store.add(f.stem, Image.open(f))
        return store
