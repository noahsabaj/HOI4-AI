"""Labeled-screenshot corpus loader: each ``*.png`` has a sibling ``*.toml`` of labels.

A label file declares a ``task`` and the ``expected`` answer, e.g.::

    task = "read_number"
    field = "free_civ_slots"
    expected = 3

or::

    task = "which_state"
    expected = "ruhr"
    options = ["ruhr", "saxony", "rhineland"]   # optional; defaults to all
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class CorpusItem:
    image_path: Path
    labels: dict

    def load_image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")


def load_corpus(directory: str | Path) -> list[CorpusItem]:
    d = Path(directory)
    items: list[CorpusItem] = []
    if not d.is_dir():
        return items
    for png in sorted(d.glob("*.png")):
        labels: dict = {}
        sidecar = png.with_suffix(".toml")
        if sidecar.is_file():
            with open(sidecar, "rb") as f:
                labels = tomllib.load(f)
        items.append(CorpusItem(png, labels))
    return items
