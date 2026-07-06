"""Labeled-screenshot corpus loader: each ``*.png`` has a sibling ``*.toml`` of labels.

The corpus is ground truth for the M0 gate, so it is validated STRICTLY: a PNG
without a sidecar, or a sidecar without a known ``task`` and an ``expected``
value, is a ConfigError — silently tolerating it once let unlabeled images score
as correct and inflate the accuracy that gates the live loop.

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

from ..errors import ConfigError

KNOWN_TASKS = ("read_number", "read_date", "which_state", "which_tech", "yes_no")


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
        sidecar = png.with_suffix(".toml")
        if not sidecar.is_file():
            raise ConfigError(f"corpus image {png.name} has no label sidecar {sidecar.name}")
        with open(sidecar, "rb") as f:
            labels = tomllib.load(f)
        task = labels.get("task")
        if task not in KNOWN_TASKS:
            raise ConfigError(
                f"corpus label {sidecar.name}: task {task!r} not in {list(KNOWN_TASKS)}"
            )
        if "expected" not in labels:
            raise ConfigError(f"corpus label {sidecar.name}: missing 'expected'")
        items.append(CorpusItem(png, labels))
    return items
