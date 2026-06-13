"""Offline replay: re-run saved trace frames through a new prompt/model/probe.

Because actions are typed and frames are saved, you can re-ask any question of the
same pixels without the game running — the core of iterating like science.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

from ..trace.writer import JsonlTraceWriter


def _default_loader(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def replay(
    trace_path: str | Path,
    probe: Callable[[Image.Image], object],
    *,
    loader: Callable[[str], Image.Image] = _default_loader,
) -> list[dict]:
    """Apply ``probe`` to each record's pre-screenshot; collect results.

    ``probe`` is typically ``lambda img: brain.read_date(img)`` or a which_state
    call — whatever you want to re-measure on the saved frames.
    """
    out: list[dict] = []
    for r in JsonlTraceWriter.read(trace_path):
        if not r.pre_screenshot:
            continue
        try:
            img = loader(r.pre_screenshot)
        except Exception as e:  # missing/unreadable frame
            out.append({"cycle": r.cycle, "plan_step": r.plan_step, "error": str(e)})
            continue
        out.append(
            {
                "cycle": r.cycle,
                "plan_step": r.plan_step,
                "original": r.parsed_intent,
                "replayed": probe(img),
            }
        )
    return out
