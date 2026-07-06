"""M0: score a model's perception/grounding on the labeled corpus, offline.

This is the gate before the live loop: if the chosen model can't read the date or
count slots from a crop, you learn it here in an afternoon, not mid-run.
"""

from __future__ import annotations

from PIL import Image

from ..enums import GermanState, Tech
from .corpus import CorpusItem


def _run_task(brain, img: Image.Image, labels: dict):
    task = labels.get("task")
    expected = labels.get("expected")
    if task == "read_number":
        return brain.read_number(img, labels.get("field", "value")), expected
    if task == "read_date":
        got = brain.read_date(img)
        return (got.to_str() if got else None), expected
    if task == "which_state":
        state_opts = [GermanState(s) for s in labels.get("options", [s.value for s in GermanState])]
        return brain.which_state(img, state_opts).value, expected
    if task == "which_tech":
        tech_opts = [Tech(t) for t in labels.get("options", [t.value for t in Tech])]
        return brain.which_tech(img, tech_opts).value, expected
    if task == "yes_no":
        return ("yes" if brain.yes_no(img, labels.get("question", "?")) else "no"), expected
    if task == "locate":
        return brain.locate(img, labels.get("instruction", "?")), expected
    # load_corpus validates tasks, but guard direct callers: an unknown task
    # must count as WRONG (via the caller's except), never as silently correct.
    raise ValueError(f"unknown corpus task {task!r}")


def _is_correct(task: str, got, expected, labels: dict) -> bool:
    """Locate is scored by pixel distance (0-1000 space); everything else by value."""
    if task == "locate" and isinstance(got, tuple):
        try:
            gx, gy = got
            ex, ey = expected
            tolerance = float(labels.get("tolerance", 25))
        except (TypeError, ValueError):
            return False
        return ((gx - ex) ** 2 + (gy - ey) ** 2) ** 0.5 <= tolerance
    return str(got) == str(expected)


def score_model(corpus: list[CorpusItem], brain) -> dict:
    per_task: dict[str, list[int]] = {}  # task -> [correct, total]
    mismatches: list[dict] = []
    for item in corpus:
        task = item.labels.get("task", "unknown")
        img = item.load_image()
        try:
            got, expected = _run_task(brain, img, item.labels)
        except Exception as e:  # invalid enum, infra error, etc. -> counted wrong
            got, expected = f"ERROR:{type(e).__name__}", item.labels.get("expected")
        ct = per_task.setdefault(task, [0, 0])
        ct[1] += 1
        if _is_correct(task, got, expected, item.labels):
            ct[0] += 1
        else:
            mismatches.append(
                {"image": str(item.image_path), "task": task, "got": got, "expected": expected}
            )

    correct = sum(c for c, _ in per_task.values())
    total = sum(n for _, n in per_task.values())
    return {
        "per_task": {
            t: {"correct": c, "total": n, "accuracy": (c / n if n else 0.0)}
            for t, (c, n) in per_task.items()
        },
        "correct": correct,
        "total": total,
        "accuracy": (correct / total if total else 0.0),
        "mismatches": mismatches,
    }
