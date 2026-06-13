"""JSONL trace writer + reader, with optional per-cycle screenshot saving."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ..schemas import TraceRecord


class JsonlTraceWriter:
    def __init__(self, path: str | Path, screenshot_dir: str | Path | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else None
        if self.screenshot_dir:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def append(self, record: TraceRecord) -> None:
        self._f.write(json.dumps(record.to_dict()) + "\n")
        self._f.flush()

    def save_frame(self, img: Image.Image, name: str) -> str | None:
        """Save a screenshot next to the trace; returns the path (or None if disabled)."""
        if not self.screenshot_dir:
            return None
        fp = self.screenshot_dir / name
        img.convert("RGB").save(fp, "JPEG", quality=70)
        return str(fp)

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "JsonlTraceWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def read(path: str | Path) -> list[TraceRecord]:
        records = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(TraceRecord.from_dict(json.loads(line)))
        return records
