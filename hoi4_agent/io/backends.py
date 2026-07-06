"""I/O protocols + reusable fakes (used by tests and the offline smoke CLI).

Keeping the fakes in the package (not just tests) lets ``smoke-test --offline``
run the real controller end-to-end with zero external services.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image

from ..geometry import CropRect, WindowGeometry


@runtime_checkable
class WindowLocator(Protocol):
    def find(
        self, title_substr: str, expected_client: tuple[int, int] | None = None
    ) -> WindowGeometry | None: ...


@runtime_checkable
class CaptureBackend(Protocol):
    def grab(self, geo: WindowGeometry, crop: CropRect | None = None) -> Image.Image: ...


@runtime_checkable
class InputBackend(Protocol):
    def focus(self, geo: WindowGeometry) -> bool: ...
    def key(self, name: str) -> None: ...
    def click(self, geo: WindowGeometry, crop: CropRect, nx: int, ny: int) -> None: ...


# --- fakes ------------------------------------------------------------------
class StubLocator:
    """Always returns a fixed geometry (or None)."""

    def __init__(self, geo: WindowGeometry | None) -> None:
        self._geo = geo

    def find(
        self, title_substr: str, expected_client: tuple[int, int] | None = None
    ) -> WindowGeometry | None:
        return self._geo


class RecordingInput:
    """No-op input backend that records every call for assertions."""

    def __init__(self, focus_ok: bool = True) -> None:
        self.calls: list[tuple] = []
        self._focus_ok = focus_ok

    def focus(self, geo: WindowGeometry) -> bool:
        self.calls.append(("focus",))
        return self._focus_ok

    def key(self, name: str) -> None:
        self.calls.append(("key", name))

    def click(self, geo: WindowGeometry, crop: CropRect, nx: int, ny: int) -> None:
        self.calls.append(("click", nx, ny))

    @property
    def keys(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "key"]

    @property
    def clicks(self) -> list[tuple[int, int]]:
        return [(c[1], c[2]) for c in self.calls if c[0] == "click"]


# A null backend is just a recorder you ignore.
NullInput = RecordingInput


class InputRecorder:
    """Wraps a real InputBackend and journals every event for the trace.

    ``drain()`` returns-and-clears, so the controller can attribute the events
    between two drains to one cycle's TraceRecord.
    """

    def __init__(self, inner: InputBackend) -> None:
        self.inner = inner
        self._events: list[dict] = []

    def focus(self, geo: WindowGeometry) -> bool:
        ok = bool(self.inner.focus(geo))
        self._events.append({"kind": "focus", "ok": ok})
        return ok

    def key(self, name: str) -> None:
        self.inner.key(name)
        self._events.append({"kind": "key", "name": name})

    def click(self, geo: WindowGeometry, crop: CropRect, nx: int, ny: int) -> None:
        self.inner.click(geo, crop, nx, ny)
        self._events.append({"kind": "click", "nx": nx, "ny": ny})

    def drain(self) -> list[dict]:
        out, self._events = self._events, []
        return out


class FakeCapture:
    """Returns canned PIL frames in order (the last frame repeats once exhausted).

    If a ``crop`` is supplied, the returned frame is PIL-cropped to it, so cropping
    math can be exercised against synthetic images.
    """

    def __init__(self, frames: "Image.Image | list[Image.Image]") -> None:
        if isinstance(frames, Image.Image):
            frames = [frames]
        if not frames:
            raise ValueError("FakeCapture needs at least one frame")
        self._frames = list(frames)
        self._i = 0

    def grab(self, geo: WindowGeometry, crop: CropRect | None = None) -> Image.Image:
        img = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        if crop is not None:
            box = (
                crop.client_x0,
                crop.client_y0,
                crop.client_x0 + crop.crop_w,
                crop.client_y0 + crop.crop_h,
            )
            img = img.crop(box)
        return img
