"""Human demonstration recorder: input hook + frames, and the executor run in reverse.

Two halves:

- ``InputHook``: a Windows low-level mouse/keyboard hook (pure ctypes, WH_MOUSE_LL /
  WH_KEYBOARD_LL on its own message-loop thread) that timestamps left clicks, right clicks and
  key presses with ``time.monotonic_ns``, the clock observations use. UNVERIFIED: written
  from the Win32 documentation and never run against the game in this build. It only
  observes input; it injects and blocks nothing.
- ``invert_events``: pure and tested. It replays the executor's mapping backwards: a left
  click on an own counter selects, a right click near a province centre is MOVE (ctrl held:
  SUPPORT_ATTACK), the halt key is CANCEL, the pause key PAUSE, speed keys or speed-step
  clicks SET_SPEED. Each order is aligned with the last observation captured BEFORE the input
  that completed it, which is the state the human was looking at.

Orders recovered here are demonstrations of intent. Box selection, shift-queued paths,
battle-plan drawing and multi-province paths are not represented and are skipped; a right
click on a non-adjacent province yields an order outside ``actions.choices`` that the caller
must filter with ``choice_index``.
"""
from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass
from typing import Any

from ...geometry import WindowGeometry
from ..contracts import Order, PlayerObservation, Verb
from ..layout import ArenaLayout
from .calibration import ArenaVisionCalibration
from .observe import UnitHandle

WH_KEYBOARD_LL, WH_MOUSE_LL = 13, 14
WM_KEYDOWN, WM_SYSKEYDOWN, WM_LBUTTONDOWN, WM_RBUTTONDOWN, WM_QUIT = 0x0100, 0x0104, 0x0201, 0x0204, 0x0012
VK_CONTROL, VK_SHIFT = 0x11, 0x10
VK_NAMES = {0x20: "space", 0x48: "h", 0x6B: "+", 0x6D: "-", 0xBB: "+", 0xBD: "-", 0x1B: "escape",
            **{0x30 + n: str(n) for n in range(10)}}


@dataclass(frozen=True)
class InputEvent:
    t_ns: int
    kind: str  # "left" | "right" | "key"
    x: float = 0.0  # client pixels
    y: float = 0.0
    key: str = ""
    ctrl: bool = False
    shift: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecordedFrame:
    observation: PlayerObservation
    handles: dict[int, UnitHandle]


class _MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("pt", wintypes.POINT), ("mouseData", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_void_p)]


class _KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_void_p)]


class InputHook:  # pragma: no cover - needs a Windows desktop session
    """Collects ``InputEvent``s in client pixels of ``geo`` until ``stop()``. UNVERIFIED live."""

    def __init__(self, geo: WindowGeometry) -> None:
        self.geo = geo
        self.events: list[InputEvent] = []
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._callbacks: list[Any] = []  # keep the C callbacks alive

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="hoi4-input-hook", daemon=True)
        self._thread.start()

    def stop(self) -> list[InputEvent]:
        if self._thread is not None and self._thread_id:
            ctypes.WinDLL("user32").PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(timeout=2.0)
        return list(self.events)

    def _loop(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._thread_id = kernel32.GetCurrentThreadId()
        prototype = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        user32.CallNextHookEx.restype = ctypes.c_ssize_t
        user32.SetWindowsHookExW.argtypes = (ctypes.c_int, prototype, wintypes.HINSTANCE, wintypes.DWORD)
        user32.SetWindowsHookExW.restype = wintypes.HHOOK

        def held(vk: int) -> bool:
            return bool(user32.GetAsyncKeyState(vk) & 0x8000)

        def mouse(code: int, wparam: int, lparam: int) -> int:
            if code >= 0 and wparam in (WM_LBUTTONDOWN, WM_RBUTTONDOWN):
                info = ctypes.cast(lparam, ctypes.POINTER(_MSLLHOOKSTRUCT)).contents
                self.events.append(InputEvent(
                    time.monotonic_ns(), "left" if wparam == WM_LBUTTONDOWN else "right",
                    float(info.pt.x - self.geo.screen_left), float(info.pt.y - self.geo.screen_top),
                    ctrl=held(VK_CONTROL), shift=held(VK_SHIFT)))
            return int(user32.CallNextHookEx(None, code, wparam, lparam))

        def keyboard(code: int, wparam: int, lparam: int) -> int:
            if code >= 0 and wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                info = ctypes.cast(lparam, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
                name = VK_NAMES.get(int(info.vkCode))
                if name:
                    self.events.append(InputEvent(time.monotonic_ns(), "key", key=name,
                                                  ctrl=held(VK_CONTROL), shift=held(VK_SHIFT)))
            return int(user32.CallNextHookEx(None, code, wparam, lparam))

        self._callbacks = [prototype(mouse), prototype(keyboard)]
        hooks = [user32.SetWindowsHookExW(WH_MOUSE_LL, self._callbacks[0], None, 0),
                 user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._callbacks[1], None, 0)]
        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            pass
        for hook in hooks:
            if hook:
                user32.UnhookWindowsHookEx(hook)


def _frame_before(frames: list[RecordedFrame], t_ns: int) -> RecordedFrame | None:
    earlier = [f for f in frames if f.observation.captured_monotonic_ns <= t_ns]
    return max(earlier, key=lambda f: f.observation.captured_monotonic_ns) if earlier else None


def invert_events(events: list[InputEvent], frames: list[RecordedFrame], layout: ArenaLayout,
                  calibration: ArenaVisionCalibration, size: tuple[int, int] | None = None) -> list[Order]:
    """Recover ``Order``s from raw input. ``size`` is the client size (default: the calibration's)."""
    width, height = size or (calibration.width, calibration.height)
    style, keys = calibration.counter_style, calibration.hotkeys
    reach_x, reach_y = 0.75 * style.frame_w * style.scale, 0.9 * style.frame_h * style.scale
    centres = {p.id: calibration.layout_to_pixel(p.x, p.y, width, height) for p in layout.provinces}
    speed_points = {s: calibration.point(f"speed_{s}") for s in range(1, 6)}
    orders: list[Order] = []
    selected: int | None = None

    def emit(frame: RecordedFrame, verb: Verb, units: tuple[int, ...] = (), target: int | None = None,
             speed: int | None = None) -> None:
        o = frame.observation
        orders.append(Order(f"demo-{len(orders):06d}", o.episode_id, o.sequence, o.country, verb, units,
                            target, speed))

    for event in sorted(events, key=lambda e: e.t_ns):
        frame = _frame_before(frames, event.t_ns)
        if frame is None or frame.observation.terminal:
            continue
        if event.kind == "left":
            step = next((s for s, p in speed_points.items() if p is not None
                         and abs(p[0] / 1000 * width - event.x) <= 12
                         and abs(p[1] / 1000 * height - event.y) <= 12), None)
            if step is not None:
                emit(frame, Verb.SET_SPEED, speed=step)
                continue
            hit = [h for h in frame.handles.values()
                   if abs(h.pixel[0] - event.x) <= reach_x and abs(h.pixel[1] - event.y) <= reach_y]
            nearest = min(hit, key=lambda h: (h.pixel[0] - event.x) ** 2 + (h.pixel[1] - event.y) ** 2,
                          default=None)
            selected = nearest.unit_id if nearest is not None else None
        elif event.kind == "right":
            if selected is None or selected not in frame.handles:
                continue
            province, (cx, cy) = min(centres.items(), key=lambda item: (item[1][0] - event.x) ** 2 +
                                     (item[1][1] - event.y) ** 2)
            if ((cx - event.x) ** 2 + (cy - event.y) ** 2) ** 0.5 > calibration.max_assign_distance * width:
                continue
            if province != frame.handles[selected].province_id:
                emit(frame, Verb.SUPPORT_ATTACK if event.ctrl else Verb.MOVE, (selected,), province)
        elif event.key == keys.halt and selected is not None and selected in frame.handles:
            emit(frame, Verb.CANCEL, (selected,))
        elif event.key == keys.pause:
            emit(frame, Verb.PAUSE)
        elif event.key in (keys.speed_up, keys.speed_down):
            speed = frame.observation.game_speed + (1 if event.key == keys.speed_up else -1)
            if 1 <= speed <= 5:
                emit(frame, Verb.SET_SPEED, speed=speed)
    return orders
