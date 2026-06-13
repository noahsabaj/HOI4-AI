"""Windows-native I/O via pure ctypes (user32) + mss.

One measured client rect (GetClientRect + ClientToScreen) feeds BOTH capture and
clicks, so the v3 title-bar/frame drift cannot recur. Keys are sent as hardware
SCANCODES (DirectX games like HOI4 ignore VK/WM events); the mouse uses absolute
normalized coordinates over the virtual desktop. DPI awareness is set first so
1440p coordinates aren't silently rescaled.

Import-safe on non-Windows (the DLL load is guarded); calling into it off-Windows
raises a clear RuntimeError.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import mss
from PIL import Image

from ..errors import AgentError
from ..geometry import NORM_SCALE, CropRect, WindowGeometry

# --- DLL handles (guarded so the module imports anywhere) -------------------
try:
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _AVAILABLE = True
except (OSError, AttributeError):  # pragma: no cover - non-Windows
    _user32 = None
    _AVAILABLE = False


def available() -> bool:
    return _AVAILABLE


# --- SendInput structures ---------------------------------------------------
ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_EXTENDEDKEY = 0x0001
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
SW_RESTORE = 9


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


# Set-1 hardware scancodes. Layout-independent — why DX games read them.
SCANCODES: dict[str, tuple[int, bool]] = {
    "escape": (0x01, False),
    "1": (0x02, False), "2": (0x03, False), "3": (0x04, False), "4": (0x05, False),
    "5": (0x06, False), "6": (0x07, False), "7": (0x08, False), "8": (0x09, False),
    "9": (0x0A, False), "0": (0x0B, False),
    "minus": (0x0C, False), "equal": (0x0D, False),
    "tab": (0x0F, False), "enter": (0x1C, False), "space": (0x39, False),
    "q": (0x10, False), "w": (0x11, False), "e": (0x12, False), "r": (0x13, False),
    "t": (0x14, False), "y": (0x15, False), "u": (0x16, False), "i": (0x17, False),
    "o": (0x18, False), "p": (0x19, False),
    "a": (0x1E, False), "s": (0x1F, False), "d": (0x20, False), "f": (0x21, False),
    "g": (0x22, False), "h": (0x23, False), "j": (0x24, False), "k": (0x25, False),
    "l": (0x26, False),
    "z": (0x2C, False), "x": (0x2D, False), "c": (0x2E, False), "v": (0x2F, False),
    "b": (0x30, False), "n": (0x31, False), "m": (0x32, False),
    "f1": (0x3B, False), "f2": (0x3C, False), "f3": (0x3D, False), "f4": (0x3E, False),
    "f5": (0x3F, False), "f6": (0x40, False), "f7": (0x41, False), "f8": (0x42, False),
    "f9": (0x43, False), "f10": (0x44, False),
    # Numpad +/- for game speed (no Shift needed, unlike main-row '+').
    "+": (0x4E, False), "add": (0x4E, False),
    "-": (0x4A, False), "subtract": (0x4A, False),
}

_ALIASES = {"esc": "escape", "return": "enter", "spacebar": "space"}


def _normalize_key(name: str) -> str:
    n = name.strip().lower()
    n = _ALIASES.get(n, n)
    if len(n) == 1 and n.isalpha():  # avoid Shift+letter from an uppercase char
        return n
    return n


def _require() -> None:
    if not _AVAILABLE:
        raise RuntimeError("Windows I/O unavailable on this platform")


def ensure_dpi_aware() -> str:
    """Make the process per-monitor DPI-aware. Call before any geometry query."""
    _require()
    # PER_MONITOR_AWARE_V2 = -4
    try:
        if _user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per_monitor_v2"
    except (AttributeError, OSError):
        pass
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)  # PER_MONITOR
        return "per_monitor"
    except (AttributeError, OSError):
        pass
    try:
        _user32.SetProcessDPIAware()
        return "system"
    except (AttributeError, OSError):  # pragma: no cover
        return "none"


# --- window location --------------------------------------------------------
_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class Win32Locator:
    def find(self, title_substr: str) -> WindowGeometry | None:
        _require()
        hwnds: list[int] = []

        def _cb(hwnd, _lparam):
            if not _user32.IsWindowVisible(hwnd):
                return True
            length = _user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            _user32.GetWindowTextW(hwnd, buf, length + 1)
            if title_substr.lower() in buf.value.lower():
                hwnds.append(hwnd)
            return True

        _user32.EnumWindows(_WNDENUMPROC(_cb), 0)
        if not hwnds:
            return None
        hwnd = hwnds[0]

        rect = wintypes.RECT()
        if not _user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        pt = wintypes.POINT(0, 0)
        if not _user32.ClientToScreen(hwnd, ctypes.byref(pt)):
            return None
        return WindowGeometry(
            hwnd=int(hwnd),
            screen_left=int(pt.x),
            screen_top=int(pt.y),
            client_w=int(rect.right),
            client_h=int(rect.bottom),
        )


# --- capture ----------------------------------------------------------------
class MssCapture:
    def __init__(self) -> None:
        self._sct = mss.mss()

    def grab(self, geo: WindowGeometry, crop: CropRect | None = None) -> Image.Image:
        raw = self._sct.grab(geo.monitor(crop))
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


# --- input ------------------------------------------------------------------
class Win32Input:
    def __init__(self, dwell_ms: int = 40) -> None:
        self.dwell_s = dwell_ms / 1000.0

    def focus(self, geo: WindowGeometry) -> bool:
        _require()
        hwnd = geo.hwnd
        if _user32.IsIconic(hwnd):
            _user32.ShowWindow(hwnd, SW_RESTORE)
        _user32.SetForegroundWindow(hwnd)
        if _user32.GetForegroundWindow() == hwnd:
            return True
        # Foreground-lock workaround: attach to the target's input thread.
        target_tid = _user32.GetWindowThreadProcessId(hwnd, None)
        our_tid = ctypes.windll.kernel32.GetCurrentThreadId()
        _user32.AttachThreadInput(our_tid, target_tid, True)
        try:
            _user32.BringWindowToTop(hwnd)
            _user32.SetForegroundWindow(hwnd)
        finally:
            _user32.AttachThreadInput(our_tid, target_tid, False)
        return _user32.GetForegroundWindow() == hwnd

    def key(self, name: str) -> None:
        _require()
        sc = SCANCODES.get(_normalize_key(name))
        if sc is None:
            raise AgentError(f"no scancode for key {name!r}")
        scan, extended = sc
        base = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if extended else 0)
        self._send([_kbd(scan, base), _kbd(scan, base | KEYEVENTF_KEYUP)])

    def click(self, geo: WindowGeometry, crop: CropRect, nx: int, ny: int) -> None:
        _require()
        sx, sy = geo.norm_to_screen(crop, nx, ny)
        ax, ay = _to_virtual_abs(sx, sy)
        self._send([_mouse(ax, ay, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK)])
        if self.dwell_s:
            time.sleep(self.dwell_s)
        self._send([
            _mouse(ax, ay, MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK),
            _mouse(ax, ay, MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK),
        ])

    @staticmethod
    def get_cursor_pos() -> tuple[int, int]:
        _require()
        pt = wintypes.POINT()
        _user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)

    @staticmethod
    def _send(inputs: list[INPUT]) -> None:
        n = len(inputs)
        arr = (INPUT * n)(*inputs)
        sent = _user32.SendInput(n, arr, ctypes.sizeof(INPUT))
        if sent != n:  # pragma: no cover
            raise AgentError(f"SendInput injected {sent}/{n} events")


def _kbd(scan: int, flags: int) -> INPUT:
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.u.ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0)
    return inp


def _mouse(ax: int, ay: int, flags: int) -> INPUT:
    inp = INPUT(type=INPUT_MOUSE)
    inp.u.mi = MOUSEINPUT(dx=ax, dy=ay, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0)
    return inp


def _to_virtual_abs(sx: int, sy: int) -> tuple[int, int]:
    """Screen pixel -> 0..65535 normalized over the whole virtual desktop."""
    vx = _user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = _user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    vw = _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    vh = _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    ax = int(round((sx - vx) * 65535 / max(1, vw - 1)))
    ay = int(round((sy - vy) * 65535 / max(1, vh - 1)))
    return ax, ay


def build_io(dwell_ms: int = 40) -> tuple[Win32Locator, MssCapture, Win32Input]:
    """Set DPI awareness and return (locator, capture, input) for the live agent."""
    ensure_dpi_aware()
    return Win32Locator(), MssCapture(), Win32Input(dwell_ms=dwell_ms)
