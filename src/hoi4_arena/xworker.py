"""The desktop worker for Linux: the Rust worker's stdio protocol over an X display.

`Desktop` talks to a worker process over stdin and stdout, one JSON line per request and
one per reply (a reply's `bytes` of payload follow its line). On Windows that worker is
crates/desktop-worker. On a Linux station it is this module, run by `Desktop` itself
(`worker_command`), so everything built on `Desktop` (record-ai, the scripted player, the
AI-vs-AI camera) runs there unchanged.

    python -m hoi4_arena.xworker [--mods DIR] [--userdir DIR] [--game DIR] [--observer]

It reads the display from DISPLAY and talks to X through ctypes (libX11, libXtst for
input, libXfixes for the pointer), so it needs no new package. What it does:

- capture: the game window's rectangle of the screen (XGetImage), BGRA, the pointer drawn
  in as the Windows worker draws it, and where the pointer is.
- input: XTest events from the demonstration vocabulary, only while armed and while the
  game has the input focus, with the same key rules as the Windows worker (setup may also
  press the console key, space, escape, backspace, and = - .). Held keys and buttons are
  let go when input is released, when the game loses focus, and 750 ms after the last
  input (the watchdog).
- game_log: the arena mod's `ARENA` lines from the game's own game.log, as on Windows.
- launch and quit: HOI4 started straight from its folder with `-userdir` (on Linux the
  game honours it, where on Windows it does not), so the game's settings, logs, mods and
  saves stay in one folder of this project, and nothing is written to the player's own.
  The arena mod is written into that folder's mod list at each launch.

What it does not do yet: recording streams and the policy's worker-side views (protocol
2), the F12 emergency stop (nobody sits at a station), and telemetry. A client asks for
protocol 1, so a recording is encoded by the client (x264 through ffmpeg).
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 1
# Input is let go this long after the last arm or apply, as by the Windows worker.
WATCHDOG_S = 0.75
# How long a launch waits for the game to log that it is loading, and a quit for a polite
# close before the game is killed.
LAUNCH_WAIT_S, QUIT_WAIT_S = 90, 30
GAME_NAME = "Hearts of Iron IV"
STEAM_LIBRARIES = [
    "~/.steam/debian-installation/steamapps/common",
    "~/.steam/steam/steamapps/common",
    "~/.local/share/Steam/steamapps/common",
]

# Windows virtual-key codes (the vocabulary's) to X keysyms.
KEYSYMS = {
    0x08: 0xFF08,  # BackSpace
    0x09: 0xFF09,  # Tab
    0x0D: 0xFF0D,  # Return
    0x10: 0xFFE1,  # Shift_L
    0x11: 0xFFE3,  # Control_L
    0x1B: 0xFF1B,  # Escape
    0x20: 0x0020,  # space
    0x25: 0xFF51,  # Left
    0x26: 0xFF52,  # Up
    0x27: 0xFF53,  # Right
    0x28: 0xFF54,  # Down
    0xBB: 0x003D,  # equal (the + key)
    0xBD: 0x002D,  # minus
    0xBE: 0x002E,  # period
    0xC0: 0x0060,  # grave: the console
    **{vk: vk for vk in range(0x30, 0x3A)},  # digits
    **{vk: vk + 0x20 for vk in range(0x41, 0x5B)},  # letters, lower case
}
MATCH_KEYS = {0x09, 0x0D, 0x10, 0x11, *range(0x25, 0x29), *range(0x30, 0x3A), *range(0x41, 0x5B)}
SETUP_KEYS = {0x08, 0x1B, 0x20, 0xBB, 0xBD, 0xBE, 0xC0}
# The vocabulary's buttons (left, right, middle) as X buttons; the wheel is 4 (up) and 5.
BUTTONS = {0: 1, 1: 3, 2: 2}
MAP_ERROR = re.compile(
    r"MAP_ERROR|map[/\\]|\.bmp|definition\.csv|adjacenc|railway|supply_node|strategicregion"
    r"|unitstack|weatherposition|buildings\.txt"
)


class WorkerError(Exception):
    """A refusal, said to the client as the reply's `error`."""


def valid_event(event, setup):
    """Whether one input event is allowed: the Windows worker's rules (valid_event)."""
    kind = event.get("kind")
    if kind == "move":
        x, y = event.get("x"), event.get("y")
        return all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in (x, y))
    if kind == "button":
        return event.get("button") in BUTTONS and isinstance(event.get("down"), bool)
    if kind == "wheel":
        delta = event.get("delta")
        return isinstance(delta, int) and not isinstance(delta, bool) and abs(delta) <= 1200
    if kind == "key":
        vk = event.get("vk")
        if not isinstance(event.get("down"), bool):
            return False
        return vk in MATCH_KEYS or (setup and vk in SETUP_KEYS)
    return False


def arena_lines(data: bytes):
    """The arena mod's lines in `data` (only whole lines), and how many bytes they took."""
    used = data.rfind(b"\n") + 1
    lines = []
    for line in data[:used].decode("utf8", "replace").splitlines():
        _, sep, rest = line.partition("]: ARENA ")
        if sep:
            lines.append(rest.strip())
    return lines, used


def read_arena_log(path: Path, offset: int):
    """The Windows worker's game_log: up to 1 MiB from `offset`, and the offset after it."""
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            start = 0 if offset > size else offset
            f.seek(start)
            data = f.read(1 << 20)
    except FileNotFoundError:
        return [], 0
    lines, used = arena_lines(data)
    return lines, start + used


def launch_args(userdir: Path, save=None):
    """HOI4's command line after the program."""
    args = ["-debug_mode", "-gdpr-compliant", f"-userdir={userdir}"]
    if save:
        if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", save):
            raise WorkerError("invalid_save_name")
        args.append(f"-start_save={save}")
    return args


def write_mod_list(userdir: Path, mod: Path):
    """The arena as the game's only mod: its descriptor in the user folder's mod list,
    pointing at the folder where it is kept."""
    (userdir / "mod").mkdir(parents=True, exist_ok=True)
    descriptor = (mod / "descriptor.mod").read_text(encoding="utf8")
    (userdir / "mod" / "arena.mod").write_text(
        descriptor.rstrip("\n") + f'\npath="{mod.as_posix()}"\n', encoding="utf8"
    )
    (userdir / "dlc_load.json").write_text(
        json.dumps({"disabled_dlcs": [], "enabled_mods": ["mod/arena.mod"]}), encoding="utf8"
    )


def map_error_report(error_log: Path):
    """The report's map-error section, in the words Game-Control.ps1 uses."""
    try:
        text = error_log.read_text(encoding="utf8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    found = [line for line in text if MAP_ERROR.search(line)]
    return [f"== map errors in error.log: {len(found)}", *found[:10], "== end of map errors"]


def find_game(extra=None):
    """The HOI4 install: `extra`, or the first Steam library that has it."""
    for base in [extra] if extra else STEAM_LIBRARIES:
        folder = Path(base).expanduser()
        folder = folder if extra else folder / GAME_NAME
        if (folder / "hoi4").exists():
            return folder
    raise WorkerError("hoi4_not_installed")


def game_processes(userdir=None):
    """(pid, command line) of every HOI4 of this user, or only the one using `userdir`."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if not argv or Path(argv[0].decode(errors="replace")).name != "hoi4":
            continue
        line = " ".join(a.decode(errors="replace") for a in argv if a)
        if userdir is None or f"-userdir={userdir}" in line:
            found.append((int(entry.name), line))
    return found


# --- X11 through ctypes ---------------------------------------------------------------


class XImage(ctypes.Structure):
    pass


class _Funcs(ctypes.Structure):
    _fields_ = [
        ("create_image", ctypes.c_void_p),
        ("destroy_image", ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(XImage))),
        ("get_pixel", ctypes.c_void_p),
        ("put_pixel", ctypes.c_void_p),
        ("sub_image", ctypes.c_void_p),
        ("add_pixel", ctypes.c_void_p),
    ]


XImage._fields_ = [
    ("width", ctypes.c_int), ("height", ctypes.c_int), ("xoffset", ctypes.c_int),
    ("format", ctypes.c_int), ("data", ctypes.POINTER(ctypes.c_ubyte)),
    ("byte_order", ctypes.c_int), ("bitmap_unit", ctypes.c_int),
    ("bitmap_bit_order", ctypes.c_int), ("bitmap_pad", ctypes.c_int), ("depth", ctypes.c_int),
    ("bytes_per_line", ctypes.c_int), ("bits_per_pixel", ctypes.c_int),
    ("red_mask", ctypes.c_ulong), ("green_mask", ctypes.c_ulong), ("blue_mask", ctypes.c_ulong),
    ("obdata", ctypes.c_void_p), ("f", _Funcs),
]  # fmt: skip


class XWindowAttributes(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int), ("y", ctypes.c_int), ("width", ctypes.c_int),
        ("height", ctypes.c_int), ("border_width", ctypes.c_int), ("depth", ctypes.c_int),
        ("visual", ctypes.c_void_p), ("root", ctypes.c_ulong), ("c_class", ctypes.c_int),
        ("bit_gravity", ctypes.c_int), ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int), ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong), ("save_under", ctypes.c_int),
        ("colormap", ctypes.c_ulong), ("map_installed", ctypes.c_int),
        ("map_state", ctypes.c_int), ("all_event_masks", ctypes.c_long),
        ("your_event_mask", ctypes.c_long), ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int), ("screen", ctypes.c_void_p),
    ]  # fmt: skip


class XFixesCursorImage(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_short), ("y", ctypes.c_short), ("width", ctypes.c_ushort),
        ("height", ctypes.c_ushort), ("xhot", ctypes.c_ushort), ("yhot", ctypes.c_ushort),
        ("cursor_serial", ctypes.c_ulong), ("pixels", ctypes.POINTER(ctypes.c_ulong)),
        ("atom", ctypes.c_ulong), ("name", ctypes.c_char_p),
    ]  # fmt: skip


ERROR_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
IS_VIEWABLE, ZPIXMAP, REVERT_TO_PARENT = 2, 2, 2
ALL_PLANES = ctypes.c_ulong(-1).value


def _lib(name):
    path = ctypes.util.find_library(name)
    if not path:
        raise WorkerError(f"lib{name}_missing")
    return ctypes.CDLL(path)


class X:
    """The few Xlib, XTest and XFixes calls the worker needs, on one connection, one
    caller at a time (the lock): the watchdog lets go of keys from its own thread."""

    def __init__(self, display=None):
        self.x11, self.xtst = _lib("X11"), _lib("Xtst")
        try:
            self.xfixes = _lib("Xfixes")
        except WorkerError:
            self.xfixes = None
        self._declare()
        self.x11.XInitThreads()
        # Xlib's default handler ends the process on any error, such as a window that
        # closed between two calls. Count errors instead.
        self.errors = 0

        def on_error(_display, _event):
            self.errors += 1
            return 0

        self._handler = ERROR_HANDLER(on_error)
        self.x11.XSetErrorHandler(self._handler)
        name = display or os.environ.get("DISPLAY")
        self.display = self.x11.XOpenDisplay(name.encode() if name else None)
        if not self.display:
            raise WorkerError("no_display")
        self.root = self.x11.XDefaultRootWindow(self.display)
        self.lock = threading.RLock()

    def _declare(self):
        u, ul, i, vp = ctypes.c_uint, ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p
        pul, pi, pu = ctypes.POINTER(ul), ctypes.POINTER(i), ctypes.POINTER(u)
        x = self.x11
        x.XOpenDisplay.restype, x.XOpenDisplay.argtypes = vp, [ctypes.c_char_p]
        x.XDefaultRootWindow.restype, x.XDefaultRootWindow.argtypes = ul, [vp]
        x.XGetImage.restype = ctypes.POINTER(XImage)
        x.XGetImage.argtypes = [vp, ul, i, i, u, u, ul, i]
        x.XQueryPointer.argtypes = [vp, ul, pul, pul, pi, pi, pi, pi, pu]
        x.XQueryTree.argtypes = [vp, ul, pul, pul, ctypes.POINTER(pul), pu]
        x.XFetchName.argtypes = [vp, ul, ctypes.POINTER(ctypes.c_char_p)]
        x.XGetWindowAttributes.argtypes = [vp, ul, ctypes.POINTER(XWindowAttributes)]
        x.XTranslateCoordinates.argtypes = [vp, ul, ul, i, i, pi, pi, pul]
        x.XGetInputFocus.argtypes = [vp, pul, pi]
        x.XSetInputFocus.argtypes = [vp, ul, i, ul]
        x.XRaiseWindow.argtypes = [vp, ul]
        x.XFlush.argtypes = [vp]
        x.XSync.argtypes = [vp, i]
        x.XFree.argtypes = [vp]
        x.XKeysymToKeycode.restype, x.XKeysymToKeycode.argtypes = ctypes.c_ubyte, [vp, ul]
        x.XSetErrorHandler.argtypes = [ERROR_HANDLER]
        t = self.xtst
        t.XTestFakeMotionEvent.argtypes = [vp, i, i, i, ul]
        t.XTestFakeButtonEvent.argtypes = [vp, u, i, ul]
        t.XTestFakeKeyEvent.argtypes = [vp, u, i, ul]
        if self.xfixes:
            self.xfixes.XFixesGetCursorImage.restype = ctypes.POINTER(XFixesCursorImage)
            self.xfixes.XFixesGetCursorImage.argtypes = [vp]

    def children(self, window):
        root, parent, kids, n = ctypes.c_ulong(), ctypes.c_ulong(), ctypes.POINTER(ctypes.c_ulong)(), ctypes.c_uint()  # fmt: skip
        if not self.x11.XQueryTree(self.display, window, root, parent, kids, n):
            return None, []
        found = [kids[k] for k in range(n.value)]
        if kids:
            self.x11.XFree(kids)
        return parent.value, found

    def name(self, window):
        out = ctypes.c_char_p()
        if self.x11.XFetchName(self.display, window, ctypes.byref(out)) and out.value:
            text = out.value.decode(errors="replace")
            self.x11.XFree(ctypes.cast(out, ctypes.c_void_p))
            return text
        return None

    def find(self, title, window=None, depth=0):
        """The first viewable window named `title`, searching down from the root."""
        window = self.root if window is None else window
        with self.lock:
            if window != self.root and self.name(window) == title:
                attrs = XWindowAttributes()
                self.x11.XGetWindowAttributes(self.display, window, ctypes.byref(attrs))
                if attrs.map_state == IS_VIEWABLE:
                    return window
            if depth > 4:
                return None
            for kid in self.children(window)[1]:
                hit = self.find(title, kid, depth + 1)
                if hit:
                    return hit
        return None

    def rect(self, window):
        """(x, y, width, height) of the window's inside on the screen, or None."""
        with self.lock:
            attrs = XWindowAttributes()
            if not self.x11.XGetWindowAttributes(self.display, window, ctypes.byref(attrs)):
                return None
            if attrs.map_state != IS_VIEWABLE:
                return None
            x, y, child = ctypes.c_int(), ctypes.c_int(), ctypes.c_ulong()
            self.x11.XTranslateCoordinates(self.display, window, self.root, 0, 0, x, y, child)
            return x.value, y.value, attrs.width, attrs.height

    def screen_size(self):
        attrs = XWindowAttributes()
        with self.lock:
            self.x11.XGetWindowAttributes(self.display, self.root, ctypes.byref(attrs))
        return attrs.width, attrs.height

    def grab(self, x, y, w, h):
        """BGRA bytes of that rectangle of the screen, alpha opaque."""
        import numpy as np

        with self.lock:
            image = self.x11.XGetImage(self.display, self.root, x, y, w, h, ALL_PLANES, ZPIXMAP)
            if not image:
                raise WorkerError("capture_failed")
            try:
                im = image.contents
                if im.bits_per_pixel != 32:
                    raise WorkerError(f"unsupported_depth_{im.bits_per_pixel}")
                raw = ctypes.string_at(im.data, im.bytes_per_line * h)
                stride = im.bytes_per_line
            finally:
                image.contents.f.destroy_image(image)
        pixels = np.frombuffer(raw, np.uint8).reshape(h, stride // 4, 4)[:, :w].copy()
        pixels[:, :, 3] = 255
        return pixels

    def pointer(self):
        """The pointer on the screen, (x, y)."""
        r, c = ctypes.c_ulong(), ctypes.c_ulong()
        rx, ry, wx, wy, mask = (ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int(),
                                ctypes.c_uint())  # fmt: skip
        with self.lock:
            self.x11.XQueryPointer(self.display, self.root, r, c, rx, ry, wx, wy, mask)
        return rx.value, ry.value

    def cursor(self):
        """The pointer's image as premultiplied BGRA, its hotspot, and where it is; None
        without XFixes."""
        import numpy as np

        if not self.xfixes:
            return None
        with self.lock:
            image = self.xfixes.XFixesGetCursorImage(self.display)
            if not image:
                return None
            try:
                c = image.contents
                n = c.width * c.height
                argb = np.ctypeslib.as_array(c.pixels, (n,)).astype(np.uint32)
                bgra = argb.view(np.uint8).reshape(c.height, c.width, 4).copy()
                return bgra, (c.xhot, c.yhot), (c.x, c.y)
            finally:
                self.x11.XFree(ctypes.cast(image, ctypes.c_void_p))

    def focused(self):
        focus, revert = ctypes.c_ulong(), ctypes.c_int()
        with self.lock:
            self.x11.XGetInputFocus(self.display, focus, revert)
        return focus.value

    def ancestors(self, window):
        chain = []
        with self.lock:
            while window and window != self.root and len(chain) < 16:
                chain.append(window)
                window = self.children(window)[0]
        return chain

    def raise_and_focus(self, window):
        with self.lock:
            self.x11.XRaiseWindow(self.display, window)
            self.x11.XSetInputFocus(self.display, window, REVERT_TO_PARENT, 0)
            self.x11.XSync(self.display, 0)

    def motion(self, x, y):
        with self.lock:
            self.xtst.XTestFakeMotionEvent(self.display, -1, int(x), int(y), 0)
            self.x11.XFlush(self.display)

    def button(self, button, down):
        with self.lock:
            self.xtst.XTestFakeButtonEvent(self.display, button, int(down), 0)
            self.x11.XFlush(self.display)

    def key(self, keysym, down):
        with self.lock:
            code = self.x11.XKeysymToKeycode(self.display, keysym)
            if not code:
                raise WorkerError("no_keycode")
            self.xtst.XTestFakeKeyEvent(self.display, code, int(down), 0)
            self.x11.XFlush(self.display)


def draw_pointer(frame, cursor, origin):
    """Blend the pointer (premultiplied BGRA) into `frame` (BGRA) where it is on screen;
    whether any of it landed there."""
    image, (hx, hy), (px, py) = cursor
    x0, y0 = px - hx - origin[0], py - hy - origin[1]
    h, w = image.shape[:2]
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame.shape[1], x0 + w), min(frame.shape[0], y0 + h)
    if fx0 >= fx1 or fy0 >= fy1:
        return False
    src = image[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0].astype("uint16")
    dst = frame[fy0:fy1, fx0:fx1].astype("uint16")
    alpha = src[:, :, 3:4]
    out = src[:, :, :3] + (dst[:, :, :3] * (255 - alpha) + 127) // 255
    frame[fy0:fy1, fx0:fx1, :3] = out.clip(0, 255).astype("uint8")
    return True


# --- the worker ----------------------------------------------------------------------


class Worker:
    def __init__(self, x, *, mods, userdir, game=None, observer=False, out=None):
        self.x, self.mods, self.userdir = x, Path(mods), Path(userdir)
        self.game_dir, self.observer = game, observer
        self.out = out or sys.stdout.buffer
        self.write_lock = threading.Lock()
        self.input_lock = threading.Lock()
        self.window = None
        self.armed = self.setup = False
        self.last = time.monotonic()
        self.held_keys, self.held_buttons = set(), set()
        self.control = False
        threading.Thread(target=self._watchdog, daemon=True).start()

    # replies

    def reply(self, header, payload=b""):
        header = {**header, "bytes": len(payload)}
        with self.write_lock:
            self.out.write(json.dumps(header).encode() + b"\n")
            if payload:
                self.out.write(payload)
            self.out.flush()

    def handle(self, line):
        try:
            cmd = json.loads(line)
        except ValueError:
            self.reply({"error": "invalid_json"})
            return
        rid = cmd.get("id")
        try:
            op = cmd.get("op", "")
            method = getattr(self, f"op_{op}", None)
            if method is None:
                raise WorkerError("unknown_operation")
            if self.observer and op in OBSERVER_REFUSES:
                raise WorkerError("observer_refuses_input_and_control")
            result = method(cmd)
            header, payload = result if isinstance(result, tuple) else (result, b"")
            self.reply({"id": rid, **header}, payload)
        except WorkerError as error:
            self.reply({"id": rid, "error": str(error)})
        except Exception as error:  # noqa: BLE001 - one bad request must not end the worker.
            self.reply({"id": rid, "error": f"{type(error).__name__}: {error}"})

    # the game window

    def find_window(self):
        if self.window is None or self.x.rect(self.window) is None:
            self.window = self.x.find(GAME_NAME)
        return self.window

    def foreground(self):
        window = self.find_window()
        if not window:
            return False
        focus = self.x.focused()
        return focus in self.x.ancestors(window) or window in self.x.ancestors(focus)

    def client_rect(self):
        window = self.find_window()
        rect = window and self.x.rect(window)
        if not rect:
            raise WorkerError("game_window_not_found")
        sw, sh = self.x.screen_size()
        x, y, w, h = rect
        # Only the part on the screen can be read.
        x0, y0, x1, y1 = max(0, x), max(0, y), min(sw, x + w), min(sh, y + h)
        if x1 <= x0 or y1 <= y0:
            raise WorkerError("game_window_off_screen")
        return x0, y0, x1 - x0, y1 - y0

    # input

    def _release_held(self):
        for keysym in list(self.held_keys):
            self.x.key(keysym, False)
        for button in list(self.held_buttons):
            self.x.button(button, False)
        self.held_keys.clear()
        self.held_buttons.clear()

    def _watchdog(self):
        while True:
            time.sleep(0.1)
            with self.input_lock:
                if self.armed and time.monotonic() - self.last > WATCHDOG_S:
                    self._release_held()
                    self.armed = False

    def _apply_one(self, event, rect):
        kind = event["kind"]
        if kind == "move":
            x, y, w, h = rect
            self.x.motion(x + round(event["x"] * (w - 1)), y + round(event["y"] * (h - 1)))
        elif kind == "button":
            button = BUTTONS[event["button"]]
            self.x.button(button, event["down"])
            (self.held_buttons.add if event["down"] else self.held_buttons.discard)(button)
        elif kind == "wheel":
            button = 4 if event["delta"] > 0 else 5
            for _ in range(max(1, abs(event["delta"]) // 120)):
                self.x.button(button, True)
                self.x.button(button, False)
        elif kind == "key":
            keysym = KEYSYMS[event["vk"]]
            self.x.key(keysym, event["down"])
            (self.held_keys.add if event["down"] else self.held_keys.discard)(keysym)

    # operations

    def op_attach(self, cmd):
        if not self.find_window():
            raise WorkerError("game_window_not_found")
        return {"hwnd": int(self.window), "foreground": self.foreground(),
                "clock_ns": time.monotonic_ns(), "backend": "x11_getimage",
                "computer": platform.node(), "protocol": PROTOCOL,
                "observer": self.observer}  # fmt: skip

    def op_capture(self, cmd):
        if cmd.get("views"):
            raise WorkerError("views_unsupported_on_linux")
        import numpy as np

        x, y, w, h = self.client_rect()
        start = time.monotonic_ns()
        frame = self.x.grab(x, y, w, h)
        end = time.monotonic_ns()
        px, py = self.x.pointer()
        drawn = False
        if cmd.get("pointer", True):
            cursor = self.x.cursor()
            if cursor is not None:
                drawn = draw_pointer(frame, cursor, (x, y))
        meta = {"width": w, "height": h, "t_ns": end, "start_ns": start, "end_ns": end,
                "backend": "x11_getimage", "cursor": [px - x, py - y], "pointer_drawn": drawn,
                "foreground": self.foreground(), "stopped": False, "overflow": False,
                "encoding": "raw"}  # fmt: skip
        parts = []
        # The whole frame unless only crops were asked for, as on Windows.
        if cmd.get("full", not cmd.get("regions")):
            parts.append(frame.tobytes())
            meta["full_bytes"] = len(parts[0])
        else:
            meta["full_bytes"] = 0
        if cmd.get("regions"):
            sizes = []
            for rx, ry, rw, rh in cmd["regions"]:
                crop = np.zeros((rh, rw, 4), np.uint8)
                part = frame[max(0, ry) : ry + rh, max(0, rx) : rx + rw]
                crop[: part.shape[0], : part.shape[1]] = part
                parts.append(crop.tobytes())
                sizes.append(len(parts[-1]))
            meta["region_bytes"] = sizes
        return meta, b"".join(parts)

    def op_arm(self, cmd):
        if not self.foreground():
            raise WorkerError("game_not_foreground")
        if self.control:
            raise WorkerError("arm_refused_during_control")
        with self.input_lock:
            self.setup = cmd.get("mode") == "setup"
            self.armed, self.last = True, time.monotonic()
        return {"armed": True}

    def op_release(self, cmd):
        with self.input_lock:
            self._release_held()
            self.armed = False
        return {"armed": False}

    def op_apply(self, cmd):
        if cmd.get("at_ms") is not None:
            raise WorkerError("timed_batches_need_protocol_2")
        events = cmd.get("events") or []
        with self.input_lock:
            if not self.armed:
                raise WorkerError("input_not_armed_or_focus_lost")
            if not self.foreground():
                # Keys held for the game must not go on into whatever took its place.
                self._release_held()
                self.armed = False
                raise WorkerError("input_not_armed_or_focus_lost")
            if len(events) > 64 or not all(valid_event(e, self.setup) for e in events):
                raise WorkerError("invalid_event_batch")
            rect = self.client_rect()
            for event in events:
                if not self.foreground():
                    self._release_held()
                    self.armed = False
                    raise WorkerError("focus_lost_during_batch")
                self._apply_one(event, rect)
            self.last = time.monotonic()
        return {"applied": len(events), "t_ns": time.monotonic_ns()}

    def op_focus(self, cmd):
        if self.armed:
            raise WorkerError("focus_refused_while_armed")
        window = self.find_window()
        if not window:
            raise WorkerError("game_window_not_found")
        if not self.foreground():
            self.x.raise_and_focus(window)
            time.sleep(0.2)
        focus = self.x.focused()
        owner = {
            "window": int(focus),
            "title": self.x.name(focus),
            "desktop": os.environ.get("DISPLAY"),
        }
        return {"foreground": self.foreground(), "owner": owner}

    def op_status(self, cmd):
        return {"armed": self.armed, "foreground": self.foreground(), "stopped": False,
                "held_keys": sorted(self.held_keys), "held_buttons": sorted(self.held_buttons),
                "t_ns": time.monotonic_ns(), "log": [], "protocol": PROTOCOL,
                "observer": self.observer, "streaming": False}  # fmt: skip

    def op_events(self, cmd):
        # Nobody sits at a station: there is no player's input to hand over.
        return {"events": [], "t_ns": time.monotonic_ns(), "overflow": False}

    def op_game_log(self, cmd):
        lines, offset = read_arena_log(
            self.userdir / "logs" / "game.log", int(cmd.get("offset", 0))
        )
        return {"lines": lines, "offset": offset}

    def op_pointer(self, cmd):
        import numpy as np

        cursor = self.x.cursor()
        if cursor is None:
            raise WorkerError("pointer_hidden")
        image, hotspot, _ = cursor
        straight = image.astype(np.float32)
        alpha = straight[:, :, 3:4]
        straight[:, :, :3] = np.where(alpha > 0, straight[:, :, :3] * 255 / np.maximum(alpha, 1), 0)
        bgra = straight.round().clip(0, 255).astype(np.uint8)
        return ({"width": image.shape[1], "height": image.shape[0], "hotspot": list(hotspot)},
                bgra.tobytes())  # fmt: skip

    # control: launch, quit, report, saves

    def _control(self, fn, cmd):
        if self.armed:
            raise WorkerError("control_refused_while_armed")
        self.control = True
        try:
            code, output = fn(cmd)
        finally:
            self.control = False
        return {"exit": code, "output": output}

    def op_launch(self, cmd):
        return self._control(self._launch, cmd)

    def op_quit(self, cmd):
        return self._control(self._quit, cmd)

    def op_report(self, cmd):
        return self._control(self._report, cmd)

    def op_saves(self, cmd):
        return self._control(self._saves, cmd)

    def op_restart_discord(self, cmd):
        return {"exit": 0, "output": "no Discord on a Linux station"}

    def _launch(self, cmd):
        name = cmd.get("mod") or ""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name) or name.startswith("."):
            return 1, "invalid_mod_name"
        mod = self.mods / name
        if not (mod / "descriptor.mod").exists():
            return 1, f"no arena {name} in {self.mods}"
        running = game_processes()
        if running:
            return 1, f"HOI4 is already running (pid {running[0][0]}); quit it first"
        game = find_game(self.game_dir)
        self.userdir.mkdir(parents=True, exist_ok=True)
        write_mod_list(self.userdir, mod.resolve())
        args = launch_args(self.userdir, cmd.get("save"))
        env = {**os.environ, "LD_LIBRARY_PATH": f"{game}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
        logs = self.userdir / "logs"
        launched = time.time()
        with open(self.userdir / "hoi4.out", "ab") as out:
            process = subprocess.Popen(
                [str(game / "hoi4"), *args], cwd=game, env=env, stdout=out, stderr=out,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )  # fmt: skip
        lines = [f"Arena load test PID {process.pid}", f"userdir {self.userdir}"]
        deadline = time.monotonic() + LAUNCH_WAIT_S
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return 1, "\n".join([*lines, f"HOI4 exited with {process.returncode}"])
            # Test-ArenaLoad.ps1's sign on Windows: the game has read the mods and set up
            # the map's history, and its main menu comes next. The mod list's "Active Mod"
            # line comes half a minute earlier, while the loading screen is still up, and
            # menu clicks made then are lost.
            log = logs / "game.log"
            if log.exists() and log.stat().st_mtime > launched:
                if "Executing History" in log.read_text(encoding="utf8", errors="replace"):
                    return 0, "\n".join([*lines, "The game has loaded the arena."])
            time.sleep(1)
        return 0, "\n".join([*lines, "Timed out waiting for the game log"])

    def _quit(self, cmd):
        running = game_processes(self.userdir)
        if not running:
            return 0, "HOI4 is not running"
        for pid, _ in running:
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + QUIT_WAIT_S
        while time.monotonic() < deadline and game_processes(self.userdir):
            time.sleep(1)
        killed = []
        for pid, _ in game_processes(self.userdir):
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        self.window = None
        return 0, f"closed {[p for p, _ in running]}" + (f", killed {killed}" if killed else "")

    def _report(self, cmd):
        lines = ["== processes"]
        lines += [f"{pid} {line}" for pid, line in game_processes()] or ["no HOI4"]
        window = self.find_window()
        lines += ["== window", f"{window} {self.x.rect(window) if window else None} "
                  f"foreground {self.foreground()}"]  # fmt: skip
        game_log = self.userdir / "logs" / "game.log"
        if game_log.exists():
            tail = game_log.read_text(encoding="utf8", errors="replace").splitlines()[-10:]
            lines += ["== game.log", *tail]
        lines += map_error_report(self.userdir / "logs" / "error.log")
        return 0, "\n".join(lines)

    def _saves(self, cmd):
        folder = self.userdir / "save games"
        saves = sorted(folder.glob("*.hoi4"), key=lambda p: p.stat().st_mtime, reverse=True)
        return 0, "\n".join(p.stem for p in saves)


OBSERVER_REFUSES = {"arm", "apply", "focus", "launch", "quit", "restart_discord"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mods", default=str(ROOT / "artifacts" / "mods"))
    # Where Game-Control.ps1 is on Windows; kept so both workers take the same arguments.
    parser.add_argument("--scripts", default=None)
    parser.add_argument(
        "--userdir", default=os.environ.get("HOI4_USERDIR", str(ROOT / "artifacts" / "hoi4-user"))
    )
    parser.add_argument("--game", default=os.environ.get("HOI4_GAME"))
    parser.add_argument("--display", default=None)
    parser.add_argument("--observer", action="store_true")
    args = parser.parse_args(argv)
    worker = Worker(X(args.display), mods=args.mods, userdir=Path(args.userdir).resolve(),
                    game=args.game, observer=args.observer)  # fmt: skip
    for line in sys.stdin.buffer:
        if line.strip():
            worker.handle(line)
    with worker.input_lock:
        worker._release_held()


if __name__ == "__main__":
    main()
