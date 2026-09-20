"""Small live-game toolkit for bring-up: window-only capture and real input in client pixels.

Usage: python scripts/live.py <command> [args]
  launch [extra hoi4 args]      start the vanilla-region arena profile (autostart as BLU)
  shot OUT.png [x0 y0 x1 y1] [scale]
  key NAME [repeat]
  hover X Y | click X Y | rclick X Y [ctrl] | scroll TICKS X Y | drag X0 Y0 X1 Y1
  quit                          kill the game we launched
Every input command refuses to act unless the HOI4 window is in the foreground.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hoi4_agent.arena.launch import launch_profile  # noqa: E402
from hoi4_agent.io import windows as w  # noqa: E402

GAME = Path("C:/Program Files (x86)/Steam/steamapps/common/Hearts of Iron IV")
PROFILE = ROOT / "artifacts/arena_mod/default"
MOUSEEVENTF_WHEEL = 0x0800
ABS = w.MOUSEEVENTF_MOVE | w.MOUSEEVENTF_ABSOLUTE | w.MOUSEEVENTF_VIRTUALDESK


def window():
    w.ensure_dpi_aware()
    geo = w.Win32Locator().find("Hearts of Iron IV")
    if geo is None:
        raise SystemExit("HOI4 window not found")
    return geo


def focused(geo, attempts: int = 20) -> None:
    """A freshly launched window can refuse the foreground for a while, so retry before giving up.

    Windows only grants a foreground change to a process that has input; a synthetic ALT tap is
    enough to qualify, and without it SetForegroundWindow fails silently from a background script.
    """
    for attempt in range(attempts):
        w.Win32Input._send([w._kbd(0x38, w.KEYEVENTF_SCANCODE),
                            w._kbd(0x38, w.KEYEVENTF_SCANCODE | w.KEYEVENTF_KEYUP)])
        time.sleep(0.1)
        if w.Win32Input().focus(geo):
            time.sleep(0.15)
            return
        time.sleep(0.5 + 0.1 * attempt)
    raise SystemExit("HOI4 is not the foreground window; refusing to send input")


def move(geo, x: float, y: float) -> tuple[int, int]:
    ax, ay = w._to_virtual_abs(int(geo.screen_left + x), int(geo.screen_top + y))
    w.Win32Input._send([w._mouse(ax, ay, ABS)])
    time.sleep(0.06)
    return ax, ay


def button(geo, x: float, y: float, down: int, up: int) -> None:
    ax, ay = move(geo, x, y)
    flags = w.MOUSEEVENTF_ABSOLUTE | w.MOUSEEVENTF_VIRTUALDESK
    w.Win32Input._send([w._mouse(ax, ay, down | flags)])
    time.sleep(0.06)
    w.Win32Input._send([w._mouse(ax, ay, up | flags)])
    time.sleep(0.06)


def scroll(geo, ticks: int, x: float, y: float) -> None:
    move(geo, x, y)
    for _ in range(abs(ticks)):
        event = w._mouse(0, 0, MOUSEEVENTF_WHEEL)
        event.u.mi.mouseData = ctypes.c_ulong((120 if ticks > 0 else -120) & 0xFFFFFFFF).value
        w.Win32Input._send([event])
        time.sleep(0.05)


def shot(geo, out: str, box: tuple[int, ...] | None = None, scale: float = 1.0) -> None:
    img = w.PrintWindowCapture().grab(geo)
    if box:
        img = img.crop(box)
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    img.save(out)
    print(out, img.size)


def main(argv: list[str]) -> None:
    command, args = argv[0], argv[1:]
    if command == "launch":
        print(launch_profile(GAME, PROFILE, "mod/hoi4_arena.mod",
                             ("-debug", "-start_tag=BLU", "-start_speed=1", *args)))
        return
    if command == "quit":
        subprocess.run(["taskkill", "/IM", "hoi4.exe", "/F"], capture_output=True)
        return
    geo = window()
    if command == "shot":
        numbers = [float(a) for a in args[1:]]
        box = tuple(int(n) for n in numbers[:4]) if len(numbers) >= 4 else None
        scale = numbers[4] if len(numbers) == 5 else numbers[0] if len(numbers) == 1 else 1.0
        shot(geo, args[0], box, scale)
        return
    focused(geo)
    if command == "key":
        for _ in range(int(args[1]) if len(args) > 1 else 1):
            w.Win32Input().key(args[0])
            time.sleep(0.08)
    elif command == "hover":
        move(geo, float(args[0]), float(args[1]))
    elif command == "click":
        button(geo, float(args[0]), float(args[1]), w.MOUSEEVENTF_LEFTDOWN, w.MOUSEEVENTF_LEFTUP)
    elif command == "rclick":
        button(geo, float(args[0]), float(args[1]), w.MOUSEEVENTF_RIGHTDOWN, w.MOUSEEVENTF_RIGHTUP)
    elif command == "scroll":
        scroll(geo, int(args[0]), float(args[1]), float(args[2]))
    elif command == "drag":
        ax, ay = move(geo, float(args[0]), float(args[1]))
        flags = w.MOUSEEVENTF_ABSOLUTE | w.MOUSEEVENTF_VIRTUALDESK
        w.Win32Input._send([w._mouse(ax, ay, w.MOUSEEVENTF_LEFTDOWN | flags)])
        for step in range(1, 11):
            move(geo, float(args[0]) + (float(args[2]) - float(args[0])) * step / 10,
                 float(args[1]) + (float(args[3]) - float(args[1])) * step / 10)
        bx, by = move(geo, float(args[2]), float(args[3]))
        w.Win32Input._send([w._mouse(bx, by, w.MOUSEEVENTF_LEFTUP | flags)])
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main(sys.argv[1:])
