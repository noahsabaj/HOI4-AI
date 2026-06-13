"""Smoke tests: offline (real controller + FakeGame) and live (Windows I/O)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..calibration import default_calibration
from ..config import Config
from ..context import AgentContext
from ..controller.loop import run
from ..perception.templates import TemplateStore
from ..playbook.loader import load_playbook
from ..playbook.select import all_done
from ..schemas import PlaybookState
from ..testing import FakeGame
from ..trace.writer import JsonlTraceWriter


def run_offline(cfg: Config) -> int:
    """Drive the real controller end-to-end against the in-memory FakeGame."""
    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = FakeGame(calibration=calib)
    ctx = AgentContext(
        config=cfg,
        geometry=fg.geometry,
        input=fg,
        capture=fg,
        calibration=calib,
        templates=TemplateStore(),
        brain=None,
        mode=cfg.mode,
        perceive=fg.perceive,
    )
    goals = load_playbook(cfg.paths.playbook)
    tmp = Path(tempfile.mkdtemp(prefix="hoi4_smoke_"))
    trace = tmp / "trace.jsonl"
    with JsonlTraceWriter(trace) as w:
        final = run(ctx, goals, PlaybookState(), writer=w, state_path=str(tmp / "state.json"),
                    sleep=lambda _s: None)
    done = all_done(goals, final)
    records = JsonlTraceWriter.read(trace)
    print(f"[offline smoke] completed={done} goals={len(final.completed_goal_ids)}/{len(goals)} "
          f"cycles={final.cycle_count} queue={fg.queue} trace_lines={len(records)}")
    print(f"[offline smoke] trace: {trace}")
    return 0 if done else 1


def run_live(cfg: Config, title: str = "Hearts of Iron") -> int:
    """Validate the Windows I/O contract against the running game (injects input!)."""
    from ..io import windows as win

    if not win.available():
        print("[live smoke] FAIL: Windows I/O unavailable on this platform")
        return 1

    ok = True

    def step(name, fn):
        nonlocal ok
        try:
            result = fn()
            print(f"[live smoke] PASS {name}: {result}")
            return result
        except Exception as e:  # noqa: BLE001 - smoke test reports everything
            ok = False
            print(f"[live smoke] FAIL {name}: {type(e).__name__}: {e}")
            return None

    print("[live smoke] DPI:", win.ensure_dpi_aware())
    locator, capture, inp = win.build_io(cfg.timing.action_dwell_ms)

    geo = step("find_window", lambda: locator.find(title))
    if geo is None:
        print("[live smoke] game window not found — is HOI4 running?")
        return 1
    print(f"[live smoke] geometry: {geo}")
    if (geo.client_w, geo.client_h) != (cfg.display.width, cfg.display.height):
        print(f"[live smoke] WARN: client {geo.client_w}x{geo.client_h} != "
              f"configured {cfg.display.width}x{cfg.display.height} (borderless at the locked res?)")

    out = Path(cfg.paths.trace_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _capture():
        img = capture.grab(geo)
        p = out / "smoke_capture.png"
        img.save(p)
        import numpy as np

        std = float(np.asarray(img.convert("L")).std())
        return f"saved {p} (std={std:.1f}{' — possibly exclusive fullscreen!' if std < 1 else ''})"

    step("capture", _capture)
    step("focus", lambda: inp.focus(geo))
    step("move_cursor_center", lambda: (inp.click(geo, geo.full_crop(), 500, 500), win.Win32Input.get_cursor_pos())[1])
    step("key_pause_toggle", lambda: (inp.key("space"), inp.key("space"), "sent space x2")[2])
    step("key_f1", lambda: (inp.key("f1"), "sent f1")[1])

    print(f"[live smoke] {'ALL PASS' if ok else 'SOME FAILURES'}")
    return 0 if ok else 1
