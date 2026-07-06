"""Smoke tests: offline (real controller + FakeGame) and live (Windows I/O)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..calibration import default_calibration
from ..config import Config
from ..context import AgentContext
from ..controller.loop import run
from ..perception.templates import TemplateStore
from ..playbook.loader import load_playbook, load_research_days
from ..playbook.select import all_done
from ..schemas import PlaybookState
from ..testing import FakeGame
from ..trace.writer import JsonlTraceWriter


def run_offline(cfg: Config) -> int:
    """Drive the real controller end-to-end against the in-memory FakeGame."""
    from ..brain.decide import Brain
    from ..brain.llm import ScriptedBackend

    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = FakeGame(calibration=calib)
    ctx = AgentContext(
        config=cfg,
        geometry=fg.geometry,
        input=fg,
        capture=fg,
        calibration=calib,
        templates=TemplateStore(),
        # Scripted judgment: the playbook's repeatable research_refill goal asks
        # the brain to pick a tech; offline that answer is canned.
        brain=Brain(ScriptedBackend(['{"tech": "industry_2"}'])),
        mode=cfg.mode,
        perceive=fg.perceive,
        sleep=lambda _s: None,  # no real settle delays against the fake
    )
    goals = load_playbook(cfg.paths.playbook)
    tmp = Path(tempfile.mkdtemp(prefix="hoi4_smoke_"))
    trace = tmp / "trace.jsonl"
    with JsonlTraceWriter(trace) as w:
        final = run(ctx, goals, PlaybookState(), writer=w, state_path=str(tmp / "state.json"),
                    research_days=load_research_days(cfg.paths.playbook),
                    sleep=lambda _s: None)
    done = all_done(goals, final)
    records = JsonlTraceWriter.read(trace)
    one_shots = sum(1 for g in goals if not g.repeatable)
    print(f"[offline smoke] completed={done} one-shot goals={len(final.completed_goal_ids)}/{one_shots} "
          f"(+{len(goals) - one_shots} repeatable) cycles={final.cycle_count} "
          f"queue={fg.queue} trace_lines={len(records)}")
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
    locator, capture, inp = win.build_io(cfg.timing.action_dwell_ms, cfg.capture_backend)

    geo = step("find_window", lambda: locator.find(title, (cfg.display.width, cfg.display.height)))
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

    def _capture_printwindow():
        # Validates the occlusion-immune backend regardless of [capture] backend,
        # so the operator knows whether flipping the config is safe.
        import numpy as np

        img = win.PrintWindowCapture().grab(geo)
        p = out / "smoke_capture_printwindow.png"
        img.save(p)
        std = float(np.asarray(img.convert("L")).std())
        note = " — black frame: PrintWindow unsupported for this window" if std < 1 else ""
        return f"saved {p} (std={std:.1f}{note})"

    def _move_cursor_center():
        inp.click(geo, geo.full_crop(), 500, 500)
        return win.Win32Input.get_cursor_pos()

    def _pause_toggle():
        inp.key("space")
        inp.key("space")
        return "sent space x2"

    def _key_f1():
        inp.key("f1")
        return "sent f1"

    step("capture", _capture)
    step("capture_printwindow", _capture_printwindow)
    step("focus", lambda: inp.focus(geo))
    step("move_cursor_center", _move_cursor_center)
    step("key_pause_toggle", _pause_toggle)
    step("key_f1", _key_f1)

    def _verify_hotkeys() -> None:
        """The tooling macros.py demands: press each panel hotkey, let perception
        confirm the panel actually opened, then reset. Needs calibration+templates."""
        nonlocal ok
        from ..calibration import load_calibration
        from ..context import AgentContext
        from ..enums import ToolName
        from ..errors import AgentError
        from ..schemas import Intent
        from ..tools.executor import execute

        try:
            calib = load_calibration(cfg.paths.calibration)
        except AgentError as e:
            print(f"[live smoke] hotkey verification SKIPPED: {e}")
            return
        templates = TemplateStore.load_dir(cfg.paths.templates)
        missing = [n for n in ("construction_panel", "research_panel") if not templates.has(n)]
        if missing:
            print(f"[live smoke] hotkey verification SKIPPED: missing templates {missing} (run calibrate)")
            return
        ctx = AgentContext.build(
            config=cfg, geometry=geo, input=inp, capture=capture,
            calibration=calib, templates=templates, brain=None,
        )
        print("[live smoke] hotkey verification (opens/closes panels in-game):")
        for label, intent in (
            ("construction (T)", Intent(ToolName.OPEN_CONSTRUCTION)),
            ("research (W)", Intent(ToolName.OPEN_RESEARCH)),
            ("close/reset (Esc + map mode)", Intent(ToolName.CLOSE_PANELS)),
        ):
            r = execute(intent, ctx)
            if r.ok:
                print(f"[live smoke]   PASS hotkey {label}: {r.assertion}")
            else:
                ok = False
                print(f"[live smoke]   FAIL hotkey {label}: {r.verdict.value} — {r.assertion}")

    _verify_hotkeys()
    print(f"[live smoke] {'ALL PASS' if ok else 'SOME FAILURES'}")
    return 0 if ok else 1
