"""Interactive calibration wizard -> config/calibration.toml + template PNGs.

For each ROI you hover the two corners; for each click-point you hover the
target; template steps grab the calibrated region with the game in a described
state; the glyph step auto-labels date-digit templates from the date you type.
Positions are stored resolution-independently (ROIs as client fractions, points
as 0..1000 crop-normalized).

Navigation is a step wizard (see ``wizard.py``): Enter captures, B / left-arrow
goes back, K keeps a previously recorded answer when revisiting, S skips where
allowed, Q aborts without writing. Nothing is written until the last step, and
one mistake costs one recapture, not a restart.

``--only rois|points|templates|glyphs`` redoes a subset on top of the existing
calibration (narrow with ``points:techs``, ``points:industry_1``,
``templates:pause_on``, ``rois:date`` — comma-separate to combine). Glyph
coverage is DESIGNED to grow across sessions: re-run ``--only glyphs`` at later
in-game dates until 0-9 are all captured.

The step composition / filtering / staleness helpers are pure and unit-tested;
the run loop itself requires the live game + a human.
"""

from __future__ import annotations

from pathlib import Path

from ..calibration import (
    ROI_NAMES,
    Calibration,
    dump_toml,
    load_calibration,
)
from ..enums import BuildingType, GermanState, MapMode, Tech
from ..errors import ConfigError
from .wizard import Step, Wizard

# Lessons from live bring-up, encoded so the operator doesn't rediscover them:
# panel ROIs are presence-detectors (box a STATIC region of the OPEN panel),
# number ROIs must contain exactly one digit run, idle_research_slots is
# counted from the slot row by the VLM, and the date box needs width slack.
ROI_HINTS = {
    "date": "box the top-bar date text (e.g. '12:00, 1 Jan, 1936'), slack for wider dates",
    "pause": "box the pause/play control",
    "speed": "box the game-speed indicator",
    "construction_panel": "OPEN the construction panel, box a small static region (its header)",
    "research_panel": "OPEN the research view, box a small static region (its header)",
    "event_popup": "box where an event popup's TITLE BAR sits (open one if available)",
    "pause_menu": "press ESCAPE so the game menu shows, box its title area",
    "free_civ_slots": "construction panel open, box JUST the free-civ-factory digits (one number only)",
    "idle_research_slots": "research view open, box the WHOLE row of research slot cards",
    "construction_queue": "construction panel open, box the queue list area",
}

UI_POINT_SPECS = (
    ("event_option", "hover an event popup's FIRST OPTION button (open one if available)"),
    ("minimap_anchor", "hover the MINIMAP over central Germany (camera re-anchor)"),
)

# (template_name, roi it is cropped from, operator instruction)
TEMPLATE_SPECS = (
    ("pause_on", "pause", "PAUSE the game (space)"),
    ("pause_off", "pause", "UNPAUSE the game briefly"),
    ("construction_panel", "construction_panel", "open the CONSTRUCTION panel"),
    ("research_panel", "research_panel", "open the RESEARCH panel"),
    ("event_popup", "event_popup", "if an event popup is on screen, leave it open"),
    ("pause_menu", "pause_menu", "press ESCAPE at the home map so the PAUSE/GAME MENU shows "
                                 "(press Escape again afterwards to close it)"),
    *((f"speed_{s}", "speed", f"set game speed to {s} (pause state doesn't matter)")
      for s in range(1, 6)),
)
_TEMPLATE_ROI = {name: roi for name, roi, _ in TEMPLATE_SPECS}

POINT_GROUPS = ("buildings", "techs", "states", "ui")


def parse_only(spec: str | None) -> list[tuple[str, str | None]] | None:
    """Parse ``--only`` into (section, narrow|None) pairs; None means full run.

    Unknown sections/names raise ConfigError up front — a typo must fail loudly,
    not silently select zero steps.
    """
    if spec is None:
        return None
    point_names = (
        {b.value for b in BuildingType} | {t.value for t in Tech}
        | {s.value for s in GermanState} | {n for n, _ in UI_POINT_SPECS}
    )
    valid_narrow: dict[str, set[str]] = {
        "rois": set(ROI_NAMES),
        "points": set(POINT_GROUPS) | point_names,
        "templates": set(_TEMPLATE_ROI),
        "glyphs": set(),
    }
    out: list[tuple[str, str | None]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        section, _, narrow = token.partition(":")
        if section not in valid_narrow:
            raise ConfigError(
                f"unknown --only section {section!r} (want rois|points|templates|glyphs)")
        name = narrow or None
        if name is not None and name not in valid_narrow[section]:
            raise ConfigError(
                f"unknown --only name {name!r} for section {section!r} "
                f"(valid: {', '.join(sorted(valid_narrow[section])) or 'none'})")
        out.append((section, name))
    if not out:
        raise ConfigError("--only given but selects nothing")
    return out


def build_steps(filters: list[tuple[str, str | None]] | None = None) -> list[Step]:
    """Compose the wizard's step list, optionally narrowed by parse_only output."""

    def want(section: str, name: str | None = None, group: str | None = None) -> bool:
        if filters is None:
            return True
        for sec, narrow in filters:
            if sec != section:
                continue
            if narrow is None or narrow == name or narrow == group:
                return True
        return False

    steps: list[Step] = []
    for name in ROI_NAMES:
        if want("rois", name):
            hint = ROI_HINTS.get(name, "")
            steps.append(Step(f"roi:{name}:tl", f"ROI {name!r} — {hint} — hover the TOP-LEFT corner"))
            steps.append(Step(f"roi:{name}:br", f"ROI {name!r}: hover the BOTTOM-RIGHT corner"))
    for b in BuildingType:
        if want("points", b.value, "buildings"):
            steps.append(Step(f"building:{b.value}",
                              f"building {b.value!r}: construction panel open — hover the click point"))
    for t in Tech:
        if want("points", t.value, "techs"):
            steps.append(Step(f"tech:{t.value}",
                              f"tech {t.value!r}: research view open — hover the tech's box"))
    for s in GermanState:
        if want("points", s.value, "states"):
            steps.append(Step(f"state:{s.value}", f"state {s.value!r}: hover the state on the map"))
    for name, instr in UI_POINT_SPECS:
        if want("points", name, "ui"):
            steps.append(Step(f"ui:{name}", f"ui {name!r}: {instr}", skippable=True))
    for tname, _roi, instr in TEMPLATE_SPECS:
        if want("templates", tname):
            steps.append(Step(f"template:{tname}", f"template {tname!r}: {instr}", skippable=True))
    if want("glyphs"):
        steps.append(Step(
            "glyphs",
            "glyphs: have the top-bar date visible, press Enter, then type it (no spaces)",
            skippable=True, textual=True))
    return steps


# Templates cropped from an ROI are pixel-married to it: redoing the ROI without
# recapturing them leaves them matching the OLD box. "glyph_*" stands for the
# whole glyph set (cropped from the date ROI).
_ROI_DEPENDENT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "date": ("glyph_*",),
    **{roi: tuple(n for n, r, _ in TEMPLATE_SPECS if r == roi)
       for roi in {r for _, r, _ in TEMPLATE_SPECS}},
}


def stale_templates_for(redone_rois: list[str], recaptured: set[str]) -> list[str]:
    """Template names invalidated by an ROI redo and not recaptured this run."""
    stale = []
    for roi in redone_rois:
        for t in _ROI_DEPENDENT_TEMPLATES.get(roi, ()):
            if t not in recaptured:
                stale.append(t)
    return stale


def _read_event(step: Step, has_value: bool) -> str:
    """One navigation event: 'enter' | 'back' | 'keep' | 'skip' | 'quit'.

    Single-key on Windows (msvcrt); line-based fallback elsewhere.
    """
    try:
        import msvcrt
    except ImportError:
        while True:
            line = input("    [Enter/b/k/s/q] > ").strip().lower()
            if line == "":
                return "enter"
            if line in ("b", "back"):
                return "back"
            if line == "k" and has_value:
                return "keep"
            if line == "s" and step.skippable:
                return "skip"
            if line == "q":
                return "quit"
    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            return "enter"
        if ch in ("\x00", "\xe0"):  # arrow/function key prefix
            if msvcrt.getwch() == "K":  # left arrow
                return "back"
            continue
        if ch == "\x03":
            raise KeyboardInterrupt
        c = ch.lower()
        if c == "b":
            return "back"
        if c == "k" and has_value:
            return "keep"
        if c == "s" and step.skippable:
            return "skip"
        if c == "q":
            print("    press Q again to abort (nothing will be written), any other key to continue")
            if msvcrt.getwch().lower() == "q":
                return "quit"


def run(cfg, title: str = "Hearts of Iron", only: str | None = None) -> int:
    from ..io import windows as win
    from ..perception.tiers import crop_roi

    try:
        filters = parse_only(only)
    except ConfigError as e:
        print(f"calibrate: {e}")
        return 1

    if not win.available():
        print("calibrate: Windows only")
        return 1
    win.ensure_dpi_aware()
    locator, capture, _inp = win.build_io(cfg.timing.action_dwell_ms, cfg.capture_backend)
    geo = locator.find(title, (cfg.display.width, cfg.display.height))
    if geo is None:
        print("game window not found — is HOI4 running?")
        return 1
    if (geo.client_w, geo.client_h) != (cfg.display.width, cfg.display.height):
        print(f"WARN: client {geo.client_w}x{geo.client_h} != configured "
              f"{cfg.display.width}x{cfg.display.height}")

    path = Path(cfg.paths.calibration)
    base: Calibration | None = load_calibration(path) if path.is_file() else None
    if filters is not None:
        if base is None:
            print(f"--only needs an existing calibration at {path} — run a full `calibrate` first")
            return 1
        if (base.width, base.height) != (geo.client_w, geo.client_h):
            print(f"--only refused: calibration is {base.width}x{base.height} but the window is "
                  f"{geo.client_w}x{geo.client_h} — geometry changed, run a full `calibrate`")
            return 1

    wiz = Wizard(build_steps(filters))
    cursor = win.Win32Input.get_cursor_pos

    def frac_of_cursor() -> tuple[float, float]:
        x, y = cursor()
        return (round((x - geo.screen_left) / geo.client_w, 4),
                round((y - geo.screen_top) / geo.client_h, 4))

    def roi_frac(name: str) -> tuple[float, float, float, float] | None:
        tl = wiz.results.get(f"roi:{name}:tl")
        br = wiz.results.get(f"roi:{name}:br")
        if tl is not None and br is not None:
            return (tl[0], tl[1], br[0], br[1])
        return (base.rois.get(name) if base else None)

    def capture_step(step: Step) -> tuple[bool, object]:
        """(ok, value); ok=False keeps the wizard on this step for a retry."""
        section = step.id.split(":", 1)[0]
        if section == "roi":
            return True, frac_of_cursor()
        if section in ("building", "tech", "state", "ui"):
            return True, geo.screen_to_norm(geo.full_crop(), *cursor())
        if section == "template":
            tname = step.id.split(":", 1)[1]
            frac = roi_frac(_TEMPLATE_ROI[tname])
            if frac is None:
                print(f"    no ROI {_TEMPLATE_ROI[tname]!r} available — capture it first (or S to skip)")
                return False, None
            return True, crop_roi(capture.grab(geo), geo, frac)
        # glyphs: crop the date ROI FIRST (the typed text must describe this frame)
        frac = roi_frac("date")
        if frac is None:
            print("    no 'date' ROI available — capture it first (or S to skip)")
            return False, None
        crop = crop_roi(capture.grab(geo), geo, frac)
        typed = input("    date exactly as shown, no spaces (e.g. 12:00,1Jan,1936); blank skips: ").strip()
        if not typed:
            return True, None
        from ..perception.digits import glyph_boxes
        from ..perception.ncc import to_gray_f32
        boxes = glyph_boxes(to_gray_f32(crop))
        if len(boxes) != len(typed):
            print(f"    segmented {len(boxes)} glyphs but typed {len(typed)} chars — retry (or S to skip)")
            return False, None
        staged = []
        for (x0, y0, x1, y1), ch in zip(boxes, typed):
            if ch.isdigit():
                gname = f"glyph_{ch}"
            elif ch == ".":
                gname = "glyph_dot"
            else:
                continue  # letters/colon/comma are not template-able; expected
            staged.append((gname, crop.crop((x0, y0, x1, y1))))
        print(f"    staged {len(staged)} glyph template(s)")
        return True, staged

    print("Calibration wizard — [Enter] capture  [B/←] back  [K] keep recorded  "
          "[S] skip  [Q] quit (writes nothing)")
    print("Keyboard stays on this terminal; only the mouse hovers the game.\n")

    while not wiz.done:
        step = wiz.current
        assert step is not None
        n, total = wiz.position
        mark = "  [recorded — K keeps it]" if wiz.has_value() else ""
        print(f"[{n:>2}/{total}] {step.prompt}{mark}")
        ev = _read_event(step, wiz.has_value())
        if ev == "quit":
            print("aborted — nothing written")
            return 1
        if ev == "back":
            if not wiz.back():
                print("    (already at the first step)")
        elif ev == "keep":
            wiz.keep()
        elif ev == "skip":
            wiz.skip()
        else:
            ok, value = capture_step(step)
            if ok:
                wiz.record(value)

    # --- assemble (recorded answers overlay the existing calibration) --------
    rois = dict(base.rois) if base else {}
    redone_rois = []
    for name in ROI_NAMES:
        tl = wiz.results.get(f"roi:{name}:tl")
        br = wiz.results.get(f"roi:{name}:br")
        if tl is not None and br is not None:
            rois[name] = (tl[0], tl[1], br[0], br[1])
            redone_rois.append(name)

    def merged(prefix: str, existing: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
        out = dict(existing)
        for sid, v in wiz.results.items():
            if sid.startswith(f"{prefix}:") and v is not None:
                out[sid.split(":", 1)[1]] = v
        return out

    calib = Calibration(
        width=geo.client_w,
        height=geo.client_h,
        rois=rois,
        building_buttons=merged("building", base.building_buttons if base else {}),
        state_points=merged("state", base.state_points if base else {}),
        tech_points=merged("tech", base.tech_points if base else {}),
        ui_points=merged("ui", base.ui_points if base else {}),
        home_map_mode=base.home_map_mode if base else MapMode.DEFAULT.value,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_toml(calib), encoding="utf-8")
    print(f"\nWrote {path}")

    tdir = Path(cfg.paths.templates)
    tdir.mkdir(parents=True, exist_ok=True)
    recaptured: set[str] = set()
    for tname, _roi, _instr in TEMPLATE_SPECS:
        img = wiz.results.get(f"template:{tname}")
        if img is not None:
            img.save(tdir / f"{tname}.png")
            recaptured.add(tname)
    staged_glyphs = wiz.results.get("glyphs")
    if staged_glyphs:
        for gname, img in staged_glyphs:
            img.save(tdir / f"{gname}.png")
        recaptured.add("glyph_*")
        print(f"saved {len(staged_glyphs)} glyph template(s) — re-run `calibrate --only glyphs` "
              "at a later date for full 0-9 coverage")
    if recaptured - {"glyph_*"}:
        print(f"saved {len(recaptured - {'glyph_*'})} template(s) to {tdir}")

    for t in stale_templates_for(redone_rois, recaptured):
        print(f"WARN: ROI redone but template {t!r} not recaptured — it still matches the old "
              "box; run `calibrate --only templates` (or glyphs) to refresh it")
    print("Run `python -m hoi4_agent.cli.main run` — preflight will list anything still missing.")
    return 0
