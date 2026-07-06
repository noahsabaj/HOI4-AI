"""Interactive one-time calibration -> config/calibration.toml + template PNGs.

For each ROI you hover the two corners; for each click-point you hover the target.
Positions are stored resolution-independently (ROIs as client fractions, points as
0..1000 crop-normalized). After geometry, the same session captures the template
images perception matches against (panels, pause on/off, speed 1..5) and
auto-labels date glyph templates from the date you type — without these the T0
tier is blind, and preflight will refuse a live run.

Not unit-tested (requires the live game + a human).
"""

from __future__ import annotations

from pathlib import Path

from ..calibration import ROI_NAMES, Calibration, dump_toml
from ..config import Config
from ..enums import BuildingType, GermanState, MapMode, Tech


def _capture_glyphs(capture, geo, date_frac, tdir: Path) -> int:
    """Segment the date ROI and label glyphs from the operator-typed date string.

    Coverage grows across sessions: 1936.1.1 only yields {1,3,6,9,.} — re-run at
    later dates (the reader falls back to the VLM for unknown glyphs meanwhile).
    """
    from ..perception.digits import glyph_boxes
    from ..perception.ncc import to_gray_f32
    from ..perception.tiers import crop_roi

    crop = crop_roi(capture.grab(geo), geo, date_frac)
    typed = input(
        "  glyphs: type the in-game date EXACTLY as shown (e.g. 1936.1.1), blank to skip: "
    ).strip()
    if not typed:
        print("    skipped glyph capture")
        return 0
    boxes = glyph_boxes(to_gray_f32(crop))
    if len(boxes) != len(typed):
        print(f"    WARN: segmented {len(boxes)} glyphs but typed {len(typed)} chars — skipping")
        return 0
    n = 0
    for (x0, y0, x1, y1), ch in zip(boxes, typed):
        if ch.isdigit():
            name = f"glyph_{ch}"
        elif ch == ".":
            name = "glyph_dot"
        else:
            print(f"    WARN: unsupported glyph char {ch!r} — skipped")
            continue
        crop.crop((x0, y0, x1, y1)).save(tdir / f"{name}.png")
        n += 1
    print(f"    saved {n} glyph templates (re-run calibrate on a later date for full 0-9 coverage)")
    return n


def run(cfg: Config, title: str = "Hearts of Iron") -> int:
    from ..io import windows as win
    from ..perception.tiers import crop_roi

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

    cursor = win.Win32Input.get_cursor_pos
    print("Calibration — hover the mouse then press Enter at each prompt.\n")

    rois: dict[str, tuple[float, float, float, float]] = {}
    for name in ROI_NAMES:
        input(f"  ROI {name!r}: hover TOP-LEFT corner, Enter...")
        x0, y0 = cursor()
        input(f"  ROI {name!r}: hover BOTTOM-RIGHT corner, Enter...")
        x1, y1 = cursor()
        rois[name] = (
            round((x0 - geo.screen_left) / geo.client_w, 4),
            round((y0 - geo.screen_top) / geo.client_h, 4),
            round((x1 - geo.screen_left) / geo.client_w, 4),
            round((y1 - geo.screen_top) / geo.client_h, 4),
        )

    def points(label: str, members) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for m in members:
            input(f"  {label} {m.value!r}: hover the click point, Enter...")
            nx, ny = geo.screen_to_norm(geo.full_crop(), *cursor())
            out[m.value] = (nx, ny)
        return out

    buildings = points("building", list(BuildingType))
    techs = points("tech", list(Tech))
    states = points("state", list(GermanState))

    print("\nUI points (popup dismissal + camera anchor). 's' skips — features degrade gracefully.")
    ui: dict[str, tuple[int, int]] = {}
    for name, instruction in (
        ("event_option", "hover an event popup's FIRST OPTION button (open one if available)"),
        ("minimap_anchor", "hover the MINIMAP over central Germany (camera re-anchor)"),
    ):
        ans = input(f"  ui {name!r}: {instruction}, Enter (s=skip)... ").strip().lower()
        if ans == "s":
            print(f"    skipped {name}")
            continue
        ui[name] = geo.screen_to_norm(geo.full_crop(), *cursor())

    calib = Calibration(
        width=geo.client_w,
        height=geo.client_h,
        rois=rois,
        building_buttons=buildings,
        state_points=states,
        tech_points=techs,
        ui_points=ui,
        home_map_mode=MapMode.DEFAULT.value,
    )
    path = Path(cfg.paths.calibration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_toml(calib), encoding="utf-8")
    print(f"\nWrote {path}")

    # --- template capture (what the T0 tier matches against) ------------------
    tdir = Path(cfg.paths.templates)
    tdir.mkdir(parents=True, exist_ok=True)
    print("\nTemplate capture — put the game in the described state, then Enter.")
    print("Type 's' + Enter to skip a single template (preflight will flag gaps).\n")

    def cap(template_name: str, roi_name: str, instruction: str) -> None:
        ans = input(f"  template {template_name!r}: {instruction}, Enter (s=skip)... ").strip().lower()
        if ans == "s":
            print(f"    skipped {template_name}")
            return
        img = crop_roi(capture.grab(geo), geo, rois[roi_name])
        fp = tdir / f"{template_name}.png"
        img.save(fp)
        print(f"    wrote {fp}")

    cap("pause_on", "pause", "PAUSE the game (space)")
    cap("pause_off", "pause", "UNPAUSE the game briefly")
    cap("construction_panel", "construction_panel", "open the CONSTRUCTION panel")
    cap("research_panel", "research_panel", "open the RESEARCH panel")
    cap("event_popup", "event_popup", "if an event popup is on screen, leave it open")
    cap("pause_menu", "pause_menu", "press ESCAPE at the home map so the PAUSE/GAME MENU shows "
                                    "(press Escape again afterwards to close it)")
    for s in range(1, 6):
        cap(f"speed_{s}", "speed", f"set game speed to {s} (pause state doesn't matter)")
    _capture_glyphs(capture, geo, rois["date"], tdir)

    print(f"\nTemplates in {tdir}. Run `python -m hoi4_agent.cli.main run` — preflight "
          "will list anything still missing.")
    return 0
