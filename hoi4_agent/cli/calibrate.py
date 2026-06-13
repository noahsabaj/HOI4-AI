"""Interactive one-time calibration -> config/calibration.toml (live, Windows).

For each ROI you hover the two corners; for each click-point you hover the target.
Positions are stored resolution-independently (ROIs as client fractions, points as
0..1000 crop-normalized). Not unit-tested (requires the live game + a human).
"""

from __future__ import annotations

from pathlib import Path

from ..calibration import ROI_NAMES, Calibration, dump_toml
from ..config import Config
from ..enums import BuildingType, GermanState, MapMode, Tech


def run(cfg: Config, title: str = "Hearts of Iron") -> int:
    from ..io import windows as win

    if not win.available():
        print("calibrate: Windows only")
        return 1
    win.ensure_dpi_aware()
    locator, _capture, _inp = win.build_io(cfg.timing.action_dwell_ms)
    geo = locator.find(title)
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

    calib = Calibration(
        width=geo.client_w,
        height=geo.client_h,
        rois=rois,
        building_buttons=buildings,
        state_points=states,
        tech_points=techs,
        home_map_mode=MapMode.DEFAULT.value,
    )
    path = Path(cfg.paths.calibration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_toml(calib), encoding="utf-8")
    print(f"\nWrote {path}")
    return 0
