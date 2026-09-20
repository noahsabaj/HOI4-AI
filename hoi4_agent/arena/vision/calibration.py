"""Arena-specific calibration: where the arena map, its provinces and its controls are on screen.

Separate from ``hoi4_agent/calibration.py`` (construction/research) and stored in
``config/arena/vision.toml``. Conventions match the base calibration so the same geometry
code applies: rectangles are client FRACTIONS, click points are 0..1000 over the full client.

The province model is one affine map from ``ArenaLayout`` coordinates (normalized to the
arena map view) to client fractions, fitted from three or more hovered province centres.
An affine fit absorbs offset, scale and shear; it cannot absorb the map projection's
curvature, so ``fit_affine`` reports its residual and the wizard prints it.

Every default below is a PLACEHOLDER, not a measurement: nothing here has been checked
against the live arena. ``calibrate-arena`` replaces them.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ...calibration import Calibration
from ..contracts import ArenaError, Country
from ..layout import ArenaLayout
from .counters import RGB, CounterStyle

Rect = tuple[float, float, float, float]
Point = tuple[int, int]
Affine = tuple[float, float, float, float, float, float]  # fx = a*x + b*y + c ; fy = d*x + e*y + f

DEFAULT_PATH = Path("config/arena/vision.toml")
TOP_BAR_ROIS = ("date", "speed", "pause")
# Named click points. All optional: a missing point makes the feature that needs it refuse
# with a reason instead of clicking a guess.
POINT_NAMES = ("decisions_open", "reset_decision", "decisions_close", "deselect", "army_select",
               "pause_button", *(f"speed_{s}" for s in range(1, 6)))


@dataclass(frozen=True)
class PanelCalibration:
    """Army panel division list. UNVERIFIED layout: rows at a fixed pitch below ``first_row``.

    Bar and probe boxes are fractions of ONE ROW's rectangle. ``enabled`` stays False until
    the rows have been hovered in the live game.
    """
    enabled: bool = False
    first_row: Rect = (0.005, 0.30, 0.17, 0.325)
    row_pitch: float = 0.027
    max_rows: int = 24
    org_bar: Rect = (0.55, 0.15, 0.95, 0.40)
    strength_bar: Rect = (0.55, 0.60, 0.95, 0.85)
    selected_probe: Rect = (0.0, 0.0, 0.04, 1.0)
    row_present_std: float = 12.0  # a row whose pixels vary less than this is empty background
    lit_value: float = 90.0
    selected_value: float = 150.0


@dataclass(frozen=True)
class Hotkeys:
    """Sources: ``support_attack_modifier`` from the game's PROVINCE_UNIT_CTRL_CLICK string
    ("Ctrl + Right-click ... support attack"), ``halt`` from interface/unitview.gui btn_hold
    (shortcut "h"), pause and speed from interface/topbar.gui (SPACE, KP_PLUS, KP_MINUS).
    Read from the installed files, not yet exercised in the live game. topbar.gui defines NO
    1-5 shortcuts for the speed steps; ``speed_direct`` exists only for a rebound setup.
    """
    support_attack_modifier: str = "ctrl"
    halt: str = "h"
    pause: str = "space"
    speed_up: str = "+"
    speed_down: str = "-"
    speed_direct: tuple[str, ...] = ("1", "2", "3", "4", "5")


@dataclass(frozen=True)
class ExecutorSettings:
    budget_ms: int = 1500  # hard ceiling for one order including confirmation
    settle_ms: int = 120  # wait after an input before the confirming grab
    confirm_polls: int = 3
    speed_mode: str = "auto"  # auto: click speed_N points, else step with +/-; "direct": number keys
    pause_mode: str = "key"  # "click" uses pause_button: SPACE is disabled in multiplayer
    reset_sequence: tuple[str, ...] = ("decisions_open", "reset_decision", "decisions_close")
    reset_timeout_s: float = 20.0
    arrow_min_fraction: float = 0.25  # share of the unit->target line that must change to confirm an order


@dataclass(frozen=True)
class ArenaVisionCalibration:
    width: int = 2560
    height: int = 1440
    map_rect: Rect = (0.0, 0.05, 1.0, 1.0)
    affine: Affine | None = None  # None: the layout square fills map_rect exactly
    max_assign_distance: float = 0.04  # client-width fraction between a counter and its province centre
    rois: dict[str, Rect] = field(default_factory=dict)
    points: dict[str, Point] = field(default_factory=dict)
    panel: PanelCalibration = field(default_factory=PanelCalibration)
    flag_colors: dict[str, RGB] = field(default_factory=dict)  # Country value -> mean flag colour
    map_colors: dict[str, RGB] = field(default_factory=dict)  # Country value -> province fill colour
    map_color_tolerance: float = 40.0
    counter_style: CounterStyle = field(default_factory=CounterStyle)
    templates_dir: str = "templates"
    counter_digits_dir: str = "templates/arena"
    match_threshold: float = 0.75
    hotkeys: Hotkeys = field(default_factory=Hotkeys)
    executor: ExecutorSettings = field(default_factory=ExecutorSettings)
    log_path: str = ""  # game.log that carries the mod's ARENA_RESET / ARENA_OUTCOME lines
    window_title: str = "Hearts of Iron"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or not (self.map_rect[0] < self.map_rect[2] and
                                                        self.map_rect[1] < self.map_rect[3]):
            raise ArenaError("vision calibration needs a positive resolution and map rectangle")
        unknown = set(self.flag_colors) | set(self.map_colors)
        if not unknown <= {c.value for c in Country}:
            raise ArenaError("vision calibration colours must be keyed by country")

    # --- coordinates -------------------------------------------------------
    def layout_to_fraction(self, x: float, y: float) -> tuple[float, float]:
        if self.affine is None:
            fx0, fy0, fx1, fy1 = self.map_rect
            return fx0 + x * (fx1 - fx0), fy0 + y * (fy1 - fy0)
        a, b, c, d, e, f = self.affine
        return a * x + b * y + c, d * x + e * y + f

    def layout_to_pixel(self, x: float, y: float, width: int, height: int) -> tuple[float, float]:
        fx, fy = self.layout_to_fraction(x, y)
        return fx * width, fy * height

    def layout_to_point(self, x: float, y: float) -> Point:
        fx, fy = self.layout_to_fraction(x, y)
        return int(round(min(max(fx, 0.0), 1.0) * 1000)), int(round(min(max(fy, 0.0), 1.0) * 1000))

    @staticmethod
    def pixel_to_point(px: float, py: float, width: int, height: int) -> Point:
        return int(round(px / width * 1000)), int(round(py / height * 1000))

    def point(self, name: str) -> Point | None:
        return self.points.get(name)

    def base(self) -> Calibration:
        """The top-bar ROIs as a base ``Calibration`` so ``perception.tiers`` reads them."""
        return Calibration(self.width, self.height, rois=dict(self.rois))

    def with_affine_from(self, layout: ArenaLayout, hovered: dict[int, tuple[float, float]]
                         ) -> tuple[ArenaVisionCalibration, float]:
        pairs = [((layout.province(pid).x, layout.province(pid).y), frac) for pid, frac in hovered.items()]
        affine, residual = fit_affine([p for p, _ in pairs], [s for _, s in pairs])
        return replace(self, affine=affine), residual


def fit_affine(layout_points: list[tuple[float, float]], screen_points: list[tuple[float, float]]
               ) -> tuple[Affine, float]:
    """Least-squares affine layout->screen and its RMS residual in screen units."""
    if len(layout_points) != len(screen_points) or len(layout_points) < 3:
        raise ArenaError("an affine fit needs at least three point pairs")
    source = np.asarray(layout_points, dtype=np.float64)
    design = np.column_stack([source, np.ones(len(source))])
    if np.linalg.matrix_rank(design) < 3:
        raise ArenaError("affine fit points are collinear")
    target = np.asarray(screen_points, dtype=np.float64)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = float(np.sqrt(np.mean(np.sum((design @ solution - target) ** 2, axis=1))))
    (a, d), (b, e), (c, f) = solution
    return (float(a), float(b), float(c), float(d), float(e), float(f)), residual


def anchor_provinces(layout: ArenaLayout, count: int = 5) -> tuple[int, ...]:
    """Well-spread provinces to hover: both capitals, then greedy farthest-point picks."""
    chosen = [layout.capital(country).id for country in Country]
    remaining = [p for p in layout.provinces if p.id not in chosen]
    while remaining and len(chosen) < count:
        picked = [layout.province(pid) for pid in chosen]
        best = max(remaining, key=lambda p: (min((p.x - q.x) ** 2 + (p.y - q.y) ** 2 for q in picked), -p.id))
        chosen.append(best.id)
        remaining.remove(best)
    return tuple(chosen)


# --- TOML ------------------------------------------------------------------
def _rgb(value: Any) -> RGB:
    return (int(value[0]), int(value[1]), int(value[2]))


def _rect(value: Any) -> Rect:
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def load_vision_calibration(path: str | Path = DEFAULT_PATH) -> ArenaVisionCalibration:
    file = Path(path)
    if not file.is_file():
        raise ArenaError(f"vision calibration not found: {file} (run `calibrate-arena`)")
    with open(file, "rb") as handle:
        raw = tomllib.load(handle)
    try:
        resolution, mapping, counters = raw.get("resolution", {}), raw.get("map", {}), raw.get("counters", {})
        panel_raw = dict(raw.get("panel", {}))
        for key in ("first_row", "org_bar", "strength_bar", "selected_probe"):
            if key in panel_raw:
                panel_raw[key] = _rect(panel_raw[key])
        hotkeys_raw = dict(raw.get("hotkeys", {}))
        if "speed_direct" in hotkeys_raw:
            hotkeys_raw["speed_direct"] = tuple(str(k) for k in hotkeys_raw["speed_direct"])
        executor_raw = dict(raw.get("executor", {}))
        if "reset_sequence" in executor_raw:
            executor_raw["reset_sequence"] = tuple(str(k) for k in executor_raw["reset_sequence"])
        style = CounterStyle()
        style = replace(style, scale=float(counters.get("scale", style.scale)),
                        own_frame_rgb=_rgb(counters.get("own_frame_rgb", style.own_frame_rgb)),
                        enemy_frame_rgb=_rgb(counters.get("enemy_frame_rgb", style.enemy_frame_rgb)),
                        other_frame_rgb=_rgb(counters.get("other_frame_rgb", style.other_frame_rgb)))
        affine = mapping.get("affine")
        defaults = ArenaVisionCalibration()
        return ArenaVisionCalibration(
            width=int(resolution.get("width", defaults.width)),
            height=int(resolution.get("height", defaults.height)),
            map_rect=_rect(mapping.get("rect", defaults.map_rect)),
            affine=None if not affine else (float(affine[0]), float(affine[1]), float(affine[2]),
                                            float(affine[3]), float(affine[4]), float(affine[5])),
            max_assign_distance=float(mapping.get("max_assign_distance", defaults.max_assign_distance)),
            rois={name: _rect(value) for name, value in raw.get("roi", {}).items()},
            points={name: (int(value[0]), int(value[1])) for name, value in raw.get("points", {}).items()},
            panel=PanelCalibration(**panel_raw),
            flag_colors={name: _rgb(value) for name, value in raw.get("flag_colors", {}).items()},
            map_colors={name: _rgb(value) for name, value in raw.get("map_colors", {}).items()},
            map_color_tolerance=float(mapping.get("color_tolerance", defaults.map_color_tolerance)),
            counter_style=style,
            templates_dir=str(raw.get("paths", {}).get("templates", defaults.templates_dir)),
            counter_digits_dir=str(raw.get("paths", {}).get("counter_digits", defaults.counter_digits_dir)),
            match_threshold=float(raw.get("perception", {}).get("match_threshold", defaults.match_threshold)),
            hotkeys=Hotkeys(**hotkeys_raw),
            executor=ExecutorSettings(**executor_raw),
            log_path=str(raw.get("log", {}).get("path", "")),
            window_title=str(raw.get("window", {}).get("title", defaults.window_title)),
        )
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise ArenaError(f"invalid vision calibration {file}: {exc}") from exc


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, float):
        return repr(round(value, 6))
    return str(value)


def dump_vision_toml(c: ArenaVisionCalibration) -> str:
    """Serialize (tomllib cannot write). Round-trips through ``load_vision_calibration``."""
    sections: list[tuple[str, dict[str, Any]]] = [
        ("resolution", {"width": c.width, "height": c.height}),
        ("window", {"title": c.window_title}),
        ("map", {"rect": c.map_rect, **({"affine": c.affine} if c.affine else {}),
                 "max_assign_distance": c.max_assign_distance, "color_tolerance": c.map_color_tolerance}),
        ("roi", dict(c.rois)),
        ("points", dict(c.points)),
        ("panel", dict(vars(c.panel))),
        ("flag_colors", dict(c.flag_colors)),
        ("map_colors", dict(c.map_colors)),
        ("counters", {"scale": c.counter_style.scale, "own_frame_rgb": c.counter_style.own_frame_rgb,
                      "enemy_frame_rgb": c.counter_style.enemy_frame_rgb,
                      "other_frame_rgb": c.counter_style.other_frame_rgb}),
        ("paths", {"templates": c.templates_dir, "counter_digits": c.counter_digits_dir}),
        ("perception", {"match_threshold": c.match_threshold}),
        ("hotkeys", dict(vars(c.hotkeys))),
        ("executor", dict(vars(c.executor))),
        ("log", {"path": c.log_path}),
    ]
    lines = ["# Arena vision calibration. Written by `calibrate-arena`; see",
             "# hoi4_agent/arena/vision/calibration.py for what each value means.", ""]
    for name, values in sections:
        lines.append(f"[{name}]")
        lines.extend(f"{key} = {_toml(value)}" for key, value in values.items())
        lines.append("")
    return "\n".join(lines)


# --- interactive wizard ----------------------------------------------------
def build_steps(layout: ArenaLayout) -> list[Any]:
    """The wizard's steps. Pure, so the composition is testable without the game."""
    from ...cli.wizard import Step
    steps = [Step("map:tl", "arena MAP VIEW: hover its TOP-LEFT corner"),
             Step("map:br", "arena MAP VIEW: hover its BOTTOM-RIGHT corner")]
    steps += [Step(f"province:{pid}", f"province {pid}: hover its CENTRE (where its counter sits)")
              for pid in anchor_provinces(layout)]
    for name in TOP_BAR_ROIS:
        steps += [Step(f"roi:{name}:tl", f"top-bar {name}: hover the TOP-LEFT corner", skippable=True),
                  Step(f"roi:{name}:br", f"top-bar {name}: hover the BOTTOM-RIGHT corner", skippable=True)]
    steps += [Step(f"point:{name}", f"point {name!r}: hover it", skippable=True) for name in POINT_NAMES]
    for country in Country:
        steps.append(Step(f"map_color:{country.value}",
                          f"hover plain {country.value}-controlled land (no counter, label or border)",
                          skippable=True))
    steps += [Step("panel:row1:tl", "army panel: FIRST division row, TOP-LEFT corner", skippable=True),
              Step("panel:row1:br", "army panel: FIRST division row, BOTTOM-RIGHT corner", skippable=True),
              Step("panel:row2:tl", "army panel: SECOND division row, TOP-LEFT corner", skippable=True),
              Step("digits", "hover a counter, press Enter, then type the division count it shows",
                   skippable=True, textual=True)]
    return steps


def assemble(base: ArenaVisionCalibration, layout: ArenaLayout, results: dict[str, Any],
             width: int, height: int) -> tuple[ArenaVisionCalibration, float | None]:
    """Overlay wizard answers on ``base``. Returns the calibration and the affine residual."""
    calibration = replace(base, width=width, height=height)
    top_left, bottom_right = results.get("map:tl"), results.get("map:br")
    if top_left and bottom_right:
        calibration = replace(calibration,
                              map_rect=(top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    hovered = {int(key.split(":")[1]): value for key, value in results.items()
               if key.startswith("province:") and value is not None}
    residual = None
    if len(hovered) >= 3:
        calibration, residual = calibration.with_affine_from(layout, hovered)
    rois, points = dict(calibration.rois), dict(calibration.points)
    for name in TOP_BAR_ROIS:
        first, second = results.get(f"roi:{name}:tl"), results.get(f"roi:{name}:br")
        if first and second:
            rois[name] = (first[0], first[1], second[0], second[1])
    for name in POINT_NAMES:
        value = results.get(f"point:{name}")
        if value is not None:
            points[name] = (int(round(value[0] * 1000)), int(round(value[1] * 1000)))
    colors = dict(calibration.map_colors)
    for country in Country:
        value = results.get(f"map_color:{country.value}")
        if value is not None:
            colors[country.value] = _rgb(value)
    panel = calibration.panel
    row_a, row_b = results.get("panel:row1:tl"), results.get("panel:row1:br")
    next_row = results.get("panel:row2:tl")
    if row_a and row_b and next_row:
        panel = replace(panel, enabled=True, first_row=(row_a[0], row_a[1], row_b[0], row_b[1]),
                        row_pitch=round(next_row[1] - row_a[1], 5))
    return replace(calibration, rois=rois, points=points, map_colors=colors, panel=panel), residual


def run_wizard(layout: ArenaLayout, path: Path = DEFAULT_PATH) -> int:  # pragma: no cover - needs the game
    """Interactive capture, modelled on ``cli/calibrate.run``. UNVERIFIED: never run live."""
    from ...cli.calibrate import _read_event
    from ...cli.wizard import Wizard
    from ...io import windows as win
    from ...perception.digits import DELTA_LADDER, glyph_boxes
    from ...perception.ncc import to_gray_f32
    from .counters import COUNTER_GLYPH_PREFIX, _box, find_counters

    if not win.available():
        print("calibrate-arena: Windows only")
        return 1
    base = load_vision_calibration(path) if path.is_file() else ArenaVisionCalibration()
    locator, capture, _ = win.build_io()
    geo = locator.find(base.window_title, (base.width, base.height))
    if geo is None:
        print("calibrate-arena: game window not found")
        return 1

    def hovered() -> tuple[float, float]:
        x, y = win.Win32Input.get_cursor_pos()
        return round((x - geo.screen_left) / geo.client_w, 5), round((y - geo.screen_top) / geo.client_h, 5)

    staged: list[tuple[str, Any]] = []
    wizard = Wizard(build_steps(layout))
    print("[Enter] capture  [B] back  [K] keep  [S] skip  [Q] quit (writes nothing)")
    while not wizard.done:
        step = wizard.current
        assert step is not None
        print(f"[{wizard.position[0]}/{wizard.position[1]}] {step.prompt}")
        event = _read_event(step, wizard.has_value())
        if event == "quit":
            print("aborted: nothing written")
            return 1
        if event == "back":
            wizard.back()
        elif event == "keep":
            wizard.keep()
        elif event == "skip":
            wizard.skip()
        elif step.id.startswith("map_color:"):
            fx, fy = hovered()
            frame = capture.grab(geo).convert("RGB")
            wizard.record(frame.getpixel((int(fx * geo.client_w), int(fy * geo.client_h))))
        elif step.id == "digits":
            fx, fy = hovered()
            frame = capture.grab(geo).convert("RGB")
            readings = find_counters(frame, base.counter_style)
            typed = input("    count shown on the hovered counter (blank skips): ").strip()
            if not readings or not typed.isdigit():
                wizard.record(None)
                continue
            nearest = min(readings, key=lambda r: (r.center[0] - fx * geo.client_w) ** 2 +
                          (r.center[1] - fy * geo.client_h) ** 2)
            plate = frame.crop(_box(nearest.bbox[0], nearest.bbox[1], nearest.scale,
                                    base.counter_style.plate_text_box, *frame.size))
            gray = to_gray_f32(plate)
            boxes = next((b for b in (glyph_boxes(gray, d) for d in DELTA_LADDER) if len(b) == len(typed)),
                         None)
            if boxes is None:
                print("    glyph segmentation does not match the typed count; retry or S to skip")
                continue
            staged += [(f"{COUNTER_GLYPH_PREFIX}{ch}", plate.crop(box)) for ch, box in zip(typed, boxes)]
            wizard.record(typed)
        else:
            wizard.record(hovered())
    calibration, residual = assemble(base, layout, wizard.results, geo.client_w, geo.client_h)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_vision_toml(calibration), encoding="utf-8")
    directory = Path(calibration.counter_digits_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, image in staged:
        image.save(directory / f"{name}.png")
    print(f"wrote {path}; {len(staged)} counter digit template(s); affine RMS residual "
          f"{'n/a' if residual is None else f'{residual:.4f} of the client (keep below ~0.005)'}")
    return 0
