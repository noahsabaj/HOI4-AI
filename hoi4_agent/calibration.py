"""Calibration data: ROI rectangles + named click-points, machine-specific.

Produced once by ``cli/calibrate`` and persisted to ``config/calibration.toml``
(gitignored). ROIs are stored as client *fractions* (resolution-independent);
click-points are stored as 0..1000 crop-normalized over the **full client**, so
the executor maps them with ``geometry.norm_to_screen(geo.full_crop(), nx, ny)``.

This is the heart of the categorical-not-spatial trick: the model picks a *name*
(e.g. ``GermanState.RUHR``); the executor looks up its calibrated map click-point.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .enums import BuildingType, GermanState, MapMode, ResearchTab, Tech
from .errors import ConfigError, TemplateMissingError

# Standard ROI names the perception tiers expect.
ROI_NAMES = (
    "date",
    "pause",
    "speed",
    "construction_panel",
    "research_panel",
    "event_popup",
    "pause_menu",
    "free_civ_slots",
    "idle_research_slots",
    "construction_queue",
)

# Template PNGs perception matches against. REQUIRED ones break T0 (panel/pause/
# blocking-overlay classification) when absent — preflight refuses to run live
# without them. Optional ones degrade gracefully: speed falls back to
# unreadable, numeric reads fall back to the VLM (preflight warns).
REQUIRED_TEMPLATES = (
    "construction_panel",
    "research_panel",
    "event_popup",
    "pause_menu",
    "pause_on",
    "pause_off",
)
SPEED_TEMPLATES = tuple(f"speed_{s}" for s in range(1, 6))
GLYPH_TEMPLATE_PREFIX = "glyph_"

# Named UI click-points (calibrated, optional — features degrade without them):
#   event_option   — an event popup's first/default option button (dismissal)
#   minimap_anchor — a fixed minimap spot over Germany (camera re-anchoring)
UI_POINT_NAMES = ("event_option", "minimap_anchor")

# Research-panel tab click-points (one per ResearchTab). Tech points are
# tab-relative, so the executor clicks the tab before the tech.
RESEARCH_TAB_NAMES = tuple(t.value for t in ResearchTab)


@dataclass(frozen=True, slots=True)
class Calibration:
    width: int
    height: int
    rois: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    building_buttons: dict[str, tuple[int, int]] = field(default_factory=dict)
    state_points: dict[str, tuple[int, int]] = field(default_factory=dict)
    tech_points: dict[str, tuple[int, int]] = field(default_factory=dict)
    ui_points: dict[str, tuple[int, int]] = field(default_factory=dict)
    research_tabs: dict[str, tuple[int, int]] = field(default_factory=dict)
    home_map_mode: str = MapMode.DEFAULT.value

    def roi(self, name: str) -> tuple[float, float, float, float]:
        if name not in self.rois:
            raise TemplateMissingError(f"roi:{name}")
        return self.rois[name]

    def building_point(self, b: BuildingType) -> tuple[int, int]:
        return self._pt(self.building_buttons, b.value, "building")

    def state_point(self, s: GermanState) -> tuple[int, int]:
        return self._pt(self.state_points, s.value, "state")

    def tech_point(self, t: Tech) -> tuple[int, int]:
        return self._pt(self.tech_points, t.value, "tech")

    def ui_point(self, name: str) -> tuple[int, int] | None:
        """Optional named UI point; None means the feature degrades gracefully."""
        return self.ui_points.get(name)

    def research_tab_point(self, tab: ResearchTab) -> tuple[int, int] | None:
        """Calibrated click-point for a research tab, or None if uncalibrated
        (preflight blocks a live run whose techs need an uncalibrated tab)."""
        return self.research_tabs.get(tab.value)

    @staticmethod
    def _pt(d: dict[str, tuple[int, int]], key: str, kind: str) -> tuple[int, int]:
        if key not in d:
            raise TemplateMissingError(f"{kind}_point:{key}")
        return d[key]

    def missing_points_for(
        self,
        states: list[GermanState],
        techs: list[Tech],
        buildings: list[BuildingType],
    ) -> list[str]:
        """Names referenced by a playbook that lack a calibrated click-point."""
        missing = []
        missing += [f"building:{b.value}" for b in buildings if b.value not in self.building_buttons]
        missing += [f"state:{s.value}" for s in states if s.value not in self.state_points]
        missing += [f"tech:{t.value}" for t in techs if t.value not in self.tech_points]
        return missing


def _as_point(v) -> tuple[int, int]:
    return (int(v[0]), int(v[1]))


def _as_rect(v) -> tuple[float, float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))


def load_calibration(path: str | Path) -> Calibration:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"calibration file not found: {p} (run `calibrate` first)")
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    res = raw.get("resolution", {})
    return Calibration(
        width=int(res.get("width", 0)),
        height=int(res.get("height", 0)),
        rois={k: _as_rect(v) for k, v in raw.get("roi", {}).items()},
        building_buttons={k: _as_point(v) for k, v in raw.get("building_buttons", {}).items()},
        state_points={k: _as_point(v) for k, v in raw.get("state_points", {}).items()},
        tech_points={k: _as_point(v) for k, v in raw.get("tech_points", {}).items()},
        ui_points={k: _as_point(v) for k, v in raw.get("ui_points", {}).items()},
        research_tabs={k: _as_point(v) for k, v in raw.get("research_tabs", {}).items()},
        home_map_mode=raw.get("home_view", {}).get("map_mode", MapMode.DEFAULT.value),
    )


def dump_toml(c: Calibration) -> str:
    """Serialize a Calibration to TOML text (tomllib is read-only, so we format)."""
    lines = ["[resolution]", f"width = {c.width}", f"height = {c.height}", "", "[roi]"]
    for k, v in c.rois.items():
        lines.append(f"{k} = [{v[0]}, {v[1]}, {v[2]}, {v[3]}]")
    lines.append("")
    for section, d in (
        ("building_buttons", c.building_buttons),
        ("state_points", c.state_points),
        ("tech_points", c.tech_points),
        ("ui_points", c.ui_points),
        ("research_tabs", c.research_tabs),
    ):
        lines.append(f"[{section}]")
        for k, (x, y) in d.items():
            lines.append(f"{k} = [{x}, {y}]")
        lines.append("")
    lines.append("[home_view]")
    lines.append(f'map_mode = "{c.home_map_mode}"')
    lines.append("")
    return "\n".join(lines)


def default_calibration(width: int = 2560, height: int = 1440) -> Calibration:
    """Placeholder calibration covering all enum members — for offline runs/tests.

    Positions are deterministic but arbitrary (NOT real HOI4 coordinates); a live
    run must replace this with ``calibrate``.
    """
    rois = {
        "date": (0.46, 0.012, 0.52, 0.034),
        "pause": (0.40, 0.010, 0.415, 0.032),
        "speed": (0.415, 0.010, 0.46, 0.032),
        "construction_panel": (0.00, 0.10, 0.22, 0.95),
        "research_panel": (0.10, 0.10, 0.95, 0.85),
        "event_popup": (0.30, 0.30, 0.70, 0.70),
        "pause_menu": (0.35, 0.25, 0.65, 0.75),
        "free_civ_slots": (0.02, 0.12, 0.20, 0.16),
        "idle_research_slots": (0.10, 0.86, 0.95, 0.92),
        "construction_queue": (0.78, 0.12, 0.98, 0.90),
    }

    def grid(items: list[str], y_base: float) -> dict[str, tuple[int, int]]:
        out = {}
        for i, name in enumerate(items):
            nx = int(100 + (i % 6) * 140)
            ny = int(y_base + (i // 6) * 80)
            out[name] = (nx, ny)
        return out

    return Calibration(
        width=width,
        height=height,
        rois=rois,
        building_buttons=grid([b.value for b in BuildingType], 220),
        state_points=grid([s.value for s in GermanState], 450),
        tech_points=grid([t.value for t in Tech], 350),
        ui_points={"event_option": (500, 620), "minimap_anchor": (930, 930)},
        research_tabs={tab.value: (int(500 + i * 60), 120) for i, tab in enumerate(ResearchTab)},
        home_map_mode=MapMode.DEFAULT.value,
    )
