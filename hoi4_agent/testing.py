"""FakeGame: an in-memory HOI4 simulator for offline smoke runs and tests.

It plays the roles of InputBackend + CaptureBackend and provides ``perceive`` so the
*real* controller, tools, playbook, trace, and recovery run end-to-end with zero
external services. Clicks are interpreted against the calibration's named points
(building/state/tech), and time advances while unpaused — enough to exercise the
date-driven cadence and the "wait for a free slot" path.

This is test/dev scaffolding shipped with the package so ``smoke-test --offline``
can reuse it; it is never imported by the live agent path.
"""

from __future__ import annotations

from PIL import Image

from .calibration import Calibration, default_calibration
from .enums import PanelId
from .geometry import WindowGeometry
from .schemas import GameDate, WorldState


class FakeGame:
    def __init__(
        self,
        *,
        calibration: Calibration | None = None,
        geometry: WindowGeometry | None = None,
        idle_research: int = 3,
        free_civ: int = 2,
        date: GameDate | None = None,
        max_free_civ: int = 3,
        days_per_tick: int = 7,
    ) -> None:
        self.calib = calibration or default_calibration()
        self.geometry = geometry or WindowGeometry(1, 0, 0, self.calib.width or 2560, self.calib.height or 1440)
        self.paused = True
        self.speed = 1
        self.open_panel = PanelId.NONE
        self.date = date or GameDate(1936, 1, 1)
        self.idle_research = idle_research
        self.free_civ = free_civ
        self.queue = 0
        self.selected_building: str | None = None
        self.event_popup = False
        self.max_free_civ = max_free_civ
        self.days_per_tick = days_per_tick
        self.input_calls: list[tuple] = []
        self._points: list[tuple[str, str, int, int]] = []
        for cat, d in (
            ("building", self.calib.building_buttons),
            ("state", self.calib.state_points),
            ("tech", self.calib.tech_points),
        ):
            for name, (nx, ny) in d.items():
                self._points.append((cat, name, nx, ny))

    # --- InputBackend ---
    def focus(self, geo) -> bool:
        return True

    def key(self, name: str) -> None:
        self.input_calls.append(("key", name))
        if name == "space":
            self.paused = not self.paused
        elif name == "+":
            self.speed = min(5, self.speed + 1)
        elif name == "-":
            self.speed = max(1, self.speed - 1)
        elif name == "t":
            self.open_panel = PanelId.CONSTRUCTION
        elif name == "w":
            self.open_panel = PanelId.RESEARCH
        elif name == "f1":
            self.open_panel = PanelId.NONE
        elif name == "escape":
            self.open_panel = PanelId.NONE
            self.event_popup = False

    def click(self, geo, crop, nx: int, ny: int) -> None:
        self.input_calls.append(("click", nx, ny))
        cat, name = self._nearest(nx, ny)
        if cat == "building":
            self.selected_building = name
        elif cat == "state":
            if self.open_panel is PanelId.CONSTRUCTION and self.selected_building and self.free_civ > 0:
                self.queue += 1
                self.free_civ -= 1
        elif cat == "tech":
            if self.open_panel is PanelId.RESEARCH and self.idle_research > 0:
                self.idle_research -= 1

    def _nearest(self, nx: int, ny: int) -> tuple[str, str]:
        best = ("", "")
        best_d = None
        for cat, name, px, py in self._points:
            d = (px - nx) ** 2 + (py - ny) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best = (cat, name)
        return best

    # --- CaptureBackend ---
    def grab(self, geo, crop=None) -> Image.Image:
        return Image.new("RGB", (8, 8), color=(20, 20, 40))

    # --- perceive (wired as ctx.perceive) ---
    def perceive(self, read_numbers: bool = True) -> WorldState:
        if not self.paused:
            # Simulate in-game time passing + completions freeing capacity.
            self.date = self.date.plus_days(self.days_per_tick)
            if self.free_civ < self.max_free_civ:
                self.free_civ += 1
        return WorldState(
            date=self.date if read_numbers else None,
            paused=self.paused,
            speed=self.speed if read_numbers else None,
            open_panel=self.open_panel,
            free_civ_slots=self.free_civ if read_numbers else None,
            idle_research_slots=self.idle_research if read_numbers else None,
            construction_queue_len=self.queue if read_numbers else None,
            event_popup=self.event_popup,
            confidence={"panel": 1.0, "pause": 1.0},
        )
