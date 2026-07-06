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
        research_frees_every: int = 0,
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
        self.popup_needs_click = True
        self.pause_menu = False
        self.max_free_civ = max_free_civ
        self.days_per_tick = days_per_tick
        # research_frees_every=N: every Nth unpaused tick a running tech
        # "completes" and frees a research slot (0 = never). Lets tests exercise
        # the repeatable refill goal.
        self.research_frees_every = research_frees_every
        self._ticks = 0
        self.input_calls: list[tuple] = []
        self._points: list[tuple[str, str, int, int]] = []
        for cat, d in (
            ("building", self.calib.building_buttons),
            ("state", self.calib.state_points),
            ("tech", self.calib.tech_points),
            ("ui", self.calib.ui_points),
            ("tab", self.calib.research_tabs),
        ):
            for name, (nx, ny) in d.items():
                self._points.append((cat, name, nx, ny))

    def spawn_popup(self, needs_click: bool = True) -> None:
        """Simulate a game event window; most real ones need an option click."""
        self.event_popup = True
        self.popup_needs_click = needs_click

    @property
    def _blocked(self) -> bool:
        """Blocking overlays swallow panel hotkeys, like the real game."""
        return self.pause_menu or self.event_popup

    # --- InputBackend ---
    def focus(self, geo) -> bool:
        return True

    def key(self, name: str) -> None:
        self.input_calls.append(("key", name))
        if name == "space":
            if not self.pause_menu:  # the game menu swallows space
                self.paused = not self.paused
        elif name == "+":
            self.speed = min(5, self.speed + 1)
        elif name == "-":
            self.speed = max(1, self.speed - 1)
        elif name == "t":
            if not self._blocked:
                self.open_panel = PanelId.CONSTRUCTION
        elif name == "w":
            if not self._blocked:
                self.open_panel = PanelId.RESEARCH
        elif name == "f1":
            if not self._blocked:
                self.open_panel = PanelId.NONE
        elif name == "escape":
            # Priority mirrors the real game: a click-required popup ignores
            # escape; the menu closes; a panel closes; escape at HOME opens the
            # pause menu (the v3-era trap reset_to_home must survive).
            if self.event_popup:
                if not self.popup_needs_click:
                    self.event_popup = False
            elif self.pause_menu:
                self.pause_menu = False
            elif self.open_panel is not PanelId.NONE:
                self.open_panel = PanelId.NONE
            else:
                self.pause_menu = True

    def click(self, geo, crop, nx: int, ny: int) -> None:
        self.input_calls.append(("click", nx, ny))
        cat, name = self._nearest(nx, ny)
        if cat == "ui":
            if name == "event_option" and self.event_popup:
                self.event_popup = False
            # minimap_anchor: pans the camera; no observable state change here.
            return
        if cat == "tab":
            # Selecting a research tab has no observable counter change here; the
            # executor always clicks the tech's tab before the tech, so modeling
            # it as a benign no-op keeps the tech click from being misread as one.
            return
        if self.event_popup or self.pause_menu:
            return  # overlays swallow clicks that aren't on their own buttons
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
    def perceive(self, read_numbers: bool = True, fields=None) -> WorldState:
        """Mirrors the real ``perceive`` contract: ``fields`` selects numeric
        facts, and slot/queue counters are context-gated to their panel — so the
        offline controller exercises the same attempt/NotReady paths as live."""
        if not self.paused:
            # Simulate in-game time passing + completions freeing capacity.
            self.date = self.date.plus_days(self.days_per_tick)
            if self.free_civ < self.max_free_civ:
                self.free_civ += 1
            self._ticks += 1
            if self.research_frees_every and self._ticks % self.research_frees_every == 0:
                self.idle_research = min(self.idle_research + 1, 3)
        if not read_numbers:
            want: set[str] = set()
        elif fields is None:
            want = {"date", "free_civ_slots", "idle_research_slots", "construction_queue", "speed"}
        else:
            want = set(fields)

        def gated(field: str, panel: PanelId) -> bool:
            return field in want and self.open_panel is panel

        return WorldState(
            date=self.date if "date" in want else None,
            paused=self.paused,
            speed=self.speed if "speed" in want else None,
            open_panel=self.open_panel,
            free_civ_slots=self.free_civ if gated("free_civ_slots", PanelId.CONSTRUCTION) else None,
            idle_research_slots=self.idle_research if gated("idle_research_slots", PanelId.RESEARCH) else None,
            construction_queue_len=self.queue if gated("construction_queue", PanelId.CONSTRUCTION) else None,
            event_popup=self.event_popup,
            pause_menu=self.pause_menu,
            confidence={"panel": 1.0, "pause": 1.0},
        )
