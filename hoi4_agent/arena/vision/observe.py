"""One captured frame -> ``PlayerObservation``, with a confidence record beside it.

Only the screen is read, so fog of war is correct by construction: an enemy stack exists
here only while its counter is drawn. What each field rests on:

- units: ``counters.find_counters`` on the map rectangle, each counter assigned to the
  nearest calibrated province centre within ``max_assign_distance``; farther ones are dropped
  and counted in ``ObservationDetail.unassigned``.
- enemy ids are ephemeral: kept while a stack stays put or steps to an adjacent province
  between two frames, otherwise new. They are contacts, not engine identities.
- own units are per stack (id derived from the province). They become per division only when
  the army panel is calibrated AND the division tracker still agrees with the stack counts;
  the tracker dead-reckons from accepted orders and gives up (back to stacks) on any
  disagreement such as a forced retreat. UNVERIFIED end to end.
- controller: median map colour around each centre against calibrated country colours;
  None (unknown) without calibration or outside tolerance. UNVERIFIED on the real map.
- game_hour: the top-bar date AND clock ("12:00, 1 Jan, 1936") through the existing
  ``GlyphReader``, as hours since the episode's first read. When the strip is unreadable
  (the shipped glyph set lacks 4, 5, 7, 8 and the month letters) the hour is ESTIMATED from wall time and
  ``hours_per_second`` for the current speed, a rough guess, and its confidence drops to 0.2.
  It never moves backwards.
- speed/pause: the existing ``speed_1..5`` / ``pause_on/off`` templates via ``perception.tiers``;
  unreadable keeps the last known value with confidence 0.
- terminal/winner: the mod's ``ARENA_OUTCOME`` line in the game log. This is a LOG-BASED
  SHORTCUT, not vision. ``outcome_detector`` is the hook for a victory-popup check (TODO).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from ...geometry import WindowGeometry
from ...perception.digits import GlyphReader
from ...perception.templates import TemplateStore
from ...perception.tiers import crop_roi, read_pause, read_speed
from ..contracts import Country, PlayerObservation, UnitView
from ..layout import ArenaLayout
from .calibration import ArenaVisionCalibration
from .counters import CounterDigits, CounterReading, find_counters
from .panel import PanelRow, read_panel

OWN_STACK_BASE = 1_000_000  # own stack id = base + province * 10 + index within the province
ENEMY_BASE = 10_000_000
RESET_RE = re.compile(r"ARENA_RESET")
OUTCOME_RE = re.compile(r"ARENA_OUTCOME\W+(?:winner\W+)?(BLU|RED|DRAW|NONE)\b", re.IGNORECASE)
_CLOCK_RE = re.compile(r"(\d{1,2}):00")
_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)  # HOI4 has no leap days
# GUESS: game hours per wall second at speeds 1..5 (speed 5 is uncapped and machine-bound).
HOURS_PER_SECOND = (1.0, 2.0, 4.0, 8.0, 24.0)
OutcomeDetector = Callable[[Image.Image], tuple[bool, Country | None]]


class LogTail:
    """New complete lines of a growing text log since the last poll; survives truncation."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._offset = self._size()

    def _size(self) -> int:
        try:
            return self.path.stat().st_size if self.path else 0
        except OSError:
            return 0

    def poll(self) -> list[str]:
        if self.path is None:
            return []
        size = self._size()
        if size < self._offset:
            self._offset = 0  # the game rewrote the log
        if size == self._offset:
            return []
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self._offset)
                data = handle.read(size - self._offset)
        except OSError:
            return []
        end = data.rfind(b"\n") + 1
        self._offset += end
        return data[:end].decode("utf-8", errors="replace").splitlines()


def parse_outcome(lines: list[str]) -> tuple[bool, Country | None]:
    for line in lines:
        match = OUTCOME_RE.search(line)
        if match:
            token = match.group(1).upper()
            return True, Country(token) if token in ("BLU", "RED") else None
    return False, None


def absolute_hour(year: int, month: int, day: int, hour: int) -> int:
    return ((year * 365 + sum(_MONTH_DAYS[:month - 1]) + day - 1) * 24) + hour


@dataclass
class UnitHandle:
    """What the executor needs to act on an observed own unit."""
    unit_id: int
    province_id: int
    pixel: tuple[float, float]  # counter centre in client pixels
    stack_count: int
    panel_row: int | None = None


@dataclass
class ObservationDetail:
    confidence: dict[str, float] = field(default_factory=dict)
    handles: dict[int, UnitHandle] = field(default_factory=dict)
    counters: tuple[CounterReading, ...] = ()
    unassigned: int = 0
    per_division: bool = False
    hour_estimated: bool = False


class EnemyTracker:
    """Ephemeral contact ids by province continuity between consecutive frames."""

    def __init__(self, layout: ArenaLayout) -> None:
        self._neighbors = {p.id: set(p.neighbors) for p in layout.provinces}
        self._previous: list[tuple[int, int]] = []  # (id, province)
        self._next = ENEMY_BASE

    def reset(self) -> None:
        self._previous = []

    def assign(self, provinces: list[int]) -> list[int]:
        free = list(self._previous)
        result: list[int | None] = [None] * len(provinces)
        for adjacent in (False, True):  # stationary stacks claim their ids first
            for index, province in enumerate(provinces):
                if result[index] is not None:
                    continue
                match = next((item for item in free if (item[1] in self._neighbors.get(province, set())
                                                        if adjacent else item[1] == province)), None)
                if match is not None:
                    free.remove(match)
                    result[index] = match[0]
        ids: list[int] = []
        for value in result:
            if value is None:
                value, self._next = self._next, self._next + 1
            ids.append(value)
        self._previous = list(zip(ids, provinces))
        return ids


class OwnDivisionTracker:
    """Which province each army-panel row is in, dead-reckoned from accepted orders.

    Starts from the scenario's spawn order, moves a division when its origin stack shrank and
    its ordered target grew, and declares itself inconsistent the moment its per-province
    counts disagree with the map. It never guesses its way back; reset re-seeds it.
    """

    def __init__(self, initial: tuple[int, ...] = ()) -> None:
        self._initial = initial
        self.location: dict[int, int] = {}
        self.target: dict[int, int] = {}
        self.consistent = False
        self.reset()

    def reset(self) -> None:
        self.location = {row + 1: province for row, province in enumerate(self._initial)}
        self.target = {}
        self.consistent = bool(self.location)

    def note_order(self, unit_id: int, target: int | None) -> None:
        if unit_id in self.location:
            if target is None:
                self.target.pop(unit_id, None)
            else:
                self.target[unit_id] = target

    def update(self, seen: dict[int, int]) -> bool:
        if not self.consistent:
            return False
        believed: dict[int, int] = {}
        for province in self.location.values():
            believed[province] = believed.get(province, 0) + 1
        for unit_id, target in sorted(self.target.items()):
            origin = self.location[unit_id]
            left_origin = believed.get(origin, 0) > seen.get(origin, 0)
            if left_origin and believed.get(target, 0) < seen.get(target, 0):
                believed[origin] -= 1
                believed[target] = believed.get(target, 0) + 1
                self.location[unit_id] = target
                del self.target[unit_id]
        self.consistent = {k: v for k, v in believed.items() if v} == {k: v for k, v in seen.items() if v}
        return self.consistent


class ArenaObserver:
    def __init__(self, layout: ArenaLayout, calibration: ArenaVisionCalibration, country: Country, *,
                 templates: TemplateStore | None = None, digits: CounterDigits | None = None,
                 log: LogTail | None = None, outcome_detector: OutcomeDetector | None = None,
                 initial_divisions: tuple[int, ...] = (),
                 clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self.layout, self.calibration, self.country = layout, calibration, country
        self.templates = (templates if templates is not None
                          else TemplateStore.load_dir(calibration.templates_dir))
        self.digits = digits or CounterDigits.load(calibration.counter_digits_dir)
        self.reader = GlyphReader(self.templates, calibration.match_threshold)
        self.log = log or LogTail(calibration.log_path or None)
        self.outcome_detector = outcome_detector
        self.enemies = EnemyTracker(layout)
        self.divisions = OwnDivisionTracker(initial_divisions)
        self.stack_targets: dict[int, int] = {}  # own stack id -> ordered target province
        self.detail = ObservationDetail()
        self._clock_ns = clock_ns
        self.episode_id = ""
        self.begin_episode("unset")

    def begin_episode(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self._sequence = 0
        self._start_hour: int | None = None
        self._hour = 0
        self._hour_ns = self._clock_ns()
        self._speed, self._paused = 1, False
        self._terminal = False
        self._winner: Country | None = None
        self.enemies.reset()
        self.divisions.reset()
        self.stack_targets.clear()

    # --- pieces ------------------------------------------------------------
    def note_order(self, unit_id: int, target: int | None) -> None:
        """The executor reports an accepted move (or a cancel) so targets can be shown."""
        self.divisions.note_order(unit_id, target)
        if unit_id >= OWN_STACK_BASE:
            if target is None:
                self.stack_targets.pop(unit_id, None)
            else:
                self.stack_targets[unit_id] = target

    def centres(self, width: int, height: int) -> dict[int, tuple[float, float]]:
        return {p.id: self.calibration.layout_to_pixel(p.x, p.y, width, height)
                for p in self.layout.provinces}

    def assign(self, readings: list[CounterReading], width: int, height: int
               ) -> tuple[list[tuple[CounterReading, int]], int]:
        centres = self.centres(width, height)
        ids = np.array(list(centres))
        points = np.array([centres[i] for i in ids], dtype=np.float64)
        limit = self.calibration.max_assign_distance * width
        placed: list[tuple[CounterReading, int]] = []
        for reading in readings:
            distance = np.hypot(points[:, 0] - reading.center[0], points[:, 1] - reading.center[1])
            nearest = int(np.argmin(distance))
            if distance[nearest] <= limit:
                placed.append((reading, int(ids[nearest])))
        return placed, len(readings) - len(placed)

    def controllers(self, rgb: np.ndarray, readings: list[CounterReading]) -> dict[int, Country | None]:
        colors = self.calibration.map_colors
        if len(colors) < len(Country):
            return {p.id: None for p in self.layout.provinces}
        height, width = rgb.shape[:2]
        blocked = np.zeros((height, width), dtype=bool)
        for reading in readings:
            x0, y0, x1, y1 = reading.bbox
            pad = int(12 * reading.scale)
            blocked[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + pad] = True
        offsets = [(0.0, 0.0)] + [(r * width * np.cos(a), r * width * np.sin(a)) for r in (0.012, 0.022)
                                  for a in np.linspace(0, 2 * np.pi, 9)[:-1]]
        result: dict[int, Country | None] = {}
        for province_id, (cx, cy) in self.centres(width, height).items():
            samples = [rgb[int(cy + dy), int(cx + dx)] for dx, dy in offsets
                       if 0 <= int(cy + dy) < height and 0 <= int(cx + dx) < width
                       and not blocked[int(cy + dy), int(cx + dx)]]
            result[province_id] = None
            if len(samples) >= 3:
                median = np.median(np.array(samples), axis=0)
                best = min(colors, key=lambda name: float(np.linalg.norm(median - np.array(colors[name]))))
                distance = float(np.linalg.norm(median - np.array(colors[best])))
                if distance <= self.calibration.map_color_tolerance:
                    result[province_id] = Country(best)
        return result

    def _read_clock(self, frame: Image.Image, geo: WindowGeometry, now_ns: int) -> float:
        """Advance ``self._hour``; returns its confidence."""
        text = None
        if "date" in self.calibration.rois and self.reader.available():
            text = self.reader.read_text(crop_roi(frame, geo, self.calibration.rois["date"]))
        from ...schemas import GameDate
        date = GameDate.from_ui_text(text) if text else None
        clock = _CLOCK_RE.search(text) if text else None
        if date is not None and clock is not None and int(clock.group(1)) < 24:
            absolute = absolute_hour(date.year, date.month, date.day, int(clock.group(1)))
            if self._start_hour is None:
                self._start_hour = absolute - self._hour
            if absolute - self._start_hour >= self._hour:
                self._hour, self._hour_ns = absolute - self._start_hour, now_ns
                self.detail.hour_estimated = False
                return 1.0
        if not self._paused:
            elapsed = (now_ns - self._hour_ns) / 1e9
            self._hour += int(elapsed * HOURS_PER_SECOND[self._speed - 1])
        self._hour_ns = now_ns
        self.detail.hour_estimated = True
        return 0.2

    # --- the observation ---------------------------------------------------
    def observe(self, frame: Image.Image, panel_frame: Image.Image | None = None) -> PlayerObservation:
        now_ns = self._clock_ns()
        frame = frame.convert("RGB")
        width, height = frame.size
        geo = WindowGeometry(0, 0, 0, width, height)
        self.detail = detail = ObservationDetail()
        base, threshold = self.calibration.base(), self.calibration.match_threshold
        speed, detail.confidence["game_speed"] = read_speed(frame, geo, base, self.templates, threshold)
        paused, detail.confidence["paused"] = read_pause(frame, geo, base, self.templates, threshold)
        if speed is None:
            detail.confidence["game_speed"] = 0.0
        if paused is None:
            detail.confidence["paused"] = 0.0
        self._speed = speed if speed is not None else self._speed
        self._paused = paused if paused is not None else self._paused
        detail.confidence["game_hour"] = self._read_clock(frame, geo, now_ns)

        fx0, fy0, fx1, fy1 = self.calibration.map_rect
        left, top = int(fx0 * width), int(fy0 * height)
        region = frame.crop((left, top, int(fx1 * width), int(fy1 * height)))
        readings = [_shift(r, left, top)
                    for r in find_counters(region, self.calibration.counter_style, self.digits)]
        detail.counters = tuple(readings)
        placed, detail.unassigned = self.assign(readings, width, height)
        rgb = np.asarray(frame)
        controllers = self.controllers(rgb, readings)
        known = sum(value is not None for value in controllers.values())
        detail.confidence["controllers"] = known / len(controllers)

        units: list[UnitView] = []
        own = [(r, p) for r, p in placed if r.relation == "own"]
        enemy = [(r, p) for r, p in placed if r.relation == "enemy"]
        rows = read_panel(panel_frame if panel_frame is not None else frame, self.calibration.panel)
        own_counts: dict[int, int] = {}
        for reading, province in own:
            own_counts[province] = own_counts.get(province, 0) + (reading.count or 1)
        detail.per_division = bool(rows) and all(r.count is not None for r, _ in own) and \
            len(rows) == sum(own_counts.values()) and self.divisions.update(own_counts)
        if detail.per_division:
            units += self._division_views(rows, own)
        else:
            units += self._stack_views(own, self.country, None)
        units += self._stack_views(enemy, self.country.opponent, self.enemies.assign([p for _, p in enemy]))
        if units:
            detail.confidence["units"] = float(min(unit.confidence for unit in units))

        terminal, winner = parse_outcome(self.log.poll())
        if not terminal and self.outcome_detector is not None:  # TODO: vision-based victory popup check
            terminal, winner = self.outcome_detector(frame)
        if terminal:
            self._terminal, self._winner = True, winner
        observation = PlayerObservation(
            self.episode_id, self.country, self._sequence, self._hour, now_ns,
            tuple(p.view(controllers[p.id]) for p in self.layout.provinces), tuple(units),
            self._terminal, self._winner, self._speed, self._paused)
        self._sequence += 1
        return observation

    def _confidence(self, reading: CounterReading, country: Country) -> float:
        value = reading.confidence * (1.0 if reading.count is not None else 0.5)
        flags = self.calibration.flag_colors
        if len(flags) == len(Country):
            nearest = min(flags, key=lambda name: sum((a - b) ** 2
                                                      for a, b in zip(flags[name], reading.flag_rgb)))
            if nearest != country.value:
                value *= 0.5  # frame hue and flag disagree about the owner
        return round(float(min(max(value, 0.0), 1.0)), 4)

    def _stack_views(self, stacks: list[tuple[CounterReading, int]], country: Country,
                     enemy_ids: list[int] | None) -> list[UnitView]:
        views: list[UnitView] = []
        per_province: dict[int, int] = {}
        for index, (reading, province) in enumerate(stacks):
            if enemy_ids is None:
                slot = per_province.get(province, 0)
                per_province[province] = slot + 1
                unit_id = OWN_STACK_BASE + province * 10 + slot
                target = self.stack_targets.get(unit_id)
                self.detail.handles[unit_id] = UnitHandle(unit_id, province, reading.center,
                                                          reading.count or 1)
            else:
                unit_id, target = enemy_ids[index], None
            views.append(UnitView(unit_id, country, province, reading.organization, reading.strength, None,
                                  reading.in_combat, "unknown", target, None, reading.count or 1,
                                  self._confidence(reading, country)))
        for stale in [k for k in self.stack_targets if enemy_ids is None and k not in self.detail.handles]:
            del self.stack_targets[stale]  # the stack left the province its order was issued from
        return views

    def _division_views(self, rows: tuple[PanelRow, ...], own: list[tuple[CounterReading, int]]
                        ) -> list[UnitView]:
        by_province = {province: reading for reading, province in own}
        views = []
        for row in rows:
            unit_id = row.index + 1
            province = self.divisions.location[unit_id]
            reading = by_province[province]
            self.detail.handles[unit_id] = UnitHandle(unit_id, province, reading.center,
                                                      reading.count or 1, row.index)
            views.append(UnitView(unit_id, self.country, province, row.organization, row.strength, None, None,
                                  "unknown", self.divisions.target.get(unit_id), None, 1,
                                  round(min(row.confidence, self._confidence(reading, self.country)), 4)))
        return views


def _shift(reading: CounterReading, dx: int, dy: int) -> CounterReading:
    from dataclasses import replace
    x0, y0, x1, y1 = reading.bbox
    return replace(reading, bbox=(x0 + dx, y0 + dy, x1 + dx, y1 + dy))
