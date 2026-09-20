"""``Order`` -> mouse and keyboard, then a visual check that the game took it.

Sequence for a unit order: focus, select (click the counter; or ``army_select`` then the
division's army-panel row when the unit is one division of a larger stack), issue, confirm,
deselect. Input mapping and where each binding comes from:

    MOVE            right-click the target province centre
    SUPPORT_ATTACK  ctrl + right-click   (game loc PROVINCE_UNIT_CTRL_CLICK)
    CANCEL          select + "h"         (interface/unitview.gui btn_hold, tooltip HALT_UNIT)
    PAUSE           space                (interface/topbar.gui; the file notes SPACE is disabled in
                                          multiplayer, so LAN play needs pause_mode = "click")
    SET_SPEED       click ``speed_N`` when calibrated, else step with numpad +/- from the observed
                    speed. topbar.gui binds KP_PLUS/KP_MINUS and gives the five speed steps NO
                    number shortcut, so "number keys 1-5" is only available as speed_mode="direct".

The bindings were read from the installed game's files; none has been exercised live yet.

Confirmation makes no assumption about what an order arrow looks like: it compares the frame
grabbed after selecting with frames grabbed after ordering, along the unit->target line, and
accepts when enough of that line changed. A rejected receipt therefore means "no visible
change", which also happens when the same order was already standing. UNVERIFIED live.

There is no action cap and no pacing here. The only limit is ``budget_ms``: an order never
blocks longer, and running out of budget is reported as not accepted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

from ...geometry import WindowGeometry
from ...io.backends import CaptureBackend, OrderInputBackend
from ..contracts import Order, OrderReceipt, PlayerObservation, Verb
from ..layout import ArenaLayout
from .calibration import ArenaVisionCalibration
from .observe import ObservationDetail, UnitHandle
from .panel import row_point


@dataclass
class OrderTiming:
    order_id: str
    verb: str
    inputs: int
    elapsed_ms: float
    confirmed: bool | None  # None: nothing to confirm or no way to


@dataclass
class _Run:
    started: float
    inputs: int = 0
    notes: list[str] = field(default_factory=list)


def line_change(before: Image.Image, after: Image.Image, start: tuple[float, float], end: tuple[float, float],
                skip_start: float, skip_end: float, threshold: float = 40.0, half_width: int = 3) -> float:
    """Share of sample points on start->end whose neighbourhood changed between two frames.

    ``skip_*`` (pixels) leave out the counter itself and the target's own counter or label.
    """
    a = np.asarray(before.convert("RGB"), dtype=np.float32)
    b = np.asarray(after.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    if length <= skip_start + skip_end + 4:
        return 0.0
    height, width = a.shape[:2]
    steps = np.linspace(skip_start / length, 1.0 - skip_end / length, max(8, int(length / 4)))
    changed = 0
    for t in steps:
        x, y = int(start[0] + t * (end[0] - start[0])), int(start[1] + t * (end[1] - start[1]))
        x0, y0 = max(0, x - half_width), max(0, y - half_width)
        x1, y1 = min(width, x + half_width + 1), min(height, y + half_width + 1)
        if x1 > x0 and y1 > y0 and float(np.abs(a[y0:y1, x0:x1] - b[y0:y1, x0:x1]).max()) > threshold:
            changed += 1
    return changed / len(steps)


class OrderExecutor:
    def __init__(self, layout: ArenaLayout, calibration: ArenaVisionCalibration, capture: CaptureBackend,
                 input_backend: OrderInputBackend, *, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 on_accepted: Callable[[int, int | None], None] | None = None) -> None:
        self.layout, self.calibration = layout, calibration
        self.capture, self.input = capture, input_backend
        self._sleep, self._clock = sleep, clock
        self._on_accepted = on_accepted  # (unit id, target or None) -> observer bookkeeping
        self.timings: list[OrderTiming] = []

    # --- input helpers -----------------------------------------------------
    def _remaining(self, run: _Run) -> float:
        return self.calibration.executor.budget_ms / 1000.0 - (self._clock() - run.started)

    def _wait(self, run: _Run, seconds: float) -> bool:
        if self._remaining(run) < seconds:
            return False
        self._sleep(seconds)
        return True

    def _click_point(self, run: _Run, geo: WindowGeometry, point: tuple[int, int]) -> None:
        self.input.click(geo, geo.full_crop(), point[0], point[1])
        run.inputs += 1

    def _pixel_point(self, geo: WindowGeometry, pixel: tuple[float, float]) -> tuple[int, int]:
        return self.calibration.pixel_to_point(pixel[0], pixel[1], geo.client_w, geo.client_h)

    def _key(self, run: _Run, name: str) -> None:
        self.input.key(name)
        run.inputs += 1

    def _deselect(self, run: _Run, geo: WindowGeometry) -> None:
        point = self.calibration.point("deselect")
        if point is None:
            run.notes.append("no deselect point calibrated")
        else:
            self._click_point(run, geo, point)

    # --- verbs -------------------------------------------------------------
    def execute(self, order: Order, geo: WindowGeometry, observation: PlayerObservation,
                detail: ObservationDetail) -> OrderReceipt:
        run = _Run(self._clock())
        accepted, reason, confirmed = self._dispatch(order, geo, observation, detail, run)
        if run.notes:
            reason = f"{reason} ({'; '.join(run.notes)})"
        self.timings.append(OrderTiming(order.id, order.verb.value, run.inputs,
                                        round((self._clock() - run.started) * 1000.0, 3), confirmed))
        return OrderReceipt(order.id, order.episode_id, accepted, reason,
                            observation.game_hour if accepted else None)

    def _dispatch(self, order: Order, geo: WindowGeometry, observation: PlayerObservation,
                  detail: ObservationDetail, run: _Run) -> tuple[bool, str, bool | None]:
        if order.verb is Verb.NOOP:
            return True, "noop", None
        if not self.input.focus(geo):
            return False, "could not focus the game window", None
        if order.verb is Verb.PAUSE:
            return self._pause(geo, run)
        if order.verb is Verb.SET_SPEED:
            assert order.speed is not None
            return self._speed(order.speed, geo, observation, detail, run)
        outcomes = [self._unit_order(order, unit_id, geo, detail, run) for unit_id in order.unit_ids]
        accepted = all(ok for ok, _, _ in outcomes)
        return accepted, "; ".join(text for _, text, _ in outcomes), all(c is True for _, _, c in outcomes)

    def _pause(self, geo: WindowGeometry, run: _Run) -> tuple[bool, str, bool | None]:
        if self.calibration.executor.pause_mode == "click":
            point = self.calibration.point("pause_button")
            if point is None:
                return False, "pause_mode is click but pause_button is not calibrated", None
            self._click_point(run, geo, point)
            return True, "pause toggled by click; effect shows in the next observation", None
        self._key(run, self.calibration.hotkeys.pause)
        return True, "pause key sent; effect shows in the next observation", None

    def _speed(self, speed: int, geo: WindowGeometry, observation: PlayerObservation,
               detail: ObservationDetail, run: _Run) -> tuple[bool, str, bool | None]:
        settings, keys = self.calibration.executor, self.calibration.hotkeys
        point = self.calibration.point(f"speed_{speed}")
        if settings.speed_mode == "direct":
            self._key(run, keys.speed_direct[speed - 1])
            return True, f"speed key {keys.speed_direct[speed - 1]!r} sent (UNVERIFIED binding)", None
        if point is not None:
            self._click_point(run, geo, point)
            return True, f"clicked speed step {speed}", None
        if detail.confidence.get("game_speed", 0.0) <= 0.0:
            return False, "current speed unreadable and no speed point calibrated: cannot step", None
        delta = speed - observation.game_speed
        for _ in range(abs(delta)):
            self._key(run, keys.speed_up if delta > 0 else keys.speed_down)
        return True, f"stepped speed {observation.game_speed} -> {speed}", None

    def _select(self, handle: UnitHandle, geo: WindowGeometry, run: _Run) -> str | None:
        panel = self.calibration.panel
        if handle.panel_row is not None and handle.stack_count > 1:
            army = self.calibration.point("army_select")
            if not panel.enabled or army is None:
                return "division shares its province and the army panel is not calibrated"
            self._click_point(run, geo, army)
            self._click_point(run, geo, row_point(panel, handle.panel_row))
        else:
            self._click_point(run, geo, self._pixel_point(geo, handle.pixel))
        return None

    def _unit_order(self, order: Order, unit_id: int, geo: WindowGeometry, detail: ObservationDetail,
                    run: _Run) -> tuple[bool, str, bool | None]:
        handle = detail.handles.get(unit_id)
        if handle is None:
            return False, f"unit {unit_id} is not an own unit of the latest observation", None
        settings = self.calibration.executor
        problem = self._select(handle, geo, run)
        if problem is not None:
            return False, problem, None
        try:
            if order.verb is Verb.CANCEL:
                self._key(run, self.calibration.hotkeys.halt)
                if self._on_accepted:
                    self._on_accepted(unit_id, None)
                return True, f"unit {unit_id}: halt key sent (not visually confirmed)", None
            assert order.target_province_id is not None
            target = self.layout.province(order.target_province_id)
            target_pixel = self.calibration.layout_to_pixel(target.x, target.y, geo.client_w, geo.client_h)
            if not self._wait(run, settings.settle_ms / 1000.0):
                return False, f"unit {unit_id}: budget exhausted before the order", None
            before = self.capture.grab(geo)
            support = order.verb is Verb.SUPPORT_ATTACK
            modifiers = (self.calibration.hotkeys.support_attack_modifier,) if support else ()
            self.input.right_click(geo, geo.full_crop(), *self._pixel_point(geo, target_pixel), modifiers)
            run.inputs += 1
            style = self.calibration.counter_style
            skip = 0.6 * style.frame_w * style.scale
            best = 0.0
            for _ in range(settings.confirm_polls):
                if not self._wait(run, settings.settle_ms / 1000.0):
                    return False, f"unit {unit_id}: confirmation budget exhausted (best {best:.2f})", False
                after = self.capture.grab(geo)
                best = max(best, line_change(before, after, handle.pixel, target_pixel, skip, skip))
                if best >= settings.arrow_min_fraction:
                    if self._on_accepted:
                        self._on_accepted(unit_id, order.target_province_id)
                    return True, f"unit {unit_id}: order arrow seen ({best:.2f} of the line changed)", True
            return False, f"unit {unit_id}: no order arrow appeared ({best:.2f} of the line changed)", False
        finally:
            self._deselect(run, geo)
