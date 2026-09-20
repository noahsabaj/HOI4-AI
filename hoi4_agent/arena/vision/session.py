"""``VisionSession``: the ``ArenaSession`` for ONE model country played through this PC's screen.

Built from injected ``WindowLocator`` / ``CaptureBackend`` / ``OrderInputBackend`` so the whole
path runs offline on ``StubLocator`` / ``FakeCapture`` / ``RecordingInput``. Trajectories
collected through this class carry provenance source ``"hoi4_vision"`` (``PROVENANCE_SOURCE``);
that label is only truthful when the injected backends are the real Win32 ones.

Reset clicks the mod's scripted reset decision (``executor.reset_sequence`` names calibrated
points; an entry ``key:<name>`` presses a key) and then waits for the mod's ``ARENA_RESET``
line in the game log. Like the outcome line this is a log-based shortcut, not vision, and
without a configured log path reset refuses rather than assume the click worked. UNVERIFIED
live: the point sequence, the log path and the line format all depend on the arena mod.
"""
from __future__ import annotations

import time
import uuid
from typing import Callable

from ...geometry import WindowGeometry
from ...io.backends import CaptureBackend, OrderInputBackend, WindowLocator
from ..contracts import ArenaError, ArenaSpec, Country, Order, OrderReceipt, PlayerObservation
from ..layout import ArenaLayout
from ..trajectory import REAL_SOURCE
from .calibration import ArenaVisionCalibration
from .executor import OrderExecutor
from .observe import RESET_RE, ArenaObserver, LogTail

PROVENANCE_SOURCE = REAL_SOURCE


class VisionSession:
    def __init__(self, layout: ArenaLayout, calibration: ArenaVisionCalibration, locator: WindowLocator,
                 capture: CaptureBackend, input_backend: OrderInputBackend, *,
                 observer_factory: Callable[[Country, LogTail], ArenaObserver] | None = None,
                 log: LogTail | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.layout, self.calibration = layout, calibration
        self.locator, self.capture, self.input = locator, capture, input_backend
        self.log = log or LogTail(calibration.log_path or None)
        self._factory = observer_factory or (lambda country, tail: ArenaObserver(layout, calibration, country,
                                                                                 log=tail))
        self._sleep, self._clock = sleep, clock
        self.episode_id: str | None = None
        self.country: Country | None = None
        self.observer: ArenaObserver | None = None
        self.executor: OrderExecutor | None = None
        self._spec: ArenaSpec | None = None
        self._latest: PlayerObservation | None = None
        self._latest_at = 0.0
        self._orders: set[str] = set()

    def _window(self) -> WindowGeometry:
        geo = self.locator.find(self.calibration.window_title,
                                (self.calibration.width, self.calibration.height))
        if geo is None:
            raise ArenaError("HOI4 window not found")
        if (geo.client_w, geo.client_h) != (self.calibration.width, self.calibration.height):
            raise ArenaError(f"window is {geo.client_w}x{geo.client_h} but the vision calibration is "
                             f"{self.calibration.width}x{self.calibration.height}")
        return geo

    def reset(self, spec: ArenaSpec, model_countries: tuple[Country, ...]) -> str:
        if len(model_countries) != 1:
            raise ArenaError("a vision session plays exactly one country; the opponent runs elsewhere")
        if self.log.path is None:
            raise ArenaError("vision reset needs the game log path (ARENA_RESET confirmation)")
        geo = self._window()
        if not self.input.focus(geo):
            raise ArenaError("could not focus the game window for reset")
        self.log.poll()  # discard everything written before this reset
        for step in self.calibration.executor.reset_sequence:
            if step.startswith("key:"):
                self.input.key(step[4:])
                continue
            point = self.calibration.point(step)
            if point is None:
                raise ArenaError(f"reset needs the calibrated point {step!r}")
            self.input.click(geo, geo.full_crop(), point[0], point[1])
            self._sleep(self.calibration.executor.settle_ms / 1000.0)
        deadline = self._clock() + self.calibration.executor.reset_timeout_s
        while not any(RESET_RE.search(line) for line in self.log.poll()):
            if self._clock() >= deadline:
                raise ArenaError("no ARENA_RESET line appeared in the game log after the reset decision")
            self._sleep(0.1)
        self.country = model_countries[0]
        self.episode_id = f"vision-{uuid.uuid4().hex[:12]}"
        self._spec = spec
        self._orders.clear()
        self.observer = self._factory(self.country, self.log)
        self.observer.begin_episode(self.episode_id)
        self.executor = OrderExecutor(self.layout, self.calibration, self.capture, self.input,
                                      sleep=self._sleep, clock=self._clock,
                                      on_accepted=self.observer.note_order)
        self.observe(self.country)  # a fresh post-reset observation proves the pipeline end to end
        return self.episode_id

    def observe(self, country: Country) -> PlayerObservation:
        if self.observer is None or self.episode_id is None or country is not self.country:
            raise ArenaError("reset the session before observing, and only observe its own country")
        geo = self._window()
        frame = self.capture.grab(geo)
        panel_frame = None
        army = self.calibration.point("army_select")
        if self.calibration.panel.enabled and army is not None:
            # UNVERIFIED: the division list only shows while the army is selected.
            self.input.click(geo, geo.full_crop(), army[0], army[1])
            self._sleep(self.calibration.executor.settle_ms / 1000.0)
            panel_frame = self.capture.grab(geo)
            deselect = self.calibration.point("deselect")
            if deselect is not None:
                self.input.click(geo, geo.full_crop(), deselect[0], deselect[1])
        self._latest = self.observer.observe(frame, panel_frame)
        self._latest_at = self._clock()
        return self._latest

    def submit(self, order: Order) -> OrderReceipt:
        observation = self._latest
        if (self.executor is None or self.observer is None or observation is None or
                order.episode_id != self.episode_id or order.country is not self.country):
            raise ArenaError("order is outside this session's country/episode")
        if order.observation_sequence != observation.sequence or observation.terminal:
            raise ArenaError("order requires the latest nonterminal player observation")
        if order.id in self._orders:
            raise ArenaError("duplicate order ID")
        self._orders.add(order.id)
        assert self._spec is not None
        age_ms = (self._clock() - self._latest_at) * 1000.0
        if age_ms > self._spec.max_observation_age_ms:
            limit = self._spec.max_observation_age_ms
            return OrderReceipt(order.id, order.episode_id, False,
                                f"observation is {age_ms:.0f} ms old (limit {limit})", None)
        return self.executor.execute(order, self._window(), observation, self.observer.detail)

    def close(self) -> None:
        self.episode_id = self.country = self.observer = self.executor = self._latest = None
        self._orders.clear()
