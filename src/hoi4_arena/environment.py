from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import gymnasium as gym
import numpy as np

from .actions import GRID, PERIOD, SLOTS, VOCAB, decode
from .desktop import DesktopError
from .vision import TERMINAL_GRACE_FRAMES, UNKNOWN_FRAMES, clock_advanced, run_setup

log = logging.getLogger(__name__)

# Any fault inside a step invalidates the episode instead of escaping the loop.
# ScreenRules raises ValueError on a resolution/template mismatch and KeyError on an
# uncalibrated rule name; both must end the match, not abort the coordinator.
FAULTS = (DesktopError, TimeoutError, OSError, ValueError, KeyError)

# How long one side waits for the other to agree on a result once it has one of its own.
# It only has to cover the debounce depth and a tick, because both sides now share a match
# clock; it does not have to absorb the difference between two independent setups.
PAIR_CONFIRM_SECONDS = 3


def disarm(env):
    env.active = False
    # Stop the interval in flight first: releasing while a dispatch thread is still
    # applying events would be undone by its next apply.
    try:
        env.abort_dispatch()
    except Exception as error:  # noqa: BLE001 - cleanup must not raise.
        log.debug("abort during disarm: %s", error)
    try:
        env.desktop.release()
    except (DesktopError, OSError, ValueError):
        pass  # The independent worker watchdog also releases on connection loss.


class ArenaEnv(gym.Env):
    """Real-time transitions: elapsed time includes policy latency; never pause the game.

    Five Hz is the requested cadence, not an assumed property. Overruns are measured.
    A slow policy must fail the runtime gate, not silently receive a faster game clock.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self, desktop, rules, recipe, seconds=1800, recorder=None, downscale=True, view_size=224
    ):
        rules.require_match_rules()
        self.desktop, self.rules, self.recipe = desktop, rules, recipe
        self.seconds, self.recorder = seconds, recorder
        # The worker downscales and crops on the capture side, so a tick moves the five
        # policy views plus the calibrated template regions instead of a 33 MB frame.
        # Recording asks for the full frame back, because video needs the pixels.
        self.downscale, self.view_size = downscale, view_size
        self.action_space = gym.spaces.MultiDiscrete(np.tile([len(VOCAB), GRID, GRID], (SLOTS, 1)))
        self.observation_space = gym.spaces.Box(0, 255, (rules.height, rules.width, 3), np.uint8)
        self.active = False
        self.last = None
        self.record_full = False
        self.deadline = 0.0
        self.dispatch = None
        self.stop_dispatch = threading.Event()
        # One thread: an interval is only ever dispatched after the previous one joined.
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dispatch")

    def observe(self):
        """Capture exactly what this tick will look at."""
        if not self.downscale:
            return self.desktop.capture()
        _, regions = self.rules.capture_regions()
        return self.desktop.capture(views=self.view_size, regions=regions, full=self.record_full)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.desktop.release()
        run_setup(self.desktop, self.rules, self.recipe)
        self.rules.last, self.rules.count = None, 0
        self.last = self.observe()
        self.active = True
        self.desktop.arm()
        self.start = self.last_time = time.monotonic()
        self.deadline = self.start + PERIOD
        self.clock_pixels = None
        self.clock_changed = self.start
        self.unhealthy = self.unknown = 0
        return self.last.rgb, {"capture": self.last.meta, "valid": True}

    def _dispatch(self, action, start, stop):
        """Apply the eight slots across one interval, on their own thread."""
        receipts = []
        for index, token in enumerate(action):
            if stop.is_set():
                break
            time.sleep(max(0, start + index * PERIOD / SLOTS - time.monotonic()))
            if stop.is_set():
                break
            receipts.append(self.desktop.apply(decode(token)))
        return receipts

    def _join_dispatch(self):
        """Wait for the interval in flight and surface anything it raised."""
        if self.dispatch is None:
            return []
        try:
            return self.dispatch.result(timeout=PERIOD * 8)
        finally:
            self.dispatch = None

    def abort_dispatch(self):
        """Stop the interval in flight; the match is over or has faulted."""
        if self.dispatch is None:
            return
        self.stop_dispatch.set()
        try:
            self.dispatch.result(timeout=PERIOD * 8)
        except Exception as error:  # noqa: BLE001 - already unwinding.
            log.debug("dispatch aborted with %s", error)
        finally:
            self.dispatch = None

    def step(self, action):
        """One 200 ms tick, with input dispatch overlapping capture and inference.

        The eight event slots occupy the whole interval but leave the thread idle, so
        they run on their own thread while the caller captures, evaluates the screen and
        decides the next action. The next tick blocks on that dispatch finishing, which
        is the cadence barrier. Dispatching serially and only then capturing, as this
        loop used to, made a tick cost interval + capture + inference and put 5 Hz out
        of reach by construction rather than by measurement.

        The action given here is executed over the interval that *starts* now, while the
        frame returned is the screen at that same instant. That one interval of lag
        between seeing and acting was always present; it is now the only one.

        `applied` therefore holds the receipts for the interval that just ended, which is
        the input that produced the frame being returned.
        """
        if not self.active:
            raise RuntimeError("reset must succeed before step")
        if not self.action_space.contains(np.asarray(action)):
            self.desktop.release()
            self.active = False
            raise ValueError("Invalid physical action")
        try:
            applied = self._join_dispatch()
            time.sleep(max(0, self.deadline - time.monotonic()))
            begin = time.monotonic()
            late = begin - self.deadline
            # Phase from the tick that actually happened, so a slow policy falls behind
            # visibly instead of compressing the next interval to catch up.
            self.deadline = begin + PERIOD
            self.stop_dispatch = threading.Event()
            self.dispatch = self.pool.submit(self._dispatch, action, begin, self.stop_dispatch)
            frame = self.observe()
            screen = self.rules.observe(frame)
            outcome = screen.outcome()
            # A calibrated healthy HUD is required for every nonterminal step and timeout.
            healthy = screen.matches("healthy")
            if outcome in {"disconnect", "desync", "invalid"}:
                raise DesktopError(f"visual_fault:{outcome}")
            if healthy:
                self.unhealthy = self.unknown = 0
            elif outcome is None:
                # A terminal screen legitimately replaces the HUD while its template
                # renders and then debounces, so grant the whole stretch a bounded budget
                # rather than only the debounce depth. A screen matching nothing at all has
                # nothing in flight and gets the shorter budget. Both are finite, so a
                # candidate that never converges cannot suppress the gates below.
                self.unhealthy += 1
                self.unknown = self.unknown + 1 if self.rules.last is None else 0
                if self.unknown > UNKNOWN_FRAMES:
                    raise DesktopError("unrecognized_match_screen")
                if self.unhealthy > TERMINAL_GRACE_FRAMES:
                    raise DesktopError("terminal_screen_never_confirmed")
            if healthy and outcome is None:
                if not screen.matches("running_speed_two"):
                    raise DesktopError("paused_or_wrong_game_speed")
                clock = screen.clock_pixels()
                if clock.size == 0:
                    raise DesktopError("invalid_clock_calibration")
                if clock_advanced(clock, self.clock_pixels):
                    self.clock_changed = time.monotonic()
                    self.clock_pixels = clock.copy()
                elif time.monotonic() - self.clock_changed > 60:
                    raise DesktopError("game_clock_stalled")
            now = time.monotonic()
            reward = {"win": 1.0, "loss": -1.0}.get(outcome, 0.0)
            done = outcome in {"win", "loss"}
            timeout = now - self.start >= self.seconds and not done
            if timeout and not healthy:
                raise DesktopError("uncertain_timeout")
            info = {
                "valid": True,
                "outcome": outcome or ("draw" if timeout else None),
                "elapsed_seconds": now - self.last_time,
                "action_seconds": now - begin,
                "late_seconds": late,
                "deadline_miss": now - self.last_time > PERIOD * 1.25,
                "capture": frame.meta,
                "applied": applied,
            }
            self.last_time = now
            self.last = frame
            if self.recorder:
                self.recorder.append(frame, action=np.asarray(action).tolist(), transition=info)
            if info["deadline_miss"]:
                log.warning(
                    "deadline miss: %.3f s for a %.3f s interval", info["elapsed_seconds"], PERIOD
                )
            if done or timeout:
                log.info(
                    "episode finished: outcome=%s elapsed=%.1f s", info["outcome"], now - self.start
                )
                self.abort_dispatch()
                self.desktop.release()
                self.active = False
            return frame.rgb, reward, done, timeout, info
        except FAULTS as error:
            log.warning("episode invalidated: %s: %s", type(error).__name__, error)
            disarm(self)
            return (
                self.last.rgb,
                0.0,
                False,
                True,
                {"valid": False, "outcome": "invalid", "error": str(error)},
            )

    def render(self):
        return self.last.rgb if self.last else None

    def close(self):
        self.active = False
        try:
            self.abort_dispatch()
        finally:
            self.pool.shutdown(wait=True)
            self.desktop.close()


class ArenaPair:
    """Reset barrier and concurrent stepping; observations never cross player boundaries."""

    def __init__(self, left, right):
        self.envs = [left, right]
        self.pool = ThreadPoolExecutor(max_workers=2)

    def reset(self):
        futures = [self.pool.submit(e.reset) for e in self.envs]
        results, errors = [], []
        # Join every future before disarming: run_setup re-arms at each recipe boundary,
        # so disarming a side that is still running its recipe is immediately undone.
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(error)
        if errors:
            for e in self.envs:
                disarm(e)
            raise errors[0]
        # One match, one clock. Each side started its own `seconds` the moment its own
        # setup finished, and two setups never finish together: the lobby recipe waits on
        # templates and one side is a LAN round trip away. Whichever finished first then
        # reached its own timeout first and reported a draw, while the other was still
        # short of its own and had only PAIR_CONFIRM_SECONDS to catch up, so any reset
        # skew past that invalidated every timeout draw. Both sides take the later start,
        # so they expire within a tick of each other.
        start = max(e.start for e in self.envs)
        for env in self.envs:
            env.start = start
        return results

    def step(self, actions):
        futures = [self.pool.submit(e.step, a) for e, a in zip(self.envs, actions, strict=True)]
        results, errors = [], []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(error)
        if errors:
            for e in self.envs:
                disarm(e)
            raise errors[0]
        if any(not r[4]["valid"] for r in results):
            for e in self.envs:
                disarm(e)
            return [
                (r[0], 0.0, False, True, {**r[4], "valid": False, "outcome": "invalid"})
                for r in results
            ]
        outcomes = [r[4]["outcome"] for r in results]
        if any(o in {"win", "loss", "draw"} for o in outcomes):
            # Both players stop sending input while each screen confirms its result.
            for env in self.envs:
                disarm(env)
            expiry = time.monotonic() + PAIR_CONFIRM_SECONDS
            while None in outcomes and time.monotonic() < expiry:
                time.sleep(0.1)
                for i, env in enumerate(self.envs):
                    if outcomes[i] is not None:
                        continue
                    try:
                        frame = env.observe()
                        screen = env.rules.observe(frame)
                        outcomes[i] = screen.outcome()
                        if (
                            outcomes[i] is None
                            and time.monotonic() - env.start >= env.seconds
                            and screen.matches("healthy")
                        ):
                            outcomes[i] = "draw"
                        if env.recorder:
                            env.recorder.append(frame, confirmation=True, outcome=outcomes[i])
                        info = {**results[i][4], "outcome": outcomes[i]}
                        results[i] = (
                            frame.rgb,
                            {"win": 1.0, "loss": -1.0}.get(outcomes[i], 0.0),
                            outcomes[i] in {"win", "loss"},
                            outcomes[i] == "draw",
                            info,
                        )
                    except FAULTS:
                        outcomes[i] = "invalid"
            consistent = outcomes in [["win", "loss"], ["loss", "win"], ["draw", "draw"]]
            if not consistent:
                return [
                    (
                        r[0],
                        0.0,
                        False,
                        True,
                        {
                            **r[4],
                            "valid": False,
                            "outcome": "invalid",
                            "error": "unconfirmed_pair_result",
                        },
                    )
                    for r in results
                ]
        return results

    def close(self):
        errors = []
        try:
            for e in self.envs:
                try:
                    e.close()
                except Exception as error:
                    errors.append(error)
        finally:
            self.pool.shutdown(wait=True)
        if errors:
            raise errors[0]
