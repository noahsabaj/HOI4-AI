from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import gymnasium as gym
import numpy as np

from .actions import GRID, PERIOD, SLOTS, VOCAB, decode
from .arena_log import ArenaLog
from .desktop import DesktopError
from .vision import (
    BLUE,
    RED,
    TERMINAL_GRACE_FRAMES,
    UNKNOWN_FRAMES,
    clock_advanced,
    occupation_balance,
    run_setup,
)

COUNTRY_COLOUR = {"BLU": BLUE, "RED": RED}

# How long a running, unpaused game may show the same clock before the match is called
# off. The clock shows the hour, which advances every 0.1 s at speed 4 and every 2 s even
# at speed 1, so a still clock means the game has stopped. In a two-player match that is
# what a dropped connection looks like while both games keep running: the client's clock
# freezes and its "Server Lost!" popup follows only 25 to 65 s later (measured
# 2026-09-22), while the host keeps playing on. At 60 s this limit let that stretch run
# on as live match time.
CLOCK_STALL_SECONDS = 15
TERMINAL_NAMES = {"win", "loss", "disconnect", "desync"}

log = logging.getLogger(__name__)

# Any fault inside a step invalidates the episode instead of escaping the loop.
# ScreenRules raises ValueError on a resolution/template mismatch and KeyError on an
# uncalibrated rule name; both must end the match, not abort the coordinator.
FAULTS = (DesktopError, TimeoutError, OSError, ValueError, KeyError)

# How long one side waits for the other to agree on a result once it has one of its own.
# It only has to cover the debounce depth and a tick, because both sides now share a match
# clock; it does not have to absorb the difference between two independent setups.
PAIR_CONFIRM_SECONDS = 3

# How often a match scored from the log asks for new lines: every 2 s at 5 Hz. The mod
# writes weekly, which is 17 s of wall time at speed 4 and 84 s at speed 2, so reading
# more often only finds nothing.
LOG_EVERY = 10


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
        self,
        desktop,
        rules,
        recipe,
        seconds=1800,
        recorder=None,
        downscale=True,
        view_size=224,
        country="BLU",
        reward="screen",
    ):
        rules.require_match_rules()
        if country not in COUNTRY_COLOUR:
            raise ValueError("country must be BLU or RED")
        if reward not in {"screen", "log"}:
            raise ValueError("reward must be 'screen' or 'log'")
        # "log" scores the match from the arena mod's game.log lines (arena_log): the
        # surrender names the winner, and the weekly counts pay for ground taken wherever
        # the camera is. "screen" reads both from pixels, which is all a vanilla lobby has.
        self.reward_source = reward
        self.arena_log = None
        self.country = country
        self.colour = COUNTRY_COLOUR[country]
        self.desktop, self.rules, self.recipe = desktop, rules, recipe
        self.seconds, self.recorder = seconds, recorder
        # The worker downscales and crops on the capture side, so a tick moves the policy
        # views (global, four quadrants, and the cursor crop) plus the calibrated
        # template regions, instead of a 33 MB frame.
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
        # Join gives the eight slots time to land. The grace is how long to wait after
        # telling a late interval to stop, long enough for one apply to return.
        self.join_timeout = PERIOD * 8
        self.join_grace = 2.0
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
        self.territory = None
        self.ticks = 0
        self.potential = None
        self.unpaid = 0.0
        if self.reward_source == "log":
            self.arena_log = ArenaLog(self.desktop)
            self.arena_log.poll()
            self.potential = self.arena_log.potential(self.country)
        return self.last.rgb, {"capture": self.last.meta, "valid": True}

    def _read_log(self):
        """New mod lines: the change in potential since the last reading, and any outcome."""
        self.arena_log.poll()
        change = 0.0
        potential = self.arena_log.potential(self.country)
        if potential is not None:
            if self.potential is not None:
                change = potential - self.potential
            self.potential = potential
        return change, self.arena_log.outcome(self.country)

    def confirm_outcome(self, screen):
        """The result a finished step is waiting to agree on: the log's, or the screen's.

        A reading here can also move the potential. That change is kept in `unpaid` for
        the confirming step to add, so the shaping still sums to the final potential.
        """
        if self.arena_log is not None:
            change, outcome = self._read_log()
            self.unpaid += change
            return outcome
        return screen.outcome()

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
        """Wait for the interval in flight and surface anything it raised.

        The future is cleared only after a timeout has asked the thread to stop.
        Clearing it first made abort a no-op, and the slots kept injecting into
        an episode that had already been invalidated.
        """
        dispatch = self.dispatch
        if dispatch is None:
            return []
        try:
            return dispatch.result(timeout=self.join_timeout)
        except TimeoutError:
            self.stop_dispatch.set()
            try:
                receipts = dispatch.result(timeout=self.join_grace)
            except TimeoutError:
                log.warning("dispatch still running after stop")
                raise
            # The stop skipped slots, so the stored action is not what the game received.
            # That transition must not train as valid.
            if len(receipts) < SLOTS:
                raise DesktopError(f"dispatch_truncated:{len(receipts)}_of_{SLOTS}_slots")
            return receipts
        finally:
            self.dispatch = None

    def abort_dispatch(self):
        """Stop the interval in flight; the match is over or has faulted."""
        dispatch = self.dispatch
        if dispatch is None:
            self.stop_dispatch.set()
            return
        self.stop_dispatch.set()
        try:
            dispatch.result(timeout=self.join_timeout + self.join_grace)
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
            self.abort_dispatch()
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
            # A terminal template that is already matching has not finished debouncing.
            # Pause and stall must not throw that episode away: the popup stops the
            # clock, and the pause glyph can light, before the third confirming frame.
            if healthy and outcome is None and self.rules.last not in TERMINAL_NAMES:
                if screen.matches("paused"):
                    raise DesktopError("game_paused")
                # The speed bars show the selected speed. They do not move when the game
                # pauses, so this is checked only on a frame that is still running. A
                # click on the speed control would otherwise leave the manifest's speed,
                # and every later clip's in-game length, false.
                if "speed" in self.rules.rules and not screen.matches("speed"):
                    raise DesktopError("game_speed_changed")
                clock = screen.clock_pixels()
                if clock.size == 0:
                    raise DesktopError("invalid_clock_calibration")
                if clock_advanced(clock, self.clock_pixels):
                    self.clock_changed = time.monotonic()
                    self.clock_pixels = clock.copy()
                elif time.monotonic() - self.clock_changed > CLOCK_STALL_SECONDS:
                    raise DesktopError("game_clock_stalled")
            self.ticks += 1
            log_change = 0.0
            if self.arena_log is not None:
                # Every LOG_EVERY ticks, and on every tick a result may be forming: the
                # surrender line is written as the peace screens open.
                if self.ticks % LOG_EVERY == 0 or not healthy or outcome is not None:
                    log_change, outcome = self._read_log()
                else:
                    outcome = self.arena_log.outcome(self.country)
            now = time.monotonic()
            reward = {"win": 1.0, "loss": -1.0}.get(outcome, 0.0)
            # The change in the acting country's share of the arena, read from the main
            # view only when it shows the whole arena at the calibrated zoom. The camera
            # moves, so any other frame is not a reading: it adds nothing and the last
            # reading stands. An uncalibrated crop adds nothing either. Scored from the
            # log instead, it is still measured but only reported.
            territory_reward = 0.0
            territory = "uncalibrated"
            if self.rules.minimap_rect is not None:
                territory = occupation_balance(
                    screen.minimap_pixels(), self.colour, span=self.rules.minimap_span
                )
                if territory is not None and self.territory is not None:
                    territory_reward = territory - self.territory
                if territory is not None:
                    self.territory = territory
            if self.arena_log is not None:
                territory_reward = log_change
            reward += territory_reward
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
                "territory": territory,
                "territory_reward": territory_reward,
                "reward_source": self.reward_source,
                "potential": self.potential,
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
        # Each side's own reset stamped its deadline and last_time. Copying only the
        # timeout clock left the early side's first elapsed covering the whole lobby
        # skew, and the two action intervals stayed a tick apart after that.
        # __dict__ so a stand-in that never stamped a deadline still syncs from start.
        deadline = max(e.__dict__.get("deadline", e.start) for e in self.envs)
        now = time.monotonic()
        for env in self.envs:
            env.start = start
            env.last_time = now
            env.deadline = max(deadline, now)
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
                        outcomes[i] = env.confirm_outcome(screen)
                        if (
                            outcomes[i] is None
                            and time.monotonic() - env.start >= env.seconds
                            and screen.matches("healthy")
                        ):
                            outcomes[i] = "draw"
                        if env.recorder:
                            env.recorder.append(frame, confirmation=True, outcome=outcomes[i])
                        info = {**results[i][4], "outcome": outcomes[i]}
                        # The confirming frame replaces the step reward. Keep the
                        # territory change already measured on that step.
                        reward = {"win": 1.0, "loss": -1.0}.get(outcomes[i], 0.0)
                        reward += float(info.get("territory_reward") or 0.0)
                        reward += getattr(env, "unpaid", 0.0)
                        env.unpaid = 0.0
                        results[i] = (
                            frame.rgb,
                            reward,
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
