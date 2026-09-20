from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import gymnasium as gym
import numpy as np

from .actions import GRID, PERIOD, SLOTS, VOCAB, decode
from .desktop import DesktopError
from .vision import run_setup


def disarm(env):
    env.active = False
    try:
        env.desktop.release()
    except (DesktopError, OSError):
        pass  # The independent worker watchdog also releases on connection loss.


class ArenaEnv(gym.Env):
    """Real-time transitions: elapsed time includes policy latency; never pause the game.

    Five Hz is the requested cadence, not an assumed property. Overruns are measured.
    A slow policy must fail the runtime gate, not silently receive a faster game clock.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, desktop, rules, recipe, seconds=1800, recorder=None):
        rules.require_match_rules()
        self.desktop, self.rules, self.recipe = desktop, rules, recipe
        self.seconds, self.recorder = seconds, recorder
        self.action_space = gym.spaces.MultiDiscrete(np.tile([len(VOCAB), GRID, GRID], (SLOTS, 1)))
        self.observation_space = gym.spaces.Box(0, 255, (rules.height, rules.width, 3), np.uint8)
        self.active = False
        self.last = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.desktop.release()
        run_setup(self.desktop, self.rules, self.recipe)
        self.rules.last, self.rules.count = None, 0
        self.last = self.desktop.capture()
        self.active = True
        self.desktop.arm()
        self.start = self.last_time = time.monotonic()
        self.clock_pixels = None
        self.clock_changed = self.start
        return self.last.rgb, {"capture": self.last.meta, "valid": True}

    def step(self, action):
        if not self.active:
            raise RuntimeError("reset must succeed before step")
        if not self.action_space.contains(np.asarray(action)):
            self.desktop.release()
            self.active = False
            raise ValueError("Invalid physical action")
        begin = time.monotonic()
        applied = []
        try:
            for index, token in enumerate(action):
                due = begin + index * PERIOD / SLOTS
                time.sleep(max(0, due - time.monotonic()))
                applied.append(self.desktop.apply(decode(token)))
            time.sleep(max(0, begin + PERIOD - time.monotonic()))
            frame = self.desktop.capture()
            outcome = self.rules.outcome(frame.rgb)
            # A calibrated healthy HUD is required for every nonterminal step and timeout.
            healthy = self.rules.matches("healthy", frame.rgb)
            if healthy and outcome is None:
                if not self.rules.matches("running_speed_two", frame.rgb):
                    raise DesktopError("paused_or_wrong_game_speed")
                x, y, w, h = self.rules.clock_rect
                clock = frame.rgb[y : y + h, x : x + w]
                if clock.size == 0:
                    raise DesktopError("invalid_clock_calibration")
                if self.clock_pixels is None or not np.array_equal(clock, self.clock_pixels):
                    self.clock_changed = time.monotonic()
                    self.clock_pixels = clock.copy()
                elif time.monotonic() - self.clock_changed > 60:
                    raise DesktopError("game_clock_stalled")
            if outcome in {"disconnect", "desync", "invalid"}:
                raise DesktopError(f"visual_fault:{outcome}")
            if outcome is None and not healthy and self.rules.last is None:
                raise DesktopError("unrecognized_match_screen")
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
                "deadline_miss": now - self.last_time > PERIOD * 1.25,
                "capture": frame.meta,
                "applied": applied,
            }
            self.last_time = now
            self.last = frame
            if self.recorder:
                self.recorder.append(frame, action=np.asarray(action).tolist(), transition=info)
            if done or timeout:
                self.desktop.release()
                self.active = False
            return frame.rgb, reward, done, timeout, info
        except (DesktopError, TimeoutError, OSError) as error:
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
        self.desktop.close()


class ArenaPair:
    """Reset barrier and concurrent stepping; observations never cross player boundaries."""

    def __init__(self, left, right):
        self.envs = [left, right]
        self.pool = ThreadPoolExecutor(max_workers=2)

    def reset(self):
        futures = [self.pool.submit(e.reset) for e in self.envs]
        try:
            return [f.result() for f in futures]
        except Exception:
            for e in self.envs:
                disarm(e)
            raise

    def step(self, actions):
        futures = [self.pool.submit(e.step, a) for e, a in zip(self.envs, actions, strict=True)]
        results = [f.result() for f in futures]
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
            expiry = time.monotonic() + 3
            while None in outcomes and time.monotonic() < expiry:
                time.sleep(0.1)
                for i, env in enumerate(self.envs):
                    if outcomes[i] is not None:
                        continue
                    try:
                        frame = env.desktop.capture()
                        outcomes[i] = env.rules.outcome(frame.rgb)
                        if (
                            outcomes[i] is None
                            and time.monotonic() - env.start >= env.seconds
                            and env.rules.matches("healthy", frame.rgb)
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
                    except (DesktopError, OSError):
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
