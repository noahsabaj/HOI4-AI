"""Practice at the setup: short games of the learned player from the start save, each scored
by what its setup got done, with the scripted player standing by as a coach.

A live test gives two games a training run, and each learned game so far lost in its
setup (0 of 12 by 2026-09-26: an army formed with no front, a front with no army, the
camera scrolled off the map). Practice asks the question that fails, many times: from the
paused start, does the policy form the army, give it a general, draw its front and get the
game running? An episode lasts `seconds` and the next loads the save from inside the game
(ai_games.load_in_game, ~10 s) instead of launching HOI4 again (~3 minutes).

The coach is the scripted player (scripted.Planner), steps of whose setup it takes over
when the policy falls behind: a step not done by its deadline (about twice the scripted
player's own time), or the camera off the arena for `drift_seconds`. It hands back once
the step is done. What it does is recorded as the recording's inputs, tagged "by": "coach",
and the manifest names the frames it held (`coached`), so training on practice games
learns from the coach's steps only: states the policy led the game into, and the way
out of them (HG-DAgger, Kelly et al. 2019, arXiv 1810.02890). The policy's own decisions
in them weigh nothing (dataset.session_labels).
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# The setup's steps, in order.
STEPS = ("army", "general", "front", "running")
# When the coach takes a step over: about twice the scripted player's time for it from the
# start saves (army ~6 s, general ~12 s, front 18-28 s, running 31-40 s, 2026-09-25).
DEADLINES = {"army": 20.0, "general": 35.0, "front": 60.0}
MAIN_ARENA = "arena-12x8-v4"


class Coach:
    """Watches the policy's setup on a full screenshot every `look_every` seconds, and says
    which step to take over (`look`); takes it over (`take_over`); scores the setup
    (`score`). With `intervene` off, it only watches and scores."""

    def __init__(
        self,
        country,
        rules,
        *,
        plan=None,
        intervene=True,
        deadlines=DEADLINES,
        look_every=3.0,
        drift_seconds=6.0,
        rng=None,
        planner=None,
    ):
        from .scripted import TEMPLATES, Planner, best_plan, load_templates

        self.rng = rng or random.Random()
        self.plan = plan or best_plan(self.rng)
        self.frames = lambda: 0
        # `planner` stands in for the scripted player's (tests: its screen templates are
        # calibration files, not in the repository).
        self.planner = planner or Planner(
            country, self.plan, load_templates(TEMPLATES), rules, 5, lambda: self.frames()
        )
        self.intervene, self.deadlines = intervene, dict(deadlines)
        self.look_every, self.drift_seconds = look_every, drift_seconds
        # When each step was first seen done, and whether the policy or the coach did it.
        self.done = {}
        self.coached = []
        self.failures = []
        self.next_look = 0.0
        self.lost_since = None

    def attach(self, rec):
        """The recording whose frames stamp the coach's orders and spans."""
        self.frames = lambda: rec.manifest["frames"]

    def seen(self, step, seconds, by="policy"):
        if step not in self.done:
            self.done[step] = {"at": round(seconds, 1), "by": by}

    def look(self, desk, seconds, running):
        """The step the coach should take over now, or None. A full screenshot at most
        every `look_every` seconds; what it shows done stays done."""
        from .ai_games import MAP_BOTTOM, MAP_TOP, screen
        from .scripted import plan_shown
        from .vision import country_pixels

        if running:
            self.seen("running", seconds)
        if seconds < self.next_look:
            return None
        self.next_look = seconds + self.look_every
        rgb = screen(desk)
        find = self.planner.find
        if find(rgb, "unassigned") is None:
            self.seen("army", seconds)
        if find(rgb, "plans_bar") is not None and find(rgb, "no_commander") is None:
            # The army's panel is open, and its commander slot is filled.
            self.seen("general", seconds)
        if plan_shown(rgb):
            self.seen("front", seconds)
        blue, red = country_pixels(rgb[MAP_TOP : rgb.shape[0] - MAP_BOTTOM])
        if blue is None and red is None:
            self.lost_since = seconds if self.lost_since is None else self.lost_since
        else:
            self.lost_since = None
        if not self.intervene:
            return None
        if self.lost_since is not None and seconds - self.lost_since >= self.drift_seconds:
            return "camera"
        for step in STEPS:
            if step in self.done:
                continue
            if step in self.deadlines and seconds >= self.deadlines[step]:
                return step
            # A later step waits for this one.
            return None
        return None

    def take_over(self, desk, step, seconds):
        """The scripted player does `step`. Returns its inputs, tagged "by": "coach"."""
        from .ai_games import Logged, recentre

        logged = Logged(desk)
        began = self.frames()
        try:
            if step == "camera":
                recentre(logged)
                self.lost_since = None
            elif step == "army":
                self.planner.form_army(logged)
            elif step == "general":
                if not self.planner.assign_general(logged):
                    raise RuntimeError("no commander could be assigned")
            elif step == "front":
                self.planner.draw_front(logged)
            if step in STEPS:
                self.seen(step, seconds, by="coach")
        except Exception as error:  # noqa: BLE001 - noted; the policy goes on.
            self.failures.append({"step": step, "at": round(seconds, 1), "error": str(error)})
            log.info("the coach could not %s: %s", step, error)
            # Not asked again for this step: it is late either way.
            self.deadlines.pop(step, None)
            if step in STEPS:
                self.seen(step, seconds, by="nobody")
        self.coached.append({"from_frame": began, "to_frame": self.frames(), "step": step})
        return [{**item, "by": "coach"} for item in logged.take()]

    def score(self):
        """Per step: when it was done and by whom, or None; the policy's own share."""
        steps = {step: self.done.get(step) for step in STEPS}
        own = sum(1 for v in steps.values() if v and v["by"] == "policy")
        return {"steps": steps, "own": own, "coached": len(self.coached)}


def summary(results):
    """How often the policy did each step itself, over the episodes that played."""
    played = [r for r in results if r.get("setup")]
    out = {"episodes": len(played)}
    for step in STEPS:
        own = sum(1 for r in played if (r["setup"]["steps"].get(step) or {}).get("by") == "policy")
        out[step] = f"{own}/{len(played)}"
    out["own_steps_mean"] = (
        round(float(np.mean([r["setup"]["own"] for r in played])), 2) if played else None
    )
    return out


def practice(
    checkpoint,
    output,
    *,
    peer,
    episodes=20,
    minutes=60.0,
    seconds=90.0,
    countries=("BLU", "RED"),
    coach=True,
    reservation=None,
    held_previous=False,
    temperature=1.0,
    rules="artifacts/calibration-1080p/rules.json",
    model_path=None,
    seed=None,
):
    """Up to `episodes` practice episodes of `seconds` each (or until `minutes` run out) on
    the second PC, from the main arena's start saves, alternating countries. Returns the
    episodes and the summary (practice-peer.json in `output`)."""
    from PIL import Image

    from .ai_games import (
        EVENT_OK,
        Station,
        can_load,
        focus,
        load_in_game,
        log_end,
        start_game,
    )
    from .play import hand_back, play_policy_game, reserve, room_for_a_game
    from .runner import Actor
    from .vision import ScreenRules

    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)
    screen_rules = ScreenRules(rules)
    ok = [np.asarray(Image.open(path).convert("RGB")) for path in (
        "artifacts/screens-1080p/ok-button.png", EVENT_OK)]  # fmt: skip
    rng = random.Random(seed)
    if reservation:
        reserve(reservation, minutes)
    station = Station("peer", peer)
    end = time.monotonic() + minutes * 60
    results, running = [], False
    try:
        actor = Actor(checkpoint, model_path, game_speed=5, temperature=temperature)
        actor.lean = True
        actor.held_previous = actor.held_previous or held_previous
        for index in range(episodes):
            if time.monotonic() + seconds + 60 > end:
                break
            country = countries[index % len(countries)]
            save = f"arenav4{country.lower()}"
            name = time.strftime("practice-peer-%Y%m%d-%H%M%S")
            entry = {"game": name, "station": "peer", "started_as": country, "arena": MAIN_ARENA,
                     "start_save": save, "checkpoint": actor.digest, "coach": coach}  # fmt: skip
            failure_shot = out_root / f"{name}-start-failed.png"
            try:
                log_from, loaded = None, False
                if running and can_load(save):
                    try:
                        with station.connect() as desk:
                            focus(desk)
                            load_in_game(desk, save, ok, screen_rules, failure_shot)
                        loaded = True
                    except Exception as error:  # noqa: BLE001 - launched afresh instead.
                        log.info("[peer] loading in the game failed (%s): launching", error)
                if not loaded:
                    station.quit()
                    if not room_for_a_game(station):
                        raise RuntimeError("the second PC has no commit room for a game")
                    station.launch(MAIN_ARENA, save=save)
                entry["loaded_in_game"] = loaded
                running = False
                with station.connect() as desk:
                    if not focus(desk):
                        raise RuntimeError("could not bring the game window to the front")
                    if loaded:
                        # Where this game's log lines begin: the last game's surrender is
                        # in the log before them (ai_games.log_end).
                        log_from = log_end(desk)
                    start_game(
                        desk, screen_rules, failure_shot, country, 5, observe=False,
                        declarer=rng.choice(("BLU", "RED")), saved=True,
                    )  # fmt: skip
                    watcher = Coach(country, screen_rules, intervene=coach, rng=rng)
                    outcome, reason, manifest = play_policy_game(
                        desk, actor, out_root / name, rules=screen_rules, country=country,
                        cap_minutes=seconds / 60, setup_seconds=seconds, arena_name=MAIN_ARENA,
                        coach=watcher, log_from=log_from,
                    )  # fmt: skip
            except Exception as error:  # noqa: BLE001 - reported, then the next episode.
                entry["error"] = f"{type(error).__name__}: {error}"
                log.warning("[peer] %s failed: %s", name, entry["error"])
            else:
                entry.update(
                    setup=watcher.score(),
                    coached=watcher.coached,
                    coach_failures=watcher.failures,
                    reason=reason,
                    frames=manifest["frames"],
                    forced_releases=manifest.get("forced_releases"),
                    milestones=manifest.get("milestones"),
                )
                running = reason is None
                log.info("[peer] %s: %s", name, json.dumps(entry["setup"]))
            results.append(entry)
            (out_root / "practice-peer.json").write_text(
                json.dumps({"summary": summary(results), "episodes": results}, indent=2)
            )
    finally:
        try:
            station.quit()
        except Exception as error:  # noqa: BLE001 - the episodes are saved.
            log.warning("quit failed: %s", error)
        if reservation:
            hand_back(reservation, {"episodes": len(results), "summary": summary(results)})
    return {"summary": summary(results), "episodes": results}
