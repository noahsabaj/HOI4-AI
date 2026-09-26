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
# The first army's card in the bottom bar, at 1080p (scripted.ARMY_CARD), the patch of it
# read (x0, y0, x1, y1), and what tells an army and a general there. With no army the
# spot is the dark gap between two empty "+" slots (mean 39-45); an army's card is lit
# (88), and a general's portrait on it is ~63% skin tones against 0-4% without one
# (2026-09-26, practice episodes). The unassigned alert's absence was no proof: a tooltip
# over the top bar hid it, and an episode scored an army that never was.
CARD = (929, 985, 965, 1025)
CARD_LIT, PORTRAIT_SKIN = 65.0, 0.3


def army_card(rgb):
    """(an army's card is shown, a general's portrait is on it), from its patch."""
    x0, y0, x1, y1 = CARD
    patch = rgb[y0:y1, x0:x1].astype(np.int32)
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
    skin = float(((r > 120) & (r > g + 15) & (g > b)).mean())
    return float(patch.mean()) > CARD_LIT, skin > PORTRAIT_SKIN


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

    def attach(self, rec, root=None):
        """The recording whose frames stamp the coach's orders and spans; `root`, its folder,
        keeps the planner's own view when a step fails (scripted.Planner.keep)."""
        self.frames = lambda: rec.manifest["frames"]
        if root is not None and hasattr(self.planner, "debug_dir"):
            self.planner.debug_dir = Path(root)

    def seen(self, step, seconds, by="policy"):
        if step not in self.done:
            self.done[step] = {"at": round(seconds, 1), "by": by}
            if by == "coach":
                # The scripted player ran without an error, which is not the step done:
                # 29 of 87 front takeovers by bc5's time had left no front. The screen
                # confirms it at a later look, and only a confirmed span teaches.
                self.done[step]["confirmed"] = None
        elif by == "policy" and self.done[step].get("confirmed", True) is None:
            self.done[step]["confirmed"] = round(seconds, 1)

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
        army, general = army_card(rgb)
        # All its divisions, too: bc5 formed armies of one division of eight (a click on the
        # unassigned alert without Shift), and the alert stayed up (2026-09-26).
        if army and self.planner.find(rgb, "unassigned") is None:
            self.seen("army", seconds)
        if general:
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


def coach_managed(done):
    """Whether a step's record (Coach.done) says the coach did it and the screen showed it
    done after. Records from before the check (no "confirmed") count the coach's word."""
    return done.get("by") == "coach" and done.get("confirmed", True) is not None


def summary(results):
    """How often the policy did each step itself, over the episodes that played, and how
    often the coach managed a step it took over (`coach_<step>`)."""
    played = [r for r in results if r.get("setup")]
    out = {"episodes": len(played)}
    for step in STEPS:
        own = sum(1 for r in played if (r["setup"]["steps"].get(step) or {}).get("by") == "policy")
        out[step] = f"{own}/{len(played)}"
    for step in ("army", "general", "front"):
        done = [r["setup"]["steps"].get(step) or {} for r in played]
        tried = sum(1 for d in done if d.get("by") in ("coach", "nobody"))
        managed = sum(1 for d in done if coach_managed(d))
        if tried:
            out[f"coach_{step}"] = f"{managed}/{tried}"
    out["own_steps_mean"] = (
        round(float(np.mean([r["setup"]["own"] for r in played])), 2) if played else None
    )
    return out


def load_or_launch(station, arena, save, running, ok, rules, failure_shot):
    """`save` of `arena` up on the station, paused: loaded from inside the game running
    there (`running`: the same arena, left cleanly) when the save's name is calibrated
    (ai_games.can_load), else HOI4 launched afresh into it. True if it was loaded."""
    from .ai_games import can_load, focus, load_in_game
    from .play import room_for_a_game

    if running and can_load(save):
        try:
            with station.connect() as desk:
                focus(desk)
                load_in_game(desk, save, ok, rules, failure_shot)
            return True
        except Exception as error:  # noqa: BLE001 - launched afresh instead.
            log.info("[peer] loading in the game failed (%s): launching", error)
    station.quit()
    if not room_for_a_game(station):
        raise RuntimeError("the second PC has no commit room for a game")
    station.launch(arena, save=save)
    return False


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

    from .ai_games import EVENT_OK, Station, focus, log_end, start_game
    from .play import hand_back, play_policy_game, reserve
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
                log_from = None
                loaded = load_or_launch(
                    station, MAIN_ARENA, save, running, ok, screen_rules, failure_shot
                )
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


# The scrambles a drill may start from (scramble): the camera knocked off an edge or right
# in (ai_games.kick_camera), a state's panel open from a click on the map, the political
# screen (Q, the learned player's most pressed key), the pointer somewhere else.
SCRAMBLES = ("camera", "click", "panel", "pointer")


def scramble(desk, rng):
    """One to three SCRAMBLES, straight to the desktop (so none is a label). Returns their
    kinds."""
    from .ai_games import ZOOM_MAX, act, click, kick_camera, tap

    kinds = rng.sample(SCRAMBLES, rng.randint(1, 3))
    for kind in kinds:
        if kind == "camera":
            kick_camera(desk, rng, rng.randint(0, ZOOM_MAX))
        elif kind == "click":
            click(desk, rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))
        elif kind == "panel":
            act(desk, tap(0x51))
        else:
            at = {"kind": "move", "x": rng.uniform(0.05, 0.95), "y": rng.uniform(0.05, 0.95)}
            act(desk, [at])
        time.sleep(0.5)
    return kinds


def drill_episode(
    desk,
    root,
    *,
    country,
    rules,
    rng,
    arena,
    scrambled=False,
    after=8.0,
    log_from=None,
    codec="nvenc",
):
    """The scripted player's setup alone, recorded as a scripted game: army, general, front,
    offensive, the game running, then `after` seconds. With `scrambled`, from a start
    scramble() has messed up first, unrecorded: those frames weigh nothing in training
    (the manifest's camera_kicks), and what the recording shows is the way back.

    A setup is most of what the learned player gets wrong, and a whole game holds about
    40 s of it; a drill is little else. A failed step leaves the drill incomplete, which
    training skips. Returns the manifest.
    """
    from .ai_games import Logged, on_screen
    from .arena_log import ArenaLog
    from .recording import open_recorder
    from .scripted import TEMPLATES, Planner, best_plan, load_templates

    inputs = Logged(desk)
    first = on_screen(desk.capture(full=True))
    rec = open_recorder(desk, root, first, game_speed=5, source="scripted", hz=5, codec=codec)
    if not getattr(rec, "streamed", False):
        rec.close(complete=False, reason="a drill needs the worker's stream")
        raise RuntimeError("a drill needs the worker's stream (protocol 2)")
    arena_log = ArenaLog(desk)
    if log_from is not None:
        arena_log.offset = log_from
    plan = best_plan(rng)
    frames = lambda: rec.manifest["frames"]  # noqa: E731
    planner = Planner(country, plan, load_templates(TEMPLATES), rules, 5, frames, rng=rng)
    planner.debug_dir = Path(root)
    kicked, reason, began = [], None, time.monotonic()
    try:
        if scrambled:
            start = frames()
            kinds = scramble(desk, rng)
            kicked.append({"kind": "+".join(kinds), "from_frame": start, "to_frame": frames()})
        planner.setup(inputs)
        time.sleep(after)
    except Exception as error:  # noqa: BLE001 - recorded in the manifest.
        reason = f"{type(error).__name__}: {error}"
        log.info("[peer] drill failed: %s", reason)
    finally:
        rec.append(scripted_events=inputs.take())
        arena_log.poll()
        last = frames()
        Path(root, "arena-log.txt").write_text("\n".join(arena_log.lines) + "\n")
        with Path(root, "arena-log.jsonl").open("w") as out:
            out.writelines(
                json.dumps({"frame": last, "line": line}) + "\n" for line in arena_log.lines
            )
        driver = "the scripted player's setup alone, a drill"
        rec.manifest.update(
            drill=True, started_as=country, arena=arena, winner=None, plan=plan,
            orders=planner.orders, planner_errors=planner.failures, camera_kicks=kicked,
            declarer=arena_log.declarer, players=[country], labels="scripted_events",
            seconds=round(time.monotonic() - began), station="peer",
            driver=driver + (", from a scramble" if scrambled else ""),
        )  # fmt: skip
        rec.close(complete=reason is None, reason=reason)
    return rec.manifest


def drill_saves(registry="artifacts/arenas/saves-peer.json"):
    """{(arena, country): start save} for every arena with one (ai_games.known_saves)."""
    from .ai_games import known_saves

    saves = known_saves(registry)
    for country in ("BLU", "RED"):
        saves.setdefault((MAIN_ARENA, country), f"arenav4{country.lower()}")
    return saves


def drill_order(arenas, countries, episodes, block):
    """Which arena and country each drill plays: `block` in a row on each arena in turn
    (a load from inside the game is ~10 s, a launch ~3 minutes), countries alternating."""
    return [
        (arenas[(index // block) % len(arenas)], countries[index % len(countries)])
        for index in range(episodes)
    ]


def drills(
    output,
    *,
    peer,
    episodes=40,
    minutes=60.0,
    arenas=(MAIN_ARENA,),
    countries=("BLU", "RED"),
    scrambled=0.7,
    block=4,
    after=8.0,
    reservation=None,
    rules="artifacts/calibration-1080p/rules.json",
    seed=None,
):
    """Up to `episodes` drills (drill_episode) on the second PC in drill_order, a `scrambled`
    share of them from a scrambled start. Returns the drills and a summary (drills-peer.json
    in `output`): how many completed, and how many an hour."""
    from PIL import Image

    from .ai_games import EVENT_OK, Station, focus, log_end, start_game
    from .play import hand_back, reserve
    from .vision import ScreenRules

    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)
    screen_rules = ScreenRules(rules)
    ok = [np.asarray(Image.open(path).convert("RGB")) for path in (
        "artifacts/screens-1080p/ok-button.png", EVENT_OK)]  # fmt: skip
    rng = random.Random(seed)
    saves = drill_saves()
    arenas = [a for a in arenas if all((a, c) in saves for c in countries)]
    if not arenas:
        raise ValueError("no arena has start saves for every country")
    if reservation:
        reserve(reservation, minutes)
    station = Station("peer", peer)
    began = time.monotonic()
    end = began + minutes * 60
    results, running = [], None

    def tally():
        done = sum(1 for r in results if r.get("complete"))
        hours = max((time.monotonic() - began) / 3600, 1e-6)
        return {"drills": len(results), "complete": done, "per_hour": round(done / hours, 1)}

    try:
        for arena, country in drill_order(arenas, countries, episodes, block):
            if time.monotonic() + 120 > end:
                break
            save = saves[(arena, country)]
            name = time.strftime("drill-peer-%Y%m%d-%H%M%S")
            entry = {"game": name, "arena": arena, "started_as": country, "start_save": save,
                     "scrambled": rng.random() < scrambled}  # fmt: skip
            failure_shot = out_root / f"{name}-start-failed.png"
            try:
                loaded = load_or_launch(
                    station, arena, save, running == arena, ok, screen_rules, failure_shot
                )
                entry["loaded_in_game"] = loaded
                running = None
                with station.connect() as desk:
                    if not focus(desk):
                        raise RuntimeError("could not bring the game window to the front")
                    log_from = log_end(desk) if loaded else None
                    start_game(
                        desk, screen_rules, failure_shot, country, 5, observe=False,
                        declarer=rng.choice(("BLU", "RED")), saved=True,
                    )  # fmt: skip
                    manifest = drill_episode(
                        desk, out_root / name, country=country, rules=screen_rules, rng=rng,
                        arena=arena, scrambled=entry["scrambled"], after=after, log_from=log_from,
                    )  # fmt: skip
            except Exception as error:  # noqa: BLE001 - reported, then the next drill.
                entry["error"] = f"{type(error).__name__}: {error}"
                log.warning("[peer] %s failed: %s", name, entry["error"])
            else:
                entry.update(
                    complete=manifest["complete"], reason=manifest.get("reason"),
                    frames=manifest["frames"], orders=len(manifest["orders"]),
                )  # fmt: skip
                running = arena
                log.info("[peer] %s: %s", name, entry["reason"] or "complete")
            results.append(entry)
            (out_root / "drills-peer.json").write_text(
                json.dumps({"summary": tally(), "drills": results}, indent=2)
            )
    finally:
        try:
            station.quit()
        except Exception as error:  # noqa: BLE001 - the drills are saved.
            log.warning("quit failed: %s", error)
        if reservation:
            hand_back(reservation, tally())
    return {"summary": tally(), "drills": results}
