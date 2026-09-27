"""A curriculum that skips the setup: games that start from saves where part of the setup
is already done, so that a learned player meets the war before it can set it up, and the
start moves earlier, rung by rung, until it does the whole setup itself.

Every learned game so far lost in its setup (0 of 14 by 2026-09-26: no army, no general or
no front), so no learned game has yet shown whether the policy can fight a war once it has
one. A rung save answers that directly. Each is the arena's start save (paused at 12:00 on
1 January 1936, before the war is declared) with the scripted player's first setup steps
done while still paused, and saved there, so the game's date, divisions and manpower are
exactly the start's:

    S0  the start save itself;
    S1  the army formed from every division (no general);
    S2  the army with its general (no front);
    S3  the army, its general, its front along the border and a broad offensive drawn:
        only running the game, the conscription laws, executing the plan and the war remain.

In the arenas every division starts unassigned, so "divisions exist, no armies" is S0; S1
is the army without its general instead. A game from a rung declares its war afresh
(start_game's coin), exactly as a game from the start save does.

Rung saves are named after the last step they hold, with the start save's name after it:
"arenav4blu" gives "armyv4blu" (S1), "generalv4blu" (S2) and "frontv4blu" (S3). No name
contains another, so the load dialog's name templates (ai_games.can_load) cannot pick the
wrong one. They are registered per PC in artifacts/arenas/rungs-<station>.json, as
{arena: {country: {rung: save}}}.

`make_ladder` makes them (the scripted player's steps, a console `savegame` after each),
`rung_games` plays full games from them, by a checkpoint or by the scripted player, and
reports each rung's wins and setup, and `promote` is the rule for moving a checkpoint down
the ladder.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RUNGS = ("S0", "S1", "S2", "S3")
# The setup's steps each rung's save holds done.
DONE = {
    "S0": (),
    "S1": ("army",),
    "S2": ("army", "general"),
    "S3": ("army", "general", "front", "offensive"),
}
# The word a rung's save is named with: its last step.
WORDS = {"S1": "army", "S2": "general", "S3": "front"}
# The scripted player's order that completes each rung, in a recorded full game
# (scripted.Planner.orders): the moment a rung save stands for.
ORDERS = {"S1": "army", "S2": "general", "S3": "offensive"}
MAIN_ARENA = "arena-12x8-v4"
# What a game from each rung is scored on, beyond its winner (RungWatch): the army, its
# general, its front, the plan executed (the execute arrow lit) and the game running.
WATCHED = ("army", "general", "front", "execute", "running")
# Who may have done a step for the setup to count as complete: the rung's save, the
# policy, the practice coach or the scripted player.
DOERS = ("rung", "policy", "coach", "scripted")
# The promotion rule's defaults (promote): a checkpoint moves one rung down once it has
# won at least this share of its last games at its rung, over at least this many.
PROMOTE_SHARE, PROMOTE_GAMES = 0.6, 10


def registry_path(station):
    return Path("artifacts/arenas") / f"rungs-{station}.json"


def start_saves_path(station):
    return Path("artifacts/arenas") / f"saves-{station}.json"


def rung_save(start, rung):
    """The name of `rung`'s save made from start save `start` (letters and digits only)."""
    if rung not in RUNGS:
        raise ValueError(f"a rung is one of {RUNGS}, not {rung}")
    if rung == "S0":
        return start
    stem = start[len("arena") :] if start.startswith("arena") else start
    return WORDS[rung] + stem


def default_start(arena, country):
    """The start save's name practice.drill_saves and the load dialog's templates expect:
    arenav4blu and arenav4red on the main arena, else ai_games.save_name's."""
    from .ai_games import save_name

    if arena == MAIN_ARENA:
        return f"arenav4{country.lower()}"
    return save_name(arena, country)


def start_saves(station):
    """{(arena, country): start save} on `station` ("here" or "peer"): its registry, with the
    main arena's hand-made saves (arenav4blu, arenav4red) where it names none."""
    from .ai_games import known_saves

    saves = known_saves(start_saves_path(station))
    for country in ("BLU", "RED"):
        saves.setdefault((MAIN_ARENA, country), default_start(MAIN_ARENA, country))
    return saves


def ladder(station):
    """{(arena, country, rung): save} for every rung save made on `station`, S0 included."""
    path = registry_path(station)
    found = {(a, c, "S0"): s for (a, c), s in start_saves(station).items()}
    if path.exists():
        for arena, sides in json.loads(path.read_text()).items():
            for country, rungs in sides.items():
                for rung, save in rungs.items():
                    found[(arena, country, rung)] = save
    return found


def remember_rung(path, arena, country, rung, save):
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(arena, {}).setdefault(country, {})[rung] = save
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def rung_start(manifest, times, rung):
    """When a recorded game reaches `rung`, on the recording's clock (ns): the frame of the
    scripted player's order that completes it (ORDERS). A game that started from a rung
    save (its manifest's `rung`), a game without the order and S0 start at their first
    frame. Decisions before it are the setup the rung skips (dataset.session_labels)."""
    if rung in (None, "S0") or manifest.get("rung"):
        return int(times[0])
    wanted = ORDERS[rung]
    for order in manifest.get("orders") or []:
        if order.get("order") == wanted:
            return int(times[min(max(int(order["frame"]), 0), len(times) - 1)])
    if rung == "S3":
        # A game whose plan drew no offensive ("front" attacks): its front completes it.
        for order in manifest.get("orders") or []:
            if order.get("order") == "front":
                return int(times[min(max(int(order["frame"]), 0), len(times) - 1)])
    return int(times[0])


def setup_complete(game):
    """Whether a game from a rung completed the setup: an army with a general, a front, and
    its plan executed (the arrow seen lit), counting the steps its rung save held. The
    shared measure, practice.setup_complete, asks the policy to have done every step
    itself, which a game from S1-S3 cannot; summary reports both."""
    steps = (game.get("setup") or {}).get("steps") or {}
    need = ("army", "general", "front", "execute")
    return all((steps.get(step) or {}).get("by") in DOERS for step in need)


def parse_saves(listing):
    """The save names in the worker's `saves` listing (Game-Control.ps1: a table of name,
    MB and time, newest first; "saves: none" when there are none)."""
    names = []
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] not in ("name", "----") and not parts[0].startswith("-"):
            names.append(parts[0])
    return names


def saves_on(station):
    """The save games on `station`'s PC, by name."""
    with station.connect(attach=False) as desk:
        return parse_saves(desk.saves())


def make_rung(desk, planner, rung, country):
    """The scripted player's steps from the rung before `rung` up to it, on the paused game.
    Raises RuntimeError when a step did not take (checked on the screen)."""
    from .ai_games import screen
    from .practice import army_card
    from .scripted import plan_shown

    if rung == "S1":
        planner.form_army(desk)
        army, _ = army_card(screen(desk))
        if not army or planner.find(screen(desk), "unassigned") is not None:
            raise RuntimeError("the army did not form from every division")
    elif rung == "S2":
        if not planner.assign_general(desk):
            raise RuntimeError("no general could be assigned")
    elif rung == "S3":
        planner.draw_front(desk)
        planner.draw_offensive(desk)
        if not plan_shown(screen(desk)):
            raise RuntimeError("no plan shows on the army's card")


def save_game(desk, name):
    """The paused game saved as `name` from the console (single player, debug mode)."""
    from .ai_games import console

    console(desk, f"savegame {name}")
    time.sleep(3)


def make_ladder(
    output,
    *,
    peer=None,
    arenas=(MAIN_ARENA,),
    countries=("BLU", "RED"),
    rungs=("S1", "S2", "S3"),
    rules="artifacts/calibration-1080p/rules.json",
    seed=None,
    fresh_starts=False,
):
    """The rung saves of every arena and country on a PC (the second PC with `peer`, else
    this one), made from each start save by the scripted player's own setup steps while the
    game stays paused, a save after each. An arena and side with no start save gets one
    first, through the menus (start_game's save_as), registered in saves-<station>.json;
    with `fresh_starts`, every arena and side does (a start save copied from another PC
    need not load: one showed a red "!" in the load dialog and stayed unloaded). Each rung's screen is kept in `output`, and ladder.json there lists what was made."""
    from PIL import Image

    from .ai_games import (
        EVENT_OK,
        Station,
        focus,
        known_saves,
        remember_save,
        screen,
        start_game,
    )
    from .practice import load_or_launch
    from .scripted import TEMPLATES, Planner, best_plan, load_templates
    from .vision import ScreenRules

    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)
    name = "peer" if peer else "here"
    station = Station(name, peer)
    screen_rules = ScreenRules(rules)
    ok = [np.asarray(Image.open(path).convert("RGB")) for path in (
        "artifacts/screens-1080p/ok-button.png", EVENT_OK)]  # fmt: skip
    buttons = load_templates(TEMPLATES)
    rng = random.Random(seed)
    registry = registry_path(name)
    made, running = [], None
    order = sorted(rungs, key=RUNGS.index)
    try:
        for arena in arenas:
            for country in countries:
                entry = {"arena": arena, "country": country, "rungs": {}}
                failure_shot = out_root / f"{arena}-{country}-start-failed.png"
                try:
                    base = start_saves(name).get((arena, country)) or default_start(arena, country)
                    if fresh_starts:
                        base = default_start(arena, country)
                    fresh = fresh_starts or base not in saves_on(station)
                    if fresh:
                        # No start save yet: through the menus, saving the start first.
                        station.quit()
                        station.launch(arena)
                        time.sleep(25)
                        loaded = False
                    else:
                        loaded = load_or_launch(
                            station, arena, base, running == arena, ok, screen_rules,
                            failure_shot,
                        )  # fmt: skip
                    running = None
                    entry["start_save"] = base
                    with station.connect() as desk:
                        if not focus(desk):
                            raise RuntimeError("could not bring the game window to the front")
                        start_game(
                            desk, screen_rules, failure_shot, country, 5, observe=False,
                            saved=not fresh, save_as=base if fresh else None,
                        )  # fmt: skip
                        if (
                            fresh
                            and known_saves(start_saves_path(name)).get((arena, country)) != base
                        ):
                            remember_save(start_saves_path(name), arena, country, base)
                        plan = {**best_plan(rng), "attack": "broad"}
                        planner = Planner(country, plan, buttons, screen_rules, 5, lambda: 0,
                                          rng=rng)  # fmt: skip
                        planner.debug_dir = out_root
                        for rung in RUNGS[1 : RUNGS.index(order[-1]) + 1]:
                            make_rung(desk, planner, rung, country)
                            if rung not in order:
                                continue
                            save = rung_save(base, rung)
                            save_game(desk, save)
                            remember_rung(registry, arena, country, rung, save)
                            shot = out_root / f"{save}.jpg"
                            Image.fromarray(screen(desk)).resize((960, 540)).save(shot)
                            entry["rungs"][rung] = save
                            log.info("[%s] %s %s %s saved as %s", name, arena, country, rung, save)
                    entry["loaded_in_game"] = loaded
                    running = arena
                except Exception as error:  # noqa: BLE001 - reported, then the next side.
                    entry["error"] = f"{type(error).__name__}: {error}"
                    log.warning("[%s] %s %s: %s", name, arena, country, entry["error"])
                made.append(entry)
                (out_root / "ladder.json").write_text(json.dumps(made, indent=2))
    finally:
        try:
            station.quit()
        except Exception as error:  # noqa: BLE001 - the saves are made.
            log.warning("quit failed: %s", error)
    return made


class RungWatch:
    """Scores a game from a rung on the screen, never acting: a full screenshot every
    `look_every` seconds until every WATCHED step has shown, each step's first sighting
    and by whom (the rung's save for the steps it holds, else the policy). Plugs into
    play.play_policy_game as its `coach`."""

    def __init__(self, country, rules, rung, *, look_every=3.0, planner=None):
        from .scripted import TEMPLATES, Planner, best_plan, load_templates

        self.rung = rung
        self.planner = planner or Planner(
            country, best_plan(random.Random()), load_templates(TEMPLATES), rules, 5, lambda: 0
        )
        self.look_every, self.next_look = look_every, 0.0
        self.done = {s: {"at": 0.0, "by": "rung"} for s in DONE[rung] if s in WATCHED}
        self.coached, self.failures = [], []
        self.frames = lambda: 0

    def attach(self, rec, root=None):
        self.frames = lambda: rec.manifest["frames"]

    def seen(self, step, seconds):
        self.done.setdefault(step, {"at": round(seconds, 1), "by": "policy"})

    def read(self, rgb, running, seconds):
        """What `rgb` shows done (the practice coach's tests, and the lit execute arrow)."""
        from .practice import army_card
        from .scripted import arrow_lit, plan_shown

        army, general = army_card(rgb)
        if army and self.planner.find(rgb, "unassigned") is None:
            self.seen("army", seconds)
        if general:
            self.seen("general", seconds)
        if plan_shown(rgb):
            self.seen("front", seconds)
            waiting = self.planner.find(rgb, "activate", top=0.8) or self.planner.find(
                rgb, "ready", top=0.8
            )
            if running and waiting is None and arrow_lit(rgb):
                self.seen("execute", seconds)

    def look(self, desk, seconds, running):
        from .ai_games import screen

        if running:
            self.seen("running", seconds)
        if seconds < self.next_look or all(s in self.done for s in WATCHED):
            return None
        self.next_look = seconds + self.look_every
        self.read(screen(desk), running, seconds)
        return None

    def take_over(self, desk, step, seconds):  # pragma: no cover - never asked.
        return []

    def score(self):
        steps = {step: self.done.get(step) for step in WATCHED}
        own = sum(1 for v in steps.values() if v and v["by"] == "policy")
        return {"steps": steps, "own": own, "coached": 0, "rung": self.rung}


def wilson(wins, games, z=1.96):
    from .scripted import wilson as scripted_wilson

    return scripted_wilson(wins, games, z)


def summary(results):
    """Per rung: games that ended (a win, a loss or the cap), wins with a 95% interval,
    setups complete, and how often each step showed done (and by the policy itself)."""
    from .practice import setup_rate

    out = {}
    for rung in RUNGS:
        played = [
            r for r in results if r.get("rung") == rung and r.get("winner") and not r.get("reason")
        ]
        if not played:
            continue
        wins = sum(1 for r in played if r["winner"] == r["started_as"])
        low, high = wilson(wins, len(played))
        row = {
            "games": len(played),
            "wins": wins,
            "win_rate": round(wins / len(played), 3),
            "wilson95": [round(low, 3), round(high, 3)],
            "setup_complete": sum(1 for r in played if setup_complete(r)),
            # The shared measure (route 2's): the whole setup by the policy itself.
            "setup_rate_own": setup_rate(played),
            "timeouts": sum(1 for r in played if r["winner"] == "timeout"),
            "seconds_mean": round(float(np.mean([r.get("seconds", 0) for r in played])), 1),
        }
        for step in WATCHED:
            steps = [((r.get("setup") or {}).get("steps") or {}).get(step) or {} for r in played]
            row[step] = f"{sum(1 for s in steps if s.get('by'))}/{len(played)}"
            if any(s.get("by") == "policy" for s in steps):
                row[f"{step}_own"] = sum(1 for s in steps if s.get("by") == "policy")
        by_arena = {}
        for r in played:
            a = by_arena.setdefault(f"{r['arena']}:{r['started_as']}", [0, 0])
            a[0] += r["winner"] == r["started_as"]
            a[1] += 1
        row["by_arena"] = {k: f"{w}/{n}" for k, (w, n) in sorted(by_arena.items())}
        out[rung] = row
    return out


def promote(results, rung, *, share=PROMOTE_SHARE, games=PROMOTE_GAMES):
    """The rung a checkpoint plays next: one down (S3 to S2, ...) once its last `games`
    ended games at `rung` won at least `share` of them; else the same rung. S0 is the
    bottom: promotion there means it plays whole games."""
    ended = [
        r for r in results if r.get("rung") == rung and r.get("winner") and not r.get("reason")
    ]
    last = ended[-games:]
    if len(last) < games:
        return rung
    won = sum(1 for r in last if r["winner"] == r["started_as"]) / len(last)
    if won >= share and rung != "S0":
        return RUNGS[RUNGS.index(rung) - 1]
    return rung


def success(game, setup_credit=0.25):
    """A game's success for the frontier scheduler, in [0, 1]: 1 for a win, `setup_credit`
    for a completed setup without one (wins are rare, and a setup done is progress), else
    0. None for a game that did not end (an error, a crash)."""
    if not game.get("winner") or game.get("reason"):
        return None
    if game["winner"] == game.get("started_as"):
        return 1.0
    return setup_credit if setup_complete(game) else 0.0


def frontier(
    history,
    cells,
    *,
    rng,
    target=0.5,
    prior=(1.0, 1.0),
    window=5,
    progress_weight=1.0,
    setup_credit=0.25,
):
    """The (rung, arena, country) cell to play next, and why: an adaptive curriculum after
    Self-Play Pretraining with Zero Data (arXiv 2609.30063), whose generator proposes
    tasks at the learner's frontier. Each cell's chance of success (success) is a Beta
    posterior over its games in `history` from `prior`; a draw from it (Thompson sampling)
    scores how near `target` the student sits there (1 at the target, 0 at certainty
    either way), plus `progress_weight` times its learning progress: the change in mean
    success between its last `window` games and the `window` before (an unplayed or
    barely played cell counts as full progress, so each gets tried). The highest score
    wins. Returns (cell, {cell: its numbers}) so the choice can be logged."""
    scores = {}
    for cell in cells:
        rung, arena, country = cell
        outcomes = [
            s
            for g in history
            if (g.get("rung"), g.get("arena"), g.get("started_as")) == cell
            and (s := success(g, setup_credit)) is not None
        ]
        won = sum(outcomes)
        a, b = prior[0] + won, prior[1] + len(outcomes) - won
        draw = rng.betavariate(a, b)
        near = 1.0 - abs(draw - target) / max(target, 1 - target)
        if len(outcomes) < 2 * window:
            progress = (
                1.0
                if len(outcomes) < window
                else abs(float(np.mean(outcomes[-window:])) - float(np.mean(outcomes[:-window])))
            )
        else:
            progress = abs(
                float(np.mean(outcomes[-window:])) - float(np.mean(outcomes[-2 * window : -window]))
            )
        scores[cell] = {
            "games": len(outcomes), "mean": round(a / (a + b), 3), "draw": round(draw, 3),
            "near": round(near, 3), "progress": round(progress, 3),
            "score": round(near + progress_weight * progress, 3),
        }  # fmt: skip
    best = max(cells, key=lambda c: scores[c]["score"])
    return best, scores


def load_history(paths):
    """Every game of earlier rung-games.json files (globs), oldest file first."""
    import glob

    games = []
    for pattern in paths or []:
        for path in sorted(glob.glob(pattern)):
            try:
                games.extend(json.loads(Path(path).read_text()).get("games", []))
            except (OSError, ValueError):
                continue
    return games


def game_order(arenas, countries, rungs, games, block=4):
    """(arena, country, rung) of each game: `block` games in a row on an arena (a load
    inside the game is ~10 s, a launch minutes), countries alternating, rungs interleaved
    so every rung is measured under the same conditions."""
    order = []
    for index in range(games):
        arena = arenas[(index // block) % len(arenas)]
        country = countries[index % len(countries)]
        rung = rungs[(index // len(countries)) % len(rungs)]
        order.append((arena, country, rung))
    return order


def rung_games(
    checkpoint,
    output,
    *,
    peer=None,
    rungs=("S3",),
    arenas=(MAIN_ARENA,),
    countries=("BLU", "RED"),
    games=10,
    minutes=60.0,
    cap_minutes=10.0,
    setup_seconds=None,
    block=4,
    opening=(0.0, 14.0),
    held_previous=False,
    temperature=1.0,
    pointer_temperature=None,
    point=False,
    fast=False,
    rules="artifacts/calibration-1080p/rules.json",
    model_path=None,
    seed=None,
    schedule="fixed",
    history=(),
):
    """Full games from rung saves (the second PC with `peer`, else this one): the learned
    `checkpoint` playing, or the scripted player for `checkpoint` "scripted" (the ceiling of
    each rung, and demonstrations that start at it). Each game loads its rung save (inside
    the running game when its name is calibrated), declares the war by a coin, runs a
    few opening hours at speed 1, and plays to a surrender or `cap_minutes`. A policy's
    game is scored by RungWatch; the harness runs a game the policy has not started by
    `setup_seconds` (30 s from S3, where only running it remains, else 90 s).

    `schedule` "fixed" plays game_order (the ladder's baseline); "adaptive" asks frontier
    before each game which (rung, arena, country) to play, from this session's games and
    those of the rung-games.json files `history` names (globs), keeping to the arena in
    play for `block` games, and each game keeps the choice and every cell's numbers
    (`chosen`).

    Writes rung-games.json in `output`: every game, the summary per rung (summary) and the
    rung each rung's record promotes to (promote). Returns the same."""
    from PIL import Image

    from .ai_games import (
        EVENT_OK,
        Popups,
        Station,
        focus,
        log_end,
        play,
        run_briefly,
        start_game,
    )
    from .practice import load_or_launch
    from .scripted import TEMPLATES, best_plan, load_templates
    from .vision import ScreenRules

    scripted = checkpoint == "scripted"
    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)
    name = "peer" if peer else "here"
    station = Station(name, peer)
    screen_rules = ScreenRules(rules)
    ok = [np.asarray(Image.open(path).convert("RGB")) for path in (
        "artifacts/screens-1080p/ok-button.png", EVENT_OK)]  # fmt: skip
    rng = random.Random(seed)
    saves = ladder(name)
    wanted = [(a, c, r) for a in arenas for c in countries for r in rungs]
    missing = [w for w in wanted if w not in saves]
    if missing:
        raise ValueError(f"no rung save on {name} for {missing} (make-ladder first)")
    actor = None
    if not scripted:
        from .play import play_policy_game, sampling_of
        from .runner import Actor

        actor = Actor(
            checkpoint, model_path, game_speed=5, temperature=temperature,
            pointer_temperature=pointer_temperature, point=point, lean=True, fast=fast,
        )  # fmt: skip
        actor.held_previous = actor.held_previous or held_previous
    buttons = load_templates(TEMPLATES) if scripted else None
    results, running = [], None
    end = time.monotonic() + minutes * 60

    def write():
        report = {"summary": summary(results), "promote": {
            r: promote(results, r) for r in rungs}, "games": results}  # fmt: skip
        (out_root / "rung-games.json").write_text(json.dumps(report, indent=2))
        return report

    if schedule not in ("fixed", "adaptive"):
        raise ValueError("schedule is fixed or adaptive")
    fixed = game_order(list(arenas), list(countries), list(rungs), games, block)
    past = load_history(history)
    try:
        for index in range(games):
            if time.monotonic() + (cap_minutes + 2) * 60 > end:
                break
            chosen = None
            if schedule == "fixed":
                arena, country, rung = fixed[index]
            else:
                # A new arena only every `block` games: a load inside the game is ~10 s.
                keep = results[-1]["arena"] if results and index % block else None
                cells = [(r, a, c) for r in rungs for a in arenas for c in countries
                         if keep is None or a == keep]  # fmt: skip
                (rung, arena, country), scores = frontier(past + results, cells, rng=rng)
                chosen = {"cell": [rung, arena, country],
                          "scores": {"/".join(k): v for k, v in scores.items()}}  # fmt: skip
                log.info("[%s] frontier picks %s %s %s: %s", name, rung, arena, country,
                         scores[(rung, arena, country)])  # fmt: skip
            save = saves[(arena, country, rung)]
            kind = "scripted" if scripted else "policy"
            game = time.strftime(f"rung-{kind}-{name}-%Y%m%d-%H%M%S")
            entry = {"game": game, "station": name, "arena": arena, "started_as": country,
                     "rung": rung, "start_save": save, "declare_drawn": rng.choice(("BLU", "RED")),
                     "opening": round(rng.uniform(*opening), 2) if opening else 0.0,
                     "schedule": schedule}  # fmt: skip
            if chosen:
                entry["chosen"] = chosen
            if actor is not None:
                entry.update(checkpoint=actor.digest, **sampling_of(actor))
            failure_shot = out_root / f"{game}-start-failed.png"
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
                    start_game(desk, screen_rules, failure_shot, country, 5, observe=False,
                               saved=True)  # fmt: skip
                    if entry["opening"]:
                        run_briefly(desk, screen_rules, entry["opening"])
                    start_game_declare(desk, entry["declare_drawn"])
                    if scripted:
                        plan = best_plan(rng)
                        entry["plan"] = plan
                        player = {"plan": plan, "templates": buttons, "rules": screen_rules,
                                  "done": DONE[rung]}  # fmt: skip
                        settings = {"hz": 5, "codec": "nvenc-hevc",
                                    "mod": arena, "cap_minutes": cap_minutes,
                                    "camera_kicks": None}  # fmt: skip
                        outcome, reason, manifest = play(
                            desk, out_root / game, Popups(ok, rng=rng), settings, name,
                            country=country, speed=5, player=player, start_save=save,
                            log_from=log_from,
                        )  # fmt: skip
                    else:
                        watch = RungWatch(country, screen_rules, rung)
                        wait = setup_seconds or (30.0 if rung == "S3" else 90.0)
                        outcome, reason, manifest = play_policy_game(
                            desk, actor, out_root / game, rules=screen_rules, country=country,
                            station=name, cap_minutes=cap_minutes, setup_seconds=wait,
                            arena_name=arena, coach=watch, log_from=log_from,
                        )  # fmt: skip
                    stamp_rung(out_root / game, rung, save)
            except Exception as error:  # noqa: BLE001 - reported, then the next game.
                entry["error"] = f"{type(error).__name__}: {error}"
                log.warning("[%s] %s failed: %s", name, game, entry["error"])
            else:
                entry.update(
                    winner=outcome, reason=reason, seconds=manifest.get("seconds"),
                    frames=manifest.get("frames"), declarer=manifest.get("declarer"),
                    complete=manifest.get("complete"), setup=manifest.get("setup"),
                    milestones=manifest.get("milestones"), orders=len(manifest.get("orders") or []),
                    planner_errors=len(manifest.get("planner_errors") or []),
                )  # fmt: skip
                if scripted:
                    entry["setup"] = scripted_setup(manifest, rung)
                entry["setup_complete"] = setup_complete(entry)
                running = arena if reason is None else None
                log.info("[%s] %s (%s %s %s): winner %s after %s s", name, game, arena, country,
                         rung, outcome, entry["seconds"])  # fmt: skip
            results.append(entry)
            write()
    finally:
        try:
            station.quit()
        except Exception as error:  # noqa: BLE001 - the games are saved.
            log.warning("quit failed: %s", error)
    return write()


def start_game_declare(desk, declarer):
    """The war declared by `declarer`, as start_game does (its event, then closed)."""
    from .ai_games import DECLARE_EVENT, close_event, console

    console(desk, f"event {DECLARE_EVENT[declarer]}")
    close_event(desk)


def stamp_rung(root, rung, save):
    """The recording's manifest names the rung it started from (rung_start reads it)."""
    path = Path(root) / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        manifest.update(rung=rung, rung_save=save)
        path.write_text(json.dumps(manifest, indent=2))


def scripted_setup(manifest, rung):
    """A scripted game's setup, in RungWatch's terms, from its orders."""
    steps = {s: {"at": 0.0, "by": "rung"} for s in DONE[rung] if s in WATCHED}
    kinds = {o.get("order") for o in manifest.get("orders") or []}
    for step, order in (("army", "army"), ("general", "general"), ("front", "front"),
                        ("execute", "activate"), ("running", "run")):  # fmt: skip
        if order in kinds:
            steps.setdefault(step, {"at": None, "by": "scripted"})
    return {"steps": {s: steps.get(s) for s in WATCHED}, "rung": rung}
