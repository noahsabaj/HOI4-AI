"""A learned policy plays one side of an arena game against the game's AI, from pixels.

The policy sees what a player sees, five times a second: the screen, taken by the worker
of the PC the game runs on, with the pointer drawn in. It answers with up to eight mouse
and key events per decision (models.ActionHead), which the worker applies across the next
200 ms in match mode: no console, no pause key, no speed keys. Nothing else reads the
game for it. The arena log is read here only to score the game and to referee it.

The harness does what the match protocol does for any player, and nothing that plays:
it launches the game, picks the country and fires the fair coin for who declares (as
record-ai does for the scripted player), and it runs the game at speed 5. A new game
starts paused, and the scripted player, whose games the policy learned from, sets up
while it is paused and then starts the game: three clicks on the speed control's + and
the space bar. Space is not an input a policy may send, so when the policy clicks + while
the game is paused (its sign that its setup is done), or after `setup_seconds` in any
case, the harness presses space and sets speed 5. It unpauses the game, and sets speed 5
again, if the game ever stops running (no daily report for `stall_seconds`), which a
click on the pause mark or a menu could do. Every such step is counted in the result.

Each game is recorded like the scripted player's (screen.mkv, frames.jsonl with the
policy's applied events as `scripted_events`, arena-log.jsonl), with source "policy", so
the games are data too.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .actions import GRID, PERIOD, SLOTS, decode
from .ai_games import SPEED_UP, Station, act, focus, on_screen, start_game, tap
from .arena_log import ArenaLog
from .desktop import DesktopError, EmergencyStop
from .recording import Recorder

log = logging.getLogger(__name__)

SPACE = 0x20
# The speed control's + at 1080p, and how near a press must land to count as a click on
# it: the button is about 20 px across.
SPEED_BOX = 14
# Clicks on + that raise the speed from 1 to 5, whatever it is: a click at 5 does nothing.
SPEED_CLICKS = 4
# How long after the policy's first click on + the harness starts the game, so that its
# other clicks (the scripted player clicks three times) land first.
RUN_DELAY = 1.0
EVAL = Path("artifacts/eval")


def lattice_to_pixels(x, y, width, height):
    return x / (GRID - 1) * (width - 1), y / (GRID - 1) * (height - 1)


class Dispatcher:
    """Applies a decision's eight slots across one interval, on a thread of its own.

    Keeps each applied event with the worker's time of it, as the recorder stores inputs,
    and follows the pointer, to notice a press on the speed control's +.
    """

    def __init__(self, desk, width=1920, height=1080, clock=time.monotonic):
        self.desk, self.width, self.height, self.clock = desk, width, height, clock
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-dispatch")
        self.future = None
        self.lock = threading.Lock()
        self.applied = []
        self.pointer = None
        self.speed_clicks = 0
        # Intervals in a row the worker refused, as when the game lost focus: the main
        # loop refocuses it, and gives up on the game if it never comes back.
        self.refused = 0
        self.error = None

    def start(self, action, begin, pointer):
        self.pointer = pointer
        self.future = self.pool.submit(self._run, np.asarray(action), begin)

    def _run(self, action, begin):
        try:
            self.desk.arm()
            for index, token in enumerate(action):
                time.sleep(max(0.0, begin + index * PERIOD / SLOTS - self.clock()))
                events = decode(token)
                for event in events:
                    if event["kind"] == "move":
                        self.pointer = lattice_to_pixels(
                            int(token[1]), int(token[2]), self.width, self.height
                        )
                    elif event["kind"] == "button" and event["down"] and self.on_speed_up():
                        self.speed_clicks += 1
                reply = self.desk.apply(events)
                if events:
                    with self.lock:
                        self.applied.extend({"t_ns": reply["t_ns"], "event": e} for e in events)
        except EmergencyStop:
            raise
        except DesktopError as error:
            self.refused += 1
            self.error = str(error)
            return
        self.refused = 0

    def on_speed_up(self):
        if self.pointer is None:
            return False
        x, y = SPEED_UP[0] * self.width, SPEED_UP[1] * self.height
        return abs(self.pointer[0] - x) <= SPEED_BOX and abs(self.pointer[1] - y) <= SPEED_BOX

    def join(self, timeout=PERIOD * 8):
        """Wait for the interval in flight; raise what it raised."""
        future, self.future = self.future, None
        if future is not None:
            future.result(timeout=timeout)

    def take(self):
        with self.lock:
            taken, self.applied = self.applied, []
        return taken

    def close(self):
        try:
            self.join()
        except Exception as error:  # noqa: BLE001 - closing; the game is over either way.
            log.debug("dispatch at close: %s", error)
        self.pool.shutdown(wait=True)


class Referee:
    """When the harness must start the game or set it running again.

    The game starts paused. It is started once the policy has clicked + (after
    RUN_DELAY) or after `setup_seconds`. Once running, a stretch of `stall_seconds`
    without a daily report means it stopped. Days come about every 0.4 s at speed 5.
    """

    def __init__(self, setup_seconds=90.0, stall_seconds=8.0, clock=time.monotonic):
        self.setup_seconds, self.stall_seconds, self.clock = setup_seconds, stall_seconds, clock
        self.began = clock()
        self.running = False
        self.first_click = None
        self.last_day = None
        self.starts = self.restarts = 0

    def clicked_speed_up(self):
        if self.first_click is None:
            self.first_click = self.clock()

    def saw_day(self):
        self.last_day = self.clock()

    def due(self):
        """ "start", "restart" or None."""
        now = self.clock()
        if not self.running:
            clicked = self.first_click is not None and now - self.first_click >= RUN_DELAY
            if clicked or now - self.began >= self.setup_seconds:
                return "start"
            return None
        if self.last_day is not None and now - self.last_day >= self.stall_seconds:
            return "restart"
        return None

    def started(self):
        self.running = True
        self.starts += 1
        self.last_day = self.clock()

    def restarted(self):
        self.restarts += 1
        self.last_day = self.clock()


def run_game(desk, pointer, rules=None, space=True):
    """Unpause (space) and set speed 5 with clicks on +, then put the pointer back.

    Setup-mode input, as record-ai's run_at. The pointer goes back where the policy left
    it, so the next frame shows it where the policy believes it is.
    """
    events = []
    if space:
        events += tap(SPACE)
    press = [{"kind": "button", "button": 0, "down": d} for d in (True, False)]
    events.append({"kind": "move", "x": SPEED_UP[0], "y": SPEED_UP[1]})
    events += press * SPEED_CLICKS
    if pointer is not None:
        events.append({"kind": "move", "x": pointer[0], "y": pointer[1]})
    act(desk, events, pause=0.12)


def is_paused(desk, rules, looks=6):
    """Whether the pause mark shows in any of a few frames: it blinks."""
    for _ in range(looks):
        rgb = on_screen(desk.capture(full=True)).rgb
        if rules.matches("paused", rgb):
            return True
        time.sleep(0.15)
    return False


def play_policy_game(
    desk,
    actor,
    root,
    *,
    rules,
    country,
    station="peer",
    speed=5,
    hz=5,
    cap_minutes=15.0,
    setup_seconds=90.0,
    stall_seconds=8.0,
    codec="x264",
    arena_name=None,
    after_surrender=5.0,
):
    """Record one game in which `actor` plays `country` against the game's AI.

    Starts from the game paused at its first hour, as start_game(observe=False) leaves
    it. Ends `after_surrender` seconds after the arena log names a winner, or at
    `cap_minutes` (a draw). Returns the outcome, the reason it ended early if it did, and
    the manifest.
    """
    clock = time.monotonic
    actor.reset_episode()
    first = on_screen(desk.capture(full=True))
    rec = Recorder(root, first, game_speed=speed, source="policy", hz=hz, codec=codec)
    height, width = first.rgb.shape[:2]
    dispatcher = Dispatcher(desk, width, height)
    referee = Referee(setup_seconds, stall_seconds)
    arena = ArenaLog(desk)
    stamped, timings = [], []
    outcome, reason, ending = "timeout", None, None
    late = away = 0
    start = deadline = next_poll = clock()
    try:
        rec.append(first)
        frame = first
        while clock() - start < cap_minutes * 60:
            dispatcher.join()
            if dispatcher.speed_clicks and not referee.running:
                referee.clicked_speed_up()
            wait = referee.due()
            if wait is not None:
                pointer = None
                cursor = frame.meta.get("cursor")
                if cursor:
                    pointer = (cursor[0] / width, cursor[1] / height)
                desk.release()
                if wait == "start":
                    run_game(desk, pointer)
                    referee.started()
                    log.info("[%s] the game runs, %.0f s after the start", station, clock() - start)
                else:
                    paused = is_paused(desk, rules) if rules is not None else True
                    run_game(desk, pointer, space=paused)
                    referee.restarted()
                    log.info(
                        "[%s] the game stalled; set running again (paused=%s)", station, paused
                    )
                deadline = clock()
            time.sleep(max(0.0, deadline - clock()))
            begin = clock()
            if begin - deadline > PERIOD / 2:
                late += 1
            deadline = begin + 1 / hz
            try:
                captured_frame = desk.capture(full=True)
            except EmergencyStop:
                raise
            except DesktopError as error:
                if "not_foreground" not in str(error):
                    raise
                captured_frame = None
            if (
                captured_frame is None
                or not captured_frame.meta.get("foreground")
                or dispatcher.refused
            ):
                away += 1
                if away > 25 * hz:
                    raise RuntimeError(f"the game was out of reach for 25 s ({dispatcher.error})")
                desk.release()
                focus(desk, tries=2)
                dispatcher.refused = 0
                continue
            away = 0
            frame = on_screen(captured_frame)
            rec.append(frame, scripted_events=dispatcher.take())
            captured = clock()
            action, _sample = actor.act(frame.rgb, frame.meta["t_ns"], cursor=frame.meta["cursor"])
            acted = clock()
            cursor = frame.meta["cursor"]
            dispatcher.start(action, clock(), (float(cursor[0]), float(cursor[1])))
            timings.append(
                {
                    "capture_ms": round((captured - begin) * 1e3, 1),
                    "act_ms": round((acted - captured) * 1e3, 1),
                }
            )
            now = clock()
            if now >= next_poll:
                next_poll = now + 1
                seen = len(arena.lines)
                arena.poll()
                frames = rec.manifest["frames"]
                fresh = arena.lines[seen:]
                stamped.extend({"frame": frames, "line": line} for line in fresh)
                if any(line.startswith("day") for line in fresh):
                    referee.saw_day()
                if arena.winner and ending is None:
                    ending = now + after_surrender
            if ending is not None and now >= ending:
                outcome = arena.winner
                break
    except EmergencyStop:
        reason = "F12"
    except Exception as error:  # noqa: BLE001 - recorded in the manifest.
        reason = f"{type(error).__name__}: {error}"
        log.warning("[%s] game ended early: %s", station, reason)
    finally:
        dispatcher.close()
        try:
            desk.release()
        except DesktopError:
            pass
        root = Path(root)
        (root / "arena-log.txt").write_text("\n".join(arena.lines) + "\n")
        with (root / "arena-log.jsonl").open("w") as out:
            out.writelines(json.dumps(entry) + "\n" for entry in stamped)
        with (root / "timings.jsonl").open("w") as out:
            out.writelines(json.dumps(entry) + "\n" for entry in timings)
        act_ms = [t["act_ms"] for t in timings]
        rec.manifest.update(
            winner=outcome,
            surrendered=arena.surrendered,
            started_as=country,
            declarer=arena.declarer,
            players=arena.players,
            seconds=round(clock() - start),
            late_ticks=late,
            act_ms_p50=float(np.median(act_ms)) if act_ms else None,
            act_ms_p95=float(np.percentile(act_ms, 95)) if act_ms else None,
            arena=arena_name,
            driver="learned policy from pixels; harness starts the game and keeps speed 5",
            labels="scripted_events",
            station=station,
            checkpoint=actor.digest,
            harness={"starts": referee.starts, "restarts": referee.restarts},
        )
        rec.close(complete=reason is None, reason=reason)
    return outcome, reason, rec.manifest


def reserve(name, minutes, *, root=EVAL, wait_minutes=90.0, clock=time.monotonic):
    """Ask the scripted player's agent for the second PC, and wait until it grants it.

    Writes queue/<name>.json; that agent quits HOI4 between its games and answers with
    granted/<name>.json. Raises TimeoutError after `wait_minutes`.
    """
    root = Path(root)
    for folder in ("queue", "granted", "done"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    granted = root / "granted" / f"{name}.json"
    (root / "queue" / f"{name}.json").write_text(
        json.dumps({"minutes": minutes, "requested": time.strftime("%Y-%m-%d %H:%M:%S")})
    )
    until = clock() + wait_minutes * 60
    while not granted.exists():
        if clock() > until:
            raise TimeoutError(f"the second PC was not granted within {wait_minutes} min")
        time.sleep(10)
    log.info("the second PC is granted: %s", granted.read_text())


def hand_back(name, summary, *, root=EVAL):
    """Tell the scripted player's agent the second PC is free again (HOI4 left closed)."""
    done = Path(root) / "done"
    done.mkdir(parents=True, exist_ok=True)
    (done / f"{name}.json").write_text(
        json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), **summary}, indent=2)
    )


def evaluate_policy(
    checkpoint,
    output,
    *,
    games,
    minutes,
    peer,
    mod="arena-12x8-v4",
    rules="artifacts/calibration-1080p/rules.json",
    reservation=None,
    countries=("BLU", "RED"),
    cap_minutes=15.0,
    setup_seconds=90.0,
    memory_window=None,
    model_path=None,
    seed=None,
):
    """Play up to `games` games (or until `minutes` run out) on the second PC and record them.

    With `reservation`, the second PC is first reserved from the scripted player's agent
    (reserve) and handed back when done, with HOI4 closed. Countries alternate. Returns
    the results, as record-ai writes them, so win-rate reads them too.
    """
    from .runner import Actor
    from .scripted import win_rate
    from .vision import ScreenRules

    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)
    screen_rules = ScreenRules(rules)
    rng = random.Random(seed)
    if reservation:
        reserve(reservation, minutes)
    results, summary = [], {}
    station = Station("peer", peer)
    end = time.monotonic() + minutes * 60
    try:
        actor = Actor(checkpoint, model_path, game_speed=5, memory_window=memory_window)
        for index in range(games):
            # A game takes about 3 minutes to launch and up to cap_minutes to play.
            if time.monotonic() + (cap_minutes + 4) * 60 > end:
                break
            country = countries[index % len(countries)]
            name = time.strftime("policy-peer-%Y%m%d-%H%M%S")
            entry = {"game": name, "station": "peer", "started_as": country, "speed": 5}
            entry["declare_drawn"] = rng.choice(("BLU", "RED"))
            entry["checkpoint"] = actor.digest
            try:
                station.quit()
                station.launch(mod)
                time.sleep(25)
                with station.connect() as desk:
                    if not focus(desk):
                        raise RuntimeError("could not bring the game window to the front")
                    start_game(
                        desk, screen_rules, out_root / f"{name}-start-failed.png", country, 5,
                        observe=False, declarer=entry["declare_drawn"],
                    )  # fmt: skip
                    log.info("[peer] %s: the policy plays %s", name, country)
                    outcome, reason, manifest = play_policy_game(
                        desk, actor, out_root / name, rules=screen_rules, country=country,
                        cap_minutes=cap_minutes, setup_seconds=setup_seconds, arena_name=mod,
                    )  # fmt: skip
            except Exception as error:  # noqa: BLE001 - reported, then the next game.
                entry["error"] = f"{type(error).__name__}: {error}"
                log.warning("[peer] %s failed: %s", name, entry["error"])
            else:
                entry.update(
                    winner=outcome,
                    seconds=manifest["seconds"],
                    frames=manifest["frames"],
                    declarer=manifest["declarer"],
                    players=manifest["players"],
                    complete=manifest["complete"],
                    reason=reason,
                    harness=manifest["harness"],
                    act_ms_p50=manifest["act_ms_p50"],
                    late_ticks=manifest["late_ticks"],
                )
                log.info("[peer] %s: winner %s after %s s", name, outcome, manifest["seconds"])
            results.append(entry)
            (out_root / "results-peer.json").write_text(json.dumps(results, indent=2))
    finally:
        try:
            station.quit()
        except Exception as error:  # noqa: BLE001 - the games are saved.
            log.warning("quit failed: %s", error)
        summary = win_rate(results) if results else {}
        if reservation:
            hand_back(reservation, {"games": len(results), "record": summary.get("all")})
    return {"results": results, "win_rate": summary}
