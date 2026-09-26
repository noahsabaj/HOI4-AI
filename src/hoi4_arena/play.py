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
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .actions import GRID, PERIOD, SLOTS, decode
from .ai_games import SPEED_UP, Station, act, focus, on_screen, start_game, tap
from .arena_log import ArenaLog
from .dataset import DETAIL_SIZE, FOVEA_SIZE, VIEW_SIZE
from .desktop import DesktopError, EmergencyStop
from .recording import (
    STREAM_CODECS,
    Recorder,
    StreamRecorder,
    StreamUnavailable,
    stream_codecs,
)
from .telemetry import game_fits

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
# The longest the policy may hold a key or button down before the harness lets it go, in
# seconds. The scripted games it learns from never hold a key past 0.8 s or the left
# button past 0.4 s; the right button, drawing fronts, up to 2.6 s (scripted-v5, measured
# 2026-09-26). The policy sees only its previous decision's action, so a key held longer
# is soon forgotten and never released: bc5 held Right for over three minutes and the
# camera scrolled off the map. Past these limits the key goes up as a player's would.
HOLD_LIMITS = {"key": 1.0, "button": 1.0, "button1": 4.0}


def _held(event):
    """The key or button an event presses or releases, or None."""
    if event["kind"] == "key":
        return ("key", event["vk"])
    if event["kind"] == "button":
        return ("button", event["button"])
    return None


class Holds:
    """What the policy holds down, since which interval, and the releases HOLD_LIMITS call
    for: at the start of the first interval past a key's limit, before the policy's own
    slots. `forced` counts them, by key or button."""

    def __init__(self, limits=HOLD_LIMITS):
        self.limits = limits
        self.since = {}
        self.interval = 0
        self.forced = {}

    def limit(self, held):
        kind, code = held
        return self.limits.get(f"{kind}{code}", self.limits[kind])

    def due(self):
        """The release events due now, forgetting what they release."""
        releases = []
        for held, start in list(self.since.items()):
            if (self.interval - start) * PERIOD >= self.limit(held) - 1e-9:
                del self.since[held]
                kind, code = held
                name = f"{kind}{code}"
                self.forced[name] = self.forced.get(name, 0) + 1
                field = "vk" if kind == "key" else "button"
                releases.append({"kind": kind, field: code, "down": False})
        return releases

    def follow(self, events):
        """Keep track of the interval's own presses and releases, in order."""
        for event in events:
            held = _held(event)
            if held is None:
                continue
            if event["down"]:
                self.since.setdefault(held, self.interval)
            else:
                self.since.pop(held, None)

    def advance(self):
        self.interval += 1

    def clear(self):
        """Everything went up (the harness released the input)."""
        self.since.clear()


def lattice_to_pixels(x, y, width, height):
    return x / (GRID - 1) * (width - 1), y / (GRID - 1) * (height - 1)


class Dispatcher:
    """Applies a decision's eight slots across one interval, on a thread of its own.

    Keeps each applied event with the worker's time of it, as the recorder stores inputs,
    and follows the pointer, to notice a press on the speed control's +. With `timed` (a
    worker of protocol 2), the slots go in one request, each at its offset on the worker's
    own clock, and the reply gives each event's time; before, each slot was a request of
    its own, sent as its time came, and a slot waited on the network.
    """

    def __init__(self, desk, width=1920, height=1080, clock=time.perf_counter, timed=False):
        # perf_counter, not monotonic: on Windows monotonic ticks in 15.6 ms steps, and the
        # slots are 25 ms apart, as training bins the inputs.
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
        # Input is armed once and kept armed by an apply every slot, empty ones too (the
        # worker disarms after 750 ms without one). Anything that releases it, such as
        # the harness's own input, clears this so the next interval arms again. Arming
        # each interval would also clear an F12 the capture had not yet reported.
        self.armed = False
        self.timed = timed
        self.holds = Holds()

    def start(self, action, begin, pointer):
        self.pointer = pointer
        self.future = self.pool.submit(self._run, np.asarray(action), begin)

    def _run(self, action, begin):
        try:
            if not self.armed:
                self.desk.arm()
                self.armed = True
            if self.timed:
                self._timed(action)
                self.refused = 0
                return
            released = self.holds.due()
            if released:
                reply = self.desk.apply(released)
                with self.lock:
                    self.applied.extend(
                        {"t_ns": reply["t_ns"], "event": e, "by": "harness"} for e in released
                    )
            for index, token in enumerate(action):
                time.sleep(max(0.0, begin + index * PERIOD / SLOTS - self.clock()))
                events = decode(token)
                self._follow(token, events)
                self.holds.follow(events)
                reply = self.desk.apply(events)
                if events:
                    with self.lock:
                        self.applied.extend({"t_ns": reply["t_ns"], "event": e} for e in events)
        except EmergencyStop:
            raise
        except DesktopError as error:
            self.refused += 1
            self.error = str(error)
            self.armed = False
            return
        finally:
            self.holds.advance()
        self.refused = 0

    def released(self):
        """The harness let go of every input (desk.release): arm again, nothing held."""
        self.armed = False
        self.holds.clear()

    def _follow(self, token, events):
        """Where the pointer goes, and whether a press lands on the speed control's +."""
        for event in events:
            if event["kind"] == "move":
                self.pointer = lattice_to_pixels(
                    int(token[1]), int(token[2]), self.width, self.height
                )
            elif event["kind"] == "button" and event["down"] and self.on_speed_up():
                self.speed_clicks += 1

    def _timed(self, action):
        # Releases the limits call for go first, at the interval's start.
        released = self.holds.due()
        events, offsets = list(released), [0.0] * len(released)
        for index, token in enumerate(action):
            decoded = decode(token)
            self._follow(token, decoded)
            self.holds.follow(decoded)
            events += decoded
            offsets += [index * PERIOD / SLOTS * 1000] * len(decoded)
        if not events:
            # An empty apply still feeds the worker's watchdog, which disarms after 750 ms.
            self.desk.apply([])
            return
        reply = self.desk.apply(events, at_ms=offsets)
        times = reply.get("times_ns") or [reply["t_ns"]] * len(events)
        with self.lock:
            self.applied.extend(
                {"t_ns": int(t), "event": e, **({"by": "harness"} if i < len(released) else {})}
                for i, (t, e) in enumerate(zip(times, events, strict=True))
            )

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


class Watch:
    """What a game in progress shows someone following it: a small picture of the screen
    every `every` seconds (snaps/, written off the decision thread) and a log line of the
    policy's presses so far."""

    def __init__(self, root, station, every=30.0, clock=time.monotonic):
        self.folder = Path(root) / "snaps"
        self.station, self.every, self.clock = station, every, clock
        self.next = clock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snaps")
        self.presses = {}
        self.events = []

    def count(self, events):
        self.events.extend(events)
        for item in events:
            event = item["event"]
            if event["kind"] == "key" and event["down"]:
                name = chr(event["vk"]) if 0x41 <= event["vk"] <= 0x5A else str(event["vk"])
            elif event["kind"] == "button" and event["down"]:
                name = f"b{event['button']}"
            else:
                continue
            self.presses[name] = self.presses.get(name, 0) + 1

    def look(self, frame, seconds, running):
        if not self.every or self.clock() < self.next:
            return
        self.next = self.clock() + self.every
        rgb = frame.rgb if frame.rgb is not None else frame.views.global_view
        self.pool.submit(self._save, rgb, int(seconds))
        log.info(
            "[%s] %d s, running %s, presses so far %s", self.station, seconds, running,
            json.dumps(self.presses, sort_keys=True),
        )  # fmt: skip

    def _save(self, rgb, seconds):
        from PIL import Image

        self.folder.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(rgb).resize((960, 540), Image.BILINEAR)
        image.save(self.folder / f"{seconds:04d}.jpg", quality=80)

    def close(self):
        self.pool.shutdown(wait=True)


def sampling_of(actor):
    """How `actor` draws its actions (runner.Actor.sampling), for a manifest: its
    temperatures, or those of plain sampling for an actor that names none."""
    return {"temperature": 1.0, "pointer_temperature": 1.0, **getattr(actor, "sampling", {})}


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
    codec="nvenc-hevc",
    arena_name=None,
    after_surrender=5.0,
    snap_every=30.0,
    coach=None,
    log_from=None,
):
    """Record one game in which `actor` plays `country` against the game's AI.

    Starts from the game paused at its first hour, as start_game(observe=False) leaves
    it. Ends `after_surrender` seconds after the arena log names a winner, or at
    `cap_minutes` (a draw). Returns the outcome, the reason it ended early if it did, and
    the manifest.

    With `coach` (practice.Coach), the scripted player watches the setup and takes a step
    over when the policy falls behind; its inputs are recorded, tagged "by": "coach", and
    the frames it held are the manifest's `coached`. `log_from` is where this game's lines
    begin in the game log, for a game loaded from inside the last one (ai_games.log_end).
    """
    clock = time.perf_counter
    actor.reset_episode()
    first = on_screen(desk.capture(full=True))
    # With a worker of protocol 2 the game is recorded where it runs, on the worker's own
    # 5 Hz clock, and each frame of that stream carries the policy's views, so one clock
    # paces both the video and the policy (open_stream); its slots go in one timed request.
    rec = open_stream(desk, root, first, speed=speed, hz=hz, codec=codec)
    streamed = bool(getattr(rec, "streamed", False))
    height, width = first.rgb.shape[:2]
    timed = _protocol(desk) >= 2
    dispatcher = Dispatcher(desk, width, height, timed=timed)
    referee = Referee(setup_seconds, stall_seconds)
    arena = ArenaLog(desk)
    if log_from is not None:
        arena.offset = log_from
    if coach is not None:
        coach.attach(rec, root)
    watch = Watch(root, station, snap_every)
    stamped, timings = [], []
    outcome, reason, ending = "timeout", None, None
    late = away = 0
    last_index = None
    start = deadline = next_poll = clock()
    try:
        if not streamed:
            rec.append(first)
        frame = first
        while clock() - start < cap_minutes * 60:
            # The interval in flight keeps applying while the next frame is taken and
            # read, as in ArenaEnv.step: the loop holds 5 Hz, and an action goes out
            # about 150 ms after the frame it answers.
            if not streamed:
                time.sleep(max(0.0, deadline - clock()))
            begin = clock()
            if not streamed and begin - deadline > 0.005:
                late += 1
            deadline = begin + 1 / hz
            try:
                if streamed:
                    # The stream's next frame, or its newest if the loop fell behind (then
                    # the frames between are skipped, and counted).
                    captured_frame = rec.next_frame(timeout=10)
                    index = captured_frame.meta.get("index")
                    if index is not None and last_index is not None and index > last_index + 1:
                        late += index - last_index - 1
                    last_index = index if index is not None else last_index
                    if captured_frame.views is None and captured_frame.meta.get("foreground"):
                        captured_frame = desk.capture(views=VIEW_SIZE)
                else:
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
                dispatcher.join()
                desk.release()
                dispatcher.released()
                actor.let_go()
                focus(desk, tries=2)
                dispatcher.refused = 0
                continue
            away = 0
            frame = on_screen(captured_frame)
            captured = clock()
            action, _sample = actor.act(
                frame.rgb, frame.meta["t_ns"], precomputed=frame.views, cursor=frame.meta["cursor"]
            )
            acted = clock()
            dispatcher.join()
            # Every input of the interval that just ended, now that it has: recorded with
            # this frame, which was taken while they were being applied.
            applied = dispatcher.take()
            for item in applied:
                if item.get("by") == "harness":
                    actor.let_go(item["event"])
            watch.count(applied)
            if dispatcher.speed_clicks and not referee.running:
                referee.clicked_speed_up()
            step = coach.look(desk, clock() - start, referee.running) if coach else None
            if step is not None:
                # The coach's own input, while no interval is in flight, as the harness's.
                desk.release()
                dispatcher.released()
                actor.let_go()
                log.info("[%s] the coach takes over: %s", station, step)
                taken = coach.take_over(desk, step, clock() - start)
                record(rec, frame, applied + taken, streamed)
                deadline = clock()
                continue
            wait = referee.due()
            if wait is not None:
                # The harness's own input, while no interval is in flight; the action
                # just chosen answered a screen this changes, so it is dropped.
                pointer = dispatcher.pointer or frame.meta["cursor"]
                pointer = (pointer[0] / width, pointer[1] / height)
                desk.release()
                dispatcher.released()
                actor.let_go()
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
                record(rec, frame, applied, streamed)
                deadline = clock()
                continue
            cursor = frame.meta["cursor"]
            dispatcher.start(action, clock(), (float(cursor[0]), float(cursor[1])))
            sent = clock()
            # The frame goes to the video after the action is on its way.
            record(rec, frame, applied, streamed)
            watch.look(frame, clock() - start, referee.running)
            timings.append(
                {
                    "capture_ms": round((captured - begin) * 1e3, 1),
                    "act_ms": round((acted - captured) * 1e3, 1),
                    "send_ms": round((sent - acted) * 1e3, 1),
                    "record_ms": round((clock() - sent) * 1e3, 1),
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
                # The game as it stands, for the live view (live.py): the arena's latest
                # daily reports, and the steps of the scripted player's setup made so far.
                from .ai_games import publish

                publish(Path(root) / "live-state.json", {
                    "station": station, "arena": arena_name, "started_as": country,
                    "started_unix": (rec.manifest.get("recorder") or {}).get("started_unix"),
                    "updated_unix": time.time(), "frames": frames, "hz": hz,
                    "seconds": round(now - start), "plan": {"variant": "learned"},
                    "declarer": arena.declarer, "days": arena.days, "weeks": arena.weeks,
                    "milestones": milestones(watch.events), "presses": watch.presses,
                    "winner": arena.winner, "surrendered": arena.surrendered,
                })  # fmt: skip
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
        watch.close()
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
            milestones=milestones(watch.events),
            live={"streamed": streamed, "timed_applies": timed},
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
            # The temperatures it played at, so games at each can be told apart.
            **sampling_of(actor),
            harness={"starts": referee.starts, "restarts": referee.restarts},
            presses=watch.presses,
            forced_releases=dispatcher.holds.forced,
        )
        if coach is not None:
            rec.manifest.update(coached=coach.coached, setup=coach.score())
        rec.close(complete=reason is None, reason=reason)
    return outcome, reason, rec.manifest


def open_stream(desk, root, first, *, speed, hz, codec):
    """A recording clocked and encoded by the worker whose frames carry the policy's views
    (StreamRecorder), when the worker can; else the classic recording, here, in x264."""
    if codec in STREAM_CODECS and _protocol(desk) >= 2:
        for tried in stream_codecs(codec):
            try:
                return StreamRecorder(
                    root, desk, game_speed=speed, source="policy", hz=hz, codec=tried,
                    views=VIEW_SIZE, detail=DETAIL_SIZE, fovea=FOVEA_SIZE,
                )  # fmt: skip
            except StreamUnavailable as error:
                log.warning("no %s recording stream (%s)", tried, error)
        log.warning("no recording stream; recording here in x264")
    codec = "x264" if codec in STREAM_CODECS else codec
    return Recorder(root, first, game_speed=speed, source="policy", hz=hz, codec=codec)


def _protocol(desk):
    try:
        return int(desk.protocol())
    except (AttributeError, DesktopError, TypeError, ValueError):
        return 1


def record(rec, frame, applied, streamed):
    """The policy's inputs into the recording: beside the worker's own frames when it
    records them (a stream), else with the frame taken here."""
    if streamed:
        rec.append(scripted_events=applied)
    else:
        rec.append(frame, scripted_events=applied)


# The scripted player's setup and orders at 1080p on the second PC, whose interface never
# moved in 31 games: where each press lands, with how far off it may be (x, y) in pixels.
MILESTONES = {
    "alert": ((825, 57), (25, 20)),  # Unassigned divisions, shift+click: select them.
    "plus": ((988, 1013), (25, 25)),  # Create the army.
    "portrait": ((30, 140), (30, 30)),  # The army's commander slot.
    "commander": ((950, 352), (180, 25)),  # The first commander in the list.
    "law_slot": ((60, 593), (25, 25)),  # The conscription law.
    "confirm": ((1054, 677), (70, 18)),  # OK on "Replace idea?" or "Delete all orders?".
    "trash": ((1265, 885), (18, 18)),  # Delete orders (a right-click).
    "army_card": ((947, 1010), (35, 30)),
    "arrow": ((947, 957), (40, 12)),  # Execute the plan.
}


def milestones(events, width=1920, height=1080):
    """Which of the scripted player's steps a game's inputs made: presses near each place in
    MILESTONES, a Z followed by a left press within 3 s (a front line), an X followed by a
    right press within 3 s (an offensive), Q presses. Counts, from `scripted_events` rows."""
    counts = {name: 0 for name in MILESTONES}
    counts.update(front=0, offensive=0, q=0)
    pointer, last_key = None, {}
    for item in sorted(events, key=lambda e: e["t_ns"]):
        event, t = item["event"], item["t_ns"] / 1e9
        if event["kind"] == "move":
            pointer = (event["x"] * (width - 1), event["y"] * (height - 1))
        elif event["kind"] == "key" and event["down"]:
            last_key[event["vk"]] = t
            counts["q"] += event["vk"] == 0x51
        elif event["kind"] == "button" and event["down"] and pointer is not None:
            if event["button"] == 0 and t - last_key.get(0x5A, -99) < 3:
                counts["front"] += 1
            if event["button"] == 1 and t - last_key.get(0x58, -99) < 3:
                counts["offensive"] += 1
            for name, ((x, y), (dx, dy)) in MILESTONES.items():
                if abs(pointer[0] - x) <= dx and abs(pointer[1] - y) <= dy:
                    counts[name] += 1
    return counts


# What a game needs on the second PC: HOI4 takes 5.3-5.9 GB of commit and the stream's
# encoder about 1 GB; allocations past the commit limit fail and can crash the game.
GAME_COMMIT_MB = 6.5 * 1024
COMMIT_SHARE = float(os.environ.get("HOI4_COMMIT_SHARE", "0.95"))


def room_for_a_game(station, tries=10, wait=30.0):
    """Whether the second PC, with HOI4 closed, has room for a game (game_fits: its commit
    plus GAME_COMMIT_MB under COMMIT_SHARE of the limit it can grow to, and RAM for it).
    Waits and looks again a few times (the recorder that lent the PC may still be closing
    its game). The reading, logged."""
    pagefile = station.pagefile()
    for _ in range(tries):
        try:
            with station.connect(attach=False) as desk:
                reply = desk.telemetry(timeout=15)
        except Exception as error:  # noqa: BLE001 - no reading is no room.
            log.warning("[peer] telemetry failed: %s", error)
            reply = {}
        memory = reply.get("memory") or {}
        commit, limit = memory.get("commit_mb"), memory.get("commit_limit_mb")
        why = game_fits(reply, pagefile, GAME_COMMIT_MB, COMMIT_SHARE) if commit else "no reading"
        if not why:
            log.info("[peer] room for a game: commit %s of %s MB", commit, limit)
            return True
        log.warning("[peer] no room for a game yet: %s (commit %s of %s MB)", why, commit, limit)
        time.sleep(wait)
    return False


def reserve(name, minutes, *, root=EVAL, wait_minutes=90.0, clock=time.monotonic):
    """Ask the scripted player's agent for the second PC, and wait until it grants it.

    Writes queue/<name>.json; that agent quits HOI4 between its games and answers with
    granted/<name>.json. A request already queued or granted under that name (made ahead,
    so the grant's wait overlaps other work) is not made again. Raises TimeoutError after
    `wait_minutes`.
    """
    root = Path(root)
    for folder in ("queue", "granted", "done"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    granted = root / "granted" / f"{name}.json"
    queued = root / "queue" / f"{name}.json"
    taken = root / "queue" / f"{name}.taken"
    if not (granted.exists() or queued.exists() or taken.exists()):
        queued.write_text(
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
    point=False,
    temperature=1.0,
    pointer_temperature=None,
    model_path=None,
    seed=None,
    saves=None,
    held_previous=False,
):
    """Play up to `games` games (or until `minutes` run out) on the second PC and record them.

    With `reservation`, the second PC is first reserved from the scripted player's agent
    (reserve) and handed back when done, with HOI4 closed. Countries alternate. Returns
    the results, as record-ai writes them, so win-rate reads them too. `saves`, {country:
    save name}, launches each game straight into a save made paused at the start of a new
    game as that country, as record-ai does, skipping the menus. `temperature`,
    `pointer_temperature` and `point` are how it samples (runner.resolve_temperatures),
    and each result names the two temperatures.
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
        actor = Actor(
            checkpoint,
            model_path,
            game_speed=5,
            memory_window=memory_window,
            point=point,
            temperature=temperature,
            pointer_temperature=pointer_temperature,
        )
        actor.lean = True  # Only the action is needed: no training sample, no clip.
        # Shown what it holds even if trained without (a checkpoint trained with it is).
        actor.held_previous = actor.held_previous or held_previous
        for index in range(games):
            # A game takes about 3 minutes to launch and up to cap_minutes to play.
            if time.monotonic() + (cap_minutes + 4) * 60 > end:
                break
            country = countries[index % len(countries)]
            name = time.strftime("policy-peer-%Y%m%d-%H%M%S")
            entry = {"game": name, "station": "peer", "started_as": country, "speed": 5}
            entry.update(arena=mod, plan={"variant": "learned"})
            entry["declare_drawn"] = rng.choice(("BLU", "RED"))
            entry["checkpoint"] = actor.digest
            entry.update(sampling_of(actor))
            save = (saves or {}).get(country)
            entry["start_save"] = save
            try:
                station.quit()
                if not room_for_a_game(station):
                    raise RuntimeError("the second PC has no commit room for a game")
                station.launch(mod, save=save)
                if not save:
                    time.sleep(25)
                with station.connect() as desk:
                    if not focus(desk):
                        raise RuntimeError("could not bring the game window to the front")
                    start_game(
                        desk, screen_rules, out_root / f"{name}-start-failed.png", country, 5,
                        observe=False, declarer=entry["declare_drawn"], saved=bool(save),
                    )  # fmt: skip
                    log.info("[peer] %s: the policy plays %s", name, country)
                    try:
                        # The game about to be played, for the live view (live.py).
                        (out_root / "live.json").write_text(json.dumps(entry))
                    except OSError:
                        pass
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
                    milestones=manifest.get("milestones"),
                    act_ms_p50=manifest["act_ms_p50"],
                    late_ticks=manifest["late_ticks"],
                )
                log.info("[peer] %s: winner %s after %s s", name, outcome, manifest["seconds"])
            results.append(entry)
            (out_root / "results-peer.json").write_text(json.dumps(results, indent=2))
            if entry.get("reason") and "no commit room" not in str(entry.get("error", "")):
                # A game that ended early (the game exited or crashed, the worker was lost)
                # ends the test: no second game on a PC that just failed one.
                log.warning("[peer] %s ended early (%s); stopping the test", name, entry["reason"])
                break
    finally:
        try:
            station.quit()
        except Exception as error:  # noqa: BLE001 - the games are saved.
            log.warning("quit failed: %s", error)
        summary = win_rate(results) if results else {}
        if reservation:
            hand_back(reservation, {"games": len(results), "record": summary.get("all")})
    return {"results": results, "win_rate": summary}
