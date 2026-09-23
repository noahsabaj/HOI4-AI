"""Record AI-vs-AI games on an arena map, one after another, until a time budget runs out.

Each game launches HOI4 with the arena, starts as Blue or Red in turn, hands both countries to
the AI with the `observe` console command, sets speed 4 or 5, and records native frames while the
camera watches the front, zooms in and out and clears popups the way a player would. The arena mod reports itself in the game's
log, so a game ends when the log names a surrender and its winner, or at the cap, and a
game whose weekly report stops has a stuck clock.

With a peer, the second PC records its own games at the same time, driven over its worker
connection; its frames are recorded here. The second PC's worker must be running and the
arena deployed to it (Deploy-Peer.ps1 -Mod), which this does itself.

The recordings carry the camera's own inputs as labels, but none of the AI's orders. They are
for the encoder, for predicting who wins, for camera control and clearing popups, and for
measuring how often a match ends inside the time limit.

It takes over the screen of each PC it uses. Anything else that takes focus stops input to
the game until the recorder brings it back.
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .arena_log import ArenaLog
from .desktop import Desktop, DesktopError, local_control_args
from .recording import Recorder
from .remote import RemoteDesktop
from .vision import ScreenRules, country_pixels, find_template

log = logging.getLogger(__name__)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
# The game runs in a 1920x1080 window (Test-ArenaLoad -Window), and these are fractions of
# it, measured on 2026-09-22.
WINDOW = "1920x1080"
# A popup's Ok button, matched anywhere on screen: TM_SQDIFF_NORMED measured below 0.001
# on the research and peace popups and above 0.07 everywhere else in two recorded games.
OK_MATCH = 0.05
# At speed 4 a game week is about 17 s, so a minute and a half without the mod's weekly
# report means the clock has stopped.
WEEK_SILENCE = 90
SINGLE_PLAYER, NEW_GAME = (0.5, 290 / 1080), (0.5, 420 / 1080)
SELECT_COUNTRY, START = (1043 / 1920, 875 / 1080), (1777 / 1920, 1038 / 1080)
SPEED_UP = (1789 / 1920, 20 / 1080)
GRAVE, ENTER = 0xC0, 0x0D
# The top bar and the bottom panels are chrome, not map.
MAP_TOP, MAP_BOTTOM = 80, 120
# Close enough to the middle, as a fraction of the screen.
CENTRED = 0.03
# Camera zoom in mouse-wheel notches in from fully out, measured at 1080p (see camera).
# Unit counters show from 9; past 22 the map is terrain; 26 is the closest.
VIEW_NEAR, VIEW_FAR, ZOOM_TERRAIN, ZOOM_MAX = 9, 20, 22, 26


def say(station, *parts):
    log.info("[%s] %s", station, " ".join(str(p) for p in parts))


def pwsh(*args, timeout=300):
    command = ["pwsh", "-NoProfile", *args]
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


class Popups:
    """Finds popups' Ok buttons in the recorded frames and hands them to the camera.

    A popup left open covered the map in 45% of the frames of two recorded games. A
    player clears them, and an agent must learn to, so the camera clicks Ok the way a
    player would: after a short, varying pause to read it.
    """

    def __init__(self, templates, rng=None, clock=time.monotonic):
        self.templates = templates
        self.rng = rng or random.Random()
        self.clock = clock
        self.lock = threading.Lock()
        self.pending = None

    def look(self, rgb):
        with self.lock:
            if self.pending is not None:
                return
        for template in self.templates:
            found = find_template(rgb, template, OK_MATCH)
            if found is not None:
                with self.lock:
                    self.pending = (found, self.clock() + self.rng.uniform(1, 4))
                return

    def due(self):
        """The Ok button to click now, if one has waited its reading time."""
        with self.lock:
            if self.pending and self.clock() >= self.pending[1]:
                at, self.pending = self.pending[0], None
                return at
        return None


class Station:
    """This PC, or the second PC, driven through its worker.

    Launching and closing the game are worker operations (Game-Control.ps1), on either
    PC, over a connection that does not attach to a game, since there may be none yet.
    The second PC's bridge takes one connection at a time, so each operation's
    connection is closed before the recording one opens. Arena mods are named by folder:
    they live in artifacts/mods here and in the deployed mods folder there.
    """

    def __init__(self, name, peer=None):
        self.name, self.peer = name, peer

    def connect(self, attach=True):
        if self.peer:
            return RemoteDesktop(self.peer, attach=attach)
        return Desktop(worker_args=local_control_args(), attach=attach)

    def quit(self):
        with self.connect(attach=False) as desk:
            desk.quit()
        # Let the previous launch's watcher put the player's display settings back first.
        time.sleep(5)

    def launch(self, mod):
        name = Path(mod).name
        with self.connect(attach=False) as desk:
            out = desk.launch(name, window=WINDOW)
            if "Timed out waiting for the game log" in out:
                # A game that never gets far enough to log has hung at startup; a stuck
                # Discord overlay did that once. Restart Discord and try once more.
                say(self.name, "launch hung; restarting Discord and retrying")
                desk.restart_discord()
                desk.quit()
                time.sleep(5)
                out = desk.launch(name, window=WINDOW)
        if "Arena load test PID" not in out or "Timed out" in out:
            raise RuntimeError(f"{self.name} did not launch: {out}")
        say(self.name, "launch:", out.replace("\n", " | "))


class Logged:
    """A desktop whose applied inputs are kept, with the worker's time of each.

    The camera's pans, zooms and popup clicks are real inputs to the game, even though a
    script chose them rather than a player. Kept beside the frames as `scripted_events`,
    they make an AI game a labelled recording of camera control and popup clearing, and
    session_labels reads them the way it reads a player's events.
    """

    def __init__(self, desk):
        self.desk = desk
        self.lock = threading.Lock()
        self.events = []

    def __getattr__(self, name):
        return getattr(self.desk, name)

    def apply(self, events):
        reply = self.desk.apply(events)
        with self.lock:
            self.events.extend({"t_ns": reply["t_ns"], "event": e} for e in events)
        return reply

    def take(self):
        """The inputs applied since the last call, oldest first."""
        with self.lock:
            taken, self.events = self.events, []
        return taken


def focus(desk, tries=5):
    for _ in range(tries):
        try:
            if desk.focus():
                return True
        except DesktopError:
            pass  # Refused while the camera has input armed; it disarms within a second.
        time.sleep(1)
    return False


def act(desk, events, pause=0.15):
    focus(desk)
    desk.arm(setup=True)
    try:
        for event in events:
            desk.apply([event])
            time.sleep(pause)
    finally:
        desk.release()


def click(desk, x, y):
    press = [{"kind": "button", "button": 0, "down": d} for d in (True, False)]
    act(desk, [{"kind": "move", "x": x, "y": y}, *press])


def tap(vk):
    return [{"kind": "key", "vk": vk, "down": d} for d in (True, False)]


def console(desk, command):
    """Type a console command through the worker, which allows the console key in setup."""
    act(desk, tap(GRAVE))
    time.sleep(0.6)
    keys = [e for c in command.upper() for e in tap(0x20 if c == " " else ord(c))]
    act(desk, keys + tap(ENTER), pause=0.05)
    time.sleep(0.4)
    act(desk, tap(GRAVE))


def on_screen(frame):
    """The frame, unless the worker had to fall back from desktop duplication.

    A monitor switched off disconnects on DisplayPort: Windows shrinks the desktop to
    1024x768, the game window fits on no screen, and the fallback frames are wrong.
    """
    if frame.meta.get("backend") == "gdi_bgra":
        raise RuntimeError("the game window is on no screen; is the monitor switched off?")
    return frame


def screen(desk, tries=5):
    """A full frame. The first capture on a new connection is sometimes all black."""
    for _ in range(tries):
        rgb = on_screen(desk.capture(full=True)).rgb
        if rgb.max() > 0:
            break
        time.sleep(0.5)
    return rgb


def hold(desk, vk, seconds):
    desk.arm(setup=True)
    desk.apply([{"kind": "key", "vk": vk, "down": True}])
    try:
        time.sleep(seconds)
    finally:
        desk.apply([{"kind": "key", "vk": vk, "down": False}])


def arena_offset(rgb):
    """Where the middle of the arena is, as (down, right) fractions from the screen centre.

    None when no country's land is on screen, such as over a menu.
    """
    top, bottom = MAP_TOP, rgb.shape[0] - MAP_BOTTOM
    blue, red = country_pixels(rgb[top:bottom])
    if blue is None or not (blue | red).any():
        return None
    ys, xs = np.nonzero(blue | red)
    centre = ((ys.min() + ys.max()) / 2 + top, (xs.min() + xs.max()) / 2)
    return centre[0] / rgb.shape[0] - 0.5, centre[1] / rgb.shape[1] - 0.5


def recentre(desk, tries=10):
    """Zoom fully out and pan until the arena sits in the middle of the screen.

    Steered by what is on screen, not by a fixed recipe, so it works for any arena size
    and undoes whatever drift the camera built up. True if the arena ended up centred.
    """
    act(
        desk,
        [{"kind": "move", "x": 0.5, "y": 0.5}]
        + [{"kind": "wheel", "delta": -120}] * (ZOOM_MAX + 4),
        0.03,
    )
    keys = {(0, 1): 0x27, (0, -1): 0x25, (1, 1): 0x28, (1, -1): 0x26}
    for _ in range(tries):
        time.sleep(0.5)
        offsets = arena_offset(screen(desk))
        if offsets is None:
            return False
        if max(abs(o) for o in offsets) < CENTRED:
            return True
        # Pan along whichever axis is further out: right/left for x, down/up for y.
        axis = 0 if abs(offsets[1]) >= abs(offsets[0]) else 1
        offset = offsets[1] if axis == 0 else offsets[0]
        focus(desk, tries=1)
        hold(desk, keys[(axis, 1 if offset > 0 else -1)], min(0.4, max(0.03, abs(offset))))
        desk.release()
    return False


def pick_country(desk, country):
    """On the country picker, click the middle of `country`'s land. False if none shows."""
    rgb = screen(desk)
    top, bottom = MAP_TOP, rgb.shape[0] - MAP_BOTTOM
    masks = dict(zip(("BLU", "RED"), country_pixels(rgb[top:bottom]), strict=True))
    if masks[country] is None or not masks[country].any():
        return False
    ys, xs = np.nonzero(masks[country])
    # The land pixel nearest the median, so the click is on the land whatever its shape.
    k = np.argmin((ys - np.median(ys)) ** 2 + (xs - np.median(xs)) ** 2)
    click(desk, xs[k] / rgb.shape[1], (ys[k] + top) / rgb.shape[0])
    return True


def start_game(desk, rules, failure_shot, country="BLU", speed=4):
    """From the main menu to an AI-vs-AI game running at `speed` (4 or 5), as `country`.

    Which country the game starts as is varied because only one side ever won while the
    recorder always started as Blue; the arena logs the country each human started as.
    """
    click(desk, *SINGLE_PLAYER)
    time.sleep(8)
    click(desk, *NEW_GAME)
    time.sleep(40)
    click(desk, *SELECT_COUNTRY)  # Blue is preselected.
    time.sleep(40)
    if country != "BLU":
        # Clicking a country's land on the picker's map selects it.
        if not pick_country(desk, country):
            Image.fromarray(screen(desk)).resize((960, 540)).save(failure_shot)
            raise RuntimeError(f"no {country} land on the country picker")
        time.sleep(3)
    click(desk, *START)
    time.sleep(20)
    # A new game starts paused. The pause mark is the same on both PCs; the alert row that
    # "healthy" reads is not, as each PC shows different alerts. The mark blinks, so one
    # frame can catch it faded: look for it over a few seconds.
    for _ in range(40):
        rgb = screen(desk)
        if rules.matches("paused", rgb):
            break
        time.sleep(0.25)
    else:
        Image.fromarray(rgb).resize((960, 540)).save(failure_shot)
        raise RuntimeError("game did not reach the map")
    console(desk, "observe")
    recentre(desk)
    for _ in range(3):
        click(desk, *SPEED_UP)  # The on-screen + button: speed 1 to 4.
    act(desk, [{"kind": "move", "x": 0.5, "y": 0.75}])
    act(desk, tap(0x20))  # Unpause
    time.sleep(2)
    rgb = screen(desk)
    if not rules.matches("speed", rgb) or rules.matches("paused", rgb):
        Image.fromarray(rgb).resize((960, 540)).save(failure_shot)
        raise RuntimeError("game is not running at speed 4")
    if speed == 5:
        # Checked at 4, where the rule was calibrated, then one more.
        click(desk, *SPEED_UP)


def front_points(rgb):
    """Where Blue's land meets Red's, as (x, y) screen fractions: the front line.

    Occupied land takes its occupier's colour, so this follows the fighting, not the
    starting border. Empty when the two do not touch on screen.
    """
    import cv2

    top, bottom = MAP_TOP, rgb.shape[0] - MAP_BOTTOM
    blue, red = country_pixels(rgb[top:bottom])
    if blue is None or not blue.any() or not red.any():
        return []
    near_blue = cv2.dilate(blue.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ys, xs = np.nonzero(near_blue & red)
    return [(x / rgb.shape[1], (y + top) / rgb.shape[0]) for y, x in zip(ys, xs, strict=True)]


def land_points(rgb):
    """Any land on screen, as (x, y) screen fractions."""
    top, bottom = MAP_TOP, rgb.shape[0] - MAP_BOTTOM
    blue, red = country_pixels(rgb[top:bottom])
    if blue is None:
        return []
    ys, xs = np.nonzero(blue | red)
    return [(x / rgb.shape[1], (y + top) / rgb.shape[0]) for y, x in zip(ys, xs, strict=True)]


def camera(desk, stop, station, popups, overview_every=(20, 60), rng=None):
    """Watch the war like a player: close in on the front, look around, step back.

    Measured at 1080p on 2026-09-23, in wheel notches in from fully out: the whole arena
    fills the middle half of the screen at 0, unit counters appear from 9 (the game hides
    them beyond a camera distance of 900), about a third of the arena shows at 18, the map
    turns to terrain past about 22, and 26 is as close as it goes. So the camera spends
    its time between VIEW_NEAR and VIEW_FAR, where the counters are, zooming in and out
    and panning, and every 20 to 60 seconds zooms fully out for a few seconds to see the
    whole map, recentres, and picks somewhere new to look: mostly a point on the front,
    where Blue's land meets Red's, sometimes anywhere on the land. Zooming is toward the
    pointer, as in the game, so it closes in on what it points at. Fully zoomed out the
    counters vanish, which is also a view players use, but only in passing. It shares
    the recorder's connection: the second PC's bridge accepts only one.
    """
    rng = rng or random.Random()
    arrows = [0x25, 0x26, 0x27, 0x28]
    zoom = 0
    next_overview = time.monotonic()

    def do(events, pause=0.08):
        # The worker disarms after 750 ms without input, so arm for each burst.
        desk.arm(setup=True)
        for event in events:
            desk.apply([event])
            time.sleep(pause)

    def clear_popup():
        at = popups.due()
        if at:
            press = [{"kind": "button", "button": 0, "down": d} for d in (True, False)]
            do([{"kind": "move", "x": at[0], "y": at[1]}, *press])

    def wheel(notches, at):
        nonlocal zoom
        step = 120 if notches > 0 else -120
        notches = max(-zoom, min(ZOOM_MAX - zoom, notches))
        do(
            [{"kind": "move", "x": at[0], "y": at[1]}]
            + [{"kind": "wheel", "delta": step}] * abs(notches),
            rng.uniform(0.04, 0.12),
        )
        zoom += notches

    def linger(low=0.6, high=1.8, times=(1, 3)):
        # Point around while looking, like a player reading the map.
        for _ in range(rng.randint(*times)):
            if stop.wait(rng.uniform(low, high)):
                return
            clear_popup()
            do([{"kind": "move", "x": rng.uniform(0.1, 0.9), "y": rng.uniform(0.15, 0.85)}])

    def interest(rgb):
        """Somewhere worth looking on screen: mostly the front, else any land."""
        points = front_points(rgb) if rng.random() < 0.75 else []
        points = points or land_points(rgb)
        return rng.choice(points) if points else None

    def pan():
        # Towards something worth looking at, most of the time; a player does not scroll
        # out over the sea for long. Otherwise anywhere.
        target = interest(screen(desk)) if rng.random() < 0.7 else None
        if target is None:
            hold(desk, rng.choice(arrows), rng.uniform(0.1, 0.4))
            return
        dx, dy = target[0] - 0.5, target[1] - 0.5
        if abs(dx) >= abs(dy):
            key, offset = (0x27 if dx > 0 else 0x25), dx
        else:
            key, offset = (0x28 if dy > 0 else 0x26), dy
        hold(desk, key, min(0.4, max(0.08, abs(offset) * 0.8)))

    def overview():
        nonlocal zoom
        recentre(desk)
        zoom = 0
        linger(1.0, 2.5, (1, 2))
        wheel(rng.randint(VIEW_NEAR + 1, VIEW_FAR), interest(screen(desk)) or (0.5, 0.5))

    try:
        while not stop.wait(rng.uniform(0.8, 2.5)):
            try:
                clear_popup()
                if time.monotonic() >= next_overview:
                    overview()
                    next_overview = time.monotonic() + rng.uniform(*overview_every)
                    continue
                roll = rng.random()
                here = (rng.uniform(0.25, 0.75), rng.uniform(0.3, 0.7))
                if 0.35 <= roll < 0.8 and rng.random() < 0.5:
                    here = interest(screen(desk)) or here
                if roll < 0.35:
                    pan()
                elif roll < 0.55:
                    wheel(rng.randint(1, 4), here)
                elif roll < 0.75:
                    wheel(-rng.randint(1, 4), here)
                elif roll < 0.8:
                    wheel(ZOOM_TERRAIN - zoom + rng.randint(0, 2), here)  # A close look.
                linger()
                if zoom < VIEW_NEAR or zoom > VIEW_FAR + 2:
                    # Drifted out of the counters' range: come back into it.
                    wheel(rng.randint(VIEW_NEAR + 1, VIEW_FAR) - zoom, here)
            except DesktopError as error:
                say(station, "camera:", error)
                try:
                    desk.release()
                except DesktopError:
                    pass
                focus(desk, tries=1)
    finally:
        try:
            desk.release()
        except DesktopError:
            pass


def play(desk, root, popups, settings, station, country="BLU", speed=4):
    stop = threading.Event()
    inputs = Logged(desk)
    mover = threading.Thread(target=camera, args=(inputs, stop, station, popups), daemon=True)
    outcome, reason = "timeout", None
    first = desk.capture()
    hz = settings["hz"]
    rec = Recorder(root, first, game_speed=speed, source="ai", hz=hz, codec=settings["codec"])
    arena = ArenaLog(desk, silence=WEEK_SILENCE)
    start = deadline = next_poll = time.monotonic()
    late, ending = 0, None
    try:
        rec.append(first)
        mover.start()
        while time.monotonic() - start < settings["cap_minutes"] * 60:
            deadline += 1 / hz
            time.sleep(max(0, deadline - time.monotonic()))
            frame = on_screen(desk.capture())
            if not frame.meta.get("foreground"):
                focus(desk, tries=1)
                continue
            rec.append(frame, scripted_events=inputs.take())
            now = time.monotonic()
            if now - deadline > 1:
                late += 1
                deadline = now
            if rec.manifest["frames"] % int(hz) == 0:
                popups.look(frame.rgb)  # About once a second; a search costs ~70 ms.
            if now >= next_poll:
                next_poll = now + 2
                arena.poll()
                if arena.winner and ending is None:
                    # Keep a few seconds of the surrender on screen, then stop.
                    ending = now + 5
                    Image.fromarray(frame.rgb).save(Path(root) / "capitulation.png")
            if ending is not None and now >= ending:
                outcome = arena.winner
                break
            if rec.manifest["frames"] % (60 * int(hz)) == 0:
                say(station, f"{rec.manifest['frames'] // int(hz) // 60} min recorded")
    except Exception as error:  # noqa: BLE001 - recorded in the manifest.
        reason = f"{type(error).__name__}: {error}"
    finally:
        stop.set()
        mover.join(timeout=10)
        (Path(root) / "arena-log.txt").write_text("\n".join(arena.lines) + "\n")
        rec.manifest.update(
            winner=outcome,
            surrendered=arena.surrendered,
            # The country the recorder picked, and what the arena logged: who declared the
            # war and the country the human started as (None on arenas before v2).
            started_as=country,
            declarer=arena.declarer,
            players=arena.players,
            seconds=round(time.monotonic() - start),
            late_ticks=late,
            arena=Path(settings["mod"]).name,
            driver="observe + scripted camera + popup clicks",
            # frames.jsonl carries the camera's inputs as scripted_events.
            labels="scripted_events",
            station=station,
        )
        rec.close(complete=reason is None, reason=reason)
    return outcome, reason, rec.manifest


def game_plan(station, index, speeds):
    """The country and speed of a station's `index`-th game.

    The country alternates, the two PCs out of step so both sides are covered at once, and
    each speed is played as both countries in turn, so side and speed are not confounded.
    """
    countries = ("BLU", "RED") if station == "here" else ("RED", "BLU")
    return countries[index % 2], speeds[index // 2 % len(speeds)]


def run_station(station, out_root, rules, templates, settings, end):
    results = []
    # A game needs about 3 minutes to launch and most end within 10; do not start one
    # that cannot plausibly finish.
    while time.monotonic() + 12 * 60 < end:
        name = time.strftime(f"ai-{station.name}-%Y%m%d-%H%M%S")
        country, speed = game_plan(station.name, len(results), settings["speeds"])
        entry = {"game": name, "station": station.name, "started_as": country, "speed": speed}
        try:
            station.quit()
            station.launch(settings["mod"])
            time.sleep(25)
            with station.connect() as desk:
                if not focus(desk):
                    raise RuntimeError("could not bring the game window to the front")
                start_game(desk, rules, out_root / f"{name}-start-failed.png", country, speed)
                say(station.name, "recording", name, "as", country, "at speed", speed)
                outcome, reason, manifest = play(
                    desk,
                    out_root / name,
                    Popups(templates),
                    settings,
                    station.name,
                    country,
                    speed,
                )
        except Exception as error:  # noqa: BLE001 - reported, then the next game is tried.
            say(station.name, "start failed:", error)
            entry["error"] = f"{type(error).__name__}: {error}"
        else:
            say(
                station.name,
                "finished",
                name,
                "winner",
                outcome,
                "after",
                manifest["seconds"],
                "s, declared by",
                manifest["declarer"],
                reason or "",
            )
            entry.update(
                winner=outcome,
                seconds=manifest["seconds"],
                frames=manifest["frames"],
                declarer=manifest["declarer"],
                players=manifest["players"],
                complete=manifest["complete"],
                reason=reason,
            )
        results.append(entry)
        path = out_root / f"results-{station.name}-{time.strftime('%Y%m%d')}.json"
        path.write_text(json.dumps(results, indent=2))
    try:
        station.quit()
    except Exception as error:  # noqa: BLE001 - the games are already saved.
        say(station.name, "quit failed:", error)
    say(station.name, "done", json.dumps(results))
    return results


def record_ai_games(
    output,
    minutes,
    *,
    mod="artifacts/mods/arena-12x8-v2",
    rules="artifacts/calibration-1080p/rules.json",
    ok_button=("artifacts/screens-1080p/ok-button.png",),
    hz=5,
    codec="x264",
    cap_minutes=45,
    speeds=(4, 5),
    peer=None,
    peer_only=False,
):
    """Record on this PC, the second PC, or both at once, until `minutes` run out."""
    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)
    screen_rules = ScreenRules(rules)
    templates = [np.asarray(Image.open(path).convert("RGB")) for path in ok_button]
    if not speeds or any(speed not in (4, 5) for speed in speeds):
        raise ValueError("speeds are 4 or 5")
    settings = {
        "mod": mod,
        "hz": hz,
        "codec": codec,
        "cap_minutes": cap_minutes,
        "speeds": list(speeds),
    }
    stations = [] if peer_only else [Station("here")]
    if peer:
        deploy = pwsh(
            "-File", str(SCRIPTS / "Deploy-Peer.ps1"), "-SkipBuild", "-PeerConfig", peer,
            "-Mod", mod,
        )  # fmt: skip
        if deploy.returncode:
            raise RuntimeError(f"Deploy-Peer failed: {deploy.stdout}{deploy.stderr}")
        stations.append(Station("peer", peer))
    if not stations:
        raise ValueError("--peer-only needs --peer")
    end = time.monotonic() + minutes * 60
    results = {}

    def run(station):
        results[station.name] = run_station(
            station, out_root, screen_rules, templates, settings, end
        )

    threads = [threading.Thread(target=run, args=(s,)) for s in stations]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results
