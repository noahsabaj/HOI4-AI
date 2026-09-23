"""Record AI-vs-AI games on an arena map, one after another, until a time budget runs out.

Each game launches HOI4 with the arena, starts as Blue, hands both countries to the AI with
the `observe` console command, sets speed 4, and records native frames while the camera
moves and clears popups the way a player would. The arena mod reports itself in the game's
log, so a game ends when the log names a surrender and its winner, or at the cap, and a
game whose weekly report stops has a stuck clock.

With a peer, the second PC records its own games at the same time, driven over its worker
connection; its frames are recorded here. The second PC's worker must be running and the
arena deployed to it (Deploy-Peer.ps1 -Mod), which this does itself.

The recordings carry no actions, so they cannot teach clicks. They are for the encoder, for
predicting who wins, and for measuring how often a match ends inside the time limit.

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
from .desktop import Desktop, DesktopError
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
    """This PC, or the second PC through its worker bridge and shared folder."""

    def __init__(self, name, peer=None):
        self.name, self.peer = name, peer
        if peer:
            host = json.loads(Path(peer).read_text())["host"]
            self.share = Path(rf"\\{host}\HOI4Worker")

    def connect(self):
        return RemoteDesktop(self.peer) if self.peer else Desktop()

    def _request(self, line, timeout=240):
        # The second PC's idle bridge notices launch.txt, runs it, and writes the outcome.
        # It only does so between connections, so none may be open while this waits.
        result = self.share / "launch-result.txt"
        result.unlink(missing_ok=True)
        (self.share / "launch.txt").write_text(line)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(2)
            try:
                text = result.read_text(errors="replace").strip()
            except OSError:
                continue
            if text.endswith("end of request"):
                return text.removesuffix("end of request").strip()
        raise RuntimeError(f"second PC did not answer '{line}' within {timeout} s")

    def quit(self):
        if self.peer:
            self._request("quit")
        else:
            # Asked to close first, as on the second PC: a game killed outright left the
            # next one there hanging at startup.
            pwsh(
                "-Command",
                "Get-Process hoi4 -ErrorAction SilentlyContinue | ForEach-Object { [void]$_.CloseMainWindow() }; "
                "$end = (Get-Date).AddSeconds(30); "
                "while ((Get-Process hoi4 -ErrorAction SilentlyContinue) -and (Get-Date) -lt $end) { Start-Sleep 1 }; "
                "Get-Process hoi4 -ErrorAction SilentlyContinue | Stop-Process -Force; "
                "while (Get-Process hoi4 -ErrorAction SilentlyContinue) { Start-Sleep 1 }",
            )  # fmt: skip
        # Let the previous launch's watcher put the player's display settings back first.
        time.sleep(5)

    def launch(self, mod):
        if self.peer:
            out = self._request(f"{Path(mod).name} {WINDOW}")
            if "Timed out waiting for the game log" in out:
                # A game that never gets far enough to log has hung at startup; a stuck
                # Discord overlay did that once. Restart Discord and try once more.
                say(self.name, "launch hung; restarting Discord and retrying")
                self._request("restart-discord")
                self.quit()
                out = self._request(f"{Path(mod).name} {WINDOW}")
            if "Arena load test PID" not in out or "Timed out" in out:
                raise RuntimeError(f"second PC did not launch: {out}")
        else:
            run = pwsh(
                "-File", str(SCRIPTS / "Test-ArenaLoad.ps1"), "-Mod", str(mod), "-Window", WINDOW
            )
            out = run.stdout.strip() + run.stderr.strip()[-200:]
        say(self.name, "launch:", out.replace("\n", " | "))


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
        desk, [{"kind": "move", "x": 0.5, "y": 0.5}] + [{"kind": "wheel", "delta": -120}] * 14, 0.05
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


def start_game(desk, rules, failure_shot):
    """From the main menu to an AI-vs-AI game running at speed 4."""
    click(desk, *SINGLE_PLAYER)
    time.sleep(8)
    click(desk, *NEW_GAME)
    time.sleep(40)
    click(desk, *SELECT_COUNTRY)  # Blue is preselected.
    time.sleep(40)
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


def camera(desk, stop, station, popups, recentre_every=(60, 150)):
    """Look around like a player, clear popups, and come back to the whole arena.

    Every pan is undone by the opposite pan of the same length, and every zoom-in by a
    zoom-out at the same pointer position, but that still drifted: two 32-minute games
    both ended zoomed in on one side, because pan speed changes with zoom. So every
    minute or two the camera zooms fully out and recentres on what it sees, which is also
    the view the territory reward reads. It shares the recorder's connection: the second
    PC's bridge accepts only one.
    """
    rng = random.Random()
    opposite = {0x25: 0x27, 0x27: 0x25, 0x26: 0x28, 0x28: 0x26}
    next_recentre = time.monotonic() + rng.uniform(*recentre_every)

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

    def linger():
        # Point around while away, like a player reading the map.
        for _ in range(rng.randint(1, 3)):
            if stop.wait(rng.uniform(0.6, 1.8)):
                return
            clear_popup()
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.15, 0.85)
            do([{"kind": "move", "x": x, "y": y}])

    try:
        while not stop.wait(rng.uniform(0.8, 2.5)):
            try:
                clear_popup()
                roll = rng.random()
                if time.monotonic() >= next_recentre:
                    recentre(desk)
                    next_recentre = time.monotonic() + rng.uniform(*recentre_every)
                    linger()
                elif roll < 0.35:
                    vk = rng.choice(list(opposite))
                    seconds = rng.uniform(0.08, 0.25)
                    hold(desk, vk, seconds)
                    linger()
                    hold(desk, opposite[vk], seconds)
                elif roll < 0.65:
                    x, y = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
                    notches = rng.randint(1, 4)
                    at = {"kind": "move", "x": x, "y": y}
                    do([at] + [{"kind": "wheel", "delta": 120}] * notches)
                    linger()
                    do([at] + [{"kind": "wheel", "delta": -120}] * notches)
                else:
                    linger()
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


def play(desk, root, popups, settings, station):
    stop = threading.Event()
    mover = threading.Thread(target=camera, args=(desk, stop, station, popups), daemon=True)
    outcome, reason = "timeout", None
    first = desk.capture()
    hz = settings["hz"]
    rec = Recorder(root, first, game_speed=4, source="ai", hz=hz, codec=settings["codec"])
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
            rec.append(frame)
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
            seconds=round(time.monotonic() - start),
            late_ticks=late,
            arena=Path(settings["mod"]).name,
            driver="observe + scripted camera + popup clicks",
            station=station,
        )
        rec.close(complete=reason is None, reason=reason)
    return outcome, reason, rec.manifest


def run_station(station, out_root, rules, templates, settings, end):
    results = []
    # A game needs about 3 minutes to launch and most end within 10; do not start one
    # that cannot plausibly finish.
    while time.monotonic() + 12 * 60 < end:
        name = time.strftime(f"ai-{station.name}-%Y%m%d-%H%M%S")
        entry = {"game": name, "station": station.name}
        try:
            station.quit()
            station.launch(settings["mod"])
            time.sleep(25)
            with station.connect() as desk:
                if not focus(desk):
                    raise RuntimeError("could not bring the game window to the front")
                start_game(desk, rules, out_root / f"{name}-start-failed.png")
                say(station.name, "recording", name)
                outcome, reason, manifest = play(
                    desk, out_root / name, Popups(templates), settings, station.name
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
                "s",
                reason or "",
            )
            entry.update(
                winner=outcome,
                seconds=manifest["seconds"],
                frames=manifest["frames"],
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
    mod="artifacts/mods/arena-12x8-v1",
    rules="artifacts/calibration-1080p/rules.json",
    ok_button=("artifacts/screens-1080p/ok-button.png",),
    hz=5,
    codec="x264",
    cap_minutes=45,
    peer=None,
    peer_only=False,
):
    """Record on this PC, the second PC, or both at once, until `minutes` run out."""
    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)
    screen_rules = ScreenRules(rules)
    templates = [np.asarray(Image.open(path).convert("RGB")) for path in ok_button]
    settings = {"mod": mod, "hz": hz, "codec": codec, "cap_minutes": cap_minutes}
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
