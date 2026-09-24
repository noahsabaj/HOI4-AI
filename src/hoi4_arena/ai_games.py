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

With `player="scripted"` the recorder's country is not handed to the AI: the scripted player
(scripted.py) fights it through the interface, and its orders are recorded as labels too.

It takes over the screen of each PC it uses. Anything else that takes focus stops input to
the game until the recorder brings it back.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .arena_log import ArenaLog
from .desktop import Desktop, DesktopError, local_control_args
from .recording import open_recorder, pixels
from .remote import RemoteDesktop
from .telemetry import game_fits, pagefile_policy
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
# Keys whose character is not their virtual-key code: the space, and the period in an
# event id (VK_OEM_PERIOD). Letters and digits are their own codes.
CONSOLE_KEYS = {" ": 0x20, ".": 0xBE}
# The arena's events that start the war (arenas since v4): Blue declares, Red declares.
DECLARE_EVENT = {"BLU": "arena.1", "RED": "arena.2"}
# Fired from the console, even a hidden event opens its window ("Blue declares war on
# Red", Ok), wider than a popup's Ok. Left open, it covered the map through the scripted
# player's setup on 2026-09-23. The crop of its Ok scored 0.000 there, 0.26 and up
# elsewhere.
EVENT_OK = "artifacts/screens-1080p/event-ok.png"
# The top bar and the bottom panels are chrome, not map.
MAP_TOP, MAP_BOTTOM = 80, 120
# The selected country's flag in the picker's top bar, 1080p pixels (x0, y0, x1, y1), and
# the player's own flag at the top left of the map.
PICKER_FLAG, TOP_FLAG = (1480, 25, 1545, 60), (10, 10, 50, 40)
# Close enough to the middle, as a fraction of the screen.
CENTRED = 0.03
# Camera zoom in mouse-wheel notches in from fully out, measured at 1080p (see camera).
# Unit counters show from 9; past 22 the map is terrain; 26 is the closest.
VIEW_NEAR, VIEW_FAR, ZOOM_TERRAIN, ZOOM_MAX = 9, 20, 22, 26
# The camera's view of the front (camera): the zooms it keeps the front at, with its
# counters shown and most of a 12x8 arena's front on screen; the notches a close look at
# a battle adds; how far the front's middle may sit from the map's before the camera pans
# to it, as a fraction of the screen; and the fewest front points that count as a front.
FRONT_ZOOM, INSPECT, OFF_CENTRE, MIN_FRONT = (10, 13), (4, 7), 0.12, 40
# The middle of the map between the top bar and the bottom panels, as a screen fraction.
MAP_MIDDLE = (MAP_TOP + 1080 - MAP_BOTTOM) / 2 / 1080
# Seconds to wait for a start save to load onto its paused map, from the launch.
SAVE_LOAD = 150
# The share of the commit limit the second PC can grow to (telemetry.commit_ceiling) at
# which no new game is started, and the commit charge a running arena game takes there
# (about 5.2 GB, 2026-09-24).
MEMORY_LIMIT, GAME_MB = 0.95, 5300
# Loading the next game from inside the running one (load_in_game), 1080p: the menu
# button at the top right, the menu's Load Game, the winner's peace conference's Confirm
# and Exit, the load dialog's Load, and a point over its list of saves. Each HOI4 launch
# on the second PC left about 100 MB of commit charge behind until a reboot
# (2026-09-24); a load from the menu took 3.3 s to the paused map.
MENU_BUTTON, MENU_LOAD_GAME = (1900 / 1920, 15 / 1080), (960 / 1920, 340 / 1080)
CONFERENCE_EXIT, LOAD_BUTTON = (1770 / 1920, 54 / 1080), (1042 / 1920, 790 / 1080)
SAVE_LIST, CONFIRM_OK = (960 / 1920, 600 / 1080), (1052 / 1920, 676 / 1080)
SCREENS = Path("artifacts/screens-1080p")
# The least TM_CCOEFF_NORMED at which these screens count as shown: 1.00 where each
# showed, at most 0.79 on every other screen tried.
SHOWN = 0.9
# Games loaded in a row before HOI4 is launched afresh anyway, and the seconds allowed for
# clearing the end of a game until the menu opens.
LOADS_PER_LAUNCH, MENU_SECONDS = 8, 30
# More buttons that close a popup at the end of a game (open_menu), as SCREENS/<name>.png:
# the winner's "equipment seized" popup's OK, in capitals. Its conference's Confirm and
# Exit (conference-exit.png) is cropped short of where the pointer rests after a click.
END_POPUPS = ("ok-capitals",)
# Seconds between moving onto something and pressing it: longer than one 200 ms decision,
# so a recorded click is always pressed where the pointer already was, as a player sees
# the button light up before clicking it.
LOOK = (0.25, 0.45)


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
        # Searched at half size, then confirmed at full size where the best match was:
        # two full-frame searches took ~350 ms on the second PC and cost frames.
        self.small = [np.ascontiguousarray(t[::2, ::2]) for t in templates]
        self.rng = rng or random.Random()
        self.clock = clock
        self.lock = threading.Lock()
        self.pending = None
        self.busy = threading.Event()

    def look_aside(self, rgb):
        """look() on its own thread, so that the recording does not wait for it; skipped
        while the last one still runs."""
        if self.busy.is_set():
            return
        self.busy.set()

        def run():
            try:
                self.look(rgb)
            finally:
                self.busy.clear()

        threading.Thread(target=run, daemon=True).start()

    def look(self, rgb):
        import cv2

        with self.lock:
            if self.pending is not None:
                return
        small = np.ascontiguousarray(rgb[::2, ::2])
        for template, little in zip(self.templates, self.small, strict=True):
            scores = cv2.matchTemplate(small, little, cv2.TM_SQDIFF_NORMED)
            score, _, (x, y), _ = cv2.minMaxLoc(scores)
            if score > 4 * OK_MATCH:
                continue
            h, w = template.shape[:2]
            x0, y0 = max(0, 2 * x - 6), max(0, 2 * y - 6)
            window = rgb[y0 : y0 + h + 12, x0 : x0 + w + 12]
            found = find_template(window, template, OK_MATCH)
            if found is not None:
                at = (
                    (x0 + found[0] * window.shape[1]) / rgb.shape[1],
                    (y0 + found[1] * window.shape[0]) / rgb.shape[0],
                )
                with self.lock:
                    self.pending = (at, self.clock() + self.rng.uniform(1, 4))
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
        self._pagefile = None

    def pagefile(self):
        """How this PC's pagefile may grow (pagefile_policy), read once from its worker's
        report; None where the report cannot be read or does not say."""
        if self._pagefile is None:
            self._pagefile = {}
            try:
                with self.connect(attach=False) as desk:
                    self._pagefile = pagefile_policy(desk.report()) or {}
            except Exception:  # noqa: BLE001 - no reading: the limit as it stands.
                pass
        return self._pagefile or None

    def connect(self, attach=True):
        if self.peer:
            return RemoteDesktop(self.peer, attach=attach)
        return Desktop(worker_args=local_control_args(), attach=attach)

    def quit(self):
        with self.connect(attach=False) as desk:
            desk.quit()
        # Let the previous launch's watcher put the player's display settings back first.
        time.sleep(5)

    def launch(self, mod, save=None):
        name = Path(mod).name
        with self.connect(attach=False) as desk:
            out = desk.launch(name, window=WINDOW, save=save)
            if "Timed out waiting for the game log" in out:
                # A game that never gets far enough to log has hung at startup; a stuck
                # Discord overlay did that once. Restart Discord and try once more.
                say(self.name, "launch hung; restarting Discord and retrying")
                desk.restart_discord()
                desk.quit()
                time.sleep(5)
                out = desk.launch(name, window=WINDOW, save=save)
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
    keys = [e for c in command.upper() for e in tap(CONSOLE_KEYS.get(c, ord(c)))]
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
    """A full frame. The first capture on a new connection is sometimes all black.

    Black but for the pointer, too, since the worker draws it: judged by the mean.
    """
    for _ in range(tries):
        rgb = on_screen(desk.capture(full=True)).rgb
        if rgb.mean() > 1:
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


def picked(rgb, box=PICKER_FLAG):
    """The country the picker shows as selected, by its flag in the top bar, or None; with
    `box` TOP_FLAG, the country played, by its flag at the top left of the map."""
    x0, y0, x1, y1 = box
    r, _, b = rgb[y0:y1, x0:x1].reshape(-1, 3).mean(0)
    return "RED" if r - b > 40 else "BLU" if b - r > 40 else None


def shown(rgb, name, threshold=SHOWN):
    """Where a calibrated screen element (SCREENS/<name>.png) is, as fractions, or None."""
    import cv2

    template = np.asarray(Image.open(SCREENS / f"{name}.png").convert("RGB"))
    scores = cv2.matchTemplate(rgb, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, (x, y) = cv2.minMaxLoc(scores)
    if best < threshold:
        return None
    h, w = template.shape[:2]
    return (x + w / 2) / rgb.shape[1], (y + h / 2) / rgb.shape[0]


def can_load(save):
    """Whether a start save can be picked from the load dialog: its name is calibrated."""
    return bool(save) and (SCREENS / f"save-{save}.png").exists()


def open_menu(desk, templates, seconds=MENU_SECONDS):
    """Open the game menu over whatever the end of a game left up: each popup's Ok, the
    winner's peace conference (Confirm and Exit, then OK) and the events after it, as
    they come, then the menu button, until the menu shows. True if it did in time.

    After a win, 33 of 36 loads fell back to a launch (2026-09-24): the winner's
    conference opens under an "equipment seized" popup, whose button reads OK in capitals,
    which the popups' Ok templates missed (END_POPUPS), and that popup blocks every other
    click."""
    templates = [*templates, *(
        np.asarray(Image.open(path).convert("RGB"))
        for path in (SCREENS / f"{name}.png" for name in END_POPUPS) if path.exists()
    )]  # fmt: skip
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rgb = screen(desk)
        if shown(rgb, "menu-load-game") is not None:
            return True
        popup = next((f for t in templates if (f := find_template(rgb, t, OK_MATCH))), None)
        if popup is not None:
            click(desk, *popup)
            time.sleep(0.8)
        elif shown(rgb, "conference-exit") is not None:
            click(desk, *CONFERENCE_EXIT)
            time.sleep(1)
            click(desk, *CONFIRM_OK)
            time.sleep(1)
        else:
            click(desk, *MENU_BUTTON)
            time.sleep(1)
    return False


def load_in_game(desk, save, templates, rules, failure_shot):
    """The next game from inside the running one: the end-of-game screens cleared and the
    menu opened (open_menu), Load Game, the save picked from the list by its name
    (scrolling down to it), Load, and the paused map. Raises RuntimeError where a screen
    does not show, keeping a screenshot when it is the menu."""
    if not open_menu(desk, templates):
        shot = failure_shot.with_name(failure_shot.name.replace("start-failed", "menu-failed"))
        Image.fromarray(screen(desk)).save(shot)
        raise RuntimeError(f"the game menu did not open (screen in {shot.name})")
    click(desk, *MENU_LOAD_GAME)
    time.sleep(1.5)
    if shown(screen(desk), "load-dialog-load") is None:
        raise RuntimeError("the load dialog did not open")
    for _ in range(12):
        # Off the list first: a row under the pointer shows a tooltip over the next ones.
        act(desk, [{"kind": "move", "x": 0.3, "y": 0.5}])
        time.sleep(0.4)
        name = shown(screen(desk), f"save-{save}", 0.93)
        if name is not None:
            break
        wheel = [{"kind": "wheel", "delta": -120}] * 3
        act(desk, [{"kind": "move", "x": SAVE_LIST[0], "y": SAVE_LIST[1]}, *wheel], 0.05)
    else:
        raise RuntimeError(f"no save called {save} in the load dialog")
    click(desk, *name)
    time.sleep(0.6)
    click(desk, *LOAD_BUTTON)
    wait_paused(desk, rules, failure_shot, 90)


def menu_ready(desk, seconds=60, still=2.0):
    """Wait for the main menu after a launch: a lit screen that holds still. It was up
    when the launch returned on the second PC (2026-09-24), but a cold start is slower."""
    deadline, last = time.monotonic() + seconds, None
    while time.monotonic() < deadline:
        rgb = screen(desk)[::8, ::8].astype(np.int16)
        if rgb.mean() > 20 and last is not None and np.abs(rgb - last).mean() < still:
            return True
        last = rgb
        time.sleep(1)
    return False


def own_land(crop, country):
    """The largest patch of `country`'s land colour in `crop`, or None if there is little.

    Not vision.country_pixels, which keeps the largest patch of either colour: on the
    picker that is the selected country, and the other shows only as a sliver beside it,
    cut off by the glowing border.
    """
    import cv2

    pixels = np.asarray(crop, dtype=np.int32)
    r, b = pixels[..., 0], pixels[..., 2]
    land = pixels.sum(-1) > 250
    mask = land & ((r - b > 15) if country == "RED" else (b - r > 10))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=4
    )
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 2000:
        return None
    return labels == largest


def pick_country(desk, country, tries=8):
    """On the country picker, click the middle of `country`'s land until its flag shows.

    The picker's map can still be black when the recorder gets there, and on the second
    PC a click made then left Blue selected twice, so each click waits for the land and
    is checked against the selected flag. The picker's camera closes in on Blue: on the
    first marsh arena Red's land was off the screen (2026-09-24), so when the country
    does not show, the map is zoomed out a little and looked at again. False if the pick
    never took.
    """
    for _ in range(tries):
        # The drawn pointer's glove reads as red land, and where little of Red shows it
        # was the largest red patch: move it up onto the top bar, off the map, first.
        act(desk, [{"kind": "move", "x": 0.3, "y": 0.015}])
        rgb = screen(desk)
        if picked(rgb) == country:
            return True
        top, bottom = MAP_TOP, rgb.shape[0] - MAP_BOTTOM
        land = own_land(rgb[top:bottom], country)
        if land is None:
            act(
                desk,
                [{"kind": "move", "x": 0.5, "y": 0.5}] + [{"kind": "wheel", "delta": -120}] * 4,
            )
            time.sleep(2)
            continue
        ys, xs = np.nonzero(land)
        # The land pixel nearest the median, so the click is on the land whatever its shape.
        k = np.argmin((ys - np.median(ys)) ** 2 + (xs - np.median(xs)) ** 2)
        click(desk, xs[k] / rgb.shape[1], (ys[k] + top) / rgb.shape[0])
        time.sleep(1.5)
    return picked(screen(desk)) == country


def run_at(desk, rules, speed, failure_shot=None):
    """Unpause a game paused at speed 1 and run it at `speed`, 4 or 5."""
    for _ in range(3):
        click(desk, *SPEED_UP)  # The on-screen + button: speed 1 to 4.
    act(desk, [{"kind": "move", "x": 0.5, "y": 0.75}])
    act(desk, tap(0x20))  # Unpause
    time.sleep(2)
    rgb = screen(desk)
    if not rules.matches("speed", rgb) or rules.matches("paused", rgb):
        if failure_shot:
            Image.fromarray(rgb).resize((960, 540)).save(failure_shot)
        raise RuntimeError("game is not running at speed 4")
    if speed == 5:
        # Checked at 4, where the rule was calibrated, then one more.
        click(desk, *SPEED_UP)


def wait_paused(desk, rules, failure_shot, seconds):
    """Wait up to `seconds` for a paused game's map. The pause mark is the same on both
    PCs; the alert row that "healthy" reads is not, as each PC shows different alerts. The
    mark blinks, so one frame can catch it faded: it is looked for again and again."""
    deadline = time.monotonic() + seconds
    while True:
        rgb = screen(desk)
        if rules.matches("paused", rgb):
            return
        if time.monotonic() > deadline:
            Image.fromarray(rgb).resize((960, 540)).save(failure_shot)
            raise RuntimeError("game did not reach the map")
        time.sleep(0.25)


def run_briefly(desk, rules, seconds):
    """Let a game paused at its start run for `seconds` at speed 1, then pause it again.

    Games that start alike play alike: with the same side and the same declarer, every
    game's daily reports were identical to the hour until the script's first law change,
    about 100 days in (2026-09-24). A few game hours before the war begins, a different
    number each game, give each game its own random draws from the start.
    """
    act(desk, [{"kind": "move", "x": 0.5, "y": 0.75}])
    act(desk, tap(0x20))
    time.sleep(seconds)
    act(desk, tap(0x20))
    for _ in range(2):
        # The mark blinks: look for it over two seconds before pressing again.
        for _ in range(8):
            time.sleep(0.25)
            if rules.matches("paused", screen(desk)):
                return
        act(desk, tap(0x20))
    raise RuntimeError("the game would not pause again after its opening hours")


def start_game(
    desk,
    rules,
    failure_shot,
    country="BLU",
    speed=4,
    observe=True,
    declarer=None,
    saved=False,
    opening=0.0,
    save_as=None,
):
    """From the main menu to an AI-vs-AI game running at `speed` (4 or 5), as `country`.

    Which country the game starts as is varied because only one side ever won while the
    recorder always started as Blue; the arena logs the country each human started as.
    Without `observe` the game is left paused at its start, as `country`, for the scripted
    player (scripted.Planner) to set up and run while it is recorded.

    `declarer` starts the war from the console (arenas since v4). The game's own coin
    flip at startup is not random: its random draw comes out the same for the same setup,
    and Red declared in 36 of 44 AI games and in all 13 started as Red on the second PC.
    The recorder flips the coin instead. Arenas before v4 have declared already, and the
    command does nothing there; the arena log's `declare` line is the truth either way.

    With `saved`, the game was launched straight into a start save of `country` (made
    paused at the start of a new game), so the menus are skipped and the map is waited
    for. `opening` runs the game that many seconds at speed 1 first (run_briefly).
    """
    if saved:
        wait_paused(desk, rules, failure_shot, SAVE_LOAD)
        if picked(screen(desk), TOP_FLAG) != country:
            Image.fromarray(screen(desk)).resize((960, 540)).save(failure_shot)
            raise RuntimeError(f"the start save does not play {country}")
    else:
        # Each menu was up within a second of its click on the second PC (2026-09-24),
        # where the recorder had waited 8, 40 and 40 s.
        menu_ready(desk)
        click(desk, *SINGLE_PLAYER)
        time.sleep(3)
        click(desk, *NEW_GAME)
        time.sleep(4)
        click(desk, *SELECT_COUNTRY)  # Blue is preselected.
        time.sleep(4)
        # Clicking a country's land on the picker's map selects it; Blue is the default.
        if not pick_country(desk, country):
            Image.fromarray(screen(desk)).resize((960, 540)).save(failure_shot)
            raise RuntimeError(f"could not pick {country} on the country picker")
        click(desk, *START)
        # A new game starts paused, its map up 3.5-4.5 s after Start.
        wait_paused(desk, rules, failure_shot, 60)
        if save_as:
            # Saved paused at the start, before the opening hours or the war: the next
            # game on this arena and side starts from here, skipping the menus.
            console(desk, f"savegame {save_as}")
            time.sleep(3)
    if opening > 0:
        run_briefly(desk, rules, opening)
    if declarer:
        console(desk, f"event {DECLARE_EVENT[declarer]}")
        close_event(desk)
    if not observe:
        return
    console(desk, "observe")
    recentre(desk)
    run_at(desk, rules, speed, failure_shot)


def close_event(desk, seconds=6):
    """Click Ok on the event window the console's `event` opens. True if one closed."""
    template = np.asarray(Image.open(EVENT_OK).convert("RGB"))
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        found = find_template(screen(desk), template, OK_MATCH)
        if found is not None:
            click(desk, *found)
            time.sleep(0.5)
            return True
        time.sleep(0.5)
    return False


def closeups(desk, root, per_side=4):
    """Full-resolution views of the map zoomed fully in, past the political map into the
    terrain view, over a spread of points in each country's land: whoever builds an arena
    wants to see its terrain, rivers and cities as a player does, and the recordings never
    zoom that far. Taken paused, before the recording starts. The paths, in order."""
    recentre(desk)
    time.sleep(1)
    rgb = screen(desk)
    top, bottom = MAP_TOP, rgb.shape[0] - MAP_BOTTOM
    blue, red = country_pixels(rgb[top:bottom])
    shots = []
    if blue is None:
        return shots
    for land in (blue, red):
        ys, xs = np.nonzero(land)
        if not len(xs):
            continue
        side = int(round(per_side**0.5))
        for fy in np.linspace(0.25, 0.75, side):
            for fx in np.linspace(0.25, 0.75, side):
                # The land pixel nearest this point of the country's box.
                x = xs.min() + fx * (xs.max() - xs.min())
                y = ys.min() + fy * (ys.max() - ys.min())
                k = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
                at = (xs[k] / rgb.shape[1], (ys[k] + top) / rgb.shape[0])
                act(desk, [{"kind": "move", "x": at[0], "y": at[1]}])
                act(desk, [{"kind": "wheel", "delta": 120}] * ZOOM_MAX, 0.04)
                # Off the map, onto the top bar's blank middle, so that neither the
                # pointer nor a tooltip is in the view.
                act(desk, [{"kind": "move", "x": 0.65, "y": 0.012}])
                time.sleep(1.5)
                path = Path(root) / f"closeup-{len(shots) + 1}.png"
                Image.fromarray(screen(desk)).save(path)
                shots.append(path)
                recentre(desk)
    return shots


def front_points(rgb):
    """Where Blue's land meets Red's, as (x, y) screen fractions: the front line.

    Occupied land takes its occupier's colour, so this follows the fighting, not the
    starting border. Empty when the two do not touch on screen. A thin line is not land:
    the battle plan's red lines over Blue's land read as a front until 2026-09-24, and
    held the camera on empty land for most of a game.
    """
    import cv2

    top, bottom = MAP_TOP, rgb.shape[0] - MAP_BOTTOM
    blue, red = country_pixels(rgb[top:bottom])
    if blue is None or not blue.any() or not red.any():
        return []
    solid = np.ones((9, 9), np.uint8)
    blue = cv2.morphologyEx(blue.astype(np.uint8), cv2.MORPH_OPEN, solid)
    red = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN, solid).astype(bool)
    near_blue = cv2.dilate(blue, solid).astype(bool)
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


def front_middle(front):
    """The middle of the front's points: the median, so a pocket does not pull it off."""
    points = np.asarray(front)
    return float(np.median(points[:, 0])), float(np.median(points[:, 1]))


def _blobs(mask, top, shape, fits):
    import cv2

    n, _, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    found = [
        (int(stats[k, 4]), (centres[k][0] / shape[1], (centres[k][1] + top) / shape[0]))
        for k in range(1, n)
        if fits(stats[k, 2], stats[k, 3], stats[k, 4])
    ]
    return [at for _, at in sorted(found, reverse=True)]


def battle_points(rgb, front=None, reach=0.05):
    """Battles on screen, largest first, as (x, y) screen fractions: the game's round badges
    on the front, whose ring is pure green (about 3, 158, 3 at 1080p, 2026-09-24), a colour
    nothing else on the map has. With `front` points, only badges within `reach` of one.
    """
    import cv2

    top, bottom = MAP_TOP, rgb.shape[0] - MAP_BOTTOM
    area = rgb[top:bottom].astype(np.int16)
    green = (area[..., 1] > 120) & (area[..., 0] < 50) & (area[..., 2] < 50)
    # The number printed over the ring cuts it into pieces: join them first.
    ring = cv2.dilate(green.astype(np.uint8), np.ones((7, 7), np.uint8))
    points = _blobs(ring, top, rgb.shape, lambda w, h, a: 22 <= w <= 70 and 22 <= h <= 70
                    and 0.6 <= w / h <= 1.7)  # fmt: skip
    if front is not None and len(front):
        f = np.asarray(front)
        points = [p for p in points if np.hypot(*(f - p).T).min() < reach]
    return points


def counter_points(rgb):
    """Unit counters on screen, as (x, y) screen fractions: their flags, small boxes of
    saturated blue or red, where the land around them is pastel."""
    top, bottom = MAP_TOP, rgb.shape[0] - MAP_BOTTOM
    area = rgb[top:bottom].astype(np.int16)
    r, g, b = area[..., 0], area[..., 1], area[..., 2]
    blue = (b > 150) & (b - r > 90) & (b - g > 60)
    red = (r > 150) & (r - g > 110) & (r - b > 90)

    def fits(w, h, a):
        return 60 <= a <= 900 and 8 <= w <= 60 and 6 <= h <= 30 and a >= 0.5 * w * h

    return _blobs(blue, top, rgb.shape, fits) + _blobs(red, top, rgb.shape, fits)


def busiest(points, radius=0.08):
    """The point with the most others within `radius`: where the counters crowd."""
    p = np.asarray(points)
    near = (np.hypot(*(p[:, None] - p[None]).transpose(2, 0, 1)) < radius).sum(1)
    return tuple(p[int(near.argmax())])


def near_front(points, front, reach=0.06):
    """The `points` within `reach` of the front, as screen fractions."""
    if not points or not len(front):
        return []
    f = np.asarray(front)
    return [p for p in points if np.hypot(*(f - p).T).min() < reach]


def camera(desk, stop, station, popups, overview_every=(40, 80), rng=None, planner=None):
    """Watch the war like a commander: the front in view with its counters, close looks at
    its battles, and the whole map now and then.

    Zoom is in wheel notches in from fully out, measured at 1080p on 2026-09-23: the whole
    arena fills the middle half of the screen at 0, unit counters appear from 9 (the game
    hides them beyond a camera distance of 900), about a third of the arena shows at 18,
    the map turns to terrain past about 22, and 26 is as close as it goes.

    The camera keeps the front, where Blue's land meets Red's, in the middle of the map at
    FRONT_ZOOM, where the counters show and most of the front fits. It pans back when the
    front's middle drifts OFF_CENTRE away, and when the front leaves the screen it zooms
    out until the front shows and closes in on it again, never panning blind. From that
    view it looks closer: in by INSPECT notches on a battle (or where the counters crowd),
    the pointer on it to read it, and back out by as many, which returns to the same view
    since zooming keeps the point under the pointer; or it runs the pointer along the
    front. Every 40 to 80 seconds it zooms fully out for a few seconds and comes straight
    back in on the front. Every move follows from what is on screen, so a policy trained on
    these recordings can learn it: the camera before 2026-09-24 moved at random, and sat
    on empty land or sea for seconds at a time. It shares the recorder's connection: the
    second PC's bridge accepts only one.

    With a scripted player's `planner`, the planner first sets the paused game up and
    starts it, then gives its later orders between the camera's moves. A setup that fails
    is left in `planner.error`, which ends the game; a later order that fails is retried.
    """
    rng = rng or random.Random()
    zoom = 0
    next_overview = time.monotonic() + rng.uniform(*overview_every)
    # Screen fractions an arrow key pans the camera in a second, across and down, learnt
    # from each pan; and the last pan, to learn from.
    speed, last_pan = [1.25, 1.25], None

    def do(events, pause=0.08):
        # The worker disarms after 750 ms without input, so arm for each burst.
        desk.arm(setup=True)
        for event in events:
            desk.apply([event])
            time.sleep(pause)

    def point(at):
        # Clear of the screen's edges, where the pointer would scroll the map.
        x, y = min(0.94, max(0.06, at[0])), min(0.9, max(0.1, at[1]))
        return {"kind": "move", "x": float(x), "y": float(y)}

    def clear_popup():
        at = popups.due()
        if at:
            press = [{"kind": "button", "button": 0, "down": d} for d in (True, False)]
            do([{"kind": "move", "x": at[0], "y": at[1]}])
            # Look, then click: the press comes a decision or more after the move, so the
            # fovea has seen the button first (models.ActionHead `look`).
            stop.wait(rng.uniform(*LOOK))
            do(press)

    def wait(seconds):
        """Wait, clearing popups as they come due. True once the game has ended."""
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            if stop.wait(min(left, 0.5)):
                return True
            clear_popup()
        return stop.is_set()

    def wheel(notches, at):
        nonlocal zoom
        step = 120 if notches > 0 else -120
        notches = max(-zoom, min(ZOOM_MAX - zoom, notches))
        do([point(at)] + [{"kind": "wheel", "delta": step}] * abs(notches), rng.uniform(0.04, 0.1))
        zoom += notches

    def close_in(front):
        """Zoom toward the front's middle, into the view of the front."""
        wheel(rng.randint(*FRONT_ZOOM) - zoom, front_middle(front))

    def find_front():
        """Zoom out toward the map's middle until the front shows, then back in on it.
        False if it shows not even on the whole map."""
        nonlocal zoom
        while zoom > 0:
            wheel(-min(zoom, 4), (0.5, MAP_MIDDLE))
            front = front_points(screen(desk))
            if len(front) >= MIN_FRONT:
                close_in(front)
                return True
        recentre(desk)
        zoom = 0
        front = front_points(screen(desk))
        if len(front) < MIN_FRONT:
            return False
        close_in(front)
        return True

    def pan_to(middle):
        """Hold the arrow keys that bring `middle` to the map's middle, an axis at a time."""
        nonlocal last_pan
        held = [0.0, 0.0]
        offsets = (middle[0] - 0.5, middle[1] - MAP_MIDDLE)
        for axis, keys in ((0, (0x25, 0x27)), (1, (0x26, 0x28))):
            if abs(offsets[axis]) > OFF_CENTRE / 2:
                held[axis] = min(0.6, max(0.06, abs(offsets[axis]) / speed[axis]))
                hold(desk, keys[offsets[axis] > 0], held[axis])
        last_pan = (middle, held, zoom)

    def learn(middle):
        """The pan speed, from how far the front moved on screen after the last pan."""
        nonlocal last_pan
        if last_pan is not None and last_pan[2] == zoom:
            for axis in (0, 1):
                if last_pan[1][axis] >= 0.1:
                    moved = abs(last_pan[0][axis] - middle[axis]) / last_pan[1][axis]
                    if 0.2 <= moved <= 5:
                        speed[axis] = 0.5 * speed[axis] + 0.5 * moved
        last_pan = None

    def overview():
        nonlocal zoom
        recentre(desk)
        zoom = 0
        front = front_points(screen(desk))
        if front:
            do([point(rng.choice(front))])  # A glance over the whole map, at the front.
        if wait(rng.uniform(1.5, 3.0)):
            return
        front = front_points(screen(desk))
        if len(front) >= MIN_FRONT:
            close_in(front)

    def inspect(rgb, front):
        """In on a battle, or where the counters crowd on the front, read it, and back
        out to this view."""
        battles = battle_points(rgb, front)
        crowd = near_front(counter_points(rgb), front)
        if battles:
            at = rng.choice(battles[:3])
        else:
            at = busiest(crowd) if crowd else rng.choice(front)
        notches = min(rng.randint(*INSPECT), ZOOM_TERRAIN - 1 - zoom)
        if notches <= 0:
            return
        wheel(notches, at)
        for _ in range(rng.randint(1, 3)):
            # The pointer on it and just around it, as a player reads its tooltip.
            do([point((at[0] + rng.uniform(-0.01, 0.01), at[1] + rng.uniform(-0.01, 0.01)))])
            if wait(rng.uniform(0.6, 1.5)):
                return
        wheel(-notches, at)

    def sweep(front):
        """Run the pointer along the front, reading its provinces."""
        picks = rng.sample(front, min(len(front), rng.randint(3, 6)))
        for at in sorted(picks, key=lambda p: p[1]):
            do([point(at)])
            if wait(rng.uniform(0.3, 0.8)):
                return

    try:
        if planner is not None:
            try:
                planner.setup(desk)
            except Exception as error:  # noqa: BLE001 - play() ends the game with it.
                planner.error = error
                return
            zoom = 0  # Setup left the camera fully zoomed out over the arena.
        while not stop.wait(rng.uniform(0.3, 0.9)):
            try:
                clear_popup()
                if planner is not None and planner.due():
                    try:
                        moved = planner.step(desk)
                    except (ValueError, OSError):
                        return  # The game ended and its connection closed mid-order.
                    except RuntimeError as error:
                        say(station, "planner:", error)
                        planner.failures.append({"frame": planner.frame(), "error": str(error)})
                        moved = True
                    if moved:
                        # The planner's moves end fully zoomed out, an overview already: its
                        # guard looks every 30 s (scripted.GUARD_CHECK).
                        zoom = 0
                        next_overview = time.monotonic() + rng.uniform(*overview_every)
                    continue
                if time.monotonic() >= next_overview:
                    overview()
                    next_overview = time.monotonic() + rng.uniform(*overview_every)
                    continue
                rgb = screen(desk)
                front = front_points(rgb)
                if len(front) < MIN_FRONT:
                    if not find_front():
                        wait(rng.uniform(2, 4))  # No front, even on the whole map.
                    continue
                middle = front_middle(front)
                learn(middle)
                if not FRONT_ZOOM[0] <= zoom <= FRONT_ZOOM[1]:
                    close_in(front)
                elif max(abs(middle[0] - 0.5), abs(middle[1] - MAP_MIDDLE)) > OFF_CENTRE:
                    pan_to(middle)
                else:
                    roll = rng.random()
                    if roll < 0.45:
                        inspect(rgb, front)
                    elif roll < 0.65:
                        sweep(front)
                    else:
                        # A rest, the pointer by the front's middle.
                        do([point((middle[0] + rng.uniform(-0.05, 0.05),
                                   middle[1] + rng.uniform(-0.05, 0.05)))])  # fmt: skip
                        wait(rng.uniform(0.8, 2.0))
            except (ValueError, OSError):
                return  # The game ended and its connection closed.
            except DesktopError as error:
                if stop.is_set():
                    return  # The game ended mid-move, and its input was disarmed.
                say(station, "camera:", error)
                try:
                    desk.release()
                except DesktopError:
                    pass
                focus(desk, tries=1)
    finally:
        try:
            desk.release()
        except (DesktopError, ValueError, OSError):
            pass


def play(
    desk,
    root,
    popups,
    settings,
    station,
    country="BLU",
    speed=4,
    player=None,
    start_save=None,
    log_from=None,
):
    """Record one game to its end. With `player` (a plan, templates and screen rules), the
    scripted player fights it from its paused start; without, the game's AI plays both.
    A game loaded from a `start_save` logs no `player` line (the arena logs it at a new
    game's start): the save's country is recorded instead, checked by its flag."""
    from .scripted import Planner, arena_layout

    stop = threading.Event()
    inputs = Logged(desk)
    outcome, reason = "timeout", None
    first = desk.capture()
    hz = settings["hz"]
    source = "scripted" if player else "ai"
    rec = open_recorder(
        desk, root, first, game_speed=speed, source=source, hz=hz, codec=settings["codec"]
    )
    planner = None
    shots = {}
    if player:
        planner = Planner(
            country, player["plan"], player["templates"], player["rules"], speed,
            frame=lambda: rec.manifest["frames"], layout=arena_layout(settings["mod"]),
        )  # fmt: skip
        planner.debug_dir = Path(root)
        if player.get("shots"):
            # An arena under test: keep the planner's full views of the map, and the
            # first and last frames, for whoever built it.
            planner.overview_dir = Path(root) / "overviews"
            planner.overview_dir.mkdir(parents=True, exist_ok=True)
            shots["start"] = Path(root) / "start.png"
            Image.fromarray(pixels(desk, first)).save(shots["start"])
    mover = threading.Thread(
        target=camera,
        args=(inputs, stop, station, popups),
        kwargs={"planner": planner},
        daemon=True,
    )
    # A scripted game starts paused, and no weekly report comes until it runs.
    arena = ArenaLog(desk, silence=None if planner else WEEK_SILENCE)
    if log_from is not None:
        # A game loaded inside a running one: the log holds the games before it.
        arena.offset = log_from
    start = deadline = next_poll = time.monotonic()
    late, ending = 0, None
    # Each mod line with the number of frames recorded when it was read, which aligns the
    # arena's daily reports (v3) with the video: at speed 5 a day passes in about 0.4 s.
    stamped = []
    try:
        rec.append(first)
        mover.start()
        while time.monotonic() - start < settings["cap_minutes"] * 60:
            if rec.streamed:
                # The worker keeps the clock and records every frame itself (--codec
                # nvenc); this loop follows its frames and looks at the screen when it must.
                frame = on_screen(rec.next_frame())
                deadline = time.monotonic()
            else:
                deadline += 1 / hz
                time.sleep(max(0, deadline - time.monotonic()))
                frame = on_screen(desk.capture())
            if not frame.meta.get("foreground"):
                focus(desk, tries=1)
                continue
            rec.append(frame, scripted_events=inputs.take())
            now = time.monotonic()
            if planner is not None:
                if planner.error is not None:
                    raise RuntimeError(f"the scripted player's setup failed: {planner.error}")
                if planner.running and arena.silence is None:
                    arena.silence, arena.last_week = WEEK_SILENCE, arena.clock()
            if now - deadline > 1:
                late += 1
                deadline = now
            if rec.manifest["frames"] % int(hz) == 0:
                # About once a second, beside the recording. A streamed frame has no pixels,
                # so they are captured for it.
                popups.look_aside(pixels(desk, frame))
            if now >= next_poll:
                next_poll = now + 1
                seen = len(arena.lines)
                arena.poll()
                frames = rec.manifest["frames"]
                stamped.extend({"frame": frames, "line": line} for line in arena.lines[seen:])
                if arena.winner and ending is None:
                    # Keep a few seconds of the surrender on screen, then stop.
                    ending = now + 2
                    Image.fromarray(pixels(desk, frame)).save(Path(root) / "capitulation.png")
            if ending is not None and now >= ending:
                outcome = arena.winner
                break
            if rec.manifest["frames"] % (60 * int(hz)) == 0:
                say(station, f"{rec.manifest['frames'] // int(hz) // 60} min recorded")
    except Exception as error:  # noqa: BLE001 - recorded in the manifest.
        reason = f"{type(error).__name__}: {error}"
    finally:
        stop.set()
        mover.join(timeout=3)  # A planner step cut short fails harmlessly once the game ends.
        if planner is not None and planner.overview_dir is not None:
            shots["end"] = Path(root) / "capitulation.png"
            if not shots["end"].exists():
                shots["end"] = Path(root) / "end.png"
                try:
                    Image.fromarray(desk.capture(full=True).rgb).save(shots["end"])
                except Exception:  # noqa: BLE001 - the recording is what matters.
                    shots.pop("end")
            views = sorted(planner.overview_dir.glob("*.png"))
            if views:
                shots["mid"] = views[len(views) // 2]
        (Path(root) / "arena-log.txt").write_text("\n".join(arena.lines) + "\n")
        with (Path(root) / "arena-log.jsonl").open("w") as out:
            out.writelines(json.dumps(entry) + "\n" for entry in stamped)
        rec.manifest.update(
            winner=outcome,
            surrendered=arena.surrendered,
            # The country the recorder picked, and what the arena logged: who declared the
            # war and the country the human started as (None on arenas before v2).
            started_as=country,
            declarer=arena.declarer,
            players=arena.players or ([country] if start_save else []),
            start_save=start_save,
            seconds=round(time.monotonic() - start),
            late_ticks=late,
            arena=Path(settings["mod"]).name,
            driver=(
                "scripted player + scripted camera + popup clicks"
                if planner
                else "observe + scripted camera + popup clicks"
            ),
            # frames.jsonl carries the camera's inputs, and the scripted player's orders,
            # as scripted_events.
            labels="scripted_events",
            station=station,
        )
        if planner:
            rec.manifest.update(
                plan=player["plan"],
                orders=planner.orders,
                # Every later order that failed, with when: each is a fault to fix.
                planner_errors=planner.failures,
                screenshots={k: str(v) for k, v in shots.items()},
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


def game_arena(mods, index, accepted=()):
    """The arena of a station's `index`-th game, played as both countries in turn
    (game_plan alternates them game by game), so arena and side are not confounded.

    The arenas the recorder was given take two pairs of games in every three, and the
    arenas accepted from the test queue take turns in the rest: the main arena keeps two
    thirds of the games, where its record is measured, however many new ones join, and
    four games in a row on one arena can load inside one running game (load_in_game).
    """
    pair = index // 2
    if not accepted:
        return mods[pair % len(mods)]
    if pair % 3 != 2:
        return mods[(pair - pair // 3) % len(mods)]
    return accepted[pair // 3 % len(accepted)]


def latest_versions(arenas):
    """The arenas without those a later version replaces (arena-plains-v1 once
    arena-plains-v2 is there), in their order."""
    import re

    def split(arena):
        match = re.fullmatch(r"(.*)-v(\d+)", Path(arena).name)
        return (match.group(1), int(match.group(2))) if match else (Path(arena).name, 0)

    newest = {}
    for arena in arenas:
        base, version = split(arena)
        newest[base] = max(newest.get(base, version), version)
    return [a for a in arenas if split(a)[1] == newest[split(a)[0]]]


def take_request(queue, station, settle=2.0):
    """The oldest arena test request in `queue`, claimed for `station`, or None.

    A request is `<name>.json` holding {"mod": "<the mod's folder>"}, written by whoever
    builds arenas. Claiming renames it, so two stations never take the same one; a file
    changed in the last `settle` seconds may still be being written, and waits.
    """
    queue = Path(queue)
    if not queue.is_dir():
        return None
    for path in sorted(queue.glob("*.json"), key=lambda p: p.stat().st_mtime):
        if time.time() - path.stat().st_mtime < settle:
            continue
        claimed = path.with_name(f"{path.stem}.{owner(station)}.taken")
        try:
            path.rename(claimed)
        except OSError:
            continue  # The other station took it first.
        try:
            mod = str(json.loads(claimed.read_text(encoding="utf-8-sig"))["mod"])
        except (ValueError, KeyError, TypeError) as error:
            answer(queue, path.stem, {"accepted": False, "error": f"bad request: {error}"})
            claimed.unlink()
            continue
        return {"name": path.stem, "mod": mod, "claimed": claimed}
    return None


def owner(station):
    """Who claims a queued request: the station and this process, as `<station>-<pid>`."""
    return f"{station}-{os.getpid()}"


def alive(pid):
    """Whether a process with this id is still running."""
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def reoffer(queue):
    """Put back every request a recorder claimed and died holding (`<name>.<station>-<pid>
    .taken` whose process is gone): a run stopped between claiming a reservation and
    granting it lost one on 2026-09-24. The names put back, in order."""
    queue = Path(queue)
    back = []
    for claimed in sorted(queue.glob("*.taken")) if queue.is_dir() else []:
        stem, _, holder = claimed.name[: -len(".taken")].rpartition(".")
        pid = holder.rpartition("-")[2]
        if pid.isdigit() and alive(int(pid)):
            continue
        claimed.rename(queue / f"{stem or holder}.json")
        back.append(stem or holder)
    return back


def save_name(arena, country):
    """A start save's name for an arena and side: letters and digits only, as the game's
    -start_save and the console's savegame take them."""
    import re

    return re.sub(r"[^A-Za-z0-9]", "", Path(arena).name).lower() + country.lower()


def known_saves(path):
    """{(arena, country): save} from a registry of saves made by earlier games."""
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    return {
        (arena, country): name for arena, sides in data.items() for country, name in sides.items()
    }


def remember_save(path, arena, country, name):
    data = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    data.setdefault(arena, {})[country] = name
    Path(path).write_text(json.dumps(data, indent=2))


def parse_saves(specs):
    """{(arena, country): save} from "ARENA:COUNTRY:SAVE" strings."""
    saves = {}
    for spec in specs:
        arena, country, save = spec.split(":")
        if country not in ("BLU", "RED"):
            raise ValueError(f"a start save's country is BLU or RED, not {country}")
        saves[(arena, country)] = save
    return saves


class MemoryStop(Exception):
    """No game started: it would not fit the second PC's memory."""


def memory_room(station, need_mb=0):
    """Why a game needing `need_mb` more would not fit the second PC (game_fits: its
    commit charge under MEMORY_LIMIT of the limit it can grow to, and RAM for the game),
    or None if it would or there is nothing to read. Each HOI4 launch there leaves about
    115 MB of commit charge behind until a sign-out (2026-09-24), and a failed allocation
    could crash a game mid-game."""
    if not station.peer:
        return None
    pagefile = station.pagefile()
    try:
        from .telemetry import open_observer

        with open_observer(station.peer) as desk:
            reply = desk.telemetry()
    except Exception:  # noqa: BLE001 - no reading: carry on as before.
        return None
    return game_fits(reply, pagefile, need_mb, MEMORY_LIMIT)


def check_memory(station, need_mb=0):
    """Raise MemoryStop if a game would not fit the second PC (memory_room)."""
    reason = memory_room(station, need_mb)
    if reason:
        raise MemoryStop(reason)


def take_reservation(root, settle=2.0):
    """A request to have the second PC for a live evaluation, claimed, or None.

    Whoever evaluates a learned player writes `<root>/queue/<name>.json` holding
    {"minutes": N}. It comes before every game of the recorder's own (lend).
    """
    queue = Path(root) / "queue"
    if not queue.is_dir():
        return None
    for path in sorted(queue.glob("*.json"), key=lambda p: p.stat().st_mtime):
        if time.time() - path.stat().st_mtime < settle:
            continue
        claimed = path.with_name(f"{path.stem}.{owner('peer')}.taken")
        try:
            path.rename(claimed)
        except OSError:
            continue
        try:
            minutes = float(json.loads(claimed.read_text(encoding="utf-8-sig"))["minutes"])
        except (ValueError, KeyError, TypeError):
            minutes = 30.0  # Unreadable: lend the PC for half an hour rather than refuse.
        return {"name": path.stem, "minutes": minutes, "claimed": claimed}
    return None


def lend(station, root, reservation, poll=5.0, clock=time.monotonic, sleep=time.sleep):
    """Hand `station` over for a live evaluation: HOI4 closed, then `granted/<name>.json`
    written. It is taken back once `done/<name>.json` appears, or once the minutes asked
    for and 15 more have passed."""
    name, minutes = reservation["name"], reservation["minutes"]
    station.quit()
    granted = Path(root) / "granted"
    granted.mkdir(parents=True, exist_ok=True)
    note = {"station": station.name, "minutes": minutes, "granted": time.strftime("%H:%M:%S")}
    (granted / f"{name}.json").write_text(json.dumps(note))
    say(station.name, "lent for an evaluation:", name, f"({minutes:g} min)")
    done = Path(root) / "done" / f"{name}.json"
    deadline = clock() + (minutes + 15) * 60
    while clock() < deadline and not done.exists():
        sleep(poll)
    reservation["claimed"].unlink(missing_ok=True)
    say(
        station.name, "back from the evaluation", name, "(done)" if done.exists() else "(timed out)"
    )
    return done.exists()


def answer(queue, name, result):
    """Write a test request's result beside the queue, in results/<name>.json."""
    out = Path(queue).parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2))


def accepted_arenas(queue):
    """The mods that passed a test game, in the order they passed (accepted.json)."""
    path = Path(queue).parent / "accepted.json"
    if not path.exists():
        return []
    return [str(m) for m in json.loads(path.read_text())]


def accept_arena(queue, mod):
    arenas = accepted_arenas(queue)
    if mod not in arenas:
        arenas.append(mod)
        (Path(queue).parent / "accepted.json").write_text(json.dumps(arenas, indent=2))


def local_mod(mod, mods_dir=None):
    """Make a mod folder launchable here: the worker looks for arenas by folder name in
    artifacts/mods, so one kept elsewhere is linked in by a junction."""
    mods_dir = Path(mods_dir or Path(__file__).resolve().parents[2] / "artifacts" / "mods")
    source = Path(mod).resolve()
    link = mods_dir / source.name
    if link.exists():
        if link.resolve() != source:
            raise RuntimeError(f"another arena is already called {source.name} in {mods_dir}")
        return
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(source)], capture_output=True, text=True
    )
    if made.returncode:
        raise RuntimeError(f"could not link {source} into {mods_dir}: {made.stdout}{made.stderr}")


def deploy_mod(station, mod, settings):
    """The arena ready to launch on `station`: mirrored to the second PC, linked here."""
    if mod in settings["deployed"].setdefault(station.name, set()):
        return
    if station.peer:
        deploy = pwsh(
            "-File", str(SCRIPTS / "Deploy-Peer.ps1"), "-SkipBuild", "-PeerConfig", station.peer,
            "-Mod", mod,
        )  # fmt: skip
        if deploy.returncode:
            raise RuntimeError(f"Deploy-Peer failed: {deploy.stdout}{deploy.stderr}")
    else:
        local_mod(mod)
    settings["deployed"][station.name].add(mod)


def map_errors(station):
    """The map errors the game logged since its launch, from the worker's report (the
    report counts them in error.log): {"count", "examples"}, or None."""
    try:
        with station.connect(attach=False) as desk:
            out = desk.report()
    except Exception:  # noqa: BLE001 - a missing count must not stop the games.
        return None
    return parse_map_errors(out)


def parse_map_errors(report):
    lines = report.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("== map errors in error.log:"):
            examples = []
            for more in lines[i + 1 :]:
                if more.startswith("== end of map errors"):
                    break
                examples.append(more.strip())
            return {"count": int(line.rsplit(":", 1)[1]), "examples": examples}
    return None


def test_result(request, entry, root, stages):
    """What a queued arena's test game showed, for whoever built the arena.

    It passes if the game launched, started, and the scripted player set up and played
    it to a surrender or to the time cap.
    """
    result = {
        "request": request["name"],
        "mod": request["mod"],
        "station": entry["station"],
        "tested": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loaded": stages["loaded"],
        "started": stages["started"],
        "started_as": entry["started_as"],
        "plan": entry.get("plan"),
        "outcome": entry.get("winner"),
        "seconds": entry.get("seconds"),
        "declarer": entry.get("declarer"),
        "error": entry.get("error") or entry.get("reason"),
        "planner_errors": entry.get("planner_errors", []),
        # Map errors the game logged; compare with the plain arena's (its count is in the
        # main arena's first game of each run, `map_errors` in the results).
        "map_errors": stages.get("map_errors"),
        "recording": str(Path(root).resolve()),
        "screenshots": {k: str(Path(v).resolve()) for k, v in stages["shots"].items()},
    }
    played = entry.get("winner") in ("BLU", "RED", "timeout") and not entry.get("reason")
    result["accepted"] = bool(stages["loaded"] and stages["started"] and played)
    return result


def log_end(desk):
    """Where the game log ends now, as an offset for desk.game_log. The worker reads at
    most 1 MiB a request, so a session's log past that takes several: until 2026-09-24 a
    game loaded in-game read from the first 1 MiB, found the last game's surrender, and
    ended after 5 s, as did every load after it until a launch (seven in a row once)."""
    offset = 0
    while True:
        _, after = desk.game_log(offset)
        if after <= offset:
            return offset
        offset = after


def run_station(station, out_root, rules, templates, settings, end):
    from .scripted import best_plan, choose_plan

    results = []
    scripted = settings.get("player") == "scripted"
    queue = settings.get("queue")
    rng = random.Random()
    rotation, baseline = 0, False
    # The arena of the game the last game left running, when it ended cleanly, and how
    # many games have been loaded inside it since HOI4 was launched.
    running, loads = None, 0
    path = out_root / f"results-{station.name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    # A game needs about 3 minutes to launch and most end within 10; do not start one
    # that cannot plausibly finish.
    while time.monotonic() + 12 * 60 < end:
        # A drain file ends the run between games, so that a restart loses none.
        if (out_root / "DRAIN").exists():
            say(station.name, "draining: no more games")
            break
        # A live evaluation that reserved the second PC comes before anything else.
        if station.peer and settings.get("eval"):
            reservation = take_reservation(settings["eval"])
            if reservation:
                lend(station, settings["eval"], reservation)
                continue
        # An arena someone asked to have tested comes next, played to the best plan, but
        # never two in a row: the station's own games come first (2026-09-24).
        request = None
        serves = queue and scripted and station.name in settings["queue_stations"]
        if serves and not (results and results[-1].get("request")):
            request = take_request(queue, station.name)
        if request:
            mod = request["mod"]
        else:
            # Read afresh each game: another recorder may have accepted an arena since.
            accepted = latest_versions(accepted_arenas(queue)) if queue else []
            if settings.get("main_only"):
                accepted = []
            accepted = [m for m in accepted if m not in settings["mods"]]
            mod = game_arena(settings["mods"], rotation, accepted)
            rotation += 1
        kind = "scripted" if scripted else "ai"
        name = time.strftime(f"{kind}-{station.name}-%Y%m%d-%H%M%S")
        country, speed = game_plan(station.name, len(results), settings["speeds"])
        if request:
            # Tests come every other game, which game_plan's turns would always give the
            # same side: all of the first eight were played as Red (2026-09-24).
            country = rng.choice(("BLU", "RED"))
        entry = {"game": name, "station": station.name, "started_as": country, "speed": speed}
        entry["arena"] = Path(mod).name
        # A fair coin for who declares, independent of the side played (start_game).
        entry["declare_drawn"] = rng.choice(("BLU", "RED"))
        player = None
        if scripted:
            plan = best_plan(rng) if request else choose_plan(rng)
            player = {"plan": plan, "templates": settings["buttons"], "rules": rules}
            entry["plan"] = player["plan"]
        if request:
            entry["request"] = request["name"]
            player["shots"] = True
            say(station.name, "testing the arena", request["name"], mod)
        # Straight into a start save where there is one for this arena and side, and a
        # few opening hours of a different length each game (run_briefly).
        saves = {**known_saves(settings["save_registry"]), **settings["saves"]}
        save = saves.get((entry["arena"], country))
        if save in settings["broken_saves"]:
            save = None
        entry["start_save"] = save
        # A game through the menus saves its start for the next ones (a scripted game;
        # an observed one gives both countries to the AI straight after).
        save_as = None
        if not save and scripted and settings["save_registry"]:
            save_as = save_name(entry["arena"], country)
        opening = round(rng.uniform(*settings["opening"]), 2) if settings["opening"] else 0.0
        entry["opening"] = opening
        stages = {"loaded": False, "started": False, "shots": {}}
        failure_shot = out_root / f"{name}-start-failed.png"
        try:
            deploy_mod(station, mod, settings)
            reused = False
            if running == entry["arena"] and can_load(save) and loads < LOADS_PER_LAUNCH:
                check_memory(station)  # With the running game's own memory in it.
                try:
                    with station.connect() as desk:
                        focus(desk)
                        load_in_game(desk, save, templates, rules, failure_shot)
                    reused, loads = True, loads + 1
                except Exception as error:  # noqa: BLE001 - launched afresh instead.
                    say(station.name, "loading in the game failed:", error, "- launching")
            running = None
            if not reused:
                station.quit()
                check_memory(station, GAME_MB)  # What is left, and what a game takes.
                station.launch(mod, save=save)
                loads = 0
            entry["loaded_in_game"] = reused
            stages["loaded"] = True
            with station.connect() as desk:
                if not focus(desk):
                    raise RuntimeError("could not bring the game window to the front")
                # Where this game's log lines begin, before its declaration is logged.
                log_from = log_end(desk) if reused else None
                try:
                    start_game(
                        desk, rules, failure_shot, country, speed,
                        observe=not scripted, declarer=entry["declare_drawn"],
                        saved=bool(save), opening=opening, save_as=save_as,
                    )  # fmt: skip
                    if save_as:
                        remember_save(settings["save_registry"], entry["arena"], country, save_as)
                        entry["saved_start"] = save_as
                except Exception:
                    if request:
                        shot = out_root / f"{name}-start-failed-full.png"
                        Image.fromarray(screen(desk)).save(shot)
                        stages["shots"]["start"] = shot
                    raise
                stages["started"] = True
                if request:
                    root = out_root / f"{name}-closeups"
                    root.mkdir(parents=True, exist_ok=True)
                    for k, path in enumerate(closeups(desk, root), 1):
                        stages["shots"][f"closeup_{k}"] = path
                say(station.name, "recording", name, "as", country, "at speed", speed)
                outcome, reason, manifest = play(
                    desk,
                    out_root / name,
                    Popups(templates),
                    {**settings, "mod": mod},
                    station.name,
                    country,
                    speed,
                    player,
                    start_save=save,
                    log_from=log_from,
                )
        except MemoryStop as stop:
            say(station.name, "memory:", stop, "- no more games")
            (out_root / "MEMORY-STOP").write_text(f"{time.strftime('%H:%M:%S')} {stop}" + chr(10))
            break
        except Exception as error:  # noqa: BLE001 - reported, then the next game is tried.
            say(station.name, "start failed:", error)
            entry["error"] = f"{type(error).__name__}: {error}"
            if save and not stages["started"]:
                # From now on this side starts from the menus.
                settings["broken_saves"].add(save)
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
            if not reason:
                running = entry["arena"]
            entry.update(
                winner=outcome,
                seconds=manifest["seconds"],
                frames=manifest["frames"],
                declarer=manifest["declarer"],
                players=manifest["players"],
                complete=manifest["complete"],
                reason=reason,
                planner_errors=manifest.get("planner_errors", []),
            )
            stages["shots"].update(manifest.get("screenshots", {}))
        # The map errors logged, before the next launch rewrites the log: after an arena
        # test, and once a run on the main arena, to compare with.
        main = entry["arena"] == Path(settings["mods"][0]).name
        if stages["started"] and (request or (main and not baseline)):
            stages["map_errors"] = entry["map_errors"] = map_errors(station)
            baseline = baseline or main
        if request:
            result = test_result(request, entry, out_root / name, stages)
            answer(queue, request["name"], result)
            request["claimed"].unlink(missing_ok=True)
            if result["accepted"]:
                accept_arena(queue, mod)
            say(station.name, "arena", request["name"], "accepted:", result["accepted"])
        results.append(entry)
        # One file per run: runs sharing a folder had each rewritten the day's file.
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
    ok_button=("artifacts/screens-1080p/ok-button.png", EVENT_OK),
    hz=5,
    codec="x264",
    cap_minutes=45,
    speeds=(4, 5),
    peer=None,
    peer_only=False,
    player="observe",
    arena_queue=None,
    queue_stations=None,
    eval_dir=None,
    start_saves=None,
    opening=None,
    main_only=False,
):
    """Record on this PC, the second PC, or both at once, until `minutes` run out.

    `player` "observe" hands both countries to the game's AI; "scripted" has the scripted
    player (scripted.py) fight the recorder's country against the AI. `mod` is one arena
    or several, played in turn.

    With `arena_queue`, a scripted station first plays any arena test request waiting
    there (take_request), once, and answers it in the queue's sibling `results` folder.
    Arenas that pass are listed in its `accepted.json` and join the turn. Only the
    stations named in `queue_stations` serve the queue (default: all of them).

    With `eval_dir`, the second PC is lent out between games to whoever reserves it in
    `<eval_dir>/queue` (take_reservation, lend).

    `start_saves` ("ARENA:COUNTRY:SAVE") launch a game straight into a save made paused at
    the start of a new game, skipping the menus. `opening` is a range of seconds the game
    runs at speed 1 before the war begins (run_briefly).
    """
    from .scripted import TEMPLATES, load_templates

    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)
    screen_rules = ScreenRules(rules)
    templates = [np.asarray(Image.open(path).convert("RGB")) for path in ok_button]
    if not speeds or any(speed not in (4, 5) for speed in speeds):
        raise ValueError("speeds are 4 or 5")
    if player not in ("observe", "scripted"):
        raise ValueError("player is observe or scripted")
    mods = [mod] if isinstance(mod, (str, Path)) else list(mod)
    mods = [str(Path(m).resolve()) for m in mods]
    settings = {
        "mods": mods,
        "deployed": {},
        "queue": arena_queue,
        "eval": eval_dir,
        "saves": parse_saves(start_saves or []),
        "broken_saves": set(),
        # Start saves made by games that came through the menus, beside the arena queue.
        "save_registry": arena_queue and Path(arena_queue).parent / "saves-peer.json",
        "opening": tuple(opening) if opening else None,
        "main_only": main_only,
        "queue_stations": list(queue_stations or ("here", "peer")),
        "hz": hz,
        "codec": codec,
        "cap_minutes": cap_minutes,
        "speeds": list(speeds),
        "player": player,
        "buttons": load_templates(TEMPLATES) if player == "scripted" else None,
    }
    stations = [] if peer_only else [Station("here")]
    if peer:
        stations.append(Station("peer", peer))
    if not stations:
        raise ValueError("--peer-only needs --peer")
    for station in stations:
        for arena in mods:
            deploy_mod(station, arena, settings)
    # Requests claimed by a recorder that was stopped before answering them.
    for queue in (arena_queue, eval_dir and Path(eval_dir) / "queue"):
        if queue:
            for name in reoffer(queue):
                say("recorder", "offered again:", name)
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
