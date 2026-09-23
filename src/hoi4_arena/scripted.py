"""A scripted player: fights one side of an arena game through the real interface.

In an AI game the game's AI fights the war, and the only inputs recorded are the
camera's and the popup clicks: they teach camera control and clearing popups, and they
never decide who wins. This player fights the war itself, the way a player does and
with the same mouse and keys. It forms its divisions into an army, draws a front line
along the border and an offensive into enemy land, activates the plan, and sometimes
redraws the offensive later, while the camera watches the front and popups are cleared.
Its choices vary at random from game to game: where it attacks, how deep, how long it
lets the plan prepare, whether it attacks at all, and whether it redraws. So its games
are labelled recordings of real orders whose outcome depends on them, in any number.
That is what imitation, a win predictor and offline reinforcement learning need. Its
win rate against the game's AI is also the first baseline a learned agent must beat.

Each order is also logged in structured form (`orders` in the manifest): what it was,
the frame it was given at, and for an offensive the enemy state it aims at. With the
arena's daily state reports, that is the data for a model of the war's numbers.

Calibrated live at 1920x1080 on 2026-09-23:
- The top bar's "Unassigned divisions" alert, the red fists. A shift+click selects all
  of them. The alert row differs between PCs, so the alert is found by template.
- With divisions selected, a green + in the army bar at the bottom creates an army,
  and the new army is selected.
- Z is the Front Line tool. A click on the border sets the front along all of it.
- X is the Offensive Line tool. A right-drag from the front into enemy land draws the
  offensive, which the game routes along its own path.
- The green arrow above the army card activates the plan. Once active, it changes.
- The Battle Plans bar shows while an army is selected.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np

from .ai_games import MAP_BOTTOM, MAP_TOP, act, recentre, run_at, screen, tap
from .vision import country_pixels

# The least normalised correlation (TM_CCOEFF_NORMED, 1 an exact copy) at which a button
# counts as found. Correlation shrugs off a button lit under the pointer, where squared
# difference did not: on the calibration screens and the first live game each button
# scored 1.00 where it was (0.92 lit, 0.88 hovered) and at most 0.55 anywhere else.
FOUND = {"unassigned": 0.85, "activate": 0.8, "plans_bar": 0.8}
# The create-army + glows while divisions are selected, so no fixed picture of it holds:
# the first live game's frames scored 0.17 against a template taken a minute earlier.
# It is found by colour instead, the only green in the army bar before an army exists:
# 201 to 336 pixels of it where it showed, none while it was grey.
GREEN_PLUS = 60
FRONT_LINE, OFFENSIVE_LINE, SHIFT = 0x5A, 0x58, 0x10
# The first army's card in the army bar at the bottom, 1080p fractions.
ARMY_CARD = (947 / 1920, 1010 / 1080)
# The army bar, where the create-army + shows: the bottom tenth of the screen.
ARMY_BAR_TOP = 0.88
# The arena's land grid: 24 province columns (12 a side) by 8 rows, and states of 3 by 4
# provinces, 8 a side. Blue's ids run 1 to 8 and Red's 9 to 16 (mapgen.state_cell).
COLUMNS, ROWS, STATE_WIDTH, STATE_HEIGHT, STATE_ROWS = 24, 8, 3, 4, 2
ENEMY = {"BLU": "RED", "RED": "BLU"}
ATTACKS = {"near": 0.45, "deep": 0.4, "none": 0.15}


def choose_plan(rng):
    """One game's strategy, drawn at random.

    `attack` is where the offensive goes: "near", into the enemy's border states;
    "deep", into its rear; or "none", a front line only, held by the game's own general.
    `wait` is how long the plan prepares before it is activated: preparation raises the
    plan's bonus, while the enemy may strike first. `redraw` is how often a new
    offensive is drawn toward a new target, or None for never.
    """
    attack = rng.choices(list(ATTACKS), weights=list(ATTACKS.values()))[0]
    return {
        "attack": attack,
        "wait": 0 if rng.random() < 0.4 else round(rng.uniform(5, 60)),
        "redraw": None if attack == "none" or rng.random() < 0.5 else round(rng.uniform(40, 120)),
    }


def state_at(u, v):
    """The arena state under a point at (u, v), fractions of the arena's land box.

    The box is both countries' land fully zoomed out, Blue's on the left. Red's half is
    Blue's turned half a turn, so its columns count from the east and its rows from the
    bottom. Provinces are hexagons, so a point near a state's edge can be off by one.
    """
    column = min(COLUMNS - 1, max(0, int(u * COLUMNS)))
    row = min(ROWS - 1, max(0, int(v * ROWS)))
    first = 1
    if column >= COLUMNS // 2:
        per_side = (COLUMNS // 2 // STATE_WIDTH) * STATE_ROWS
        column, row, first = COLUMNS - 1 - column, ROWS - 1 - row, 1 + per_side
    return first + (column // STATE_WIDTH) * STATE_ROWS + row // STATE_HEIGHT


def land_box(blue, red):
    """The land's bounding box in the map crop (top, left, bottom, right), or None."""
    ys, xs = np.nonzero(blue | red)
    if not len(ys):
        return None
    return ys.min(), xs.min(), ys.max() + 1, xs.max() + 1


class Planner:
    """The scripted player's orders, given between the camera's moves.

    `setup` runs while the game is still paused, as a player sets up at the start, and
    then unpauses at `speed`; `due` and `step` carry out the rest of the plan as the
    game runs. `frame` returns the number of frames recorded so far, to stamp orders.
    """

    def __init__(self, country, plan, templates, rules, speed, frame, rng=None):
        self.country, self.enemy = country, ENEMY[country]
        self.plan, self.templates, self.rules, self.speed = plan, templates, rules, speed
        self.frame = frame
        self.rng = rng or random.Random()
        self.orders = []
        self.activate_at = self.redraw_at = math.inf
        self.active = self.running = False
        # A setup that failed, for play() to end the game with.
        self.error = None

    def order(self, kind, **details):
        self.orders.append({"frame": self.frame(), "order": kind, **details})

    def find(self, rgb, name, top=0.0):
        """Where a calibrated button is, as screen fractions, or None."""
        import cv2

        cut = int(top * rgb.shape[0])
        template = self.templates[name]
        scores = cv2.matchTemplate(rgb[cut:], template, cv2.TM_CCOEFF_NORMED)
        _, best, _, (x, y) = cv2.minMaxLoc(scores)
        if best < FOUND[name]:
            return None
        h, w = template.shape[:2]
        return (x + w / 2) / rgb.shape[1], (y + h / 2 + cut) / rgb.shape[0]

    def click(self, desk, at, button=0, shift=False):
        press = [{"kind": "button", "button": button, "down": d} for d in (True, False)]
        events = [{"kind": "move", "x": at[0], "y": at[1]}, *press]
        if shift:
            events = [{"kind": "key", "vk": SHIFT, "down": True}, *events]
            events.append({"kind": "key", "vk": SHIFT, "down": False})
        act(desk, events)

    def form_army(self, desk, tries=4):
        """Every unassigned division into one new army, which is left selected."""
        for _ in range(tries):
            # Off the top bar first: an alert under the pointer is drawn lit.
            act(desk, [{"kind": "move", "x": 0.5, "y": 0.5}])
            alert = self.find(screen(desk), "unassigned")
            if alert is None:
                time.sleep(1)
                continue
            self.click(desk, alert, shift=True)
            time.sleep(0.8)
            plus = green_plus(screen(desk))
            if plus is None:
                continue
            self.click(desk, plus)
            time.sleep(1.2)
            if self.selected(desk):
                self.order("army")
                return
        raise RuntimeError("could not form an army from the unassigned divisions")

    def selected(self, desk):
        return self.find(screen(desk), "plans_bar") is not None

    def select_army(self, desk):
        if not self.selected(desk):
            self.click(desk, ARMY_CARD)
            time.sleep(0.8)
        return self.selected(desk)

    def overview(self, desk):
        """Zoomed fully out over the arena: the land masks and their box, or Nones."""
        recentre(desk)
        rgb = screen(desk)
        top, bottom = MAP_TOP, rgb.shape[0] - MAP_BOTTOM
        blue, red = country_pixels(rgb[top:bottom])
        if blue is None or not blue.any() or not red.any():
            return rgb, None, None, None
        return rgb, blue, red, land_box(blue, red)

    def draw_front(self, desk):
        rgb, blue, red, box = self.overview(desk)
        if box is None or not self.select_army(desk):
            raise RuntimeError("no arena or army to draw a front line with")
        front = self.front(blue, red)
        if not front:
            raise RuntimeError("the two countries do not touch on screen")
        # The border's middle: the front line tool follows the whole border from there.
        x, y = sorted(front, key=lambda p: p[1])[len(front) // 2]
        act(desk, tap(FRONT_LINE))
        self.click(desk, self.screen_point(rgb, x, y))
        time.sleep(0.8)
        self.order("front", at=self.box_point(box, x, y))

    def front(self, blue, red):
        """Crop pixels where the two countries meet, as (x, y).

        Not within a twentieth of the land's height of its top or bottom edge: the first
        live game's offensive started from a stray "front" pixel on the top coast.
        """
        import cv2

        near_blue = cv2.dilate(blue.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ys, xs = np.nonzero(near_blue & red)
        box = land_box(blue, red)
        if box is not None:
            margin = (box[2] - box[0]) / 20
            keep = (ys > box[0] + margin) & (ys < box[2] - margin)
            ys, xs = ys[keep], xs[keep]
        return list(zip(xs.tolist(), ys.tolist(), strict=True))

    def screen_point(self, rgb, x, y):
        return x / rgb.shape[1], (y + MAP_TOP) / rgb.shape[0]

    def box_point(self, box, x, y):
        top, left, bottom, right = box
        return [round((x - left) / (right - left), 4), round((y - top) / (bottom - top), 4)]

    def draw_offensive(self, desk):
        """An offensive from the front toward an enemy state, near or deep."""
        rgb, blue, red, box = self.overview(desk)
        if box is None or not self.select_army(desk):
            raise RuntimeError("no arena or army to draw an offensive with")
        front = self.front(blue, red)
        enemy = red if self.enemy == "RED" else blue
        ys, xs = np.nonzero(enemy)
        if not front or not len(xs):
            raise RuntimeError("no front or enemy land on screen")
        # How far each enemy pixel lies from the seam, as a fraction of the enemy's width:
        # near targets sit in the first third of its land, deep ones in the last.
        top, left, bottom, right = box
        seam = (left + right) / 2
        depth = np.abs(xs - seam) / max(1.0, (right - left) / 2)
        pick = depth < 1 / 3 if self.plan["attack"] == "near" else depth > 2 / 3
        if not pick.any():
            pick = np.ones_like(depth, dtype=bool)
        k = self.rng.choice(np.flatnonzero(pick).tolist())
        target = xs[k], ys[k]
        # From just across the front, level with the target, so the line lies in enemy
        # land: one of the front's pixels nearest the target's row, a step into the enemy.
        near = sorted(front, key=lambda p: abs(p[1] - target[1]))[:20]
        x, y = self.rng.choice(near)
        step = (right - left) / 50 * (1 if self.enemy == "RED" else -1)
        start = x + step, y
        act(desk, tap(OFFENSIVE_LINE))
        a, b = self.screen_point(rgb, *start), self.screen_point(rgb, *target)
        steps = [
            {"kind": "move", "x": a[0] + (b[0] - a[0]) * i / 8, "y": a[1] + (b[1] - a[1]) * i / 8}
            for i in range(1, 9)
        ]
        press = {"kind": "button", "button": 1}
        act(
            desk,
            [{"kind": "move", "x": a[0], "y": a[1]}, {**press, "down": True}, *steps,
             {**press, "down": False}],
            pause=0.08,
        )  # fmt: skip
        time.sleep(0.8)
        u, v = self.box_point(box, *target)
        self.order(
            "offensive",
            attack=self.plan["attack"],
            start=self.box_point(box, *start),
            target=[u, v],
            target_state=state_at(u, v),
        )

    def activate(self, desk):
        if not self.select_army(desk):
            return False
        button = self.find(screen(desk), "activate", top=0.8)
        if button is None:
            return False
        self.click(desk, button)
        time.sleep(0.8)
        self.active = True
        self.order("activate")
        return True

    def setup(self, desk):
        """While paused: the army, its front, its offensive; then run the game."""
        self.form_army(desk)
        self.draw_front(desk)
        if self.plan["attack"] != "none":
            self.draw_offensive(desk)
        run_at(desk, self.rules, self.speed)
        self.running = True
        self.order("run", speed=self.speed)
        now = time.monotonic()
        if self.plan["attack"] != "none":
            self.activate_at = now + self.plan["wait"]
        if self.plan["redraw"]:
            self.redraw_at = now + self.plan["redraw"]

    def due(self):
        now = time.monotonic()
        return now >= self.activate_at or now >= self.redraw_at

    def step(self, desk):
        """The next due order, once the game runs. True if the camera was moved."""
        now = time.monotonic()
        if now >= self.activate_at:
            self.activate_at = math.inf if self.activate(desk) else now + 5
            return False
        if now >= self.redraw_at:
            self.draw_offensive(desk)
            if self.active:
                self.activate(desk)
            self.redraw_at = now + self.plan["redraw"]
            return True
        return False


def green_plus(rgb):
    """The centre of the green + in the army bar, as screen fractions, or None."""
    top = int(ARMY_BAR_TOP * rgb.shape[0])
    bar = rgb[top:].astype(np.int32)
    r, g, b = bar[..., 0], bar[..., 1], bar[..., 2]
    ys, xs = np.nonzero((g - np.maximum(r, b) > 25) & (g > 90))
    if len(xs) < GREEN_PLUS:
        return None
    return float(np.median(xs)) / rgb.shape[1], (float(np.median(ys)) + top) / rgb.shape[0]


def load_templates(paths):
    from PIL import Image

    return {name: np.asarray(Image.open(path).convert("RGB")) for name, path in paths.items()}


TEMPLATES = {
    "unassigned": "artifacts/screens-1080p/unassigned-divisions.png",
    "activate": "artifacts/screens-1080p/activate-plan.png",
    "plans_bar": "artifacts/screens-1080p/battle-plans-bar.png",
}


def wilson(wins, games, z=1.96):
    """A 95% interval for a win rate (Wilson), as (low, high); (0, 1) with no games."""
    if not games:
        return 0.0, 1.0
    p = wins / games
    centre = (p + z * z / (2 * games)) / (1 + z * z / games)
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / (1 + z * z / games)
    return max(0.0, centre - half), min(1.0, centre + half)


def win_rate(results):
    """The scripted player's record from `run_station` results: overall, by side, by attack.

    Games that ended without a surrender count as neither a win nor a loss, and are
    reported apart.
    """

    def tally(games):
        decided = [g for g in games if g.get("winner") in ("BLU", "RED")]
        wins = sum(g["winner"] == g["started_as"] for g in decided)
        low, high = wilson(wins, len(decided))
        return {
            "games": len(games),
            "decided": len(decided),
            "wins": wins,
            "rate": round(wins / len(decided), 3) if decided else None,
            "interval95": [round(low, 3), round(high, 3)],
        }

    played = [g for g in results if "error" not in g]
    report = {"all": tally(played), "errors": len(results) - len(played)}
    for side in ("BLU", "RED"):
        report[side] = tally([g for g in played if g["started_as"] == side])
    for attack in ATTACKS:
        report[attack] = tally([g for g in played if (g.get("plan") or {}).get("attack") == attack])
    return report
