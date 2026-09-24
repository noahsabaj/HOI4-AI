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
- Q opens the political screen. Its first law slot is conscription: a click lists the
  laws, a click on one asks "Replace Idea?", and OK changes it for political power (150
  a step up the list; a country starts with 2 and gains about 2 a day). With too little
  power OK does nothing and the question stays. After OK the list closes by itself. The
  AI changes the law once it can afford to: in the first v4 games its manpower pool
  began refilling in late May and its deployed manpower rose from 15k to 16-21k, while
  the script's, which never changed the law, only fell. The starting law is read from
  the slot's own icon, since the open list shows every law's icon.
- A new army has no commander ("No Commander" in its panel). A click on the panel's
  portrait lists the country's commanders, and a click on one assigns them. Each arena
  country has three generals and a field marshal, all skill 3; the AI puts one in charge
  of its army, and the script's first 12 games fought without.
- The Battle Plans bar's Delete Order button (the bin at its right end): a right-click
  asks "Delete all orders from this Army?", on the same dialog as a law change. Without
  it every redraw added an offensive to the ones before, and the army was split between
  them: in the first game with conscription the AI took the map's top and bottom rows
  while the script's divisions stood in the middle.
- An offensive line spreads the front's divisions along it. One drawn across the whole
  front, parallel to it, moves the front forward as one ("broad"); one drawn from the
  front toward a single state draws the divisions toward that state.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np

from .ai_games import LOOK, MAP_BOTTOM, MAP_TOP, act, recentre, run_at, screen, tap
from .vision import country_pixels

# The least normalised correlation (TM_CCOEFF_NORMED, 1 an exact copy) at which a button
# counts as found. Correlation shrugs off a button lit under the pointer, where squared
# difference did not: on the calibration screens and the first live game each button
# scored 1.00 where it was (0.92 lit, 0.88 hovered) and at most 0.55 anywhere else.
FOUND = {
    "unassigned": 0.85,
    "activate": 0.8,
    "plans_bar": 0.8,
    "confirm_ok": 0.9,
    "political_title": 0.8,
    # 1.00 where each showed on the calibration screens; at most 0.48 ("No Commander")
    # and 0.64 (the bin, under a tooltip) anywhere else.
    "no_commander": 0.8,
    "trash": 0.8,
    # The open law list's title: 1.00 open, at most 0.51 on other screens.
    "law_list": 0.8,
}
# The create-army + glows while divisions are selected, so no fixed picture of it holds:
# the first live game's frames scored 0.17 against a template taken a minute earlier.
# It is found by colour instead, the only green in the army bar before an army exists:
# 201 to 336 pixels of it where it showed, none while it was grey.
GREEN_PLUS = 60
FRONT_LINE, OFFENSIVE_LINE, SHIFT, POLITICS = 0x5A, 0x58, 0x10, 0x51
# The political screen at 1080p: the conscription slot (its icon's box), each law's row
# in the list it opens, the confirmation's OK and Cancel, and the list's close button.
LAW_SLOT = (38, 571, 82, 615)
CONFIRM, CANCEL, CLOSE_LIST = (1054, 677), (884, 677), (1005, 100)
# The conscription laws up the list from the start's Volunteer Only, one row (74 px) each;
# every step costs 150 political power. Scraping the Barrel, the last, was not on offer.
LAWS = ["volunteer", "limited", "extensive", "service", "all_adults"]
LAW_ROWS = {law: (703, 322 + 74 * i) for i, law in enumerate(LAWS[1:])}
# Divisions start at 31% strength and fill from the manpower pool, which the law sets.
# Extensive alone left them at 81% after five years of a held front, with 2,000
# political power unspent (2026-09-23), so most games go further.
CONSCRIPTION = {"limited": 0.1, "extensive": 0.2, "service": 0.3, "all_adults": 0.4}
# Tries at one law step before giving up on it (every 10 s).
LAW_TRIES = 30
# The army panel's commander portrait, and the first commander in the list it opens.
COMMANDER_SLOT, FIRST_COMMANDER = (30, 140), (950, 352)
# How far a broad offensive goes: this share of the way from the front to the enemy's
# far edge.
BROAD_DEPTH = 1 / 3
# Morphological opening of the land masks, in pixels. The map draws thin lines that
# are classed as land: the glow along Red's outer coast is blue, an offensive's arrow
# inside Blue is red. Without it the first games took them for the front, and drew
# offensives from Red's far coast or from deep in Blue.
CLEAN = 5
# The first army's card in the army bar at the bottom, 1080p fractions.
ARMY_CARD = (947 / 1920, 1010 / 1080)
# The army bar, where the create-army + shows: the bottom tenth of the screen.
ARMY_BAR_TOP = 0.88
# The arena's land grid: 24 province columns (12 a side) by 8 rows, and states of 3 by 4
# provinces, 8 a side. Blue's ids run 1 to 8 and Red's 9 to 16 (mapgen.state_cell).
COLUMNS, ROWS, STATE_WIDTH, STATE_HEIGHT, STATE_ROWS = 24, 8, 3, 4, 2
ENEMY = {"BLU": "RED", "RED": "BLU"}
# Arrows toward one state lost the front's flanks or its rear in every game with them
# (2026-09-23), so most games attack broad. None holds only: a held front with a general
# and Extensive conscription stood unbroken for five years, a draw.
ATTACKS = {"broad": 0.7, "near": 0.15, "deep": 0.15}


def choose_plan(rng):
    """One game's strategy, drawn at random.

    `attack` is where the offensive goes: "broad", the whole front forward by a third of
    the enemy's land; "near", toward one of the enemy's border states; or "deep", toward
    its rear. `wait` is how long the front holds, with the offensive drawn, before the
    plan is executed: preparation raises the plan's bonus, and a longer hold lets the
    divisions fill up, while the enemy may strike first. `redraw` is how often the plan
    is drawn afresh (every order deleted, then a new front and offensive), or None.
    """
    attack = rng.choices(list(ATTACKS), weights=list(ATTACKS.values()))[0]
    return {
        # How far up the conscription laws to go as political power allows.
        "conscription": rng.choices(list(CONSCRIPTION), weights=list(CONSCRIPTION.values()))[0],
        "attack": attack,
        # Planning reaches its full 30% bonus in 15 days, about 6 s at speed 5; some games
        # hold for up to a game year and a half first.
        "wait": round(rng.uniform(6, 60) if rng.random() < 0.6 else rng.uniform(60, 240)),
        # A broad offensive stops at its line, so it is always drawn again, further on.
        "redraw": (
            round(rng.uniform(30, 90))
            if attack == "broad"
            else None
            if rng.random() < 0.5
            else round(rng.uniform(40, 120))
        ),
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


def clean(mask):
    """A land mask without the lines thinner than CLEAN pixels that the map draws."""
    import cv2

    kernel = np.ones((CLEAN, CLEAN), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)


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
        self.activate_at = self.redraw_at = self.law_at = math.inf
        self.active = self.running = False
        # Attempts at activating the current plan.
        self.tries = 0
        # The conscription law in force, as an index into LAWS, and failed tries at the
        # next step.
        self.law_step = self.law_fails = 0
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
        """Move onto `at`, look (ai_games.LOOK), then press there."""
        press = [{"kind": "button", "button": button, "down": d} for d in (True, False)]
        act(desk, [{"kind": "move", "x": at[0], "y": at[1]}])
        time.sleep(self.rng.uniform(*LOOK))
        if shift:
            press = [{"kind": "key", "vk": SHIFT, "down": True}, *press]
            press.append({"kind": "key", "vk": SHIFT, "down": False})
        act(desk, press)

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
        if blue is None:
            return rgb, None, None, None
        blue, red = clean(blue), clean(red)
        if not blue.any() or not red.any():
            return rgb, None, None, None
        return rgb, blue, red, land_box(blue, red)

    def assign_general(self, desk, tries=3):
        """A commander for the army, as the AI gives its own. True once it has one."""
        for _ in range(tries):
            if not self.select_army(desk):
                continue
            if self.find(screen(desk), "no_commander") is None:
                self.order("general")
                return True
            self.click(desk, pixels(*COMMANDER_SLOT))
            time.sleep(0.8)
            self.click(desk, pixels(*FIRST_COMMANDER))
            time.sleep(0.8)
        return False

    def clear_orders(self, desk):
        """Every order of the army deleted: a right-click on Delete Order, then OK."""
        if not self.select_army(desk):
            return False
        # Off the bar first, so that no tooltip covers the bin.
        act(desk, [{"kind": "move", "x": 0.5, "y": 0.5}])
        trash = self.find(screen(desk), "trash", top=0.7)
        if trash is None:
            return False
        self.click(desk, trash, button=1)
        time.sleep(0.8)
        if self.find(screen(desk), "confirm_ok") is None:
            return False
        self.click(desk, pixels(*CONFIRM))
        time.sleep(0.8)
        self.order("clear")
        return True

    def draw_front(self, desk, tries=4):
        """A front line along the whole border, checked: the plan's activate arrow shows.

        The front line tool takes a click on the enemy's side of the border. On 2026-09-23
        two games in three as Red drew no front and so had no plan to activate: the click
        had gone to the player's own side, where the tool does nothing.
        """
        rgb, blue, red, box = self.overview(desk)
        if box is None or not self.select_army(desk):
            raise RuntimeError("no arena or army to draw a front line with")
        front = self.front(blue, red)
        if not front:
            raise RuntimeError("the two countries do not touch on screen")
        # The border's middle first: the tool follows the whole border from there.
        middle = sorted(front, key=lambda p: p[1])[len(front) // 2]
        for attempt in range(tries):
            x, y = middle if attempt == 0 else self.rng.choice(front)
            act(desk, tap(FRONT_LINE))
            self.click(desk, self.screen_point(rgb, x, y))
            time.sleep(0.8)
            if self.find(screen(desk), "activate", top=0.8) is not None:
                self.order("front", at=self.box_point(box, x, y), tries=attempt + 1)
                return
            self.select_army(desk)
        raise RuntimeError("no front line took: the plan's activate arrow never showed")

    def front(self, blue, red):
        """Crop pixels on the enemy's side of the border, as (x, y).

        The enemy's land next to the player's: where the front line tool takes its click,
        and where an offensive starts. Not within a twentieth of the land's height of its
        top or bottom edge: the first live game's offensive started from a stray "front"
        pixel on the top coast.
        """
        import cv2

        own, enemy = (blue, red) if self.country == "BLU" else (red, blue)
        near_own = cv2.dilate(own.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ys, xs = np.nonzero(near_own & enemy)
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

    def drag(self, desk, points, steps):
        """The Offensive Line tool, then a right-drag through `points` (screen fractions),
        `steps` moves to each."""
        moves = [
            {"kind": "move", "x": x0 + (x1 - x0) * i / steps, "y": y0 + (y1 - y0) * i / steps}
            for (x0, y0), (x1, y1) in zip(points, points[1:])
            for i in range(1, steps + 1)
        ]
        press = {"kind": "button", "button": 1}
        act(desk, tap(OFFENSIVE_LINE))
        act(desk, [{"kind": "move", "x": points[0][0], "y": points[0][1]}])
        time.sleep(self.rng.uniform(*LOOK))
        act(desk, [{**press, "down": True}, *moves, {**press, "down": False}], pause=0.08)
        time.sleep(0.8)

    def broad_line(self, front, box, depth=BROAD_DEPTH):
        """Points across the whole front, `depth` of the way on to the enemy's far edge.

        One for each band of rows from the land's top to its bottom: the front's pixel
        furthest into the enemy within the band, moved on into enemy land. So the line
        follows the front's bends, and the whole front moves forward to it.
        """
        top, left, bottom, right = box
        sign = 1 if self.enemy == "RED" else -1
        far = right if sign > 0 else left
        xs = np.array([p[0] for p in front])
        ys = np.array([p[1] for p in front])
        height = bottom - top
        points = []
        for y in np.linspace(top + height / 20, bottom - height / 20, 9):
            band = np.abs(ys - y) < height / 16
            if band.any():
                x = xs[band].max() if sign > 0 else xs[band].min()
                points.append((float(x + sign * depth * abs(far - x)), float(y)))
        return points

    def draw_offensive(self, desk):
        """An offensive: the whole front forward ("broad"), or toward one enemy state."""
        rgb, blue, red, box = self.overview(desk)
        if box is None or not self.select_army(desk):
            raise RuntimeError("no arena or army to draw an offensive with")
        front = self.front(blue, red)
        enemy = red if self.enemy == "RED" else blue
        ys, xs = np.nonzero(enemy)
        if not front or not len(xs):
            raise RuntimeError("no front or enemy land on screen")
        if self.plan["attack"] == "broad":
            line = self.broad_line(front, box)
            if len(line) < 2:
                raise RuntimeError("no front to draw a broad offensive along")
            self.drag(desk, [self.screen_point(rgb, *p) for p in line], steps=3)
            points = [self.box_point(box, *p) for p in line]
            self.order(
                "offensive",
                attack="broad",
                line=points,
                target_states=sorted({state_at(u, v) for u, v in points}),
            )
            return
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
        self.drag(desk, [self.screen_point(rgb, *start), self.screen_point(rgb, *target)], steps=8)
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
        """While paused: the army, its general, its front, its offensive; then run."""
        self.form_army(desk)
        self.assign_general(desk)
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
        if self.plan.get("conscription") in LAWS[1:]:
            # About 150 political power after half a minute at speed 5.
            self.law_at = now + 30

    def due(self):
        now = time.monotonic()
        return min(self.activate_at, self.redraw_at, self.law_at) <= now

    def step(self, desk):
        """The next due order, once the game runs. True if the camera was moved."""
        now = time.monotonic()
        if now >= self.activate_at:
            self.tries += 1
            if self.activate(desk) or self.tries >= 6:
                self.activate_at, self.tries = math.inf, 0
            else:
                self.activate_at = now + 5
            return False
        if now >= self.redraw_at:
            # Soon again, should the redraw fail part way.
            self.redraw_at = now + 5
            self.clear_orders(desk)
            self.draw_front(desk)
            if self.plan["attack"] != "none":
                self.draw_offensive(desk)
            if self.active:
                # A new plan waits to be executed, like the first.
                self.activate_at = now
            self.redraw_at = now + self.plan["redraw"]
            return True
        if now >= self.law_at:
            done = self.raise_conscription(desk)
            self.law_at = math.inf if done else now + 10
        return False

    def law(self, rgb):
        """The conscription law the political screen's slot shows, or None."""
        import cv2

        x0, y0, x1, y1 = LAW_SLOT
        patch = rgb[y0:y1, x0:x1]
        for name in ("limited", "volunteer"):
            template = self.templates[f"law_{name}"]
            if cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED).max() > 0.9:
                return name
        return "other"

    def politics(self, desk, shown, tries=3):
        """The political screen opened or closed, checked. True once it is as asked.

        Q toggles it, so a press made without looking can leave it the wrong way round:
        in a live game on 2026-09-23 it stayed open from the second law change to the
        end, over the map the camera and the other orders work on.
        """
        for _ in range(tries):
            if (self.find(screen(desk), "political_title") is not None) == shown:
                return True
            act(desk, tap(POLITICS))
            time.sleep(0.8)
        return False

    def raise_conscription(self, desk):
        """One step up the conscription laws toward the plan's. True once it is there, or
        once a step has failed LAW_TRIES times running.

        The law in force is read from the slot's icon only at the start (Volunteer Only or
        Limited). After that a step counts as taken when its confirmation closes on OK,
        which it does only with the political power to pay. Every click is made only where
        the screen it is meant for shows: after OK the law list closes by itself, and a
        click on its close button then lands on the map.
        """
        goal = LAWS.index(self.plan["conscription"])
        if not self.politics(desk, True):
            return False
        if self.law_step == 0 and self.law(screen(desk)) == "limited":
            self.law_step = 1
        if self.law_step >= goal:
            self.politics(desk, False)
            return True
        target = LAWS[self.law_step + 1]
        self.click(desk, pixels(LAW_SLOT[0] + 22, LAW_SLOT[1] + 22))
        time.sleep(0.8)
        changed = False
        if self.find(screen(desk), "law_list") is not None:
            self.click(desk, pixels(*LAW_ROWS[target]))
            time.sleep(0.8)
            if self.find(screen(desk), "confirm_ok") is not None:
                self.click(desk, pixels(*CONFIRM))
                time.sleep(0.8)
                changed = self.find(screen(desk), "confirm_ok") is None
                if not changed:
                    self.click(desk, pixels(*CANCEL))  # Not enough political power yet.
                    time.sleep(0.5)
            if self.find(screen(desk), "law_list") is not None:
                self.click(desk, pixels(*CLOSE_LIST))
                time.sleep(0.5)
        self.politics(desk, False)
        if changed:
            self.law_step, self.law_fails = self.law_step + 1, 0
            self.order("law", law=target)
            return self.law_step >= goal
        self.law_fails += 1
        return self.law_fails >= LAW_TRIES


def green_plus(rgb):
    """The centre of the green + in the army bar, as screen fractions, or None."""
    top = int(ARMY_BAR_TOP * rgb.shape[0])
    bar = rgb[top:].astype(np.int32)
    r, g, b = bar[..., 0], bar[..., 1], bar[..., 2]
    ys, xs = np.nonzero((g - np.maximum(r, b) > 25) & (g > 90))
    if len(xs) < GREEN_PLUS:
        return None
    return float(np.median(xs)) / rgb.shape[1], (float(np.median(ys)) + top) / rgb.shape[0]


def pixels(x, y):
    """1080p pixels as screen fractions."""
    return x / 1920, y / 1080


def load_templates(paths):
    from PIL import Image

    return {name: np.asarray(Image.open(path).convert("RGB")) for name, path in paths.items()}


TEMPLATES = {
    "unassigned": "artifacts/screens-1080p/unassigned-divisions.png",
    "activate": "artifacts/screens-1080p/activate-plan.png",
    "plans_bar": "artifacts/screens-1080p/battle-plans-bar.png",
    "law_volunteer": "artifacts/screens-1080p/law-volunteer.png",
    "law_limited": "artifacts/screens-1080p/law-limited.png",
    "confirm_ok": "artifacts/screens-1080p/confirm-ok.png",
    "political_title": "artifacts/screens-1080p/political-title.png",
    "no_commander": "artifacts/screens-1080p/no-commander.png",
    "trash": "artifacts/screens-1080p/plan-trash.png",
    "law_list": "artifacts/screens-1080p/law-list-title.png",
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
        # While wins are rare, how long the script holds out is the finer measure.
        lost = [
            g["seconds"] for g in decided if g["winner"] != g["started_as"] and g.get("seconds")
        ]
        return {
            "games": len(games),
            "decided": len(decided),
            "wins": wins,
            "rate": round(wins / len(decided), 3) if decided else None,
            "interval95": [round(low, 3), round(high, 3)],
            "lost_after_s": round(float(np.median(lost))) if lost else None,
        }

    played = [g for g in results if "error" not in g]
    report = {"all": tally(played), "errors": len(results) - len(played)}
    for side in ("BLU", "RED"):
        report[side] = tally([g for g in played if g["started_as"] == side])
    for attack in ATTACKS:
        report[attack] = tally([g for g in played if (g.get("plan") or {}).get("attack") == attack])
    for law in CONSCRIPTION:
        chosen = [g for g in played if (g.get("plan") or {}).get("conscription") == law]
        report[f"conscription_{law}"] = tally(chosen)
    return report
