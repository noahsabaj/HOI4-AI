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
- The green arrow above the army card activates the plan. It has three looks: idle
  (dots), ready (a green check, once the divisions are in place, as after a redraw)
  and executing (lit, on a green ground). Idle and ready both need a click. The red
  stop button beside it shows with any plan and not without one.
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
- U opens Recruit & Deploy. Train on the army's template adds a deployment line that
  trains division after division ("∞") while manpower and equipment last; its "No
  location set" asks for a state, picked on the map, where the player's own land shows
  green; Add Unit adds a slot to the line. Trained divisions deploy there unassigned,
  and the top bar's alert shows again: shift+click on it selects them, and a
  right-click on the army's card adds them to the army (8/24 became 16/24 in the
  calibration game). Winning games left up to 148k manpower unused while the AI never
  had more than 8 divisions.
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
    # The plan's arrow with a green check (ready, not executing): 1.00 where it showed.
    "ready": 0.8,
    # Recruit & Deploy's title: 1.00 open, at most 0.33 closed. A deployment line's red
    # "No location set": 0.96-1.00 shown, 0.73 on every other screen.
    "recruit_title": 0.8,
    "no_location": 0.9,
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
# Recruit & Deploy at 1080p (U): the first template's Train button, and on the line it
# adds, the location button, Add Unit and the line's delete button. The panels cover the
# screen's left PANELS_RIGHT pixels, so the deployment state is picked to their right.
RECRUIT = 0x55
TRAIN, DEPLOY_AT, ADD_UNIT, DROP_LINE = (742, 276), (220, 433), (386, 433), (480, 435)
PANELS_RIGHT = 910
# Training slots a game opens for new divisions; 0 recruits none, to measure the rest.
RECRUITS = {0: 0.25, 2: 0.35, 4: 0.4}
# Seconds after the conscription goal before recruiting, about 150 days at speed 5. Slots
# opened at the start took every man the divisions needed to fill up from 31%: in the
# first game with them, as Blue, the army's deployed manpower fell from 14.7k to 5.5k, no
# new division ever finished, and Blue surrendered in December 1936.
RECRUIT_AFTER = 60
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
# The first army's card in the army bar at the bottom, 1080p fractions, and the red stop
# button above it (x0, y0, x1, y1 in 1080p pixels), which shows while the army has a plan.
ARMY_CARD = (947 / 1920, 1010 / 1080)
STOP_BUTTON = (924, 953, 938, 961)
# The execute arrow beside it (x0, y0, x1, y1), dark while the plan is idle or ready and
# lit while it executes: its green averaged 54-65 in the first two looks and 90-105 lit
# (2026-09-24). The dots or the check before it come and go while executing too.
ARROW, LIT_GREEN = (962, 951, 996, 965), 78
# The army bar, where the create-army + shows: the bottom tenth of the screen.
ARMY_BAR_TOP = 0.88
# The arena's land grid: 24 province columns (12 a side) by 8 rows, and states of 3 by 4
# provinces, 8 a side. Blue's ids run 1 to 8 and Red's 9 to 16 (mapgen.state_cell).
COLUMNS, ROWS, STATE_WIDTH, STATE_HEIGHT, STATE_ROWS = 24, 8, 3, 4, 2
ENEMY = {"BLU": "RED", "RED": "BLU"}
# Arrows toward one state lost the front's flanks or its rear in every game with them
# (2026-09-23), so most games attack broad. A "front" attack draws no offensive line: the
# front line itself is executed, and pushes along the whole border.
ATTACKS = {"broad": 0.55, "front": 0.15, "near": 0.15, "deep": 0.15}
# The attacks that draw an offensive line.
OFFENSIVES = ("broad", "near", "deep")
# The share of games for each kind of plan: the best found so far (best_plan), the best
# with one change under test (CHALLENGER), and the rest with every choice drawn at random,
# so the recordings still show varied plans, good and bad.
SHARES = {"best": 0.4, "challenger": 0.3, "explore": 0.3}
# The change under test. Broad offensives pulled the middle of the front forward while
# the AI kept pockets and a salient in the script's own land (2026-09-23), and one won
# game swung back from the AI's 96% surrender to 24%: a front line executed alone pushes
# along the whole border and follows it as it moves.
CHALLENGER = {"variant": "front", "attack": "front"}


def best_plan(rng):
    """The best plan found so far: hold the line, then broad offensives.

    Since the fixes of 2026-09-23, every game that held for 90 s or more before
    attacking won (6 of 6, holds of 113 to 227 s), and three of the four that attacked
    within 60 s lost. While the front holds, the AI loses 3 to 6 men for each of the
    script's attacking it, and the script's divisions fill up from the manpower that All
    Adults Serve brings, to about 48k against the AI's 14-18k.
    """
    return {
        "best": True,
        "variant": "best",
        "conscription": "all_adults",
        "attack": "broad",
        "recruit": 0,
        "wait": round(rng.uniform(120, 240)),
        "redraw": round(rng.uniform(30, 90)),
    }


def choose_plan(rng, shares=None):
    """One game's strategy: the best plan (best_plan), the best plan with the change under
    test (CHALLENGER), or every choice drawn at random, in the given `shares` (SHARES).

    `attack` is where the offensive goes: "broad", the whole front forward by a third of
    the enemy's land; "near", toward one of the enemy's border states; or "deep", toward
    its rear. `wait` is how long the front holds, with the offensive drawn, before the
    plan is executed: preparation raises the plan's bonus, and a longer hold lets the
    divisions fill up, while the enemy may strike first. `redraw` is how often the plan
    is drawn afresh: every order deleted, then a new front and offensive.
    """
    shares = shares or SHARES
    kind = rng.choices(list(shares), weights=list(shares.values()))[0]
    if kind == "best":
        return best_plan(rng)
    if kind == "challenger":
        return {**best_plan(rng), "best": False, **CHALLENGER}
    attack = rng.choices(list(ATTACKS), weights=list(ATTACKS.values()))[0]
    return {
        "best": False,
        "variant": "explore",
        # How far up the conscription laws to go as political power allows.
        "conscription": rng.choices(list(CONSCRIPTION), weights=list(CONSCRIPTION.values()))[0],
        "attack": attack,
        # Slots training new divisions with the manpower the laws bring.
        "recruit": rng.choices(list(RECRUITS), weights=list(RECRUITS.values()))[0],
        # Planning reaches its full 30% bonus in 15 days, about 6 s at speed 5. Most games
        # hold far longer: the first win held 142 s while the AI lost 29k men against the
        # line to its 10k and the script's divisions filled up, and the next broad game,
        # which held 54 s, lost.
        "wait": round(rng.uniform(6, 60) if rng.random() < 0.2 else rng.uniform(90, 240)),
        # An offensive stops at its line, so it is always drawn again, further on: a single
        # push took one state and then stood for four years (2026-09-23).
        "redraw": round(rng.uniform(30, 90)),
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
        self.activate_at = self.redraw_at = self.law_at = self.reinforce_at = math.inf
        self.recruit_at = math.inf
        self.recruit_tries = 0
        # Whether the plan executes now (lit), the game runs, and the hold is over.
        self.active = self.running = self.attacking = False
        # Attempts at activating the current plan.
        self.tries = 0
        # The conscription law in force, as an index into LAWS, and failed tries at the
        # next step.
        self.law_step = self.law_fails = 0
        # A setup that failed, for play() to end the game with; later orders that failed.
        self.error = None
        self.failures = []
        # Where to keep each settled full view of the map, if anywhere (arena tests), and
        # the land masks of the last one.
        self.overview_dir = None
        self.last_view = None

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
            # Off the card: the general's tooltip over it covers the plan's buttons.
            act(desk, [{"kind": "move", "x": 0.5, "y": 0.5}])
            time.sleep(0.3)
        return self.selected(desk)

    def overview(self, desk, tries=8):
        """Zoomed fully out over the arena: the land masks and their box, or Nones.

        Taken once the camera has come to rest, when two looks 0.3 s apart find the same
        land box. The zoom out glides on after `recentre` returns while a game runs, and
        orders placed from a screen taken during the glide missed: one redraw's four front
        clicks landed off the border, the fourth on a state, which opened its panel.
        """
        recentre(desk)
        last = None
        for _ in range(tries):
            time.sleep(0.3)
            rgb = screen(desk)
            top, bottom = MAP_TOP, rgb.shape[0] - MAP_BOTTOM
            blue, red = country_pixels(rgb[top:bottom])
            if blue is None:
                return rgb, None, None, None
            blue, red = clean(blue), clean(red)
            if not blue.any() or not red.any():
                return rgb, None, None, None
            box = land_box(blue, red)
            if last is not None and max(abs(int(a) - int(b)) for a, b in zip(box, last)) <= 2:
                break
            last = box
        if self.overview_dir is not None:
            from PIL import Image

            Image.fromarray(rgb).save(self.overview_dir / f"{self.frame():06d}.png")
        if blue is not None:
            self.last_view = (blue, red)
        return rgb, blue, red, box

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
        """A front line along the whole border, checked: the army card shows a plan.

        The front line tool takes a click on the enemy's side of the border, on one of
        the fronts it highlights ("You cannot draw Front Line here" anywhere else, deep in
        enemy land included). Its tooltip calls the result a defensive line. On
        2026-09-23 two games in three as Red drew no front at first: the click had gone
        to the player's own side, where the tool does nothing.
        """
        for attempt in range(tries):
            # A fresh look each time: the camera may have moved since the last.
            rgb, blue, red, box = self.overview(desk)
            if box is None or not self.select_army(desk):
                raise RuntimeError("no arena or army to draw a front line with")
            front = self.front(blue, red)
            if not front:
                raise RuntimeError("the two countries do not touch on screen")
            # The border's middle first: the tool follows the whole border from there.
            middle = sorted(front, key=lambda p: p[1])[len(front) // 2]
            x, y = middle if attempt == 0 else self.rng.choice(front)
            act(desk, tap(FRONT_LINE))
            self.click(desk, self.screen_point(rgb, x, y))
            time.sleep(0.8)
            if plan_shown(screen(desk)):
                self.order("front", at=self.box_point(box, x, y), tries=attempt + 1)
                return
            self.select_army(desk)
        raise RuntimeError("no front line took: the army card never showed a plan")

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
        """The plan executed: a click on the card's arrow while it is idle (dots) or
        ready (a green check), checked. True once the arrow is lit, that is executing.

        The check shows when a plan is redrawn for divisions already in place, and it
        does not execute: taken for executing, it left one game's army holding its line
        for five years after the first redraw, from 1936 to the 15-minute cap.
        """
        if not self.select_army(desk):
            return False
        act(desk, [{"kind": "move", "x": 0.5, "y": 0.5}])
        time.sleep(0.3)
        rgb = screen(desk)
        button = self.find(rgb, "activate", top=0.8) or self.find(rgb, "ready", top=0.8)
        if button is None:
            return self.lit(desk)
        self.click(desk, button)
        time.sleep(0.8)
        if not self.lit(desk):
            return False
        self.order("activate")
        return True

    def lit(self, desk):
        """Whether the army's plan is executing: a plan shows and its arrow is lit.

        Looked at with the pointer off the bar. In a game on 2026-09-24 the general's
        tooltip covered the arrow, neither the idle nor the ready look was found, and the
        plan was taken for executing: it stood ready for 65 s of an attack.
        """
        act(desk, [{"kind": "move", "x": 0.5, "y": 0.5}])
        time.sleep(0.3)
        rgb = screen(desk)
        waiting = self.find(rgb, "activate", top=0.8) or self.find(rgb, "ready", top=0.8)
        self.active = plan_shown(rgb) and waiting is None and arrow_lit(rgb)
        return self.active

    def guarding(self):
        """Whether the home half needs the army: with a plan's `guard`, a redraw during the
        attack executes the front line alone (it pushes along the whole border, the
        enemy's incursion included) instead of pushing on, while the enemy holds at
        least that share of the home half (incursion)."""
        guard = self.plan.get("guard")
        if not guard or self.last_view is None:
            return False
        held = incursion(*self.last_view, self.country)
        if held >= guard:
            self.order("guard", held=round(held, 3))
        return held >= guard

    def setup(self, desk):
        """While paused: the army, its general, its front, its offensive; then run."""
        self.form_army(desk)
        self.assign_general(desk)
        self.draw_front(desk)
        if self.plan["attack"] in OFFENSIVES:
            self.draw_offensive(desk)
        run_at(desk, self.rules, self.speed)
        self.running = True
        self.order("run", speed=self.speed)
        self.start(time.monotonic())

    def start(self, now):
        """The later orders' times, from the moment the game runs. The redraws start with
        the attack (step): while the front holds, the plan stands."""
        if self.plan["attack"] != "none":
            self.activate_at = now + self.plan["wait"]
        if self.plan.get("conscription") in LAWS[1:]:
            # About 150 political power after half a minute at speed 5.
            self.law_at = now + 30
        if self.plan.get("recruit") and self.law_at == math.inf:
            self.recruit_at = now + RECRUIT_AFTER

    def due(self):
        now = time.monotonic()
        waits = (self.activate_at, self.redraw_at, self.law_at, self.recruit_at, self.reinforce_at)
        return min(waits) <= now

    def step(self, desk):
        """The next due order, once the game runs. True if the camera was moved.

        Conscription comes before redraws, and the plan is first redrawn once it executes.
        In a game on 2026-09-24 the plan was redrawn every 33 s while the front held, each
        redraw taking 15-30 s: the law steps found few turns between them (Limited at 67 s
        and Extensive at 146 s, against 41 s and 65 s in the wins), every redraw deleted
        the front line under the divisions, and the AI broke through before the attack.
        """
        now = time.monotonic()
        if now >= self.activate_at:
            # The hold is over: from now on every new plan is executed.
            self.attacking = True
            self.tries += 1
            if not self.activate(desk) and self.tries < 6:
                self.activate_at = now + 5
                return False
            self.activate_at, self.tries = math.inf, 0
            if self.redraw_at == math.inf and self.plan["redraw"]:
                # The first redraw; soon, if the plan would not execute, to draw it afresh.
                self.redraw_at = time.monotonic() + (self.plan["redraw"] if self.active else 5)
            return False
        if now >= self.law_at:
            done = self.raise_conscription(desk)
            self.law_at = math.inf if done else now + 10
            if done and self.plan.get("recruit"):
                self.recruit_at = now + RECRUIT_AFTER
            return False
        if now >= self.redraw_at:
            # Soon again, should the redraw fail part way.
            self.redraw_at = now + 5
            self.clear_orders(desk)
            self.draw_front(desk)
            if self.plan["attack"] in OFFENSIVES and not self.guarding():
                self.draw_offensive(desk)
            if self.attacking:
                # A new plan waits to be executed, like the first.
                self.activate_at = time.monotonic()
            # The period counts from the redraw's end, so it never crowds out the rest.
            self.redraw_at = time.monotonic() + self.plan["redraw"]
            return True
        if now >= self.recruit_at:
            self.overview(desk)
            self.recruit_tries += 1
            if self.recruit(desk):
                self.recruit_at, self.reinforce_at = math.inf, now + 20
            else:
                self.recruit_at = now + 30 if self.recruit_tries < 3 else math.inf
            return True
        if now >= self.reinforce_at:
            alert = self.find(screen(desk), "unassigned") is not None
            # A full army (24) leaves the alert up: then stop trying.
            self.reinforce_at = now + 20 if not alert or self.reinforce(desk) else math.inf
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
        return self.panel(desk, "political_title", POLITICS, shown, tries)

    def panel(self, desk, title, key, shown, tries=3):
        """A screen that `key` toggles, opened or closed, checked by its `title`."""
        for _ in range(tries):
            if (self.find(screen(desk), title) is not None) == shown:
                return True
            act(desk, tap(key))
            time.sleep(0.8)
        return False

    def recruit(self, desk):
        """Training slots for more divisions of the army's template, deployed in one of
        the player's own states. True once they are queued with a place to deploy.

        A line left without a place would train and never deploy, so it is deleted.
        """
        slots = self.plan.get("recruit", 0)
        if not slots or not self.panel(desk, "recruit_title", RECRUIT, True):
            return False
        self.click(desk, pixels(*TRAIN))
        time.sleep(0.8)
        placed = False
        if self.find(screen(desk), "no_location") is not None:
            self.click(desk, pixels(*DEPLOY_AT))
            time.sleep(0.8)
            spot = own_land_lit(screen(desk))
            if spot is not None:
                self.click(desk, spot)
                time.sleep(0.8)
                placed = self.find(screen(desk), "no_location") is None
            if not placed:
                self.click(desk, pixels(*DROP_LINE))
                time.sleep(0.5)
        if placed:
            for _ in range(slots - 1):
                self.click(desk, pixels(*ADD_UNIT))
                time.sleep(0.4)
            self.order("recruit", slots=slots)
        self.panel(desk, "recruit_title", RECRUIT, False)
        return placed

    def reinforce(self, desk):
        """New divisions into the army: shift+click on the Unassigned divisions alert
        selects them all, and a right-click on the army's card adds them. True if they
        joined, as the alert went away; False with no alert, or a full army."""
        act(desk, [{"kind": "move", "x": 0.5, "y": 0.5}])
        alert = self.find(screen(desk), "unassigned")
        if alert is None:
            return False
        self.click(desk, alert, shift=True)
        time.sleep(0.8)
        self.click(desk, ARMY_CARD, button=1)
        time.sleep(0.8)
        act(desk, [{"kind": "move", "x": 0.5, "y": 0.5}])
        time.sleep(0.3)
        joined = self.find(screen(desk), "unassigned") is None
        if joined:
            self.order("reinforce")
            if self.active:
                # Execute again, should the new divisions have left the plan waiting.
                self.activate_at = time.monotonic()
        return joined

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


def incursion(blue, red, country):
    """The share of `country`'s home half of the arena that its enemy holds, from the
    land masks of a full view: the half of the land box on its own side of the seam.

    In the wins the enemy held 0-14% of it when the attack began, and none after the
    push. In the loss of 2026-09-24 it grew while the army pushed on, 10%, 17%, then
    89%, as the AI's last divisions walked into the empty rear and took its victory
    points; in a win that swung back it reached 22%.
    """
    box = land_box(blue, red)
    if box is None:
        return 0.0
    top, left, bottom, right = box
    seam = (left + right) // 2
    own, enemy = (blue, red) if country == "BLU" else (red, blue)
    half = slice(left, seam) if country == "BLU" else slice(seam, right)
    held = enemy[top:bottom, half].sum()
    return float(held / max(1, held + own[top:bottom, half].sum()))


def green_plus(rgb):
    """The centre of the green + in the army bar, as screen fractions, or None."""
    top = int(ARMY_BAR_TOP * rgb.shape[0])
    bar = rgb[top:].astype(np.int32)
    r, g, b = bar[..., 0], bar[..., 1], bar[..., 2]
    ys, xs = np.nonzero((g - np.maximum(r, b) > 25) & (g > 90))
    if len(xs) < GREEN_PLUS:
        return None
    return float(np.median(xs)) / rgb.shape[1], (float(np.median(ys)) + top) / rgb.shape[0]


def plan_shown(rgb):
    """Whether the first army's card shows a plan: its red stop button is lit.

    The execute arrow beside it has three looks (idle, executing, and a green check once a
    plan is redrawn while executing), but the stop button shows with any plan and not
    without one: red in 71% (executing) to 96% (idle) of its box, 0% with no plan.
    """
    x0, y0, x1, y1 = STOP_BUTTON
    box = rgb[y0:y1, x0:x1].astype(np.int32)
    red = (box[..., 0] > 100) & (box[..., 0] - np.maximum(box[..., 1], box[..., 2]) > 50)
    return bool(red.mean() > 0.3)


def arrow_lit(rgb):
    """Whether the first army's execute arrow is lit, as while its plan executes."""
    x0, y0, x1, y1 = ARROW
    return float(rgb[y0:y1, x0:x1, 1].mean()) > LIT_GREEN


def own_land_lit(rgb, least=2000):
    """Where to click to pick a deployment state, as screen fractions, or None: the middle
    of the player's own land, which the map lights green while a location is being picked,
    right of the recruitment panels. 22,102 such pixels while picking in the calibration
    game, at most 332 green ones on other screens."""
    part = rgb[100:900, PANELS_RIGHT:].astype(np.int32)
    r, g, b = part[..., 0], part[..., 1], part[..., 2]
    ys, xs = np.nonzero((g - r > 40) & (g - b > 15) & (g > 100))
    if len(xs) < least:
        return None
    return (float(np.median(xs)) + PANELS_RIGHT) / rgb.shape[1], (
        float(np.median(ys)) + 100
    ) / rgb.shape[0]


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
    "ready": "artifacts/screens-1080p/plan-ready.png",
    "recruit_title": "artifacts/screens-1080p/recruit-title.png",
    "no_location": "artifacts/screens-1080p/no-location.png",
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
    # The best plan's record apart from the rest (plans before 2026-09-24 have no `best`
    # and count as exploring), and each variant's: the best, challengers, exploration.
    best = [g for g in played if (g.get("plan") or {}).get("best")]
    report["best"] = tally(best)
    report["explore"] = tally([g for g in played if not (g.get("plan") or {}).get("best")])
    variants = {(g.get("plan") or {}).get("variant") for g in played} - {None}
    for variant in sorted(variants):
        chosen = [g for g in played if (g.get("plan") or {}).get("variant") == variant]
        report[f"variant_{variant}"] = tally(chosen)
    for side in ("BLU", "RED"):
        report[side] = tally([g for g in played if g["started_as"] == side])
        report[f"best_{side}"] = tally([g for g in best if g["started_as"] == side])
    # By arena, where the results name it (since 2026-09-24).
    for arena in sorted({g["arena"] for g in played if g.get("arena")}):
        report[f"arena_{arena}"] = tally([g for g in played if g.get("arena") == arena])
        report[f"best_arena_{arena}"] = tally([g for g in best if g.get("arena") == arena])
    for attack in ATTACKS:
        report[attack] = tally([g for g in played if (g.get("plan") or {}).get("attack") == attack])
    for law in CONSCRIPTION:
        chosen = [g for g in played if (g.get("plan") or {}).get("conscription") == law]
        report[f"conscription_{law}"] = tally(chosen)
    for slots in RECRUITS:
        chosen = [g for g in played if (g.get("plan") or {}).get("recruit", 0) == slots]
        report[f"recruit_{slots}"] = tally(chosen)
    return report
