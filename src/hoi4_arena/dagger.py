"""DAgger: the scripted player labels the states the learned player leads the game into.

Behaviour cloning learns from the scripted player's own games, so it never sees the
states its own mistakes lead to, and it has no idea how to recover from them (covariate
shift; bc5 formed an army in 24 of 80 practice episodes and drew a front in none). DAgger
(Ross et al. 2011, arXiv 1011.0686) lets the learner play, asks the expert what it would
do in every state the learner reached, adds those labels to the data and trains again.

The expert is the scripted player's setup (scripted.Planner), which is a program and so
can be asked about any screen. But the Planner is a procedure: it clicks, sleeps, looks
again and clicks on, and it keeps its progress in its own call stack. It has no answer
to "what now, on this screen?". `Expert` gives it one. It is the same setup (the army
from the unassigned divisions, its general, a front along the whole border, a broad
offensive, then the game started with a click on +), the same detectors (the Planner's
templates and colour checks, practice.army_card) and the same geometry (Planner.front,
Planner.broad_line), made reactive: each decision it looks at the frame and the inputs
applied so far, and says what the next 200 ms should hold.

What it may know is what a player at that moment could: the frame (the one the learned
player saw at that decision), and the inputs already applied (whoever made them, the
policy, the coach or the harness). It has no plan state of its own that the screen and
those inputs cannot rebuild:
- which step is due comes from the screen alone: the army's card (lit, with a portrait),
  the unassigned alert, the plan's stop button, the Battle Plans bar;
- a click under way is finished from the inputs: a key or button still down is let go,
  a shift+click whose shift is down gets its press, then its shift released; nothing is
  pressed until the pointer is on the target (the policy looks before it clicks);
- a list or tool the screen does not show is read from the inputs: the commander list
  is open after a click on the portrait, the offensive tool is on after an X;
- whether an offensive was drawn, which no screen check sees, is read from the inputs
  too: an X, then a right-drag, since the front showed. Rebuilt the same way every
  decision, so a learner that went its own way is followed, never assumed to be where
  the script would be.
- It waits SETTLE after any press, as the Planner sleeps after its clicks, so a screen
  that has not caught up yet is not answered twice.

Labels are made offline, from the recording of a practice episode (relabel): the decoded
frame each decision read, and the inputs recorded before it. Nothing extra runs while the
learner plays. Where the expert has no opinion (the setup done, a screen it cannot read)
it abstains, and the decision is not a label. During a coach's takeover the coach's own
inputs are the label, since the coach is the scripted player. `train-bc --dagger W`
trains practice recordings on these labels (dataset.session_labels).

Its faithfulness is measured on the scripted player's own recordings (validate), where
what it would do is what it did.
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path

import numpy as np

from .actions import SLOTS, VOCAB, encode_interval

log = logging.getLogger(__name__)

DAGGER_LABELS = "labels-dagger.npz"
# The left, right and middle button presses, by kind index.
VOCAB_BUTTONS = {
    i: e["button"] for i, e in enumerate(VOCAB) if e and e["kind"] == "button" and e["down"]
}
# How near an expert's move must go to where the scripted player clicked to count as
# aiming at it (agreement), in pixels.
INTENT_PX = 40
# Seconds of a scripted recording validate reads: its setup takes 25-40 s.
SETUP_SECONDS = 60
PERIOD_NS = 200_000_000
# Seconds after a press before the expert acts again: the Planner sleeps 0.8 s after
# each click so that the screen shows what it did before it looks again.
SETTLE = 0.8
# How near the pointer must be to a target to press there, in 1080p pixels.
NEAR = 12
# Seconds within which a click on the portrait still has the commander list open, an X
# still has the offensive tool on, a shift+click on the alert still has its divisions
# selected.
RECENT = 4.0
# Wheel notches down a full zoom out takes (ai_games.recentre: ZOOM_MAX + 4).
ZOOM_OUT = 30
# The fewest front points the front and offensive are drawn from (ai_games.MIN_FRONT).
MIN_FRONT = 40
# 1080p places (pixels): the first army's card, the army panel's commander portrait,
# the first commander in the list it opens, the speed control's +, and the confirmation
# dialog's Cancel and the law list's close button (scripted.py).
ARMY_CARD = (947, 1010)
COMMANDER_SLOT, FIRST_COMMANDER = (30, 140), (950, 352)
SPEED_UP = (1789, 20)
CANCEL, CLOSE_LIST = (884, 677), (1005, 100)
SHIFT, FRONT_LINE, OFFENSIVE_LINE, POLITICS, RECRUIT = 0x10, 0x5A, 0x58, 0x51, 0x55
# Where each template is searched (x0, y0, x1, y1), 1080p pixels: round where it showed
# on a scripted game and two practice episodes (2026-09-26: the alert at (809, 41), the
# Battle Plans bar at (662, 870), "No Commander" at (68, 133), the line tools' note at
# (1040, 845), the political screen's title at (29-40, 94), the law list's at (620-700,
# 86), the dialog's OK at (1030, 667); top left corners). A whole 1080p frame costs about
# 0.1 s a template, these a few milliseconds. Recruit & Deploy never showed: a guess.
REGIONS = {
    "unassigned": (0, 0, 1920, 110),
    "plans_bar": (300, 780, 1300, 960),
    "no_commander": (0, 80, 500, 250),
    "front_tool": (700, 800, 1500, 900),
    "political_title": (0, 50, 400, 160),
    "law_list": (400, 40, 1100, 150),
    "confirm_ok": (850, 600, 1250, 740),
    "recruit_title": (0, 0, 960, 300),
}
# Where the create-army + shows before any army exists (x0, y0, x1, y1): the scripted
# player clicked it at (988, 1013) in every game (play.MILESTONES).
PLUS_BOX = (948, 978, 1028, 1048)
# Seconds past the harness's hold limit (play.HOLD_LIMITS) after which an input still
# down in the record counts as released.
HELD_MARGIN = 0.4
# How long the pause mark may stay unseen and the game still count as paused: it blinks.
PAUSE_BLINK = 1.5
# How long after the army's card lit the unassigned alert may linger before it means
# divisions were left out of the army (it went ~0.8 s after the card lit, 2026-09-26).
ALERT_LINGERS = 2.0


def pixels(point, width=1920, height=1080):
    return point[0] / width, point[1] / height


class Inputs:
    """What the inputs applied so far say, kept up to date one event at a time (`feed`):
    what is held down, where the pointer went, and when each key and button was last
    pressed, and where."""

    def __init__(self):
        self.held = set()
        self.pointer = None
        self.pressed = {}  # (kind, code) -> (t_ns, pointer at the press)
        self.released = {}
        self.wheel_down = []  # t_ns of each notch out
        self.drag = []  # pointer positions while the right button is down
        self.last_drag = None  # (t_ns of release, positions) of the last right-drag
        self.last_press = -math.inf

    def feed(self, item):
        event, t = item["event"], item["t_ns"]
        kind = event["kind"]
        if kind == "move":
            self.pointer = (event["x"], event["y"])
            if ("button", 1) in self.held:
                self.drag.append(self.pointer)
            return
        if kind == "wheel":
            if event.get("delta", 0) < 0:
                self.wheel_down.append(t)
            return
        code = (kind, event["vk"] if kind == "key" else event["button"])
        if event["down"]:
            self.held.add(code)
            self.pressed[code] = (t, self.pointer)
            self.last_press = t
            if code == ("button", 1):
                self.drag = [self.pointer]
        else:
            self.held.discard(code)
            self.released[code] = t
            if code == ("button", 1):
                self.last_drag = (t, list(self.drag))

    def down(self, t):
        """What is held at `t`. A key or button down longer than the harness lets one be
        (play.HOLD_LIMITS, and a little over) counts as up: the harness let it go, or
        released every input before its own (desk.release, which leaves no event)."""
        from .play import HOLD_LIMITS

        out = set()
        for kind, code in self.held:
            limit = HOLD_LIMITS.get(f"{kind}{code}", HOLD_LIMITS[kind]) + HELD_MARGIN
            if (t - self.pressed[(kind, code)][0]) / 1e9 <= limit:
                out.add((kind, code))
        return out

    def let_go(self):
        """Everything up: the harness or the coach released every input."""
        self.held.clear()

    def since(self, code, t):
        """Seconds since `code` was last pressed, or inf."""
        at = self.pressed.get(code)
        return (t - at[0]) / 1e9 if at else math.inf

    def pressed_at(self, code):
        at = self.pressed.get(code)
        return at[1] if at else None


def near(a, b, tolerance=NEAR, width=1920, height=1080):
    """Whether two points (screen fractions) are within `tolerance` pixels."""
    if a is None or b is None:
        return False
    return math.hypot((a[0] - b[0]) * width, (a[1] - b[1]) * height) <= tolerance


def create_plus(rgb, box=PLUS_BOX):
    """The army bar's create-army +, lit green while divisions are selected, as screen
    fractions, or None: scripted.green_plus, looked for only where the first army's + is.
    Over the whole bar, green elsewhere passed for it once the game ran (a practice
    episode, 2026-09-26: at (281, 958))."""
    from .scripted import GREEN_PLUS

    x0, y0, x1, y1 = box
    part = rgb[y0:y1, x0:x1].astype(np.int32)
    r, g, b = part[..., 0], part[..., 1], part[..., 2]
    ys, xs = np.nonzero((g - np.maximum(r, b) > 25) & (g > 90))
    if len(xs) < GREEN_PLUS:
        return None
    return (float(np.median(xs)) + x0) / rgb.shape[1], (float(np.median(ys)) + y0) / rgb.shape[0]


class Seen:
    """What one frame shows, read with the scripted player's own checks."""

    def __init__(self, rgb, expert):
        from .practice import army_card
        from .scripted import plan_shown

        self.rgb, self.expert = rgb, expert
        self.army, self.general = army_card(rgb)
        self.plan = plan_shown(rgb)
        self.plus = create_plus(rgb)
        self._found = {}
        self._land = None

    def find(self, name):
        if name not in self._found:
            self._found[name] = self.expert.find(self.rgb, name)
        return self._found[name]

    def paused(self):
        rules = self.expert.rules
        return None if rules is None else bool(rules.matches("paused", self.rgb))

    def land(self):
        """(front points as screen fractions, the land box, blue, red) or Nones: the
        Planner's reading of a full view of the map."""
        if self._land is None:
            from .ai_games import MAP_BOTTOM, MAP_TOP
            from .scripted import clean, land_box
            from .vision import country_pixels

            crop = self.rgb[MAP_TOP : self.rgb.shape[0] - MAP_BOTTOM]
            blue, red = country_pixels(crop)
            self._land = (None, None, None, None)
            if blue is not None:
                crop16 = crop.astype(np.int16)
                blue = clean(blue & (crop16[..., 1] - crop16[..., 0] > 5))
                red = clean(red)
                if blue.any() and red.any():
                    box = land_box(blue, red)
                    front = self.expert.planner.front(blue, red)
                    self._land = (front, box, blue, red)
        return self._land


class Expert:
    """The scripted player's setup, answering one decision at a time (see the module's
    docstring). `label(rgb, t_ns, cursor)` after `feed`ing it every input applied before
    t_ns gives (events with their offsets in ms, phase), events None to abstain.

    `planner` and `templates` stand in for the Planner's (tests); `rules` reads the pause
    mark (vision.ScreenRules), without which + is clicked once the setup is done."""

    def __init__(self, country, *, rules=None, templates=None, planner=None, rng=None):
        from .scripted import TEMPLATES, Planner, best_plan, load_templates

        self.country, self.rules = country, rules
        self.rng = rng or random.Random(0)
        self.templates = templates if templates is not None else load_templates(TEMPLATES)
        self.planner = planner or Planner(
            country, best_plan(self.rng), self.templates, rules, 5, lambda: 0
        )
        self.inputs = Inputs()
        # When the army's card first lit, and the pause mark was last seen.
        self.army_since = None
        self.paused_at = None
        # When the front first showed, and the offensive's line while one is drawn.
        self.front_since = None
        self.line = None
        self.offensive_at = None
        self.width, self.height = 1920, 1080

    def feed(self, item):
        self.inputs.feed(item)

    def find(self, rgb, name):
        """Where a calibrated template is (screen fractions), searched in REGIONS[name]."""
        import cv2

        from .scripted import FOUND

        x0, y0, x1, y1 = REGIONS.get(name, (0, 0, rgb.shape[1], rgb.shape[0]))
        part = rgb[y0:y1, x0:x1]
        template = self.templates[name]
        if part.shape[0] < template.shape[0] or part.shape[1] < template.shape[1]:
            return None
        scores = cv2.matchTemplate(part, template, cv2.TM_CCOEFF_NORMED)
        _, best, _, (x, y) = cv2.minMaxLoc(scores)
        if best < FOUND[name]:
            return None
        h, w = template.shape[:2]
        return (x0 + x + w / 2) / rgb.shape[1], (y0 + y + h / 2) / rgb.shape[0]

    # The actions it can take, as (events, offsets in ms).
    def move(self, at):
        return [{"kind": "move", "x": float(at[0]), "y": float(at[1])}], [0.0]

    def click(self, at, button=0, shift=False, pointer=None):
        """Onto `at` first; pressed only once the pointer is there (it looks first)."""
        if not near(pointer, at, width=self.width, height=self.height):
            return self.move(at)
        press = [{"kind": "button", "button": button, "down": d} for d in (True, False)]
        if shift:
            return [{"kind": "key", "vk": SHIFT, "down": True}, press[0]], [0.0, 150.0]
        return press, [0.0, 150.0]

    def tap(self, vk):
        return [{"kind": "key", "vk": vk, "down": d} for d in (True, False)], [0.0, 150.0]

    def zoom_out(self, pointer):
        centre = (0.5, 0.5)
        if not near(pointer, centre, 40, self.width, self.height):
            return self.move(centre)
        return [{"kind": "wheel", "delta": -120}] * 7, [i * 25.0 for i in range(7)]

    def finish(self, t):
        """The rest of an input under way: what is held, let go, in the Planner's order.
        None if nothing is."""
        inputs = self.inputs
        held = inputs.down(t)
        if ("button", 1) in held:
            return self.drag_on(t)
        if ("button", 0) in held:
            up = [{"kind": "button", "button": 0, "down": False}]
            if ("key", SHIFT) in held:
                # The end of a shift+click: the button, then shift.
                return [*up, {"kind": "key", "vk": SHIFT, "down": False}], [0.0, 150.0]
            return up, [0.0]
        if ("button", 2) in held:
            return [{"kind": "button", "button": 2, "down": False}], [0.0]
        from .actions import KEYS

        for kind, code in sorted(held):
            # Not a key no policy may press (space, which the harness presses to start).
            if kind == "key" and code in KEYS:
                return [{"kind": "key", "vk": code, "down": False}], [0.0]
        return None

    def drag_on(self, t):
        """The offensive's right-drag under way: on along its line, 3 moves a stretch, or
        let go at its end. A right button the expert did not press is let go."""
        up = [{"kind": "button", "button": 1, "down": False}], [0.0]
        if self.line is None:
            return up
        pointer = self.inputs.pointer
        steps = dense(self.line, 3)
        # The next point past where the pointer is along the line.
        at = min(range(len(steps)), key=lambda i: math.dist(steps[i], pointer or steps[0]))
        rest = steps[at + 1 : at + 3]
        if not rest:
            self.offensive_at = t
            return up
        return [{"kind": "move", "x": x, "y": y} for x, y in rest], [0.0, 80.0][: len(rest)]

    def label(self, rgb, t, cursor=None):
        """(events, offsets in ms, phase) for the decision at `t` (ns) on frame `rgb`;
        events None where it abstains. `cursor` is where the frame shows the pointer
        (pixels), used until an input has moved it."""
        self.width, self.height = rgb.shape[1], rgb.shape[0]
        inputs = self.inputs
        pointer = inputs.pointer
        if pointer is None and cursor is not None:
            pointer = (cursor[0] / (self.width - 1), cursor[1] / (self.height - 1))
        ongoing = self.finish(t)
        if ongoing is not None:
            return (*ongoing, "finish")
        if (t - inputs.last_press) / 1e9 < SETTLE:
            return [], [], "settle"
        seen = Seen(rgb, self)
        if seen.paused():
            self.paused_at = t
        paused = self.paused_at is not None and (t - self.paused_at) / 1e9 < PAUSE_BLINK
        overlay = self.clear(seen, pointer)
        if overlay is not None:
            return overlay
        panel = self.close_panel(seen)
        card = pixels(ARMY_CARD, self.width, self.height)
        if seen.plan and self.front_since is None:
            self.front_since = t
        if not seen.plan:
            self.front_since = self.offensive_at = None
        self.army_since = (self.army_since or t) if seen.army else None
        if not seen.army:
            if seen.plus is not None:
                return (*self.click(seen.plus, pointer=pointer), "army:plus")
            alert = seen.find("unassigned")
            if alert is None:
                return None, None, "army:no-alert"
            return (*self.click(alert, shift=True, pointer=pointer), "army:alert")
        alert = seen.find("unassigned")
        if alert is not None and (t - self.army_since) / 1e9 > ALERT_LINGERS:
            # Divisions left out of the army (a click on the alert without shift): all of
            # them selected, then a right-click on the army's card adds them.
            shifted = inputs.pressed.get(("key", SHIFT))
            if (
                shifted
                and shifted[0] > self.army_since
                and inputs.since(("key", SHIFT), t) < RECENT
            ):
                return (*self.click(card, button=1, pointer=pointer), "army:join")
            return (*self.click(alert, shift=True, pointer=pointer), "army:alert")
        if panel is not None:
            return panel
        if not seen.general:
            if seen.find("plans_bar") is None:
                return (*self.click(card, pointer=pointer), "general:select")
            portrait = pixels(COMMANDER_SLOT, self.width, self.height)
            opened = inputs.since(("button", 0), t) < RECENT and near(
                inputs.pressed_at(("button", 0)), portrait, 30, self.width, self.height
            )
            if opened:
                first = pixels(FIRST_COMMANDER, self.width, self.height)
                return (*self.click(first, pointer=pointer), "general:commander")
            return (*self.click(portrait, pointer=pointer), "general:portrait")
        if not seen.plan:
            return self.front_step(seen, pointer, card, t)
        if self.offensive_at is None:
            drag = inputs.last_drag
            if drag is not None and drag[0] >= self.front_since and len(drag[1]) >= 3:
                if inputs.since(("key", OFFENSIVE_LINE), drag[0]) < RECENT:
                    self.offensive_at = drag[0]
        if self.offensive_at is None:
            return self.offensive_step(seen, pointer, card, t)
        if paused:
            if inputs.since(("button", 0), t) < 2 and near(
                inputs.pressed_at(("button", 0)), pixels(SPEED_UP), 14
            ):
                return [], [], "run:wait"
            return (*self.click(pixels(SPEED_UP), pointer=pointer), "run")
        return None, None, "done"

    def clear(self, seen, pointer):
        """A dialog or list the setup does not use, closed the way the Planner closes it."""
        if seen.find("confirm_ok") is not None:
            return (*self.click(pixels(CANCEL), pointer=pointer), "clear:confirm")
        if seen.find("law_list") is not None:
            return (*self.click(pixels(CLOSE_LIST), pointer=pointer), "clear:laws")
        return None

    def close_panel(self, seen):
        """A screen over the map the setup does not use, closed with its key. Not before the
        army is formed: the alert and the + are clear of it, and the scripted player forms
        its army through the political screen a scrambled drill opened (2026-09-26); the
        army's panel, its portrait and the map are under it."""
        if seen.find("political_title") is not None:
            return (*self.tap(POLITICS), "clear:politics")
        if seen.find("recruit_title") is not None:
            return (*self.tap(RECRUIT), "clear:recruit")
        return None

    def tool_on(self, seen, key, t):
        """Whether the line tool `key` (Z or X) is on: `key` was the last of the two
        pressed, lately, and no click since; for Z, the Battle Plans bar says so too ("N
        divisions will be assigned"; for X it shows only once the drag has begun)."""
        if key == FRONT_LINE and seen.find("front_tool") is None:
            return False
        inputs = self.inputs
        other = OFFENSIVE_LINE if key == FRONT_LINE else FRONT_LINE
        mine, theirs = inputs.since(("key", key), t), inputs.since(("key", other), t)
        clicks = min(inputs.since(("button", 0), t), inputs.since(("button", 1), t))
        return mine < theirs and mine < RECENT * 2 and mine < clicks

    def front_step(self, seen, pointer, card, t):
        if seen.find("plans_bar") is None:
            return (*self.click(card, pointer=pointer), "front:select")
        front, box, _, _ = seen.land()
        if not front or len(front) < MIN_FRONT:
            if len([w for w in self.inputs.wheel_down if t - w < 5e9]) >= ZOOM_OUT:
                return None, None, "front:no-border"
            return (*self.zoom_out(pointer), "front:zoom-out")
        if self.tool_on(seen, FRONT_LINE, t):
            # The border's middle, as the Planner's first try.
            from .ai_games import MAP_TOP

            x, y = sorted(front, key=lambda p: p[1])[len(front) // 2]
            at = (x / self.width, (y + MAP_TOP) / self.height)
            return (*self.click(at, pointer=pointer), "front:click")
        if seen.find("front_tool") is not None:
            return (*self.tap(OFFENSIVE_LINE if self.last_tool(t) == "x" else FRONT_LINE),
                    "front:tool-off")  # fmt: skip
        return (*self.tap(FRONT_LINE), "front:tool")

    def last_tool(self, t):
        z = self.inputs.since(("key", FRONT_LINE), t)
        x = self.inputs.since(("key", OFFENSIVE_LINE), t)
        return None if math.isinf(min(z, x)) else ("z" if z < x else "x")

    def offensive_step(self, seen, pointer, card, t):
        if seen.find("plans_bar") is None:
            return (*self.click(card, pointer=pointer), "offensive:select")
        if self.tool_on(seen, OFFENSIVE_LINE, t) and self.line is not None:
            start = self.line[0]
            if not near(pointer, start, NEAR, self.width, self.height):
                return (*self.move(start), "offensive:start")
            return [{"kind": "button", "button": 1, "down": True}], [0.0], "offensive:press"
        front, box, _, _ = seen.land()
        if not front or len(front) < MIN_FRONT:
            if len([w for w in self.inputs.wheel_down if t - w < 5e9]) >= ZOOM_OUT:
                return None, None, "offensive:no-border"
            return (*self.zoom_out(pointer), "offensive:zoom-out")
        from .ai_games import MAP_TOP

        line = self.planner.broad_line(front, box)
        if len(line) < 2:
            return None, None, "offensive:no-line"
        self.line = [(x / self.width, (y + MAP_TOP) / self.height) for x, y in line]
        if seen.find("front_tool") is not None and self.last_tool(t) == "z":
            return (*self.tap(FRONT_LINE), "offensive:tool-off")
        return (*self.tap(OFFENSIVE_LINE), "offensive:tool")


def dense(points, steps):
    """The moves a drag through `points` makes, `steps` to each (Planner.drag)."""
    out = [points[0]]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        out += [
            (x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps) for i in range(1, steps + 1)
        ]
    return out


def as_action(events, offsets, start_ns):
    """An expert's events as the action a decision at `start_ns` holds (SLOTS, 3)."""
    timed = [
        {"t_ns": start_ns + int(ms * 1e6), "event": event}
        for event, ms in zip(events, offsets, strict=True)
    ]
    return encode_interval(timed, start_ns)


def label_recording(recording, *, rules="artifacts/calibration-1080p/rules.json", seconds=None):
    """The expert's label for every decision of a recording (on the grid session_labels
    makes with lead_in 0), from its frames and the inputs recorded before each, or for
    those of its first `seconds`: {"decisions", "actions", "valid" (the expert gave
    one), "phase" (the step it was on)}.
    """
    from .nvdec import FfmpegFrames
    from .vision import ScreenRules

    recording = Path(recording)
    manifest = json.loads((recording / "manifest.json").read_text())
    rows = [json.loads(line) for line in (recording / "frames.jsonl").read_text().splitlines()]
    times = np.array([row["t_ns"] for row in rows], dtype=np.int64)
    events = sorted(
        (e for row in rows for e in row.get("scripted_events", [])), key=lambda e: e["t_ns"]
    )
    country = manifest.get("started_as") or "BLU"
    # A coach's takeover starts by releasing every input, which leaves no event.
    let_go = sorted(
        int(times[min(int(span["from_frame"]), len(times) - 1)])
        for span in manifest.get("coached") or []
        if span.get("from_frame") is not None
    )
    expert = Expert(country, rules=ScreenRules(rules) if rules else None)
    decisions = np.arange(times[0], times[-1] - PERIOD_NS, PERIOD_NS, dtype=np.int64)
    if seconds is not None:
        decisions = decisions[decisions < times[0] + int(seconds * 1e9)]
    frame_ids = np.searchsorted(times, decisions, side="right") - 1
    actions = np.zeros((len(decisions), SLOTS, 3), dtype=np.int64)
    valid = np.zeros(len(decisions), dtype=bool)
    phases = []
    reader = FfmpegFrames(recording / "screen.mkv", manifest["width"], manifest["height"])
    try:
        frame, index, fed = None, -1, 0
        for d, t in enumerate(decisions):
            while index < frame_ids[d]:
                frame, index = reader.read(), index + 1
                if frame is None:
                    raise ValueError(f"{recording.name}: the video ends at frame {index}")
            while fed < len(events) and events[fed]["t_ns"] < t:
                while let_go and let_go[0] <= events[fed]["t_ns"]:
                    expert.inputs.let_go()
                    let_go.pop(0)
                expert.feed(events[fed])
                fed += 1
            got, offsets, phase = expert.label(frame, int(t), rows[index].get("cursor"))
            if got is not None:
                try:
                    actions[d] = as_action(got, offsets, int(t))
                    valid[d] = True
                except ValueError:
                    phase = "unencodable"
            phases.append(phase)
    finally:
        reader.close()
    return {"decisions": decisions, "actions": actions, "valid": valid, "phase": np.array(phases)}


def summarise(recording, labels):
    counts = {}
    for phase in labels["phase"].tolist():
        counts[phase] = counts.get(phase, 0) + 1
    return {"recording": Path(recording).name, "decisions": len(labels["decisions"]),
            "labelled": int(labels["valid"].sum()), "phases": counts}  # fmt: skip


def relabel(recording, **options):
    """label_recording, written to DAGGER_LABELS in the recording. Returns a summary."""
    labels = label_recording(recording, **options)
    np.savez(Path(recording) / DAGGER_LABELS, **labels)
    return summarise(recording, labels)


def recorded_actions(recording, decisions):
    """The inputs a recording holds, as actions on the decision grid `decisions`: every
    input, whoever made it (session_labels leaves some out; the expert saw them all)."""
    lines = (Path(recording) / "frames.jsonl").read_text().splitlines()
    events = sorted(
        (e for line in lines for e in json.loads(line).get("scripted_events", [])),
        key=lambda e: e["t_ns"],
    )
    from .actions import KEYS

    # Keys no policy may press (space, which starts the game) are the harness's.
    events = [e for e in events if e["event"]["kind"] != "key" or e["event"]["vk"] in KEYS]
    times = np.array([e["t_ns"] for e in events], dtype=np.int64)
    actions = np.zeros((len(decisions), SLOTS, 3), dtype=np.int64)
    for d, t in enumerate(decisions):
        lo, hi = np.searchsorted(times, [t, t + PERIOD_NS])
        try:
            actions[d] = encode_interval(events[lo:hi], int(t))
        except ValueError:
            pass
    return actions


def presses(actions, pointer=None):
    """[(decision, kind index, pointer)] of every key or button press in `actions`, the
    pointer where it was pressed (screen fractions): from `pointer` (the pointer's place
    each decision starts at) when given, else followed through the actions' own moves."""
    from .actions import GRID, PRESSES

    out, at = [], None
    for d in range(len(actions)):
        if pointer is not None:
            at = pointer[d]
        for kind, x, y in actions[d]:
            kind = int(kind)
            if kind == 1:
                at = (x / (GRID - 1), y / (GRID - 1))
            elif kind in PRESSES:
                out.append((d, kind, at))
    return out


def moves(actions):
    """[(decision, target)] of every move in `actions`."""
    from .actions import GRID

    return [
        (d, (int(x) / (GRID - 1), int(y) / (GRID - 1)))
        for d in range(len(actions))
        for kind, x, y in actions[d]
        if int(kind) == 1
    ]


def own_presses(recorded, until):
    """The scripted player's presses before `until` that a learned player makes itself:
    not the clicks on + after the first. The scripted player clicks + four times, from
    speed 1 to 5; a learned player's first click on + has the harness start the game and
    set its speed (play.run_game)."""
    speed_up = pixels(SPEED_UP)
    out, seen = [], False
    for d, k, at in presses(recorded):
        if d >= until:
            continue
        if near(at, speed_up, 14) and VOCAB_BUTTONS.get(k) == 0:
            if seen:
                continue
            seen = True
        out.append((d, k, at))
    return out


def pointer_track(recorded):
    """Where the recorded pointer is as each decision starts (screen fractions or None)."""
    from .actions import GRID

    track, at = [], None
    for d in range(len(recorded)):
        track.append(at)
        for kind, x, y in recorded[d]:
            if int(kind) == 1:
                at = (int(x) / (GRID - 1), int(y) / (GRID - 1))
    return track


def agreement(recorded, expert, valid, until=None, window=3, aim=INTENT_PX):
    """How well the expert matches the scripted player on the scripted player's own
    recording, over its decisions before `until`:
    - `recall`: the share of the scripted player's presses the expert made too, the same
      key or button within `window` decisions;
    - `aim_recall`: those, or, for a click, a move of the expert's within `window`
      decisions to within `aim` pixels of where the scripted player clicked (the expert
      moves first and presses at the next decision, and on a recording the click it
      asks for never comes, so its press is often not there to match);
    - `precision`: the share of the expert's presses the scripted player made.
    The expert's label does not happen, so on a recording it asks for the same press
    until the scripted player makes it: a run of the same press counts once."""
    until = len(recorded) if until is None else until
    track = pointer_track(recorded)
    theirs = own_presses(recorded, until)
    labelled = np.where(valid[:, None, None], expert, 0)
    ours = [p for p in presses(labelled, track) if p[0] < until]
    # Runs of the same press in consecutive decisions, each counted once.
    runs = []
    for d, k, _ in ours:
        if runs and runs[-1][1] == k and runs[-1][0][-1] == d - 1:
            runs[-1][0].append(d)
        else:
            runs.append(([d], k))
    aims = moves(labelled)

    def same(d, k, b):
        return any(k == kb and abs(d - db) <= window for db, kb, _ in b)

    def aimed(d, k, at):
        from .actions import VOCAB

        if VOCAB[k]["kind"] != "button" or at is None:
            return False
        return any(abs(d - db) <= window and near(to, at, aim) for db, to in aims)

    recalled = [same(d, k, ours) for d, k, _ in theirs]
    precise = sum(any(same(d, k, theirs) for d in run) for run, k in runs)
    result = rates(len(theirs), len(runs), sum(recalled), precise)
    hit = sum(r or aimed(d, k, at) for r, (d, k, at) in zip(recalled, theirs, strict=True))
    result.update(aimed=hit, aim_recall=round(hit / len(theirs), 3) if theirs else None)
    return result


def missed(recorded, expert, valid, until=None, window=3, aim=INTENT_PX):
    """The scripted player's presses the expert neither made nor aimed at (agreement), as
    "decision:key@x,y" names (pixels)."""
    from .actions import VOCAB

    until = len(recorded) if until is None else until
    labelled = np.where(valid[:, None, None], expert, 0)
    ours = presses(labelled, pointer_track(recorded))
    aims = moves(labelled)
    out = []
    for d, k, at in own_presses(recorded, until):
        if any(k == kb and abs(d - db) <= window for db, kb, _ in ours):
            continue
        event = VOCAB[k]
        if event["kind"] == "button" and at is not None:
            if any(abs(d - db) <= window and near(to, at, aim) for db, to in aims):
                continue
        where = f"@{round(at[0] * 1919)},{round(at[1] * 1079)}" if at else ""
        out.append(f"{d}:{event['kind'][0]}{event.get('vk', event.get('button'))}{where}")
    return out


def rates(theirs, ours, recalled, precise):
    return {
        "scripted_presses": theirs,
        "expert_presses": ours,
        "recalled": recalled,
        "precise": precise,
        "recall": round(recalled / theirs, 3) if theirs else None,
        "precision": round(precise / ours, 3) if ours else None,
    }


def validate(recording, arrays=False, **options):
    """The expert against the scripted player on the scripted player's own recording
    (a drill or a game): agreement over its setup, up to a second after its `run` order."""
    recording = Path(recording)
    labels = label_recording(recording, seconds=SETUP_SECONDS, **options)
    summary = summarise(recording, labels)
    decisions, expert, valid, phase = (
        labels[k] for k in ("decisions", "actions", "valid", "phase")
    )
    manifest = json.loads((recording / "manifest.json").read_text())
    lines = (recording / "frames.jsonl").read_text().splitlines()
    until = None
    for order in manifest.get("orders") or []:
        if order.get("order") == "run":
            frame = min(int(order["frame"]), len(lines) - 1)
            ran = json.loads(lines[frame])["t_ns"]
            until = int(np.searchsorted(decisions, ran)) + 5
    recorded = recorded_actions(recording, decisions)
    result = {**summary, **agreement(recorded, expert, valid, until), "until": until}
    result["missed"] = missed(recorded, expert, valid, until)
    if arrays:
        result["arrays"] = (decisions, recorded, expert, valid, phase)
    return result


def recordings_in(paths):
    """Recording folders: each path that is one, and those inside each that is not."""
    found = []
    for path in map(Path, paths):
        if (path / "manifest.json").exists():
            found.append(path)
        else:
            found += sorted(p.parent for p in path.glob("*/manifest.json"))
    return found


def _each(work, items, jobs):
    """work(item) for each item, in `jobs` processes (1: here), as (item, result or the
    error's text) in order."""
    if jobs <= 1:
        for item in items:
            try:
                yield item, work(item)
            except Exception as error:  # noqa: BLE001 - reported; the next one.
                yield item, f"{type(error).__name__}: {error}"
        return
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(jobs) as pool:
        futures = [(item, pool.submit(work, item)) for item in items]
        for item, future in futures:
            try:
                yield item, future.result()
            except Exception as error:  # noqa: BLE001 - reported; the next one.
                yield item, f"{type(error).__name__}: {error}"


def label_all(paths, force=False, jobs=1, **options):
    """relabel every complete practice game among `paths` that has no labels yet (all,
    with `force`), in `jobs` processes; a game that cannot be labelled is reported, not
    fatal."""
    from functools import partial

    wanted, skipped = [], 0
    for recording in recordings_in(paths):
        manifest = json.loads((recording / "manifest.json").read_text())
        fresh = force or not (recording / DAGGER_LABELS).exists()
        if manifest.get("complete") and manifest.get("source") == "policy" and fresh:
            wanted.append(recording)
        else:
            skipped += 1
    done, failed = [], []
    for recording, result in _each(partial(relabel, **options), wanted, jobs):
        if isinstance(result, str):
            failed.append({"recording": recording.name, "error": result})
            log.warning("%s: %s", recording.name, result)
            continue
        done.append(result)
        log.info("%s: %d of %d decisions labelled", recording.name, result["labelled"],
                 result["decisions"])  # fmt: skip
    phases = {}
    for summary in done:
        for phase, n in summary["phases"].items():
            phases[phase] = phases.get(phase, 0) + n
    return {"labelled": len(done), "skipped": skipped, "failed": failed,
            "decisions": sum(d["decisions"] for d in done),
            "labels": sum(d["labelled"] for d in done), "phases": phases}  # fmt: skip


def aggregate(folder, paths, holdout=0.1):
    """DAgger's data set: a link in `folder` to every practice game among `paths` that has
    the expert's labels, and a splits.json that trains on them (train-bc --dagger-data)
    but for about a `holdout` share, chosen by name, kept for validation: how well a
    policy acts as the expert would in states a learner led the game into. Links already
    there keep their split; returns how many were added and how many it holds."""
    import zlib

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    splits_path = folder / "splits.json"
    splits = json.loads(splits_path.read_text()) if splits_path.exists() else {}
    added = 0
    for recording in recordings_in(paths):
        if not (recording / DAGGER_LABELS).exists():
            continue
        link = folder / recording.name
        if not link.exists():
            link_folder(recording.resolve(), link)
            added += 1
        if recording.name not in splits:
            drawn = zlib.crc32(recording.name.encode()) % 1000 / 1000
            splits[recording.name] = "validation" if drawn < holdout else "train"
    splits_path.write_text(json.dumps(dict(sorted(splits.items())), indent=2))
    held = sum(v == "validation" for v in splits.values())
    return {"added": added, "games": len(splits), "validation": held}


def link_folder(target, link):
    """A directory junction on Windows (no privilege needed), a symlink elsewhere."""
    import os

    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def check_all(paths, jobs=1, **options):
    """validate over every scripted recording among `paths`, in `jobs` processes, and the
    totals."""
    from functools import partial

    wanted = []
    for recording in recordings_in(paths):
        manifest = json.loads((recording / "manifest.json").read_text())
        if manifest.get("complete") and manifest.get("source") == "scripted":
            wanted.append(recording)
    games, failed = [], []
    for recording, result in _each(partial(validate, **options), wanted, jobs):
        if isinstance(result, str):
            failed.append({"recording": recording.name, "error": result})
            continue
        games.append(result)
        log.info("%s: %s", recording.name,
                 {k: result[k] for k in ("recall", "aim_recall", "precision")})  # fmt: skip
    total = rates(*(sum(g[k] for g in games) for k in
                    ("scripted_presses", "expert_presses", "recalled", "precise")))  # fmt: skip
    aimed = sum(g["aimed"] for g in games)
    total.update(aimed=aimed, aim_recall=round(aimed / max(1, total["scripted_presses"]), 3))
    return {"games": games, "failed": failed, "total": total}


def expert_nll(checkpoint, paths, *, model_path=None, window=64):
    """How well a policy acts as the expert would, in states learners led the game into:
    its negative log-likelihood of the expert's labels (DAGGER_LABELS) over every labelled
    decision of the practice games among `paths` (outside a coach's spans), each game
    played through the policy from its start with its memory carried, as it plays live.
    Games a DAgger aggregate held out (its validation split) are ones no checkpoint trained
    on. Returns the mean over all labelled decisions (`nll`), over those that press a key
    or a button (`nll_presses`), and per game."""
    import torch
    from torch.utils.data import default_collate

    from .dataset import _Stream, batch_to_device, camera_keys_dropped, cover_starts, session_labels
    from .models import reads_clip
    from .runner import load_policy
    from .train import presses as pressing

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = json.loads(Path(checkpoint).with_suffix(".json").read_text())["config"]
    policy, _, digest = load_policy(checkpoint, model_path, device)
    policy.eval()
    clips = reads_clip(policy.encoder)
    autocast = {"device_type": device, "dtype": torch.bfloat16}
    games, every, pressed = [], [], []
    for recording in recordings_in(paths):
        if not (recording / DAGGER_LABELS).exists():
            continue
        manifest = json.loads((recording / "manifest.json").read_text())
        keys = tuple(config.get("drop_keys") or ())
        labels = session_labels(
            recording,
            sources=(manifest["source"],),
            lead_in=config.get("lead_in"),
            drop_keys=keys + camera_keys_dropped(manifest, config.get("camera_since")),
            drop_parking=bool(config.get("drop_parking")),
            held_previous=bool(config.get("held_previous")),
            dagger=1.0,
        )
        if config.get("tower_cache"):
            from .tower_cache import tower_paths

            found = tower_paths(config["tower_cache"], recording)
            if found is not None:
                labels["tower"] = found
        with np.load(recording / DAGGER_LABELS) as stored:
            grid, labelled = stored["decisions"], stored["valid"]
        at = np.clip(np.searchsorted(grid, labels["decisions"]), 0, len(grid) - 1)
        expert = labelled[at] & (grid[at] == labels["decisions"]) & (labels["weight"] > 0)
        wanted = set(np.flatnonzero(expert & labels["valid"] & labels["readable"]).tolist())
        if not wanted:
            continue
        scores, acts = [], []
        stream = _Stream(
            labels, window, 0, device, starts=cover_starts(labels, window), clips=clips
        )
        hidden, done_until, last = None, 0, max(wanted)
        try:
            with torch.no_grad():
                while (done := stream.advance()) is not None and done_until <= last:
                    for piece in done:
                        start = piece.pop("start")
                        batch = batch_to_device(default_collate([piece]), device)
                        with torch.autocast(**autocast):
                            tower = None
                            if "tower_grid" in batch:
                                tower = (batch["tower_summary"], batch["tower_grid"])
                            summary, cells, centre = policy.perceive_window(
                                batch.get("clips"), batch["quadrants"], batch["fovea"],
                                tower=tower,
                            )  # fmt: skip
                            if hidden is None:
                                hidden = summary.new_zeros(1, policy.memory_dim)
                            for t in range(summary.shape[1]):
                                if start + t < done_until:
                                    continue
                                hidden, _ = policy.recall(
                                    summary[:, t], cells[:, t], centre[:, t],
                                    batch["previous"][:, t], batch["speed"][:, t], hidden,
                                    batch["quadrants"].dtype,
                                )  # fmt: skip
                                if start + t in wanted:
                                    action = batch["actions"][:, t]
                                    logp = policy.actor(hidden, cells[:, t], action)[1]
                                    scores.append(-float(logp.float().sum()))
                                    acts.append(bool(pressing(action.cpu())[0]))
                        done_until = max(done_until, start + summary.shape[1])
        finally:
            stream.close()
        every += scores
        pressed += [x for x, a in zip(scores, acts, strict=True) if a]
        games.append({"recording": recording.name, "decisions": len(scores),
                      "nll": round(float(np.mean(scores)), 4)})  # fmt: skip
        log.info("%s: %s", recording.name, games[-1])
    return {
        "checkpoint": str(checkpoint),
        "digest": digest,
        "decisions": len(every),
        "nll": round(float(np.mean(every)), 4) if every else None,
        "presses": len(pressed),
        "nll_presses": round(float(np.mean(pressed)), 4) if pressed else None,
        "games": games,
    }
