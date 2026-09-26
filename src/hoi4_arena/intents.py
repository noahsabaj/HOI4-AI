"""Intents: what a player means to do, between the strategy and the clicks that do it.

A policy that goes straight from the screen to the mouse must learn long, precise click
sequences (forming an army takes a shift+click on an alert and a click on a + that glows
only then; a front line takes Z and a click on the enemy's side of the border) at the
same time as when to attack. Most of the learned player's failures were in the first.
Here the two are split: a high-level policy reads the screen and picks an intent every
second or so, and skills turn an intent into clicks and keys. Pixels in, mouse out still
holds end to end.

The vocabulary is the scripted player's (scripted.Planner) and the recorder's camera's
(ai_games.camera), so every recorded scripted game can be relabelled into intents
automatically (relabel): its inputs are cut into gestures (a click, a drag, a key tap, a
zoom), and the gestures into skill segments by the planner's own procedures. The same
vocabulary serves the privileged teacher and the Claude strategist.

Arguments are optional. None means the hand chooses from the screen, as the scripted
hand does, so the relabelled games stay valid; a strategist that knows better (which
front to hold, where to push) may name them.
"""

from __future__ import annotations

import functools
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# What the high-level policy chooses between. `wait` leaves the hand to its default, the
# camera director, which watches the front and clears popups.
INTENTS = (
    "wait",
    "form_army",
    "assign_general",
    "draw_front",
    "draw_offensive",
    "execute",
    "set_law",
    "redraw",
    "pause",
    "run",
    "reinforce",
    "recruit",
    "camera",
    "popup",
)
# The optional arguments each intent takes; None (or absent) is the hand's choice.
#
# Arena states are numbered 1 to 16, Blue's 1 to 8 and Red's 9 to 16 (mapgen.state_cell),
# as in the arena log. `army` is an index into the army bar, 0 the first army; `general`
# an index into the commander list.
ARGS = {
    "wait": ("seconds",),
    "form_army": ("divisions",),
    "assign_general": ("army", "general"),
    # `front_state`: the front is drawn where the border meets this state (the stretch of
    # it there, when water splits the border), None for the whole border; `stance`: push
    # or hold, what the front is for; `at`: the clicked point as fractions of the arena's
    # land box.
    "draw_front": ("army", "front_state", "stance", "at"),
    # `attack`: broad, near or deep; `target_state`: the state to push toward, None for
    # the hand's choice.
    "draw_offensive": ("army", "attack", "target_state", "line"),
    "execute": ("army",),
    "set_law": ("law",),
    # Every order deleted, then a new front and offensive; `guard` when the front is drawn
    # round the enemy's incursion into the home land.
    "redraw": ("army", "guard", "attack", "target_state", "front_state"),
    "pause": ("paused",),
    "run": ("speed",),
    "reinforce": (),
    "recruit": ("slots",),
    # `kind`: overview (fully out), front (in on it), inspect (a closer look), pan, sweep
    # or rest; `at`: where, as screen fractions.
    "camera": ("kind", "at"),
    "popup": ("at",),
}
STATES = tuple(range(1, 17))
# The values an argument may take, where it is one of a few.
CHOICES = {
    "attack": ("broad", "near", "deep"),
    "law": ("limited", "extensive", "service", "all_adults"),
    "stance": ("push", "hold"),
    "target_state": STATES,
    "front_state": STATES,
    "kind": ("overview", "front", "inspect", "out", "pan", "look", "sweep", "rest"),
    "speed": (1, 2, 3, 4, 5),
}
# What the hand does: a segment of inputs, one skill each. A redraw is clear_orders then
# draw_front and draw_offensive; select_army and survey (the planner's full view of the
# map) are folded into the skill they serve. `console` is the harness's, never a
# player's; `unknown` is what no rule explains.
SKILLS = (
    "form_army",
    "reinforce",
    "assign_general",
    "draw_front",
    "draw_offensive",
    "execute",
    "clear_orders",
    "set_law",
    "recruit",
    "pause",
    "run",
    "survey",
    "camera",
    "popup",
    "console",
    "unknown",
)
# The planner's skills, as against the camera's and the harness's.
PLANNER_SKILLS = (
    "form_army",
    "reinforce",
    "assign_general",
    "draw_front",
    "draw_offensive",
    "execute",
    "clear_orders",
    "set_law",
    "recruit",
    "pause",
    "run",
    "survey",
)
# The high-level intent each skill serves (clear_orders begins a redraw, and the front and
# offensive after it in the same burst join it: group_redraws).
SKILL_INTENT = {
    "form_army": "form_army",
    "reinforce": "reinforce",
    "assign_general": "assign_general",
    "draw_front": "draw_front",
    "draw_offensive": "draw_offensive",
    "execute": "execute",
    "clear_orders": "redraw",
    "set_law": "set_law",
    "recruit": "recruit",
    "pause": "pause",
    "run": "run",
    "survey": "camera",
    "camera": "camera",
    "popup": "popup",
    "console": "wait",
    "unknown": "wait",
}
# The manifest's order kinds (scripted.Planner.order), and the skill whose segment ends
# just before each is stamped.
ORDER_SKILL = {
    "army": "form_army",
    "general": "assign_general",
    "front": "draw_front",
    "offensive": "draw_offensive",
    "run": "run",
    "law": "set_law",
    "clear": "clear_orders",
    "activate": "execute",
    "recruit": "recruit",
    "reinforce": "reinforce",
    "pause": "pause",
}


@dataclass
class Intent:
    """One intent and its arguments (ARGS; None or absent: the hand's choice)."""

    name: str
    args: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name not in INTENTS:
            raise ValueError(f"unknown intent {self.name!r}")
        unknown = set(self.args) - set(ARGS[self.name])
        if unknown:
            raise ValueError(f"{self.name} takes no {sorted(unknown)}")
        for key, allowed in CHOICES.items():
            value = self.args.get(key)
            if value is not None and value not in allowed:
                raise ValueError(
                    f"{self.name}: {key} must be one of {list(allowed)}, not {value!r}"
                )
        for key in ("army", "general"):
            value = self.args.get(key)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{self.name}: {key} must be an index from 0, not {value!r}")

    def arg(self, key):
        return self.args.get(key)

    def to_json(self):
        return {"intent": self.name, **{k: v for k, v in self.args.items() if v is not None}}

    @classmethod
    def from_json(cls, data):
        data = dict(data)
        return cls(data.pop("intent"), data)


# Where the scripted player's buttons sit at 1080p, as screen fractions (scripted.py).
def _at(x, y):
    return x / 1920, y / 1080


ARMY_CARD = (947 / 1920, 1010 / 1080)
COMMANDER_SLOT, FIRST_COMMANDER = _at(30, 140), _at(950, 352)
CONFIRM, CANCEL, CLOSE_LIST = _at(1054, 677), _at(884, 677), _at(1005, 100)
LAW_SLOT = _at(60, 593)
LAW_X, LAW_FIRST_Y, LAW_ROW = 703 / 1920, 322, 74
LAW_NAMES = ("limited", "extensive", "service", "all_adults")
SPEED_UP = (1789 / 1920, 20 / 1080)
TRAIN, DEPLOY_AT, ADD_UNIT, DROP_LINE = _at(742, 276), _at(220, 433), _at(386, 433), _at(480, 435)
# The execute arrow and the bin are found by template; they sit about here.
ARROW_BOX = (930 / 1920, 935 / 1080, 1010 / 1920, 975 / 1080)
TRASH_BOX = (1235 / 1920, 860 / 1080, 1295 / 1920, 910 / 1080)
# Where the planner parks the pointer to read the screen (dataset.PARKING).
PARK_CENTRE, PARK_TOP, PARK_LOW = (0.5, 0.5), (0.65, 0.012), (0.5, 0.75)
# The army bar, where the create-army + shows, and the top bar, where the alert is.
ARMY_BAR, TOP_BAR = 0.88, 0.08
SHIFT, SPACE, GRAVE = 0x10, 0x20, 0xC0
FRONT_KEY, OFFENSIVE_KEY, POLITICS_KEY, RECRUIT_KEY = 0x5A, 0x58, 0x51, 0x55
ARROWS = (0x25, 0x26, 0x27, 0x28)
# recentre zooms out by ZOOM_MAX + 4 notches (30); the camera's own zooms are a dozen at
# most.
FULL_ZOOM = 20
# A planner step's gestures come at most this far apart (its longest sleep is 1.2 s, and
# a screen search can take a second more on the second PC).
STEP_GAP = 3.0
NEAR = 0.012


def _near(point, anchor, tolerance=NEAR):
    return abs(point[0] - anchor[0]) <= tolerance and abs(point[1] - anchor[1]) <= tolerance * 1.8


def _inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


@dataclass
class Gesture:
    """A unit of input: a click, a drag, a key tap, a zoom or a move alone."""

    kind: str  # click, drag, tap, zoom, move
    first: int  # index of its first event
    last: int  # index of its last event
    t0: int
    t1: int
    at: tuple = (0.0, 0.0)
    button: int = 0
    shift: bool = False
    vk: int = 0
    notches: int = 0
    path: list = field(default_factory=list)

    @property
    def seconds(self):
        return (self.t1 - self.t0) / 1e9


def gestures(events):
    """Cut `events` ({t_ns, event} in time order) into gestures.

    A move straight before a press or a zoom joins it, as where it happens; moves while a
    button is down make a drag; a shift held round a click makes it a shift+click.
    """
    found = []
    cursor = (0.5, 0.5)
    shift = False
    pressed = None  # (button, index, t, at, path)
    held = {}  # vk -> (index, t)
    for i, item in enumerate(events):
        event, t = item["event"], int(item["t_ns"])
        kind = event.get("kind")
        if kind == "move":
            cursor = (float(event["x"]), float(event["y"]))
            if pressed is not None:
                pressed[4].append(cursor)
                continue
            found.append(Gesture("move", i, i, t, t, at=cursor))
        elif kind == "button":
            if event["down"]:
                pressed = (event["button"], i, t, cursor, [cursor])
                continue
            if pressed is None:
                continue
            button, first, t0, at, path = pressed
            pressed = None
            first, t0 = _absorb_move(found, first, t0, at)
            if len(path) > 1:
                found.append(Gesture("drag", first, i, t0, t, at=at, button=button, path=path))
            else:
                found.append(Gesture("click", first, i, t0, t, at=at, button=button, shift=shift))
        elif kind == "key":
            vk = event["vk"]
            if vk == SHIFT:
                if event["down"]:
                    shift = True
                    held[vk] = (i, t)
                else:
                    shift = False
                    held.pop(vk, None)
                    # The shift's own events join the click it wrapped.
                    if found and found[-1].kind == "click" and found[-1].shift:
                        found[-1].last, found[-1].t1 = i, t
                continue
            if event["down"]:
                held[vk] = (i, t)
                continue
            first, t0 = held.pop(vk, (i, t))
            found.append(Gesture("tap", first, i, t0, t, at=cursor, vk=vk))
        elif kind == "wheel":
            step = 1 if event.get("delta", 0) > 0 else -1
            last = found[-1] if found else None
            if (
                last is not None
                and last.kind == "zoom"
                and np.sign(last.notches) == step
                and last.last == i - 1
            ):
                last.notches += step
                last.last, last.t1 = i, t
                continue
            first, t0 = _absorb_move(found, i, t, cursor)
            found.append(Gesture("zoom", first, i, t0, t, at=cursor, notches=step))
    # A shift+click's shift went down before the click's move: take its event in.
    return found


def _absorb_move(found, first, t0, at):
    """Take the move just before a press or zoom into it, when it went to the same place."""
    if found and found[-1].kind == "move" and found[-1].last == first - 1 and found[-1].at == at:
        move = found.pop()
        return move.first, move.t0
    if found and found[-1].kind == "move" and found[-1].at == at and found[-1].last == first - 2:
        # A shift pressed between the move and the click.
        move = found.pop()
        return move.first, move.t0
    return first, t0


@dataclass
class Segment:
    """A run of gestures that serve one skill."""

    skill: str
    first: int  # gesture indices, inclusive
    last: int
    args: dict = field(default_factory=dict)
    intent: str = ""
    group: int = -1  # segments of one redraw share it

    def __post_init__(self):
        if not self.intent:
            self.intent = SKILL_INTENT[self.skill]


def _is_park(g, point=None):
    if g.kind != "move":
        return False
    points = (PARK_CENTRE, PARK_TOP, PARK_LOW) if point is None else (point,)
    return any(abs(g.at[0] - p[0]) < 1e-6 and abs(g.at[1] - p[1]) < 1e-6 for p in points)


def _card_click(g):
    return g.kind == "click" and g.button == 0 and not g.shift and _near(g.at, ARMY_CARD, 0.01)


def _law_row(y):
    index = round((y * 1080 - LAW_FIRST_Y) / LAW_ROW)
    return LAW_NAMES[index] if 0 <= index < len(LAW_NAMES) else None


class _Parser:
    """Segments of a recording's gestures, by the planner's procedures (scripted.Planner)
    and, for the rest, the camera's (ai_games.camera)."""

    def __init__(self, found):
        self.g = found
        self.segments = []

    def close(self, i, j):
        """Whether gesture j follows gesture i within a planner step's gap."""
        return (
            0 <= i < len(self.g)
            and j < len(self.g)
            and (self.g[j].t0 - self.g[i].t1) / 1e9 <= STEP_GAP
        )

    def skip_parks(self, i):
        while i < len(self.g) and _is_park(self.g[i]) and self.close(i, i + 1):
            i += 1
        return i

    def run(self):
        i = 0
        while i < len(self.g):
            matched = self.match(i)
            if matched is None:
                self.segments.append(self.camera(i))
                i += 1
            else:
                self.segments.append(matched)
                i = matched.last + 1
        self.fold()
        return self.segments

    # Each matcher: a Segment starting at gesture i, or None.
    def match(self, i):
        g = self.g
        start = i
        # A survey (the planner's full view of the map), then what it served.
        survey_end = self.survey(i)
        if survey_end is not None:
            j = self.skip_parks(survey_end + 1)
            if j < len(g) and self.close(survey_end, j):
                if _card_click(g[j]):
                    j = self.skip_parks(j + 1)
                inner = self.procedure(j) if j < len(g) and self.close(j - 1, j) else None
                if inner is not None and inner.skill in ("draw_front", "draw_offensive", "recruit"):
                    inner.first = start
                    return inner
            return Segment("survey", start, survey_end, {"kind": "overview", "planner": True})
        # Parking and selecting the army lead into a procedure.
        j = i
        if _is_park(g[j]) or _card_click(g[j]):
            while j < len(g) and (_is_park(g[j]) or _card_click(g[j])) and self.close(j, j + 1):
                j += 1
            if j < len(g) and j > i:
                inner = self.procedure(j)
                if inner is not None:
                    inner.first = start
                    return inner
            if _card_click(g[i]):
                # A selection with nothing after it: activate's look at the arrow, whose
                # button was not found (the plan already executing) or assign_general's.
                end = i
                while end + 1 < len(g) and _is_park(g[end + 1]) and self.close(end, end + 1):
                    end += 1
                return Segment("execute", start, end, {"checked": True})
            return None
        return self.procedure(i)

    def survey(self, i):
        """The last gesture of a planner's look at the whole map starting at i, or None:
        a full zoom out, the arrow taps that centre it, and the pointer parked on the top
        bar. The camera's own overviews park it on the front instead."""
        g = self.g
        if not (g[i].kind == "zoom" and g[i].notches <= -FULL_ZOOM):
            return None
        j = i + 1
        while j < len(g) and g[j].kind == "tap" and g[j].vk in ARROWS and self.close(j - 1, j):
            j += 1
        if j < len(g) and _is_park(g[j], PARK_TOP) and self.close(j - 1, j):
            return j
        return None

    def procedure(self, i):
        g = self.g
        x = g[i]
        # Forming an army, or reinforcing it: shift+click on the alert in the top bar.
        if x.kind == "click" and x.shift and x.at[1] < TOP_BAR:
            j = self.skip_parks(i + 1)
            if j < len(g) and self.close(i, j) and g[j].kind == "click":
                if g[j].button == 1 and _near(g[j].at, ARMY_CARD, 0.01):
                    return Segment("reinforce", i, self.trail(j))
                if g[j].button == 0 and g[j].at[1] > ARMY_BAR:
                    return Segment("form_army", i, self.trail(j))
            return Segment("form_army", i, i, {"plus": False})
        # A general: the panel's portrait, then the first in the list.
        if x.kind == "click" and _near(x.at, COMMANDER_SLOT):
            j = i + 1
            if (
                j < len(g)
                and self.close(i, j)
                and g[j].kind == "click"
                and _near(g[j].at, FIRST_COMMANDER)
            ):
                return Segment("assign_general", i, j)
            return Segment("assign_general", i, i)
        # A front line: Z, a click on the border, and Z again to put the tool away.
        if x.kind == "tap" and x.vk == FRONT_KEY:
            return self.front(i)
        # An offensive: X, then a right-drag.
        if x.kind == "tap" and x.vk == OFFENSIVE_KEY:
            j = i + 1
            if j < len(g) and self.close(i, j) and g[j].kind == "drag" and g[j].button == 1:
                first, last = g[j].path[0], g[j].path[-1]
                return Segment(
                    "draw_offensive", i, j, {"start": first, "end": last, "points": len(g[j].path)}
                )
            return Segment("draw_offensive", i, i, {"drawn": False})
        # Executing: a click on the arrow above the army card.
        if x.kind == "click" and x.button == 0 and _inside(x.at, ARROW_BOX):
            return Segment("execute", i, self.trail(i))
        # Deleting every order: a right-click on the bin, then OK.
        if x.kind == "click" and x.button == 1 and _inside(x.at, TRASH_BOX):
            j = i + 1
            if j < len(g) and self.close(i, j) and g[j].kind == "click" and _near(g[j].at, CONFIRM):
                return Segment("clear_orders", i, j)
            return Segment("clear_orders", i, i, {"confirmed": False})
        # A law: Q opens the political screen, clicks choose, Q closes it.
        if x.kind == "tap" and x.vk == POLITICS_KEY:
            return self.law(i)
        if x.kind == "tap" and x.vk == RECRUIT_KEY:
            return self.panel(i, RECRUIT_KEY, "recruit")
        # Running the game: the speed + clicks, then space.
        if x.kind == "click" and _near(x.at, SPEED_UP):
            j = i
            while (
                j + 1 < len(g)
                and self.close(j, j + 1)
                and (
                    (g[j + 1].kind == "click" and _near(g[j + 1].at, SPEED_UP))
                    or _is_park(g[j + 1], PARK_LOW)
                    or (g[j + 1].kind == "tap" and g[j + 1].vk == SPACE)
                )
            ):
                j += 1
            clicks = sum(1 for k in range(i, j + 1) if g[k].kind == "click")
            return Segment("run", i, j, {"speed_clicks": clicks})
        # Pausing or unpausing: space, with the pointer parked.
        if x.kind == "tap" and x.vk == SPACE:
            return Segment("pause", i, i)
        if x.kind == "tap" and x.vk == GRAVE:
            return self.console(i)
        return None

    def trail(self, j):
        """Past gesture j, the parking moves that belong to its step (the look after)."""
        while j + 1 < len(self.g) and _is_park(self.g[j + 1]) and self.close(j, j + 1):
            # A park that leads into another procedure is that procedure's.
            k = self.skip_parks(j + 1)
            if k < len(self.g) and self.close(k - 1, k) and self.procedure_start(k):
                break
            j += 1
        return j

    def procedure_start(self, k):
        g = self.g[k]
        return (
            (
                g.kind == "tap"
                and g.vk in (FRONT_KEY, OFFENSIVE_KEY, POLITICS_KEY, RECRUIT_KEY, SPACE)
            )
            or (g.kind == "click" and (g.shift or _card_click(g) or _inside(g.at, ARROW_BOX)))
            or (g.kind == "zoom" and g.notches <= -FULL_ZOOM)
        )

    def front(self, i):
        """Z taps and the clicks on the map between them, with the retries' looks and
        selections, as one draw_front."""
        g = self.g
        j, clicks, points = i, 0, []
        while j + 1 < len(g) and self.close(j, j + 1):
            nxt = g[j + 1]
            if nxt.kind == "tap" and nxt.vk == FRONT_KEY:
                j += 1
            elif nxt.kind == "click" and nxt.button == 0 and not nxt.shift and not _card_click(nxt):
                if _near(nxt.at, COMMANDER_SLOT) or _inside(nxt.at, ARROW_BOX):
                    break
                j += 1
                clicks += 1
                points.append(nxt.at)
            elif _card_click(nxt) or _is_park(nxt):
                # A retry selects the army again and looks afresh; only if a Z follows.
                k = j + 1
                while k < len(g) and (_card_click(g[k]) or _is_park(g[k])) and self.close(k - 1, k):
                    k += 1
                survey = self.survey(k) if k < len(g) else None
                if survey is not None:
                    k = self.skip_parks(survey + 1)
                    if k < len(g) and _card_click(g[k]):
                        k = self.skip_parks(k + 1)
                if (
                    k < len(g)
                    and self.close(k - 1, k)
                    and g[k].kind == "tap"
                    and g[k].vk == FRONT_KEY
                ):
                    j = k
                else:
                    break
            elif nxt.kind == "zoom" and nxt.notches <= -FULL_ZOOM:
                survey = self.survey(j + 1)
                if survey is None:
                    break
                k = self.skip_parks(survey + 1)
                if k < len(g) and _card_click(g[k]):
                    k = self.skip_parks(k + 1)
                if (
                    k < len(g)
                    and self.close(k - 1, k)
                    and g[k].kind == "tap"
                    and g[k].vk == FRONT_KEY
                ):
                    j = k
                else:
                    break
            else:
                break
        return Segment("draw_front", i, j, {"clicks": clicks, "points": points})

    def law(self, i):
        """From Q to the Q that closes the political screen; the law aimed at, and whether
        its confirmation was OK'd or cancelled."""
        g = self.g
        j, law, answer = i, None, None
        while j + 1 < len(g) and self.close(j, j + 1):
            nxt = g[j + 1]
            j += 1
            if nxt.kind == "tap" and nxt.vk == POLITICS_KEY:
                break
            if nxt.kind == "click" and abs(nxt.at[0] - LAW_X) < 0.01:
                law = _law_row(nxt.at[1]) or law
            elif nxt.kind == "click" and _near(nxt.at, CONFIRM):
                answer = "ok"
            elif nxt.kind == "click" and _near(nxt.at, CANCEL):
                answer = "cancel"
            elif nxt.kind in ("zoom", "drag") or (nxt.kind == "tap" and nxt.vk != POLITICS_KEY):
                j -= 1
                break
        return Segment("set_law", i, j, {"law": law, "answer": answer})

    def panel(self, i, key, skill):
        g = self.g
        j = i
        while j + 1 < len(g) and self.close(j, j + 1):
            j += 1
            if g[j].kind == "tap" and g[j].vk == key:
                break
        return Segment(skill, i, j)

    def console(self, i):
        g = self.g
        j = i
        while j + 1 < len(g) and self.close(j, j + 1) and g[j + 1].kind == "tap":
            j += 1
            if g[j].vk == GRAVE:
                break
        return Segment("console", i, j)

    def camera(self, i):
        """What the camera director did with gesture i (ai_games.camera)."""
        x = self.g[i]
        if x.kind == "zoom":
            if x.notches <= -FULL_ZOOM:
                return Segment("camera", i, i, {"kind": "overview", "at": x.at})
            return Segment(
                "camera", i, i, {"kind": "front" if x.notches > 0 else "out", "at": x.at}
            )
        if x.kind == "tap" and x.vk in ARROWS:
            return Segment("camera", i, i, {"kind": "pan", "vk": x.vk})
        if x.kind == "move":
            return Segment("camera", i, i, {"kind": "look", "at": x.at})
        if x.kind == "click" and x.button == 0 and not x.shift:
            # A popup's Ok: the camera clicks nothing else.
            return Segment("popup", i, i, {"at": x.at})
        return Segment("unknown", i, i)

    def fold(self):
        """Neighbouring camera gestures into one camera segment of the same kind; the
        camera's inspect (in, looks, out) into one; clear_orders with the front and
        offensive after it into one redraw."""
        merged = []
        for s in self.segments:
            last = merged[-1] if merged else None
            if (
                last is not None
                and last.skill == s.skill == "camera"
                and self._camera_joins(last, s)
            ):
                last.last = s.last
                if s.args.get("kind") == "out" and last.args.get("kind") in ("front", "inspect"):
                    last.args["kind"] = "inspect"
                continue
            if (
                last is not None
                and last.skill == s.skill == "draw_front"
                and self.close(last.last, s.first)
            ):
                # Another stretch of the border, or a retry (Planner.more_front).
                last.last = s.last
                last.args["clicks"] = last.args.get("clicks", 0) + s.args.get("clicks", 0)
                last.args.setdefault("points", []).extend(s.args.get("points", []))
                continue
            merged.append(s)
        group = 0
        for k, s in enumerate(merged):
            if s.skill != "clear_orders":
                continue
            group += 1
            s.group = group
            for later in merged[k + 1 :]:
                if later.skill in ("camera", "survey") and later.args.get("kind") == "look":
                    continue
                if later.skill in ("draw_front", "draw_offensive"):
                    later.group, later.intent = group, "redraw"
                    continue
                break
        self.segments = merged

    def _camera_joins(self, a, b):
        if not self.close(a.last, b.first):
            return False
        ka, kb = a.args.get("kind"), b.args.get("kind")
        if ka in ("front", "inspect") and kb in ("look", "out"):
            return True
        if ka == "look" and kb == "look":
            return True
        if ka == kb == "pan":
            return True
        if ka == "overview" and kb in ("pan", "look"):
            return True
        return False


@contextmanager
def doing(desk, skill):
    """Tag the inputs `desk` applies meanwhile with `skill` (ai_games.Logged keeps each
    event's tag in the recording). An outer tag wins: a survey inside a draw_front is the
    front's. A desktop that keeps no tags is left alone."""
    if "skill" not in getattr(desk, "__dict__", {}) or desk.skill is not None:
        yield
        return
    desk.skill = skill
    try:
        yield
    finally:
        desk.skill = None


def tagged(skill):
    """A Planner method whose inputs are `skill`'s (doing), its desktop the first
    argument."""

    def wrap(method):
        @functools.wraps(method)
        def inner(self, desk, *args, **kwargs):
            with doing(desk, skill):
                return method(self, desk, *args, **kwargs)

        return inner

    return wrap


def agreement(events, segments):
    """How far relabel's skills agree with the skills the recorder tagged each event with
    (doing; an untagged event is the camera's): (share agreeing, events compared,
    {(tagged, inferred): count} of the disagreements). None if nothing was tagged."""
    if not any("skill" in e for e in events):
        return None
    inferred = ["unknown"] * len(events)
    for s in segments:
        for k in range(s["first_event"], s["last_event"] + 1):
            inferred[k] = s["skill"]
    wrong = {}
    for e, guess in zip(events, inferred, strict=True):
        truth = e.get("skill") or "camera"
        if truth != guess:
            wrong[(truth, guess)] = wrong.get((truth, guess), 0) + 1
    return 1 - sum(wrong.values()) / len(events), len(events), wrong


def load_events(root):
    """A scripted recording's inputs ({t_ns, event}, in time order), its frame times and
    manifest."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
    key = "events" if manifest.get("source") == "human" else "scripted_events"
    events = [e for row in rows for e in row.get(key, []) if e.get("by") != "harness"]
    events.sort(key=lambda e: e["t_ns"])
    times = np.array([row["t_ns"] for row in rows], np.int64)
    return events, times, manifest


def relabel(events, manifest=None, times=None):
    """A recording's inputs as skill segments: a list of dicts with the skill, the intent
    it serves, the indices of its first and last event, its start and end (ns), its
    arguments, and for a planner skill the manifest order it ended in, if any. With the
    manifest, each segment also carries the intent's own arguments (`intent_args`, ARGS)
    as far as the recording tells them.
    """
    found = gestures(events)
    segments = _Parser(found).run() if found else []
    out = []
    for s in segments:
        first, last = found[s.first], found[s.last]
        out.append(
            {
                "skill": s.skill,
                "intent": s.intent,
                "first_event": first.first,
                "last_event": last.last,
                "t0": first.t0,
                "t1": last.t1,
                "args": _plain(s.args),
                "group": s.group,
            }
        )
    orders = []
    if manifest is not None and times is not None:
        match_orders(out, manifest, times)
        orders = order_times(manifest, times)
    intent_args(out, orders)
    return out


def intent_args(segments, orders=()):
    """Each segment's `intent_args`: what its intent was asked to do, from its inputs and
    the manifest's orders ((order, ns) pairs, order_times)."""
    running = False
    groups = {}
    for s in segments:
        if s["group"] > 0:
            groups.setdefault(s["group"], []).append(s)
    for s in segments:
        args = {}
        skill = s["skill"]
        if skill == "run":
            running = True
            args["speed"] = next(
                (o.get("speed") for o, t in orders if o.get("order") == "run" and t >= s["t0"]),
                None,
            )
        elif skill == "pause":
            # Space toggles: before the run it pauses nothing the policy chose.
            args["paused"] = running
            running = not running
        elif skill == "set_law":
            args["law"] = s["args"].get("law")
        elif skill in ("camera", "survey"):
            args["kind"] = s["args"].get("kind")
        elif skill == "draw_offensive" and s.get("order"):
            order = _order_at(orders, "offensive", s)
            if order is not None:
                args["attack"] = order.get("attack")
                states = order.get("target_states") or [order.get("target_state")]
                if len([x for x in states if x]) == 1:
                    args["target_state"] = states[0]
        s["intent_args"] = {k: v for k, v in args.items() if v is not None}
    for members in groups.values():
        t0, t1 = members[0]["t0"], members[-1]["t1"] + 6e9
        guard = any(o.get("order") == "guard" and t0 <= t <= t1 for o, t in orders)
        attack = next(
            (m["intent_args"].get("attack") for m in members if m["skill"] == "draw_offensive"),
            None,
        )
        for m in members:
            if m["intent"] == "redraw":
                m["intent_args"] = {"guard": guard, **({"attack": attack} if attack else {})}


def _order_at(orders, kind, segment, reach=6.0):
    for order, t in orders:
        if (
            order.get("order") == kind
            and segment["t0"] <= t + 5e8
            and t - segment["t1"] <= reach * 1e9
        ):
            return order
    return None


def _plain(args):
    return {
        k: (
            [list(p) if isinstance(p, tuple) else p for p in v]
            if isinstance(v, list)
            else list(v)
            if isinstance(v, tuple)
            else v
        )
        for k, v in args.items()
    }


def order_times(manifest, times):
    """(order, ns) for each manifest order: the time its frame was recorded."""
    out = []
    for order in manifest.get("orders") or []:
        frame = min(max(int(order["frame"]) - 1, 0), len(times) - 1)
        out.append((order, int(times[frame])))
    return out


def match_orders(segments, manifest, times, reach=6.0):
    """Mark each planner segment with the manifest order that stamped it: the order of its
    skill stamped after the segment began and within `reach` seconds of its end, taking
    each segment once (a front's other stretches join their draw_front).
    Returns (orders matched, orders that name a skill)."""
    matched = named = 0
    taken = set()
    for order, t in order_times(manifest, times):
        skill = ORDER_SKILL.get(order.get("order"))
        if skill is None:
            continue
        named += 1
        best = None
        # A front drawn on another stretch of the border is part of the same draw_front.
        again = bool(order.get("stretch"))
        for k, s in enumerate(segments):
            if (k in taken and not again) or s["skill"] != skill:
                continue
            # A stamp comes once the step is checked done, which may be before the inputs
            # that tidy up after it (the front tool put away, a panel closed); frames are
            # 0.2 s apart.
            if s["t0"] > t + 0.5e9:
                continue
            gap = max(0.0, (t - s["t1"]) / 1e9)
            if gap <= reach and (best is None or gap < best[1]):
                best = (k, gap)
        if best is not None:
            taken.add(best[0])
            segments[best[0]].setdefault("order", order.get("order"))
            segments[best[0]].setdefault("order_frame", order.get("frame"))
            matched += 1
    return matched, named


def coverage(segments, events_total):
    """How much of a recording the segments explain: the share of its events in a known
    skill, the share in a planner skill, and the events by skill."""
    by_skill = {}
    for s in segments:
        n = s["last_event"] - s["first_event"] + 1
        by_skill[s["skill"]] = by_skill.get(s["skill"], 0) + n
    unknown = by_skill.get("unknown", 0)
    planner = sum(by_skill.get(k, 0) for k in PLANNER_SKILLS)
    return {
        "events": events_total,
        "explained": (events_total - unknown) / events_total if events_total else math.nan,
        "planner_events": planner,
        "by_skill": by_skill,
    }


def decision_intents(segments, decisions, period_ns):
    """Per decision (its start, ns), the index into INTENTS and SKILLS of the segment in
    progress over its interval, and whether a segment starts inside it. Decisions in no
    segment are `wait` with skill -1."""
    intents = np.zeros(len(decisions), np.int64)
    skills = np.full(len(decisions), -1, np.int64)
    starts = np.zeros(len(decisions), bool)
    decisions = np.asarray(decisions, np.int64)
    for s in segments:
        lo = np.searchsorted(decisions, s["t0"] - period_ns, side="right")
        hi = np.searchsorted(decisions, s["t1"], side="right")
        if hi <= lo:
            continue
        intents[lo:hi] = INTENTS.index(s["intent"])
        skills[lo:hi] = SKILLS.index(s["skill"])
        starts[lo] = True
    return intents, skills, starts


def recording_report(root):
    """relabel() on one recording, with its coverage and how many of its orders a segment
    ended in."""
    events, times, manifest = load_events(root)
    segments = relabel(events, manifest, times)
    report = coverage(segments, len(events))
    report["orders_matched"], report["orders_named"] = match_orders(
        [dict(s) for s in segments], manifest, times
    )
    planner = [s for s in segments if s["skill"] in PLANNER_SKILLS and s["skill"] != "survey"]
    report["planner_segments"] = len(planner)
    report["planner_segments_ordered"] = sum(1 for s in planner if s.get("order"))
    agreed = agreement(events, segments)
    if agreed is not None:
        report["tag_agreement"], _, wrong = agreed
        report["tag_disagreements"] = {f"{a}->{b}": n for (a, b), n in wrong.items()}
    return segments, report


def folder_report(roots):
    """recording_report over recordings, summed: the share of their inputs relabel
    explains, the share of the planner's orders a segment of the right skill ends in,
    the share of planner segments that ended in an order (the rest are attempts, such as
    a law step without the political power for it), and the events and segments by skill.
    """
    total = {"recordings": 0, "events": 0, "unknown": 0, "orders_matched": 0, "orders_named": 0}
    total.update(planner_segments=0, planner_segments_ordered=0)
    events_by, segments_by, failed = {}, {}, []
    tagged_events = tagged_agree = 0
    for root in roots:
        try:
            segments, report = recording_report(root)
        except (OSError, ValueError, KeyError) as error:
            failed.append({"recording": str(root), "error": str(error)})
            continue
        total["recordings"] += 1
        total["events"] += report["events"]
        total["unknown"] += report["by_skill"].get("unknown", 0)
        for key in (
            "orders_matched",
            "orders_named",
            "planner_segments",
            "planner_segments_ordered",
        ):
            total[key] += report[key]
        for skill, n in report["by_skill"].items():
            events_by[skill] = events_by.get(skill, 0) + n
        for s in segments:
            segments_by[s["skill"]] = segments_by.get(s["skill"], 0) + 1
        if "tag_agreement" in report:
            tagged_events += report["events"]
            tagged_agree += report["tag_agreement"] * report["events"]
    events = max(total["events"], 1)
    return {
        **total,
        "explained": 1 - total["unknown"] / events,
        "planner_share": sum(events_by.get(k, 0) for k in PLANNER_SKILLS) / events,
        "orders_explained": total["orders_matched"] / max(total["orders_named"], 1),
        "segments_ordered": total["planner_segments_ordered"] / max(total["planner_segments"], 1),
        "tag_agreement": tagged_agree / tagged_events if tagged_events else None,
        "events_by_skill": dict(sorted(events_by.items(), key=lambda kv: -kv[1])),
        "segments_by_skill": dict(sorted(segments_by.items(), key=lambda kv: -kv[1])),
        "failed": failed,
    }
