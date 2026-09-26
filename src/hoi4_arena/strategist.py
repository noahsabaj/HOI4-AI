"""A strategist over the scripted player's hand: someone else decides, the script clicks.

The scripted player (scripted.Planner) plays a fixed plan with a few random or tuned
settings, and wins about 56% of its games against the game's AI; a student that copies it
is capped there. With a strategist the planner stops at decision points instead, pauses
the game, and asks: it writes a full view of the map and the game's numbers to a folder,
waits for a decision file, carries the decision out with its usual clicks, and runs on.
The strategist can be a person, Claude reading the screenshot, or a model.

Decision points: the start (army, general and front drawn, still paused), every
`next_days` game days (DECIDE_EVERY by default), and on events: the enemy takes one of the
player's states, the player takes one of the enemy's, or an attack stalls for STALL_DAYS.

A decision is JSON: intents in the shared vocabulary (intents.Intent, checked there) plus
plan-level settings:

    {"note": "why, in words (kept as a label)",
     "plan": {"conscription": "all_adults", "redraw": 30, "guard": 0.15,
              "depth": 0.33, "rows": [0.0, 0.5]},
     "intents": [{"intent": "set_law", "law": "all_adults"},
                 {"intent": "draw_offensive", "target_state": 12},
                 {"intent": "execute"}],
     "next_days": 45}

What the planner does with each: draw_front deletes every order and draws a front alone
(at `front_state`, if given), so the army holds and an attack halts; draw_offensive
deletes every order and draws the front and an offensive (toward `target_state`, or as
`attack` says), which later redraws keep; redraw draws the plan afresh now, with any
`guard`, `attack`, `target_state` or `front_state` given; execute, set_law, recruit,
reinforce, form_army and assign_general do what they say; wait does nothing. Pause, run,
camera and popup belong to the hand and are refused with a note. The plan settings `depth`
and `rows` (a band of the land's height, from 0 at its top to 1 at its bottom) shape a
broad offensive: how far it goes, and which part of the front pushes. There is one army.

The numbers come through a small reader (ArenaState, from the arena mod's daily log
lines), so a richer state channel can take its place without changing the planner.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from .arena_log import ENEMY, parse

# Game days between decisions, unless a decision says otherwise; the fewest game days
# between two decisions an event (a state changing hands) calls; and the game days an
# attack may go without a state changing hands before a decision is called.
DECIDE_EVERY, EVENT_GAP, STALL_DAYS = 45, 10, 40
# Seconds to wait for a decision before the game goes on as it was. (The plan settings a
# decision may change and the intents it may give are scripted.PLAN_KEYS and
# scripted.Planner.decide.)
TIMEOUT = 1200
START = datetime(1936, 1, 1)


def strategist_plan():
    """The plan a strategist starts from: the best plan's settings, but nothing happens on a
    timer that the strategist decides (no hold that ends by itself, no law until asked)."""
    return {
        "best": False,
        "variant": "strategist",
        "strategist": True,
        "conscription": "volunteer",
        "attack": "broad",
        "recruit": 0,
        "wait": None,
        "redraw": 30,
        "pause_redraw": True,
        "guard": 0.15,
    }


def day_number(date):
    """Game days since 1 January 1936 from a log date ("12:00, 1 January, 1936"), or None."""
    try:
        when = datetime.strptime(" ".join(date.split()), "%H:%M, %d %B, %Y")
    except (AttributeError, ValueError):
        return None
    return round((when - START).total_seconds() / 86400, 2)


SLIM = (
    "states", "owned", "divisions", "surrender", "strength", "casualties", "manpower",
    "deployed", "rifles", "needed", "at",
)  # fmt: skip


class ArenaState:
    """The game's numbers as the strategist sees them, from an arena_log.ArenaLog that the
    recorder keeps polling: the latest daily report of each side and every state that has
    changed hands. Read from the planner's thread; the log's own thread only appends."""

    def __init__(self, arena, country):
        self.arena, self.country, self.enemy = arena, country, ENEMY[country]
        self.seen = 0
        self.controls = []

    def snapshot(self):
        lines = self.arena.lines
        for line in lines[self.seen : len(lines)]:
            event = parse(line)
            if event and event["kind"] == "control":
                self.controls.append({
                    "date": event["date"], "day": day_number(event["date"]),
                    "tag": event["tag"], "from": event["previous"], "state": event["state"],
                })  # fmt: skip
        self.seen = len(lines)
        days = dict(self.arena.days)
        own, enemy = days.get(self.country), days.get(self.enemy)
        dates = [d["date"] for d in (own, enemy) if d]
        date = max(dates, key=lambda d: day_number(d) or 0) if dates else None

        def slim(day):
            return {k: day[k] for k in SLIM if k in day} if day else None

        return {
            "date": date,
            "day": day_number(date) if date else None,
            "country": self.country,
            "own": slim(own),
            "enemy": slim(enemy),
            "controls": list(self.controls),
        }


def write_json(path, value):
    """`value` as JSON at `path`, whole: a temporary file renamed over it."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, default=str))
    os.replace(temporary, path)


class Strategist:
    """The decision files. Each request is `<game>-<n>.request.json` with its screenshot
    `<game>-<n>.png` in `folder`, and `pending.json` names the one waited on; the answer
    is `<game>-<n>.decision.json`. Everything is then kept in the game's own folder
    (`archive`), with a line per decision in its `log.jsonl`."""

    def __init__(self, folder, timeout=TIMEOUT, poll=0.5, sleep=time.sleep, clock=time.monotonic):
        self.folder = Path(folder)
        self.timeout, self.poll, self.sleep, self.clock = timeout, poll, sleep, clock

    def ask(self, stem, request, rgb=None):
        """The decision for `request`, or None if none came in time (or it was not JSON)."""
        from PIL import Image

        self.folder.mkdir(parents=True, exist_ok=True)
        if rgb is not None:
            image = self.folder / f"{stem}.png"
            Image.fromarray(rgb).save(image)
            request["image"] = str(image.resolve())
        request["answer"] = str((self.folder / f"{stem}.decision.json").resolve())
        write_json(self.folder / f"{stem}.request.json", request)
        write_json(self.folder / "pending.json", request)
        answer = self.folder / f"{stem}.decision.json"
        end = self.clock() + self.timeout
        decision = None
        try:
            while self.clock() < end:
                if answer.exists():
                    try:
                        decision = json.loads(answer.read_text())
                        break
                    except (OSError, ValueError):
                        pass  # Still being written.
                self.sleep(self.poll)
        finally:
            (self.folder / "pending.json").unlink(missing_ok=True)
        return decision if isinstance(decision, dict) else None

    def keep(self, stem, archive, entry):
        """The request, screenshot and decision moved into the game's folder, and the
        decision's line added to its log."""
        archive = Path(archive)
        archive.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".request.json", ".decision.json"):
            source = self.folder / f"{stem}{suffix}"
            if source.exists():
                os.replace(source, archive / source.name)
        with (archive / "log.jsonl").open("a") as out:
            out.write(json.dumps(entry, default=str) + "\n")
