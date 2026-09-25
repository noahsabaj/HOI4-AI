"""The live view's feed, like a stream's chat: what happens in the games as it happens
(a game starts, the scripted player gives an order, a side nears surrender, a game ends),
beside messages from whoever watches, and from Claude (`hoi4-arena live-say`).

Messages are kept in a JSON-lines file, so the feed outlives restarts, and a flag (a
moment someone marks for a closer look) goes to its own file too. What watchers write is
shown as it is, never acted on.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

# How much of the feed the page can ask for, and how long a message may be.
KEEP, LONGEST, NAME = 500, 300, 24
SIDE = {"BLU": "Blue", "RED": "Red"}
ORDERS = {
    "army": "formed the army",
    "general": "gave the army its general",
    "front": "drew the front line",
    "offensive": "drew the offensive",
    "activate": "launched the attack",
    "run": "started the game",
    "law": "raised conscription",
    "clear": "cleared the army's orders",
    "pause": "paused to redraw",
    "guard": "redrew round an incursion",
    "pocket": "left a stable pocket behind",
    "reinforce": "reinforced the front",
    "recruit": "queued new divisions",
}
MILESTONES = {
    "alert": "clicked the unassigned-divisions alert",
    "plus": "clicked create army",
    "portrait": "opened the general's portrait",
    "law_slot": "opened the conscription law",
    "front": "drew a front",
    "offensive": "drew an offensive",
    "arrow": "pressed execute",
}
# Surrender progress worth a line, as each side passes it.
MARKS = (0.25, 0.5, 0.75)


def clock(seconds):
    seconds = int(seconds or 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def arena_name(arena):
    name = (arena or "?").removeprefix("arena-")
    return name.rsplit("-v", 1)[0] if "-v" in name else name


class Feed:
    """The feed's messages, in memory and in `path` (JSON lines): each {id, t, who, kind,
    text} and, for a game's events, its station and game. `kind` is "event", "user" or
    "claude"."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.messages = []
        self.next_id = 1
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-KEEP:]
            self.messages = [json.loads(line) for line in lines if line.strip()]
        except (OSError, ValueError):
            self.messages = []
        if self.messages:
            self.next_id = max(m.get("id", 0) for m in self.messages) + 1
        for message in self.messages:
            if "id" not in message:  # Appended from outside (say) and never taken in.
                message["id"] = self.next_id
                self.next_id += 1
        self.seen = self.path.stat().st_size if self.path.exists() else 0

    def add(self, who, text, kind="user", **extra):
        text = " ".join(str(text).split())[:LONGEST]
        who = " ".join(str(who).split())[:NAME] or "anon"
        if not text:
            return None
        with self.lock:
            message = {"id": self.next_id, "t": round(time.time(), 1), "who": who,
                       "kind": kind, "text": text, **extra}  # fmt: skip
            self.next_id += 1
            self.messages = [*self.messages, message][-KEEP:]
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(message) + "\n")
                self.seen = self.path.stat().st_size
            except OSError:
                pass
        return message

    def absorb(self):
        """Messages another process appended to the file (live-say), taken in."""
        try:
            size = self.path.stat().st_size
            if size <= self.seen:
                return
            with self.path.open("rb") as file:
                file.seek(self.seen)
                fresh = file.read().decode("utf-8", "replace").splitlines()
            self.seen = size
        except OSError:
            return
        with self.lock:
            for line in fresh:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if message.get("id", 0) >= self.next_id or "id" not in message:
                    message["id"] = self.next_id
                    self.next_id += 1
                    self.messages = [*self.messages, message][-KEEP:]

    def since(self, after=0, limit=200):
        with self.lock:
            return [m for m in self.messages if m["id"] > after][-limit:]


def say(path, text, who="Claude", kind="claude"):
    """A message from outside the running view, appended to its feed's file."""
    message = {"t": round(time.time(), 1), "who": who, "kind": kind, "text": text[:LONGEST]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(message) + "\n")
    return message


class Narrator:
    """Turns the games' cards, round after round, into the feed's event lines: a game
    starting, each new order or setup step, a side passing a mark of surrender, a state
    changing hands, and the game's end."""

    def __init__(self, feed, labels):
        self.feed, self.labels = feed, labels
        self.games = {}  # game name: what was already said about it

    def step(self, cards, finished):
        """`cards`, the live games' cards by station; `finished(game)`, the history entry
        of a game that ended, or None."""
        live = {card["game"]: card for card in cards.values() if card}
        for name, card in live.items():
            said = self.games.get(name)
            where = self.labels.get(card["station"], card["station"])
            if said is None:
                said = self.games[name] = {"frame": -1, "marks": set(), "owned": {},
                                           "start": {}, "steps": {}, "station": card["station"]}  # fmt: skip
                plan = (card.get("plan") or {}).get("variant") or "?"
                self.event(card, f"{where}: {arena_name(card['arena'])} as "
                           f"{SIDE.get(card['side'], card['side'])}, {plan} plan")  # fmt: skip
            for order in card.get("orders") or []:
                if order.get("frame", 0) <= said["frame"]:
                    continue
                said["frame"] = order.get("frame", 0)
                text = ORDERS.get(order["order"], order["order"])
                if order.get("attack"):
                    text += f" ({order['attack']})"
                self.event(card, f"{clock(order['seconds'])} {text}")
            for step, count in (card.get("milestones") or {}).items():
                if step in MILESTONES and count and not said["steps"].get(step):
                    self.event(card, f"learned player {MILESTONES[step]}")
                said["steps"][step] = count
            for side, report in (card.get("sides") or {}).items():
                for mark in MARKS:
                    if (report.get("surrender") or 0) >= mark and (side, mark) not in said["marks"]:
                        said["marks"].add((side, mark))
                        self.event(card, f"{SIDE.get(side, side)} is {int(mark * 100)}% of the way "
                                   "to surrender")  # fmt: skip
                # The report's `states` are the side's states now, lost ones gone, so the
                # count to hold out of is the most it has had (2026-09-25: "7 of 7 held").
                owned, before = report.get("owned"), said["owned"].get(side)
                start = said["start"][side] = max(
                    said["start"].get(side, 0), report.get("states") or 0, owned or 0
                )
                if owned is not None and before is not None and owned < before:
                    self.event(card, f"{SIDE.get(side, side)} lost a state "
                               f"({owned} of {start} held)")  # fmt: skip
                if owned is not None:
                    said["owned"][side] = owned
        for name in [n for n in self.games if n not in live]:
            # Its results come a few seconds after its video stops: wait up to 2 minutes.
            said = self.games[name]
            said.setdefault("gone", time.time())
            done = finished(name)
            if done is None and time.time() - said["gone"] < 120:
                continue
            self.games.pop(name)
            if done:
                verdict = {"win": "won", "loss": "lost", "timeout": "timed out"}[done["result"]]
                self.feed.add("game", f"{self.labels.get(said['station'], said['station'])}: "
                              f"{SIDE.get(done['side'], done['side'])} {verdict} on "
                              f"{arena_name(done['arena'])} in {clock(done['seconds'])}",
                              kind="event", station=said["station"], game=name)  # fmt: skip

    def event(self, card, text):
        self.feed.add("game", text, kind="event", station=card["station"], game=card["game"])


class Flags:
    """Moments someone marked on the page for a closer look, in `path` (JSON lines)."""

    def __init__(self, path):
        self.path = Path(path)

    def add(self, flag):
        flag = {"t": round(time.time(), 1), **flag, "note": str(flag.get("note", ""))[:LONGEST]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(flag) + "\n")
        return flag
