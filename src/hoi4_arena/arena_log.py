"""The arena mod's own report in game.log: surrenders, weekly counts, and a reward from them.

The mod logs, without changing any rule (see mapgen's on_actions):

    start  12:00, 1 January, 1936
    week  1:00, 4 January, 1936 BLU states 8 owned 8 divisions 8 surrender 0
    control RED from BLU West 3 12:00, 9 March, 1936
    capitulated RED winner BLU 12:00, 2 June, 1936
    peace RED BLU 12:00, 3 June, 1936

The worker's game_log request returns these lines with the "ARENA " prefix removed, on
either PC. They are exact where the screen is not: a surrender names both sides, and the
weekly counts do not depend on where the camera is. They are for scoring only. The agent
never sees them, and a vanilla lobby has no mod, so its outcome still comes from the screen.
"""

from __future__ import annotations

import re
import time

# "12:00, 1 January, 1936". The hour has no leading zero; the game pads it with a space
# instead, so a single-digit hour follows two spaces.
DATE = r"(?P<date>\d{1,2}:\d{2}, \d{1,2} \w+, \d{4})"
PATTERNS = {
    "start": re.compile(rf"^start\s+{DATE}$"),
    "week": re.compile(
        rf"^week\s+{DATE} (?P<tag>[A-Z]{{3}}) states (?P<states>\d+) owned (?P<owned>\d+)"
        r" divisions (?P<divisions>\d+) surrender (?P<surrender>[\d.]+)$"
    ),
    "control": re.compile(
        rf"^control (?P<tag>[A-Z]{{3}}) from (?P<previous>[A-Z]{{3}}) (?P<state>.+?)\s+{DATE}$"
    ),
    "capitulated": re.compile(
        rf"^capitulated (?P<loser>[A-Z]{{3}}) winner (?P<winner>[A-Z]{{3}})\s+{DATE}$"
    ),
    "peace": re.compile(rf"^peace (?P<tag>[A-Z]{{3}}) (?P<other>[A-Z]{{3}})\s+{DATE}$"),
}
NUMBERS = {"states", "owned", "divisions"}
ENEMY = {"BLU": "RED", "RED": "BLU"}
# How much a state held counts against a whole surrender, in the potential below. A side
# surrenders at a progress of 1.0 (BASE_SURRENDER_LEVEL), and the arena has 8 states a side,
# so taking all of the enemy's land at this weight is worth half a surrender. States move
# first, before any victory point falls, so this is the part that pays early.
STATE_WEIGHT = 0.5


def parse(line):
    """One mod line as a dict with its `kind`, or None for a line this module does not know.

    Unknown lines are skipped rather than raised on, so a newer mod that logs more does not
    break an older reader.
    """
    for kind, pattern in PATTERNS.items():
        match = pattern.match(line.strip())
        if match:
            event = {"kind": kind, **match.groupdict()}
            for key in NUMBERS & event.keys():
                event[key] = int(event[key])
            if "surrender" in event:
                event["surrender"] = float(event["surrender"])
            return event
    return None


def potential(weeks, country, states_per_country):
    """How far ahead `country` is, from the latest weekly report of each side.

    The enemy's surrender progress minus this side's, plus the difference in states held
    as a fraction of one side's states, weighted by STATE_WEIGHT. A reward that is the
    change in this value is potential-based shaping (Ng, Harada and Russell, 1999), so it
    adds signal without changing which policy is best: over a whole match it sums to the
    final potential minus the first. None until both sides have reported.
    """
    enemy = ENEMY[country]
    if country not in weeks or enemy not in weeks:
        return None
    own, other = weeks[country], weeks[enemy]
    states = (own["states"] - other["states"]) / states_per_country
    return other["surrender"] - own["surrender"] + STATE_WEIGHT * states


class ArenaLog:
    """Follows one game's mod lines through a desktop's game_log request.

    `poll` asks the worker for the lines written since the last call. The latest weekly
    report per country, the winner and every line read are kept here.
    """

    def __init__(self, desktop, *, silence=None, clock=time.monotonic):
        self.desktop = desktop
        self.offset = 0
        self.lines = []
        self.weeks = {}
        # The first report of each side: what it owned before any fighting.
        self.first_weeks = {}
        self.winner = self.loser = self.surrendered = None
        # A weekly report missing for this long means the game clock has stopped. None
        # turns the check off: at speed 2 a week takes 84 s, longer than the screen's own
        # clock check allows.
        self.silence = silence
        self.clock = clock
        self.last_week = clock()

    def poll(self):
        """The new events, oldest first."""
        lines, self.offset = self.desktop.game_log(self.offset)
        events = []
        for line in lines:
            self.lines.append(line)
            event = parse(line)
            if event is None:
                continue
            events.append(event)
            if event["kind"] == "week":
                self.first_weeks.setdefault(event["tag"], event)
                self.weeks[event["tag"]] = event
                self.last_week = self.clock()
            elif event["kind"] == "capitulated" and self.winner is None:
                self.winner, self.loser = event["winner"], event["loser"]
                self.surrendered = event["date"]
        if self.silence is not None and self.clock() - self.last_week > self.silence:
            raise RuntimeError(f"the game clock stopped: no weekly report for {self.silence} s")
        return events

    def potential(self, country):
        """`potential` for this game, scaled by what `country` owned at its first report."""
        first = self.first_weeks.get(country)
        if first is None or not first["owned"]:
            return None
        return potential(self.weeks, country, first["owned"])

    def outcome(self, country):
        """Whether `country` won ("win") or lost ("loss"), or None before a surrender."""
        if self.winner is None:
            return None
        return "win" if self.winner == country else "loss"
