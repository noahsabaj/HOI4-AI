"""The arena's true state at each decision of a recording, as a training target only.

The arena mod logs each side's state every game day (arena_log's "day" lines, arenas since
v3), and recordings stamp each line with the frame it was read at (arena-log.jsonl). Here
that state becomes one vector per decision, from the recording player's side: what a
player who reads the screen well would know. Training can ask the policy's memory to
predict it (train-bc --state-weight), so that what the policy perceives is shaped by what
decides the war, not only by the next click: learning with privileged information
(Vapnik and Vashist, 2009), as AlphaStar's critic read what its agent could not. The
policy never sees these numbers when it plays, and a vanilla lobby has no mod.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .arena_log import ENEMY, parse
from .state_value import day_number, state_id

STATES = 16
# Each side's state as logged, in a form near unit scale: counts and men on a log scale
# (manpower, deployed and casualties are logged in thousands), surrender progress as it
# is (0 to 1), states held as a share of a side's eight, and rifles held over rifles
# needed, capped at 2.
SIDE = (
    "divisions", "strength", "manpower", "deployed", "casualties", "surrender", "owned",
    "supply",
)  # fmt: skip
NAMES = (
    [f"own_{name}" for name in SIDE]
    + [f"enemy_{name}" for name in SIDE]
    + [f"held_{s}" for s in range(1, STATES + 1)]
    + [f"own_at_{s}" for s in range(1, STATES + 1)]
    + [f"enemy_at_{s}" for s in range(1, STATES + 1)]
    + ["year"]
)
DIM = len(NAMES)


def side_vector(day):
    supply = day["rifles"] / max(day["needed"], 1e-3)
    return [
        math.log1p(day["divisions"]),
        math.log1p(day["strength"]),
        math.log1p(day["manpower"]),
        math.log1p(day["deployed"]),
        math.log1p(day["casualties"]),
        day["surrender"],
        day["owned"] / (STATES // 2),
        min(supply, 2.0),
    ]


def vector(own, other, held):
    at_own = [math.log1p(own["at"].get(s, 0)) for s in range(1, STATES + 1)]
    at_other = [math.log1p(other["at"].get(s, 0)) for s in range(1, STATES + 1)]
    when = max(day_number(own["date"]), day_number(other["date"])) / 365
    return side_vector(own) + side_vector(other) + list(held) + at_own + at_other + [when]


def state_rows(stamped, player):
    """(frame, vector) after every daily report, from `player`'s side.

    `held` is +1 for a state `player` controls and -1 for one the enemy does, starting
    from ownership (Blue holds 1 to 8, Red 9 to 16) and following the control lines.
    """
    enemy = ENEMY[player]
    own_first = 1 if player == "BLU" else STATES // 2 + 1
    held = np.array(
        [1.0 if own_first <= s < own_first + STATES // 2 else -1.0 for s in range(1, STATES + 1)]
    )
    latest, rows = {}, []
    for entry in stamped:
        event = parse(entry["line"])
        if event is None:
            continue
        if event["kind"] == "control":
            state = state_id(event["state"])
            if state is not None:
                held[state - 1] = 1.0 if event["tag"] == player else -1.0
        elif event["kind"] == "day":
            latest[event["tag"]] = event
            if player in latest and enemy in latest:
                rows.append((entry["frame"], vector(latest[player], latest[enemy], held)))
    return rows


ORDERS = ("army", "general", "front", "offensive", "run", "law", "clear", "activate", "recruit")
# The next order's kind: one of ORDERS, "other" for a kind added since, or "none" once the
# game's last order is given.
ORDER_KINDS = (*ORDERS, "other", "none")


def decision_orders(manifest, frame_ids):
    """The scripted player's next order at each decision: its kind and the seconds until.

    The manifest's `orders` stamp each order with the number of frames recorded when it
    was complete. A decision reading frame f is working toward the first order stamped
    after f. Kinds index ORDER_KINDS; -1 where the recording has no orders (a learned
    policy's game, an AI game). The time is log(1 + seconds), NaN when unknown or after
    the last order. A training target only: it asks the memory to know where in its
    procedure the player is and what comes next, which the next click alone does not.
    """
    frame_ids = np.asarray(frame_ids)
    kinds = np.full(len(frame_ids), -1, np.int64)
    eta = np.full(len(frame_ids), np.nan, np.float32)
    orders = sorted(manifest.get("orders") or [], key=lambda o: o["frame"])
    if not orders:
        return kinds, eta
    stamps = np.array([o["frame"] for o in orders])
    names = [o["order"] if o["order"] in ORDERS else "other" for o in orders]
    hz = manifest.get("nominal_fps") or 5
    upcoming = np.searchsorted(stamps, frame_ids, side="right")
    for i, k in enumerate(upcoming):
        if k >= len(orders):
            kinds[i] = ORDER_KINDS.index("none")
            continue
        kinds[i] = ORDER_KINDS.index(names[k])
        eta[i] = math.log1p((stamps[k] - frame_ids[i]) / hz)
    return kinds, eta


def recording_player(manifest):
    players = manifest.get("players") or []
    player = players[0] if len(players) == 1 else manifest.get("started_as")
    return player if player in ("BLU", "RED") else None


def decision_states(root, manifest, frame_ids):
    """(len(frame_ids), DIM) float32: the state at each decision's frame; NaN if unknown.

    A line stamped with frame n was read once n frames had been recorded, so it describes
    the game at about frame n - 1. Before the first report the game has not yet run (it
    starts paused, and the scripted player sets up before it runs), so the first report
    stands for those frames too.
    """
    frame_ids = np.asarray(frame_ids)
    unknown = np.full((len(frame_ids), DIM), np.nan, np.float32)
    player = recording_player(manifest)
    path = Path(root) / "arena-log.jsonl"
    if player is None or not path.exists():
        return unknown
    stamped = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = state_rows(stamped, player)
    if not rows:
        return unknown
    frames = np.array([frame for frame, _ in rows])
    values = np.array([v for _, v in rows], np.float32)
    index = np.searchsorted(frames, frame_ids + 1, side="right") - 1
    return values[np.maximum(index, 0)]
