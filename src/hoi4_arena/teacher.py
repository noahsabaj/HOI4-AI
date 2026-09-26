"""A privileged teacher: the next intent from the game's true state, for distillation.

Training may read the game's state; only the player that plays is held to pixels. So a
teacher that decides from the logged state (arena_log's day and control lines; the state
channel's fuller report in games recorded since it) and acts through intents, executed by
the scripted player's hand, can be made strong where a pixels policy cannot yet, and its
decisions then label the pixels student ("learning by cheating", Chen et al., 2019).

This is its first stage, behaviour cloning of the scripted player:

- `decisions` turns a scripted recording into (features, intent, target state) rows, one
  a second while the hand is idle. The intent is the order the scripted player gave next,
  in the intent vocabulary the strategy/execution split uses (INTENTS), or "wait".
- `features` is the player's state vector (privileged.vector: both sides' divisions,
  strength, manpower, casualties, supply, states held and divisions in each state, the
  date) with the player's own history (its army, general, front, offensive, plan
  executing, law, redraws, seconds since its last order), its side and its arena.
- `Teacher` is a small network with an intent head, a target-state head (which enemy
  state an offensive aims at) and a win head, trained on the CPU in a minute
  (`train`), and scored on held-out games against the same network without the state
  (`--blind`): what the state adds to the scripted player's own sequence.

Later stages (see STATUS.md): the same network trained by RL in live games through the
scripted hand, then its intents label the pixels student's own states (a teacher that
reads the state can label any state at once, which the scripted player's screen reading
cannot), and the student's memory is asked to predict the teacher's state encoding.
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import intents as intent_vocabulary
from .arena_log import ENEMY, parse
from .privileged import DIM as STATE_DIM
from .privileged import recording_player, vector
from .state_value import state_id

# The strategy/execution split's intents (intents.py) that a strategist chooses: the camera
# and popups are the camera director's between intents. "wait" leaves the plan as it
# stands. The scripted player's orders map onto them through intents.ORDER_SKILL.
INTENTS = tuple(i for i in intent_vocabulary.INTENTS if i not in ("camera", "popup"))
ORDER_INTENT = {
    order: intent_vocabulary.SKILL_INTENT[skill]
    for order, skill in intent_vocabulary.ORDER_SKILL.items()
}
ARENAS = ("arena-12x8-v4", "plains", "passes", "marsh", "bay", "river", "salient", "ford")
LAWS = ("volunteer", "limited", "extensive", "service", "all_adults")
HISTORY = (
    "army", "general", "front", "offensive", "executing", "law", "redraws", "running",
    "since_order", "since_run", "guarded",
)  # fmt: skip
DIM = STATE_DIM + len(HISTORY) + 2 + len(ARENAS) + 1
STATES = 16
# One decision a second (5 frames at 5 Hz) while the hand is idle; an order more than this
# many frames after the one before is taken to have been chosen that long before it was
# complete, the rest of the gap being idle.
STEP = 5
SKILL = 25
CHAINED = 50


def arena_index(name):
    for i, key in enumerate(ARENAS):
        if name and key in name:
            return i
    return len(ARENAS)


def history_vector(history, frame):
    h = history
    return [
        float(h["army"]), float(h["general"]), float(h["front"]), float(h["offensive"]),
        float(h["executing"]), h["law"] / (len(LAWS) - 1), math.log1p(h["redraws"]),
        float(h["running"]),
        math.log1p(max(frame - h["last_order"], 0) / 5),
        math.log1p(max(frame - h["run_at"], 0) / 5) if h["running"] else 0.0,
        float(h["guarded"]),
    ]  # fmt: skip


def apply_order(history, order):
    """The player's own history after `order`, in place."""
    kind = order["order"]
    history["last_order"] = order["frame"]
    if kind == "army":
        history["army"] = True
    elif kind == "general":
        history["general"] = True
    elif kind == "front":
        history["front"] = True
    elif kind == "offensive":
        history["offensive"] = True
    elif kind == "activate":
        history["executing"] = True
    elif kind == "clear":
        history.update(front=False, offensive=False, executing=False)
        history["redraws"] += 1
    elif kind == "law" and order.get("law") in LAWS:
        history["law"] = LAWS.index(order["law"])
    elif kind == "run":
        history.update(running=True, run_at=order["frame"])
    elif kind == "guard":
        history["guarded"] = True


def intents_of(orders):
    """The orders as intents: (frame, intent, target state or -1, order) in time order.

    A redraw is one intent: the clear and the front and offensive drawn after it until
    another kind of order. Guard and pocket orders are notes, not intents.
    """
    out, redrawing = [], False
    for order in sorted(orders, key=lambda o: o["frame"]):
        kind = order["order"]
        if kind in ("front", "offensive") and redrawing:
            continue
        redrawing = kind == "clear" or (redrawing and kind in ("guard", "pocket"))
        intent = ORDER_INTENT.get(kind)
        if intent is None:
            continue
        target = order.get("target_state")
        out.append((order["frame"], intent, int(target) - 1 if target else -1, order))
    return out


def _states(stamped, player):
    """(frames, vectors) of the player's state after every pair of daily reports."""
    enemy = ENEMY[player]
    first = 1 if player == "BLU" else STATES // 2 + 1
    held = [1.0 if first <= s < first + STATES // 2 else -1.0 for s in range(1, STATES + 1)]
    latest, frames, rows = {}, [], []
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
                frames.append(entry["frame"])
                rows.append(vector(latest[player], latest[enemy], held))
    return np.array(frames), np.array(rows, np.float32)


def decisions(root):
    """A scripted recording's teacher rows: (features, intents, targets, win), or None
    for a recording without orders, a player or a daily report."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    player = recording_player(manifest)
    orders = manifest.get("orders") or []
    path = root / "arena-log.jsonl"
    if player is None or not orders or not path.exists():
        return None
    stamped = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    frames, states = _states(stamped, player)
    if not len(frames):
        return None
    context = [float(player == "RED"), float(player == "BLU")]
    arena = [0.0] * (len(ARENAS) + 1)
    arena[arena_index(manifest.get("arena"))] = 1.0
    end = manifest.get("frames") or orders[-1]["frame"]
    history = dict(
        army=False, general=False, front=False, offensive=False, executing=False, law=0,
        redraws=0, running=False, last_order=0, run_at=0, guarded=False,
    )  # fmt: skip

    def row(frame):
        index = max(np.searchsorted(frames, frame + 1, side="right") - 1, 0)
        return [*states[index], *history_vector(history, frame), *context, *arena]

    feats, intents, targets = [], [], []
    by_frame = sorted(orders, key=lambda o: o["frame"])
    applied = 0
    previous = 0
    for frame, intent, target, _ in intents_of(orders):
        chosen = previous if frame - previous <= CHAINED else frame - SKILL
        for t in range(previous + STEP, chosen, STEP):
            feats.append(row(t))
            intents.append(0)
            targets.append(-1)
        # The history as it stood when the intent was chosen: every order before it.
        while applied < len(by_frame) and by_frame[applied]["frame"] < frame:
            apply_order(history, by_frame[applied])
            applied += 1
        feats.append(row(chosen))
        intents.append(INTENTS.index(intent))
        targets.append(target)
        previous = frame
        while applied < len(by_frame) and by_frame[applied]["frame"] <= frame:
            apply_order(history, by_frame[applied])
            applied += 1
    for t in range(previous + STEP, end, STEP):
        feats.append(row(t))
        intents.append(0)
        targets.append(-1)
    win = 1.0 if manifest.get("winner") == player else 0.0
    return (
        np.array(feats, np.float32),
        np.array(intents, np.int64),
        np.array(targets, np.int64),
        np.full(len(intents), win, np.float32),
    )


def held_out(root, share=0.15):
    """Whether a recording is held out: a fixed hash of its folder name."""
    return zlib.crc32(Path(root).name.encode()) % 1000 < share * 1000


class Teacher(nn.Module):
    def __init__(self, width=256, blind=False):
        super().__init__()
        self.blind = blind
        self.body = nn.Sequential(
            nn.Linear(DIM, width), nn.GELU(), nn.Linear(width, width), nn.GELU()
        )
        self.intent = nn.Linear(width, len(INTENTS))
        self.target = nn.Linear(width, STATES)
        self.win = nn.Linear(width, 1)

    def forward(self, x):
        if self.blind:
            # Without the game's state: the player's own history, side and arena only.
            x = torch.cat([torch.zeros_like(x[:, :STATE_DIM]), x[:, STATE_DIM:]], 1)
        h = self.body(x)
        return self.intent(h), self.target(h), self.win(h).squeeze(1)


def choose(model, features, rng=None, temperature=0.0, aim=False):
    """The teacher's intent for one row of `features`, as an intents.Intent for the
    scripted hand (hand.ScriptedHand): the likeliest intent at temperature 0, else a draw.
    With `aim`, an offensive or a redraw names the state the target head likes best;
    without, the hand aims as the plan says (a broad line, most games)."""
    with torch.no_grad():
        logits, target_logits, _ = model(torch.as_tensor(features, dtype=torch.float32)[None])
    logits = logits[0].double()
    if temperature > 0:
        rng = rng or np.random.default_rng()
        p = torch.softmax(logits / temperature, 0).numpy()
        index = int(rng.choice(len(p), p=p / p.sum()))
    else:
        index = int(logits.argmax())
    name = INTENTS[index]
    args = {}
    if aim and name in ("draw_offensive", "redraw"):
        args["target_state"] = int(target_logits[0].argmax()) + 1
    return intent_vocabulary.Intent(name, args)


def load(roots):
    parts = [d for d in (decisions(r) for r in roots) if d is not None]
    if not parts:
        return None
    return [np.concatenate(p) for p in zip(*parts)], len(parts)


def evaluate(model, data):
    x, y, target, win = (torch.from_numpy(a) for a in data)
    with torch.no_grad():
        logits, target_logits, win_logit = model(x)
    guess = logits.argmax(1)
    acts = y != 0
    recall = {
        name: float((guess[y == i] == i).float().mean())
        for i, name in enumerate(INTENTS)
        if (y == i).any()
    }
    aimed = target >= 0
    return {
        "rows": len(y),
        "loss": float(F.cross_entropy(logits, y)),
        "accuracy": float((guess == y).float().mean()),
        "act_accuracy": float((guess[acts] == y[acts]).float().mean()),
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "recall": recall,
        "target_accuracy": float(
            (target_logits[aimed].argmax(1) == target[aimed]).float().mean()
        ) if aimed.any() else None,
        "win_loss": float(F.binary_cross_entropy_with_logits(win_logit, win)),
    }  # fmt: skip


def train(roots, output, *, epochs=30, blind=False, seed=0, width=256, batch=512):
    """Fits a Teacher on the scripted recordings `roots` (held_out ones scored, not fitted)
    and saves it with its report. The intent loss weighs each intent by the inverse square
    root of its frequency, since "wait" fills most rows."""
    torch.manual_seed(seed)
    roots = [Path(r) for r in roots]
    fit, fit_games = load([r for r in roots if not held_out(r)])
    test, test_games = load([r for r in roots if held_out(r)])
    x, y, target, win = (torch.from_numpy(a) for a in fit)
    counts = torch.bincount(y, minlength=len(INTENTS)).float().clamp(min=1)
    weights = (counts.sum() / counts).sqrt()
    weights = weights / weights[y].mean()
    model = Teacher(width, blind=blind)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(y))
        model.train()
        for start in range(0, len(y), batch):
            i = order[start : start + batch]
            logits, target_logits, win_logit = model(x[i])
            loss = F.cross_entropy(logits, y[i], weight=weights)
            aimed = target[i] >= 0
            if aimed.any():
                loss = loss + F.cross_entropy(target_logits[aimed], target[i][aimed])
            loss = loss + 0.5 * F.binary_cross_entropy_with_logits(win_logit, win[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        history.append({"epoch": epoch, **evaluate(model, test)})
    report = {
        "fit_games": fit_games,
        "test_games": test_games,
        "fit_rows": len(y),
        "blind": blind,
        "majority_accuracy": float((torch.from_numpy(test[1]) == 0).float().mean()),
        "final": history[-1],
        "history": history,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "width": width, "blind": blind}, output / "teacher.pt")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report
