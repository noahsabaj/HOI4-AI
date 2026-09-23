"""A win predictor that reads the arena's logged state instead of the screen.

The screen shows part of the map at a time, and reading it has to be learned. The
arena's daily reports (v3 arenas; arena_log's "day" lines) give each side's true state:
divisions in every state, the game's estimate of army strength, casualties, manpower,
rifles, surrender progress, and, from the control lines, who holds each state. A
predictor of who wins from that state is small, trains in seconds on the CPU, and is
only ever used in training: to value each recorded moment for offline reinforcement
learning (offline.advantage_weights), and to say who is ahead at any time. The agent
never sees these numbers, and a vanilla lobby has no mod. AlphaStar's value network
likewise saw what its agent could not, during training only.

Each game gives one example per daily report: the state that day, and the game's
result from Blue's side discounted by the decisions left, the same target the screen
critic learns (dataset.session_labels' `outcome`). A recording's frames get the value
of the last report read before them (arena-log.jsonl stamps each line with the frame).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .arena_log import parse
from .learning import GAMMA, scale_return, value_estimate

SIDE = ("states", "owned", "divisions", "surrender", "strength", "casualties", "manpower",
        "deployed", "rifles", "needed")  # fmt: skip
STATES = 16
FEATURES = 2 * (len(SIDE) + STATES) + STATES + 1
MONTHS = {m: i for i, m in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August", "September",
     "October", "November", "December"), 1)}  # fmt: skip


def state_id(name, per_side=STATES // 2):
    """A state's id from its logged name: Blue's are "West n", Red's "East n" (mapgen)."""
    match = re.fullmatch(r"(West|East) (\d+)", name.strip())
    if not match:
        return None
    return int(match.group(2)) + (per_side if match.group(1) == "East" else 0)


def day_number(date):
    """Days since 1 January 1936 from a logged date, "24:00, 4 January, 1936"."""
    match = re.fullmatch(r"\s*(\d+):\d+, (\d+) (\w+), (\d+)", date)
    hour, day, month, year = int(match[1]), int(match[2]), MONTHS[match[3]], int(match[4])
    days = (year - 1936) * 365 + sum((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[: month - 1])
    return days + day - 1 + hour / 24


def snapshots(stamped):
    """(frame, features) after every daily report, from arena-log.jsonl entries.

    Starts once both sides have reported. The control map starts from ownership (Blue
    holds 1 to 8, Red 9 to 16) and follows the control lines.
    """
    latest, rows = {}, []
    control = np.array([1.0] * (STATES // 2) + [-1.0] * (STATES // 2))
    for entry in stamped:
        event = parse(entry["line"])
        if event is None:
            continue
        if event["kind"] == "control":
            state = state_id(event["state"])
            if state is not None:
                control[state - 1] = 1.0 if event["tag"] == "BLU" else -1.0
        elif event["kind"] == "day":
            latest[event["tag"]] = event
            if len(latest) == 2:
                rows.append((entry["frame"], features(latest["BLU"], latest["RED"], control)))
    return rows


def features(blue, red, control):
    side = [
        [day[key] for key in SIDE] + [day["at"].get(s, 0) for s in range(1, STATES + 1)]
        for day in (blue, red)
    ]
    when = max(day_number(blue["date"]), day_number(red["date"])) / 365
    return np.array(side[0] + side[1] + list(control) + [when], dtype=np.float32)


def game_examples(root):
    """A recorded game's (features, target, frame) arrays, or None without v3 reports."""
    root = Path(root)
    path = root / "arena-log.jsonl"
    manifest = json.loads((root / "manifest.json").read_text())
    sign = {"BLU": 1.0, "RED": -1.0}.get(manifest.get("winner"))
    if not path.exists() or sign is None:
        return None
    rows = snapshots([json.loads(line) for line in path.read_text().splitlines() if line])
    if not rows:
        return None
    frames = np.array([frame for frame, _ in rows])
    # The same return the screen critic learns: the result, discounted by the decisions
    # left (one per frame at 5 a second) until the recording ends.
    target = sign * GAMMA ** (manifest["frames"] - frames).astype(np.float64)
    return np.stack([x for _, x in rows]), target.astype(np.float32), frames


class StateValue(nn.Module):
    def __init__(self, mean, std, width=128):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(FEATURES, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 1),
        )  # fmt: skip

    def forward(self, x):
        """The value logit, as the screen critic's (learning.value_estimate reads it)."""
        return self.net((x - self.mean) / self.std).squeeze(-1)


def split_of(root):
    return json.loads((Path(root) / "manifest.json").read_text()).get("split", "train")


def train_state_value(roots, output, *, epochs=300, lr=3e-3, seed=0):
    """Fit the predictor on recorded v3 games and report it on the validation split."""
    torch.manual_seed(seed)
    games = {"train": [], "validation": []}
    for root in roots:
        examples = game_examples(root)
        if examples is not None:
            games["validation" if split_of(root) == "validation" else "train"].append(examples)
    if not games["train"]:
        raise ValueError("no finished v3 games with daily reports to learn from")

    def stack(chosen):
        return (
            torch.from_numpy(np.concatenate([x for x, _, _ in chosen])),
            torch.from_numpy(np.concatenate([t for _, t, _ in chosen])),
        )

    x, y = stack(games["train"])
    std = x.std(0)
    model = StateValue(x.mean(0), torch.where(std > 1e-6, std, torch.ones_like(std)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x), scale_return(y))
        loss.backward()
        optimizer.step()
    report = {"train_games": len(games["train"]), "train_loss": loss.item()}
    if games["validation"]:
        report.update(evaluate(model, games["validation"]))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "report": report}, output)
    return report


@torch.no_grad()
def evaluate(model, games):
    """Loss, and how often the predicted return has the winner's sign, by game third."""
    report = {"validation_games": len(games)}
    thirds = {0: [], 1: [], 2: []}
    losses = []
    for x, y, _ in games:
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        logit = model(x)
        losses.append(F.binary_cross_entropy_with_logits(logit, scale_return(y)).item())
        right = (value_estimate(logit).sign() == y.sign()).float().numpy()
        for i, part in enumerate(np.array_split(right, 3)):
            thirds[i].extend(part.tolist())
    report["validation_loss"] = float(np.mean(losses))
    report["sign_accuracy_by_third"] = [
        round(float(np.mean(v)), 3) if v else None for v in thirds.values()
    ]
    return report


def load_state_value(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = StateValue(saved["state"]["mean"], saved["state"]["std"])
    model.load_state_dict(saved["state"])
    return model.eval()


@torch.no_grad()
def frame_values(model, root, frames):
    """The predicted return from Blue's side at each of a recording's `frames` frames.

    Each frame takes the value of the last daily report read before it; frames before
    the first report take 0, an even game.
    """
    stamped = [
        json.loads(line)
        for line in (Path(root) / "arena-log.jsonl").read_text().splitlines()
        if line
    ]
    rows = snapshots(stamped)
    values = np.zeros(frames, dtype=np.float32)
    if not rows:
        return values
    at = np.array([frame for frame, _ in rows])
    predicted = value_estimate(model(torch.from_numpy(np.stack([x for _, x in rows])))).numpy()
    index = np.searchsorted(at, np.arange(frames), side="right") - 1
    values[index >= 0] = predicted[index[index >= 0]]
    return values
