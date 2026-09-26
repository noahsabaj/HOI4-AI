"""What each learned checkpoint has scored, in one table: the numbers to hill-climb.

    python scripts/scoreboard.py [--json artifacts/learned/scoreboard.json]

Per checkpoint (by its folder's name: live-bc5-e0000 and practice-bc5-e0000-2 are both
"bc5-e0000") and the temperatures it played at (practice and play-policy --temperature,
--pointer-temperature; a run from before they were recorded played at 1), from what the
runs left in artifacts/learned:
- practice: episodes, and how often the policy did each setup step itself
  (practice-peer.json, practice.summary). The fast number: ~30 episodes an hour.
- coach: how often the coach managed a step it took over, which sets how much each
  practice hour teaches.
- live: games and wins against the game's AI (results-peer.json). The slow number.
Then the setup drills (artifacts/drills/*/drills-peer.json): how many completed, and an hour.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

STEPS = ("army", "general", "front", "running")


def checkpoint_of(folder):
    """ "bc5-e0000" from live-bc5-e0000, practice-bc5-e0000-2, live-bc5h-e0000 (a variant keeps
    its letter: bc5h is bc5 shown what it holds)."""
    name = re.sub(r"^(live|practice)-", "", Path(folder).name)
    return re.sub(r"-\d+$", "", name) if re.search(r"-e\d{4}-\d+$", name) else name


def sampled_at(game, default=None):
    """(temperature, pointer temperature) a game or episode was played at: its own, else the
    run's (`default`, a practice summary), else 1 (all runs before they were recorded)."""
    default = default or {}
    t = game.get("temperature", default.get("temperature", 1.0))
    pointer = game.get("pointer_temperature", default.get("pointer_temperature", t))
    return float(t), float(pointer)


def score(root):
    """{(checkpoint, temperature, pointer temperature): its practice and live scores}."""
    board = defaultdict(lambda: {"practice": 0, "own": defaultdict(int), "coach": defaultdict(lambda: [0, 0]),
                                 "live": 0, "wins": 0})  # fmt: skip
    for path in sorted(Path(root).glob("practice-*/practice-peer.json")):
        run = json.loads(path.read_text())
        # Each episode names its own; a summary's are a list when its episodes differ.
        summary = run.get("summary", {})
        default = {
            k: summary[k]
            for k in ("temperature", "pointer_temperature")
            if isinstance(summary.get(k), float | int)
        }
        for episode in run["episodes"]:
            steps = (episode.get("setup") or {}).get("steps")
            if not steps:
                continue
            entry = board[(checkpoint_of(path.parent), *sampled_at(episode, default))]
            entry["practice"] += 1
            for step in STEPS:
                by = (steps.get(step) or {}).get("by")
                entry["own"][step] += by == "policy"
                if by in ("coach", "nobody"):
                    entry["coach"][step][0] += by == "coach"
                    entry["coach"][step][1] += 1
    for path in sorted(Path(root).glob("live-*/results-peer.json")):
        for game in json.loads(path.read_text()):
            if game.get("winner") in ("BLU", "RED", "timeout") and not game.get("reason"):
                entry = board[(checkpoint_of(path.parent), *sampled_at(game))]
                entry["live"] += 1
                entry["wins"] += game["winner"] == game.get("started_as")
    return board


def drills(root):
    done = hours = 0.0
    for path in Path(root).glob("*/drills-peer.json"):
        summary = json.loads(path.read_text())["summary"]
        done += summary["complete"]
        if summary.get("per_hour"):
            hours += summary["complete"] / summary["per_hour"]
    return {"complete": int(done), "per_hour": round(done / hours, 1) if hours else None}


def table(board):
    head = "| checkpoint | temperature | pointer | practice | " + " | ".join(STEPS)
    rows = [head + " | coach front | live wins |", "|" + "---|" * (len(STEPS) + 6)]
    for key in sorted(board):
        name, t, pointer = key
        e = board[key]
        n = e["practice"]
        own = [f"{e['own'][s]}/{n}" if n else "" for s in STEPS]
        won, tried = e["coach"]["front"]
        coach = f"{won}/{tried}" if tried else ""
        live = f"{e['wins']}/{e['live']}" if e["live"] else ""
        at = f"| {name} | {t:g} | {pointer:g} | {n or ''} | "
        rows.append(at + " | ".join(own) + f" | {coach} | {live} |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("artifacts/learned"))
    parser.add_argument("--drills", type=Path, default=Path("artifacts/drills"))
    parser.add_argument("--json", type=Path, help="Also write the scores here")
    args = parser.parse_args()
    board = score(args.root)
    print(table(board))
    made = drills(args.drills)
    print(f"\nsetup drills: {made['complete']} complete, {made['per_hour']} an hour")
    if args.json:
        args.json.write_text(json.dumps({"checkpoints": as_rows(board), "drills": made}, indent=2))


def as_rows(board):
    """The board as a list, one row per checkpoint and temperatures, for JSON."""
    return [
        {"checkpoint": name, "temperature": t, "pointer_temperature": pointer,
         **v, "own": dict(v["own"]), "coach": dict(v["coach"])}
        for (name, t, pointer), v in sorted(board.items())
    ]  # fmt: skip


if __name__ == "__main__":
    main()
