"""What each learned checkpoint has scored, in one table: the numbers to hill-climb.

    python scripts/scoreboard.py [--json artifacts/learned/scoreboard.json]

Per checkpoint (by its folder's name: live-bc5-e0000 and practice-bc5-e0000-2 are both
"bc5-e0000"), from what the runs left in artifacts/learned:
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


def score(root):
    board = defaultdict(lambda: {"practice": 0, "own": defaultdict(int), "coach": defaultdict(lambda: [0, 0]),
                                 "live": 0, "wins": 0})  # fmt: skip
    for path in sorted(Path(root).glob("practice-*/practice-peer.json")):
        entry = board[checkpoint_of(path.parent)]
        for episode in json.loads(path.read_text())["episodes"]:
            steps = (episode.get("setup") or {}).get("steps")
            if not steps:
                continue
            entry["practice"] += 1
            for step in STEPS:
                by = (steps.get(step) or {}).get("by")
                entry["own"][step] += by == "policy"
                if by in ("coach", "nobody"):
                    entry["coach"][step][0] += by == "coach"
                    entry["coach"][step][1] += 1
    for path in sorted(Path(root).glob("live-*/results-peer.json")):
        entry = board[checkpoint_of(path.parent)]
        for game in json.loads(path.read_text()):
            if game.get("winner") in ("BLU", "RED", "timeout") and not game.get("reason"):
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
    head = "| checkpoint | practice | " + " | ".join(STEPS) + " | coach front | live wins |"
    rows = [head, "|" + "---|" * (len(STEPS) + 4)]
    for name in sorted(board):
        e = board[name]
        n = e["practice"]
        own = [f"{e['own'][s]}/{n}" if n else "" for s in STEPS]
        won, tried = e["coach"]["front"]
        coach = f"{won}/{tried}" if tried else ""
        live = f"{e['wins']}/{e['live']}" if e["live"] else ""
        rows.append(f"| {name} | {n or ''} | " + " | ".join(own) + f" | {coach} | {live} |")
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
        plain = {
            k: {**v, "own": dict(v["own"]), "coach": dict(v["coach"])} for k, v in board.items()
        }
        args.json.write_text(json.dumps({"checkpoints": plain, "drills": made}, indent=2))


if __name__ == "__main__":
    main()
