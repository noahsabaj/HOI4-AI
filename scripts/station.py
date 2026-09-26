"""The second PC's HOI4 loop, run on it as a fleet service that lends its GPU:

    fleet service add hoi4-station --on <second-pc-node> --name hoi4-ai --yields -- \
        uv run --frozen python scripts/station.py

Round after round: setup drills, then, once the plan names a learned checkpoint, that
checkpoint's one-off evaluation sessions and its coached practice. Each session is a
`hoi4-arena` command run here against this PC's own worker, and each finished session's
folder is listed in artifacts/station/finished.jsonl for scripts/collect_station.py on the
training PC, which pulls it back and deletes it here.

Between sessions it answers fleet:
- FLEET_YIELD_WANTED (a GPU job waits for this PC): HOI4 is closed, FLEET_YIELD_LENT is
  created, and the loop waits until WANTED is gone, then deletes LENT and goes on.
- FLEET_RESTART_WANTED (new code pushed): the loop exits, and fleet starts it again.
- artifacts/station/DRAIN: the loop ends after the session in progress.

The plan (artifacts/station/plan.json, pushed from the training PC; read before each
session, so it changes without a restart):
    {"checkpoint": "artifacts/learned/bc6/epoch-0000.pt", "name": "bc6-e0000",
     "practice": ["--held-previous"],
     "evaluate": [{"name": "t05", "command": "practice", "args": ["--temperature", "0.5"]}, ...],
     "drill_minutes": 30, "practice_minutes": 30, "arenas": [...]}
A session that plays nothing (its summary counts no drill, episode or game) makes the
loop wait 10 minutes, still answering fleet, rather than start the next one at once.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

STATION = Path("artifacts/station")
PLAN = STATION / "plan.json"
FINISHED = STATION / "finished.jsonl"
DRAIN = STATION / "DRAIN"
PEER = "artifacts/pairing/peer.json"
ARENAS = ["arena-12x8-v4", "arena-bay-v6", "arena-12x8-v4", "arena-river-v6", "arena-12x8-v4",
          "arena-plains-v6", "arena-12x8-v4", "arena-passes-v6", "arena-12x8-v4",
          "arena-marsh-v6", "arena-12x8-v4", "arena-salient-v6", "arena-12x8-v4",
          "arena-ford-v6"]  # fmt: skip
SUMMARIES = {"practice": "practice-peer.json", "drills": "drills-peer.json",
             "play-policy": "results-peer.json"}  # fmt: skip


def log(text):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} station: {text}", flush=True)


def flag(name):
    """The file fleet names in environment variable `name`, if it exists now."""
    path = os.environ.get(name)
    return Path(path) if path and Path(path).exists() else None


def hoi4(*args):
    return subprocess.run([sys.executable, "-m", "hoi4_arena", *args]).returncode


def lend_if_wanted(poll=10.0, quit_game=None):
    """Between sessions: if fleet wants the GPU, close HOI4, lend, and wait for it back.
    Returns whether it lent."""
    wanted = flag("FLEET_YIELD_WANTED")
    if wanted is None:
        return False
    log("fleet wants the GPU: closing HOI4 and lending it")
    (quit_game or (lambda: hoi4("control", "quit", "--peer", PEER)))()
    lent = Path(os.environ["FLEET_YIELD_LENT"])
    lent.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    while wanted.exists():
        time.sleep(poll)
    lent.unlink(missing_ok=True)
    log("the GPU is back")
    return True


def should_stop():
    if DRAIN.exists():
        log("drained")
        return True
    if flag("FLEET_RESTART_WANTED"):
        log("fleet asks for a restart (new code): exiting between sessions")
        return True
    return False


def played(command, output):
    try:
        written = json.loads((Path(output) / SUMMARIES[command]).read_text())
    except (OSError, ValueError):
        return False
    if isinstance(written, list):
        return bool(written)
    summary = written.get("summary") or {}
    return bool(summary.get("complete" if command == "drills" else "episodes"))


def session(command, output, args, run=hoi4):
    """One session here; its folder is listed for collection whatever happened."""
    log(f"{command} {output} {' '.join(args)}")
    code = run(command, output, "--peer", PEER, *args)
    STATION.mkdir(parents=True, exist_ok=True)
    ok = code == 0 and played(command, output)
    with FINISHED.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"output": output, "command": command, "exit": code,
                                 "played": ok, "at": time.time()}) + "\n")  # fmt: skip
    return ok


def rest(minutes=10.0, poll=10.0):
    """After a session that played nothing: wait, still answering fleet."""
    log(f"the session played nothing: waiting {minutes:.0f} min")
    end = time.monotonic() + minutes * 60
    while time.monotonic() < end and not should_stop():
        lend_if_wanted()
        time.sleep(poll)


def plan():
    try:
        return json.loads(PLAN.read_text())
    except (OSError, ValueError):
        return {}


def main():
    STATION.mkdir(parents=True, exist_ok=True)
    log(f"started (pid {os.getpid()})")
    while not should_stop():
        lend_if_wanted()
        now = plan()
        stamp = time.strftime("%Y%m%d-%H%M")
        arenas = now.get("arenas") or ARENAS
        args = ["--episodes", "60", "--minutes", str(now.get("drill_minutes", 30)),
                "--arenas", *arenas, "--block", "4"]  # fmt: skip
        if not session("drills", f"artifacts/drills/{stamp}", args):
            rest()
        if should_stop():
            break
        lend_if_wanted()
        now = plan()
        checkpoint = now.get("checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            continue
        name = now.get("name") or Path(checkpoint).parent.name

        def with_checkpoint(command, output, *args):
            # practice and play-policy take the checkpoint before the output folder.
            return hoi4(command, checkpoint, output, *args)

        for entry in now.get("evaluate", []):
            done = STATION / f"evaluated-{name}-{entry['name']}"
            if done.exists():
                continue
            if should_stop():
                break
            lend_if_wanted()
            kind = "live" if entry["command"] == "play-policy" else "practice"
            output = f"artifacts/learned/{kind}-{name}-{entry['name']}"
            session(entry["command"], output, entry.get("args", []), run=with_checkpoint)
            done.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        if should_stop():
            break
        lend_if_wanted()
        output = f"artifacts/learned/practice-{name}-s{time.strftime('%H%M')}"
        args = ["--episodes", "20", "--minutes", str(now.get("practice_minutes", 30)),
                *now.get("practice", [])]  # fmt: skip
        if not session("practice", output, args, run=with_checkpoint):
            rest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
