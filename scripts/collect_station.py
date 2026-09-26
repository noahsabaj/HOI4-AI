"""The training PC's side of the second PC's station (scripts/station.py, a fleet service):
what it needs goes there, and what it recorded comes back.

    python scripts/collect_station.py deploy [--checkpoint CKPT --name bc6-e0000 --evaluate]
    python scripts/collect_station.py collect [--every 120]

- deploy: a staging folder (a git worktree at origin/main, because a fleet push copies a
  whole folder and this checkout holds hundreds of GB) gets the code and the data the
  sessions read: the screen templates and rules, the start saves, the pairing file, and
  with --checkpoint that checkpoint, its manifest and its tower, plus the plan naming it.
  It is pushed to the station's project, and the service is asked to restart at its next
  idle moment if its code changed.
- collect: every --every seconds, the station's list of finished sessions is pulled, each
  new session folder is pulled into this checkout at the same path (where the dataset
  builders and the scoreboard look), and deleted on the second PC, whose disk is shared.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent / f"{ROOT.name}-fleet"
NODE, PROJECT, SERVICE = "kat-pc", "hoi4-ai", "hoi4-station"
DATA_FOLDERS = ["artifacts/screens-1080p", "artifacts/calibration-1080p"]
DATA_FILES = ["artifacts/pairing/peer.json", "artifacts/arenas/saves-peer.json"]
COLLECTED = ROOT / "artifacts" / "station" / "collected.txt"
EVALUATE = [
    {"name": "t05", "command": "practice", "args": ["--episodes", "16", "--minutes", "30",
                                                    "--temperature", "0.5"]},
    {"name": "p0", "command": "practice", "args": ["--episodes", "16", "--minutes", "30",
                                                   "--pointer-temperature", "0"]},
    {"name": "e", "command": "play-policy", "args": ["--games", "2", "--minutes", "40",
                                                     "--start-save", "BLU:arenav4blu",
                                                     "RED:arenav4red"]},
]  # fmt: skip


def log(text):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} collect-station: {text}", flush=True)


def fleet(*args, check=True, quiet=False):
    done = subprocess.run(["fleet", *args], cwd=STAGE, capture_output=quiet, text=True)
    if check and done.returncode:
        raise RuntimeError(f"fleet {' '.join(args)} failed ({done.returncode})")
    return done


def link(relative):
    source, target = ROOT / relative, STAGE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        a, b = source.stat(), target.stat()
        if a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime):
            return
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def mirror(relative):
    done = subprocess.run(["robocopy", str(ROOT / relative), str(STAGE / relative), "/MIR",
                           "/NFL", "/NDL", "/NJH", "/NJS", "/NP"], capture_output=True)  # fmt: skip
    if done.returncode >= 8:
        raise RuntimeError(f"staging {relative} failed (robocopy {done.returncode})")


def tower_of(checkpoint):
    manifest = json.loads(Path(ROOT / checkpoint).with_suffix(".json").read_text())
    return f"models/{PureWindowsPath(manifest['config']['model_path']).name}"


def deploy(checkpoint=None, name=None, evaluate=False, practice=()):
    if not STAGE.exists():
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(STAGE),
                        "origin/main"], check=True)  # fmt: skip
    before = subprocess.run(["git", "-C", str(STAGE), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()  # fmt: skip
    subprocess.run(["git", "-C", str(STAGE), "fetch", "-q", "origin"], check=True)
    subprocess.run(["git", "-C", str(STAGE), "checkout", "-q", "--detach", "origin/main"],
                   check=True)  # fmt: skip
    after = subprocess.run(["git", "-C", str(STAGE), "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip()  # fmt: skip
    for relative in DATA_FILES:
        link(relative)
    for relative in DATA_FOLDERS:
        mirror(relative)
    if checkpoint:
        link(checkpoint)
        link(Path(checkpoint).with_suffix(".json").as_posix())
        mirror(tower_of(checkpoint))
        plan = {"checkpoint": checkpoint, "name": name or Path(checkpoint).parent.name,
                "practice": list(practice), "evaluate": EVALUATE if evaluate else []}  # fmt: skip
        (STAGE / "artifacts" / "station").mkdir(parents=True, exist_ok=True)
        (STAGE / "artifacts" / "station" / "plan.json").write_text(json.dumps(plan, indent=2))
    fleet("push", "--on", NODE, "--name", PROJECT)
    if before != after:
        log(f"code {before[:7]} -> {after[:7]}: the station restarts at its next idle moment")
        fleet("service", "restart", SERVICE, "--on", NODE, check=False)
    log("deployed")


def collected():
    return set(COLLECTED.read_text().split()) if COLLECTED.exists() else set()


def collect_once():
    listing = "artifacts/station/finished.jsonl"
    if fleet("pull", listing, "--on", NODE, "--name", PROJECT, check=False,
             quiet=True).returncode:  # fmt: skip
        return 0
    done = collected()
    count = 0
    for line in (STAGE / listing).read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        output = entry["output"]
        if output in done:
            continue
        if fleet("pull", output, "--on", NODE, "--name", PROJECT, check=False,
                 quiet=True).returncode or not (STAGE / output).exists():  # fmt: skip
            log(f"{output} could not be pulled; trying again next time")
            continue
        moved = subprocess.run(["robocopy", str(STAGE / output), str(ROOT / output), "/E",
                                "/MOVE", "/NFL", "/NDL", "/NJH", "/NJS", "/NP"],
                               capture_output=True)  # fmt: skip
        if moved.returncode >= 8:
            log(f"moving {output} here failed (robocopy {moved.returncode})")
            continue
        fleet("run", "--on", NODE, "--name", PROJECT, "--no-push", "--", "uv", "run",
              "--frozen", "python", "-c",
              f"import shutil; shutil.rmtree(r'{output}', ignore_errors=True)",
              check=False, quiet=True)  # fmt: skip
        COLLECTED.parent.mkdir(parents=True, exist_ok=True)
        with COLLECTED.open("a", encoding="utf-8") as handle:
            handle.write(output + "\n")
        log(f"collected {output} ({'played' if entry.get('played') else 'played nothing'})")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    push = sub.add_parser("deploy")
    push.add_argument("--checkpoint")
    push.add_argument("--name")
    push.add_argument(
        "--evaluate",
        action="store_true",
        help="Also its one-off evaluation (T=0.5, pointer T=0, 2 live games)",
    )
    push.add_argument(
        "--held-previous",
        action="store_true",
        help="Practice with --held-previous (for checkpoints from before #113)",
    )
    pull = sub.add_parser("collect")
    pull.add_argument("--every", type=float, default=120.0)
    pull.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.action == "deploy":
        deploy(
            args.checkpoint,
            args.name,
            args.evaluate,
            ["--held-previous"] if args.held_previous else [],
        )
        return 0
    while True:
        try:
            collect_once()
        except (OSError, RuntimeError, ValueError) as error:
            log(f"collect failed: {error}")
        if args.once:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
