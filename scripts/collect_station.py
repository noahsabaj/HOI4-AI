"""The training PC's side of the second PC's station (scripts/station.py, a fleet service):
what it needs goes there, and what it recorded comes back.

    python scripts/collect_station.py deploy [--checkpoint CKPT --name bc6-e0000 --evaluate]
                                             [--worker]
    python scripts/collect_station.py collect [--every 120]
    python scripts/collect_station.py tunnel

- deploy: a staging folder (a git worktree at origin/main, because a fleet push copies a
  whole folder and this checkout holds hundreds of GB) gets the code and the data the
  sessions read: the screen templates and rules, the start saves, the pairing file, and
  with --checkpoint that checkpoint, its manifest and its tower, plus the plan naming it.
  It is pushed to the station's project, and the service is asked to restart at its next
  idle moment if its code changed.
  With --worker it also ships the desktop worker, which runs in the same project as the
  fleet service `hoi4-worker` (scripts/Start-Worker.ps1 -Service): this checkout's release
  build, the pairing's second-PC half, the arenas the station plays (their .mod
  descriptors are written into the game's mod folder at each launch, pointing here), and
  peer-fleet.json, the pairing on 127.0.0.1 that the station then uses (station.peer). The
  worker service is asked to restart at its next idle moment if any of it changed.
- collect: every --every seconds, the station's list of finished sessions is pulled, each
  new session folder is pulled into this checkout at the same path (where the dataset
  builders and the scoreboard look), and deleted on the second PC, whose disk is shared.
- tunnel: the worker service's port here on 127.0.0.1 (`fleet tunnel`), opened again
  whenever it ends, for the live view and anything else here with --peer
  artifacts/pairing/peer-fleet.json. A tunnel's connections end when the node restarts,
  and clients using that pairing connect again (reconnect_seconds).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
# The desktop worker as a fleet service (deploy --worker, tunnel).
WORKER_SERVICE, WORKER_PORT = "hoi4-worker", 47941
WORKER_BUILD = "target/release/hoi4-desktop-worker.exe"
WORKER_EXE = "artifacts/worker/hoi4-desktop-worker.exe"
WORKER_PAIRING = [
    "artifacts/pairing/second-pc/server.json",
    "artifacts/pairing/second-pc/worker.pfx",
]
FLEET_PEER = "artifacts/pairing/peer-fleet.json"
# What the service reads when it starts: a change to any of them wants a restart. The
# control scripts are read at each operation, and the arenas at each launch.
WORKER_STARTS_FROM = [WORKER_EXE, "scripts/Start-Worker.ps1", *WORKER_PAIRING]
# How long a client with the fleet pairing keeps trying to connect: a service restart
# (fleet starts it again within ~5 s, and it compiles its bridge in a few more) or a node
# restart (its tunnel's connections end) is waited out, not a session's failure.
RECONNECT_SECONDS = 60
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


def link(relative, source=None, copy=False):
    """`relative` staged from this checkout (or `source`): hard-linked, or with `copy` a
    copy of its own, for a file rewritten in place, whose staged link would change with it
    before the deploy could see that it changed."""
    source, target = Path(source) if source else ROOT / relative, STAGE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        a, b = source.stat(), target.stat()
        if a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime):
            return
        target.unlink()
    if copy:
        shutil.copy2(source, target)
        return
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


def fleet_pairing(port=WORKER_PORT):
    """artifacts/pairing/peer-fleet.json: peer.json's token and certificate pin at
    127.0.0.1:`port`. On the second PC that is the worker service itself; here, fleet's
    tunnel to it, opened on the same port (tunnel). Its clients keep trying to connect for
    RECONNECT_SECONDS (remote.open_tls). Written only when it changes."""
    spec = json.loads((ROOT / DATA_FILES[0]).read_text())
    spec.update(host="127.0.0.1", port=port, reconnect_seconds=RECONNECT_SECONDS)
    text = json.dumps(spec, indent=2)
    path = ROOT / FLEET_PEER
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return path


def station_arenas():
    """The arenas the station plays: its own list, and its plan's."""
    spec = importlib.util.spec_from_file_location("station", ROOT / "scripts" / "station.py")
    station = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(station)
    try:
        plan = json.loads((STAGE / station.PLAN).read_text())
    except (OSError, ValueError):
        plan = {}
    return sorted({*station.ARENAS, *(plan.get("arenas") or [])})


def worker_stamp():
    """The staged files the worker service starts from (WORKER_STARTS_FROM), as one hash."""
    digest = hashlib.sha256()
    for relative in WORKER_STARTS_FROM:
        path = STAGE / relative
        digest.update(relative.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest() if path.exists() else b"-")
    return digest.hexdigest()


def stage_worker():
    """The worker service's files into the staging folder (see deploy --worker)."""
    build = ROOT / WORKER_BUILD
    if not build.exists():
        raise RuntimeError(
            f"no {WORKER_BUILD}: cargo build --release --locked -p hoi4-desktop-worker"
        )
    link(WORKER_EXE, build, copy=True)
    for relative in WORKER_PAIRING:
        link(relative)
    fleet_pairing()
    link(FLEET_PEER)
    for arena in station_arenas():
        mirror(f"artifacts/mods/{arena}")


def deploy(checkpoint=None, name=None, evaluate=False, practice=(), worker=False):
    if not STAGE.exists():
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(STAGE),
                        "origin/main"], check=True)  # fmt: skip
    before = subprocess.run(["git", "-C", str(STAGE), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()  # fmt: skip
    worker_before = worker_stamp()
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
    if worker:
        stage_worker()
    fleet("push", "--on", NODE, "--name", PROJECT)
    if before != after:
        log(f"code {before[:7]} -> {after[:7]}: the station restarts at its next idle moment")
        fleet("service", "restart", SERVICE, "--on", NODE, check=False)
    if worker and worker_stamp() != worker_before:
        # Before the service exists (the cutover's first deploy) this only fails.
        log("the worker changed: its service restarts at its next idle moment")
        fleet("service", "restart", WORKER_SERVICE, "--on", NODE, check=False)
    log("deployed")


def tunnel(port=WORKER_PORT, local=WORKER_PORT, pause=5.0, rounds=None):
    """The worker service's port here on 127.0.0.1, opened again whenever `fleet tunnel`
    ends (it should outlive node restarts, whose connections it loses; this covers the
    rest), until stopped."""
    while rounds is None or rounds > 0:
        rounds = None if rounds is None else rounds - 1
        log(f"tunnel to {NODE}:{port} on 127.0.0.1:{local}")
        code = subprocess.run(
            ["fleet", "tunnel", NODE, str(port), "--local", str(local)]
        ).returncode
        log(f"the tunnel ended ({code}); again in {pause:.0f} s")
        time.sleep(pause)


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
    push.add_argument(
        "--worker",
        action="store_true",
        help=f"Also the desktop worker, for the fleet service {WORKER_SERVICE} (and "
        f"{FLEET_PEER}, which the station then uses)",
    )
    pull = sub.add_parser("collect")
    pull.add_argument("--every", type=float, default=120.0)
    pull.add_argument("--once", action="store_true")
    sub.add_parser("tunnel")
    args = parser.parse_args()
    if args.action == "deploy":
        deploy(
            args.checkpoint,
            args.name,
            args.evaluate,
            ["--held-previous"] if args.held_previous else [],
            worker=args.worker,
        )
        return 0
    if args.action == "tunnel":
        tunnel()
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
