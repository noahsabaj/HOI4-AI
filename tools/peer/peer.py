"""peer: run work on the second PC from any project on this one.

The second PC runs the HOI4-AI project's worker, which the user opened on 2026-09-24 as
"a universal bus" for every project here: training a model there, say. This reaches it
as the HOI4 recorder does, by the worker's pinned certificate and token, but on a
read-only connection, which never waits for a recording and gives no input to the game,
and through its share, where each project has a folder of its own,
compute\\projects\\<name>. A job runs that project's own command there, hidden and
detached, usually `uv run ...`, whose environment is the project's own. Its output and
state go to the share's jobs folder.

    peer push [DIR] [--name NAME] [--mirror]   copy a project there
    peer run [--name NAME] [--gpu-gb N] [--minutes M] [--detach] [--no-push] -- COMMAND ...
    peer logs JOB [--follow]
    peer status
    peer stop JOB
    peer pull PATH [DEST] [--name NAME]        copy results back
    peer gpu

The second PC is shared with the HOI4 runs, whose game holds about 3 GB of its 8 GB card.
A job that needs more video memory than is free reserves the PC through the HOI4
recorder's queue: the recorder finishes its game, closes HOI4 and grants the PC, and
resumes when the job ends, or when its minutes and 15 more have passed.

Setup: ~/.peer/config.json names the pairing file and the reservation queue:
{"pairing": ".../artifacts/pairing/peer.json", "eval": ".../artifacts/eval"}, or set
PEER_PAIRING and PEER_EVAL.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path

CONFIG = Path.home() / ".peer" / "config.json"
# Never copied: version control, environments (rebuilt there by uv), caches, builds.
SKIP_DIRS = (
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".idea", ".vscode", ".tox", "target",
)  # fmt: skip
FINAL = ("done", "failed", "stopped", "lost")
# A reservation waits this long for the HOI4 recorder to finish its game and grant it.
GRANT_WAIT = 20 * 60


class PeerError(Exception):
    pass


def settings():
    """The pairing (host, port, certificate pin, token), the share and the queue."""
    found = {}
    if CONFIG.exists():
        found = json.loads(CONFIG.read_text(encoding="utf-8"))
    pairing = os.environ.get("PEER_PAIRING") or found.get("pairing")
    if not pairing or not Path(pairing).exists():
        raise PeerError(f"peer is not set up: no pairing file (set PEER_PAIRING or {CONFIG})")
    spec = json.loads(Path(pairing).read_text(encoding="utf-8"))
    return {
        "pairing": spec,
        "share": Path(f"//{spec['host']}/HOI4Worker"),
        "eval": os.environ.get("PEER_EVAL") or found.get("eval"),
    }


def project_name(folder):
    """A project's name on the second PC: its folder's, in letters, digits, _ and -."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(folder).resolve().name).strip("-")
    return name[:40] or "project"


class Worker:
    """A read-only connection to the second PC's worker; requests and their replies."""

    def __init__(self, spec, stream=None):
        self.socket = None
        if stream is None:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE  # The exact certificate is pinned below.
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            raw = socket.create_connection((spec["host"], spec["port"]), timeout=30)
            self.socket = context.wrap_socket(raw, server_hostname=spec["host"])
            pin = hashlib.sha256(self.socket.getpeercert(binary_form=True)).hexdigest()
            if not hmac.compare_digest(pin, spec["certificate_sha256"]):
                self.socket.close()
                raise PeerError("the second PC's certificate does not match its pairing file")
            self.socket.settimeout(300)
            stream = self.socket.makefile("rwb")
            stream.write(spec["token"].encode("ascii") + b" observer\n")
            stream.flush()
        self.stream = stream
        self.next_id = 1

    def request(self, op, **fields):
        """One operation's reply; its error raised."""
        request_id = self.next_id
        self.next_id += 1
        message = json.dumps({"op": op, "id": request_id, **fields}, allow_nan=False)
        self.stream.write(message.encode() + b"\n")
        self.stream.flush()
        while True:
            line = self.stream.readline()
            if not line:
                raise PeerError("the second PC closed the connection")
            reply = json.loads(line)
            size = reply.pop("bytes", 0)
            if size:
                self.stream.read(size)  # Another stream's data: none is asked for here.
            if reply.get("id") != request_id:
                continue
            if "error" in reply:
                raise PeerError(f"{op}: {reply['error']}")
            return reply

    def job(self, action, **fields):
        """A job operation's output (Run-Job.ps1 on the second PC); a failure raised."""
        reply = self.request("job", action=action, **fields)
        output = reply.get("output", "")
        if reply.get("exit") != 0:
            raise PeerError(f"job {action} failed: {output.strip()}")
        return output

    def close(self):
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def gpu_memory(status_output):
    """(free, total) MiB of the second PC's GPU, from Run-Job's status (nvidia-smi's CSV
    line), or None if it has none."""
    for line in status_output.splitlines():
        found = re.search(r",\s*(\d+)\s*MiB,\s*(\d+)\s*MiB", line)
        if found:
            used, total = int(found.group(1)), int(found.group(2))
            return total - used, total
    return None


def push_command(source, target, mirror=False):
    """robocopy copying a project's folder there: new and changed files, never the
    skipped folders; with `mirror`, also deleting what is gone here (a job's outputs
    inside the project folder too)."""
    return [
        "robocopy", str(source), str(target), "/MIR" if mirror else "/E",
        "/XD", *SKIP_DIRS, "/XF", "*.pyc",
        "/R:2", "/W:2", "/MT:8", "/NFL", "/NDL", "/NP",
    ]  # fmt: skip


def push(cfg, folder, name=None, mirror=False):
    source = Path(folder).resolve()
    name = name or project_name(source)
    target = cfg["share"] / "compute" / "projects" / name
    code = subprocess.run(push_command(source, target, mirror)).returncode
    # robocopy's exit codes below 8 are successes (1: files copied).
    if code >= 8:
        raise PeerError(f"push failed: robocopy exit {code}")
    print(f"pushed {source} as project {name}")
    return name


def job_state(cfg, job):
    try:
        raw = (cfg["share"] / "jobs" / f"{job}.json").read_text(encoding="utf-8-sig")
        return json.loads(raw)
    except (OSError, ValueError):
        return {}


def text(chunk):
    """A log's bytes as text with plain line ends (Windows writes the logs with CRLF)."""
    return chunk.decode("utf-8", "replace").replace("\r\n", "\n")


def follow(cfg, job, poll=2.0, out=None):
    """Print a job's log as it grows until the job ends; its final state."""
    out = out or sys.stdout
    log = cfg["share"] / "jobs" / f"{job}.log"
    offset = 0
    while True:
        state = job_state(cfg, job)
        if log.exists():
            with open(log, "rb") as file:
                file.seek(offset)
                chunk = file.read()
            offset += len(chunk)
            if chunk:
                out.write(text(chunk))
                out.flush()
        if state.get("state") in FINAL:
            return state
        time.sleep(poll)


def reserve(cfg, job, minutes, wait=GRANT_WAIT, poll=10.0):
    """The second PC to itself, through the HOI4 recorder's queue (see the docstring):
    asked for, and waited for until granted. False if no recorder answered in time."""
    if not cfg.get("eval"):
        raise PeerError("the GPU is short and no reservation queue is set up (eval)")
    folder = Path(cfg["eval"])
    queue, granted = folder / "queue" / f"{job}.json", folder / "granted" / f"{job}.json"
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(json.dumps({"minutes": minutes, "by": "peer"}))
    print(f"reserving the second PC for {minutes} min: the HOI4 run finishes its game first")
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if granted.exists():
            print("granted")
            return True
        time.sleep(poll)
    queue.unlink(missing_ok=True)
    return False


def release(cfg, job):
    """The reservation done: the HOI4 run resumes."""
    done = Path(cfg["eval"]) / "done" / f"{job}.json"
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps({"ended": time.strftime("%Y-%m-%dT%H:%M:%S")}))


def run(cfg, command, name=None, folder=".", gpu_gb=0.0, minutes=60, detach=False, job=None):
    """Start `command` in the project's folder there, reserving the PC first if the GPU is
    short, and follow its log to the end unless `detach`. The job's exit code."""
    name = name or project_name(folder)
    job = (job or f"{name}-{time.strftime('%m%d-%H%M%S')}")[:40]
    reserved = False
    with Worker(cfg["pairing"]) as worker:
        memory = gpu_memory(worker.job("status"))
        if gpu_gb and memory and memory[0] < gpu_gb * 1024:
            print(f"{memory[0] / 1024:.1f} GB of the GPU free, {gpu_gb:g} GB needed")
            reserved = reserve(cfg, job, minutes)
            if not reserved:
                raise PeerError("no HOI4 recorder granted the PC in time; the GPU is still short")
        try:
            worker.job("start", job=job, kind="project", project=name, args=list(command))
        except BaseException:
            if reserved:
                release(cfg, job)
            raise
    print(f"started {job} in project {name}")
    if detach:
        print(f"follow it with: peer logs {job} --follow")
        if reserved:
            print(f"the reservation ends by itself after {minutes + 15} min")
        return 0
    try:
        state = follow(cfg, job)
    finally:
        if reserved:
            release(cfg, job)
    print(f"\n{job}: {state.get('state')} (exit {state.get('exit')})")
    return state.get("exit") if isinstance(state.get("exit"), int) else 1


def pull(cfg, path, dest=None, name=None, folder="."):
    """A file or folder of the project's there copied back here (to `dest`, or the same
    relative path)."""
    name = name or project_name(folder)
    source = cfg["share"] / "compute" / "projects" / name / path
    dest = Path(dest or path)
    if source.is_dir():
        code = subprocess.run(
            [
                "robocopy",
                str(source),
                str(dest),
                "/E",
                "/R:2",
                "/W:2",
                "/MT:8",
                "/NFL",
                "/NDL",
                "/NP",
            ]
        ).returncode
        if code >= 8:
            raise PeerError(f"pull failed: robocopy exit {code}")
    elif source.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    else:
        raise PeerError(f"no {path} in project {name} there")
    print(f"pulled {path} to {dest}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="peer", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("push", help="copy a project to the second PC")
    p.add_argument("dir", nargs="?", default=".")
    p.add_argument("--name")
    p.add_argument("--mirror", action="store_true", help="also delete what is gone here")
    r = sub.add_parser("run", help="run a command in the project's folder there")
    r.add_argument("--name")
    r.add_argument("--dir", default=".")
    r.add_argument("--gpu-gb", type=float, default=0.0, help="video memory the job needs")
    r.add_argument("--minutes", type=int, default=60, help="how long a reservation lasts")
    r.add_argument("--detach", action="store_true")
    r.add_argument("--no-push", action="store_true")
    r.add_argument("--id")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    lg = sub.add_parser("logs", help="a job's output")
    lg.add_argument("job")
    lg.add_argument("--follow", "-f", action="store_true")
    sub.add_parser("status", help="jobs there, the GPU and free disk")
    s = sub.add_parser("stop", help="end a job")
    s.add_argument("job")
    pl = sub.add_parser("pull", help="copy results back")
    pl.add_argument("path")
    pl.add_argument("dest", nargs="?")
    pl.add_argument("--name")
    pl.add_argument("--dir", default=".")
    sub.add_parser("gpu", help="the GPU's free memory there")
    args = parser.parse_args(argv)
    try:
        cfg = settings()
        if args.command == "push":
            push(cfg, args.dir, args.name, args.mirror)
        elif args.command == "run":
            command = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
            if not command:
                raise PeerError("give the command after --, e.g. peer run -- uv run train.py")
            if not args.no_push:
                push(cfg, args.dir, args.name)
            name = args.name or project_name(args.dir)
            return run(
                cfg, command, name, args.dir, args.gpu_gb, args.minutes, args.detach, args.id
            )
        elif args.command == "logs":
            if args.follow:
                follow(cfg, args.job)
            else:
                log = cfg["share"] / "jobs" / f"{args.job}.log"
                if not log.exists():
                    raise PeerError(f"no log for {args.job}")
                sys.stdout.write(text(log.read_bytes()))
        elif args.command == "status":
            with Worker(cfg["pairing"]) as worker:
                print(worker.job("status"))
        elif args.command == "stop":
            with Worker(cfg["pairing"]) as worker:
                print(worker.job("stop", job=args.job))
        elif args.command == "pull":
            pull(cfg, args.path, args.dest, args.name, args.dir)
        elif args.command == "gpu":
            with Worker(cfg["pairing"]) as worker:
                memory = gpu_memory(worker.job("status"))
            if memory is None:
                raise PeerError("no GPU reading from the second PC")
            print(f"{memory[0] / 1024:.1f} GB free of {memory[1] / 1024:.1f} GB")
    except PeerError as error:
        print(f"peer: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
