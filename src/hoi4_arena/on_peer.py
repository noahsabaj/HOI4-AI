"""Sessions on the second PC, run there: its own Python plays its own game with the policy on
its own GPU, and this PC's GPU is left to training.

Practice, drills and play-policy drive the second PC's game through its worker. Run here,
the policy decides on this PC's GPU, beside a training run: a decision took ~80 ms with a
trainer sharing the card against ~40 alone, evaluations paused training outright, and the
trainer was capped at half the card to leave room for them. The second PC has the same
card, and HOI4 leaves it about half free.

`run_on_peer` sends the session there as a compute job (the worker's `job` op,
Run-Job.ps1): the code, the checkpoint and its tower, the screen templates and rules, the
start saves' registry and the pairing file go into the share's compute folder at the same
paths (Deploy-Peer.ps1 -ComputeOnly), and the command runs there exactly as it would here.
Its `--peer` reaches the bridge on its own PC, which since this change takes connections
from its own address too, so the session holds the game as a session from here does: a
recording from here is told worker_busy meanwhile, a new worker is swapped in only
between connections, and the live view's observer watches as before. This waits for the
job, prints its log as it grows, and brings the session's folder back to the same path
here, where training's dataset builders look.

The reservation files stay here (artifacts/eval: queue/, granted/, done/), the fleet
project's interface: `--reservation` books the second PC here before the job and hands it
back after, and between sessions a loop here serves the queue as before
(ai_games.take_reservation, lend). A session given `--reservation` itself would book on
the second PC's copy of the folder, which no one reads, so that is refused.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path, PureWindowsPath

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
# The sessions Run-Job.ps1 runs as jobs ($Commands), and the summary each writes.
SUMMARIES = {"practice": "practice-peer.json", "drills": "drills-peer.json",
             "play-policy": "results-peer.json"}  # fmt: skip
# A job argument as the worker and Run-Job check it: a flag, a value or a relative path.
ARGUMENT = re.compile(r"^[A-Za-z0-9_.=+-][A-Za-z0-9_./=+-]{0,199}$")
FINAL = ("done", "failed", "stopped", "lost")
# What every session reads beside its checkpoint and rules: the screen templates (the
# start, the popups, the scripted player's and the coach's) and the start saves' names.
TEMPLATES = "artifacts/screens-1080p"
SAVES = "artifacts/arenas/saves-peer.json"
# The Python environment there was built from this lock (a copy, written after its setup).
LOCK_STAMP = ".venv-lock.sha256"


def share(peer):
    """The second PC's shared worker folder, as Deploy-Peer.ps1 finds it."""
    return Path(rf"\\{json.loads(Path(peer).read_text())['host']}\HOI4Worker")


def session_plan(session):
    """What `session` (a hoi4-arena command line) needs there and leaves behind: its
    command, output folder, time budget, and the files and folders to send."""
    from .cli import build_parser

    for argument in session:
        if not ARGUMENT.match(argument) or ".." in argument:
            raise ValueError(
                f"{argument!r} cannot cross to the second PC: a job's arguments are flags, "
                "values and paths inside the repository (letters, digits and _ . / = + -)"
            )
    parsed = build_parser().parse_args(session)
    command = parsed.command
    if command not in SUMMARIES:
        raise ValueError(f"on-peer runs {', '.join(SUMMARIES)}, not {command}")
    if parsed.reservation:
        raise ValueError(
            "--reservation belongs to on-peer, not the session: the bookings are kept here"
        )
    files = [parsed.peer]
    if Path(SAVES).exists():
        files.append(SAVES)
    folders = [str(Path(parsed.rules).parent.as_posix()), TEMPLATES]
    checkpoint = getattr(parsed, "checkpoint", None)
    if checkpoint:
        files += [checkpoint, Path(checkpoint).with_suffix(".json").as_posix()]
    for path in files + folders:
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} is not here to send")
    if checkpoint:
        tower = parsed.model_path or tower_of(files[-1])
        if not Path(tower).is_dir():
            raise FileNotFoundError(f"{tower}, the checkpoint's tower, is not here to send")
        folders.append(tower)
    return {
        "command": command,
        "output": parsed.output,
        "minutes": parsed.minutes,
        "files": list(dict.fromkeys(files)),
        "folders": list(dict.fromkeys(folders)),
    }


def tower_of(manifest):
    """The folder under models/ of the tower a checkpoint names (by its absolute path on
    the PC that trained it; runner.tower_folder finds it there the same way)."""
    named = json.loads(Path(manifest).read_text())["config"]["model_path"]
    return f"models/{PureWindowsPath(named).name}"


def quoted(text):
    return "'" + str(text).replace("'", "''") + "'"


def deploy(peer, files, folders, root=ROOT, run=subprocess.run):
    """The code, `folders` (mirrored) and `files` (copied when changed) into the share's
    compute folder, and nothing else: Deploy-Peer.ps1 -ComputeOnly."""
    command = [f"& {quoted(Path(root) / 'scripts' / 'Deploy-Peer.ps1')}", "-ComputeOnly",
               "-Compute", "-PeerConfig", quoted(Path(peer).resolve())]  # fmt: skip
    if folders:
        command += ["-Data", ",".join(quoted(f) for f in folders)]
    if files:
        command += ["-File", ",".join(quoted(f) for f in files)]
    # -Command, not -File: through -File a list reaches the script as one string.
    shell = shutil.which("pwsh") or "pwsh"
    done = run([shell, "-NoProfile", "-Command", " ".join(command)],
               capture_output=True, text=True)  # fmt: skip
    if done.returncode:
        raise RuntimeError(f"Deploy-Peer failed: {done.stdout}{done.stderr}")
    for line in done.stdout.splitlines():
        if line.startswith("deployed"):
            log.info("[peer] %s", line)


def job(peer, action, job_id=None, kind=None, args=None):
    """A compute job operation on an observer connection, which a game holding the full
    connection leaves free."""
    from .remote import RemoteDesktop

    with RemoteDesktop(peer, attach=False, observer=True) as desk:
        return desk.job(action, job_id, kind, args)


def read_state(root, job_id, suffix=".json"):
    """A job's state file from the share, or {} while it is not there or half-seen."""
    try:
        state = json.loads((root / "jobs" / f"{job_id}{suffix}").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def listed_state(status, job_id):
    """The state Run-Job's status (`status`, its text) gives the job, as its process shows
    it: lost when the process is gone. None when the status does not list it as active.
    Its `active:` line says; a Run-Job from before that line has only its table, whose
    row for the job starts with the id and has the state among its words."""
    table = None
    for line in status.splitlines():
        if line.startswith("active: "):
            try:
                active = json.loads(line[len("active: ") :])
            except ValueError:
                break
            if isinstance(active, dict):  # One job, from a PowerShell that unwrapped it.
                active = [active]
            for entry in active:
                if isinstance(entry, dict) and entry.get("id") == job_id:
                    return entry.get("state")
            return None
        words = line.split()
        if words and words[0] == job_id and table is None:
            table = next((w for w in words[1:] if w in ("starting", "running", *FINAL)), None)
    return table if table in ("starting", "running", "lost") else None


def job_state_there(peer, job_id):
    """The job's state as its process shows it there (listed_state), through the worker's
    `job` op: the share alone cannot tell whether a process runs."""
    return listed_state(job(peer, "status"), job_id)


def wait(root, job_id, *, deadline=None, poll=10.0, echo=None, check=None, summary_path=None,
         check_after=120.0, quiet=600.0, clock=time.monotonic, sleep=time.sleep):  # fmt: skip
    """Until the job ends: its last state. Each new line of its log goes to `echo`. At
    `deadline` (a `clock` time) the state so far, which is not final.

    A job can end without its state file saying so. On 2026-09-26 the second PC's bridge
    restarted as a job ended: its file stayed `running` beside the `.tmp` its last write
    left, and this would have waited out the deadline, the session's minutes and half an
    hour more. So an end left in the `.tmp` counts. And once the log has been quiet for
    `check_after` seconds, `check(job_id)` (job_state_there) is asked, every
    `check_after` seconds, whether the process still runs: lost if it is gone. While
    `check` cannot say (the bridge is down, say), a job whose log and summary
    (`summary_path`, which a session writes after each episode) have both been still for
    `quiet` seconds is taken as lost too. The state returned says `why`."""
    log_path, offset = root / "jobs" / f"{job_id}.log", 0
    seen, changed = {}, {}
    last_check, verdict = None, None

    def still_for(name, path):
        """Seconds since `path` last changed as seen from here (its size or time)."""
        try:
            status = path.stat()
            signature = (status.st_size, status.st_mtime_ns)
        except OSError:
            signature = None
        now = clock()
        if name not in seen or signature != seen[name]:
            seen[name], changed[name] = signature, now
        return now - changed[name], signature is not None

    def ended_there():
        """Why the job, recorded as not ended, has ended, or None."""
        nonlocal last_check, verdict
        quiet_log, _ = still_for("log", log_path)
        if check is not None and quiet_log >= check_after:
            now = clock()
            if last_check is None or now - last_check >= check_after:
                last_check = now
                try:
                    verdict = check(job_id)
                except Exception as error:  # noqa: BLE001 - the bridge may be restarting.
                    log.warning("[peer] cannot ask whether %s still runs: %s", job_id, error)
                    verdict = None
                if verdict == "lost":
                    return "its process is gone there, and it never recorded an end"
        if verdict in ("starting", "running") or summary_path is None:
            return None
        quiet_summary, written = still_for("summary", summary_path)
        if written and min(quiet_log, quiet_summary) >= quiet:
            return (
                f"its log and summary have not changed for {quiet / 60:.0f} min and "
                "whether its process runs cannot be asked"
            )
        return None

    def drain(whole=False):
        nonlocal offset
        try:
            with log_path.open("rb") as handle:
                handle.seek(offset)
                data = handle.read()
        except OSError:
            return
        if not whole:  # A line still being written waits for its end.
            data = data[: data.rfind(b"\n") + 1]
        offset += len(data)
        for line in data.decode("utf-8", errors="replace").splitlines():
            if echo and line.strip():
                echo(line)

    while True:
        state = read_state(root, job_id)
        drain()
        if state.get("state") not in FINAL:
            stranded = read_state(root, job_id, ".tmp")
            if stranded.get("state") in FINAL:
                log.warning("[peer] %s ended %s, left in %s.tmp by a write cut short",
                            job_id, stranded["state"], job_id)  # fmt: skip
                state = stranded
            elif state.get("state") in ("starting", "running"):
                why = ended_there()
                if why:
                    log.warning("[peer] %s taken as lost: %s", job_id, why)
                    state = {**state, "state": "lost", "why": why}
        if state.get("state") in FINAL:
            # The share shows a growing file's size up to ~10 s late: read to log_bytes.
            end = clock() + 30
            while True:
                drain(whole=True)
                if offset >= state.get("log_bytes", 0) or clock() > end:
                    return state
                sleep(1.0)
        if deadline is not None and clock() > deadline:
            return state
        sleep(poll)


def lock_hash(root=ROOT):
    return hashlib.sha256((Path(root) / "uv.lock").read_bytes()).hexdigest()


def ensure_environment(peer, root, *, wait_minutes=30.0):
    """The Python environment there built from this uv.lock, by a setup job if it was not
    (uv sync; a few seconds when little changed)."""
    compute = root / "compute"
    stamp = compute / LOCK_STAMP
    want = lock_hash()
    ready = (compute / ".venv" / "Scripts" / "python.exe").exists()
    if ready and stamp.exists() and stamp.read_text().strip() == want:
        return False
    job_id = time.strftime("setup-%Y%m%d-%H%M%S")
    log.info("[peer] setting up the Python environment there (%s)", job_id)
    job(peer, "start", job_id, "setup", [])
    state = wait(root, job_id, deadline=time.monotonic() + wait_minutes * 60, poll=5.0,
                 echo=lambda line: log.info("[peer setup] %s", line))  # fmt: skip
    if state.get("state") != "done":
        raise RuntimeError(f"the environment's setup there ended {state.get('state')}")
    stamp.write_text(want)
    return True


def bring_back(root, output, keep_there=False):
    """The session's folder from the share to the same path here. Moved, unless
    `keep_there`: robocopy deletes each file there once it is copied."""
    source = root / "compute" / Path(output)
    if not source.exists():
        return False
    Path(output).mkdir(parents=True, exist_ok=True)
    flags = ["/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP"] + ([] if keep_there else ["/MOVE"])
    done = subprocess.run(["robocopy", str(source), str(Path(output)), *flags],
                          capture_output=True, text=True)  # fmt: skip
    # Robocopy exit codes below 8 are success.
    if done.returncode >= 8:
        raise RuntimeError(f"bringing {output} back failed (robocopy {done.returncode})")
    return True


def summary(path):
    """A session's summary from the file it wrote: practice's and drills' own, or
    play-policy's record from its results."""
    try:
        written = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if isinstance(written, list):
        from .scripted import win_rate

        return win_rate(written) if written else {}
    return written.get("summary")


def played_nothing(result):
    """Whether a session ended without one episode or game played: its job failed, or its
    summary counts nothing done. A loop of sessions then stops or waits instead of starting
    the next one at once: on 2026-09-26 every drill failed in a second for want of a file
    there, and a loop started 186 empty sessions in 50 minutes."""
    if result.get("state") != "done" or result.get("exit"):
        return True
    done = result.get("summary")
    if not done:
        return True
    # play-policy's summary is its record, which has games only if some were played.
    key = {"practice": "episodes", "drills": "complete"}.get(result.get("command"))
    return key is not None and not done.get(key)


def run_on_peer(
    session,
    peer,
    *,
    reservation=None,
    job_id=None,
    deploy_first=True,
    keep_there=False,
    grace_minutes=30.0,
    poll=10.0,
):
    """Run `session` (e.g. ["practice", CKPT, OUT, "--peer", PEER, "--minutes", "30"]) on
    the second PC and wait for it; return the job's end and the session's summary. Its
    folder comes back to OUT here. Stopped there if it outlasts its --minutes by
    `grace_minutes`, or if this is interrupted."""
    from .play import hand_back, reserve

    if Path.cwd().resolve() != ROOT:
        # Deploy-Peer sends its own checkout's code and finds the session's files there.
        raise RuntimeError(f"run on-peer from the checkout it comes from, {ROOT}")
    plan = session_plan(session)
    root = share(peer)
    if not root.exists():
        raise RuntimeError(f"{root} cannot be reached (README.md, Second PC)")
    job_id = job_id or f"{plan['command']}-{time.strftime('%Y%m%d-%H%M%S')}"
    if reservation:
        reserve(reservation, plan["minutes"] + grace_minutes)
    result = {"job": job_id, "command": plan["command"], "output": plan["output"]}
    try:
        if deploy_first:
            deploy(peer, plan["files"], plan["folders"])
            ensure_environment(peer, root)
        log.info("[peer] %s", job(peer, "start", job_id, "run", list(session)))
        deadline = time.monotonic() + (plan["minutes"] + grace_minutes) * 60
        summary_there = root / "compute" / plan["output"] / SUMMARIES[plan["command"]]
        try:
            state = wait(root, job_id, deadline=deadline, poll=poll,
                         echo=lambda line: log.info("[peer] %s", line),
                         check=lambda j: job_state_there(peer, j),
                         summary_path=summary_there)  # fmt: skip
        except KeyboardInterrupt:
            log.warning("[peer] interrupted: stopping %s there", job_id)
            job(peer, "stop", job_id)
            raise
        if state.get("state") not in FINAL:
            log.warning("[peer] %s outlasted its time: stopping it", job_id)
            job(peer, "stop", job_id)
            state = wait(root, job_id, deadline=time.monotonic() + 60, poll=5.0)
        result.update(state=state.get("state"), exit=state.get("exit"))
        if state.get("why"):
            result["why"] = state["why"]
    finally:
        try:
            if bring_back(root, plan["output"], keep_there):
                shutil.copyfile(root / "jobs" / f"{job_id}.log",
                                Path(plan["output"]) / f"job-{job_id}.log")  # fmt: skip
        except (OSError, RuntimeError) as error:
            log.warning("[peer] %s stays on the second PC: %s", plan["output"], error)
            result["left_there"] = str(root / "compute" / plan["output"])
        result["summary"] = summary(Path(plan["output"]) / SUMMARIES[plan["command"]])
        if reservation:
            hand_back(reservation, {"job": job_id, "summary": result["summary"]})
    return result
