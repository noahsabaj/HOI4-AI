"""Sessions run on the second PC itself (on_peer.py): what they take along, the job they run
as there, and the folder they bring back; and the scripts that carry them (Run-Job.ps1's
session commands, Deploy-Peer.ps1 -ComputeOnly)."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from hoi4_arena import on_peer

ROOT = Path(__file__).resolve().parents[1]
TOWER = "D:\\Elsewhere\\HOI4-AI\\models\\qwen3-vit-88m"


def _repo(root):
    """The files a practice session reads, as a checkout here has them."""
    for folder in ("artifacts/calibration-1080p", "artifacts/screens-1080p", "models/qwen3-vit-88m",
                   "artifacts/learned/bc9", "artifacts/pairing", "artifacts/arenas"):  # fmt: skip
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "artifacts/calibration-1080p/rules.json").write_text("{}")
    (root / "artifacts/pairing/peer.json").write_text(json.dumps({"host": "second-pc"}))
    (root / "artifacts/arenas/saves-peer.json").write_text("{}")
    (root / "artifacts/learned/bc9/epoch-0000.pt").write_bytes(b"weights")
    (root / "artifacts/learned/bc9/epoch-0000.json").write_text(
        json.dumps({"sha256": "x", "config": {"model_path": TOWER}})
    )


PRACTICE = ["practice", "artifacts/learned/bc9/epoch-0000.pt", "artifacts/learned/practice-9",
            "--peer", "artifacts/pairing/peer.json", "--minutes", "30"]  # fmt: skip


def test_a_session_takes_its_checkpoint_its_tower_and_the_screens_along(tmp_path, monkeypatch):
    _repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    plan = on_peer.session_plan(PRACTICE)
    assert plan["command"] == "practice"
    assert plan["output"] == "artifacts/learned/practice-9"
    assert plan["minutes"] == 30
    assert plan["files"] == [
        "artifacts/pairing/peer.json",
        "artifacts/arenas/saves-peer.json",
        "artifacts/learned/bc9/epoch-0000.pt",
        "artifacts/learned/bc9/epoch-0000.json",
    ]
    # The tower by its folder's name, wherever the training PC kept it.
    assert plan["folders"] == [
        "artifacts/calibration-1080p",
        "artifacts/screens-1080p",
        "models/qwen3-vit-88m",
    ]
    drills = on_peer.session_plan(
        "drills artifacts/drills/d1 --peer artifacts/pairing/peer.json --minutes 30 "
        "--arenas arena-12x8-v4 arena-bay-v6".split()
    )
    assert drills["command"] == "drills" and "models/qwen3-vit-88m" not in drills["folders"]


@pytest.mark.parametrize(
    "change",
    [
        lambda s: [*s, "--rules", "C:/rules.json"],  # a drive
        lambda s: [*s, "--rules", "/rules.json"],  # the root
        lambda s: [*s, "--rules", "../rules.json"],  # out of the repository
        lambda s: [*s, "--seed", "1;calc"],  # anything a shell would read
        lambda s: [*s, "--reservation", "live-1"],  # a booking kept on the second PC
        lambda s: ["record-ai", "out", "--minutes", "5"],  # not a session
    ],
)
def test_what_cannot_run_there_is_refused_here(tmp_path, monkeypatch, change):
    _repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        on_peer.session_plan(change(PRACTICE))


def test_a_file_the_session_needs_must_be_here_to_send(tmp_path, monkeypatch):
    _repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    shutil.rmtree(tmp_path / "models")
    with pytest.raises(FileNotFoundError):
        on_peer.session_plan(PRACTICE)


def test_the_log_is_read_whole_lines_at_a_time_and_to_its_end(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    log, state = jobs / "j.log", jobs / "j.json"
    log.write_bytes(b"one\ntw")
    seen, ticks = [], iter(range(1000))

    def sleep(_):
        # The job goes on: its line ends, a last one without a newline, and it is done.
        log.write_bytes(b"one\ntwo\nthree")
        state.write_text(json.dumps({"state": "done", "exit": 0, "log_bytes": 13}))

    got = on_peer.wait(tmp_path, "j", echo=seen.append, clock=lambda: next(ticks), sleep=sleep)
    assert got["state"] == "done"
    assert seen == ["one", "two", "three"]


class _Second:
    """The second PC's share and its job op, as the test runs them: a job writes its log,
    its state and the session's folder there."""

    def __init__(self, root, output, finish=True):
        self.root, self.output, self.finish = root, output, finish
        self.calls = []
        (root / "jobs").mkdir(parents=True)

    def job(self, peer, action, job_id=None, kind=None, args=None):
        self.calls.append((action, job_id, kind, args))
        state = self.root / "jobs" / f"{job_id}.json"
        if action == "start" and self.finish:
            folder = self.root / "compute" / self.output
            (folder / "practice-peer-1").mkdir(parents=True)
            (folder / "practice-peer-1" / "screen.mkv").write_bytes(b"video")
            (folder / "practice-peer.json").write_text(json.dumps({"summary": {"episodes": 1}}))
            (self.root / "jobs" / f"{job_id}.log").write_text("episode 1\n")
            state.write_text(json.dumps({"state": "done", "exit": 0, "log_bytes": 10}))
        elif action == "start":
            state.write_text(json.dumps({"state": "running"}))
        elif action == "stop":
            state.write_text(json.dumps({"state": "stopped", "exit": 1}))
        return f"{action} {job_id}"


@pytest.fixture
def second(tmp_path, monkeypatch):
    _repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(on_peer, "ROOT", tmp_path.resolve())
    fake = _Second(tmp_path / "share", "artifacts/learned/practice-9")
    monkeypatch.setattr(on_peer, "share", lambda peer: fake.root)
    monkeypatch.setattr(on_peer, "job", fake.job)
    fake.deployed = []
    monkeypatch.setattr(on_peer, "deploy", lambda *a: fake.deployed.append(a))
    monkeypatch.setattr(on_peer, "ensure_environment", lambda peer, root: False)
    return fake


@pytest.mark.skipif(shutil.which("robocopy") is None, reason="robocopy is Windows'")
def test_a_session_runs_there_as_a_job_and_its_folder_comes_back(second, tmp_path):
    # Booked here, as play-policy books: granted by whoever serves the queue.
    (tmp_path / "artifacts/eval/granted").mkdir(parents=True)
    (tmp_path / "artifacts/eval/granted/live-9.json").write_text("{}")
    result = on_peer.run_on_peer(PRACTICE, "artifacts/pairing/peer.json",
                                 reservation="live-9", job_id="practice-t1", poll=0)  # fmt: skip
    assert result["state"] == "done" and result["summary"] == {"episodes": 1}
    # The command runs there exactly as it would here, on its own PC's bridge.
    assert second.calls == [("start", "practice-t1", "run", PRACTICE)]
    files, folders = second.deployed[0][1:]
    assert "artifacts/learned/bc9/epoch-0000.pt" in files and "models/qwen3-vit-88m" in folders
    here = tmp_path / "artifacts/learned/practice-9"
    assert (here / "practice-peer-1" / "screen.mkv").read_bytes() == b"video"
    assert (here / "job-practice-t1.log").read_text() == "episode 1\n"
    # Moved, not copied: the second PC keeps no second copy of the recordings.
    assert not list((second.root / "compute").rglob("screen.mkv"))
    assert json.loads((tmp_path / "artifacts/eval/done/live-9.json").read_text())["job"]


def test_a_session_that_outlasts_its_time_is_stopped_there(second, tmp_path):
    second.finish = False
    session = [*PRACTICE[:-1], "0.001"]
    result = on_peer.run_on_peer(session, "artifacts/pairing/peer.json", job_id="practice-t2",
                                 grace_minutes=0, poll=0)  # fmt: skip
    assert [c[0] for c in second.calls] == ["start", "stop"]
    assert result["state"] == "stopped" and result["summary"] is None


def test_on_peer_runs_from_the_checkout_it_sends(second, monkeypatch, tmp_path):
    monkeypatch.setattr(on_peer, "ROOT", tmp_path / "elsewhere")
    with pytest.raises(RuntimeError, match="checkout"):
        on_peer.run_on_peer(PRACTICE, "artifacts/pairing/peer.json")
    assert second.calls == []


def test_a_checkpoint_finds_its_tower_under_models_on_another_pc(tmp_path, monkeypatch):
    from hoi4_arena.runner import tower_folder

    monkeypatch.chdir(tmp_path)
    assert tower_folder(TOWER) == TOWER  # Not here at all: as named, to fail as before.
    (tmp_path / "models/qwen3-vit-88m").mkdir(parents=True)
    assert Path(tower_folder(TOWER)) == Path("models/qwen3-vit-88m")
    assert tower_folder(str(tmp_path / "models/qwen3-vit-88m")) == str(
        tmp_path / "models/qwen3-vit-88m"
    )


def _pwsh(*arguments):
    return subprocess.run([shutil.which("pwsh"), "-NoProfile", "-File", *map(str, arguments)],
                          capture_output=True, text=True, timeout=120)  # fmt: skip


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs PowerShell 7")
def test_run_job_runs_the_sessions_and_nothing_else_new(tmp_path):
    shutil.copy(ROOT / "scripts" / "Run-Job.ps1", tmp_path)
    (tmp_path / "compute").mkdir()
    for index, command in enumerate(("practice", "drills", "play-policy")):
        spec = json.dumps({"kind": "run", "args": [command, "out"]}).encode().hex()
        started = _pwsh(tmp_path / "Run-Job.ps1", "-Action", "start", "-Id", f"s{index}",
                        "-Spec", spec)  # fmt: skip
        assert started.returncode == 0, started.stderr
    spec = json.dumps({"kind": "run", "args": ["record-ai", "out"]}).encode().hex()
    refused = _pwsh(tmp_path / "Run-Job.ps1", "-Action", "start", "-Id", "r", "-Spec", spec)
    assert refused.returncode != 0 and "command not allowed" in refused.stderr
    # No environment here, so each ends failed; wait for that before the folder goes.
    deadline = time.monotonic() + 60
    states = {}
    while time.monotonic() < deadline and len(states) < 3:
        time.sleep(0.5)
        for index in range(3):
            path = tmp_path / "jobs" / f"s{index}.json"
            if path.exists():
                state = json.loads(path.read_text(encoding="utf-8-sig")).get("state")
                if state in ("done", "failed"):
                    states[index] = state
    assert len(states) == 3


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs PowerShell 7")
def test_deploy_peer_compute_only_leaves_the_worker_alone(tmp_path):
    share = tmp_path / "share"
    share.mkdir()
    peer = tmp_path / "peer.json"
    peer.write_text(json.dumps({"host": "second-pc"}))
    script = ROOT / "scripts" / "Deploy-Peer.ps1"
    first = _pwsh(script, "-ComputeOnly", "-PeerConfig", peer, "-Share", share,
                  "-File", "pyproject.toml")  # fmt: skip
    assert first.returncode == 0, first.stderr
    assert "deployed file pyproject.toml" in first.stdout
    copy = share / "compute" / "pyproject.toml"
    assert copy.read_bytes() == (ROOT / "pyproject.toml").read_bytes()
    # Only compute\ was touched: no worker, bridge script, pairing or mods.
    assert sorted(p.name for p in share.iterdir()) == ["compute"]
    again = _pwsh(script, "-ComputeOnly", "-PeerConfig", peer, "-Share", share,
                  "-File", "pyproject.toml")  # fmt: skip
    assert again.returncode == 0 and "deployed file" not in again.stdout


def test_the_session_commands_are_the_ones_run_job_runs():
    text = (ROOT / "scripts" / "Run-Job.ps1").read_text()
    for command in on_peer.SUMMARIES:
        assert f"'{command}'" in text


@pytest.mark.parametrize(
    "result, nothing",
    [
        (
            {
                "command": "drills",
                "state": "done",
                "exit": 0,
                "summary": {"drills": 60, "complete": 0},
            },
            True,
        ),
        (
            {
                "command": "drills",
                "state": "done",
                "exit": 0,
                "summary": {"drills": 4, "complete": 3},
            },
            False,
        ),
        ({"command": "practice", "state": "done", "exit": 0, "summary": {"episodes": 0}}, True),
        ({"command": "practice", "state": "done", "exit": 0, "summary": {"episodes": 14}}, False),
        ({"command": "play-policy", "state": "done", "exit": 0, "summary": {}}, True),
        ({"command": "play-policy", "state": "done", "exit": 0, "summary": {"games": 2}}, False),
        ({"command": "practice", "state": "done", "exit": 0, "summary": None}, True),
        ({"command": "drills", "state": "failed", "exit": 1, "summary": {"complete": 3}}, True),
    ],
)
def test_a_session_that_played_nothing_is_a_failure(result, nothing):
    assert on_peer.played_nothing(result) is nothing


class _Clock:
    """A clock that `sleep` moves on, and the second PC's side effects at given times."""

    def __init__(self, events=()):
        self.now, self.events = 0.0, sorted(events, key=lambda e: e[0])

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 1.0)
        while self.events and self.events[0][0] <= self.now:
            self.events.pop(0)[1]()


def _running(jobs, job_id="j", log=b"episode 1\n"):
    jobs.mkdir(exist_ok=True)
    (jobs / f"{job_id}.log").write_bytes(log)
    (jobs / f"{job_id}.json").write_text(json.dumps({"id": job_id, "state": "running", "pid": 7}))


def test_an_end_left_in_a_stranded_tmp_ends_the_wait(tmp_path):
    # The bridge restarted as the job ended: its last write never became the state file.
    _running(tmp_path / "jobs")
    end = {"id": "j", "state": "done", "exit": 0, "log_bytes": 10}
    (tmp_path / "jobs" / "j.tmp").write_text(json.dumps(end))
    clock = _Clock()
    got = on_peer.wait(tmp_path, "j", deadline=3600, poll=10, clock=clock, sleep=clock.sleep)
    assert got["state"] == "done" and got["exit"] == 0
    assert clock.now < 60


def test_a_half_written_tmp_is_not_an_end(tmp_path):
    _running(tmp_path / "jobs")
    (tmp_path / "jobs" / "j.tmp").write_text('{"id": "j", "state": "do')
    clock = _Clock()
    got = on_peer.wait(tmp_path, "j", deadline=100, poll=10, clock=clock, sleep=clock.sleep)
    assert got["state"] == "running"


def test_a_running_job_whose_process_is_gone_ends_the_wait_as_lost(tmp_path):
    _running(tmp_path / "jobs")
    asked = []

    def check(job_id):
        asked.append(clock.now)
        return "running" if len(asked) < 2 else "lost"

    clock = _Clock()
    got = on_peer.wait(tmp_path, "j", deadline=7200, poll=10, check=check, check_after=120,
                       clock=clock, sleep=clock.sleep)  # fmt: skip
    assert got["state"] == "lost" and "process is gone" in got["why"]
    # Asked only once the log was quiet, then no more often than every check_after.
    assert asked[0] >= 120 and asked[1] - asked[0] >= 120
    assert clock.now < 400


def test_a_growing_log_is_not_asked_about(tmp_path):
    jobs = tmp_path / "jobs"
    _running(jobs)
    lines = [b"episode 1\n"]

    def grow():
        lines.append(b"step\n")
        (jobs / "j.log").write_bytes(b"".join(lines))

    clock = _Clock([(t, grow) for t in range(10, 1000, 30)])
    asked = []
    got = on_peer.wait(tmp_path, "j", deadline=900, poll=10, check=asked.append,
                       check_after=120, clock=clock, sleep=clock.sleep)  # fmt: skip
    assert got["state"] == "running" and asked == []


def test_a_quiet_job_is_lost_only_when_its_process_cannot_be_asked(tmp_path):
    jobs = tmp_path / "jobs"
    summary = tmp_path / "compute" / "out" / "practice-peer.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"summary": {"episodes": 3}}))

    def unreachable(job_id):
        raise ConnectionError("the bridge is restarting")

    _running(jobs)
    clock = _Clock()
    got = on_peer.wait(tmp_path, "j", deadline=7200, poll=10, check=unreachable,
                       summary_path=summary, quiet=600, clock=clock, sleep=clock.sleep)  # fmt: skip
    assert got["state"] == "lost" and "have not changed for 10 min" in got["why"]
    assert 600 <= clock.now < 700
    # A process there that still runs is waited for, however quiet.
    clock = _Clock()
    got = on_peer.wait(tmp_path, "j", deadline=3000, poll=10, check=lambda j: "running",
                       summary_path=summary, quiet=600, clock=clock, sleep=clock.sleep)  # fmt: skip
    assert got["state"] == "running"
    # No summary yet: nothing played to go by, so waited for too.
    summary.unlink()
    clock = _Clock()
    got = on_peer.wait(tmp_path, "j", deadline=3000, poll=10, check=unreachable,
                       summary_path=summary, quiet=600, clock=clock, sleep=clock.sleep)  # fmt: skip
    assert got["state"] == "running"


STATUS_TABLE = """
id                        kind project state   exit started                  ended
--                        ---- ------- -----   ---- -------                  -----
drills-20260926-101500    run          done       0 9/26/2026 10:15:00 AM    9/26/2026 10:45:00 AM
drills-20260926-114500    run          lost         9/26/2026 11:45:00 AM
practice-20260926-120000  run          running      9/26/2026 12:00:00 PM

free disk: 400 GB
environment: ready
"""


def test_the_status_says_which_jobs_still_run_there():
    active = 'active: [{"id":"a","state":"lost","pid":7},{"id":"b","state":"running","pid":8}]'
    assert on_peer.listed_state(STATUS_TABLE + active, "a") == "lost"
    assert on_peer.listed_state(STATUS_TABLE + active, "b") == "running"
    assert on_peer.listed_state(STATUS_TABLE + active, "drills-20260926-114500") is None
    assert on_peer.listed_state('active: {"id":"a","state":"lost","pid":7}', "a") == "lost"
    # A Run-Job from before the active line: its table.
    assert on_peer.listed_state(STATUS_TABLE, "drills-20260926-114500") == "lost"
    assert on_peer.listed_state(STATUS_TABLE, "practice-20260926-120000") == "running"
    assert on_peer.listed_state(STATUS_TABLE, "drills-20260926-101500") is None
    assert on_peer.listed_state(STATUS_TABLE, "absent") is None


def test_a_session_whose_process_is_lost_there_is_not_waited_out(second, monkeypatch):
    second.finish = False
    monkeypatch.setattr(on_peer, "job_state_there", lambda peer, job_id: "lost")
    clock = _Clock()
    real_wait = on_peer.wait
    monkeypatch.setattr(
        on_peer, "wait",
        lambda *a, **k: real_wait(*a, **{**k, "clock": clock, "sleep": clock.sleep}),
    )  # fmt: skip
    result = on_peer.run_on_peer(PRACTICE, "artifacts/pairing/peer.json", job_id="practice-t3",
                                 poll=10)  # fmt: skip
    assert result["state"] == "lost" and "process is gone" in result["why"]
    # Ended there already: nothing to stop.
    assert [c[0] for c in second.calls] == ["start"]
    assert on_peer.played_nothing(result)
    assert clock.now < 400


def _stranded_jobs(folder):
    jobs = folder / "jobs"
    jobs.mkdir()
    started = "2026-09-26T11:45:00.0000000+00:00"
    running = {"kind": "run", "state": "running", "started": started}
    # A running job whose process is gone, its end stranded in a .tmp an hour old.
    (jobs / "gone.json").write_text(json.dumps({**running, "id": "gone", "pid": 999999}))
    end = {**running, "id": "gone", "state": "done", "exit": 0, "log_bytes": 3, "ended": started}
    (jobs / "gone.tmp").write_text(json.dumps(end))
    hour_ago = time.time() - 3600
    os.utime(jobs / "gone.tmp", (hour_ago, hour_ago))
    # One whose process is gone with no end at all, and one whose end is being written now.
    (jobs / "cut.json").write_text(json.dumps({**running, "id": "cut", "pid": 999998}))
    (jobs / "fresh.json").write_text(json.dumps({**running, "id": "fresh", "pid": 999997}))
    (jobs / "fresh.tmp").write_text(json.dumps({**end, "id": "fresh"}))
    return jobs


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs PowerShell 7")
def test_run_job_status_finishes_a_stranded_end_and_reports_lost_jobs(tmp_path):
    shutil.copy(ROOT / "scripts" / "Run-Job.ps1", tmp_path)
    jobs = _stranded_jobs(tmp_path)
    status = _pwsh(tmp_path / "Run-Job.ps1", "-Action", "status")
    assert status.returncode == 0, status.stderr
    # The stranded end became the state file; a .tmp still being written was left alone.
    assert not (jobs / "gone.tmp").exists()
    assert json.loads((jobs / "gone.json").read_text(encoding="utf-8-sig"))["state"] == "done"
    assert (jobs / "fresh.tmp").exists()
    assert on_peer.listed_state(status.stdout, "cut") == "lost"
    assert on_peer.listed_state(status.stdout, "fresh") == "lost"
    assert on_peer.listed_state(status.stdout, "gone") is None
    # Stopping a job that has ended keeps its end.
    stop = _pwsh(tmp_path / "Run-Job.ps1", "-Action", "stop", "-Id", "gone")
    assert stop.returncode == 0 and "already done" in stop.stdout
    assert json.loads((jobs / "gone.json").read_text(encoding="utf-8-sig"))["state"] == "done"
