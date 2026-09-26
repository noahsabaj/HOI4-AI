"""Sessions run on the second PC itself (on_peer.py): what they take along, the job they run
as there, and the folder they bring back; and the scripts that carry them (Run-Job.ps1's
session commands, Deploy-Peer.ps1 -ComputeOnly)."""

import json
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
