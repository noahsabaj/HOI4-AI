import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("peer", ROOT / "tools" / "peer" / "peer.py")
peer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(peer)

STATUS = (
    "id  kind  project  state\n"
    "name, driver_version, memory.used [MiB], memory.total [MiB], utilization.gpu [%]\n"
    "NVIDIA GeForce RTX 4060 Ti, 581.29, 3120 MiB, 8188 MiB, 23 %\n"
    "free disk: 684 GB\n"
)


class Stream:
    """The worker's side of a connection: the replies it will send, and what it got."""

    def __init__(self, replies):
        self.data = io.BytesIO(b"".join(replies))
        self.sent = []

    def write(self, data):
        self.sent.append(json.loads(data))

    def flush(self):
        pass

    def readline(self):
        return self.data.readline()

    def read(self, size):
        return self.data.read(size)


def test_a_project_s_name_there_is_its_folder_s_in_safe_letters(tmp_path):
    assert peer.project_name(tmp_path / "stocks") == "stocks"
    assert peer.project_name(tmp_path / "My Stock Model!") == "My-Stock-Model"
    assert len(peer.project_name(tmp_path / ("x" * 60))) == 40


def test_the_gpu_s_free_memory_comes_from_the_job_status():
    assert peer.gpu_memory(STATUS) == (8188 - 3120, 8188)
    assert peer.gpu_memory("free disk: 684 GB\n") is None


def test_requests_skip_other_streams_and_raise_the_worker_s_errors():
    stream = Stream(
        [
            b'{"stream": "view", "data": 0, "bytes": 3}\n',
            b"abc",  # A view's bytes on the same connection: skipped.
            b'{"id": 1, "output": "' + STATUS.encode().replace(b"\n", b"\\n") + b'", "exit": 0}\n',
            b'{"id": 2, "output": "no project nope here: push it first", "exit": 1}\n',
            b'{"id": 3, "error": "invalid_project"}\n',
        ]
    )
    worker = peer.Worker(None, stream=stream)
    assert peer.gpu_memory(worker.job("status")) == (5068, 8188)
    with pytest.raises(peer.PeerError, match="push it first"):
        worker.job("start", job="j", kind="project", project="nope", args=["uv"])
    with pytest.raises(peer.PeerError, match="invalid_project"):
        worker.job("start", job="j", kind="project", project="../x", args=["uv"])
    assert stream.sent[1] == {
        "op": "job", "id": 2, "action": "start", "job": "j", "kind": "project",
        "project": "nope", "args": ["uv"],
    }  # fmt: skip


def test_a_push_copies_new_files_and_deletes_only_when_asked(tmp_path):
    command = peer.push_command(tmp_path / "proj", Path("//pc/HOI4Worker/compute/projects/proj"))
    assert command[0] == "robocopy" and "/E" in command and "/MIR" not in command
    skipped = command[command.index("/XD") + 1 : command.index("/XF")]
    assert {".git", ".venv", "__pycache__"} <= set(skipped)
    assert "/MIR" in peer.push_command(tmp_path, tmp_path, mirror=True)


def test_a_reservation_waits_for_the_grant_and_its_end_lets_the_run_resume(tmp_path):
    cfg = {"eval": str(tmp_path)}

    def grant():
        time.sleep(0.2)
        (tmp_path / "granted").mkdir()
        (tmp_path / "granted" / "big.json").write_text("{}")

    threading.Thread(target=grant).start()
    assert peer.reserve(cfg, "big", 45, wait=5, poll=0.05)
    assert json.loads((tmp_path / "queue" / "big.json").read_text())["minutes"] == 45
    peer.release(cfg, "big")
    assert (tmp_path / "done" / "big.json").exists()
    # No recorder answering: the request is taken back.
    assert not peer.reserve(cfg, "lonely", 10, wait=0.1, poll=0.05)
    assert not (tmp_path / "queue" / "lonely.json").exists()


def test_following_a_job_prints_its_log_until_it_ends(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    cfg = {"share": tmp_path}
    (jobs / "j.json").write_text(json.dumps({"state": "running"}), encoding="utf-8-sig")

    def work():
        for line in ("epoch 1\n", "epoch 2\n"):
            with open(jobs / "j.log", "a") as log:
                log.write(line)
            time.sleep(0.1)
        # Ended, with the log's whole length, while the share still shows it short (it
        # caches a file's size): the last line comes into view only after the end.
        length = (jobs / "j.log").stat().st_size + len("saved\n".encode()) + (len(os.linesep) - 1)
        final = {"state": "done", "exit": 0, "log_bytes": length}
        (jobs / "j.json").write_text(json.dumps(final), encoding="utf-8-sig")
        time.sleep(0.3)
        with open(jobs / "j.log", "a") as log:
            log.write("saved\n")

    threading.Thread(target=work).start()
    out = io.StringIO()
    assert peer.follow(cfg, "j", poll=0.05, out=out)["state"] == "done"
    assert out.getvalue() == "epoch 1\nepoch 2\nsaved\n"


def test_a_job_the_gpu_cannot_hold_reserves_the_pc_first(monkeypatch):
    events = []

    class FakeWorker:
        def __init__(self, spec):
            pass

        def job(self, action, **fields):
            events.append((action, fields.get("kind")))
            return STATUS

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(peer, "Worker", FakeWorker)
    monkeypatch.setattr(peer, "reserve", lambda cfg, job, minutes: events.append("reserve") or True)
    monkeypatch.setattr(peer, "release", lambda cfg, job: events.append("release"))
    monkeypatch.setattr(peer, "follow", lambda cfg, job: {"state": "done", "exit": 0})
    cfg = {"pairing": {}, "eval": "q"}
    # 4.9 GB free: a 4 GB job shares the GPU with the game.
    assert peer.run(cfg, ["uv", "run", "train.py"], name="stocks", gpu_gb=4) == 0
    assert events == [("status", None), ("start", "project")]
    # An 6 GB job does not fit: the PC is reserved, and given back when the job ends.
    events.clear()
    assert peer.run(cfg, ["uv", "run", "train.py"], name="stocks", gpu_gb=6) == 0
    assert events == [("status", None), "reserve", ("start", "project"), "release"]


def test_run_needs_a_command(monkeypatch, capsys):
    monkeypatch.setattr(peer, "settings", lambda: {"pairing": {}, "share": Path("."), "eval": None})
    assert peer.main(["run", "--no-push", "--"]) == 2
    assert "give the command" in capsys.readouterr().err


def run_job(folder, *arguments):
    shell = shutil.which("pwsh")
    return subprocess.run(
        [shell, "-NoProfile", "-File", str(folder / "Run-Job.ps1"), *arguments],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs PowerShell 7")
def test_run_job_runs_a_project_s_own_command_in_its_folder(tmp_path):
    shutil.copy(ROOT / "scripts" / "Run-Job.ps1", tmp_path)
    project = tmp_path / "compute" / "projects" / "demo"
    project.mkdir(parents=True)
    code = "import os, sys; print('ran in', os.path.basename(os.getcwd()), sys.argv[1:])"
    spec = {
        "kind": "project",
        "project": "demo",
        "args": [sys.executable, "-c", code, "a b", "x;y"],
    }
    started = run_job(
        tmp_path, "-Action", "start", "-Id", "t1", "-Spec", json.dumps(spec).encode().hex()
    )
    assert started.returncode == 0, started.stderr
    state = {}
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and state.get("state") not in peer.FINAL:
        time.sleep(0.5)
        path = tmp_path / "jobs" / "t1.json"
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8-sig"))
    assert state.get("state") == "done" and state.get("project") == "demo"
    assert state.get("log_bytes") == (tmp_path / "jobs" / "t1.log").stat().st_size
    log = (tmp_path / "jobs" / "t1.log").read_text(encoding="utf-8", errors="replace")
    assert "ran in demo ['a b', 'x;y']" in log
    # A project that was never pushed, or a name that is a path, is refused at once.
    for name in ("absent", "..\\escape"):
        bad = {"kind": "project", "project": name, "args": ["uv"]}
        refused = run_job(
            tmp_path, "-Action", "start", "-Id", "t2", "-Spec", json.dumps(bad).encode().hex()
        )
        assert refused.returncode != 0
    assert not (tmp_path / "jobs" / "t2.json").exists()
