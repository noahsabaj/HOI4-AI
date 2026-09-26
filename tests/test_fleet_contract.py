"""What the fleet project's `fleet` command relies on here, kept working: a `project`
job run by Run-Job.ps1, with its state and log in jobs/. The worker's side (the `job`
op on observer connections) is tested in the worker's crate, and the reservation files
(queue/, granted/, done/) in test_ai_games.py."""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The states a job ends in, as fleet reads them.
FINAL = ("done", "failed", "stopped", "lost")


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
    while time.monotonic() < deadline and state.get("state") not in FINAL:
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
