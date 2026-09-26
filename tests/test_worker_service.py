"""The desktop worker as a fleet service on the second PC (scripts/Start-Worker.ps1
-Service, shipped by scripts/collect_station.py deploy --worker), and its clients: the
station there on the loopback, and this PC through `fleet tunnel`, which connect again
while the service or the tunnel restarts."""

import importlib.util
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hoi4_arena import remote

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Clients connect again.


class Context:
    def __init__(self, fails=0):
        self.fails, self.wrapped = fails, 0

    def wrap_socket(self, raw, server_hostname=None):
        self.wrapped += 1
        if self.fails:
            self.fails -= 1
            raise ssl.SSLEOFError("closed during the handshake: no bridge behind the tunnel")
        return ("tls", raw)


class Raw:
    closed = 0

    def close(self):
        Raw.closed += 1


def test_a_worker_that_is_not_there_yet_is_waited_for(monkeypatch):
    tries = []

    def connect(address, timeout):
        tries.append(address)
        if len(tries) <= 2:
            raise ConnectionRefusedError("the tunnel is down")
        return Raw()

    monkeypatch.setattr(remote.socket, "create_connection", connect)
    Raw.closed = 0
    context = Context(fails=1)
    spec = {"host": "127.0.0.1", "port": 47941}
    tls = remote.open_tls(spec, context, wait=5, pause=0.01)
    assert tls[0] == "tls" and len(tries) == 4, "refused twice, reset once, then in"
    assert Raw.closed == 1, "the socket of a failed handshake is closed"


def test_without_a_wait_the_first_failure_is_raised_as_before(monkeypatch):
    tries = []

    def refused(address, timeout):
        tries.append(address)
        raise ConnectionRefusedError("nobody there")

    monkeypatch.setattr(remote.socket, "create_connection", refused)
    with pytest.raises(ConnectionRefusedError):
        remote.open_tls({"host": "second-pc", "port": 1}, Context(), wait=0, pause=0.01)
    assert len(tries) == 1
    started = time.monotonic()
    with pytest.raises(ConnectionRefusedError):
        remote.open_tls({"host": "127.0.0.1", "port": 1}, Context(), wait=0.3, pause=0.05)
    assert 0.2 < time.monotonic() - started < 3


def test_the_pairing_says_how_long_to_wait_and_a_caller_may_say_otherwise(tmp_path, monkeypatch):
    asked = []

    def open_tls(spec, context, wait=0.0, **_):
        asked.append(wait)
        raise ConnectionRefusedError("stop here")

    monkeypatch.setattr(remote, "open_tls", open_tls)
    lan, fleet = tmp_path / "peer.json", tmp_path / "peer-fleet.json"
    lan.write_text(json.dumps({"host": "second-pc", "port": 47941}))
    fleet.write_text(json.dumps({"host": "127.0.0.1", "port": 47941, "reconnect_seconds": 60}))
    for config, wait in ((lan, None), (fleet, None), (fleet, 0)):
        with pytest.raises(ConnectionRefusedError):
            remote.RemoteDesktop(config, attach=False, observer=True, wait=wait)
    assert asked == [0.0, 60.0, 0]
    assert remote.is_loopback({"host": "127.0.0.1"}) and remote.is_loopback({"host": "::1"})
    assert not remote.is_loopback({"host": "second-pc"}) and not remote.is_loopback({})


# The station uses the service once it has shipped.


def test_the_station_uses_the_fleet_pairing_once_it_is_there(tmp_path, monkeypatch):
    station = load_script("station")
    monkeypatch.chdir(tmp_path)
    assert station.peer() == station.PEER, "the old bridge until the cutover"
    Path(station.FLEET_PEER).parent.mkdir(parents=True)
    Path(station.FLEET_PEER).write_text("{}")
    assert station.peer() == station.FLEET_PEER
    ran = []
    monkeypatch.setattr(station, "FINISHED", tmp_path / "finished.jsonl")
    station.session("drills", "out", [], run=lambda *a: ran.append(a) or 1)
    assert ran == [("drills", "out", "--peer", station.FLEET_PEER)]


# Deploying the worker.


@pytest.fixture
def collect(tmp_path, monkeypatch):
    module = load_script("collect_station")
    root, stage = tmp_path / "repo", tmp_path / "repo-fleet"
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "STAGE", stage)
    (root / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "station.py", root / "scripts")
    (stage / "scripts").mkdir(parents=True)
    (stage / "scripts" / "Start-Worker.ps1").write_text("bridge")
    pairing = root / "artifacts" / "pairing"
    (pairing / "second-pc").mkdir(parents=True)
    (pairing / "peer.json").write_text(
        json.dumps({"host": "second-pc", "port": 47941, "token": "t", "certificate_sha256": "c"})
    )
    (pairing / "second-pc" / "server.json").write_text("{}")
    (pairing / "second-pc" / "worker.pfx").write_bytes(b"pfx")
    (root / "artifacts" / "arenas").mkdir(parents=True)
    (root / "artifacts" / "arenas" / "saves-peer.json").write_text("{}")
    (root / "target" / "release").mkdir(parents=True)
    (root / module.WORKER_BUILD).write_bytes(b"MZ worker 1")
    mirrored = []
    monkeypatch.setattr(module, "mirror", mirrored.append)
    module.mirrored = mirrored
    return module


def test_the_fleet_pairing_is_the_lan_one_on_the_loopback(collect):
    path = collect.fleet_pairing()
    spec = json.loads(path.read_text())
    assert spec == {"host": "127.0.0.1", "port": 47941, "token": "t", "certificate_sha256": "c",
                    "reconnect_seconds": collect.RECONNECT_SECONDS}  # fmt: skip
    written = path.stat().st_mtime_ns
    time.sleep(0.02)
    collect.fleet_pairing()
    assert path.stat().st_mtime_ns == written, "unchanged, so the next push sends nothing"


def test_the_worker_ships_with_its_pairing_arenas_and_the_station_s_pairing(collect):
    collect.stage_worker()
    stage = collect.STAGE
    assert (stage / collect.WORKER_EXE).read_bytes() == b"MZ worker 1"
    for relative in [*collect.WORKER_PAIRING, collect.FLEET_PEER]:
        assert (stage / relative).exists(), relative
    station = load_script("station")
    assert collect.FLEET_PEER == station.FLEET_PEER, "where the station looks for it"
    assert sorted(collect.mirrored) == [f"artifacts/mods/{a}" for a in sorted(set(station.ARENAS))]


def test_deploying_the_worker_restarts_its_service_only_when_it_changed(collect, monkeypatch):
    calls = []
    monkeypatch.setattr(
        collect.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="same-commit\n"),
    )
    monkeypatch.setattr(collect, "fleet", lambda *a, **k: calls.append(a))
    restarts = lambda: [c for c in calls if c[:3] == ("service", "restart", "hoi4-worker")]  # noqa: E731
    collect.deploy(worker=True)
    assert ("push", "--on", collect.NODE, "--name", "hoi4-ai") in calls
    assert len(restarts()) == 1, "the first shipment"
    calls.clear()
    collect.deploy(worker=True)
    assert not restarts(), "nothing changed"
    (collect.ROOT / collect.WORKER_BUILD).write_bytes(b"MZ worker 2, rebuilt")
    collect.deploy(worker=True)
    assert len(restarts()) == 1, "a new build"
    calls.clear()
    (collect.ROOT / collect.WORKER_BUILD).write_bytes(b"MZ worker 3, rebuilt again")
    collect.deploy()
    assert not restarts(), "without --worker nothing of the worker ships"
    assert not [c for c in calls if c[:2] == ("service", "restart")], "and the code is the same"


# The bridge as a service, for real: a throwaway pairing, the worker, observers only.


def worker_exe():
    for candidate in (
        os.environ.get("HOI4_WORKER_EXE"),
        ROOT / "target" / "release" / "hoi4-desktop-worker.exe",
        ROOT / "artifacts" / "worker" / "hoi4-desktop-worker.exe",
    ):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


@pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("pwsh") is None or worker_exe() is None,
    reason="needs Windows, PowerShell 7 and a built worker",
)
def test_the_service_bridge_serves_the_loopback_and_restarts_only_when_let(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    for name in ("Start-Worker.ps1", "Game-Control.ps1", "Test-ArenaLoad.ps1"):
        shutil.copy(ROOT / "scripts" / name, project / "scripts")
    (project / "artifacts" / "worker").mkdir(parents=True)
    shutil.copy(worker_exe(), project / "artifacts" / "worker" / "hoi4-desktop-worker.exe")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(remote, "worker_executable", lambda: str(worker_exe()))
    remote.bundle(project / "artifacts" / "pairing", "127.0.0.1", "127.0.0.1", port)
    peer = project / "artifacts" / "pairing" / "peer.json"
    peer.write_text(json.dumps({**json.loads(peer.read_text()), "reconnect_seconds": 60}))
    restart = tmp_path / "restart-wanted"
    env = {**os.environ, "FLEET_RESTART_WANTED": str(restart)}
    bridge = subprocess.Popen(
        [shutil.which("pwsh"), "-NoProfile", "-File", "scripts/Start-Worker.ps1", "-Service",
         "-QuietSeconds", "4"],
        cwd=project, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    try:
        # It compiles its bridge first: the client waits for it (reconnect_seconds).
        with remote.RemoteDesktop(peer, attach=False, observer=True) as desk:
            status = desk.request("status")
            assert "protocol" in status
        runs = list((project / "artifacts" / "worker").glob("run-*/hoi4-desktop-worker.exe"))
        assert len(runs) == 1, "it runs a copy, so a push can replace the shipped worker"
        # A watcher (the live view) stays connected; a restart is wanted.
        watcher = remote.RemoteDesktop(peer, attach=False, observer=True)
        restart.write_text("asked\n")
        time.sleep(2)
        assert bridge.poll() is None, "a young observer holds the restart back"
        deadline = time.monotonic() + 30
        while bridge.poll() is None and time.monotonic() < deadline:
            time.sleep(0.25)
        assert bridge.poll() == 0, "an old one does not: the bridge exits"
        with pytest.raises(Exception):
            watcher.request("status", timeout=5)
        watcher._shutdown()
    finally:
        if bridge.poll() is None:
            bridge.kill()
        out = bridge.communicate(timeout=30)[0]
    assert f"ready at 127.0.0.1:{port}" in out and "nothing holds it back: exiting" in out, out
