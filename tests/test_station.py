"""The second PC's station loop (scripts/station.py): lending the GPU to fleet between
sessions, stopping for a restart or a drain, telling a session that played nothing, and
the plan's one-off recording run."""

import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "station.py"


@pytest.fixture
def station(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("station", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    for name in ("FLEET_YIELD_WANTED", "FLEET_YIELD_LENT", "FLEET_RESTART_WANTED"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    return module


def test_nothing_is_lent_unless_fleet_wants_the_gpu(station, tmp_path):
    quits = []
    assert station.lend_if_wanted(quit_game=lambda: quits.append(1)) is False
    assert not quits and not (tmp_path / "fleet_yield_lent").exists()


def test_the_gpu_is_lent_with_hoi4_closed_until_fleet_is_done(station, tmp_path):
    wanted, lent = tmp_path / "fleet_yield_wanted", tmp_path / "fleet_yield_lent"
    wanted.write_text("gpu job")
    order = []

    def fleet():
        while not lent.exists():
            time.sleep(0.01)
        order.append("lent seen")
        wanted.unlink()

    helper = threading.Thread(target=fleet)
    helper.start()
    assert station.lend_if_wanted(poll=0.01, quit_game=lambda: order.append("quit")) is True
    helper.join()
    assert order == ["quit", "lent seen"], "HOI4 closes before the GPU is lent"
    assert not lent.exists(), "LENT goes once the GPU is back"


def test_it_stops_between_sessions_for_a_restart_or_a_drain(station, tmp_path):
    assert not station.should_stop()
    (tmp_path / "fleet_restart_wanted").write_text("")
    assert station.should_stop()
    (tmp_path / "fleet_restart_wanted").unlink()
    station.DRAIN.parent.mkdir(parents=True)
    station.DRAIN.write_text("")
    assert station.should_stop()


@pytest.mark.parametrize(
    "command, written, played",
    [
        ("drills", {"summary": {"drills": 60, "complete": 0}}, False),
        ("drills", {"summary": {"drills": 4, "complete": 3}}, True),
        ("practice", {"summary": {"episodes": 0}}, False),
        ("practice", {"summary": {"episodes": 14}}, True),
        ("play-policy", [], False),
        ("play-policy", [{"winner": "BLU"}], True),
        ("practice", None, False),
        ("record-ai", [{"game": "a", "error": "no launch"}], False),
        ("record-ai", [{"game": "a", "error": "x"}, {"game": "b", "winner": "timeout"}], True),
    ],
)
def test_a_session_counts_as_played_only_if_its_summary_says_so(
    station, tmp_path, command, written, played
):
    folder = tmp_path / "out"
    folder.mkdir()
    if written is not None:
        name = station.SUMMARIES[command].replace("*", "20260926-120000")
        (folder / name).write_text(json.dumps(written))
    assert station.played(command, str(folder)) is played


def test_every_session_is_listed_for_collection(station):
    ran = []
    ok = station.session("drills", "artifacts/drills/x", ["--episodes", "1"],
                         run=lambda *a: ran.append(a) or 0)  # fmt: skip
    assert ran == [("drills", "artifacts/drills/x", "--peer", station.PEER, "--episodes", "1")]
    assert ok is False, "no summary: nothing played"
    listed = [json.loads(line) for line in station.FINISHED.read_text().splitlines()]
    assert listed[0]["output"] == "artifacts/drills/x" and listed[0]["played"] is False


def test_the_plan_s_recording_run_plays_here_once_per_name(station):
    ran = []
    entry = {"name": "sv6", "minutes": 120, "args": ["--player", "scripted", "--speeds", "5"]}
    assert station.record(entry, run=lambda *a: ran.append(a) or 0) is True
    assert ran == [("record-ai", "artifacts/record-sv6", "--peer", station.PEER, "--peer-only",
                    "--minutes", "120", "--player", "scripted", "--speeds", "5")]  # fmt: skip
    assert station.record(entry, run=lambda *a: ran.append(a) or 0) is False, "once"
    assert len(ran) == 1
    listed = [json.loads(line) for line in station.FINISHED.read_text().splitlines()]
    assert listed[0]["output"] == "artifacts/record-sv6", "collected like any session"
