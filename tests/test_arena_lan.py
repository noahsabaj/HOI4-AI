"""Loopback checks of the LAN coordinator protocol.

Everything here runs on 127.0.0.1 with threads and hand-written outcomes. It establishes protocol
behaviour only: nothing in this file is evidence about real HOI4, two physical PCs, or strength.
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from hoi4_agent.arena.contracts import ArenaError, BuildFingerprint, Country
from hoi4_agent.arena.evaluation import improvement_report
from hoi4_agent.arena.lan import (
    LAN_PROTOCOL_VERSION, MAX_LINE_BYTES, LanError, LanGuest, LanHost, LanOversized, LanRejected,
    LeagueMatchmaker, MatchSpec, Outcome, PausePolicy, SpeedPolicy, add_commands, cross_check,
    is_lan_address, pause_permission, side_swapped_schedule, verify_frozen_checkpoint,
)
from hoi4_agent.arena.league import League

FP = BuildFingerprint("a" * 64, "1.16.4", "b" * 64, "c" * 64)
SIDES = (("pc1", Country.BLUE), ("pc2", Country.RED))


def make_host(**options: Any) -> LanHost:
    options.setdefault("heartbeat_seconds", 0.2)
    options.setdefault("peer_timeout_seconds", 3.0)
    host = LanHost(FP, "pc1", Country.BLUE, **options)
    host.start_server()
    return host


def make_guest(host: LanHost, agent_id: str = "pc2", **options: Any) -> LanGuest:
    options.setdefault("heartbeat_seconds", 0.2)
    options.setdefault("peer_timeout_seconds", 3.0)
    guest = LanGuest(options.pop("fingerprint", FP), agent_id, options.pop("country", Country.RED), **options)
    guest.connect(*host.address, timeout=3.0)
    return guest


def spec(episode: str = "ep-1", **options: Any) -> MatchSpec:
    return MatchSpec(episode, "crossing_a", "train", 7, SIDES, **options)


def outcome(agent: str, winner: Country | None, episode: str = "ep-1", hour: int = 2160) -> Outcome:
    return Outcome(episode, agent, winner, "victory points at day 90", hour, 41, 2, {"decision_p50": 12.5})


def eventually(predicate: Callable[[], bool], seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def begin(host: LanHost, guest: LanGuest, match: MatchSpec) -> None:
    host.propose(match)
    assert guest.wait_proposal(3.0) == match
    guest.ready()
    host.wait_ready(3.0)
    host.start(game_hour=0)
    assert guest.wait_start(3.0) == 0


def raw_handshake(host: LanHost, agent_id: str = "raw") -> tuple[socket.socket, Any]:
    sock = socket.create_connection(host.address, timeout=3.0)
    stream = sock.makefile("rb")
    challenge = json.loads(stream.readline())
    assert challenge["type"] == "challenge" and challenge["auth"] == "none"
    hello = {"v": LAN_PROTOCOL_VERSION, "type": "hello", "agent_id": agent_id, "role": "guest",
             "kind": "agent",
             "nonce": "n" * 8, "country": "RED", "fingerprint": FP.__dict__}
    sock.sendall(json.dumps(hello).encode() + b"\n")
    return sock, stream


def test_handshake_success_and_audit_log(tmp_path: Path) -> None:
    log = tmp_path / "host.jsonl"
    host = make_host(log_path=log)
    guest = make_guest(host)
    try:
        assert guest.host_info == {"agent_id": "pc1", "country": "BLU", "role": "host", "kind": "agent"}
        assert not guest.resumed and guest.phase == "idle"
        assert host.peers()["pc2"]["connected"] and host.peers()["pc2"]["country"] == "RED"
        assert 0 < guest.ping() < 1.0
        assert "unauthenticated" in host.security_note  # no secret: says so, LAN peers only
        with pytest.raises(LanRejected) as taken:
            make_guest(host, "pc3", country=Country.BLUE)
        assert taken.value.code == "country_taken"
    finally:
        guest.close()
        host.close()
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    events = {row["event"] for row in rows}
    assert {"listening", "send", "recv", "peer_joined", "rejected", "closed"} <= events
    assert any(row["event"] == "recv" and row["message"]["type"] == "hello" for row in rows)


def test_bind_and_address_rules() -> None:
    with pytest.raises(LanError, match="refusing to bind every interface"):
        LanHost(FP, "pc1", Country.BLUE, bind="0.0.0.0")
    lan = ("127.0.0.1", "10.1.2.3", "172.16.0.9", "172.31.255.1", "192.168.1.10")
    assert all(is_lan_address(a) for a in lan)
    assert not any(is_lan_address(a) for a in ("8.8.8.8", "172.32.0.1", "169.254.1.1", "::1", "example.com"))


def test_fingerprint_mismatch_is_rejected_with_the_field() -> None:
    host = make_host()
    try:
        with pytest.raises(LanRejected) as rejected:
            make_guest(host, fingerprint=replace(FP, mod_sha256="d" * 64))
        assert rejected.value.code == "fingerprint_mismatch"
        assert "mod_sha256" in str(rejected.value) and "executable_sha256" not in str(rejected.value)
        assert "checksum" in str(rejected.value)
        assert "pc2" not in host.peers()
    finally:
        host.close()


def test_hmac_challenge() -> None:
    host = make_host(secret=b"right")
    try:
        with pytest.raises(LanRejected) as rejected:
            make_guest(host, secret=b"wrong")
        assert rejected.value.code == "auth_failed"
        with pytest.raises(LanError, match="authentication mismatch"):
            make_guest(host)
        assert host.peers() == {}
        guest = make_guest(host, secret=b"right")  # mutual: the guest also verified the host's MAC
        assert host.peers()["pc2"]["connected"]
        guest.close()
        assert all("right" not in json.dumps(row, default=str) for row in host.log.recent)
    finally:
        host.close()


def test_full_match_lifecycle() -> None:
    host = make_host()
    guest = make_guest(host)
    try:
        match = spec(speed=SpeedPolicy("free", max_speed=4, initial_speed=2), opponent_id="scripted-rush")
        assert not guest.may_pause()  # nothing is running yet
        begin(host, guest, match)
        assert host.effective_speed() == guest.effective_speed() == 2
        assert guest.may_pause() and host.may_pause()  # default policy: free

        faster = guest.request_speed(5, game_hour=30)
        assert faster.granted and faster.speed == 4 and "capped" in faster.reason
        assert faster.actor == "pc1" and faster.applies_from_game_hour == 30
        assert host.effective_speed() == 4
        host.set_game_hour(100)
        paused = guest.request_pause("waiting for a DeepSeek brief", max_wall_seconds=30, game_hour=90)
        assert paused.granted and paused.paused and paused.applies_from_game_hour == 100
        assert host.is_paused() and guest.is_paused() and not guest.may_pause()
        assert not host.request_pause("second pause").granted
        assert guest.request_resume(game_hour=100).granted and not host.is_paused()
        assert not guest.request_speed(9).granted

        guest.report_outcome(outcome("pc2", Country.BLUE))
        host.report_outcome(outcome("pc1", Country.BLUE, hour=2159))
        record = host.wait_outcomes(3.0)
        assert record.agreed and record.cross_checked and record.winner is Country.BLUE and not record.aborted
        assert record.result_for("pc1") == "win" and record.result_for("pc2") == "loss"
        assert [d.action for d in record.decisions] == ["init", "speed", "pause", "pause", "resume", "speed"]
        assert guest.wait_result(3.0)["winner"] == "BLU"

        acker = threading.Thread(target=lambda: (guest.wait_reset(3.0), guest.ack_reset()))
        acker.start()
        host.reset(3.0)
        acker.join(3.0)
        assert host.phase == "idle" and eventually(lambda: guest.phase == "idle" and guest.match is None)
        begin(host, guest, spec("ep-2"))  # the next episode starts cleanly
        assert len(host.log.events("control_decision")) == 5 and host.decisions[0].action == "init"
    finally:
        guest.close()
        host.close()


def test_outcome_disagreement_is_recorded_not_resolved() -> None:
    host = make_host()
    guest = make_guest(host)
    try:
        begin(host, guest, spec())
        guest.report_outcome(outcome("pc2", Country.RED))
        host.report_outcome(outcome("pc1", Country.BLUE))
        record = host.wait_outcomes(3.0)
        assert not record.agreed and record.winner is None and not record.cross_checked
        assert record.disagreements == ("winner: pc1=BLU, pc2=RED",)
        with pytest.raises(LanError, match="no agreed completed result"):
            record.result_for("pc1")
        assert host.log.events("match_record")[0]["record"]["agreed"] is False
    finally:
        guest.close()
        host.close()
    late = cross_check(spec(), [outcome("pc1", None)], ["pc1", "pc2"])
    assert not late.agreed and late.missing == ("pc2",)
    hours = cross_check(spec(), [outcome("pc1", None, hour=100), outcome("pc2", None, hour=900)],
                        ["pc1", "pc2"])
    assert not hours.agreed and "final_game_hour" in hours.disagreements[0]


def test_heartbeat_timeout_drops_a_silent_peer() -> None:
    host = make_host(heartbeat_seconds=0.05, peer_timeout_seconds=0.4)
    sock, stream = raw_handshake(host)
    try:
        assert json.loads(stream.readline())["type"] == "welcome"
        assert host.peers()["raw"]["connected"]
        assert eventually(lambda: not host.peers()["raw"]["connected"], 3.0)  # never answers a ping
        assert host.log.events("peer_timeout")[0]["agent_id"] == "raw"
    finally:
        sock.close()
        host.close()


def test_guest_notices_a_dead_host() -> None:
    host = make_host()
    guest = make_guest(host)
    host.close()
    assert eventually(lambda: not guest.connected)
    guest.close()


def test_crashed_guest_rejoins_the_same_match() -> None:
    host = make_host()
    first = make_guest(host)
    try:
        match = spec()
        begin(host, first, match)
        assert first.request_speed(3, game_hour=12).granted
        assert first._wire is not None
        first._wire.sock.close()  # abrupt loss, no bye: what a crashed process looks like to the host
        assert eventually(lambda: not host.peers()["pc2"]["connected"])
        assert host.phase == "running"  # the match stays open for the peer to return

        second = make_guest(host)  # a fresh process with the same agent id
        assert second.resumed and second.phase == "running" and second.wait_proposal(1.0) == match
        assert second.effective_speed() == 3
        second.report_outcome(outcome("pc2", None))
        host.report_outcome(outcome("pc1", None))
        record = host.wait_outcomes(3.0)
        assert record.agreed and record.result_for("pc2") == "draw"
        assert host.peers()["pc2"]["connections"] == 2 and host.log.events("peer_resumed")
        second.close()
    finally:
        first.close()
        host.close()


def test_pause_budget_is_enforced() -> None:
    host = make_host()
    guest = make_guest(host)
    try:
        begin(host, guest, spec(pause=PausePolicy("budgeted", 0.3)))
        assert guest.may_pause()
        granted = guest.request_pause("thinking", max_wall_seconds=60)
        assert granted.granted and granted.max_wall_seconds == pytest.approx(0.3)
        assert eventually(lambda: not guest.is_paused())  # the host ends the pause when the grant expires
        assert host.decisions[-1].action == "expire" and host.decisions[-1].budgets["pc2"] == 0.0
        assert host.decisions[-1].budgets["pc1"] == pytest.approx(0.3)  # budgets are per agent
        denied = guest.request_pause("again")
        assert not denied.granted and "exhausted" in denied.reason and not guest.may_pause()
        assert host.may_pause() and host.request_pause("host turn").granted
        logged = [row["decision"] for row in host.log.events("control_decision")]
        assert [(d["action"], d["granted"]) for d in logged] == [
            ("pause", True), ("expire", True), ("pause", False), ("pause", True)]
    finally:
        guest.close()
        host.close()
    assert pause_permission(PausePolicy("forbidden"), "agent", True, 99.0, None)[0] is False
    assert pause_permission(PausePolicy("forbidden"), "human", False, 0.0, None)[0] is True
    assert pause_permission(PausePolicy("host_only"), "agent", False, 0.0, None)[0] is False
    assert pause_permission(PausePolicy("host_only"), "agent", True, 0.0, None)[0] is True


def test_human_host_versus_model() -> None:
    host = make_host(kind="human")
    guest = make_guest(host)
    try:
        begin(host, guest, spec(pause=PausePolicy("forbidden")))
        assert not guest.request_pause("not allowed for the agent").granted
        speed = guest.request_speed(5)
        assert not speed.granted and "human" in speed.reason
        assert host.request_pause("coffee").granted  # the policy binds the agent only
        with pytest.raises(LanError):
            host.report_outcome(outcome("pc1", Country.RED))
        guest.report_outcome(outcome("pc2", Country.RED))
        record = host.wait_outcomes(3.0)
        assert record.agreed and not record.cross_checked and record.result_for("pc2") == "win"
    finally:
        guest.close()
        host.close()


def test_human_guest_thin_client_acknowledges_by_itself() -> None:
    host = make_host()
    human = make_guest(host, kind="human")
    try:
        host.propose(spec(pause=PausePolicy("host_only")))
        host.wait_ready(3.0)
        host.start()
        assert eventually(lambda: human.phase == "running")
        pause = human.request_pause("phone call")
        assert pause.granted and pause.actor == "pc1"
        with pytest.raises(LanError):
            human.report_outcome(outcome("pc2", None))
        host.report_outcome(outcome("pc1", Country.BLUE))
        assert host.wait_outcomes(3.0).agreed
        host.reset(3.0)
    finally:
        human.close()
        host.close()


def test_side_swapped_schedule_builds_valid_evaluation_blocks() -> None:
    schedule = list(side_swapped_schedule(["crossing_a", "defence_b"], "ai_panel_1", training_seed=3,
                                          pairs_per_scenario=2, split="held_out"))
    assert len(schedule) == 16 and len({game.pair_id for game in schedule}) == 4
    games = []
    for index, game in enumerate(schedule):
        match = game.match_spec(f"ep-{index}", "pc1", "pc2")
        assert match.country_of("pc1") is game.side and match.country_of("pc2") is game.side.opponent
        assert match.split == "held_out" and match.opponent_id == "ai_panel_1"
        winner = game.side if game.candidate == "candidate" else game.side.opponent
        reports = [outcome(agent, winner, f"ep-{index}") for agent in ("pc1", "pc2")]
        record = cross_check(match, reports, ["pc1", "pc2"])
        # Hand-written fixture outcomes: "hoi4_vision" only exercises the report's structure rules.
        games.append(game.evaluation_game(record, "pc1", source="hoi4_vision"))
    assert len({game.match_seed for game in schedule if game.pair_id == schedule[0].pair_id}) == 1
    report = improvement_report(games)
    assert report["per_seed"][0]["seed"] == 3 and report["per_seed"][0]["score_improvement"] == 1.0
    assert report["per_seed"][0]["games_per_policy"] == 8
    with pytest.raises(ArenaError):
        improvement_report(games[:-1])  # an incomplete block is still refused
    disputed = cross_check(schedule[0].match_spec("ep-x", "pc1", "pc2"),
                           [outcome("pc1", Country.BLUE, "ep-x"), outcome("pc2", Country.RED, "ep-x")],
                           ["pc1", "pc2"])
    with pytest.raises(LanError):
        schedule[0].evaluation_game(disputed, "pc1", source="hoi4_vision")


def test_league_matchmaker_distributes_frozen_opponents_by_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"not a real checkpoint; only its bytes matter here")
    league = League(seed=1)
    league.add_scripted("scripted-hold")
    frozen = league.freeze(checkpoint, tmp_path / "frozen", champion=True)
    maker = LeagueMatchmaker(league, ["pc1", "pc2", "pc3"])
    assert maker.idle_peer == "pc3"
    seen = set()
    for episode in range(12):
        (assignment,) = maker.assign(episode)
        match = maker.match_spec(assignment, f"ep-{episode}", "crossing_a", episode)
        assert match.country_of("pc1") is (Country.BLUE if episode % 2 == 0 else Country.RED)
        assert MatchSpec.from_dict(json.loads(json.dumps(match.to_dict()))) == match
        seen.add(assignment.opponent.id)
        if match.opponent_sha256:
            stored = tmp_path / "frozen" / f"{frozen[7:]}.pt"
            assert verify_frozen_checkpoint(match, tmp_path / "frozen") == stored
            with pytest.raises(LanError, match="missing here"):
                verify_frozen_checkpoint(match, tmp_path / "other-pc")
        else:
            assert verify_frozen_checkpoint(match, tmp_path / "frozen") is None
        reports = [outcome(agent, match.country_of("pc1"), match.episode_id) for agent in ("pc1", "pc2")]
        assert maker.record(assignment, cross_check(match, reports, ["pc1", "pc2"])) == "win"
    assert seen == {frozen, "scripted-hold"}
    assert sum(item.wins for item in league.opponents.values()) == 12
    (assignment,) = maker.assign(12)
    match = maker.match_spec(assignment, "ep-12", "crossing_a", 12)
    split = [outcome("pc1", Country.BLUE, "ep-12"), outcome("pc2", Country.RED, "ep-12")]
    assert maker.record(assignment, cross_check(match, split, ["pc1", "pc2"])) is None
    assert len(maker.unrecorded) == 1 and sum(item.wins for item in league.opponents.values()) == 12
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    (tampered / f"{frozen[7:]}.pt").write_bytes(b"different bytes")
    with pytest.raises(LanError, match="does not hash"):
        verify_frozen_checkpoint(replace(match, opponent_id=frozen, opponent_sha256=frozen[7:]), tampered)


def test_oversized_messages_are_rejected_both_ways() -> None:
    host = make_host()
    sock = socket.create_connection(host.address, timeout=3.0)
    stream = sock.makefile("rb")
    try:
        assert json.loads(stream.readline())["type"] == "challenge"
        sock.sendall(b"x" * (MAX_LINE_BYTES + 1024))  # no newline: the host must not buffer without bound
        error = json.loads(stream.readline())
        assert error["type"] == "error" and error["code"] == "oversized"
        assert stream.readline() == b""  # and the connection is closed
        assert host.log.events("oversized")
        guest = make_guest(host)
        host.propose(spec())
        guest.wait_proposal(3.0)
        with pytest.raises(LanOversized):
            guest.decline("x" * MAX_LINE_BYTES)
        assert guest.connected  # refusing to send does not tear the connection down
        guest.close()
    finally:
        sock.close()
        host.close()


def test_cli_registration_and_ping(capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.ArgumentParser()
    handlers = add_commands(parser.add_subparsers(dest="command", required=True))
    assert set(handlers) == {"lan-host", "lan-join", "lan-ping"}
    defaults = parser.parse_args(["lan-host", "--fingerprint", "fp.json"])
    assert defaults.bind == "127.0.0.1" and defaults.pause_policy == "free" and defaults.max_speed == 5
    host = make_host()
    try:
        args = parser.parse_args(["lan-ping", "--host", "127.0.0.1", "--port", str(host.address[1]),
                                  "--count", "3"])
        assert handlers["lan-ping"](args) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["count"] == 3 and 0 < printed["rtt_ms"]["median"] < 1000
        assert host.peers() == {}  # a probe never becomes a match participant
    finally:
        host.close()
