"""Contract checks, plus a real Windows pipe round trip (not a game test)."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from hoi4_agent.arena.contracts import (
    ArenaError, ArenaSpec, BuildFingerprint, CapabilityError, Country, Order,
    PlayerObservation, Province, UnitView, Verb,
)
from hoi4_agent.arena.diagnostics import REQUIRED_CAPABILITIES
from hoi4_agent.arena.protocol import NamedPipeTransport, decode_message, encode_message
from hoi4_agent.arena.session import BridgeSession


def fingerprint() -> BuildFingerprint:
    return BuildFingerprint("a" * 64, "test-version", "b" * 64, "c" * 64)


def observation(episode: str = "episode-1", sequence: int = 0) -> PlayerObservation:
    return PlayerObservation(episode, Country.BLUE, sequence, sequence, time.monotonic_ns(),
        (Province(1, 0, 0, "plains", (2,), Country.BLUE, 1),
         Province(2, 1, 0, "forest", (1,), None, 1)),
        (UnitView(1, Country.BLUE, 1, 1, 1, 1),))


class ScriptedTransport:
    """Protocol fixture, never accepted as real integration evidence."""
    def __init__(self, backend: str = "hoi4_native") -> None:
        self.backend = backend
        self.closed = False
        self.episode = 0
        self.requests: list[dict] = []

    def exchange(self, request: dict) -> dict:
        self.requests.append(request)
        method = request["method"]
        if method == "hello":
            from dataclasses import asdict
            result = {"backend": self.backend, "fingerprint": asdict(fingerprint()),
                      "capabilities": dict.fromkeys(REQUIRED_CAPABILITIES, True)}
        elif method == "reset":
            self.episode += 1
            result = {"episode_id": f"episode-{self.episode}", "engine_state_fingerprint": "d" * 64}
        elif method == "observe":
            result = observation(f"episode-{self.episode}").to_dict()
        elif method == "submit":
            result = {"order_id": request["payload"]["id"], "episode_id": f"episode-{self.episode}",
                      "accepted": True, "reason": "accepted", "applied_game_hour": 0}
        else:
            raise AssertionError(method)
        return {"version": 1, "id": request["id"], "ok": True, "result": result}

    def close(self) -> None:
        self.closed = True


def test_framing_and_observation_roundtrip() -> None:
    original = observation()
    frame = encode_message({"version": 1, "observation": original.to_dict()})
    assert struct.unpack("<I", frame[:4])[0] == len(frame) - 4
    assert PlayerObservation.from_dict(decode_message(frame[4:])["observation"]) == original


@pytest.mark.parametrize("body", [b'[]', b'{"version":2}', b'{"version":1,"x":NaN}', b'\xff'])
def test_malformed_messages_rejected(body: bytes) -> None:
    with pytest.raises(ArenaError):
        decode_message(body)


def test_diagnostic_host_cannot_be_a_live_session() -> None:
    transport = ScriptedTransport("diagnostic")
    with pytest.raises(CapabilityError, match="verified HOI4"):
        BridgeSession(transport, fingerprint())
    assert transport.closed


def test_incompatible_executable_rejected() -> None:
    with pytest.raises(ArenaError, match="executable_sha256"):
        BridgeSession(ScriptedTransport(), replace(fingerprint(), executable_sha256="e" * 64))


def test_country_staleness_duplicate_and_reset_boundaries() -> None:
    transport = ScriptedTransport()
    session = BridgeSession(transport, fingerprint())
    spec = ArenaSpec("fixture", fingerprint())
    session.reset(spec, (Country.BLUE,))
    session.observe(Country.BLUE)
    order = Order("order-1", "episode-1", 0, Country.BLUE, Verb.MOVE, (1,), 2)
    for invalid in [replace(order, unit_ids=(999,)), replace(order, country=Country.RED),
                    replace(order, observation_sequence=1)]:
        with pytest.raises(ArenaError):
            session.submit(invalid)
    assert session.submit(order).accepted
    with pytest.raises(ArenaError, match="duplicate"):
        session.submit(order)
    session.reset(spec, (Country.BLUE,))
    with pytest.raises(ArenaError, match="episode"):
        session.submit(order)


def test_expired_observation_preserves_orders_without_submitting() -> None:
    transport = ScriptedTransport()
    session = BridgeSession(transport, fingerprint())
    session.reset(ArenaSpec("fixture", fingerprint()), (Country.BLUE,))
    session.observe(Country.BLUE)
    session._received_ns[Country.BLUE] = 1
    with pytest.raises(ArenaError, match="deadline"):
        session.submit(Order("late", "episode-1", 0, Country.BLUE, Verb.MOVE, (1,), 2))
    assert all(request["method"] != "submit" for request in transport.requests)


@pytest.mark.skipif(os.name != "nt", reason="Windows named pipes")
def test_compiled_native_transport_rejects_gameplay() -> None:
    executable = Path("artifacts/native/Release/hoi4_bridge_probe.exe").resolve()
    if not executable.is_file():
        pytest.skip("build scripts/build_native.py to exercise the C++ pipe host")
    name = "hoi4-arena-test-" + uuid.uuid4().hex
    process = subprocess.Popen([str(executable), name], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    transport = None
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready: diagnostic transport only"
        transport = NamedPipeTransport(name)
        hello = transport.exchange({"version": 1, "id": 1, "method": "hello", "payload": {}})
        assert hello["result"]["backend"] == "diagnostic"
        assert not any(hello["result"]["capabilities"].values())
        rejected = transport.exchange({"version": 1, "id": 2, "method": "submit", "payload": {}})
        assert rejected["ok"] is False
        assert rejected["error"] == "engine_adapter_unavailable"
        assert json.dumps(hello)
    finally:
        if transport:
            transport.close()
        process.terminate()
        process.communicate(timeout=10)
