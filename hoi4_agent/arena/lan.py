"""LAN coordinator: syncs the AGENT processes around a HOI4 LAN multiplayer match.

HOI4 itself synchronises the game between the PCs. This module only synchronises the agent
processes next to it: who plays which country, when an episode starts and ends, whether both
sides saw the same result, and who may change speed or pause (in multiplayer the HOI4 host owns
speed and pause is shared, so the agents need a convention).

Wire format: one JSON object per line over TCP, every message carries ``v`` (protocol version),
``type`` and a per-connection ``seq``. Lines are bounded by ``MAX_LINE_BYTES``.

Honesty caveats:
- Nothing here touches the game. A granted pause/speed decision is a permission plus the name of
  the agent that should press the key (``actor``); whether the key press reached HOI4 is the
  executor's business and is not verified here.
- The optional HMAC authenticates the handshake only. Later messages are neither signed nor
  encrypted; the LAN is assumed to contain no active man-in-the-middle.
- Frozen opponent checkpoints are NOT transferred (messages are capped at 64 KiB). The coordinator
  distributes the opponent id and SHA-256; the remote PC must already hold ``<sha256>.pt`` (shared
  folder, manual copy) and ``verify_frozen_checkpoint`` proves it is the same file.
- One ``LanHost`` is one HOI4 multiplayer game, so at most two countries play per host. More PCs
  means more host/guest pairs; ``LeagueMatchmaker`` produces one assignment per pair.
- IPv4 only. Tests run on loopback; behaviour between two physical PCs is unverified until run.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import socket
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

from .contracts import ArenaError, BuildFingerprint, Country, integer
from .evaluation import EvaluationGame
from .fingerprint import file_hash
from .league import League, Opponent

LAN_PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 64 * 1024
DEFAULT_PORT = 47814
SECRET_ENV = "HOI4_ARENA_LAN_SECRET"
PAUSE_MODES = ("free", "host_only", "budgeted", "forbidden")
SPEED_MODES = ("free", "host_only")
KINDS = ("agent", "human")
SPLITS = ("train", "validation", "held_out")
_POLL_SECONDS = 0.1
_LAN_NETWORKS = tuple(ipaddress.ip_network(net) for net in
                      ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))


class LanError(ArenaError):
    """A coordinator transport, protocol or lifecycle failure."""


class LanClosed(LanError):
    """The connection ended (clean close, crash, or heartbeat timeout)."""


class LanTimeout(LanError):
    """A deadline passed."""


class LanOversized(LanError):
    """A line exceeded ``MAX_LINE_BYTES``."""


class LanRejected(LanError):
    """The handshake was refused; ``code`` is machine readable, the message is for the operator."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def secret_from_env() -> bytes | None:
    """Shared secret from ``HOI4_ARENA_LAN_SECRET``; the value is never logged or printed."""
    value = os.environ.get(SECRET_ENV, "")
    return value.encode("utf-8") if value else None


def is_lan_address(address: str) -> bool:
    """True for loopback and RFC1918 IPv4 addresses only."""
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(parsed in network for network in _LAN_NETWORKS)


def _mac(secret: bytes, *parts: str) -> str:
    return hmac.new(secret, "|".join(parts).encode("utf-8"), hashlib.sha256).hexdigest()


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------------------------
# Contracts carried on the wire
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PausePolicy:
    """``free`` (default, every grant still logged), ``host_only``, ``budgeted``, ``forbidden``.

    ``budget_wall_seconds`` is per agent and per episode and only used by ``budgeted``.
    The policy governs agents; a human player is never restricted by it.
    """
    mode: str = "free"
    budget_wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in PAUSE_MODES:
            raise LanError(f"pause policy must be one of {PAUSE_MODES}")
        budget = self.budget_wall_seconds
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) \
                or not 0 <= budget < float("inf"):
            raise LanError("pause budget must be a finite number of wall seconds >= 0")


@dataclass(frozen=True)
class SpeedPolicy:
    """``free`` or ``host_only``; ``max_speed`` caps every grant. All five speeds by default."""
    mode: str = "free"
    max_speed: int = 5
    initial_speed: int = 1

    def __post_init__(self) -> None:
        if self.mode not in SPEED_MODES:
            raise LanError(f"speed policy must be one of {SPEED_MODES}")
        for name in ("max_speed", "initial_speed"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 5:
                raise LanError(f"{name} must be 1-5")


@dataclass(frozen=True)
class PeerInfo:
    agent_id: str
    country: Country
    role: str  # host | guest
    kind: str = "agent"  # agent | human (no agent attached; reports no outcome)

    def to_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "country": self.country.value, "role": self.role,
                "kind": self.kind}


@dataclass(frozen=True)
class MatchSpec:
    episode_id: str
    scenario_id: str
    split: str
    seed: int
    sides: tuple[tuple[str, Country], ...]  # (agent id, country); authoritative over the handshake claim
    horizon_hours: int = 90 * 24
    opponent_id: str = ""
    opponent_sha256: str | None = None  # frozen checkpoint digest the opponent PC must load
    league_id: str = ""
    pause: PausePolicy = field(default_factory=PausePolicy)
    speed: SpeedPolicy = field(default_factory=SpeedPolicy)
    humans: tuple[str, ...] = ()  # side ids played by a human without a coordinator client

    def __post_init__(self) -> None:
        integer(self.seed, "match seed")
        integer(self.horizon_hours, "horizon_hours", 1)
        if not self.episode_id or not self.scenario_id or self.split not in SPLITS:
            raise LanError(f"match needs an episode id, a scenario id and a split in {SPLITS}")
        ids = [agent for agent, _ in self.sides]
        countries = [country for _, country in self.sides]
        if not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids) or len(set(countries)) != len(countries) \
                or any(not agent or not isinstance(country, Country) for agent, country in self.sides):
            raise LanError("sides must map one or two distinct agent ids to distinct countries")
        if not set(self.humans) <= set(ids):
            raise LanError("humans must be listed in sides")
        if self.opponent_sha256 is not None and (
                len(self.opponent_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.opponent_sha256)):
            raise LanError("opponent_sha256 must be a lowercase SHA-256 digest")

    def country_of(self, agent_id: str) -> Country:
        for agent, country in self.sides:
            if agent == agent_id:
                return country
        raise LanError(f"{agent_id} does not play in episode {self.episode_id}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sides"] = [[agent, country.value] for agent, country in self.sides]
        data["humans"] = list(self.humans)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchSpec:
        try:
            values = dict(data)
            values["sides"] = tuple((str(agent), Country(country)) for agent, country in values["sides"])
            values["humans"] = tuple(values.get("humans", ()))
            values["pause"] = PausePolicy(**values.get("pause", {}))
            values["speed"] = SpeedPolicy(**values.get("speed", {}))
            return cls(**values)
        except (KeyError, TypeError, ValueError) as exc:
            raise LanError(f"invalid match spec: {exc}") from exc


@dataclass(frozen=True)
class Outcome:
    """One side's report. ``winner`` None with status ``completed`` is a draw."""
    episode_id: str
    agent_id: str
    winner: Country | None
    reason: str
    final_game_hour: int
    orders: int = 0
    invalid_orders: int = 0
    latency_ms: dict[str, float] = field(default_factory=dict)
    status: str = "completed"  # completed | aborted (crash, desync: never counted as a draw)

    def __post_init__(self) -> None:
        for name in ("final_game_hour", "orders", "invalid_orders"):
            integer(getattr(self, name), name)
        if not self.episode_id or not self.agent_id or not self.reason \
                or self.status not in ("completed", "aborted"):
            raise LanError("outcome needs episode, agent, reason and a known status")
        if self.winner is not None and (not isinstance(self.winner, Country) or self.status == "aborted"):
            raise LanError("winner must be a Country, and an aborted game has no winner")
        if any(not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, (int, float))
               for k, v in self.latency_ms.items()):
            raise LanError("latency_ms maps names to numbers")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "winner": self.winner.value if self.winner else None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Outcome:
        try:
            values = dict(data)
            values["winner"] = Country(values["winner"]) if values.get("winner") else None
            return cls(**values)
        except (KeyError, TypeError, ValueError) as exc:
            raise LanError(f"invalid outcome: {exc}") from exc


@dataclass(frozen=True)
class ControlDecision:
    """The host's answer to a speed/pause request, broadcast to every peer and logged.

    ``actor`` is the agent that should press the key: the host agent when there is one (only the
    HOI4 host can change speed), otherwise the requesting agent for pause/resume.
    """
    request_id: str
    action: str  # init | speed | pause | resume | expire
    requester: str
    requester_kind: str
    granted: bool
    reason: str
    speed: int
    paused: bool
    paused_by: str | None
    applies_from_game_hour: int
    actor: str | None = None
    max_wall_seconds: float | None = None
    budgets: dict[str, float] = field(default_factory=dict)  # remaining pause seconds per agent (budgeted)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlDecision:
        try:
            return cls(**data)
        except TypeError as exc:
            raise LanError(f"invalid control decision: {exc}") from exc


@dataclass(frozen=True)
class MatchRecord:
    """Cross-checked result. ``winner`` is only filled when every report agrees."""
    spec: MatchSpec
    outcomes: tuple[Outcome, ...]
    agreed: bool
    cross_checked: bool  # False when only one side reported (human-vs-model): agreed but unverified
    winner: Country | None
    aborted: bool
    disagreements: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    decisions: tuple[ControlDecision, ...] = ()

    def result_for(self, agent_id: str) -> str:
        if not self.agreed or self.aborted:
            raise LanError(f"episode {self.spec.episode_id} has no agreed completed result: "
                           + ("; ".join(self.disagreements) or "aborted"))
        if self.winner is None:
            return "draw"
        return "win" if self.spec.country_of(agent_id) is self.winner else "loss"

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec.to_dict(), "outcomes": [o.to_dict() for o in self.outcomes],
                "agreed": self.agreed, "cross_checked": self.cross_checked,
                "winner": self.winner.value if self.winner else None, "aborted": self.aborted,
                "disagreements": list(self.disagreements), "missing": list(self.missing),
                "decisions": [d.to_dict() for d in self.decisions]}


def cross_check(spec: MatchSpec, outcomes: Sequence[Outcome], expected: Sequence[str],
                decisions: Sequence[ControlDecision] = (), hour_tolerance: int = 24) -> MatchRecord:
    """Compare the sides' reports. Any disagreement is recorded; nothing is resolved by guessing."""
    by_agent = {outcome.agent_id: outcome for outcome in outcomes}
    missing = tuple(agent for agent in expected if agent not in by_agent)
    problems = [f"no outcome from {agent} before the deadline" for agent in missing]
    reports = [by_agent[agent] for agent in expected if agent in by_agent]
    if len({report.status for report in reports}) > 1:
        problems.append("status: " + ", ".join(f"{r.agent_id}={r.status}" for r in reports))
    if len({report.winner for report in reports}) > 1:
        problems.append("winner: " + ", ".join(
            f"{r.agent_id}={r.winner.value if r.winner else 'draw'}" for r in reports))
    hours = [report.final_game_hour for report in reports]
    if hours and max(hours) - min(hours) > hour_tolerance:
        problems.append(f"final_game_hour differs by more than {hour_tolerance}: "
                        + ", ".join(f"{r.agent_id}={r.final_game_hour}" for r in reports))
    agreed = bool(reports) and not problems
    return MatchRecord(spec, tuple(reports), agreed, agreed and len(reports) > 1,
                       reports[0].winner if agreed else None,
                       agreed and reports[0].status == "aborted", tuple(problems), missing, tuple(decisions))


def pause_permission(policy: PausePolicy, kind: str, is_host: bool, remaining: float,
                     paused_by: str | None) -> tuple[bool, str]:
    """Shared by host (deciding) and guests (``may_pause`` prediction)."""
    if paused_by is not None:
        return False, f"already paused by {paused_by}"
    if kind == "human":
        return True, "human players are not governed by the agent pause policy"
    if policy.mode == "forbidden":
        return False, "pause policy is forbidden (pause-forbidden evaluation track)"
    if policy.mode == "host_only" and not is_host:
        return False, "pause policy is host_only"
    if policy.mode == "budgeted" and remaining <= 0:
        return False, "pause budget for this episode is exhausted"
    return True, f"pause policy {policy.mode}"


# ---------------------------------------------------------------------------------------------
# Transport and audit log
# ---------------------------------------------------------------------------------------------

class EventLog:
    """Append-only JSONL audit log of every protocol event. MACs and secrets are never written."""

    def __init__(self, path: Path | None = None, keep: int = 4096) -> None:
        self.path = path
        self.recent: deque[dict[str, Any]] = deque(maxlen=keep)
        self._lock = threading.Lock()
        self._stream: IO[str] | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("a", encoding="utf-8")

    def write(self, event: str, **fields: Any) -> None:
        message = fields.get("message")
        if isinstance(message, dict) and "mac" in message:
            fields["message"] = {**message, "mac": "<redacted>"}
        row = {"wall_time": time.time(), "monotonic_ns": time.monotonic_ns(), "event": event, **fields}
        with self._lock:
            self.recent.append(row)
            if self._stream is not None:
                self._stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                self._stream.flush()

    def events(self, event: str) -> list[dict[str, Any]]:
        with self._lock:
            return [row for row in self.recent if row["event"] == event]

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None


class _Wire:
    """Line-delimited JSON over one socket. Sends are serialised; receives belong to one thread."""

    def __init__(self, sock: socket.socket, log: EventLog, label: str) -> None:
        sock.settimeout(_POLL_SECONDS)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self.sock, self.log, self.label = sock, log, label
        self._buffer = bytearray()
        self._send_lock = threading.Lock()
        self._seq = 0
        self.closed = threading.Event()

    def send(self, message: dict[str, Any]) -> None:
        with self._send_lock:
            self._seq += 1
            message = {"v": LAN_PROTOCOL_VERSION, "seq": self._seq, **message}
            data = _canonical(message).encode("utf-8") + b"\n"
            if len(data) > MAX_LINE_BYTES:
                raise LanOversized(f"refusing to send a {len(data)} byte message (limit {MAX_LINE_BYTES})")
            if self.closed.is_set():
                raise LanClosed(f"connection to {self.label} is closed")
            try:
                self.sock.sendall(data)
            except OSError as exc:
                self.close()
                raise LanClosed(f"send to {self.label} failed: {exc}") from exc
        self.log.write("send", peer=self.label, message=message)

    def recv(self, deadline: float | None = None, stop: threading.Event | None = None) -> dict[str, Any]:
        while True:
            newline = self._buffer.find(b"\n")
            if newline > MAX_LINE_BYTES or (newline < 0 and len(self._buffer) > MAX_LINE_BYTES):
                raise LanOversized(f"message from {self.label} exceeds {MAX_LINE_BYTES} bytes")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[:newline + 1]
                try:
                    message = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as exc:
                    raise LanError(f"invalid JSON line from {self.label}") from exc
                if not isinstance(message, dict) or not isinstance(message.get("type"), str):
                    raise LanError(f"message from {self.label} is not an object with a type")
                self.log.write("recv", peer=self.label, message=message)
                return message
            if self.closed.is_set() or (stop is not None and stop.is_set()):
                raise LanClosed(f"connection to {self.label} is closed")
            if deadline is not None and time.monotonic() > deadline:
                raise LanTimeout(f"deadline passed waiting for {self.label}")
            try:
                chunk = self.sock.recv(65536)
            except TimeoutError:
                continue
            except OSError as exc:
                raise LanClosed(f"connection to {self.label} failed: {exc}") from exc
            if not chunk:
                raise LanClosed(f"{self.label} closed the connection")
            self._buffer += chunk

    def close(self) -> None:
        self.closed.set()
        try:
            self.sock.close()
        except OSError:
            pass


@dataclass
class _Peer:
    info: PeerInfo
    address: str
    wire: _Wire | None = None
    last_seen: float = 0.0
    rtt_ms: float | None = None
    connections: int = 0

    @property
    def connected(self) -> bool:
        return self.wire is not None


class _Base:
    """State shared by host and guest: identity, log, condition variable, control view."""

    def __init__(self, fingerprint: BuildFingerprint | None, agent_id: str, country: Country | None,
                 role: str,
                 kind: str, secret: bytes | None, log_path: Path | None, heartbeat_seconds: float,
                 peer_timeout_seconds: float) -> None:
        if kind not in KINDS or not agent_id or len(agent_id) > 64:
            raise LanError(f"agent id (1-64 characters) and a kind in {KINDS} are required")
        if heartbeat_seconds <= 0 or peer_timeout_seconds <= heartbeat_seconds:
            raise LanError("peer timeout must exceed the heartbeat interval")
        self.fingerprint, self.agent_id, self.country = fingerprint, agent_id, country
        self.role, self.kind = role, kind
        self._secret = secret
        self.log = EventLog(log_path)
        self.heartbeat_seconds, self.peer_timeout_seconds = heartbeat_seconds, peer_timeout_seconds
        self._cond = threading.Condition(threading.RLock())
        self._stop = threading.Event()
        self.phase = "idle"  # idle | proposed | running | finished
        self.match: MatchSpec | None = None
        self.control: ControlDecision | None = None

    def _wait(self, predicate: Callable[[], bool], timeout: float, what: str) -> None:
        with self._cond:
            if not self._cond.wait_for(predicate, timeout):
                raise LanTimeout(f"deadline of {timeout:g}s passed waiting for {what}")

    # --- what the local agent asks -----------------------------------------------------------
    def effective_speed(self) -> int:
        with self._cond:
            if self.control is not None:
                return self.control.speed
            policy = self.match.speed if self.match else SpeedPolicy()
            return min(policy.initial_speed, policy.max_speed)

    def is_paused(self) -> bool:
        with self._cond:
            return bool(self.control and self.control.paused)

    def may_pause(self) -> bool:
        """Would a pause request be granted right now? Asking is free; the host still decides."""
        with self._cond:
            if self.phase != "running" or self.match is None:
                return False
            policy = self.match.pause
            remaining = (self.control.budgets.get(self.agent_id, policy.budget_wall_seconds)
                         if self.control else policy.budget_wall_seconds)
            return pause_permission(policy, self.kind, self.role == "host", remaining,
                                    self.control.paused_by if self.control else None)[0]


# ---------------------------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------------------------

class LanHost(_Base):
    """Listens for guests; owns the match lifecycle and the speed/pause decisions.

    Run it on the PC that hosts the HOI4 multiplayer game. ``secret=None`` means no HMAC: only
    loopback/RFC1918 peers are then accepted. The CLI passes ``secret_from_env()``.
    """

    def __init__(self, fingerprint: BuildFingerprint, agent_id: str, country: Country, *, kind: str = "agent",
                 bind: str = "127.0.0.1", port: int = 0, secret: bytes | None = None,
                 log_path: Path | None = None, heartbeat_seconds: float = 2.0,
                 peer_timeout_seconds: float = 6.0, handshake_timeout_seconds: float = 5.0,
                 allow_wildcard_bind: bool = False, hour_tolerance: int = 24) -> None:
        super().__init__(fingerprint, agent_id, country, "host", kind, secret, log_path, heartbeat_seconds,
                         peer_timeout_seconds)
        if not allow_wildcard_bind and bind in ("", "0.0.0.0", "::"):
            raise LanError("refusing to bind every interface; pass the PC's LAN address "
                           "(for example <lan-address>) or 127.0.0.1")
        self.info = PeerInfo(agent_id, country, "host", kind)
        self._bind, self._port = bind, port
        self._handshake_timeout, self._hour_tolerance = handshake_timeout_seconds, hour_tolerance
        self.security_note = ("HMAC-SHA256 challenge required" if secret else
                              f"{SECRET_ENV} is unset: accepting only loopback/RFC1918 peers, "
                              "unauthenticated")
        self._server: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._peers: dict[str, _Peer] = {}
        self._ready: set[str] = set()
        self._declined: dict[str, str] = {}
        self._reset_acks: set[str] = set()
        self._required: tuple[str, ...] = ()
        self._outcomes: dict[str, Outcome] = {}
        self._game_hour = 0
        self._pause_started: float | None = None
        self._pause_deadline: float | None = None
        self._pause_used: dict[str, float] = {}
        self.decisions: list[ControlDecision] = []  # every grant AND denial of the current episode
        self.records: list[MatchRecord] = []

    # --- lifecycle of the server -------------------------------------------------------------
    def start_server(self) -> tuple[str, int]:
        self._server = socket.create_server((self._bind, self._port))
        self._server.settimeout(_POLL_SECONDS)
        self.log.write("listening", address=list(self.address), security=self.security_note,
                       host=self.info.to_dict())
        for target in (self._accept_loop, self._tick_loop):
            thread = threading.Thread(target=target, daemon=True, name=f"lan-host-{target.__name__}")
            thread.start()
            self._threads.append(thread)
        return self.address

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            raise LanError("host is not listening")
        return self._server.getsockname()[:2]

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            wires = [peer.wire for peer in self._peers.values() if peer.wire is not None]
        for wire in wires:
            self._try_send(wire, {"type": "bye", "reason": "host closing"})
            wire.close()
        if self._server is not None:
            self._server.close()
        for thread in self._threads:
            thread.join(2.0)
        self.log.write("closed")
        self.log.close()

    def __enter__(self) -> LanHost:
        self.start_server()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def peers(self) -> dict[str, dict[str, Any]]:
        with self._cond:
            return {agent: {**peer.info.to_dict(), "address": peer.address, "connected": peer.connected,
                            "rtt_ms": peer.rtt_ms, "connections": peer.connections}
                    for agent, peer in self._peers.items()}

    def wait_for_guests(self, count: int, timeout: float) -> None:
        self._wait(lambda: sum(peer.connected for peer in self._peers.values()) >= count, timeout,
                   f"{count} connected guest(s)")

    def set_game_hour(self, game_hour: int) -> None:
        """The host agent reports the hour it sees so decisions can say when they apply from."""
        integer(game_hour, "game hour")
        with self._cond:
            self._game_hour = max(self._game_hour, game_hour)

    # --- threads -----------------------------------------------------------------------------
    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                sock, address = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            label = f"{address[0]}:{address[1]}"
            thread = threading.Thread(target=self._serve, args=(sock, address[0], label),
                                      daemon=True, name="lan-host-peer")
            thread.start()

    def _serve(self, sock: socket.socket, ip: str, label: str) -> None:
        wire = _Wire(sock, self.log, label)
        peer: _Peer | None = None
        try:
            peer = self._handshake(wire, ip, label)
            while True:
                message = wire.recv(None, self._stop)
                if peer is not None:
                    with self._cond:
                        peer.last_seen = time.monotonic()
                if not self._dispatch(wire, peer, message):
                    return
        except LanOversized as exc:
            self._try_send(wire, {"type": "error", "code": "oversized", "message": str(exc)})
            self.log.write("oversized", peer=label, detail=str(exc))
        except LanRejected as exc:
            self.log.write("rejected", peer=label, code=exc.code, detail=str(exc))
        except LanClosed as exc:
            self.log.write("connection_lost", peer=label, detail=str(exc))
        except LanError as exc:
            self._try_send(wire, {"type": "error", "code": "bad_message", "message": str(exc)})
            self.log.write("protocol_error", peer=label, detail=str(exc))
        finally:
            if peer is not None:
                self._drop(peer, wire, "connection ended")
            wire.close()

    @staticmethod
    def _try_send(wire: _Wire, message: dict[str, Any]) -> None:
        try:
            wire.send(message)
        except LanError:
            pass

    def _drop(self, peer: _Peer, wire: _Wire, reason: str) -> None:
        with self._cond:
            if peer.wire is wire:
                peer.wire = None
                self.log.write("peer_disconnected", agent_id=peer.info.agent_id, reason=reason,
                               phase=self.phase)
                self._cond.notify_all()
        wire.close()

    def _tick_loop(self) -> None:
        next_ping = 0.0
        while not self._stop.wait(min(self.heartbeat_seconds / 2, _POLL_SECONDS)):
            now = time.monotonic()
            with self._cond:
                for peer in list(self._peers.values()):
                    if peer.wire is not None and now - peer.last_seen > self.peer_timeout_seconds:
                        self.log.write("peer_timeout", agent_id=peer.info.agent_id,
                                       silent_seconds=round(now - peer.last_seen, 3))
                        self._drop(peer, peer.wire, "heartbeat timeout")
                if now >= next_ping:
                    next_ping = now + self.heartbeat_seconds
                    self._broadcast({"type": "ping", "t": time.perf_counter_ns()})
                expired = self._pause_deadline is not None and now >= self._pause_deadline
                if expired and self.control is not None:
                    pauser = self.control.paused_by or self.agent_id
                    self._publish(self._end_pause("expire", pauser, self._kind_of(pauser),
                                                  secrets.token_hex(8),
                                                  "pause grant expired", self._game_hour))

    # --- handshake ---------------------------------------------------------------------------
    def _reject(self, wire: _Wire, code: str, message: str) -> LanRejected:
        self._try_send(wire, {"type": "reject", "code": code, "message": message})
        return LanRejected(code, message)

    def _handshake(self, wire: _Wire, ip: str, label: str) -> _Peer | None:
        deadline = time.monotonic() + self._handshake_timeout
        nonce = secrets.token_hex(16)
        wire.send({"type": "challenge", "nonce": nonce, "auth": "hmac-sha256" if self._secret else "none",
                   "note": self.security_note})
        if self._secret is None and not is_lan_address(ip):
            raise self._reject(wire, "address_not_allowed",
                               f"{ip} is not loopback/RFC1918 and no shared secret ({SECRET_ENV}) is "
                               "configured")
        hello = wire.recv(deadline, self._stop)
        if hello["type"] != "hello":
            raise self._reject(wire, "bad_handshake", f"expected hello, got {hello['type']}")
        if hello.get("v") != LAN_PROTOCOL_VERSION:
            raise self._reject(wire, "version_mismatch", f"host speaks LAN protocol {LAN_PROTOCOL_VERSION}, "
                               f"peer sent {hello.get('v')!r}; update both PCs to the same commit")
        fields = [hello.get(key) for key in ("agent_id", "role", "kind", "nonce")]
        claimed = hello.get("fingerprint")
        if not all(isinstance(item, str) and 0 < len(item) <= 64 for item in fields) \
                or fields[2] not in KINDS or fields[1] not in ("guest", "probe"):
            raise self._reject(wire, "bad_handshake",
                               "hello needs agent_id, role guest|probe, kind and nonce")
        agent_id, role, kind, guest_nonce = (str(item) for item in fields)
        if self._secret is not None:
            expected = _mac(self._secret, "hello", nonce, guest_nonce, agent_id, role, kind,
                            str(hello.get("country")), _canonical(claimed))
            if not isinstance(hello.get("mac"), str) or not hmac.compare_digest(expected, hello["mac"]):
                raise self._reject(wire, "auth_failed", "HMAC check failed: "
                                   f"{SECRET_ENV} differs between the PCs "
                                   "or is unset on the joining PC")
        welcome: dict[str, Any] = {"type": "welcome", "host": self.info.to_dict(),
                                   "heartbeat_seconds": self.heartbeat_seconds,
                                   "peer_timeout_seconds": self.peer_timeout_seconds}
        if self._secret is not None:
            welcome["mac"] = _mac(self._secret, "welcome", guest_nonce, nonce, self.agent_id)
        if role == "probe":  # latency probe: authenticated, but not a match participant
            wire.send({**welcome, "resumed": False, "phase": self.phase, "match": None, "control": None})
            return None
        try:
            country = Country(str(hello.get("country")))
            fingerprint = BuildFingerprint(**dict(claimed or {}))
        except (ValueError, TypeError, ArenaError) as exc:
            raise self._reject(wire, "bad_handshake", f"invalid country or fingerprint: {exc}") from exc
        assert self.fingerprint is not None
        if fingerprint != self.fingerprint:
            mine, theirs = asdict(self.fingerprint), asdict(fingerprint)
            detail = "; ".join(f"{key}: host {mine[key][:16]} guest {theirs[key][:16]}"
                               for key in mine if mine[key] != theirs[key])
            raise self._reject(wire, "fingerprint_mismatch", f"build fingerprints differ in {detail}. HOI4 "
                               "multiplayer also refuses mismatched checksums; make executable, DLC and the "
                               "arena mod byte-identical on both PCs")
        info = PeerInfo(agent_id, country, "guest", kind)
        with self._cond:
            existing = self._peers.get(agent_id)
            if agent_id == self.agent_id or (existing is not None and existing.connected):
                raise self._reject(wire, "agent_id_in_use", f"agent id {agent_id} is already connected")
            taken = [self.info] + [p.info for p in self._peers.values() if p.info.agent_id != agent_id]
            if any(other.country is country for other in taken):
                raise self._reject(wire, "country_taken", f"{country.value} is already claimed")
            peer = existing or _Peer(info, label)
            peer.info, peer.address, peer.last_seen = info, label, time.monotonic()
            peer.connections += 1
            resumed = existing is not None
            wire.send({**welcome, "resumed": resumed, "phase": self.phase,
                       "match": self.match.to_dict() if self.match else None,
                       "control": self.control.to_dict() if self.control else None})
            peer.wire = wire
            self._peers[agent_id] = peer
            self.log.write("peer_resumed" if resumed else "peer_joined", agent_id=agent_id, address=label,
                           phase=self.phase, episode_id=self.match.episode_id if self.match else None)
            self._cond.notify_all()
        return peer

    # --- message handling --------------------------------------------------------------------
    def _send(self, peer: _Peer, message: dict[str, Any]) -> None:
        wire = peer.wire
        if wire is None:
            self.log.write("undelivered", agent_id=peer.info.agent_id, message=message)
            return
        try:
            wire.send(message)
        except LanError as exc:
            self._drop(peer, wire, f"send failed: {exc}")

    def _broadcast(self, message: dict[str, Any]) -> None:
        with self._cond:  # held so that decisions reach every peer in the order they were taken
            for peer in list(self._peers.values()):
                if message["type"] != "ping" or peer.connected:
                    self._send(peer, message)

    def _dispatch(self, wire: _Wire, peer: _Peer | None, message: dict[str, Any]) -> bool:
        kind = message["type"]
        if kind == "ping":
            wire.send({"type": "pong", "id": message.get("id"), "t": message.get("t")})
        elif kind == "pong":
            if peer is not None and isinstance(message.get("t"), int):
                peer.rtt_ms = (time.perf_counter_ns() - message["t"]) / 1e6
        elif kind == "bye":
            return False
        elif peer is None:
            wire.send({"type": "error", "code": "probe_only", "message": "probe connections may only ping"})
        elif kind in ("ready", "decline", "reset_ack"):
            with self._cond:
                if self.match is None or message.get("episode_id") != self.match.episode_id:
                    wire.send({"type": "error", "code": "wrong_episode",
                               "message": f"{kind} for a stale episode"})
                elif kind == "ready":
                    self._ready.add(peer.info.agent_id)
                elif kind == "decline":
                    self._declined[peer.info.agent_id] = str(message.get("reason", ""))
                else:
                    self._reset_acks.add(peer.info.agent_id)
                self._cond.notify_all()
        elif kind == "outcome":
            self._accept_outcome(peer.info.agent_id, Outcome.from_dict(message.get("outcome", {})), wire)
        elif kind in ("request_speed", "request_pause", "request_resume"):
            self._handle_request(peer.info, kind.removeprefix("request_"), message)
        else:
            wire.send({"type": "error", "code": "unknown_type", "message": f"unknown message type {kind}"})
        return True

    def _accept_outcome(self, sender: str, outcome: Outcome, wire: _Wire | None) -> None:
        problem = None
        with self._cond:
            if self.match is None or outcome.episode_id != self.match.episode_id or self.phase != "running":
                problem = f"no running episode {outcome.episode_id}"
            elif outcome.agent_id != sender or sender not in self._required:
                problem = f"{sender} may not report an outcome for {outcome.agent_id}"
            elif sender in self._outcomes and self._outcomes[sender] != outcome:
                problem = "a different outcome was already reported; the first one is kept"
            else:
                self._outcomes[sender] = outcome
                self.log.write("outcome", agent_id=sender, outcome=outcome.to_dict())
                self._cond.notify_all()
        if problem is not None:
            self.log.write("outcome_refused", agent_id=sender, detail=problem, outcome=outcome.to_dict())
            if wire is not None:
                wire.send({"type": "error", "code": "outcome_refused", "message": problem})
            else:
                raise LanError(problem)

    # --- match lifecycle ---------------------------------------------------------------------
    def _kind_of(self, agent_id: str) -> str:
        if agent_id == self.agent_id:
            return self.kind
        peer = self._peers.get(agent_id)
        return peer.info.kind if peer else "human"

    def propose(self, spec: MatchSpec) -> None:
        with self._cond:
            if self.phase != "idle":
                raise LanError(f"cannot propose while phase is {self.phase}; reset first")
            for agent, _ in spec.sides:
                peer = self._peers.get(agent)
                absent = peer is None or not peer.connected
                if agent != self.agent_id and agent not in spec.humans and absent:
                    raise LanError(f"{agent} is not connected and is not declared in spec.humans")
                if agent in spec.humans and self._kind_of(agent) != "human":
                    raise LanError(f"{agent} is declared human but connected as an agent")
            self.match, self.phase, self.control = spec, "proposed", None
            self._required = tuple(agent for agent, _ in spec.sides if self._kind_of(agent) == "agent")
            self._ready, self._declined, self._reset_acks, self._outcomes = set(), {}, set(), {}
            self._pause_used, self._pause_started, self._pause_deadline = {}, None, None
            self.decisions, self._game_hour = [], 0
            self.log.write("propose", spec=spec.to_dict(), outcome_reporters=list(self._required))
            self._broadcast({"type": "propose", "spec": spec.to_dict()})

    def _guests_in_match(self) -> list[str]:
        assert self.match is not None
        return [agent for agent, _ in self.match.sides if agent in self._peers]

    def wait_ready(self, timeout: float) -> None:
        try:
            self._wait(lambda: bool(self._declined) or set(self._guests_in_match()) <= self._ready,
                       timeout, "ready")
        except LanTimeout as exc:
            laggards = sorted(set(self._guests_in_match()) - self._ready)
            raise LanTimeout(f"{exc}; not ready: {laggards}") from exc
        if self._declined:
            raise LanError(f"match declined: {self._declined}")

    def start(self, game_hour: int = 0) -> ControlDecision:
        integer(game_hour, "game hour")
        with self._cond:
            if self.phase != "proposed" or self.match is None \
                    or not set(self._guests_in_match()) <= self._ready:
                raise LanError("start requires a proposed match that every connected participant "
                               "acknowledged")
            policy = self.match.speed
            self.phase, self._game_hour = "running", game_hour
            self.control = ControlDecision(secrets.token_hex(8), "init", self.agent_id, self.kind, True,
                                           "episode start", min(policy.initial_speed, policy.max_speed),
                                           False,
                                           None, game_hour, self.agent_id if self.kind == "agent" else None,
                                           None, self._budgets())
            self.decisions.append(self.control)
            self.log.write("start", episode_id=self.match.episode_id, control=self.control.to_dict())
            self._broadcast({"type": "start", "episode_id": self.match.episode_id, "game_hour": game_hour,
                             "control": self.control.to_dict()})
            self._cond.notify_all()
            return self.control

    def report_outcome(self, outcome: Outcome) -> None:
        """The host agent's own report. A human host reports nothing."""
        self._accept_outcome(self.agent_id, outcome, None)

    def wait_outcomes(self, timeout: float) -> MatchRecord:
        """Cross-check all reports. A missing report is recorded as such after the deadline."""
        with self._cond:
            if self.phase != "running" or self.match is None:
                raise LanError("no running match")
            self._cond.wait_for(lambda: set(self._required) <= self._outcomes.keys(), timeout)
            record = cross_check(self.match, list(self._outcomes.values()), self._required, self.decisions,
                                 self._hour_tolerance)
            self.phase = "finished"
            self.records.append(record)
            self.log.write("match_record", record=record.to_dict())
            self._broadcast({"type": "result", "episode_id": self.match.episode_id, "agreed": record.agreed,
                             "winner": record.winner.value if record.winner else None,
                             "aborted": record.aborted, "disagreements": list(record.disagreements)})
            return record

    def reset(self, timeout: float) -> None:
        """Tell every side to run the arena reset; returns once the connected guests acknowledged."""
        with self._cond:
            if self.match is None:
                raise LanError("nothing to reset")
            if self._pause_started is not None and self.control is not None:
                pauser = self.control.paused_by or self.agent_id
                self._publish(self._end_pause("expire", pauser, self._kind_of(pauser),
                                              secrets.token_hex(8),
                                              "episode reset", self._game_hour))
            episode = self.match.episode_id
            waiting = {agent for agent in self._guests_in_match() if self._peers[agent].connected}
            self.log.write("reset", episode_id=episode, waiting_for=sorted(waiting))
            self._broadcast({"type": "reset", "episode_id": episode})
        self._wait(lambda: {a for a in waiting if self._peers[a].connected} <= self._reset_acks, timeout,
                   "reset acknowledgements")
        with self._cond:
            self.phase, self.match, self.control = "idle", None, None

    # --- speed / pause -----------------------------------------------------------------------
    def _budgets(self) -> dict[str, float]:
        if self.match is None or self.match.pause.mode != "budgeted":
            return {}
        budget = self.match.pause.budget_wall_seconds
        return {agent: round(max(0.0, budget - self._pause_used.get(agent, 0.0)), 3)
                for agent in self._required}

    def _end_pause(self, action: str, requester: str, requester_kind: str, request_id: str, reason: str,
                   game_hour: int) -> ControlDecision:
        assert self.control is not None
        if self._pause_started is not None and self.control.paused_by is not None:
            pauser = self.control.paused_by
            elapsed = time.monotonic() - self._pause_started
            self._pause_used[pauser] = self._pause_used.get(pauser, 0.0) + elapsed
        self._pause_started = self._pause_deadline = None
        return self._decision(request_id, action, requester, requester_kind, True, reason, self.control.speed,
                              False, None, game_hour, None)

    def _decision(self, request_id: str, action: str, requester: str, requester_kind: str, granted: bool,
                  reason: str, speed: int, paused: bool, paused_by: str | None, game_hour: int,
                  max_wall: float | None) -> ControlDecision:
        if action == "speed" or self.kind == "agent":
            actor = self.agent_id if self.kind == "agent" else None
        else:
            actor = requester if requester_kind == "agent" else None
        return ControlDecision(request_id, action, requester, requester_kind, granted, reason, speed, paused,
                               paused_by, max(self._game_hour, game_hour), actor if granted else None,
                               max_wall,
                               self._budgets())

    def _decide(self, who: PeerInfo, action: str, message: dict[str, Any]) -> ControlDecision:
        request_id = str(message.get("request_id") or secrets.token_hex(8))
        hour = message.get("game_hour", 0)
        hour = hour if type(hour) is int and hour >= 0 else 0
        is_host = who.agent_id == self.agent_id
        with self._cond:
            current = self.control
            if self.phase != "running" or self.match is None or current is None:
                return ControlDecision(request_id, action, who.agent_id, who.kind, False,
                                       "no match is running",
                                       current.speed if current else 1, bool(current and current.paused),
                                       None,
                                       hour)
            now: ControlDecision = current
            self._game_hour = max(self._game_hour, hour)

            def deny(reason: str) -> ControlDecision:
                return self._decision(request_id, action, who.agent_id, who.kind, False, reason, now.speed,
                                      now.paused, now.paused_by, hour, None)

            if who.agent_id not in dict(self.match.sides):
                return deny(f"{who.agent_id} does not play in this episode")
            if action == "speed":
                speed, policy = message.get("speed"), self.match.speed
                if type(speed) is not int or not 1 <= speed <= 5:
                    return deny("speed must be an integer 1-5")
                if who.kind != "human" and policy.mode == "host_only" and not is_host:
                    return deny("speed policy is host_only")
                if self.kind == "human" and not is_host:
                    return deny("only the HOI4 host can change speed and the host is a human player")
                granted = min(speed, policy.max_speed)
                note = f"speed policy {policy.mode}" + (f"; capped at {granted}" if granted != speed else "")
                return self._decision(request_id, action, who.agent_id, who.kind, True, note, granted,
                                      now.paused,
                                      now.paused_by, hour, None)
            if action == "pause":
                remaining = self._budgets().get(who.agent_id, 0.0)
                allowed, why = pause_permission(self.match.pause, who.kind, is_host, remaining, now.paused_by)
                if not allowed:
                    return deny(why)
                wanted = message.get("max_wall_seconds")
                limit = float(wanted) if isinstance(wanted, (int, float)) and not isinstance(wanted, bool) \
                    and wanted > 0 else None
                if self.match.pause.mode == "budgeted" and who.kind == "agent":
                    limit = min(limit, remaining) if limit is not None else remaining
                self._pause_started = time.monotonic()
                self._pause_deadline = self._pause_started + limit if limit is not None else None
                return self._decision(request_id, action, who.agent_id, who.kind, True,
                                      f"{why}: {str(message.get('reason', ''))[:200]}", now.speed, True,
                                      who.agent_id, hour, limit)
            if not now.paused:
                return deny("the game is not paused")
            if not (who.agent_id == now.paused_by or is_host or who.kind == "human"
                    or self.match.pause.mode == "free"):
                return deny(f"only {now.paused_by} or the host may resume under policy "
                            f"{self.match.pause.mode}")
            return self._end_pause("resume", who.agent_id, who.kind, request_id, "resume", hour)

    def _handle_request(self, who: PeerInfo, action: str, message: dict[str, Any]) -> ControlDecision:
        with self._cond:  # decide and publish atomically: two racing pause requests must not both pass
            return self._publish(self._decide(who, action, message))

    def _publish(self, decision: ControlDecision) -> ControlDecision:
        with self._cond:
            if decision.granted:
                self.control = decision
            self.decisions.append(decision)
            self.log.write("control_decision", decision=decision.to_dict(),
                           episode_id=self.match.episode_id if self.match else None)
            self._broadcast({"type": "control", "decision": decision.to_dict()})
            self._cond.notify_all()
        return decision

    def request_speed(self, speed: int, game_hour: int = 0) -> ControlDecision:
        return self._handle_request(self.info, "speed", {"speed": speed, "game_hour": game_hour})

    def request_pause(self, reason: str, max_wall_seconds: float | None = None,
                      game_hour: int = 0) -> ControlDecision:
        return self._handle_request(self.info, "pause", {
            "reason": reason, "max_wall_seconds": max_wall_seconds, "game_hour": game_hour})

    def request_resume(self, game_hour: int = 0) -> ControlDecision:
        return self._handle_request(self.info, "resume", {"game_hour": game_hour})


# ---------------------------------------------------------------------------------------------
# Guest
# ---------------------------------------------------------------------------------------------

class LanGuest(_Base):
    """Connects to a ``LanHost``. ``kind="human"`` is a thin client that auto-acknowledges.

    A crashed guest process rejoins the same match by connecting again with the same agent id:
    the welcome carries the running ``MatchSpec``, phase and control state. ``role="probe"`` is a
    latency probe that needs no fingerprint and never takes part in a match.
    """

    def __init__(self, fingerprint: BuildFingerprint | None, agent_id: str, country: Country | None, *,
                 kind: str = "agent", role: str = "guest", secret: bytes | None = None,
                 log_path: Path | None = None, heartbeat_seconds: float = 2.0,
                 peer_timeout_seconds: float = 6.0) -> None:
        super().__init__(fingerprint, agent_id, country, role, kind, secret, log_path, heartbeat_seconds,
                         peer_timeout_seconds)
        if role not in ("guest", "probe") or (role == "guest" and (fingerprint is None or country is None)):
            raise LanError("a guest needs a fingerprint and a country; only a probe may omit them")
        self._wire: _Wire | None = None
        self._target: tuple[str, int] | None = None
        self._last_seen = 0.0
        self._decisions: dict[str, ControlDecision] = {}
        self._pongs: dict[str, float] = {}
        self._reset_episode: str | None = None
        self.start_game_hour = 0
        self.host_info: dict[str, Any] = {}
        self.resumed = False
        self.result: dict[str, Any] | None = None
        self.errors: list[dict[str, Any]] = []

    @property
    def connected(self) -> bool:
        wire = self._wire
        return wire is not None and not wire.closed.is_set()

    def connect(self, host: str, port: int, timeout: float = 5.0) -> dict[str, Any]:
        """Handshake; retries while the host still believes a crashed predecessor is connected."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self._connect_once(host, port, deadline)
            except LanRejected as exc:
                if exc.code != "agent_id_in_use" or time.monotonic() + 0.3 > deadline:
                    raise
                time.sleep(0.1)

    def reconnect(self, timeout: float = 5.0) -> dict[str, Any]:
        if self._target is None:
            raise LanError("never connected")
        if self._wire is not None:
            self._wire.close()
        return self.connect(*self._target, timeout=timeout)

    def _connect_once(self, host: str, port: int, deadline: float) -> dict[str, Any]:
        try:
            sock = socket.create_connection((host, port), timeout=max(0.1, deadline - time.monotonic()))
        except OSError as exc:
            raise LanClosed(f"cannot reach {host}:{port}: {exc}") from exc
        wire = _Wire(sock, self.log, f"{host}:{port}")
        try:
            ip = sock.getpeername()[0]
            if self._secret is None and not is_lan_address(ip):
                raise LanError(f"{ip} is not loopback/RFC1918 and no shared secret ({SECRET_ENV}) is "
                               "configured")
            challenge = wire.recv(deadline)
            if challenge["type"] != "challenge" or challenge.get("v") != LAN_PROTOCOL_VERSION:
                raise LanError(f"host speaks LAN protocol {challenge.get('v')!r}, "
                               f"this PC {LAN_PROTOCOL_VERSION}")
            host_nonce = str(challenge.get("nonce", ""))
            if (challenge.get("auth") == "hmac-sha256") != (self._secret is not None):
                raise LanError(f"authentication mismatch: host uses {challenge.get('auth')}, this PC has "
                               f"{SECRET_ENV} {'set' if self._secret else 'unset'}; configure both PCs alike")
            nonce = secrets.token_hex(16)
            claimed = asdict(self.fingerprint) if self.fingerprint else None
            hello: dict[str, Any] = {"type": "hello", "agent_id": self.agent_id, "role": self.role,
                                     "kind": self.kind, "nonce": nonce, "fingerprint": claimed,
                                     "country": self.country.value if self.country else None}
            if self._secret is not None:
                hello["mac"] = _mac(self._secret, "hello", host_nonce, nonce, self.agent_id, self.role,
                                    self.kind,
                                    str(hello["country"]), _canonical(claimed))
            wire.send(hello)
            welcome = wire.recv(deadline)
            if welcome["type"] in ("reject", "error"):
                raise LanRejected(str(welcome.get("code")), str(welcome.get("message")))
            host_id = str(welcome.get("host", {}).get("agent_id", ""))
            if welcome["type"] != "welcome" or (self._secret is not None and not hmac.compare_digest(
                    _mac(self._secret, "welcome", nonce, host_nonce, host_id), str(welcome.get("mac", "")))):
                raise LanError("host failed to prove knowledge of the shared secret")
        except LanError:
            wire.close()
            raise
        with self._cond:
            self._wire, self._target, self._last_seen = wire, (host, port), time.monotonic()
            self.host_info, self.resumed = welcome["host"], bool(welcome.get("resumed"))
            self.phase = str(welcome.get("phase", "idle"))
            self.match = MatchSpec.from_dict(welcome["match"]) if welcome.get("match") else None
            self.control = ControlDecision.from_dict(welcome["control"]) if welcome.get("control") else None
            self._cond.notify_all()
        self.log.write("connected", host=self.host_info, resumed=self.resumed, phase=self.phase)
        for target in (self._read_loop, self._heartbeat_loop):
            threading.Thread(target=target, args=(wire,), daemon=True,
                             name=f"lan-guest-{target.__name__}").start()
        return welcome

    def close(self) -> None:
        self._stop.set()
        wire = self._wire
        if wire is not None:
            try:
                wire.send({"type": "bye", "reason": "guest closing"})
            except LanError:
                pass
            wire.close()
        self.log.close()

    def __exit__(self, *_: object) -> None:
        self.close()

    def __enter__(self) -> LanGuest:
        return self

    # --- threads -----------------------------------------------------------------------------
    def _read_loop(self, wire: _Wire) -> None:
        reason = "closed"
        try:
            while True:
                message = wire.recv(None, self._stop)
                with self._cond:
                    self._last_seen = time.monotonic()
                if not self._dispatch(wire, message):
                    reason = "host said bye"
                    return
        except LanError as exc:
            reason = str(exc)
        finally:
            wire.close()
            with self._cond:
                self.log.write("disconnected", reason=reason)
                self._cond.notify_all()

    def _heartbeat_loop(self, wire: _Wire) -> None:
        while not wire.closed.wait(self.heartbeat_seconds):
            if time.monotonic() - self._last_seen > self.peer_timeout_seconds:
                self.log.write("host_timeout", silent_seconds=round(time.monotonic() - self._last_seen, 3))
                wire.close()
                return
            try:
                wire.send({"type": "ping", "t": time.perf_counter_ns()})
            except LanError:
                return

    def _dispatch(self, wire: _Wire, message: dict[str, Any]) -> bool:
        kind = message["type"]
        if kind == "ping":
            wire.send({"type": "pong", "id": message.get("id"), "t": message.get("t")})
            return True
        if kind == "bye":
            return False
        with self._cond:
            if kind == "pong":
                if isinstance(message.get("id"), str) and isinstance(message.get("t"), int):
                    self._pongs[message["id"]] = (time.perf_counter_ns() - message["t"]) / 1e9
            elif kind == "propose":
                self.match, self.phase = MatchSpec.from_dict(message["spec"]), "proposed"
                self.control, self.result, self._reset_episode = None, None, None
                if self.kind == "human":
                    wire.send({"type": "ready", "episode_id": self.match.episode_id})
            elif kind == "start":
                self.phase, self.start_game_hour = "running", int(message.get("game_hour", 0))
                self.control = ControlDecision.from_dict(message["control"])
            elif kind == "control":
                decision = ControlDecision.from_dict(message["decision"])
                if decision.granted:
                    self.control = decision
                elif self.control is not None:  # a denial still refreshes the budgets we predict from
                    self.control = ControlDecision(**{**asdict(self.control), "budgets": decision.budgets})
                if decision.requester == self.agent_id:
                    self._decisions[decision.request_id] = decision
            elif kind == "result":
                self.result, self.phase = message, "finished"
            elif kind == "reset":
                self._reset_episode = str(message.get("episode_id"))
                if self.kind == "human":
                    self._ack_reset(wire)
            elif kind in ("error", "reject"):
                self.errors.append(message)
            self._cond.notify_all()
        return True

    def _send(self, message: dict[str, Any]) -> None:
        wire = self._wire
        if wire is None:
            raise LanClosed("not connected")
        wire.send(message)

    # --- lifecycle as seen by the guest agent -------------------------------------------------
    def wait_proposal(self, timeout: float) -> MatchSpec:
        """The proposed (or, after a reconnect, already running) match."""
        self._wait(lambda: self.match is not None and self.phase in ("proposed", "running"), timeout,
                   "a proposal")
        assert self.match is not None
        return self.match

    def ready(self) -> None:
        if self.match is None:
            raise LanError("no proposal to acknowledge")
        self._send({"type": "ready", "episode_id": self.match.episode_id})

    def decline(self, reason: str) -> None:
        if self.match is None:
            raise LanError("no proposal to decline")
        self._send({"type": "decline", "episode_id": self.match.episode_id, "reason": reason})

    def wait_start(self, timeout: float) -> int:
        self._wait(lambda: self.phase == "running", timeout, "start")
        return self.start_game_hour

    def report_outcome(self, outcome: Outcome) -> None:
        if self.kind == "human":
            raise LanError("a human peer reports no outcome; the agent side does")
        self._send({"type": "outcome", "outcome": outcome.to_dict()})

    def wait_result(self, timeout: float) -> dict[str, Any]:
        self._wait(lambda: self.result is not None, timeout, "the host's cross-checked result")
        assert self.result is not None
        return self.result

    def wait_reset(self, timeout: float) -> str:
        self._wait(lambda: self._reset_episode is not None, timeout, "reset")
        assert self._reset_episode is not None
        return self._reset_episode

    def _ack_reset(self, wire: _Wire) -> None:
        wire.send({"type": "reset_ack", "episode_id": self._reset_episode})
        self.phase, self.match, self.control, self._reset_episode = "idle", None, None, None

    def ack_reset(self) -> None:
        """Call after the local arena reset decision was clicked."""
        with self._cond:
            if self._reset_episode is None or self._wire is None:
                raise LanError("no reset to acknowledge")
            self._ack_reset(self._wire)

    # --- speed / pause -----------------------------------------------------------------------
    def _request(self, kind: str, fields: dict[str, Any], timeout: float) -> ControlDecision:
        request_id = secrets.token_hex(8)
        self._send({"type": kind, "request_id": request_id, **fields})
        self._wait(lambda: request_id in self._decisions or not self.connected, timeout,
                   f"the {kind} decision")
        with self._cond:
            if request_id not in self._decisions:
                raise LanClosed("disconnected before the host decided")
            return self._decisions.pop(request_id)

    def request_speed(self, speed: int, game_hour: int = 0, timeout: float = 5.0) -> ControlDecision:
        return self._request("request_speed", {"speed": speed, "game_hour": game_hour}, timeout)

    def request_pause(self, reason: str, max_wall_seconds: float | None = None, game_hour: int = 0,
                      timeout: float = 5.0) -> ControlDecision:
        return self._request("request_pause", {"reason": reason, "max_wall_seconds": max_wall_seconds,
                                               "game_hour": game_hour}, timeout)

    def request_resume(self, game_hour: int = 0, timeout: float = 5.0) -> ControlDecision:
        return self._request("request_resume", {"game_hour": game_hour}, timeout)

    def ping(self, timeout: float = 5.0) -> float:
        """Application-level round trip in seconds (includes JSON and thread wake-up on both PCs)."""
        ping_id = secrets.token_hex(8)
        self._send({"type": "ping", "id": ping_id, "t": time.perf_counter_ns()})
        self._wait(lambda: ping_id in self._pongs or not self.connected, timeout, "pong")
        with self._cond:
            if ping_id not in self._pongs:
                raise LanClosed("disconnected before the pong")
            return self._pongs.pop(ping_id)


# ---------------------------------------------------------------------------------------------
# Side-swapped evaluation schedule
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ScheduledGame:
    """One of the four games of an evaluation block: {initialization, candidate} x {BLU, RED}."""
    pair_id: str
    candidate: str
    training_seed: int
    side: Country
    scenario: str
    opponent: str
    split: str
    match_seed: int

    def match_spec(self, episode_id: str, evaluated_agent: str, opponent_agent: str, *,
                   opponent_sha256: str | None = None, pause: PausePolicy | None = None,
                   speed: SpeedPolicy | None = None, horizon_hours: int = 90 * 24,
                   humans: tuple[str, ...] = ()) -> MatchSpec:
        return MatchSpec(episode_id, self.scenario, self.split, self.match_seed,
                         tuple(sorted(((evaluated_agent, self.side), (opponent_agent, self.side.opponent)))),
                         horizon_hours, self.opponent, opponent_sha256, "", pause or PausePolicy(),
                         speed or SpeedPolicy(), humans)

    def evaluation_game(self, record: MatchRecord, evaluated_agent: str, *, source: str) -> EvaluationGame:
        """``source`` is mandatory: only real LAN HOI4 matches may say ``hoi4_vision``.

        Raises for disputed or aborted records; those must be replayed, not guessed.
        """
        if record.spec.scenario_id != self.scenario \
                or record.spec.country_of(evaluated_agent) is not self.side:
            raise LanError("record does not belong to this scheduled game")
        return EvaluationGame(self.candidate, self.training_seed, self.pair_id, self.side.value,
                              self.scenario,
                              self.opponent, record.result_for(evaluated_agent), source)


def side_swapped_schedule(scenarios: Sequence[str], opponent: str, training_seed: int, *,
                          pairs_per_scenario: int = 1, split: str = "validation",
                          base_seed: int = 0) -> Iterator[ScheduledGame]:
    """Yield complete blocks for ``evaluation.improvement_report``.

    Every ``pair_id`` block holds both policies on both sides against the same scenario, fixed
    opponent and scenario seed, so a side or seed advantage cancels inside the block.
    """
    if not scenarios or not opponent or pairs_per_scenario < 1 or split not in SPLITS:
        raise LanError("schedule needs scenarios, an opponent, a known split and at least one pair")
    for scenario in scenarios:
        for index in range(pairs_per_scenario):
            pair_id = f"{scenario}|{opponent}|seed{training_seed}|pair{index}"
            digest = hashlib.sha256(f"{base_seed}|{pair_id}".encode()).digest()
            match_seed = int.from_bytes(digest[:4], "big") >> 1
            for candidate in ("initialization", "candidate"):
                for side in (Country.BLUE, Country.RED):
                    yield ScheduledGame(pair_id, candidate, training_seed, side, scenario, opponent, split,
                                        match_seed)


# ---------------------------------------------------------------------------------------------
# League matchmaking
# ---------------------------------------------------------------------------------------------

def verify_frozen_checkpoint(spec: MatchSpec, directory: Path) -> Path | None:
    """Prove this PC holds the same frozen opponent file the host sampled (None for scripted)."""
    if spec.opponent_sha256 is None:
        return None
    path = directory / (spec.opponent_sha256 + ".pt")
    if not path.is_file():
        raise LanError(f"frozen opponent {spec.opponent_id} is missing here: copy {path.name} into "
                       f"{directory}")
    if file_hash(path) != spec.opponent_sha256:
        raise LanError(f"frozen opponent file {path} does not hash to {spec.opponent_sha256}")
    return path


@dataclass(frozen=True)
class Assignment:
    learner: str  # peer running the learning policy
    opponent_peer: str  # peer that loads and plays the sampled league opponent
    opponent: Opponent
    learner_country: Country


class LeagueMatchmaker:
    """Pairs connected peers and samples one league opponent per pair and episode.

    Peers are paired in order (0-1, 2-3, ...); each pair needs its own HOI4 game and ``LanHost``.
    The learner's country alternates with the episode index so sides stay balanced.
    """

    def __init__(self, league: League, peers: Sequence[str], league_id: str = "league") -> None:
        if len(peers) < 2 or len(set(peers)) != len(peers):
            raise LanError("matchmaking needs at least two distinct peers")
        self.league, self.peers, self.league_id = league, tuple(peers), league_id
        self.unrecorded: list[MatchRecord] = []

    @property
    def idle_peer(self) -> str | None:
        return self.peers[-1] if len(self.peers) % 2 else None

    def assign(self, episode_index: int) -> list[Assignment]:
        integer(episode_index, "episode index")
        country = Country.BLUE if episode_index % 2 == 0 else Country.RED
        return [Assignment(self.peers[i], self.peers[i + 1], self.league.sample_episode(), country)
                for i in range(0, len(self.peers) - 1, 2)]

    def match_spec(self, assignment: Assignment, episode_id: str, scenario_id: str, seed: int, *,
                   split: str = "train", pause: PausePolicy | None = None, speed: SpeedPolicy | None = None,
                   horizon_hours: int = 90 * 24) -> MatchSpec:
        sides = ((assignment.learner, assignment.learner_country),
                 (assignment.opponent_peer, assignment.learner_country.opponent))
        return MatchSpec(episode_id, scenario_id, split, seed, tuple(sorted(sides)), horizon_hours,
                         assignment.opponent.id, assignment.opponent.sha256, self.league_id,
                         pause or PausePolicy(), speed or SpeedPolicy())

    def record(self, assignment: Assignment, record: MatchRecord) -> str | None:
        """Feed an agreed result to ``League.record`` from the learner's perspective.

        Disputed, incomplete or aborted matches are kept in ``unrecorded`` and return None.
        """
        if record.spec.opponent_id != assignment.opponent.id:
            raise LanError("record belongs to a different opponent")
        if not record.agreed or record.aborted:
            self.unrecorded.append(record)
            return None
        result = record.result_for(assignment.learner)
        self.league.record(assignment.opponent.id, result)
        return result


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def _load_fingerprint(path: Path) -> BuildFingerprint:
    try:
        return BuildFingerprint(**json.loads(path.read_text(encoding="utf-8")))
    except (TypeError, ValueError) as exc:
        raise LanError(f"{path} is not the output of the `fingerprint` command: {exc}") from exc


def _cmd_host(args: argparse.Namespace) -> int:
    host = LanHost(_load_fingerprint(args.fingerprint), args.agent_id, Country(args.country), kind=args.kind,
                   bind=args.bind, port=args.port, secret=secret_from_env(), log_path=args.log)
    records: list[dict[str, Any]] = []
    with host:
        print(json.dumps({"listening": list(host.address), "security": host.security_note,
                          "event_log": str(args.log)}, indent=2), flush=True)
        host.wait_for_guests(args.expect_guests, args.wait_seconds)
        print(json.dumps({"peers": host.peers()}, indent=2), flush=True)
        if args.episodes and args.kind != "human":
            raise LanError("--episodes drives a human-hosted match only; an agent host is driven through the "
                           "LanHost API by the agent loop, which must report its own outcome")
        guests = [agent for agent, row in host.peers().items() if row["connected"]]
        for index in range(args.episodes):
            sides = ((args.agent_id, Country(args.country)), (guests[0], Country(args.country).opponent))
            spec = MatchSpec(f"{args.scenario}-{args.seed + index}-{secrets.token_hex(4)}", args.scenario,
                             args.split, args.seed + index, tuple(sorted(sides)), args.horizon_hours,
                             pause=PausePolicy(args.pause_policy, args.pause_budget_seconds),
                             speed=SpeedPolicy(args.speed_policy, args.max_speed))
            host.propose(spec)
            host.wait_ready(args.wait_seconds)
            host.start()
            record = host.wait_outcomes(args.outcome_timeout_seconds)
            records.append(record.to_dict())
            print(json.dumps(records[-1], indent=2), flush=True)
            host.reset(args.wait_seconds)
        if args.serve_seconds:
            host._stop.wait(args.serve_seconds)
    return 0 if all(row["agreed"] for row in records) else 2


def _cmd_join(args: argparse.Namespace) -> int:
    guest = LanGuest(_load_fingerprint(args.fingerprint), args.agent_id, Country(args.country),
                     kind=args.kind,
                     secret=secret_from_env(), log_path=args.log)
    with guest:
        welcome = guest.connect(args.host, args.port, args.timeout_seconds)
        print(json.dumps({"connected_to": welcome["host"], "resumed": guest.resumed, "phase": guest.phase,
                          "rtt_ms": round(guest.ping() * 1000, 3), "event_log": str(args.log),
                          "note": "kind=human auto-acknowledges proposals and resets; an agent guest is "
                                  "driven through the LanGuest API by the agent loop"}, indent=2), flush=True)
        deadline = time.monotonic() + args.seconds if args.seconds else None
        while guest.connected and (deadline is None or time.monotonic() < deadline):
            time.sleep(0.2)
    return 0


def _cmd_ping(args: argparse.Namespace) -> int:
    probe = LanGuest(None, f"probe-{secrets.token_hex(4)}", None, role="probe", secret=secret_from_env())
    with probe:
        probe.connect(args.host, args.port, args.timeout_seconds)
        samples = [probe.ping(args.timeout_seconds) * 1000 for _ in range(args.count)]
    print(json.dumps({"host": args.host, "port": args.port, "count": len(samples),
                      "rtt_ms": {"min": round(min(samples), 3),
                                 "median": round(statistics.median(samples), 3),
                                 "max": round(max(samples), 3)},
                      "measures": "application-level JSON ping/pong between coordinator processes, not ICMP"},
                     indent=2))
    return 0


def add_commands(commands: Any) -> dict[str, Callable[[argparse.Namespace], int]]:
    """Register ``lan-host``, ``lan-join`` and ``lan-ping`` on an argparse subparsers object."""
    name = socket.gethostname()[:48] or "pc"
    host = commands.add_parser("lan-host", help="host the LAN coordinator next to the HOI4 multiplayer host")
    host.add_argument("--bind", default="127.0.0.1", help="explicit LAN address of this PC; never 0.0.0.0")
    host.add_argument("--expect-guests", type=int, default=1)
    host.add_argument("--wait-seconds", type=float, default=300.0)
    host.add_argument("--episodes", type=int, default=0,
                      help="human-hosted episodes to coordinate (kind=human)")
    host.add_argument("--scenario", default="arena_default")
    host.add_argument("--split", choices=SPLITS, default="train")
    host.add_argument("--seed", type=int, default=0)
    host.add_argument("--horizon-hours", type=int, default=90 * 24)
    host.add_argument("--pause-policy", choices=PAUSE_MODES, default="free")
    host.add_argument("--pause-budget-seconds", type=float, default=0.0)
    host.add_argument("--speed-policy", choices=SPEED_MODES, default="free")
    host.add_argument("--max-speed", type=int, default=5)
    host.add_argument("--outcome-timeout-seconds", type=float, default=4 * 3600.0)
    host.add_argument("--serve-seconds", type=float, default=0.0,
                      help="keep serving (pings, joins) afterwards")
    join = commands.add_parser("lan-join", help="join a LAN coordinator host from another PC")
    join.add_argument("--seconds", type=float, default=0.0,
                      help="stay connected this long; 0 = until the host closes")
    ping = commands.add_parser("lan-ping", help="round-trip latency to a LAN coordinator host")
    ping.add_argument("--count", type=int, default=20)
    for parser, country in ((host, "BLU"), (join, "RED")):
        parser.add_argument("--fingerprint", type=Path, required=True,
                            help="JSON from the `fingerprint` command")
        parser.add_argument("--agent-id", default=name)
        parser.add_argument("--country", choices=[c.value for c in Country], default=country)
        parser.add_argument("--kind", choices=KINDS, default="agent")
        parser.add_argument("--log", type=Path,
                            default=Path(f"artifacts/lan/{parser.prog.split()[-1]}-events.jsonl"))
    host.add_argument("--port", type=int, default=DEFAULT_PORT)
    for parser in (join, ping):
        parser.add_argument("--host", required=True)
        parser.add_argument("--port", type=int, default=DEFAULT_PORT)
        parser.add_argument("--timeout-seconds", type=float, default=5.0)
    return {"lan-host": _cmd_host, "lan-join": _cmd_join, "lan-ping": _cmd_ping}
