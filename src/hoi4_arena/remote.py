"""Pinned TLS connection to the same raw desktop protocol on another Windows PC: the second
PC's worker, the fleet service `hoi4-worker` on that PC's loopback, reached there directly
and from here through `fleet tunnel` on the same port."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import socket
import ssl
import threading
import time
from pathlib import Path

from .desktop import Desktop, DesktopError, read_reply

log = logging.getLogger(__name__)

# How long a client of the pairing keeps trying to connect: a service restart (fleet starts
# it again within ~5 s, and it compiles its bridge in a few more) or a node restart (its
# tunnel's connections end) is waited out, not a session's failure.
RECONNECT_SECONDS = 60


def open_tls(spec, context, wait=0.0, *, pause=2.0):
    """The TLS connection to the worker `spec` (a pairing) names, tried again every
    `pause` seconds for up to `wait` seconds while it cannot be made: while fleet's tunnel
    is down (a node restart ends its connections) or up with no bridge behind it (the
    service restarting), the connection is refused, or closed during the handshake. With
    `wait` 0 the first failure is raised, as it always was."""
    deadline = time.monotonic() + wait
    while True:
        try:
            raw = socket.create_connection((spec["host"], spec["port"]), timeout=10)
            try:
                return context.wrap_socket(raw, server_hostname=spec["host"])
            except BaseException:
                raw.close()
                raise
        except OSError as error:  # Refused, reset, closed mid-handshake (ssl's are OSErrors).
            if time.monotonic() + pause > deadline:
                raise
            log.info("no worker at %s:%s yet (%s)", spec["host"], spec["port"], error)
            time.sleep(pause)


class RemoteDesktop(Desktop):
    encoding = "lz4"

    def __init__(
        self, config, *, attach: bool = True, observer: bool = False, wait: float | None = None
    ):
        """Connect to the second PC's worker. `attach=False` as for Desktop: the control
        operations need no game running there.

        `observer=True` asks for a read-only connection beside the one that holds the game:
        telemetry, `report`, captures, the game log, and no input, launches or recording.
        A bridge from before observers refuses it by closing the connection.

        `wait`: seconds to keep trying while the worker cannot be reached (open_tls). By
        default the pairing's `reconnect_seconds` (RECONNECT_SECONDS in one bundle-peer
        writes), which waits out a tunnel or service restart; without it, none.
        """
        from collections import deque

        # The peer worker's stderr stays on the peer's console; keep the attribute so
        # callers can record a worker log uniformly for local and remote desktops.
        self.diagnostics = deque(maxlen=64)
        self.close_error = None
        spec = json.loads(Path(config).read_text())
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE  # Exact certificate pin is verified below.
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        wait = float(spec.get("reconnect_seconds", 0)) if wait is None else wait
        self.socket = open_tls(spec, context, wait)
        pin = hashlib.sha256(self.socket.getpeercert(binary_form=True)).hexdigest()
        if not hmac.compare_digest(pin, spec["certificate_sha256"]):
            self.socket.close()
            raise DesktopError("Second-PC certificate does not match its pairing file")
        # The connect timeout must not stay on the socket: the reader blocks while no
        # request is outstanding, and a 10 s idle read would drop a healthy match.
        self.socket.settimeout(None)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # Requests are small and go out at once, not after the last one is acknowledged.
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if hasattr(socket, "SIO_KEEPALIVE_VALS") and hasattr(self.socket, "ioctl"):
            # Windows probes an idle connection only after two hours by default: a peer
            # that vanished would leave the reader waiting that long. Probe after 10 s of
            # quiet, every 2 s.
            self.socket.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 10_000, 2_000))
        self.stream = self.socket.makefile("rwb")
        # Same id-demuxed protocol as a local worker. Apply and capture are in flight
        # together, and a half-open peer has to fail the request instead of blocking
        # the match forever. Closing the socket is what unblocks the reader.
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = {}
        self.next_id = 1
        self.reader_error = None
        self.streams = {}
        # An observer is read-only: it can watch and measure while another connection
        # holds the game, and the bridge starts its worker with --observer.
        role = b" observer" if observer else b""
        self.stream.write(spec["token"].encode("ascii") + role + b"\n")
        self.stream.flush()
        threading.Thread(target=self._read_remote, daemon=True).start()
        self.attached = None
        if attach:
            try:
                self.attached = self.request("attach")
            except Exception:
                self._shutdown()
                raise

    def _read_remote(self):
        try:
            while True:
                self._deliver(read_reply(self.stream))
        except Exception as error:
            self._fail_pending(error)

    def _send(self, payload: bytes):
        try:
            self.stream.write(payload)
            self.stream.flush()
        except OSError as error:
            raise DesktopError(f"Remote desktop disconnected: {error}") from error

    def _alive(self) -> bool:
        return True

    def _shutdown(self):
        # Shut the socket down before closing the stream. The reader thread is blocked in
        # a read that holds the stream's buffer lock, so closing the stream first waits on
        # that read forever. The shutdown ends the read, and the reader fails pending work.
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except Exception:  # noqa: BLE001 - already closed or never connected.
            pass
        try:
            self.stream.close()
        except Exception:  # noqa: BLE001 - the socket close is the one that matters.
            pass
        try:
            self.socket.close()
        except Exception:  # noqa: BLE001
            pass

    def worker_log(self) -> list[str]:
        # A failed request asks for the log to explain itself (Desktop._detail), and a
        # failed log request would ask again: on a worker that stopped answering, that
        # recursed every 2 s for as long as anyone waited. Ask once, and not at all once
        # the connection is gone: nothing asked on it can be answered.
        if getattr(self, "_asking_for_log", False) or getattr(self, "reader_error", None):
            return list(self.diagnostics)
        self._asking_for_log = True
        try:
            reply = self.request("status", timeout=2)
            lines = reply.get("log") or []
            if lines:
                return [str(line) for line in lines]
        except (DesktopError, OSError, ValueError) as error:
            log.warning("remote worker log unavailable: %s", error)
        finally:
            self._asking_for_log = False
        return list(self.diagnostics)

    def close(self):
        # Same contract as Desktop.close: record the failed release as evidence rather
        # than raising out of __exit__ over an exception that is already propagating.
        try:
            self.release()
        except (DesktopError, OSError, ValueError) as error:
            self.close_error = f"{type(error).__name__}: {error}"
        finally:
            self._shutdown()


def bundle(output, port):
    """A new pairing in `output` (a new folder, such as artifacts/pairing): for the second
    PC's worker service, second-pc/server.json (its port, token and certificate password)
    and second-pc/worker.pfx, which scripts/collect_station.py deploy --worker ships; for
    its clients, peer-fleet.json, the worker at 127.0.0.1:`port` with the token and the
    certificate's pin. It is the same file on both PCs: there the service listens on that
    port of its loopback, and here `fleet tunnel` opens the same port. Opens no port."""
    import datetime
    import secrets

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import pkcs12
    from cryptography.x509.oid import NameOID

    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    peer = root / "second-pc"
    peer.mkdir()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "HOI4 paired worker")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    password, token = secrets.token_hex(32), secrets.token_hex(32)
    (peer / "worker.pfx").write_bytes(
        pkcs12.serialize_key_and_certificates(
            b"hoi4", key, cert, None, serialization.BestAvailableEncryption(password.encode())
        )
    )
    (peer / "server.json").write_text(
        json.dumps({"port": port, "token": token, "pfx_password": password}, indent=2)
    )
    client = root / "peer-fleet.json"
    client.write_text(
        json.dumps(
            {
                "host": "127.0.0.1",
                "port": port,
                "token": token,
                "certificate_sha256": cert.fingerprint(hashes.SHA256()).hex(),
                "reconnect_seconds": RECONNECT_SECONDS,
            },
            indent=2,
        )
    )
    return {"worker": str(peer), "client_config": str(client)}
