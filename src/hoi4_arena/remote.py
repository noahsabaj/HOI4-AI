"""Pinned TLS connection to the same raw desktop protocol on another Windows PC."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import socket
import ssl
import threading
from pathlib import Path

from .desktop import Desktop, DesktopError, read_reply, worker_executable

log = logging.getLogger(__name__)


class RemoteDesktop(Desktop):
    def __init__(self, config, *, attach: bool = True):
        """Connect to the second PC's worker. `attach=False` as for Desktop: the control
        operations need no game running there."""
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
        raw = socket.create_connection((spec["host"], spec["port"]), timeout=10)
        self.socket = context.wrap_socket(raw, server_hostname=spec["host"])
        pin = hashlib.sha256(self.socket.getpeercert(binary_form=True)).hexdigest()
        if not hmac.compare_digest(pin, spec["certificate_sha256"]):
            self.socket.close()
            raise DesktopError("Second-PC certificate does not match its pairing file")
        # The connect timeout must not stay on the socket: the reader blocks while no
        # request is outstanding, and a 10 s idle read would drop a healthy match.
        self.socket.settimeout(None)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.stream = self.socket.makefile("rwb")
        # Same id-demuxed protocol as a local worker. Apply and capture are in flight
        # together, and a half-open peer has to fail the request instead of blocking
        # the match forever. Closing the socket is what unblocks the reader.
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = {}
        self.next_id = 1
        self.reader_error = None
        self.stream.write(spec["token"].encode("ascii") + b"\n")
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
        try:
            reply = self.request("status", timeout=2)
            lines = reply.get("log") or []
            if lines:
                return [str(line) for line in lines]
        except (DesktopError, OSError, ValueError) as error:
            log.warning("remote worker log unavailable: %s", error)
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


def bundle(output, host, coordinator, port):
    """Creates a portable worker plus private pairing credentials; never auto-opens a port."""
    import datetime
    import ipaddress
    import secrets
    import shutil

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import pkcs12
    from cryptography.x509.oid import NameOID

    host_ip = ipaddress.ip_address(host)
    ipaddress.ip_address(coordinator)
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
            x509.SubjectAlternativeName([x509.IPAddress(host_ip)]),
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
        json.dumps(
            {
                "bind": host,
                "coordinator": coordinator,
                "port": port,
                "token": token,
                "pfx_password": password,
            },
            indent=2,
        )
    )
    (root / "peer.json").write_text(
        json.dumps(
            {
                "host": host,
                "port": port,
                "token": token,
                "certificate_sha256": cert.fingerprint(hashes.SHA256()).hex(),
            },
            indent=2,
        )
    )
    shutil.copy2(worker_executable(), peer / "hoi4-desktop-worker.exe")
    script = Path(__file__).resolve().parents[2] / "scripts" / "Start-Worker.ps1"
    shutil.copy2(script if script.exists() else "scripts/Start-Worker.ps1", peer)
    (peer / "START-HERE.txt").write_text(
        "Open HOI4, then run Start-Worker.ps1 in PowerShell 7.5 or later (pwsh), not\n"
        "Windows PowerShell 5.1. Leave this window open.\n"
        "F12 stops injected inputs. Close PowerShell to disconnect.\n"
        "Only the paired coordinator can connect. Keep this folder private.\n"
        f"If Windows Firewall blocks this connection, allow TCP {port} only from the\n"
        "coordinator address in server.json on your private network.\n"
    )
    shutil.make_archive(str(root / "second-pc"), "zip", peer)
    return {"bundle": str(root / "second-pc.zip"), "client_config": str(root / "peer.json")}
