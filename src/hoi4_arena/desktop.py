"""Framed stdio transport; SSH can carry the same protocol without a listening input server."""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class DesktopError(RuntimeError):
    pass


@dataclass
class Frame:
    rgb: np.ndarray
    meta: dict
    received_ns: int


class Desktop:
    def __init__(self, command: list[str] | None = None):
        command = command or [str(Path("target/release/hoi4-desktop-worker.exe").resolve())]
        # The worker's only diagnostic channel is stderr. Capture it instead of letting it
        # escape to an inherited console, so failures land beside the run's other evidence.
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.diagnostics = deque(maxlen=64)
        self.close_error = None
        self.lock = threading.Lock()
        self.replies = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        try:
            self.attached = self.request("attach")
        except Exception:
            self._shutdown()
            raise

    def _read(self):
        try:
            while True:
                self.replies.put(read_reply(self.process.stdout))
        except Exception as error:
            self.replies.put(error)

    def _drain(self):
        try:
            for line in self.process.stderr:
                text = line.decode(errors="replace").rstrip()
                if text:
                    self.diagnostics.append(text)
                    log.warning("worker: %s", text)
        except Exception:  # noqa: S110 - a closed pipe simply ends diagnostics.
            pass

    def worker_log(self) -> list[str]:
        return list(self.diagnostics)

    def _detail(self, message: str) -> str:
        tail = self.worker_log()
        return f"{message} ({'; '.join(tail[-3:])})" if tail else message

    def _shutdown(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            # Independent worker watchdog releases keys even with blocked output.
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def request(self, op: str, **kwargs) -> dict:
        with self.lock:
            if self.process.poll() is not None:
                raise DesktopError(self._detail("Desktop worker exited"))
            self.process.stdin.write((json.dumps({"op": op, **kwargs}) + "\n").encode())
            self.process.stdin.flush()
            try:
                reply = self.replies.get(timeout=10)
            except queue.Empty:
                self._shutdown()
                raise DesktopError(self._detail("Desktop response timed out")) from None
            if isinstance(reply, Exception):
                raise DesktopError(self._detail(str(reply))) from reply
            if "error" in reply:
                raise DesktopError(reply["error"])
            return reply

    def capture(self) -> Frame:
        meta = self.request("capture", encoding="lz4")
        payload = meta.pop("payload")
        if meta.get("encoding") == "lz4":
            import lz4.block

            size = meta["height"] * meta["width"] * 4
            if not 0 < size <= 8192 * 8192 * 4:
                raise DesktopError("Invalid decompressed frame size")
            payload = lz4.block.decompress(payload, uncompressed_size=size)
        pixels = np.frombuffer(payload, np.uint8)
        rgb = pixels.reshape(meta["height"], meta["width"], 4)[:, :, [2, 1, 0]].copy()
        if meta["overflow"] or meta["stopped"]:
            raise DesktopError("Input queue overflow or F12 emergency stop")
        return Frame(rgb, meta, time.monotonic_ns())

    def arm(self, *, setup=False):
        self.request("arm", mode="setup" if setup else "match")

    def apply(self, events: list[dict]):
        reply = self.request("apply", events=events)
        reply.pop("payload", None)
        return reply

    def release(self):
        self.request("release")

    def close(self):
        # Record rather than raise: close() runs from __exit__, where raising would
        # replace whatever exception is already propagating. Callers that treat a failed
        # release as run-invalidating evidence read close_error instead.
        try:
            if self.process.poll() is None:
                self.release()
        except (DesktopError, OSError, ValueError) as error:
            self.close_error = f"{type(error).__name__}: {error}"
            log.warning("release during close failed: %s", error)
        finally:
            self._shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_reply(stream):
    line = stream.readline(16 * 1024 * 1024)
    if not line or not line.endswith(b"\n"):
        raise DesktopError("Desktop disconnected or oversized protocol header")
    reply = json.loads(line)
    n = reply.pop("bytes", 0)
    if not isinstance(n, int) or not 0 <= n <= 8192 * 8192 * 4:
        raise DesktopError("Invalid frame size")
    payload = bytearray()
    while len(payload) < n:
        block = stream.read(n - len(payload))
        if not block:
            raise DesktopError("Truncated frame")
        payload.extend(block)
    reply["payload"] = payload
    return reply
