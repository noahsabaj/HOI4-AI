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

from .dataset import DETAIL_SIZE, FOVEA_SIZE, QUADRANTS, Views, parse_cursor

log = logging.getLogger(__name__)


def worker_executable() -> str:
    """The release worker next to this repo, wherever the process was started."""
    candidate = (
        Path(__file__).resolve().parents[2] / "target" / "release" / "hoi4-desktop-worker.exe"
    )
    if candidate.exists():
        return str(candidate)
    return str(Path("target/release/hoi4-desktop-worker.exe").resolve())


def local_control_args() -> list[str]:
    """Worker arguments that point its control operations at this repo.

    The worker looks for Game-Control.ps1 and the arena mods beside itself, which is the
    second PC's layout. Here the script is in scripts/ and the mods in artifacts/mods/.
    """
    root = Path(__file__).resolve().parents[2]
    return ["--scripts", str(root / "scripts"), "--mods", str(root / "artifacts" / "mods")]


class DesktopError(RuntimeError):
    pass


@dataclass
class Frame:
    rgb: np.ndarray | None
    meta: dict
    received_ns: int
    # Populated when the worker downscaled on the capture side. `views` is a
    # dataset.Views already at policy resolution; `crops` holds the calibrated
    # template regions at native resolution, keyed by the order they were requested.
    views: Views | None = None
    crops: list | None = None


class Desktop:
    def __init__(
        self,
        command: list[str] | None = None,
        *,
        worker_args: list[str] | tuple[str, ...] = (),
        attach: bool = True,
    ):
        """Start a local worker and, by default, attach it to the running game.

        `attach=False` is for the control operations (launch, quit, report,
        restart_discord), which need no game: with none running, attach fails.
        `worker_args` go on the worker's command line, such as `local_control_args()`.
        """
        command = [*(command or [worker_executable()]), *worker_args]
        # The worker's only diagnostic channel is stderr. Capture it instead of letting it
        # escape to an inherited console, so failures land beside the run's other evidence.
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.diagnostics = deque(maxlen=64)
        self.close_error = None
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = {}
        self.next_id = 1
        self.reader_error = None
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        self.attached = None
        if attach:
            try:
                self.attached = self.request("attach")
            except Exception:
                self._shutdown()
                raise

    def _read(self):
        try:
            while True:
                self._deliver(read_reply(self.process.stdout))
        except Exception as error:
            self._fail_pending(error)

    def _deliver(self, reply):
        req_id = reply.get("id")
        with self.pending_lock:
            box = self.pending.get(req_id)
        if box is not None:
            box.put(reply)
        elif "error" in reply:
            # An error the worker could not tie to a request, such as a line it could not
            # parse. Keep it as evidence rather than dropping it silently.
            self.diagnostics.append(f"unmatched worker error: {reply['error']}")
            log.warning("unmatched worker error: %s", reply["error"])

    def _fail_pending(self, error):
        # Under the lock request() registers under, so no request can slip in after the
        # snapshot and wait on a reader that has already stopped.
        with self.pending_lock:
            self.reader_error = error
            boxes = list(self.pending.values())
        for box in boxes:
            box.put(error)

    def _send(self, payload: bytes):
        self.process.stdin.write(payload)
        self.process.stdin.flush()

    def _alive(self) -> bool:
        return self.process.poll() is None

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

    def request(self, op: str, timeout: float = 10, **kwargs) -> dict:
        """One worker operation.

        Capture and apply are in flight together: the dispatch thread applies the
        eight slots while this thread captures. Replies carry the request id, so
        they can complete in either order. The write lock only covers the send.
        """
        box = queue.Queue()
        with self.write_lock:
            if not self._alive():
                raise DesktopError(self._detail("Desktop worker exited"))
            req_id = self.next_id
            self.next_id += 1
            # Strict JSON: the worker rejects NaN, and its reply to an unparseable line
            # cannot carry the id this request waits on.
            message = json.dumps({"op": op, "id": req_id, **kwargs}, allow_nan=False)
            with self.pending_lock:
                if self.reader_error is not None:
                    raise DesktopError(self._detail(f"Desktop reader stopped: {self.reader_error}"))
                self.pending[req_id] = box
            try:
                self._send((message + "\n").encode())
            except Exception:
                with self.pending_lock:
                    self.pending.pop(req_id, None)
                raise
        try:
            reply = box.get(timeout=timeout)
        except queue.Empty:
            self._shutdown()
            raise DesktopError(self._detail("Desktop response timed out")) from None
        finally:
            with self.pending_lock:
                self.pending.pop(req_id, None)
        if isinstance(reply, Exception):
            raise DesktopError(self._detail(str(reply))) from reply
        if "error" in reply:
            raise DesktopError(reply["error"])
        return reply

    def capture(
        self, *, views=None, detail=DETAIL_SIZE, fovea=FOVEA_SIZE, regions=None, full=None
    ) -> Frame:
        """Capture a frame, optionally downscaled and cropped by the worker.

        `views` asks the worker for the policy views: the global frame at that size, the
        four quadrants at `detail`, and a native `fovea` square centered on the pointer
        (see dataset.views). `regions` is a list of [x, y, w, h] crops at native
        resolution. Asking for either keeps the full frame off the wire. The full frame
        is returned only when nothing narrower was requested, or `full=True`.
        """
        options = {}
        if views:
            options["views"] = int(views)
            options["detail"] = int(detail)
            options["fovea"] = int(fovea)
        if regions:
            options["regions"] = [[int(v) for v in r] for r in regions]
        if full is not None:
            options["full"] = bool(full)
        meta = self.request("capture", encoding="lz4", **options)
        payload = meta.pop("payload")
        # A worker built before worker-side downscaling accepts these options, ignores
        # them, and sends the whole 33 MB frame back. That is indistinguishable from a
        # working one except by the reply, so say so rather than silently paying for it
        # every tick. This is how a stale deployed binary stayed hidden once already.
        if options.get("views") and not meta.get("views_bytes"):
            raise DesktopError(
                "Worker ignored the requested views; it predates worker-side "
                "downscaling. Rebuild and redeploy hoi4-desktop-worker."
            )
        if len(meta.get("region_bytes", [])) != len(options.get("regions", [])):
            raise DesktopError("Worker returned a different number of crops than requested")
        full_bytes = meta.get("full_bytes", meta["height"] * meta["width"] * 4)
        region_bytes = meta.get("region_bytes", [])
        size = full_bytes + meta.get("views_bytes", 0) + sum(region_bytes)
        if meta.get("encoding") == "lz4":
            import lz4.block

            if not 0 < size <= 8192 * 8192 * 4:
                raise DesktopError("Invalid decompressed frame size")
            payload = lz4.block.decompress(payload, uncompressed_size=size)
        if len(payload) != size:
            raise DesktopError("Capture payload does not match its declared layout")
        if meta["overflow"] or meta["stopped"]:
            raise DesktopError("Input queue overflow or F12 emergency stop")
        buffer = np.frombuffer(payload, np.uint8)
        offset = 0
        rgb = None
        if full_bytes:
            rgb = (
                buffer[:full_bytes]
                .reshape(meta["height"], meta["width"], 4)[:, :, [2, 1, 0]]
                .copy()
            )
            offset = full_bytes
        seen = None
        if meta.get("views_bytes"):
            sizes = [meta.get(k) for k in ("view_size", "detail_size", "fovea_size")]
            if not all(isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in sizes):
                raise DesktopError(
                    "Worker view payload has no view, detail and fovea sizes. Rebuild and "
                    "redeploy hoi4-desktop-worker."
                )
            s, d, f = sizes
            if (d, f) != (options["detail"], options["fovea"]):
                raise DesktopError("Worker returned views at sizes other than requested")
            parts = [s * s * 3, QUADRANTS * d * d * 3, f * f * 3]
            if meta["views_bytes"] != sum(parts):
                raise DesktopError(
                    "Worker view payload is not the global frame, four quadrants, and "
                    "fovea. Rebuild and redeploy hoi4-desktop-worker."
                )
            # The worker already emits RGB at policy resolution; no swizzle needed.
            block = buffer[offset : offset + meta["views_bytes"]]
            ends = np.cumsum(parts)
            seen = Views(
                block[: ends[0]].reshape(s, s, 3).copy(),
                block[ends[0] : ends[1]].reshape(QUADRANTS, d, d, 3).copy(),
                block[ends[1] :].reshape(f, f, 3).copy(),
            )
            offset += meta["views_bytes"]
        crops = None
        if region_bytes:
            crops = []
            for (x, y, w, h), n in zip(options["regions"], region_bytes, strict=True):
                crops.append(buffer[offset : offset + n].reshape(h, w, 4)[:, :, [2, 1, 0]].copy())
                offset += n
        try:
            parse_cursor(meta.get("cursor"))
        except ValueError as error:
            raise DesktopError(str(error)) from error
        return Frame(rgb, meta, time.monotonic_ns(), views=seen, crops=crops)

    def arm(self, *, setup=False):
        self.request("arm", mode="setup" if setup else "match")

    def apply(self, events: list[dict]):
        # Short on purpose. A slot that blocks longer than this has lost the worker,
        # and the dispatch join is waiting to stop the interval.
        reply = self.request("apply", timeout=2, events=events)
        reply.pop("payload", None)
        return reply

    def release(self):
        self.request("release")

    def focus(self):
        """Bring the game window to the front, for setup only. True if it is now in front.

        A windowed game started on the second PC does not take focus by itself, and the
        worker refuses input and capture until it has it. Refused while armed.
        """
        return bool(self.request("focus")["foreground"])

    def game_log(self, offset=0):
        """The arena mod's new game.log lines after `offset`, and the offset to pass next.

        The mod logs each surrender, peace deal and state changing hands, and a weekly
        count per country, so a match's outcome needs no pixels. Works the same on the
        second PC, through its worker.
        """
        reply = self.request("game_log", offset=int(offset))
        return reply["lines"], reply["offset"]

    def _control(self, op: str, timeout: float, **kwargs) -> str:
        """Run one of the worker's fixed Game-Control.ps1 actions and return its output.

        The worker refuses these while input is armed, and runs one at a time. A refusal
        or a failed script is a nonzero exit, raised here with the script's own words.
        """
        reply = self.request(op, timeout=timeout, **kwargs)
        output = reply.get("output", "")
        if reply.get("exit") != 0:
            raise DesktopError(f"{op} exited {reply.get('exit')}: {output}")
        return output

    def launch(self, mod: str, window: str | None = "1920x1080", timeout: float = 600) -> str:
        """Start HOI4 with the arena mod in that folder of the worker's mods directory.

        `mod` is a folder name, not a path. `window` is the client size of a windowed
        game, or None for the player's own display mode. Refused if HOI4 is running. A
        game that never logged far enough to load is not a failure here: the output says
        "Timed out waiting for the game log", and the caller decides what to do.
        """
        return self._control("launch", timeout, mod=mod, window=window)

    def quit(self, timeout: float = 120) -> str:
        """Close HOI4, politely first, and return once it is gone."""
        return self._control("quit", timeout)

    def report(self, timeout: float = 60) -> str:
        """The game's processes, the visible windows and the ends of the game and Steam logs."""
        return self._control("report", timeout)

    def restart_discord(self, timeout: float = 60) -> str:
        """Restart Discord, whose overlay once hung every launch after a force-closed game."""
        return self._control("restart_discord", timeout)

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
