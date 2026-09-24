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

from .layout import DETAIL_SIZE, FOVEA_SIZE, QUADRANTS, Views, hw, parse_cursor

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


def bgra_to_rgb(bgra):
    """A new contiguous RGB array from a BGRA one. OpenCV's conversion gives the same bytes
    as numpy's index swizzle, eight times faster: 2.2 ms against 18.4 ms at 1080p."""
    import cv2

    return cv2.cvtColor(np.ascontiguousarray(bgra), cv2.COLOR_BGRA2RGB)


class EmergencyStop(DesktopError):
    """The player pressed F12, the worker's stop key: input stops, and so does recording."""


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
    # How captures travel. A local worker's pipe moves raw frames at gigabytes a second,
    # where lz4 cost 16-21 ms a 1080p frame to compress for a 1.2-3.2x ratio on real game
    # frames; over the network (RemoteDesktop) the saving is worth it.
    encoding = "raw"

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
        # Recording streams by key: where each stream's messages go (WorkerStream).
        self.streams = {}
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
            # A recording stream's messages carry its key instead of a request's id.
            sink = (
                None if box is not None else getattr(self, "streams", {}).get(reply.get("stream"))
            )
        if box is not None:
            box.put(reply)
        elif sink is not None and req_id is None:
            sink(reply)
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
            sinks = list(getattr(self, "streams", {}).values())
        for box in boxes:
            box.put(error)
        # A stream ends with its connection: tell whoever is writing it, so it keeps what
        # it has instead of waiting for messages that cannot come.
        for sink in sinks:
            sink({"end": {"reason": f"connection lost: {error}"}})

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

        A request that cannot be sent is explained (_detail) only once both locks are
        let go: explaining asks a remote worker for its log, another request, and asking
        from under the write lock waited on that lock forever. That hung a recorder
        twice on 2026-09-24, closing a connection the second PC's bridge had dropped.
        """
        box = queue.Queue()
        refused = None
        with self.write_lock:
            if not self._alive():
                refused = "Desktop worker exited"
            else:
                req_id = self.next_id
                self.next_id += 1
                # Strict JSON: the worker rejects NaN, and its reply to an unparseable line
                # cannot carry the id this request waits on.
                message = json.dumps({"op": op, "id": req_id, **kwargs}, allow_nan=False)
                with self.pending_lock:
                    if self.reader_error is not None:
                        refused = f"Desktop reader stopped: {self.reader_error}"
                    else:
                        self.pending[req_id] = box
                if refused is None:
                    try:
                        self._send((message + "\n").encode())
                    except Exception:
                        with self.pending_lock:
                            self.pending.pop(req_id, None)
                        raise
        if refused is not None:
            raise DesktopError(self._detail(refused))
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
        (see dataset.views). View sizes are (height, width), or an int for a square; the
        worker takes them as [width, height]. `regions` is a list of [x, y, w, h] crops at native
        resolution. Asking for either keeps the full frame off the wire. The full frame
        is returned only when nothing narrower was requested, or `full=True`.
        """
        options = {}
        if views:
            options.update(view_options(views, detail, fovea))
        if regions:
            options["regions"] = [[int(v) for v in r] for r in regions]
        if full is not None:
            options["full"] = bool(full)
        meta = self.request("capture", encoding=self.encoding, **options)
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
        if meta["stopped"]:
            raise EmergencyStop("F12 emergency stop")
        if meta["overflow"]:
            raise DesktopError("Input queue overflow")
        buffer = np.frombuffer(payload, np.uint8)
        offset = 0
        rgb = None
        if full_bytes:
            rgb = bgra_to_rgb(buffer[:full_bytes].reshape(meta["height"], meta["width"], 4))
            offset = full_bytes
        seen = None
        if meta.get("views_bytes"):
            wanted = (options["views"], options["detail"], options["fovea"])
            seen = parse_views(meta, buffer[offset : offset + meta["views_bytes"]], wanted)
            offset += meta["views_bytes"]
        crops = None
        if region_bytes:
            crops = []
            for (x, y, w, h), n in zip(options["regions"], region_bytes, strict=True):
                crops.append(bgra_to_rgb(buffer[offset : offset + n].reshape(h, w, 4)))
                offset += n
        try:
            parse_cursor(meta.get("cursor"))
        except ValueError as error:
            raise DesktopError(str(error)) from error
        # perf_counter, not monotonic: on Windows before Python 3.13 monotonic ticks in
        # steps of 15.6 ms, coarser than a frame's journey here.
        return Frame(rgb, meta, time.perf_counter_ns(), views=seen, crops=crops)

    def arm(self, *, setup=False):
        self.request("arm", mode="setup" if setup else "match")

    def apply(self, events: list[dict], at_ms=None):
        """Give input: `events` now, or each at its offset in `at_ms` (milliseconds from when
        the worker takes the request, at most 1000), applied on the worker's own clock.

        A timed batch replaces one request per 25 ms slot with one per decision: no slot
        waits on the network. Workers before protocol 2 would apply a timed batch at once,
        so it is refused for them here.
        """
        kwargs = {}
        # Short on purpose. A slot that blocks longer than this has lost the worker,
        # and the dispatch join is waiting to stop the interval.
        timeout = 2
        if at_ms is not None:
            if getattr(self, "_protocol", None) is None:
                self._protocol = self.protocol()
            if self._protocol < 2:
                raise DesktopError("this worker applies every event at once; redeploy it")
            kwargs["at_ms"] = [float(t) for t in at_ms]
            timeout += max(kwargs["at_ms"], default=0) / 1000
        reply = self.request("apply", timeout=timeout, events=events, **kwargs)
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

    def clock_offset(self, tries=3):
        """(offset, round trip) in ns: add the offset to a worker time (`t_ns`) to get
        this process's time.perf_counter_ns() of the same moment, to within half the round
        trip. The fastest of a few status requests is used."""
        best = None
        for _ in range(tries):
            before = time.perf_counter_ns()
            worker = int(self.request("status")["t_ns"])
            after = time.perf_counter_ns()
            if best is None or after - before < best[1]:
                best = ((before + after) // 2 - worker, after - before)
        return best

    def protocol(self) -> int:
        """What the worker speaks: 1, or 2 with telemetry, observers and streams."""
        return int(self.request("status").get("protocol", 1))

    def telemetry(self, timeout: float = 15) -> dict:
        """What the worker's PC is doing: CPU, memory, GPU, disks, network, per process,
        and the game's window and capture timing (protocol 2).

        The first request starts the worker's sampler and waits about a second for it.
        """
        reply = self.request("telemetry", timeout=timeout)
        reply.pop("payload", None)
        return reply

    def start_stream(self, hz=5, profile="h264_nvenc", quality=None, views=None, **sizes):
        """Start a recording stream the worker clocks and encodes (protocol 2).

        With `views` (and optionally `detail` and `fovea`, as for capture), every frame also
        brings the policy's views of itself: a live policy acts on exactly the frames the
        recording holds, without asking for a capture each tick. Not `stream`:
        RemoteDesktop's socket file is its `stream`.
        """
        return WorkerStream(
            self, hz=hz, profile=profile, quality=quality,
            views=view_options(views, **sizes) if views else None,
        )  # fmt: skip

    def game_log(self, offset=0):
        """The arena mod's new game.log lines after `offset`, and the offset to pass next.

        The mod logs each surrender, peace deal and state changing hands, and a weekly
        count per country, so a match's outcome needs no pixels. Works the same on the
        second PC, through its worker.
        """
        reply = self.request("game_log", offset=int(offset))
        return reply["lines"], reply["offset"]

    def pointer(self):
        """The pointer's current image as RGBA, and its hotspot as (x, y).

        The worker draws this image into every frame it captures; saved, it is the
        template `video_import` looks for in video that never recorded the pointer.
        """
        reply = self.request("pointer")
        w, h = reply["width"], reply["height"]
        bgra = np.frombuffer(bytes(reply["payload"]), np.uint8).reshape(h, w, 4)
        return bgra[:, :, [2, 1, 0, 3]].copy(), tuple(reply["hotspot"])

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

    def launch(
        self,
        mod: str,
        window: str | None = "1920x1080",
        timeout: float = 600,
        save: str | None = None,
    ) -> str:
        """Start HOI4 with the arena mod in that folder of the worker's mods directory.

        `mod` is a folder name, not a path. `window` is the client size of a windowed
        game, or None for the player's own display mode. `save`, a save game's name
        (letters, digits and _), loads it straight away, skipping the main menu: a match
        can start mid-game. Refused if HOI4 is running. A game that never logged far
        enough to load is not a failure here: the output says "Timed out waiting for the
        game log", and the caller decides what to do.
        """
        return self._control("launch", timeout, mod=mod, window=window, save=save)

    def saves(self, timeout: float = 60) -> str:
        """The save games on the worker's PC, newest first, to choose a mid-game start from."""
        return self._control("saves", timeout)

    def job(
        self,
        action: str,
        job_id: str | None = None,
        kind: str | None = None,
        args: list[str] | None = None,
        timeout: float = 60,
    ) -> str:
        """Start, stop or list compute jobs on the worker's PC (scripts/Run-Job.ps1).

        A job runs detached and hidden there: the Python environment's setup, one of a
        fixed set of hoi4-arena commands, or a study script. Its output and state are in
        the jobs folder beside the worker, which the share makes readable from here.
        """
        # "job", not "id": every request's id routes its reply back to the caller.
        return self._control("job", timeout, action=action, job=job_id, kind=kind, args=args)

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


class WorkerStream:
    """A recording the worker clocks and encodes (protocol 2).

    The worker captures the game `hz` times a second on its own timer, draws the pointer,
    and encodes the frames with ffmpeg on its own PC (`profile` and `quality`, from its
    fixed list). What arrives here, in order, on `messages`:
    `{"frame": {...}}` for each frame the video holds (its times, pointer and inputs, as a
    capture reply has them), `{"data": offset, "payload": bytes}` for the encoded video (NUT),
    `{"gap": {"reason"}}` for a tick that recorded nothing (the game not in front, say), and
    `{"end": {...}}` once. A frame's pixels never cross the network.
    """

    def __init__(self, desk, *, hz, profile, quality=None, views=None):
        import uuid

        self.desk = desk
        self.key = uuid.uuid4().hex[:16]
        self.messages = queue.Queue()
        self.ended = None
        self.views = views
        with desk.pending_lock:
            desk.streams[self.key] = self.messages.put
        try:
            self.info = desk.request(
                "stream", timeout=20, action="start", key=self.key, hz=int(hz), profile=profile,
                quality=quality, encoding=desk.encoding, **(views or {}),
            )  # fmt: skip
        except Exception:
            with desk.pending_lock:
                desk.streams.pop(self.key, None)
            raise
        self.info.pop("payload", None)

    def frame_views(self, message):
        """The Views a frame message brings, or None."""
        meta = message["frame"]
        if not self.views or not meta.get("views_bytes"):
            return None
        block = bytes(message.get("payload") or b"")
        if meta.get("encoding") == "lz4":
            import lz4.block

            try:
                block = lz4.block.decompress(block, uncompressed_size=meta["views_bytes"])
            except Exception as error:  # noqa: BLE001 - reported as the worker's fault.
                raise DesktopError(f"a stream frame's views are corrupt: {error}") from error
        wanted = (self.views["views"], self.views["detail"], self.views["fovea"])
        return parse_views(meta, block, wanted)

    def stop(self, timeout=90):
        """End the stream. By the time this returns every message is on `messages`."""
        try:
            reply = self.desk.request("stream", timeout=timeout, action="stop")
            reply.pop("payload", None)
            return reply
        finally:
            with self.desk.pending_lock:
                self.desk.streams.pop(self.key, None)


def view_options(views, detail=DETAIL_SIZE, fovea=FOVEA_SIZE):
    """A request's view sizes as the worker takes them: [width, height] lists, and the
    fovea's side. Sizes here are (height, width), or an int for a square."""
    return {
        "views": list(hw(views))[::-1],
        "detail": list(hw(detail))[::-1],
        "fovea": int(fovea),
    }


def parse_views(meta, block, wanted):
    """The Views in a reply's view bytes (a capture's, or a stream frame's), checked
    against the sizes asked for (`wanted`: [w, h], [w, h], fovea)."""
    sizes = [meta.get(k) for k in ("view_size", "detail_size", "fovea_size")]

    def pair(v):
        ok = isinstance(v, list) and len(v) == 2
        return ok and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in v)

    f = sizes[2]
    if not (pair(sizes[0]) and pair(sizes[1]) and isinstance(f, int) and not isinstance(f, bool)):
        raise DesktopError(
            "Worker view payload has no [width, height] view and detail sizes. "
            "Rebuild and redeploy hoi4-desktop-worker."
        )
    (sw, sh), (dw, dh) = sizes[0], sizes[1]
    if ([sw, sh], [dw, dh], f) != tuple(wanted) or f <= 0:
        raise DesktopError("Worker returned views at sizes other than requested")
    parts = [sh * sw * 3, QUADRANTS * dh * dw * 3, f * f * 3]
    if meta["views_bytes"] != sum(parts) or len(block) != sum(parts):
        raise DesktopError(
            "Worker view payload is not the global frame, four quadrants, and "
            "fovea. Rebuild and redeploy hoi4-desktop-worker."
        )
    # The worker already emits RGB at policy resolution; no swizzle needed.
    block = np.frombuffer(block, np.uint8)
    ends = np.cumsum(parts)
    return Views(
        block[: ends[0]].reshape(sh, sw, 3).copy(),
        block[ends[0] : ends[1]].reshape(QUADRANTS, dh, dw, 3).copy(),
        block[ends[1] :].reshape(f, f, 3).copy(),
    )


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
