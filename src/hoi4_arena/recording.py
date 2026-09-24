from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .arena_log import ArenaLog
from .desktop import Desktop, DesktopError, EmergencyStop
from .layout import FOVEA_SIZE, parse_cursor, recorded_speed

log = logging.getLogger(__name__)


def audit_pixels(frame):
    """What to record for this frame.

    A worker that downscaled on the capture side never sent the native frame, so the
    audit video is the global view: the pixels the policy actually saw. The manifest
    says which, because a 224x224 video is not a substitute for a native recording
    when a human needs to review a match.
    """
    if frame.rgb is not None:
        return frame.rgb
    if frame.views is not None:
        return frame.views[0]
    raise ValueError("Capture carries no pixels to record")


def split_for_session(session_id: str):
    bucket = int(hashlib.sha256(session_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


# Encoder arguments by name. ffv1 is lossless. x264 at CRF 18 in full-resolution colour
# (4:4:4, so thin coloured text keeps its edges) measured about 48 dB PSNR against ffv1
# on a 1080p arena clip, and about a hundredth of the size. Four threads encode 1080p at
# about 24 fps, several times the 5 a recording needs; x264's own default on this 28-thread
# CPU took 79 threads and 2 GB of memory per recording (2026-09-24).
CODECS = {
    "ffv1": [
        "-c:v", "ffv1", "-level", "3",
        # Sliced so the encode spreads over cores. With four threads and no slices, a
        # busy 4K map encoded at about 16 fps offline and could not hold 5 Hz beside a
        # running game; sixteen slices measured about 70 fps.
        "-slices", "16", "-threads", str(min(16, os.cpu_count() or 4)),
    ],
    "x264": [
        "-c:v", "libx264", "-preset", "faster", "-crf", "18", "-pix_fmt", "yuv444p",
        "-threads", "4",
    ],
}  # fmt: skip


class Recorder:
    """Native RGB frames; explicit frame index -> capture time, not nominal FPS."""

    # Frames are captured and encoded here, one request per frame (StreamRecorder: by the
    # worker).
    streamed = False

    def __init__(
        self,
        root,
        first,
        *,
        game_speed,
        source="human",
        hz=15,
        session_id=None,
        split=None,
        codec="ffv1",
    ):
        if codec not in CODECS:
            raise ValueError(f"codec must be one of {sorted(CODECS)}")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.id = session_id or str(uuid.uuid4())
        pixels = audit_pixels(first)
        self.manifest = {
            "schema": 1,
            "session_id": self.id,
            "split": split or split_for_session(self.id),
            "source": source,
            "width": pixels.shape[1],
            "height": pixels.shape[0],
            "video_source": "full_frame" if first.rgb is not None else "global_view",
            "nominal_fps": hz,
            "codec": codec,
            "fovea": FOVEA_SIZE,
            # Whether the worker drew the pointer into the frames, as a player sees it.
            # Recordings from before it did have no pointer in the pixels.
            "pointer_drawn": bool(first.meta.get("pointer_drawn")),
            # The operator sets this for the whole session. The match loop leaves it alone.
            **recorded_speed(game_speed),
            "complete": False,
            "frames": 0,
            "privileged_state": False,
        }
        self.events = (self.root / "frames.jsonl").open("w", encoding="utf8")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required for recordings")
        self.log = (self.root / "ffmpeg.log").open("wb")
        self.encoder = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.manifest['width']}x{self.manifest['height']}",
                "-framerate",
                str(hz),
                "-i",
                "pipe:0",
                "-an",
                *CODECS[codec],
                str(self.root / "screen.mkv"),
            ],
            stdin=subprocess.PIPE,
            stderr=self.log,
        )
        self._manifest()

    def _manifest(self):
        temp = self.root / "manifest.tmp"
        temp.write_text(json.dumps(self.manifest, indent=2), encoding="utf8")
        temp.replace(self.root / "manifest.json")

    def append(self, frame, **extra):
        pixels = audit_pixels(frame)
        # The full frame can grow the crop later. A frame that never recorded where the
        # pointer was cannot, and finding that out after the session is the failure.
        parse_cursor(frame.meta.get("cursor"))
        if pixels.shape != (self.manifest["height"], self.manifest["width"], 3):
            raise ValueError("Resolution changed during recording")
        if frame.meta.get("overflow") or not frame.meta.get("foreground"):
            raise ValueError("Invalid capture")
        self.encoder.stdin.write(pixels.tobytes())
        row = {
            "index": self.manifest["frames"],
            **frame.meta,
            "received_ns": frame.received_ns,
            **extra,
        }
        self.events.write(json.dumps(row) + "\n")
        self.events.flush()
        self.manifest["frames"] += 1

    def close(self, *, complete=True, reason=None, trailing_events=None):
        if trailing_events:
            (self.root / "trailing-events.json").write_text(json.dumps(trailing_events))
        self.encoder.stdin.close()
        rc = self.encoder.wait(timeout=60)
        self.events.close()
        self.log.close()
        self.manifest.update(complete=complete and rc == 0, reason=reason, encoder_exit=rc)
        self._manifest()


# Codecs encoded on the PC that captures, by the worker's ffmpeg (a stream), with the
# worker's encoder profile and quality for each. "nvenc" is H.264 in full-resolution colour
# (High 4:4:4) on that PC's NVIDIA video encoder at QP 14, where its frames keep what the
# policy reads better than x264 at CRF 18 did, at about the same size (STATUS.md).
STREAM_CODECS = {
    "nvenc": ("h264_nvenc", 14),
    "nvenc-hevc": ("hevc_nvenc", 16),
    "x264-source": ("x264", 18),
    "ffv1-source": ("ffv1", 0),
}


# How often a stream recording notes what its PC spends on it, and on what.
TELEMETRY_EVERY = 30
RECORDING_PROCESSES = {"hoi4.exe", "hoi4-desktop-worker.exe", "ffmpeg.exe", "pwsh.exe", "dwm.exe"}


class StreamUnavailable(RuntimeError):
    """The worker cannot record a stream: too old, no ffmpeg there, or no such encoder."""


def pixels(desk, frame):
    """A frame's RGB pixels, captured now if the frame came without them (a stream's)."""
    return frame.rgb if frame.rgb is not None else desk.capture(full=True).rgb


class StreamRecorder:
    """A recording the worker clocks and encodes on its own PC (worker protocol 2).

    The same files as Recorder: screen.mkv, frames.jsonl with a row for every frame of the
    video (its capture times, the pointer, the player's inputs), manifest.json. What moves is
    where the work happens. The worker captures on its own timer, so no frame waits for a
    network round trip or for anything else on the connection, and encodes with its
    ffmpeg; only the encoded video and the rows arrive here, and are written as they come
    (the video remuxed into Matroska, nothing decoded). A tick that finds the game out of
    focus records nothing, as the classic loop skips it.

    `append(frame, **extra)` only adds `extra` (such as a script's applied inputs) to the
    rows; the frames are the worker's. `next_frame()` waits for the worker's next frame and
    returns it without pixels (`pixels()` captures them when needed).
    """

    streamed = True

    def __init__(
        self,
        root,
        desk,
        *,
        game_speed,
        source="human",
        hz=5,
        session_id=None,
        split=None,
        codec="nvenc",
        first_data_timeout=15.0,
        views=None,
        **sizes,
    ):
        import threading

        if codec not in STREAM_CODECS:
            raise ValueError(f"a stream codec is one of {sorted(STREAM_CODECS)}")
        profile, quality = STREAM_CODECS[codec]
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required for recordings")
        self.root = Path(root)
        self.desk = desk
        self.root.mkdir(parents=True, exist_ok=False)
        try:
            self.stream = desk.start_stream(
                hz=hz, profile=profile, quality=quality, views=views, **sizes
            )
        except DesktopError as error:
            shutil.rmtree(self.root, ignore_errors=True)
            raise StreamUnavailable(str(error)) from error
        info = self.stream.info
        self.id = session_id or str(uuid.uuid4())
        self.manifest = {
            "schema": 1,
            "session_id": self.id,
            "split": split or split_for_session(self.id),
            "source": source,
            "width": info["width"],
            "height": info["height"],
            "video_source": "full_frame",
            "nominal_fps": hz,
            "codec": codec,
            # Where and how the video was encoded: by the worker, on the PC that captured.
            "encoder": {
                "where": "worker",
                "profile": info.get("profile", profile),
                "quality": info.get("quality", quality),
                "hardware": info.get("hardware"),
            },
            "clock": "worker",
            "fovea": FOVEA_SIZE,
            "pointer_drawn": True,
            **recorded_speed(game_speed),
            "complete": False,
            "frames": 0,
            "privileged_state": False,
        }
        self.events = (self.root / "frames.jsonl").open("w", encoding="utf8")
        self.log = (self.root / "ffmpeg.log").open("wb")
        # NUT from the worker, remuxed without decoding into the Matroska file training reads.
        self.muxer = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "nut", "-i", "pipe:0",
             "-c", "copy", "-f", "matroska", str(self.root / "screen.mkv")],
            stdin=subprocess.PIPE,
            stderr=self.log,
        )  # fmt: skip
        self.changed = threading.Condition()
        self.views = None  # The newest frame's views, when the stream brings them.
        self.extras = {}
        self.held = None  # The newest row: written when the next one arrives, or at close.
        self.seen = 0  # Rows next_frame has handed out.
        self.gaps = {}
        self.gap_count = self.gaps_seen = 0
        self.last_gap = None
        self.data_bytes = 0
        self.first_data = threading.Event()
        self.end = None
        self.write_error = None
        self.consumer = threading.Thread(target=self._consume, daemon=True)
        self.consumer.start()
        # What the recording PC spent on it: the game, the worker and its encoder, the GPU,
        # the network. A few lines a minute, beside the video (TELEMETRY_EVERY).
        self.stopping = threading.Event()
        self.sampler = threading.Thread(target=self._sample, daemon=True)
        # An encoder that cannot start (no NVIDIA encoder on that PC, say) fails on its
        # first frame, after the stream has started: wait for its first bytes. A game out
        # of focus sends gaps instead of frames, which is the recorder's to fix, not a
        # broken encoder.
        self.first_data.wait(first_data_timeout)
        if not self.data_bytes and (self.end is not None or not self.gap_count):
            detail = self.end or "no video within the timeout"
            try:
                self.stream.stop(timeout=60)
            except DesktopError:
                pass
            self._shut()
            shutil.rmtree(self.root, ignore_errors=True)
            raise StreamUnavailable(f"the worker's {profile} encoder gave no video: {detail}")
        self.sampler.start()
        self._manifest()

    def _sample(self):
        with (self.root / "telemetry.jsonl").open("a", encoding="utf8") as out:
            while not self.stopping.wait(TELEMETRY_EVERY):
                try:
                    reply = self.desk.telemetry(timeout=15)
                except Exception as error:  # noqa: BLE001 - telemetry must never end a recording.
                    log.warning("telemetry: %s", error)
                    continue
                row = {
                    "frame": self.manifest["frames"],
                    **{k: reply.get(k) for k in ("t_ns", "cpu", "memory", "gpu", "network")},
                    "processes": [
                        p for p in reply.get("processes") or []
                        if p.get("name", "").lower() in RECORDING_PROCESSES
                    ],
                    "game": reply.get("game"),
                    "stream": (reply.get("capture") or {}).get("stream"),
                }  # fmt: skip
                out.write(json.dumps(row) + "\n")
                out.flush()

    def _manifest(self):
        temp = self.root / "manifest.tmp"
        temp.write_text(json.dumps(self.manifest, indent=2), encoding="utf8")
        temp.replace(self.root / "manifest.json")

    def _consume(self):
        """Write the stream's messages as they arrive, on their own thread."""
        while True:
            message = self.stream.messages.get()
            if "frame" in message:
                row = {**message["frame"], "received_ns": time.monotonic_ns()}
                try:
                    seen = self.stream.frame_views(message)
                except DesktopError as error:
                    self.write_error = self.write_error or f"views: {error}"
                    seen = None
                # The views travel with the frame to next_frame(), not into the rows.
                for key in ("views_bytes", "view_size", "detail_size", "fovea_size", "encoding"):
                    row.pop(key, None)
                with self.changed:
                    self.views = seen
                    if self.held is not None:
                        self._write(self.held)
                    for key, value in self.extras.items():
                        row[key] = value
                    self.extras = {}
                    self.held = row
                    self.manifest["frames"] += 1
                    self.changed.notify_all()
            elif "data" in message:
                payload = message.get("payload") or b""
                if message["data"] != self.data_bytes:
                    self.write_error = self.write_error or (
                        f"video bytes out of order at {self.data_bytes}"
                    )
                try:
                    self.muxer.stdin.write(payload)
                except OSError as error:
                    self.write_error = self.write_error or f"muxer: {error}"
                self.data_bytes += len(payload)
                self.first_data.set()
            elif "gap" in message:
                reason = message["gap"].get("reason", "unknown")
                with self.changed:
                    self.gaps[reason] = self.gaps.get(reason, 0) + 1
                    self.gap_count += 1
                    self.last_gap = message["gap"]
                    self.changed.notify_all()
            elif "end" in message:
                with self.changed:
                    self.end = message["end"]
                    self.changed.notify_all()
                self.first_data.set()
                return

    def _write(self, row):
        self.events.write(json.dumps(row) + "\n")
        self.events.flush()

    def append(self, frame=None, **extra):
        """Add `extra` to the rows (lists are extended): a script's inputs, for one. The frame
        itself is the worker's to record, so it is not written again."""
        with self.changed:
            target = self.held if self.held is not None else self.extras
            for key, value in extra.items():
                if isinstance(value, list) and isinstance(target.get(key), list):
                    target[key] = target[key] + value
                elif isinstance(value, list):
                    target[key] = list(value)
                else:
                    target[key] = value

    def next_frame(self, timeout=10.0):
        """The worker's next frame, without pixels, once it has arrived.

        A tick that found the game out of focus comes back as a frame whose meta says
        `foreground: False`, so a loop can bring the game back as it does for a capture.
        """
        from .desktop import Frame

        deadline = time.monotonic() + timeout
        with self.changed:
            while (
                self.seen == self.manifest["frames"]
                and self.gaps_seen == self.gap_count
                and self.end is None
            ):
                left = deadline - time.monotonic()
                if left <= 0:
                    raise DesktopError(f"the worker's stream sent nothing for {timeout:g} s")
                self.changed.wait(left)
            if self.seen < self.manifest["frames"]:
                self.seen = self.manifest["frames"]
                self.gaps_seen = self.gap_count
                return Frame(None, dict(self.held), self.held["received_ns"], views=self.views)
            if self.gaps_seen < self.gap_count:
                self.gaps_seen = self.gap_count
                gap = self.last_gap or {}
                return Frame(
                    None, {"foreground": False, "gap": gap.get("reason")}, time.monotonic_ns()
                )
            raise DesktopError(f"the worker's stream ended: {self.end}")

    def _shut(self):
        try:
            self.muxer.stdin.close()
        except OSError:
            pass
        try:
            rc = self.muxer.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.muxer.kill()
            rc = self.muxer.wait()
        self.consumer.join(timeout=5)
        self.events.close()
        self.log.close()
        return rc

    def close(self, *, complete=True, reason=None, trailing_events=None):
        self.stopping.set()
        end = None
        try:
            end = self.stream.stop()
        except DesktopError as error:
            # A stream that already ended (its encoder died, say) has nothing to stop; what
            # it recorded is judged below like any other. Anything else is a real fault.
            if self.end is None:
                reason = reason or f"the stream did not stop cleanly: {error}"
                complete = False
        # Every message is queued by now; the consumer writes the last of them and exits.
        self.consumer.join(timeout=60)
        with self.changed:
            if self.held is not None:
                for key, value in self.extras.items():
                    self.held.setdefault(key, value)
                self._write(self.held)
                self.held = None
        end = end or self.end or {}
        trailing = trailing_events or (
            {"events": end.get("trailing_events", [])} if end.get("trailing_events") else None
        )
        if trailing:
            (self.root / "trailing-events.json").write_text(json.dumps(trailing))
        rc = self._shut()
        frames = self.manifest["frames"]
        video = count_frames(self.root / "screen.mkv")
        # The rows and the video must agree frame for frame, or training reads the wrong
        # frame for every decision after the first mismatch. A stream cut short (its
        # encoder or its connection lost) has rows for frames that never reached the video:
        # those rows go, and everything before them is kept, since frame i of the video is
        # row i. More video than rows cannot be matched up and is a fault.
        problem = cut = None
        if video is not None and video < frames:
            cut = frames - video
            rows = (self.root / "frames.jsonl").read_text(encoding="utf8").splitlines()
            (self.root / "frames.jsonl").write_text(
                "".join(line + "\n" for line in rows[:video]), encoding="utf8"
            )
            self.manifest["frames"] = frames = video
            log.warning("the stream was cut short: %d rows past the video's end dropped", cut)
        elif video is not None and video > frames:
            problem = f"the video holds {video} frames but only {frames} rows"
        if self.write_error:
            problem = problem or self.write_error
        if end.get("exit") not in (0, None):
            log.warning("the worker's encoder exited %s: %s", end.get("exit"), end.get("errors"))
        self.manifest.update(
            complete=complete and rc == 0 and problem is None,
            reason=reason or problem,
            rows_cut=cut,
            encoder_exit=end.get("exit"),
            muxer_exit=rc,
            video_frames=video,
            video_bytes=self.data_bytes,
            gaps=self.gaps,
            stream=end.get("stats"),
        )
        self._manifest()


def count_frames(path):
    """The number of video frames in `path` by its packets (no decoding), or None."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not Path(path).exists():
        return None
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )  # fmt: skip
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def open_recorder(desk, root, first, *, game_speed, source="human", hz=15, codec="x264", **kw):
    """A StreamRecorder for a stream codec (STREAM_CODECS) when the worker can record one,
    else a Recorder, as before. A worker that cannot (an older one, no ffmpeg or no NVIDIA
    encoder on its PC) falls back to x264 encoded here, and the manifest says which ran.
    """
    if codec in STREAM_CODECS:
        try:
            if int(desk.protocol()) >= 2:
                return StreamRecorder(root, desk, game_speed=game_speed, source=source, hz=hz,
                                      codec=codec, **kw)  # fmt: skip
            log.warning("the worker predates recording streams; encoding here with x264")
        except StreamUnavailable as error:
            log.warning("no recording stream (%s); encoding here with x264", error)
        codec = "x264"
    return Recorder(root, first, game_speed=game_speed, source=source, hz=hz, codec=codec, **kw)


# How long the game may stay out of focus before the recording ends, and how long after
# the arena log names a winner it goes on, so the end screen is in the video.
AWAY_LIMIT = 600
AFTER_SURRENDER = 15
# How often the arena log is read during a recording, in seconds.
LOG_EVERY = 5
# A recording with fewer frames than this cannot train anything (session_labels).
MIN_FRAMES = 32


def _first_frame(desktop, clock):
    """The first frame of the game in front, waiting up to AWAY_LIMIT for it to be there."""
    since = clock()
    while True:
        try:
            frame = desktop.capture()
            if frame.meta.get("foreground"):
                return frame
        except DesktopError as error:
            if "not_foreground" not in str(error) or clock() - since > AWAY_LIMIT:
                raise
        log.warning("waiting for the game to be in front")
        time.sleep(1)


def record(
    root,
    seconds,
    hz=15,
    command=None,
    split=None,
    game_speed=None,
    codec="ffv1",
    peer=None,
    clock=time.monotonic,
):
    """Record a player until `seconds` pass, they press F12 or Ctrl+C, or the arena ends.

    `game_speed` is the speed the operator set for the whole session. It is checked
    before the worker starts. With `peer`, a pairing file, the game and the player are on
    the second PC: its worker captures the screen and the player's own inputs there, both
    on its clock, and the frames come here over the network to be encoded. A full 1080p
    frame takes about 86 ms to arrive, so 5 Hz holds and 10 does not. With a stream codec
    (`nvenc`, STREAM_CODECS) the worker keeps the clock and encodes there instead, and only
    the video crosses the network.

    Nothing already recorded is lost to a mistake. While the game is out of focus (a
    click outside it, an alt-tab) or its window has another size, recording pauses and
    then resumes; the worker records no input outside the game, and training drops only
    the decisions whose frames span the pause (session_labels). A capture that falls
    behind skips ahead rather than failing. On an arena game the recording stops by
    itself AFTER_SURRENDER seconds after the log names a winner. If anything else ends it
    early, what was recorded is kept and usable, and the manifest's `ended` says why.
    Only a recording too short to train on fails.
    """
    speed = recorded_speed(game_speed)["game_speed"]
    if peer:
        from .remote import RemoteDesktop

        desktop = RemoteDesktop(peer)
    else:
        desktop = Desktop(command)
    with desktop:
        # The worker captures only a game in front. Bring it there once, before the
        # first frame: nobody may have clicked it yet, least of all on the second PC.
        desktop.focus()
        # F12 is the worker's stop key and stays latched until input is armed again, so
        # an earlier press would end this recording at once. Arming and releasing clears it.
        try:
            desktop.arm(setup=True)
            desktop.release()
        except DesktopError as error:
            log.warning("could not clear an earlier F12: %s", error)
        # The arena mod's log names who declared, which country the player took and who
        # won, as it does for the AI games (arena_log). Lines from before this recording
        # belong to earlier games of the same launch, so they are read now and dropped.
        arena = ArenaLog(desktop)
        try:
            arena.poll()
        except Exception as error:  # noqa: BLE001 - a vanilla game has no arena log.
            log.warning("no arena log: %s", error)
            arena = None
        else:
            arena.declarer, arena.players = None, []
            arena.winner = arena.loser = arena.surrendered = None
        first = _first_frame(desktop, clock)
        recorder = open_recorder(
            desktop, root, first, hz=hz, split=split, game_speed=speed, codec=codec
        )
        recorder.manifest["station"] = "peer" if peer else "here"
        size = (first.meta.get("width"), first.meta.get("height"))
        start = deadline = next_poll = clock()
        away_since = over_at = None
        pauses, skipped = 0, 0
        ended = "time"
        try:
            recorder.append(first)
            while clock() - start < seconds:
                if recorder.streamed:
                    # The worker keeps the clock and records each frame itself; a tick
                    # that found the game away comes back with foreground false.
                    frame = recorder.next_frame()
                    if frame.meta.get("stopped"):
                        ended = "F12"
                        break
                else:
                    deadline += 1 / hz
                    time.sleep(max(0, deadline - clock()))
                    if clock() - deadline > 1:
                        # Behind by over a second: skip ahead. The gap costs the decisions
                        # that span it, not the recording.
                        deadline = clock()
                        skipped += 1
                    try:
                        frame = desktop.capture()
                    except EmergencyStop:
                        ended = "F12"
                        break
                    except DesktopError as error:
                        # A screen switched off leaves the window off the desktop: wait for
                        # it as for a game out of focus.
                        if "not_foreground" not in str(error) and "off_screen" not in str(error):
                            raise
                        frame = None
                usable = (
                    frame is not None
                    and frame.meta.get("foreground")
                    and (frame.meta.get("width"), frame.meta.get("height")) == size
                )
                if not usable:
                    if away_since is None:
                        away_since = clock()
                        pauses += 1
                        log.warning("the game is not in front: recording paused")
                    elif clock() - away_since > AWAY_LIMIT:
                        ended = f"the game was out of focus for over {AWAY_LIMIT} s"
                        break
                    continue
                if away_since is not None:
                    log.warning("the game is back: recording resumed")
                    away_since = None
                recorder.append(frame)
                if arena is not None and clock() >= next_poll:
                    next_poll = clock() + LOG_EVERY
                    try:
                        arena.poll()
                    except Exception as error:  # noqa: BLE001 - keep recording regardless.
                        log.warning("arena log unavailable: %s", error)
                    if arena.winner and over_at is None:
                        over_at = clock() + AFTER_SURRENDER
                if over_at is not None and clock() >= over_at:
                    ended = "surrender"
                    break
        except KeyboardInterrupt:
            ended = "Ctrl+C"
        except Exception as error:  # noqa: BLE001 - keep what was recorded; say why it ended.
            ended = f"error: {error}"
            log.error("recording ended early, and what was recorded is kept: %s", error)
        finally:
            tail = None
            try:
                # A stream hands over its last inputs itself, when it ends.
                if not recorder.streamed:
                    tail = desktop.request("events")
                    tail.pop("payload", None)
            except Exception as error:  # noqa: BLE001
                log.warning("final input drain failed: %s", error)
            if arena is not None:
                try:
                    arena.poll()
                    recorder.manifest.update(
                        winner=arena.winner,
                        surrendered=arena.surrendered,
                        declarer=arena.declarer,
                        players=arena.players,
                    )
                except Exception as error:  # noqa: BLE001 - the video is still good.
                    log.warning("could not read the arena log: %s", error)
            try:
                lines = desktop.worker_log()
                if lines:
                    (recorder.root / "worker.log").write_text("\n".join(lines) + "\n")
            except Exception as error:  # noqa: BLE001 - diagnostics must never mask cleanup.
                log.warning("could not write worker.log: %s", error)
            recorder.manifest.update(ended=ended, focus_pauses=pauses, skipped_ahead=skipped)
            enough = recorder.manifest["frames"] >= MIN_FRAMES
            reason = None if enough else f"only {recorder.manifest['frames']} frames"
            recorder.close(complete=enough, reason=reason, trailing_events=tail)
            print(json.dumps(recorder.manifest, indent=2))
        # Recorder.close independently clears `complete` on a nonzero encoder exit without
        # setting a reason, so key the failure on the manifest rather than on `reason`.
        if not recorder.manifest["complete"]:
            detail = reason or f"encoder exit {recorder.manifest['encoder_exit']}"
            raise RuntimeError(f"Recording unusable: {detail} (ended by {ended})")
        return recorder.manifest
