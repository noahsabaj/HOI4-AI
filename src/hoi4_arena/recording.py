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
from .dataset import FOVEA_SIZE, parse_cursor, recorded_speed
from .desktop import Desktop, DesktopError, EmergencyStop

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
# on a 1080p arena clip, and about a hundredth of the size.
CODECS = {
    "ffv1": [
        "-c:v", "ffv1", "-level", "3",
        # Sliced so the encode spreads over cores. With four threads and no slices, a
        # busy 4K map encoded at about 16 fps offline and could not hold 5 Hz beside a
        # running game; sixteen slices measured about 70 fps.
        "-slices", "16", "-threads", str(min(16, os.cpu_count() or 4)),
    ],
    "x264": ["-c:v", "libx264", "-preset", "faster", "-crf", "18", "-pix_fmt", "yuv444p"],
}  # fmt: skip


class Recorder:
    """Native RGB frames; explicit frame index -> capture time, not nominal FPS."""

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
    frame takes about 86 ms to arrive, so 5 Hz holds and 10 does not.

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
        recorder = Recorder(root, first, hz=hz, split=split, game_speed=speed, codec=codec)
        recorder.manifest["station"] = "peer" if peer else "here"
        size = (first.meta.get("width"), first.meta.get("height"))
        start = deadline = next_poll = clock()
        away_since = over_at = None
        pauses, skipped = 0, 0
        ended = "time"
        try:
            recorder.append(first)
            while clock() - start < seconds:
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
                    if "not_foreground" not in str(error):
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
