"""Each game's live view kept, to watch again as it looked live.

The recordings are 5 frames a second, as the policy sees the game; the live view is 60.
While a PC streams, its view's ffmpeg also writes the stream as it comes, without encoding
it again, into 30 s pieces named by the time they began (raw/<station>/). Once a game has
ended, its stretch of those pieces is made into one small file: 720p, 30 frames a second,
HEVC from NVENC (which every iPhone and iPad plays), about 2.6 MB a minute (the map is
still readable at the quality chosen, 36; a keyframe a second cost three times as much,
2026-09-25). Those are kept up to a budget, about a week of games, the oldest going first; the pieces go once no game needs them. A game with
no archive (no view then, or a PC this view did not watch) is replayed from its recording.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from .replay import SAFE

log = logging.getLogger(__name__)

# The archives kept, in bytes.
BUDGET = 20 * 1024**3
# Seconds each raw piece lasts, and how long a piece no game needs is kept.
PIECE, RAW_KEEP = 30, 15 * 60
# Games older than this are not archived any more: their pieces are gone.
TOO_OLD = 3 * 3600
STAMP = "%Y%m%d-%H%M%S"


def piece_output(folder):
    """The ffmpeg output (after `-map 0:v`) writing a stream into raw pieces in `folder`."""
    return [
        "-c:v", "copy", "-f", "segment", "-segment_time", str(PIECE),
        "-segment_format", "mpegts", "-reset_timestamps", "1", "-strftime", "1",
        str(Path(folder) / f"{STAMP}.ts"),
    ]  # fmt: skip


def archive_command(ffmpeg, pieces, start, seconds, out, encoder="hevc_nvenc"):
    """ffmpeg joining `pieces` (a concat list file), cutting `seconds` from `start` seconds
    into them, as 720p at 30 frames a second, a keyframe every 5 s, the index at the front,
    and its progress on stdout."""
    quality = ["-preset", "p5", "-cq", "36", "-tag:v", "hvc1"]
    if encoder != "hevc_nvenc":
        quality = ["-crf", "30", "-tag:v", "hvc1"] if encoder == "libx265" else ["-crf", "28"]
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "4",
        "-f", "concat", "-safe", "0", "-i", str(pieces),
        "-ss", f"{max(0.0, start):.2f}", "-t", f"{seconds:.2f}",
        "-vf", "fps=30,scale=-2:720,format=yuv420p", "-c:v", encoder, *quality, "-g", "150",
        "-an", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats",
        "-f", "mp4", str(out),
    ]  # fmt: skip


def pieces(folder):
    """The raw pieces in `folder`, as (began unix, path), oldest first."""
    found = []
    for path in Path(folder).glob("*.ts"):
        try:
            found.append((datetime.strptime(path.stem, STAMP).timestamp(), path))
        except ValueError:
            continue
    return sorted(found)


def covering(found, start, end, now):
    """The pieces covering `start` to `end`, or None when they do not yet (or no longer):
    the first begins at most a piece before the start, and one after them began past the
    end, or the last has stopped growing past it."""
    if not found or found[0][0] > start + 5:
        return None
    chosen = [p for p in found if p[0] < end and (p[0] + PIECE + 5 > start)]
    later = [p for p in found if p[0] >= end]
    if not chosen:
        return None
    if not later:
        try:
            written = chosen[-1][1].stat().st_mtime
        except OSError:
            return None
        if written < end + 2 or now - written < 10:
            return None  # Still being written, or it stopped before the game did.
    # Keep only a run without a gap: a view that stopped mid-game leaves a hole.
    for (a, _), (b, _) in zip(chosen, chosen[1:]):
        if b - a > PIECE + 5:
            return None
    return chosen


class Archive:
    """Games' live views, joined and made small once each game ends (see the module).
    `step(played)` each round; `path(name)` is a game's archive if it has one."""

    def __init__(self, raw, folder, ffmpeg, budget=BUDGET, encoder="hevc_nvenc"):
        self.raw, self.folder, self.ffmpeg = Path(raw), Path(folder), ffmpeg
        self.budget, self.encoder = budget, encoder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.failed = set()
        self.windows = {}  # game name: (start, end), from its manifest, once it has ended
        self.job = None  # The game being made, while one is.

    def pieces_for(self, station):
        folder = self.raw / station
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def path(self, name):
        path = self.folder / f"{name}.mp4"
        return path if SAFE.match(name) and path.exists() else None

    def step(self, played, live_since=(), now=None):
        """Start making the newest game not archived yet whose pieces are all there, and
        drop pieces nobody needs: those before every live game's start and every game
        still to be archived, less RAW_KEEP."""
        now = now or time.time()
        wanted = []
        for game in played:
            name = game["game"]
            if now - (game.get("ended_unix") or 0) > TOO_OLD + 3600:
                continue  # Roughly (its folder is named before the game loads): long gone.
            if name not in self.windows:
                self.windows[name] = self.window(game)
            window = self.windows[name]
            if window is None or now - window[1] > TOO_OLD:
                continue
            if name in self.failed or (self.folder / f"{name}.mp4").exists():
                continue
            wanted.append((game, window))
        if self.job is None:
            for game, (start, end) in wanted:
                found = covering(pieces(self.raw / game["station"]), start, end, now)
                if found:
                    self.job = game["game"]
                    threading.Thread(
                        target=self.make, args=(game["game"], found, start, end), daemon=True
                    ).start()
                    break
        keep_from = min([now, *live_since, *(w[0] for _, w in wanted)]) - RAW_KEEP
        for station in self.raw.glob("*"):
            for began, path in pieces(station):
                if began + PIECE < keep_from:
                    path.unlink(missing_ok=True)

    @staticmethod
    def window(game):
        """A played game's start and end in unix seconds, from its recording: the recorder
        began with its first frame, and the frames count its length."""
        from .media import read_json

        manifest = read_json(Path(game.get("path") or "") / "manifest.json") or {}
        started = (manifest.get("recorder") or {}).get("started_unix")
        frames, hz = manifest.get("frames"), manifest.get("nominal_fps") or 5
        if not started or not frames:
            return None
        return started, started + frames / hz

    def make(self, name, found, start, end):
        partial = self.folder / f"{name}.part.mp4"
        listing = self.folder / f"{name}.txt"
        began = time.monotonic()
        try:
            listing.write_text(
                "".join(f"file '{p.resolve().as_posix()}'\n" for _, p in found), encoding="utf-8"
            )
            command = archive_command(
                self.ffmpeg, listing, start - found[0][0], end - start, partial, self.encoder
            )
            done = subprocess.run(command, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(done.stderr.strip().splitlines()[-1] if done.stderr else "")
            partial.replace(self.folder / f"{name}.mp4")
            size = (self.folder / f"{name}.mp4").stat().st_size
            log.info("archived %s in %.0f s (%.0f MB)", name, time.monotonic() - began, size / 1e6)
            self.trim()
        except Exception as error:  # noqa: BLE001 - replayed from its recording instead.
            self.failed.add(name)
            partial.unlink(missing_ok=True)
            log.warning("archiving %s failed: %s", name, error)
        finally:
            listing.unlink(missing_ok=True)
            self.job = None

    def trim(self):
        """Keep the newest archives, up to the budget."""
        kept = sorted(self.folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        total = 0
        for path in kept:
            if path.name.endswith(".part.mp4"):
                continue
            total += path.stat().st_size
            if total > self.budget:
                path.unlink(missing_ok=True)
