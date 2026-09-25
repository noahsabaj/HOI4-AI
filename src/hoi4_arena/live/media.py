"""The video side of the live view: ffmpeg commands, and one stream per PC.

Each PC with a game gets its own HLS stream in its own folder (`out/<station>/`):
live.m3u8, its segments, and latest.jpg, a snapshot each second.

- The second PC ("peer"): its worker's `view` captures the game window at 30 frames a
  second on that PC's GPU and sends H.264 over a read-only connection (PeerView). It shows
  the menus and loading between games too.
- This PC ("here"): ffmpeg captures the HOI4 window here the same way (LocalView), only
  while record-ai plays a game on this PC: otherwise the screen is the user's.
- Without either, the recording being written is followed at its 5 frames a second
  (Follower).

A stream that stops takes its playlist and segments with it (clear_stream), so a page
can never keep playing the last seconds of an old game as if they were live; its last
snapshot stays, for the page to show as what the screen showed last.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Seconds of video before the live edge that a view joining mid-game starts from.
LEAD = 4
# Seconds the video may stop growing before its game counts as over.
STALL = 15
# Seconds a segment lasts, and how many the playlist keeps.
SEGMENT, SEGMENTS = 2, 8
# The recordings' frames a second.
FPS = 5
FLAGS = "append_list+delete_segments+discont_start+omit_endlist+independent_segments"


def read_shared(path):
    """A file's bytes, read without stopping ffmpeg from replacing or deleting it: on
    Windows a file open for reading blocks both unless opened with every sharing flag,
    and ffmpeg renames each playlist over the last and deletes old segments."""
    if sys.platform != "win32":
        return Path(path).read_bytes()
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]  # fmt: skip
    generic_read, share_all, open_existing = 0x80000000, 0x7, 3
    handle = kernel.CreateFileW(str(path), generic_read, share_all, None, open_existing, 0, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise FileNotFoundError(path)
    with os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY), "rb") as file:
        return file.read()


def read_json(path):
    """A JSON file another process rewrites, or None if it is missing or half written."""
    try:
        return json.loads(read_shared(path))
    except (OSError, ValueError):
        return None


def hls_command(ffmpeg, video, out, start=0.0, number=0, stall=STALL):
    """ffmpeg following `video` as it grows, from `start` seconds in: an HLS playlist
    whose segments are numbered from `number`, after a discontinuity (the game before
    had its own clock), and latest.jpg once a second. Every output is overwritten (-y):
    without it the second game's ffmpeg refused the first one's latest.jpg and the stream
    stopped. Segments stay on disk a while after leaving the playlist, for a player that
    lags."""
    out = Path(out)
    graph = (
        f"[0:v]select='gte(t\\,{start:.2f})',setpts=PTS-STARTPTS,split=2[v][s];"
        "[v]format=yuv420p[hls];[s]fps=1[jpg]"
    )
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-follow", "1", "-rw_timeout", str(stall * 1_000_000), "-i", str(video),
        "-filter_complex", graph,
        "-map", "[hls]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-maxrate", "3M", "-bufsize", "6M", "-fps_mode", "cfr", "-r", str(FPS),
        "-g", str(SEGMENT * FPS), "-keyint_min", str(SEGMENT * FPS), "-sc_threshold", "0",
        "-f", "hls", "-hls_time", str(SEGMENT), "-hls_list_size", str(SEGMENTS),
        "-hls_delete_threshold", "5", "-hls_flags", FLAGS, "-start_number", str(number),
        "-hls_segment_filename", str(out / "seg%06d.ts"), str(out / "live.m3u8"),
        "-map", "[jpg]", "-q:v", "4", "-update", "1", str(out / "latest.jpg"),
    ]  # fmt: skip


def view_command(ffmpeg, out, number=0):
    """ffmpeg cutting a live view (MPEG-TS on its stdin, keyframes every 2 s) into the
    playlist as it comes, without encoding it again, after a discontinuity, with
    latest.jpg once a second.

    One thread each decodes, filters and encodes the snapshot. Left to choose, ffmpeg
    sized every pool to this PC's 28 threads (84 threads in all) and held ~900 MB a game
    for work one thread does at 13% of a core; capped, the peak was 83 MB (2026-09-24).
    """
    out = Path(out)
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-threads", "1", "-f", "mpegts", "-i", "pipe:0",
        "-map", "0:v", "-c:v", "copy",
        "-f", "hls", "-hls_time", str(SEGMENT), "-hls_list_size", str(SEGMENTS),
        "-hls_delete_threshold", "5", "-hls_flags", FLAGS, "-start_number", str(number),
        "-hls_segment_filename", str(out / "seg%06d.ts"), str(out / "live.m3u8"),
        "-map", "0:v", "-filter_threads", "1", "-threads", "1",
        "-vf", "fps=1", "-q:v", "4", "-update", "1", str(out / "latest.jpg"),
    ]  # fmt: skip


def capture_command(ffmpeg, hz=30):
    """ffmpeg capturing this PC's HOI4 window (by its program's name) as the second PC's
    worker captures its own (encoder.rs view_arguments): on the GPU, H.264 from NVENC with
    a keyframe every 2 s, as MPEG-TS on stdout for view_command."""
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"gfxcapture=window_exe=hoi4:max_framerate={hz}:capture_cursor=1",
        "-fps_mode", "cfr", "-r", str(hz),
        "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll", "-rc", "vbr",
        "-b:v", "5M", "-maxrate", "8M", "-bufsize", "10M", "-g", str(2 * hz), "-bf", "0",
        "-f", "mpegts", "-flush_packets", "1", "pipe:1",
    ]  # fmt: skip


def next_segment(out):
    """The number after the playlist's last segment, or 0 without one."""
    try:
        text = (Path(out) / "live.m3u8").read_text()
    except OSError:
        return 0
    numbers = [int(line[3:9]) for line in text.splitlines() if line.startswith("seg")]
    return max(numbers) + 1 if numbers else 0


def clear_stream(out):
    """Take a stopped stream's playlist and segments away, so no page plays them as live.
    The last snapshot stays: what the screen showed last."""
    for old in [*Path(out).glob("seg*.ts"), Path(out) / "live.m3u8"]:
        try:
            old.unlink(missing_ok=True)
        except OSError:
            pass  # Held for a moment; the next stream overwrites it.


def update_pending(peer, share=None):
    """Whether a new worker waits on the second PC's share (Deploy-Peer stages it as
    .new beside the running one). Its bridge swaps it in only while no connection is
    open there, so a view that never closed would keep every update out."""
    try:
        share = share or f"//{json.loads(Path(peer).read_text())['host']}/HOI4Worker"
        return (Path(share) / "hoi4-desktop-worker.exe.new").exists()
    except (OSError, ValueError, KeyError):
        return False


class Follower:
    """The recording being written, followed at its 5 frames a second (hls_command): one
    ffmpeg per game, from near the live edge, restarted 10 s after a failure."""

    def __init__(self, ffmpeg, out):
        self.ffmpeg, self.out = ffmpeg, Path(out)
        self.proc, self.current, self.retry_at = None, None, 0.0

    def step(self, found):
        game = found[0] if found else None
        proc = self.proc
        if proc is not None and (proc.poll() is not None or game != self.current):
            if proc.poll() is None:
                proc.terminate()  # A new game began while the last one's file lay still.
                proc.wait(timeout=10)
            elif game == self.current:
                self.retry_at = time.monotonic() + 10  # It failed mid-game: not at once.
                log.warning("ffmpeg stopped (exit %s); again in 10 s", proc.returncode)
            self.proc = None
            if game is None:
                clear_stream(self.out)
        if game is not None and self.proc is None and time.monotonic() >= self.retry_at:
            started = (found[1].get("recorder") or {}).get("started_unix") or time.time()
            start = max(0.0, time.time() - started - LEAD)
            command = hls_command(
                self.ffmpeg, game / "screen.mkv", self.out, start, next_segment(self.out)
            )
            self.proc = subprocess.Popen(command, stdin=subprocess.DEVNULL)
            self.current = game
            log.info("following %s from %.0f s", game, start)

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None
        clear_stream(self.out)


class PeerView:
    """The second PC's own live view (its worker's `view`): the game window captured at
    `hz` frames a second on that PC's GPU, sent as MPEG-TS over a read-only connection,
    and cut into the playlist here without encoding it again (view_command). It shows
    the menus and the loading between games too, where following the recording is 5
    frames a second and stops. `refused` once a worker without views says no."""

    def __init__(self, peer, ffmpeg, out, hz=30):
        self.peer, self.ffmpeg, self.out, self.hz = peer, ffmpeg, Path(out), hz
        self.desk = self.proc = None
        self.ended = threading.Event()
        self.refused = False
        self.since = None

    def start(self):
        from ..remote import RemoteDesktop

        self.stop()
        self.ended.clear()
        command = view_command(self.ffmpeg, self.out, next_segment(self.out))
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            self.desk = RemoteDesktop(self.peer, attach=True, observer=True)
            self.desk.streams["view"] = self.deliver
            self.desk.request("view", action="start", key="view", hz=self.hz)
        except Exception as error:
            self.refused = "refused_for_observer" in str(error) or "unknown" in str(error)
            self.stop()
            raise
        self.since = time.time()

    def deliver(self, message):
        """A message of the view's stream, on the connection's reader thread."""
        if "end" in message:
            self.ended.set()
            return
        payload, proc = message.get("payload"), self.proc
        if payload and proc is not None and proc.stdin is not None:
            try:
                proc.stdin.write(payload)
            except (OSError, ValueError):
                self.ended.set()

    def running(self):
        return self.proc is not None and self.proc.poll() is None and not self.ended.is_set()

    def stop(self):
        desk, proc = self.desk, self.proc
        self.desk = self.proc = self.since = None
        if desk is not None:
            try:
                desk.request("view", action="stop", timeout=5)
            except Exception:  # noqa: BLE001 - closing the connection stops it anyway.
                pass
            try:
                desk.close()
            except Exception:  # noqa: BLE001
                pass
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - it is going anyway.
                proc.kill()
        clear_stream(self.out)


class LocalView:
    """This PC's HOI4 window, captured here (capture_command) and cut into the playlist
    (view_command): two ffmpegs joined by a pipe. Started only while record-ai plays a
    game on this PC."""

    def __init__(self, ffmpeg, out, hz=30):
        self.ffmpeg, self.out, self.hz = ffmpeg, Path(out), hz
        self.capture = self.cutter = None
        self.since = None

    def start(self):
        self.stop()
        self.capture = subprocess.Popen(
            capture_command(self.ffmpeg, self.hz), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE
        )
        self.cutter = subprocess.Popen(
            view_command(self.ffmpeg, self.out, next_segment(self.out)), stdin=self.capture.stdout
        )
        self.capture.stdout.close()  # The cutter holds it now.
        self.since = time.time()

    def running(self):
        return all(p is not None and p.poll() is None for p in (self.capture, self.cutter))

    def stop(self):
        for proc in (self.capture, self.cutter):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.capture = self.cutter = self.since = None
        clear_stream(self.out)
