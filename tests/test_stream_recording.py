"""Recordings the worker clocks and encodes (StreamRecorder): the same files as before."""

import json
import os
import queue
import subprocess
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from test_dataset import needs_ffmpeg

from hoi4_arena import recording
from hoi4_arena.desktop import Desktop, DesktopError

W = H = 16


def _nut(frames):
    """A real encoded stream of `frames` 16x16 frames, as the worker's ffmpeg writes it."""
    raw = b"".join(np.full((H, W, 4), 10 * i % 255, np.uint8).tobytes() for i in range(frames))
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "bgra", "-video_size",
         f"{W}x{H}", "-framerate", "5", "-i", "pipe:0", "-c:v", "ffv1", "-f", "nut", "pipe:1"],
        input=raw, capture_output=True, check=True,
    ).stdout  # fmt: skip


class _Stream:
    """A worker stream that has recorded `frames` frames (and a gap), ending on stop()."""

    def __init__(self, frames=40, encoded=None, data=True, fail=False):
        self.info = {"width": W, "height": H, "profile": "ffv1", "quality": 0, "hardware": False}
        self.messages = queue.Queue()
        self.encoded = frames if encoded is None else encoded
        self.frames = frames
        self.stopped = False
        if fail:
            self.messages.put(
                {"end": {"reason": "encoder_exited", "exit": 1, "errors": ["no nvenc"]}}
            )
            return
        stream = _nut(self.encoded) if data else b""
        for i in range(frames):
            self.messages.put(
                {"frame": {"index": i, "t_ns": (i + 1) * 200_000_000, "capture_start_ns":
                           (i + 1) * 200_000_000 - 5_000_000, "cursor": [4, 4], "events": [],
                           "foreground": True, "backend": "dxgi_bgra", "width": W, "height": H}}
            )  # fmt: skip
            if i == 10:
                self.messages.put({"gap": {"reason": "game_not_foreground"}})
        for offset in range(0, len(stream), 1000):
            self.messages.put({"data": offset, "payload": stream[offset : offset + 1000]})

    def frame_views(self, message):
        return None

    def stop(self, timeout=90):
        end = {
            "reason": "stopped", "frames": self.frames, "encoded_frames": self.encoded, "exit": 0,
            "trailing_events": [{"t_ns": 1, "event": {"kind": "key", "vk": 65, "down": False}}],
            "stats": {"late_ms_p95": 1.0},
        }  # fmt: skip
        self.messages.put({"end": end})
        return end


class _Desk:
    def __init__(self, stream=None, protocol=2):
        self.made = stream
        self.version = protocol

    def protocol(self):
        return self.version

    def start_stream(self, hz, profile, quality=None, views=None, **sizes):
        if isinstance(self.made, Exception):
            raise self.made
        return self.made

    def capture(self, full=True):
        return SimpleNamespace(rgb=np.zeros((H, W, 3), np.uint8), meta={}, views=None)


def _rows(root):
    return [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]


def _decoded(root):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(root / "screen.mkv"), "-f", "rawvideo", "-pix_fmt",
         "rgb24", "pipe:1"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return len(out) // (W * H * 3)


@needs_ffmpeg
def test_a_stream_recording_writes_the_files_training_reads(tmp_path):
    rec = recording.StreamRecorder(tmp_path / "rec", _Desk(_Stream()), game_speed=5, codec="nvenc")
    assert rec.streamed
    frame = rec.next_frame(timeout=5)
    assert frame.rgb is None and frame.meta["foreground"]
    rec.append(frame, scripted_events=[{"t_ns": 7, "event": {"kind": "move", "x": 0.5, "y": 0.5}}])
    rec.close()
    root = tmp_path / "rec"
    manifest = json.loads((root / "manifest.json").read_text())
    rows = _rows(root)
    assert manifest["complete"], manifest["reason"]
    assert manifest["frames"] == len(rows) == _decoded(root) == 40
    assert [r["index"] for r in rows] == list(range(40))
    assert np.all(np.diff([r["t_ns"] for r in rows]) > 0)
    assert manifest["encoder"]["where"] == "worker" and manifest["clock"] == "worker"
    assert manifest["recorder"]["pid"] == os.getpid()
    assert manifest["gaps"] == {"game_not_foreground": 1}
    assert sum(len(r.get("scripted_events", [])) for r in rows) == 1
    assert json.loads((root / "trailing-events.json").read_text())["events"][0]["t_ns"] == 1


@needs_ffmpeg
def test_a_tick_that_found_the_game_away_reads_as_out_of_focus(tmp_path):
    rec = recording.StreamRecorder(tmp_path / "rec", _Desk(_Stream()), game_speed=5, codec="nvenc")
    # All 40 frames and the gap are already queued: the first call sees frames, then the
    # gap has been counted with them. A gap alone reads as a frame out of focus.
    rec.next_frame(timeout=5)
    with rec.changed:
        rec.gap_count += 1
        rec.last_gap = {"reason": "game_not_foreground"}
        rec.changed.notify_all()
    away = rec.next_frame(timeout=5)
    assert away.meta == {"foreground": False, "gap": "game_not_foreground"}
    assert recording.pixels(_Desk(), away).shape == (H, W, 3)
    rec.close()


@needs_ffmpeg
def test_a_stream_cut_short_keeps_the_frames_its_video_holds(tmp_path):
    # Two rows arrived for frames that never reached the video (a lost connection, say):
    # frame i of the video is row i, so the rows past the video's end go, and the rest train.
    rec = recording.StreamRecorder(
        tmp_path / "rec", _Desk(_Stream(frames=40, encoded=38)), game_speed=5, codec="nvenc"
    )
    rec.close()
    root = tmp_path / "rec"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["complete"] and manifest["rows_cut"] == 2
    assert manifest["frames"] == len(_rows(root)) == _decoded(root) == 38


@needs_ffmpeg
def test_an_encoder_that_gives_no_video_falls_back_to_x264_here(tmp_path):
    first = SimpleNamespace(
        rgb=np.zeros((H, W, 3), np.uint8), views=None, meta={"pointer_drawn": True}
    )
    desk = _Desk(_Stream(fail=True))
    rec = recording.open_recorder(desk, tmp_path / "rec", first, game_speed=5, hz=5, codec="nvenc")
    assert isinstance(rec, recording.Recorder) and rec.manifest["codec"] == "x264"
    rec.close()
    # A worker from before streams, or one that refuses, records as before too.
    for desk, name in ((_Desk(protocol=1), "old"), (_Desk(DesktopError("ffmpeg_not_found")), "no")):
        rec = recording.open_recorder(desk, tmp_path / name, first, game_speed=5, codec="nvenc")
        assert not rec.streamed
        rec.close()
    # Not a stream codec: the classic recorder, without asking the worker anything.
    rec = recording.open_recorder(None, tmp_path / "x", first, game_speed=5, codec="x264")
    assert not rec.streamed
    rec.close()


def test_stream_messages_reach_their_stream_and_replies_their_request():
    desk = Desktop.__new__(Desktop)
    desk.pending_lock = threading.Lock()
    desk.diagnostics = []
    box = queue.Queue()
    desk.pending = {3: box}
    got = []
    desk.streams = {"abc": got.append}
    desk._deliver({"stream": "abc", "frame": {"index": 0}})
    # A stream's own start reply carries its key too, but it answers a request.
    desk._deliver({"id": 3, "stream": "abc", "width": 16})
    desk._deliver({"stream": "other", "data": 0})
    assert got == [{"stream": "abc", "frame": {"index": 0}}]
    assert box.get_nowait() == {"id": 3, "stream": "abc", "width": 16}


def test_the_observer_role_rides_on_the_token_line(tmp_path):
    from hoi4_arena import remote

    written = []

    class _Socket:
        def getpeercert(self, binary_form=True):
            return b"cert"

        def settimeout(self, _):
            pass

        def setsockopt(self, *_):
            pass

        def makefile(self, mode):
            return SimpleNamespace(
                write=written.append,
                flush=lambda: None,
                readline=lambda *_: b"",
                read=lambda *_: b"",
            )

        def shutdown(self, _):
            pass

        def close(self):
            pass

    class _Context:
        def __init__(self, *_):
            pass

        def wrap_socket(self, raw, server_hostname=None):
            return _Socket()

    import hashlib

    pin = hashlib.sha256(b"cert").hexdigest()
    spec = {"host": "127.0.0.1", "port": 1, "token": "t" * 64, "certificate_sha256": pin}
    peer = tmp_path / "peer.json"
    peer.write_text(json.dumps(spec))
    for observer, line in ((False, b"t" * 64 + b"\n"), (True, b"t" * 64 + b" observer\n")):
        written.clear()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(remote.socket, "create_connection", lambda *a, **k: object())
            mp.setattr(remote.ssl, "SSLContext", _Context)
            desk = remote.RemoteDesktop(peer, attach=False, observer=observer)
            desk._shutdown()
        assert written[0] == line


def test_a_stream_frame_brings_the_policy_views():
    import lz4.block

    from hoi4_arena.desktop import WorkerStream, view_options

    stream = WorkerStream.__new__(WorkerStream)
    stream.views = view_options((2, 4), detail=(2, 2), fovea=2)
    sizes = [2 * 4 * 3, 4 * 2 * 2 * 3, 2 * 2 * 3]
    raw = bytes(range(sum(sizes)))
    meta = {
        "views_bytes": len(raw), "view_size": [4, 2], "detail_size": [2, 2], "fovea_size": 2,
        "encoding": "lz4",
    }  # fmt: skip
    # The worker sends a bare lz4 block, with no size before it.
    payload = lz4.block.compress(raw, store_size=False)
    views = stream.frame_views({"frame": meta, "payload": payload})
    assert views.global_view.shape == (2, 4, 3) and views.quadrants.shape == (4, 2, 2, 3)
    assert views.fovea.shape == (2, 2, 3) and views.fovea.reshape(-1)[-1] == sum(sizes) - 1
    # A frame without views, or a stream that asked for none, brings none.
    assert stream.frame_views({"frame": {}, "payload": b""}) is None
    with pytest.raises(DesktopError, match="other than requested"):
        stream.frame_views({"frame": {**meta, "fovea_size": 3}, "payload": payload})
    with pytest.raises(DesktopError, match="corrupt"):
        stream.frame_views({"frame": meta, "payload": b"junk"})
