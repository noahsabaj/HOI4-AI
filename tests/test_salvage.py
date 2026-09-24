"""Recordings whose recorder died before closing them: salvage makes them trainable."""

import json
import os
import subprocess
import time

import numpy as np
import pytest
from test_dataset import needs_ffmpeg

from hoi4_arena import recording

W = H = 16


def _killed(root, rows=45, video=40, torn=True, **manifest):
    """A recording as a killed recorder leaves it, long ago: `rows` rows, of which the last
    `rows - video` never reached the video, a row torn in half at the end, and a manifest
    that still says incomplete."""
    root.mkdir(parents=True)
    raw = b"".join(np.full((H, W, 3), 5 * i % 255, np.uint8).tobytes() for i in range(video))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
         f"{W}x{H}", "-framerate", "5", "-i", "pipe:0", "-c:v", "ffv1", str(root / "screen.mkv")],
        input=raw, check=True,
    )  # fmt: skip
    lines = [
        json.dumps({"index": i, "t_ns": (i + 1) * 200_000_000, "cursor": [4, 4], "events": []})
        for i in range(rows)
    ]
    text = "".join(line + "\n" for line in lines) + (f'{{"index": {rows}, "t_' if torn else "")
    (root / "frames.jsonl").write_text(text, encoding="utf8")
    meta = {"schema": 1, "source": "human", "split": "train", "width": W, "height": H,
            "nominal_fps": 5, "game_speed": 4, "complete": False, "frames": 0, **manifest}  # fmt: skip
    (root / "manifest.json").write_text(json.dumps(meta), encoding="utf8")
    _age(root)
    return root


def _age(root, seconds=3600):
    then = time.time() - seconds
    for path in root.iterdir():
        os.utime(path, (then, then))


def _files(root):
    return {p.name: p.read_bytes() for p in root.iterdir()}


@needs_ffmpeg
def test_salvage_keeps_every_frame_the_video_holds_and_undo_puts_it_back(tmp_path):
    root = _killed(tmp_path / "rec")
    before = _files(root)
    done = recording.salvage(root)
    assert done["salvaged"], done
    assert (done["rows"], done["video_frames"], done["kept"]) == (46, 40, 40)
    rows = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
    assert [r["index"] for r in rows] == list(range(40))
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["complete"] and manifest["frames"] == manifest["video_frames"] == 40
    assert manifest["salvaged"]["rows_dropped"] == 6
    assert recording.salvage(root)["why"] == "salvaged before"
    assert recording.unsalvage(root)["restored"]
    assert _files(root) == before


@needs_ffmpeg
def test_a_salvaged_recording_trains(tmp_path):
    from hoi4_arena.dataset import session_labels

    root = _killed(tmp_path / "rec", rows=60, video=58, torn=False)
    with pytest.raises(ValueError, match="complete"):
        session_labels(root, sources=("human",))
    assert recording.salvage(root)["salvaged"]
    labels = session_labels(root, sources=("human",))
    assert labels["manifest"]["frames"] == 58


@needs_ffmpeg
def test_salvage_leaves_alone_what_may_still_be_recording_or_was_closed(tmp_path):
    recent = _killed(tmp_path / "recent")
    _age(recent, 60)
    assert "written" in recording.salvage(recent)["why"]
    # Recorded by this process, which still runs.
    live = _killed(tmp_path / "live", recorder=recording.recorder_identity())
    assert "runs" in recording.salvage(live)["why"]
    closed = _killed(tmp_path / "closed", encoder_exit=0, reason="the player's setup failed")
    assert "closed" in recording.salvage(closed)["why"]
    whole = _killed(tmp_path / "whole", complete=True)
    assert recording.salvage(whole)["why"] == "complete"
    short = _killed(tmp_path / "short", rows=30, video=30)
    assert "only 30" in recording.salvage(short)["why"]
    unmatched = _killed(tmp_path / "unmatched", rows=40, video=45, torn=False)
    assert "45 frames but only 40 rows" in recording.salvage(unmatched)["why"]
    dry = _killed(tmp_path / "dry")
    before = _files(dry)
    assert recording.salvage(dry, dry_run=True)["why"] == "dry run"
    assert _files(dry) == before
    for root in (recent, live, closed, whole, short, unmatched):
        assert not json.loads((root / "manifest.json").read_text()).get("salvaged")


@needs_ffmpeg
@pytest.mark.skipif(os.name != "nt", reason="an open file is told apart on Windows only")
def test_salvage_leaves_alone_a_recording_whose_files_are_open(tmp_path):
    root = _killed(tmp_path / "rec")
    with (root / "frames.jsonl").open("a"):
        _age(root)
        assert "frames.jsonl open" in recording.salvage(root)["why"]
    assert recording.salvage(root)["salvaged"]


def test_a_recorder_that_is_gone_or_a_reused_number_is_not_alive():
    me = recording.recorder_identity()
    assert recording._recorder_alive({"recorder": me})
    # This process was created before now, so a recording begun an hour ago was not its.
    assert not recording._recorder_alive({"recorder": {**me, "started_unix": time.time() - 3600}})
    assert not recording._recorder_alive({})


def test_recordings_are_found_under_folders(tmp_path):
    for name in ("a/one", "a/two", "b"):
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "manifest.json").write_text("{}")
        (tmp_path / name / "frames.jsonl").write_text("")
    (tmp_path / "a" / "no-rows").mkdir()
    (tmp_path / "a" / "no-rows" / "manifest.json").write_text("{}")
    found = recording.recordings([tmp_path / "a", tmp_path / "b"])
    assert found == [tmp_path / "a" / "one", tmp_path / "a" / "two", tmp_path / "b"]


@needs_ffmpeg
def test_only_the_unbroken_run_of_frames_counts(tmp_path):
    # Frame 15 of 20 is missing, as a killed x264 can leave a B-frame's successor behind it.
    path = tmp_path / "gap.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=16x16:rate=5:duration=4",
         "-vf", r"select=not(eq(n\,15))", "-fps_mode", "passthrough", "-c:v", "ffv1", str(path)],
        check=True,
    )  # fmt: skip
    assert recording.count_frames(path) == 19
    assert recording.whole_frames(path, 5) == 15
