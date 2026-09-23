"""A player's recording keeps everything already recorded, whatever ends it."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_dataset import needs_ffmpeg

from hoi4_arena import recording
from hoi4_arena.desktop import DesktopError, EmergencyStop


class _Game:
    """A worker whose frames follow a script: "ok", "away", "small", "F12" or an error."""

    def __init__(self, script, log_lines=()):
        self.script, self.seq, self.log_lines = list(script), 0, list(log_lines)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def focus(self):
        return True

    def arm(self, setup=False):
        pass

    def release(self):
        pass

    def capture(self):
        step = self.script.pop(0) if self.script else "ok"
        if step == "F12":
            raise EmergencyStop("F12 emergency stop")
        if step == "unfocused":
            raise DesktopError("game_not_foreground")
        if isinstance(step, BaseException):
            raise step
        self.seq += 1
        size = 8 if step == "small" else 16
        meta = {
            "t_ns": self.seq * 100_000_000,
            "cursor": [4, 4],
            "foreground": step != "away",
            "width": size,
            "height": size,
            "events": [],
        }
        pixels = np.full((size, size, 3), self.seq % 255, np.uint8)
        return SimpleNamespace(rgb=pixels, views=None, meta=meta, received_ns=self.seq)

    def request(self, op):
        return {"events": []}

    def game_log(self, offset):
        if offset == 0:
            return [], 1
        lines, self.log_lines = self.log_lines, []
        return lines, offset + len(lines)

    def worker_log(self):
        return []


def _record(tmp_path, monkeypatch, game, seconds=6.0):
    monkeypatch.setattr(recording, "Desktop", lambda command=None: game)
    return recording.record(tmp_path / "rec", seconds, hz=20, game_speed=5, codec="ffv1")


def _frames(tmp_path):
    return [
        json.loads(line) for line in (tmp_path / "rec" / "frames.jsonl").read_text().splitlines()
    ]


@needs_ffmpeg
def test_losing_focus_pauses_the_recording_and_it_resumes(tmp_path, monkeypatch):
    game = _Game(["ok"] * 20 + ["away"] * 10 + ["unfocused"] * 5 + ["small"] * 3 + ["ok"] * 20)
    manifest = _record(tmp_path, monkeypatch, game, seconds=3.0)
    assert manifest["complete"] and manifest["ended"] == "time"
    assert manifest["focus_pauses"] == 1
    # Nothing from the pause is written, and the video matches the rows.
    assert all(row["foreground"] and row["width"] == 16 for row in _frames(tmp_path))
    assert manifest["frames"] == len(_frames(tmp_path)) >= 40


@needs_ffmpeg
@pytest.mark.parametrize(
    "stop, ended",
    [("F12", "F12"), (DesktopError("worker lost"), "error: worker lost")],
)
def test_a_recording_that_ends_early_keeps_what_it_has(tmp_path, monkeypatch, stop, ended):
    manifest = _record(tmp_path, monkeypatch, _Game(["ok"] * 40 + [stop]))
    assert manifest["complete"] and manifest["ended"] == ended and manifest["frames"] == 40


@needs_ffmpeg
def test_ctrl_c_ends_a_recording_cleanly(tmp_path, monkeypatch):
    manifest = _record(tmp_path, monkeypatch, _Game(["ok"] * 40 + [KeyboardInterrupt()]))
    assert manifest["complete"] and manifest["ended"] == "Ctrl+C"


@needs_ffmpeg
def test_a_recording_too_short_to_train_on_says_so(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="only 10 frames"):
        _record(tmp_path, monkeypatch, _Game(["ok"] * 10 + ["F12"]))
    assert not json.loads((tmp_path / "rec" / "manifest.json").read_text())["complete"]


@needs_ffmpeg
def test_an_arena_recording_stops_after_the_surrender(tmp_path, monkeypatch):
    monkeypatch.setattr(recording, "AFTER_SURRENDER", 2.5)
    monkeypatch.setattr(recording, "LOG_EVERY", 0.1)
    game = _Game(
        ["ok"] * 400, ["declare BLU", "player RED", "capitulated BLU winner RED  1:00, 3 May, 1937"]
    )
    manifest = _record(tmp_path, monkeypatch, game, seconds=60)
    assert manifest["ended"] == "surrender" and manifest["winner"] == "RED"
    assert manifest["declarer"] == "BLU" and manifest["players"] == ["RED"]
