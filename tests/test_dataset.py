import json
import shutil
import subprocess

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hoi4_arena.dataset import (
    CLIP_FRAMES,
    DETAIL_SIZE,
    FOVEA_SIZE,
    PERIOD_NS,
    QUADRANTS,
    VIEW_SIZE,
    VideoSessions,
    batch_to_device,
    clip_frame_ids,
    cursor_crop,
    quadrants,
    recorded_speed,
    session_labels,
    views,
)


def one_box_at_a_time(rgb, size=VIEW_SIZE, detail=DETAIL_SIZE, device="cpu"):
    """The resize `views` does, one interpolate call per box.

    Kept as the thing the batched version has to equal. The worker reimplements this
    resize in Rust and a test pins the two together, so a pixel that moves here moves
    the policy's input away from the pixels the worker will feed it at deployment --
    which is the failure this whole filter choice exists to prevent.
    """
    source = torch.as_tensor(np.ascontiguousarray(rgb), device=device).permute(2, 0, 1)[None]
    h, w = source.shape[-2:]

    def area(box, s):
        scaled = F.interpolate(box.float(), s, mode="area")
        return scaled.round().clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0)

    boxes = [source[..., y : y + bh, x : x + bw] for y, x, bh, bw in quadrants(h, w)]
    return area(source, size), torch.stack([area(box, detail) for box in boxes])


SHAPES = [(2160, 3840, 3), (1080, 1920, 3), (1081, 1921, 3), (65, 63, 3)]


@pytest.mark.parametrize("shape", SHAPES)
def test_batched_view_resize_is_identical_to_resizing_one_box_at_a_time(shape):
    """Odd shapes are in the list on purpose: they take the fallback, not the batch."""
    frame = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    expected_global, expected_quadrants = one_box_at_a_time(frame)
    seen = views(frame, cursor=(0, 0))
    assert torch.equal(seen.global_view, expected_global)
    assert torch.equal(seen.quadrants, expected_quadrants)
    assert seen.quadrants.shape == (QUADRANTS, *DETAIL_SIZE, 3)
    assert torch.equal(seen.fovea, torch.from_numpy(cursor_crop(frame, 0, 0, FOVEA_SIZE)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU this path runs on")
@pytest.mark.parametrize("shape", SHAPES)
def test_batched_view_resize_is_identical_on_the_device_it_actually_runs_on(shape):
    """Training resizes on CUDA when it can, so CPU agreement is not the claim."""
    frame = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)
    expected_global, expected_quadrants = one_box_at_a_time(frame, device="cuda")
    seen = views(frame, device="cuda", cursor=(0, 0))
    assert torch.equal(seen.global_view, expected_global)
    assert torch.equal(seen.quadrants, expected_quadrants)
    assert torch.equal(seen.fovea.cpu(), torch.from_numpy(cursor_crop(frame, 0, 0, FOVEA_SIZE)))


def test_live_and_training_clips_select_the_same_timestamps():
    """A 5 Hz actor and a 10 Hz recording must hand the encoder the same frames.

    Lookback used to step every 1/7.5 s. The live actor stores one view per decision,
    so that spacing repeated neighbours, while a 10 Hz recording still had a distinct
    frame at each step.
    """
    record_ns = 100_000_000
    times = np.arange(40, dtype=np.int64) * record_ns
    live = times[::2]
    decision = 20 * record_ns
    dense = times[clip_frame_ids(times, decision)]
    sparse = live[clip_frame_ids(live, decision)]
    assert np.array_equal(dense, sparse)
    assert dense.shape == (CLIP_FRAMES,)
    assert len(set(dense.tolist())) == CLIP_FRAMES
    assert np.all(np.diff(dense) == PERIOD_NS)
    assert dense[-1] == decision
    assert PERIOD_NS == 200_000_000


# Same 3x3 image as desktop-worker cursor_crop_keeps_the_pointer_pixel_and_pads_outside_with_zero.
_POINTER = np.arange(1, 28, dtype=np.uint8).reshape(3, 3, 3)


def test_cursor_crop_keeps_the_pointer_pixel_centered():
    """The pointer stays at (size // 2, size // 2), including when that needs padding."""
    assert np.array_equal(cursor_crop(_POINTER, 1, 1, 3), _POINTER)
    corner = cursor_crop(_POINTER, 0, 0, 3)
    assert not corner[0].any() and not corner[:, 0].any()
    assert np.array_equal(corner[1:, 1:], _POINTER[:2, :2])
    assert int(cursor_crop(_POINTER, 9, -4, 3).sum()) == 0
    # Center index of a size-4 crop is 2, so cursor (5, 6) opens at source (3, 4).
    bigger = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    assert np.array_equal(cursor_crop(bigger, 5, 6, 4), bigger[4:8, 3:7])


def test_views_centres_the_fovea_on_the_pointer_and_refuses_to_invent_one():
    seen = views(_POINTER, size=3, detail=2, fovea=3, cursor=(0, 0))
    assert torch.equal(seen.fovea, torch.from_numpy(cursor_crop(_POINTER, 0, 0, 3)))
    assert seen.quadrants.shape == (4, 2, 2, 3)
    with pytest.raises(ValueError, match="cursor"):
        views(_POINTER, size=3)


def _recording(root, cursor, *, game_speed=4, frames=40, source="human", events=None):
    """A recording as Recorder writes it: manifest, frames.jsonl, and ffv1 video.

    Frame i is a flat image of value i, so any view of it can be traced back to it.
    """
    root.mkdir()
    manifest = {
        "complete": True,
        "source": source,
        "split": "train",
        "width": 16,
        "height": 16,
        "frames": frames,
    }
    if game_speed is not None:
        manifest["game_speed"] = game_speed
    (root / "manifest.json").write_text(json.dumps(manifest))
    rows = []
    for i in range(frames):
        row = {"t_ns": i * 100_000_000, "events": []}
        if cursor is not None:
            row["cursor"] = cursor
        rows.append(row)
    for t_ns, event in events or []:
        rows[t_ns // 100_000_000]["events" if source == "human" else "scripted_events"] = [
            {"t_ns": t_ns, "event": event}
        ]
    (root / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    encoder = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo"]
        + ["-pixel_format", "rgb24", "-video_size", "16x16", "-framerate", "10"]
        + ["-i", "pipe:0", "-an", "-c:v", "ffv1", "-level", "3", str(root / "screen.mkv")],
        stdin=subprocess.PIPE,
    )
    for i in range(frames):
        encoder.stdin.write(np.full((16, 16, 3), i, np.uint8).tobytes())
    encoder.stdin.close()
    assert encoder.wait(timeout=30) == 0


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def test_a_recording_with_no_cursor_or_no_speed_is_refused_before_decoding(tmp_path):
    _recording(tmp_path / "raw", None)
    with pytest.raises(ValueError, match="cursor"):
        session_labels(tmp_path / "raw")
    _recording(tmp_path / "bools", [True, 0])
    with pytest.raises(ValueError, match="cursor"):
        session_labels(tmp_path / "bools")
    _recording(tmp_path / "slow", [1, 1], game_speed=None)
    with pytest.raises(ValueError, match="game speed"):
        session_labels(tmp_path / "slow")
    _recording(tmp_path / "ai", [1, 1], source="ai")
    with pytest.raises(ValueError, match="human"):
        session_labels(tmp_path / "ai")
    assert session_labels(tmp_path / "ai", sources=("ai",))["label_source"] == "ai"


def test_labels_come_from_the_player_or_from_the_scripted_camera(tmp_path):
    click = {"kind": "button", "button": 0, "down": True}
    _recording(tmp_path / "human", [1, 1], events=[(2_000_000_000, click)])
    _recording(tmp_path / "ai", [1, 1], source="ai", events=[(2_000_000_000, click)])
    for name, sources in (("human", ("human",)), ("ai", ("ai",))):
        labels = session_labels(tmp_path / name, sources=sources)
        # Decision 1 starts at 2.0 s: the lead-in is nine intervals, 1.8 s.
        assert labels["decisions"][1] == 2_000_000_000
        assert labels["actions"][1, 0, 0] != 0 and not labels["actions"][0].any()


@needs_ffmpeg
def test_training_windows_read_only_past_frames_and_carry_the_speed(tmp_path):
    _recording(tmp_path / "slow", [8, 6], game_speed=2)
    _recording(tmp_path / "fast", [8, 6], game_speed=4)
    dataset = VideoSessions(tmp_path, length=2, burn_in=1, device="cpu", shuffle=0)
    # Ten decisions from 1.8 s to 3.6 s; windows of three, every two, per recording.
    assert len(dataset) == 8
    windows = list(dataset)
    assert len(windows) == 8
    assert {int(w["speed"][0]) for w in windows} == {2, 4}, "two speeds train together"
    for window in windows:
        clips = window["clips"]
        assert clips.shape == (3, CLIP_FRAMES, *VIEW_SIZE, 3)
        assert window["quadrants"].shape == (3, QUADRANTS, *DETAIL_SIZE, 3)
        assert window["fovea"].shape == (3, FOVEA_SIZE, FOVEA_SIZE, 3)
        # Frame i reads i. Each clip ends on its decision's frame and never reaches past it.
        latest = clips[:, -1, 0, 0, 0]
        assert torch.equal(clips.amax((1, 2, 3, 4)), latest)
        assert torch.equal(window["quadrants"][:, 0, 0, 0, 0], latest)
        assert torch.equal(window["fovea"][:, FOVEA_SIZE // 2, FOVEA_SIZE // 2, 0], latest)
        assert (torch.diff(latest.int()) == 2).all(), "one decision every two 10 Hz frames"
    batch = batch_to_device(torch.utils.data.default_collate(windows[:2]), "cpu")
    assert batch["clips"].shape == (2, 3, 3, CLIP_FRAMES, *VIEW_SIZE)
    assert batch["quadrants"].shape == (2, 3, QUADRANTS, 3, *DETAIL_SIZE)
    assert batch["fovea"].shape == (2, 3, 3, FOVEA_SIZE, FOVEA_SIZE)


@needs_ffmpeg
def test_a_truncated_video_fails_the_window_rather_than_training_on_it(tmp_path):
    _recording(tmp_path / "short", [1, 1])
    manifest = tmp_path / "short" / "manifest.json"
    meta = json.loads(manifest.read_text())
    frames = tmp_path / "short" / "frames.jsonl"
    rows = frames.read_text().splitlines()
    rows += [json.dumps({"t_ns": (40 + i) * 100_000_000, "cursor": [1, 1]}) for i in range(10)]
    frames.write_text("\n".join(rows) + "\n")
    manifest.write_text(json.dumps({**meta, "frames": 50}))
    with pytest.raises(ValueError, match="truncated"):
        list(VideoSessions(tmp_path, length=2, burn_in=1, device="cpu"))


def test_no_trainable_recording_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No complete valid"):
        VideoSessions(tmp_path)


def test_recorded_speed_fixes_the_game_length_of_a_clip():
    """Speed 2 and speed 4 are not the same 1.6 s clip. Speed 5 has no fixed length."""
    wall = CLIP_FRAMES * 0.2
    assert wall / recorded_speed(2)["seconds_per_hour"] == pytest.approx(3.2)
    assert wall / recorded_speed(4)["seconds_per_hour"] == pytest.approx(16.0)
    assert recorded_speed(5) == {"game_speed": 5, "seconds_per_hour": None}
    for bad in (None, True, False, 0, 6, 2.0):
        with pytest.raises(ValueError, match="game speed"):
            recorded_speed(bad)


def test_a_capture_stall_costs_only_the_decisions_that_read_across_it(tmp_path):
    """The second PC's capture sometimes stalls for over a second; the rest still trains."""
    _recording(tmp_path / "game", [8, 8], frames=80)
    path = tmp_path / "game" / "frames.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows[40:]:
        row["t_ns"] += 1_300_000_000  # 3.9 s, then nothing until 5.3 s
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    labels = session_labels(tmp_path / "game")
    seconds = labels["decisions"] / 1e9
    # A decision reads from 1.8 s (the clip and a margin) before it to 0.2 s after.
    spans = (seconds - 1.8 < 5.3) & (seconds + 0.2 > 3.9)
    assert spans.any() and (~spans).any()
    assert not labels["valid"][spans].any() and labels["valid"][~spans].all()
    assert sum(e["reason"] == "capture gap" for e in labels["excluded"]) == spans.sum()


def test_timestamps_that_go_backwards_still_refuse_the_recording(tmp_path):
    _recording(tmp_path / "game", [8, 8], frames=40)
    path = tmp_path / "game" / "frames.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[20]["t_ns"] = rows[19]["t_ns"]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="Nonmonotonic"):
        session_labels(tmp_path / "game")


@needs_ffmpeg
def test_record_with_a_peer_focuses_its_game_and_records_it_here(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hoi4_arena import recording, remote

    calls = []

    class Peer:
        def __init__(self, config):
            calls.append(("connect", config))
            self.seq = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            calls.append(("close",))

        def focus(self):
            calls.append(("focus",))
            return True

        def arm(self, setup=False):
            calls.append(("arm",))

        def release(self):
            calls.append(("release",))

        def capture(self):
            self.seq += 1
            meta = {"t_ns": self.seq * 100_000_000, "cursor": [4, 4], "foreground": True}
            meta["events"] = []
            return SimpleNamespace(
                rgb=np.full((16, 16, 3), self.seq, np.uint8),
                views=None,
                meta=meta,
                received_ns=self.seq,
            )

        def request(self, op):
            return {"events": []}

        def game_log(self, offset):
            lines = ["declare RED", "player BLU", "capitulated BLU winner RED 1:00, 30 May, 1937"]
            if offset == 0:  # an earlier game of the same launch, already over
                return ["player RED", "capitulated RED winner BLU 1:00, 2 June, 1936"], 2
            if offset == 2:  # this recording's game, read once
                return lines, offset + len(lines)
            return [], offset

        def worker_log(self):
            return []

    monkeypatch.setattr(remote, "RemoteDesktop", Peer)
    recording.record(tmp_path / "rec", 4, hz=10, game_speed=5, codec="ffv1", peer="p.json")
    manifest = json.loads((tmp_path / "rec" / "manifest.json").read_text())
    assert manifest["complete"] and manifest["station"] == "peer" and manifest["frames"] >= 5
    assert calls[0] == ("connect", "p.json") and calls[1] == ("focus",)
    # Only this recording's game counts: its declarer, the player's country, the winner.
    assert (manifest["declarer"], manifest["players"], manifest["winner"]) == (
        "RED",
        ["BLU"],
        "RED",
    )
