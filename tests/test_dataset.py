import json
import shutil
import subprocess

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hoi4_arena.dataset import (
    CLIP_FRAMES,
    PERIOD_NS,
    TILES,
    Sessions,
    clip_frame_ids,
    cursor_crop,
    prepare_session,
    quadrants,
    recorded_speed,
    require_one_game_speed,
    views,
)


def one_box_at_a_time(rgb, size=224, device="cpu"):
    """The resize `views` used to do, one interpolate call per box.

    Kept as the thing the batched version has to equal. The worker reimplements this
    resize in Rust and a test pins the two together, so a pixel that moves here moves
    the policy's input away from the pixels the worker will feed it at deployment --
    which is the failure this whole filter choice exists to prevent.
    """
    source = torch.as_tensor(np.ascontiguousarray(rgb), device=device).permute(2, 0, 1)[None]
    h, w = source.shape[-2:]
    boxes = [source] + [
        source[..., top : top + bh, left : left + bw] for top, left, bh, bw in quadrants(h, w)
    ]
    resized = [
        F.interpolate(box.float(), (size, size), mode="area").round().clamp(0, 255).to(torch.uint8)
        for box in boxes
    ]
    stacked = torch.cat(resized).permute(0, 2, 3, 1)
    return stacked[0], stacked[1:]


SHAPES = [(2160, 3840, 3), (1080, 1920, 3), (1081, 1921, 3), (65, 63, 3)]


@pytest.mark.parametrize("shape", SHAPES)
def test_batched_view_resize_is_identical_to_resizing_one_box_at_a_time(shape):
    """Odd shapes are in the list on purpose: they take the fallback, not the batch."""
    frame = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    expected_global, expected_tiles = one_box_at_a_time(frame)
    actual_global, actual_tiles = views(frame, cursor=(0, 0))
    assert torch.equal(actual_global, expected_global)
    assert torch.equal(actual_tiles[:-1], expected_tiles)
    assert actual_tiles.shape[0] == TILES
    assert torch.equal(actual_tiles[-1].cpu(), torch.from_numpy(cursor_crop(frame, 0, 0, 224)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU this path runs on")
@pytest.mark.parametrize("shape", SHAPES)
def test_batched_view_resize_is_identical_on_the_device_it_actually_runs_on(shape):
    """prepare_session resizes on CUDA when it can, so CPU agreement is not the claim."""
    frame = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)
    expected_global, expected_tiles = one_box_at_a_time(frame, device="cuda")
    actual_global, actual_tiles = views(frame, device="cuda", cursor=(0, 0))
    assert torch.equal(actual_global, expected_global)
    assert torch.equal(actual_tiles[:-1], expected_tiles)
    assert torch.equal(actual_tiles[-1].cpu(), torch.from_numpy(cursor_crop(frame, 0, 0, 224)))


def test_exact_length_sequence_uses_only_past_video_frames(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"prepared": True, "split": "train", "game_speed": 4})
    )
    global_frames = np.broadcast_to(
        np.arange(40, dtype=np.uint8)[:, None, None, None], (40, 224, 224, 3)
    ).copy()
    data = {
        "global": global_frames,
        "details": np.zeros((3, TILES, 224, 224, 3), np.uint8),
        "times": np.arange(40, dtype=np.int64) * 100000000,
        "decisions": np.array([2100000000, 2300000000, 2500000000]),
        "frame_ids": np.array([21, 23, 25]),
        "actions": np.zeros((3, 8, 3), np.int64),
        "valid": np.ones(3, bool),
    }
    for name, value in data.items():
        np.save(root / f"{name}.npy", value)
    dataset = Sessions(tmp_path, length=2, burn_in=1)
    assert len(dataset) == 1
    batch = dataset[0]
    recovered = (batch["clips"][:, 0] * 0.229 + 0.485) * 255
    assert torch.allclose(recovered[:, -1, 0, 0], torch.tensor([21.0, 23.0, 25.0]), atol=1e-4)
    for i, latest in enumerate([21, 23, 25]):
        assert recovered[i].max() <= latest + 1e-4
    assert batch["previous"].shape == (3, 8, 3)


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
_POINTER = np.array(
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
    ],
    np.uint8,
)


def test_cursor_crop_keeps_the_pointer_pixel_centered():
    """The pointer stays at (size // 2, size // 2), including when that needs padding."""
    assert np.array_equal(cursor_crop(_POINTER, 1, 1, 3), _POINTER)
    assert cursor_crop(_POINTER, 0, 0, 3).reshape(-1).tolist() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        0,
        0,
        0,
        10,
        11,
        12,
        13,
        14,
        15,
    ]
    assert int(cursor_crop(_POINTER, 9, -4, 3).sum()) == 0
    # Center index of a size-4 crop is 2, so cursor (5, 6) opens at source (3, 4).
    # That patch is what a 1:1 area average of the same box returns.
    bigger = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    assert np.array_equal(cursor_crop(bigger, 5, 6, 4), bigger[4:8, 3:7])


def test_the_policy_fusion_is_sized_for_the_cursor_crop():
    from hoi4_arena.models import Policy

    class Encoder(torch.nn.Module):
        dim = 8

    policy = Policy(Encoder())
    assert policy.fusion.in_features == 8 + TILES * 256 + 64


def test_views_appends_the_cursor_crop_and_refuses_to_invent_one():
    _, tiles = views(_POINTER, size=3, cursor=(0, 0))
    assert tiles.shape[0] == TILES
    assert torch.equal(tiles[-1], torch.from_numpy(cursor_crop(_POINTER, 0, 0, 3)))
    with pytest.raises(ValueError, match="cursor"):
        views(_POINTER, size=3)


def test_a_prepared_session_without_the_cursor_crop_is_refused(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"prepared": True, "split": "train"}))
    for name, value in {
        "global": np.zeros((8, 224, 224, 3), np.uint8),
        "details": np.zeros((8, 4, 224, 224, 3), np.uint8),
        "times": np.arange(8, dtype=np.int64),
        "decisions": np.arange(8, dtype=np.int64),
        "frame_ids": np.arange(8, dtype=np.int64),
        "actions": np.zeros((8, 8, 3), np.int64),
        "valid": np.ones(8, bool),
    }.items():
        np.save(root / f"{name}.npy", value)
    with pytest.raises(ValueError, match="cursor crop"):
        Sessions(tmp_path, length=2, burn_in=1)


def _recording(root, cursor, *, game_speed=4):
    root.mkdir()
    frames = 40
    manifest = {
        "complete": True,
        "source": "human",
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
    (root / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_prepare_refuses_a_recording_that_has_no_cursor(tmp_path):
    source = tmp_path / "raw"
    _recording(source, None)
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="cursor"):
        prepare_session(source, destination)
    assert not destination.exists()
    _recording(tmp_path / "bools", [True, 0])
    with pytest.raises(ValueError, match="cursor"):
        prepare_session(tmp_path / "bools", tmp_path / "bools-out")
    assert not (tmp_path / "bools-out").exists()


def test_prepare_stores_the_cursor_crop_as_the_last_detail_tile(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required to prepare a session")
    source = tmp_path / "raw"
    _recording(source, [8, 6])
    frame = np.zeros((16, 16, 3), np.uint8)
    frame[:, :, 0] = np.arange(16, dtype=np.uint8)[None, :]
    frame[:, :, 1] = np.arange(16, dtype=np.uint8)[:, None]
    frame[:, :, 2] = 40
    encoder = subprocess.Popen(
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
            "16x16",
            "-framerate",
            "10",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            str(source / "screen.mkv"),
        ],
        stdin=subprocess.PIPE,
    )
    encoder.stdin.write(frame.tobytes() * 40)
    encoder.stdin.close()
    assert encoder.wait(timeout=30) == 0
    destination = tmp_path / "out"
    prepare_session(source, destination)
    details = np.load(destination / "details.npy")
    assert details.shape[1:] == (TILES, 224, 224, 3)
    # Pointer (8, 6) is crop pixel (112, 112). The pixel to its right stays to its right.
    assert details[:, -1, 112, 112].tolist() == [[8, 6, 40]] * len(details)
    assert details[0, -1, 112, 113].tolist() == [9, 6, 40]
    assert details[0, -1, 0, 0].tolist() == [0, 0, 0]
    prepared = json.loads((destination / "manifest.json").read_text())
    assert prepared["tiles"] == TILES and prepared["cursor_crop"] == 224
    assert prepared["game_speed"] == 4 and prepared["seconds_per_hour"] == 0.1


def test_recorded_speed_fixes_the_game_length_of_a_clip():
    """Speed 2 and speed 4 are not the same 1.6 s clip. Speed 5 has no fixed length."""
    wall = CLIP_FRAMES * 0.2
    assert wall / recorded_speed(2)["seconds_per_hour"] == pytest.approx(3.2)
    assert wall / recorded_speed(4)["seconds_per_hour"] == pytest.approx(16.0)
    assert recorded_speed(5) == {"game_speed": 5, "seconds_per_hour": None}
    for bad in (None, True, False, 0, 6, 2.0):
        with pytest.raises(ValueError, match="game speed"):
            recorded_speed(bad)
    with pytest.raises(ValueError, match="game time"):
        require_one_game_speed([2, 4])


def _prepared(root, game_speed):
    root.mkdir()
    manifest = {"prepared": True, "split": "train"}
    if game_speed is not None:
        manifest["game_speed"] = game_speed
    (root / "manifest.json").write_text(json.dumps(manifest))
    for name, value in {
        "global": np.zeros((1, 224, 224, 3), np.uint8),
        "details": np.zeros((1, TILES, 224, 224, 3), np.uint8),
        "times": np.zeros(1, np.int64),
        "decisions": np.zeros(1, np.int64),
        "frame_ids": np.zeros(1, np.int64),
        "actions": np.zeros((1, 8, 3), np.int64),
        "valid": np.ones(1, bool),
    }.items():
        np.save(root / f"{name}.npy", value)


def test_prepare_refuses_a_recording_with_no_game_speed(tmp_path):
    source = tmp_path / "raw"
    _recording(source, [1, 1], game_speed=None)
    with pytest.raises(ValueError, match="game speed"):
        prepare_session(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    _recording(tmp_path / "bools", [1, 1], game_speed=True)
    with pytest.raises(ValueError, match="game speed"):
        prepare_session(tmp_path / "bools", tmp_path / "bools-out")
    assert not (tmp_path / "bools-out").exists()


def test_a_prepared_session_without_a_game_speed_is_refused(tmp_path):
    _prepared(tmp_path / "session", None)
    with pytest.raises(ValueError, match="game speed"):
        Sessions(tmp_path)


def test_prepared_sessions_at_different_speeds_are_refused(tmp_path):
    _prepared(tmp_path / "slow", 2)
    _prepared(tmp_path / "fast", 4)
    with pytest.raises(ValueError, match="game time"):
        Sessions(tmp_path)
