import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hoi4_arena.dataset import Sessions, quadrants, views


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
    actual_global, actual_tiles = views(frame)
    assert torch.equal(actual_global, expected_global)
    assert torch.equal(actual_tiles, expected_tiles)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU this path runs on")
@pytest.mark.parametrize("shape", SHAPES)
def test_batched_view_resize_is_identical_on_the_device_it_actually_runs_on(shape):
    """prepare_session resizes on CUDA when it can, so CPU agreement is not the claim."""
    frame = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)
    expected_global, expected_tiles = one_box_at_a_time(frame, device="cuda")
    actual_global, actual_tiles = views(frame, device="cuda")
    assert torch.equal(actual_global, expected_global)
    assert torch.equal(actual_tiles, expected_tiles)


def test_exact_length_sequence_uses_only_past_video_frames(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"prepared": True, "split": "train"}))
    global_frames = np.broadcast_to(
        np.arange(40, dtype=np.uint8)[:, None, None, None], (40, 224, 224, 3)
    ).copy()
    data = {
        "global": global_frames,
        "details": np.zeros((3, 4, 224, 224, 3), np.uint8),
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
