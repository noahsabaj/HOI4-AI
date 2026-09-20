import json

import numpy as np
import torch

from hoi4_arena.dataset import Sessions


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
