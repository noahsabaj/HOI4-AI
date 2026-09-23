import json

import numpy as np
import pytest
import torch
from test_dataset import _recording, needs_ffmpeg

from hoi4_arena.actions import GRID, SLOTS
from hoi4_arena.dataset import (
    DETAIL_SIZE,
    FOVEA_SIZE,
    IDM_LABELS,
    QUADRANTS,
    VIEW_SIZE,
    VideoSessions,
    session_labels,
)
from hoi4_arena.idm import CLIP_SHIFT, DETAIL_SHIFT, label_accuracy, label_recording
from hoi4_arena.models import CELL_DIM, InverseDynamics


class _Encoder(torch.nn.Module):
    dim = 8

    def forward(self, clip):
        batch = clip.shape[0]
        pooled = clip.mean((1, 3, 4))  # (B, T): each frame's brightness, in order.
        summary = torch.nn.functional.pad(pooled, (0, self.dim - pooled.shape[1]))
        return summary, summary[:, :, None, None].expand(batch, self.dim, 14, 14)


@needs_ffmpeg
def test_the_inverse_model_sees_after_the_decision_and_the_policy_does_not(tmp_path):
    """Frame i reads i. The policy's clip ends on the decision; the IDM's runs past it."""
    _recording(tmp_path / "game", [8, 6])
    policy = {w["start"]: w for w in VideoSessions(tmp_path, length=2, burn_in=0, device="cpu")}
    shifted = VideoSessions(
        tmp_path,
        length=2,
        burn_in=0,
        device="cpu",
        clip_shift=CLIP_SHIFT,
        detail_shift=DETAIL_SHIFT,
    )
    ahead = {w["start"]: w for w in shifted}
    assert ahead and set(ahead) <= set(policy)
    for start, window in ahead.items():
        decision = policy[start]["clips"][:, -1, 0, 0, 0].int()
        assert torch.equal(policy[start]["quadrants"][:, 0, 0, 0, 0].int(), decision)
        # Two 10 Hz frames per decision interval.
        assert torch.equal(window["clips"][:, -1, 0, 0, 0].int(), decision + 2 * CLIP_SHIFT)
        assert torch.equal(window["quadrants"][:, 0, 0, 0, 0].int(), decision + 2 * DETAIL_SHIFT)


def test_decisions_whose_later_frames_are_missing_are_not_trained_on(tmp_path):
    _recording(tmp_path / "game", [1, 1])
    plain = session_labels(tmp_path / "game")
    ahead = session_labels(tmp_path / "game", clip_shift=CLIP_SHIFT, detail_shift=DETAIL_SHIFT)
    assert plain["valid"].all()
    # The recording ends at 3.9 s, and a decision needs the frame 0.8 s after it.
    needed = ahead["decisions"] + CLIP_SHIFT * 200_000_000
    assert np.array_equal(ahead["valid"], needed <= 3_900_000_000)
    assert 0 < ahead["valid"].sum() < len(ahead["valid"])


def test_each_label_knows_its_neighbours_in_both_directions():
    torch.manual_seed(0)
    model = InverseDynamics(_Encoder(), memory_dim=16)
    shape = (1, 5)
    clips = torch.zeros(*shape, 3, 8, VIEW_SIZE, VIEW_SIZE)
    quadrants = torch.zeros(*shape, QUADRANTS, 3, DETAIL_SIZE, DETAIL_SIZE)
    fovea = torch.zeros(*shape, 3, FOVEA_SIZE, FOVEA_SIZE)
    speed = torch.full(shape, 4)
    context, cells = model(clips, quadrants, fovea, speed)
    assert context.shape == (1, 5, 16) and cells.shape == (1, 5, GRID, CELL_DIM)
    later = clips.clone()
    later[:, 4] += 1  # Only the last decision's video changes...
    changed, _ = model(later, quadrants, fovea, speed)
    assert not torch.allclose(changed[:, 0], context[:, 0]), "...and the first one hears of it"


def test_label_accuracy_scores_kinds_and_pointer_distance():
    actual = torch.zeros(2, SLOTS, 3, dtype=torch.long)
    actual[:, 0] = torch.tensor([1, 100, 100])
    predicted = actual.clone()
    predicted[0, 0, 1] = 103  # A move three steps off.
    predicted[1, 1, 0] = 2  # A click where there was none.
    kinds, error = label_accuracy(predicted, actual)
    assert kinds == pytest.approx(15 / 16)
    assert error == pytest.approx(1.5)


@needs_ffmpeg
def test_a_labelled_video_trains_like_a_recording_with_inputs(tmp_path, monkeypatch):
    import hoi4_arena.idm as idm

    torch.manual_seed(1)
    model = InverseDynamics(_Encoder(), memory_dim=16).eval()
    config = {"clip_shift": CLIP_SHIFT, "detail_shift": DETAIL_SHIFT}
    monkeypatch.setattr(idm, "load_idm", lambda *a, **k: (model, config, "digest"))
    root = tmp_path / "videos"
    root.mkdir()
    _recording(root / "clip", [8, 6], source="video")
    with pytest.raises(ValueError, match="No complete valid"):
        VideoSessions(root, sources=("human",), device="cpu")
    result = label_recording("idm.pt", root / "clip", window=4, device="cpu")
    stored = np.load(root / "clip" / IDM_LABELS)
    assert result["decisions"] == len(stored["decisions"]) == 10
    assert result["labelled"] == int(
        session_labels(root / "clip", sources=("video",), clip_shift=CLIP_SHIFT)["valid"].sum()
    )
    assert stored["actions"].shape == (10, SLOTS, 3)
    assert (stored["actions"][..., 1:] < GRID).all()
    labels = session_labels(root / "clip", sources=("idm",))
    assert labels["label_source"] == "idm"
    assert np.array_equal(labels["actions"], stored["actions"])
    windows = list(VideoSessions(root, length=2, burn_in=0, sources=("idm",), device="cpu"))
    assert windows and all(w["valid"].all() for w in windows)
    # A label from another decision grid is refused rather than misaligned.
    manifest = json.loads((root / "clip" / "manifest.json").read_text())
    rows = (root / "clip" / "frames.jsonl").read_text().splitlines()[1:]
    (root / "clip" / "frames.jsonl").write_text("\n".join(rows) + "\n")
    (root / "clip" / "manifest.json").write_text(json.dumps({**manifest, "frames": len(rows)}))
    with pytest.raises(ValueError, match="decision grid"):
        session_labels(root / "clip", sources=("idm",))
