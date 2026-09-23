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
from hoi4_arena.idm import CLIP_SHIFT, DETAIL_SHIFT, label_accuracy, label_recording, load_idm
from hoi4_arena.learning import save_checkpoint
from hoi4_arena.models import CELL_DIM, ActionHead, InverseDynamics, WindowAttention
from hoi4_arena.train import imitation_loss, imitation_score


class _Encoder(torch.nn.Module):
    dim = 8

    def forward(self, clip, quadrants=None):
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


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_each_label_knows_its_neighbours_in_both_directions(kind):
    torch.manual_seed(0)
    model = InverseDynamics(_Encoder(), memory_dim=16, context=kind).eval()
    shape = (1, 5)
    clips = torch.zeros(*shape, 3, 8, *VIEW_SIZE)
    quadrants = torch.zeros(*shape, QUADRANTS, 3, *DETAIL_SIZE)
    fovea = torch.zeros(*shape, 3, FOVEA_SIZE, FOVEA_SIZE)
    speed = torch.full(shape, 4)
    context, cells = model(clips, quadrants, fovea, speed)
    assert context.shape == (1, 5, 16) and cells.shape == (1, 5, GRID, CELL_DIM)
    later = clips.clone()
    later[:, 4] += 1  # Only the last decision's video changes...
    changed, _ = model(later, quadrants, fovea, speed)
    assert not torch.allclose(changed[:, 0], context[:, 0]), "...and the first one hears of it"
    earlier = clips.clone()
    earlier[:, 0] += 1  # And the other way.
    changed, _ = model(earlier, quadrants, fovea, speed)
    assert not torch.allclose(changed[:, 4], context[:, 4])


@pytest.mark.parametrize("steps", [16, 32, 64])
def test_attention_covers_every_window_the_sweep_uses(steps):
    torch.manual_seed(0)
    attention = WindowAttention(16, layers=2).eval()
    window = torch.randn(2, steps, 16)
    out = attention(window)
    assert out.shape == (2, steps, 16) and torch.isfinite(out).all()
    # Unlike a causal model, the first step hears of a change to the last...
    later = window.clone()
    later[:, -1] = torch.randn(2, 16)  # Not a constant: layer norm would take it out.
    assert not torch.allclose(attention(later)[:, 0], out[:, 0])
    # ...and order matters: the positions tell the steps apart.
    assert not torch.allclose(attention(window.flip(1)).flip(1), out)


def _small(encoder, **kwargs):
    return InverseDynamics(encoder, memory_dim=16, **kwargs)


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_a_checkpoint_rebuilds_the_context_it_was_trained_with(tmp_path, monkeypatch, kind):
    import hoi4_arena.idm as idm

    monkeypatch.setattr(idm, "build_encoder", lambda *a, **k: _Encoder())
    monkeypatch.setattr(idm, "InverseDynamics", _small)
    torch.manual_seed(0)
    model = _small(_Encoder(), context=kind, layers=3)
    config = {"kind": "idm", "variant": "large", "model_path": "unused", "context": kind}
    config["context_layers"] = 3
    save_checkpoint(tmp_path / "idm.pt", model, config)
    loaded, saved, _ = load_idm(tmp_path / "idm.pt", device="cpu")
    assert saved["context"] == kind and loaded.context_kind == kind
    assert type(loaded.context) is type(model.context)
    state = loaded.state_dict()
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())


def test_a_checkpoint_from_before_the_option_is_the_gru(tmp_path, monkeypatch):
    import hoi4_arena.idm as idm

    monkeypatch.setattr(idm, "build_encoder", lambda *a, **k: _Encoder())
    monkeypatch.setattr(idm, "InverseDynamics", _small)
    config = {"kind": "idm", "variant": "large", "model_path": "unused"}
    save_checkpoint(tmp_path / "idm.pt", _small(_Encoder()), config)
    assert load_idm(tmp_path / "idm.pt", device="cpu")[0].context_kind == "gru"


def _idm_labelled(root, logp):
    """A recording without inputs, with IDM labels whose confidence is `logp`."""
    _recording(root, [8, 6], source="video")
    decisions = session_labels(root, sources=("video",))["decisions"]
    assert len(decisions) == len(logp)
    np.savez(
        root / IDM_LABELS,
        decisions=decisions,
        actions=np.zeros((len(decisions), SLOTS, 3), np.int64),
        valid=np.ones(len(decisions), bool),
        logp=np.asarray(logp, np.float32),
        checkpoint="digest",
    )


def test_unsure_idm_labels_are_not_trained_on_and_the_rest_count_less(tmp_path):
    logp = [-0.1, -5.0, -0.2, -9.0, -0.3, -0.4, -0.5, -0.6, -2.0, -0.7]
    _idm_labelled(tmp_path / "clip", logp)
    plain = session_labels(tmp_path / "clip", sources=("idm",))
    assert plain["valid"].all() and (plain["weight"] == 1).all()
    labels = session_labels(tmp_path / "clip", sources=("idm",), idm_min_logp=-1.0, idm_weight=0.5)
    assert np.array_equal(labels["valid"], np.array(logp) >= -1.0)
    assert (labels["weight"] == 0.5).all()
    assert [e["decision"] for e in labels["excluded"]] == [1, 3, 8]
    # Recorded labels keep their full weight whatever the IDM options say.
    _recording(tmp_path / "game", [8, 6])
    recorded = session_labels(
        tmp_path / "game", sources=("human", "idm"), idm_min_logp=0.0, idm_weight=0.5
    )
    assert recorded["valid"].all() and (recorded["weight"] == 1).all()


@needs_ffmpeg
def test_windows_carry_each_decisions_weight(tmp_path):
    _idm_labelled(tmp_path / "clip", [-0.1] * 10)
    _recording(tmp_path / "game", [8, 6])
    common = {"length": 2, "burn_in": 0, "sources": ("human", "idm"), "device": "cpu"}
    windows = list(VideoSessions(tmp_path, idm_weight=0.25, **common))
    assert {float(w["weight"].unique()) for w in windows} == {0.25, 1.0}
    # Every IDM label is below a threshold of 0, so only the recorded game is left.
    filtered = list(VideoSessions(tmp_path, idm_min_logp=0.0, **common))
    assert filtered and all((w["weight"] == 1).all() for w in filtered)


def test_weights_reach_the_imitation_loss():
    torch.manual_seed(0)
    policy = type("Policy", (), {"actor": ActionHead(16)})()
    memory, cells = torch.randn(2, 3, 16), torch.randn(2, 3, GRID, CELL_DIM)
    actions = torch.zeros(2, 3, SLOTS, 3, dtype=torch.long)
    actions[..., 0, :] = torch.tensor([1, 40, 700])
    score = imitation_score(policy, memory, cells, actions, "bc")
    ones = torch.ones(2, 3)
    assert torch.equal(imitation_loss(score, ones), -score.mean())
    mixed = torch.tensor([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    assert not torch.allclose(imitation_loss(score, mixed), imitation_loss(score, ones))
    # A weight the whole batch shares still scales it: IDM labels alone count less.
    assert torch.allclose(imitation_loss(score, ones / 2), imitation_loss(score, ones) / 2)


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
