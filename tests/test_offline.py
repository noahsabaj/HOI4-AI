import json

import numpy as np
import pytest
import torch
from test_dataset import _recording, needs_ffmpeg

from hoi4_arena.dataset import ADVANTAGE_LABELS, session_labels
from hoi4_arena.models import Policy
from hoi4_arena.offline import advantage_weights, player_side, value_recording


def test_an_input_before_the_position_improved_counts_more():
    values = np.array([0.0, 0.0, 0.5, 0.5, 0.5])
    outcome = np.full(5, np.nan)
    valid = np.ones(5, bool)
    advantage, weight = advantage_weights(values, outcome, 1.0, valid, n_step=1, beta=0.5)
    # Blue's value rose after the second decision: that one is credited, the rest not.
    assert advantage.tolist() == pytest.approx([0.0, 0.5, 0.0, 0.0, 0.0])
    assert weight[1] > weight[0] and weight.mean() == pytest.approx(1.0)
    # The same game played as Red: the rise was the enemy's.
    advantage, weight = advantage_weights(values, outcome, -1.0, valid, n_step=1, beta=0.5)
    assert advantage[1] == pytest.approx(-0.5) and weight[1] < weight[0]


def test_the_end_of_a_game_looks_to_its_outcome_and_weights_are_capped():
    values = np.zeros(4)
    outcome = np.full(4, 1.0)  # Blue won
    advantage, weight = advantage_weights(
        values, outcome, 1.0, np.ones(4, bool), n_step=2, beta=0.01, max_weight=5.0
    )
    assert advantage.tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])
    # exp(100) capped at 5 before scaling to a mean of one.
    raw = np.array([1.0, 1.0, 5.0, 5.0])
    assert weight == pytest.approx(raw / raw.mean())


def test_an_invalid_decision_carries_no_weight():
    _, weight = advantage_weights(
        np.zeros(3), np.full(3, np.nan), 1.0, np.array([True, False, True]), n_step=1
    )
    assert weight[1] == 0 and weight[[0, 2]].mean() == pytest.approx(1.0)


def test_only_a_players_own_game_can_be_weighed():
    assert player_side({"source": "human", "players": ["RED"]}) == -1.0
    with pytest.raises(ValueError, match="AI game"):
        player_side({"source": "ai", "players": ["BLU"]})
    with pytest.raises(ValueError, match="one player"):
        player_side({"source": "human"})


@needs_ffmpeg
def test_advantage_weights_reach_the_labels_only_when_asked(tmp_path):
    _recording(tmp_path / "game", [8, 8], frames=40)
    decisions = session_labels(tmp_path / "game")["decisions"]
    weights = np.linspace(0.5, 1.5, len(decisions)).astype(np.float32)
    np.savez(tmp_path / "game" / ADVANTAGE_LABELS, decisions=decisions, weight=weights)
    assert session_labels(tmp_path / "game")["weight"] == pytest.approx(np.ones(len(weights)))
    assert session_labels(tmp_path / "game", advantage=True)["weight"] == pytest.approx(weights)
    np.savez(tmp_path / "game" / ADVANTAGE_LABELS, decisions=decisions + 1, weight=weights)
    with pytest.raises(ValueError, match="decision grid"):
        session_labels(tmp_path / "game", advantage=True)


class _Encoder(torch.nn.Module):
    """Reads the quadrants' brightness, so each decision's frame gives a different value."""

    dim = 8
    reads_clip = False

    def forward(self, clip, quadrants):
        level = quadrants.mean((1, 2, 3, 4))[:, None]
        summary = level.expand(-1, self.dim)
        return summary, summary[:, :, None, None].expand(-1, self.dim, 14, 14)


@needs_ffmpeg
def test_values_carry_the_memory_through_the_game_whatever_the_window(tmp_path):
    """The last window overlaps the one before; the memory must see each decision once."""
    _recording(tmp_path / "game", [8, 8], frames=60)
    labels = session_labels(tmp_path / "game")
    torch.manual_seed(0)
    policy = Policy(_Encoder(), memory_dim=16).eval()
    whole = value_recording(policy, labels, "cpu", window=int(labels["readable"].sum()))
    pieces = value_recording(policy, labels, "cpu", window=5)
    assert np.isfinite(whole).all() and len(np.unique(whole.round(6))) > 1
    # Convolutions over different batch sizes round differently on the CPU, by about
    # 3e-5 here; a decision stepped twice or skipped moves every later value by far more.
    np.testing.assert_allclose(pieces, whole, rtol=0, atol=2e-4)
    assert np.abs(whole[1:] - whole[:-1]).max() > 2e-3, "values change enough to see a slip"
    assert json.loads((tmp_path / "game" / "manifest.json").read_text())["source"] == "human"
