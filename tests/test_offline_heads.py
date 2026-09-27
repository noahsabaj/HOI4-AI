import json

import numpy as np
import pytest
import torch

from hoi4_arena.actions import GRID, SLOTS
from hoi4_arena.features import CachedGame, MemoryHead, labels_as_trained
from hoi4_arena.models import CELL_DIM
from hoi4_arena.offline_heads import (
    coached_mask,
    offline_weights,
    practice_returns,
    success_windows,
    time_baseline,
    train_offline,
)

SECOND = 1_000_000_000


def test_a_setup_step_the_policy_did_pays_the_decisions_before_it():
    manifest = {"setup": {"steps": {
        "army": {"at": 2.0, "by": "policy"},
        "general": {"at": 3.0, "by": "coach"},
        "running": {"at": 1.0, "by": "policy"},
    }}}  # fmt: skip
    seconds = np.array([0.0, 1.0, 2.0, 2.5])
    ret = practice_returns(manifest, seconds, per_second=0.5)
    # Only the army counts: the coach's general pays nothing, nor does unpausing.
    assert ret.tolist() == pytest.approx([0.25, 0.5, 1.0, 0.0])
    keep = success_windows(manifest, seconds, horizon=1.0)
    assert keep.tolist() == [False, True, True, False]


def test_the_baseline_is_the_mean_return_at_each_second():
    baseline = time_baseline([np.array([0.0, 1.5]), np.array([0.2, 1.1, 9.0])],
                             [np.array([1.0, 3.0]), np.array([0.0, 1.0, 5.0])])  # fmt: skip
    assert baseline(np.array([0.5, 1.0, 20.0])).tolist() == pytest.approx([0.5, 2.0, 5.0])


def test_a_cache_labels_as_the_policy_was_trained():
    config = {"lead_in": 0, "drop_keys": [32], "look_before_click": True, "press_weight": 4.0,
              "setup_weight": 4.0, "drop_parking": True, "lr": 1e-4, "state_weight": 0.5}  # fmt: skip
    assert labels_as_trained(config) == {
        "lead_in": 0, "drop_keys": (32,), "look_before_click": True, "press_weight": 4.0,
        "setup_weight": 4.0, "drop_parking": True,
    }  # fmt: skip


def test_the_coachs_spans_are_marked_by_frame():
    mask = coached_mask({"coached": [{"from_frame": 2, "to_frame": 3}]}, np.arange(6))
    assert mask.tolist() == [False, False, True, True, False, False]


def _recording(root, manifest):
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def _cache(root, name, split, n, seed, recording, weight=None):
    rng = np.random.default_rng(seed)
    game = root / name
    game.mkdir(parents=True)
    np.save(game / "summary.npy", rng.standard_normal((n, 8)).astype(np.float16))
    np.save(game / "centre.npy", rng.standard_normal((n, CELL_DIM)).astype(np.float16))
    np.save(game / "cells.npy", rng.standard_normal((n, GRID, CELL_DIM)).astype(np.float16))
    actions = np.zeros((n, SLOTS, 3), np.int64)
    actions[::3, 0] = [1, 5, 7]
    np.savez(
        game / "labels.npz",
        actions=actions,
        valid=np.ones(n, bool),
        outcome=np.zeros(n, np.float32),
        speed=np.full(n, 5),
        weight=np.ones(n, np.float32) if weight is None else weight,
        decisions=np.arange(n, dtype=np.int64) * SECOND // 5,
        frame_ids=np.arange(n),
    )
    meta = {"split": split, "decisions": n, "recording": str(recording)}
    (game / "meta.json").write_text(json.dumps(meta))
    return game


def _games(tmp_path):
    won = _recording(tmp_path / "rec" / "won", {"source": "scripted", "players": ["BLU"],
                                                "winner": "BLU"})  # fmt: skip
    lost = _recording(tmp_path / "rec" / "lost", {"source": "scripted", "players": ["RED"],
                                                  "winner": "BLU"})  # fmt: skip
    practice = _recording(tmp_path / "rec" / "practice", {
        "source": "policy", "players": [], "winner": "timeout",
        "coached": [{"from_frame": 20, "to_frame": 29, "step": "front"}],
        "setup": {"steps": {"army": {"at": 2.0, "by": "policy"},
                            "front": {"at": 5.8, "by": "coach"}}},
    })  # fmt: skip
    cache = tmp_path / "cache"
    _cache(cache, "won", "train", 24, 1, won)
    _cache(cache, "lost", "train", 24, 2, lost)
    weight = np.zeros(30, np.float32)
    weight[20:30] = 1  # as session_labels weighs a practice episode: the coach's span only
    _cache(cache, "practice", "train", 30, 3, practice, weight)
    _cache(cache, "held", "validation", 20, 4, won)
    return cache


def test_filtered_imitation_keeps_the_wins_and_the_policys_own_successes(tmp_path):
    cache = _games(tmp_path)
    games = [CachedGame(cache / name) for name in ("won", "lost", "practice")]
    bc, _ = offline_weights(games, "bc")
    assert bc["lost"].sum() == 24 and bc["practice"].sum() == 10
    weights, summary = offline_weights(games, "filtered", horizon=1.0, press_weight=1,
                                       setup_weight=1)  # fmt: skip
    # The loss is gone, and the win carries what both carried.
    assert weights["lost"].sum() == 0 and weights["won"].sum() == pytest.approx(48)
    # The coach's span stays; the policy's own decisions in the second before its army join.
    practice = weights["practice"]
    assert practice[20:30].tolist() == [1.0] * 10
    assert np.flatnonzero(practice[:20]).tolist() == [5, 6, 7, 8, 9, 10]
    assert summary["scripted_wins"] == 1


def test_awr_raises_a_practice_decision_that_led_to_a_step(tmp_path):
    cache = _games(tmp_path)
    # The same episode, but the coach formed the army: the policy did nothing itself.
    failed = _recording(tmp_path / "rec" / "failed", {
        "source": "policy", "players": [], "winner": "timeout",
        "setup": {"steps": {"army": {"at": 4.0, "by": "coach"}}},
    })  # fmt: skip
    _cache(cache, "failed", "train", 30, 5, failed, np.zeros(30, np.float32))
    games = [CachedGame(cache / "practice"), CachedGame(cache / "failed")]
    weights, _ = offline_weights(games, "awr", practice_beta=0.1, press_weight=1,
                                 setup_weight=1)  # fmt: skip
    # Up to the army, the episode that formed it weighs more than one, the other less;
    # after it, neither got anything more, and both weigh one.
    assert (weights["practice"][:11] > 1).all() and (weights["failed"][:11] < 1).all()
    assert weights["practice"][15:20] == pytest.approx(np.ones(5))
    assert weights["failed"][15:30] == pytest.approx(np.ones(15))
    with pytest.raises(ValueError, match="win predictor"):
        offline_weights([CachedGame(cache / "won")], "awr")


def test_train_offline_writes_a_playable_policy_with_the_trained_layers(tmp_path):
    cache = _games(tmp_path)
    head = MemoryHead(8, "gru", look=True)
    policy = {**head.state_dict(), "encoder.frozen": torch.ones(3)}
    init = tmp_path / "init.pt"
    torch.save({"policy": policy, "config": {"look_before_click": True}}, init)
    report = train_offline(cache, init, tmp_path / "out", weighting="filtered", window=8,
                           decisions=16, epochs=1, device="cpu")  # fmt: skip
    assert report["updates"] > 0 and report["train_games"] == 3 and report["held_games"] == 1
    for when in ("held_before", "held_after"):
        assert np.isfinite(report[when]["scripted_win"]["nll"])
    saved = torch.load(tmp_path / "out" / "epoch-0000.pt", weights_only=True)
    assert torch.equal(saved["policy"]["encoder.frozen"], torch.ones(3))
    assert not torch.equal(saved["policy"]["fusion.weight"], policy["fusion.weight"])
    assert saved["config"]["offline"]["weighting"] == "filtered"
    manifest = json.loads((tmp_path / "out" / "epoch-0000.json").read_text())
    assert len(manifest["sha256"]) == 64
    with pytest.raises(FileExistsError):
        train_offline(cache, init, tmp_path / "out", device="cpu")
