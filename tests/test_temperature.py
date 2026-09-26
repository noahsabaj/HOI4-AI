"""Sampling a policy at a temperature: the head, the actor, what the runs record, and the
scoreboard that compares them."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from hoi4_arena import practice
from hoi4_arena.actions import GRID
from hoi4_arena.models import CELL_DIM, CELLS, ActionHead
from hoi4_arena.play import sampling_of
from hoi4_arena.runner import Actor, resolve_temperatures

ROOT = Path(__file__).resolve().parents[1]


def _head(move_bias=None):
    torch.manual_seed(0)
    head = ActionHead(memory_dim=8).eval()
    if move_bias is not None:
        with torch.no_grad():
            head.kinds.bias.zero_()
            head.kinds.bias[1] = move_bias
    return head, torch.randn(64, 8), torch.randn(64, GRID, CELL_DIM)


def _places(action):
    """Every slot's (x, y) as one tuple per sample."""
    return {tuple(row) for row in action[:, :, 1:].flatten(1).tolist()}


def test_a_temperature_of_one_is_the_very_same_draw_as_before():
    head, memory, cells = _head()
    with torch.no_grad():
        torch.manual_seed(3)
        plain = head(memory, cells)
        torch.manual_seed(3)
        tempered = head(memory, cells, temperature=1.0, pointer_temperature=1.0)
    for a, b in zip(plain, tempered, strict=True):
        assert torch.equal(a, b)


def test_a_pointer_temperature_of_zero_points_like_point_and_still_samples_the_kind():
    head, memory, cells = _head(move_bias=50.0)  # Always a move.
    memory, cells = memory[:1].expand(8, -1), cells[:1].expand(8, -1, -1)
    with torch.no_grad():
        cold = head(memory, cells, pointer_temperature=0.0)[0]
        greedy = head(memory, cells, point=True)[0]
        warm = head(memory, cells)[0]
        assert len(_places(cold)) == 1 and torch.equal(cold[0], greedy[0])
        assert len(_places(warm)) > 1
        head.kinds.bias[1] = 0.0
        head.kinds.bias[0] = 0.5
        kinds = head(memory, cells, pointer_temperature=0.0)[0][:, :, 0]
    assert len({tuple(k) for k in kinds.tolist()}) > 1, "what to do is still sampled"


def test_a_low_pointer_temperature_sharpens_where_it_points_and_leaves_scoring_alone():
    head, memory, cells = _head(move_bias=50.0)
    memory, cells = memory[:1].expand(256, -1), cells[:1].expand(256, -1, -1)
    with torch.no_grad():
        # A preference over cells worth sharpening: an untrained head's is nearly flat.
        head.cell_bias.copy_(torch.randn(GRID) * 3)
        best = head(memory[:1], cells[:1], pointer_temperature=0.0)[0][0, 0, 1:] // CELLS
        hits = {}
        for t in (1.0, 0.3):
            torch.manual_seed(1)
            first = head(memory, cells, pointer_temperature=t)[0][:, 0, 1:] // CELLS
            hits[t] = (first == best).all(-1).float().mean()
        actions = head(memory[:4], cells[:4])[0]
        scores = [
            head(memory[:4], cells[:4], actions, pointer_temperature=t)[1] for t in (1.0, 0.3)
        ]
    assert hits[0.3] > hits[1.0] + 0.1, "the likeliest cell gains"
    assert torch.equal(*scores), "a demonstration's likelihood does not depend on it"


def test_a_temperature_of_zero_takes_the_likeliest_input():
    head, memory, cells = _head()
    with torch.no_grad():
        head.kinds.bias.zero_()
        head.kinds.bias[0] = 3.0  # "none" likeliest, by far.
        kinds = head(memory, cells, temperature=0.0)[0][:, :, 0]
        expected = head(memory, cells, deterministic=True)[0][:, :, 0]
    assert torch.equal(kinds, expected)


def test_the_pointer_follows_the_temperature_unless_given_its_own():
    assert resolve_temperatures() == (1.0, 1.0)
    assert resolve_temperatures(0.5) == (0.5, 0.5)
    assert resolve_temperatures(0.5, 1.0) == (0.5, 1.0)
    assert resolve_temperatures(0.7, point=True) == (0.7, 0.0)
    assert resolve_temperatures(1.0, 0, point=True) == (1.0, 0.0)
    with pytest.raises(ValueError, match="point"):
        resolve_temperatures(1.0, 0.5, point=True)
    for bad in (-0.1, float("nan")):
        with pytest.raises(ValueError, match="0 or more"):
            resolve_temperatures(bad)
        with pytest.raises(ValueError, match="0 or more"):
            resolve_temperatures(1.0, bad)


def test_an_actor_says_how_it_samples_for_the_manifest():
    actor = Actor.__new__(Actor)
    actor.temperature, actor.pointer_temperature = resolve_temperatures(0.5, point=True)
    actor.deterministic = False
    expected = {"temperature": 0.5, "pointer_temperature": 0.0, "deterministic": False}
    assert actor.sampling == expected
    assert sampling_of(actor) == expected
    assert sampling_of(object()) == {"temperature": 1.0, "pointer_temperature": 1.0}


def _episode(own, **sampled):
    steps = {"army": {"at": 5.0, "by": "policy" if own else "coach"}}
    return {"setup": {"steps": steps, "own": int(own)}, **sampled}


def test_a_practice_summary_names_the_temperatures_played_at():
    at = {"temperature": 0.5, "pointer_temperature": 0.0}
    summary = practice.summary([_episode(True, **at), _episode(False, **at)])
    assert summary["temperature"] == 0.5 and summary["pointer_temperature"] == 0.0
    mixed = practice.summary([_episode(True, **at), _episode(True)])
    assert mixed["temperature"] == [0.5, 1.0], "episodes before it was recorded played at 1"
    assert practice.summary([]) == {"episodes": 0, **{s: "0/0" for s in practice.STEPS},
                                    "own_steps_mean": None}  # fmt: skip


def _scoreboard():
    spec = importlib.util.spec_from_file_location("scoreboard", ROOT / "scripts/scoreboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_the_scoreboard_scores_each_checkpoint_at_each_temperature(tmp_path):
    board = _scoreboard()
    cold = {"temperature": 0.5, "pointer_temperature": 0.5}
    # A run from before temperatures were recorded, one at 0.5, and one pointing greedily.
    old = [_episode(True), _episode(False)]
    _write(tmp_path / "practice-bc6-e0000/practice-peer.json", {"summary": {}, "episodes": old})
    _write(
        tmp_path / "practice-bc6-e0000-2/practice-peer.json",
        {"summary": practice.summary([_episode(True, **cold)] * 3),
         "episodes": [_episode(True, **cold)] * 3},
    )  # fmt: skip
    greedy = {"temperature": 1.0, "pointer_temperature": 0.0}
    _write(
        tmp_path / "practice-bc6-e0000-3/practice-peer.json",
        {"summary": {**greedy}, "episodes": [_episode(False)]},
    )
    won = {"winner": "BLU", "started_as": "BLU", **cold}
    _write(tmp_path / "live-bc6-e0000/results-peer.json", [won, {**won, "winner": "RED"}])
    scores = board.score(tmp_path)
    assert set(scores) == {("bc6-e0000", 1.0, 1.0), ("bc6-e0000", 0.5, 0.5),
                           ("bc6-e0000", 1.0, 0.0)}  # fmt: skip
    plain, sharp = scores[("bc6-e0000", 1.0, 1.0)], scores[("bc6-e0000", 0.5, 0.5)]
    assert (plain["practice"], plain["own"]["army"], plain["live"]) == (2, 1, 0)
    assert (sharp["practice"], sharp["own"]["army"]) == (3, 3)
    assert (sharp["live"], sharp["wins"]) == (2, 1)
    assert scores[("bc6-e0000", 1.0, 0.0)]["practice"] == 1, "the run's summary names them"
    text = board.table(scores)
    assert "| bc6-e0000 | 0.5 | 0.5 | 3 | 3/3 |" in text
    assert "| bc6-e0000 | 1 | 1 | 2 | 1/2 |" in text
    rows = board.as_rows(scores)
    assert {(r["checkpoint"], r["temperature"], r["pointer_temperature"]) for r in rows} == set(
        scores
    )
    json.dumps(rows)


@pytest.mark.parametrize(
    ("command", "target", "flags", "expected"),
    [
        ("practice", "hoi4_arena.practice.practice", ["--temperature", "0.5"],
         {"temperature": 0.5, "pointer_temperature": None, "point": False}),
        ("practice", "hoi4_arena.practice.practice", ["--pointer-temperature", "0"],
         {"temperature": 1.0, "pointer_temperature": 0.0, "point": False}),
        ("play-policy", "hoi4_arena.play.evaluate_policy", ["--temperature", "0.7", "--point"],
         {"temperature": 0.7, "pointer_temperature": None, "point": True}),
    ],
)  # fmt: skip
def test_practice_and_play_policy_take_the_temperatures(
    monkeypatch, command, target, flags, expected
):
    from hoi4_arena import cli

    seen = {}
    monkeypatch.setattr(target, lambda *a, **k: seen.update(k))
    monkeypatch.setattr("hoi4_arena.models.configure_precision", lambda tf32: None)
    monkeypatch.setattr("hoi4_arena.models.limit_gpu_memory", lambda fraction: None)
    argv = ["hoi4-arena", command, "ckpt", "out", "--peer", "p.json", "--minutes", "5", *flags]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert {k: seen[k] for k in expected} == expected
