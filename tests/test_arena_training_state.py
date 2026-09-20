"""Training plumbing on synthetic fixtures; never a real-game strength claim."""
# ruff: noqa: E402 -- skip optional Torch before importing consumers.
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hoi4_agent.arena.contracts import ArenaError, Country, OrderReceipt
from hoi4_agent.arena.evaluation import EvaluationGame, improvement_report
from hoi4_agent.arena.league import League
from hoi4_agent.arena.learner import PPOConfig, PPOLearner
from hoi4_agent.arena.policy import RecurrentPolicy
from hoi4_agent.arena.trajectory import Provenance, Trajectory, Transition, validate_demonstrations
from test_arena_bridge import fingerprint, observation


def fixture_episode(policy: RecurrentPolicy) -> Trajectory:
    before = observation()
    memory = None
    transitions = []
    with torch.no_grad():
        for i in range(3):
            output = policy(before, memory)
            memory = output.memory
            action_index = torch.tensor(i % len(output.actions))
            order = output.actions[int(action_index)].order(before, f"action-{i}")
            after = replace(before, sequence=i + 1, game_hour=(i + 1) * 2,
                            terminal=i == 2, winner=Country.BLUE if i == 2 else None)
            transitions.append(Transition(before, order,
                OrderReceipt(order.id, before.episode_id, True, "fixture", before.game_hour), after,
                float(output.distribution.log_prob(action_index)), float(output.value), 1))
            before = after
    provenance = Provenance("fixture", "scenario-a", "train", 4, fingerprint(), "d" * 64,
                            "policy-initial", "fixture-opponent")
    return Trajectory(provenance, tuple(transitions))


def test_ppo_fixture_boundary_and_exact_checkpoint_resume(tmp_path: Path) -> None:
    torch.manual_seed(1)
    torch.set_num_threads(1)
    learner = PPOLearner(RecurrentPolicy(32), PPOConfig(epochs=2, sequence_length=2))
    episode = fixture_episode(learner.policy)
    with pytest.raises(ArenaError, match="real HOI4"):
        learner.update([episode])
    original = learner.policy.value_head.weight.detach().clone()
    metrics = learner.update([episode], unit_test_fixture=True)
    assert metrics["transitions"] == 3
    assert not torch.equal(original, learner.policy.value_head.weight)
    saved = tmp_path / "checkpoint.pt"
    learner.save(saved, fingerprint(), {"test": "league"}, {"source": "fixture"})
    resumed, metadata = PPOLearner.resume(saved, fingerprint())
    assert metadata["provenance"] == {"source": "fixture"}
    assert resumed.updates == learner.updates
    fresh_episode = fixture_episode(learner.policy)
    learner.update([fresh_episode], unit_test_fixture=True)
    resumed.update([fresh_episode], unit_test_fixture=True)
    for key, value in learner.policy.state_dict().items():
        torch.testing.assert_close(value, resumed.policy.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ArenaError, match="incompatible"):
        PPOLearner.resume(saved, replace(fingerprint(), mod_sha256="f" * 64))


def test_trajectory_roundtrip_and_dataset_leakage(tmp_path: Path) -> None:
    episode = fixture_episode(RecurrentPolicy(32))
    path = tmp_path / "fixture.json"
    episode.save(path)
    assert Trajectory.load(path) == episode
    import json
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert len(stored["observations"]) == len(stored["transitions"]) + 1  # each observation stored once
    assert "observation" not in stored["transitions"][0]
    with pytest.raises(ArenaError, match="real-engine"):
        validate_demonstrations([episode], {"scenario-a": "train"})
    # Metadata branch exercise only: this test does not create live evidence.
    metadata = replace(episode.provenance, source="hoi4_vision", demonstration_source="human")
    demonstration = replace(episode, provenance=metadata)
    assert validate_demonstrations([demonstration], {"scenario-a": "train"})["transitions_by_split"] == {"train": 3}
    leaked = replace(demonstration, provenance=replace(metadata, scenario_id="scenario-b", scenario_split="validation"))
    with pytest.raises(ArenaError, match="multiple dataset splits"):
        validate_demonstrations([demonstration, leaked], {"scenario-a": "train", "scenario-b": "validation"})


def test_frozen_opponents_and_resumable_sampling(tmp_path: Path) -> None:
    source = tmp_path / "latest.pt"
    source.write_bytes(b"fixture checkpoint")
    league = League(1)
    identifier = league.freeze(source, tmp_path / "frozen", champion=True)
    league.add_scripted("hold-v1")
    league.add_scripted("advance-v1")
    league.set_evaluation_panel((identifier, "hold-v1"))
    with pytest.raises(ArenaError, match="only once"):
        league.set_evaluation_panel((identifier,))
    source.write_bytes(b"a later checkpoint")
    snapshot = league.sample_episode()
    assert snapshot.id == identifier
    league.record(identifier, "win")
    assert snapshot.wins == 0
    resumed = League.from_state_dict(league.state_dict())
    assert [league.sample_episode().id for _ in range(30)] == [resumed.sample_episode().id for _ in range(30)]
    assert Path(league.opponents[identifier].checkpoint).read_bytes() == b"fixture checkpoint"


def test_paired_evaluation_rejects_unbalanced_and_fixture_evidence() -> None:
    fixture = EvaluationGame("candidate", 0, "pair", "BLU", "a", "fixed", "win", "fixture")
    with pytest.raises(ArenaError, match="real HOI4"):
        improvement_report([fixture])
    with pytest.raises(ArenaError, match="both policies"):
        improvement_report([replace(fixture, source="hoi4_vision")])
    rows = []
    for seed in range(3):
        for block in range(50):
            for side in ("BLU", "RED"):
                for candidate in ("initialization", "candidate"):
                    # Deterministic synthetic statistics oracle: +0.5 score.
                    outcome = "draw" if candidate == "initialization" else "win"
                    rows.append(EvaluationGame(candidate, seed, str(block), side, "a", "fixed", outcome))
    result = improvement_report(rows)
    assert result["three_seed_statistical_gate"]
    assert all(row["score_improvement"] == 0.5 for row in result["per_seed"])
    assert result["acceptance"].startswith("requires separate")
