import json

import pytest
import torch
from test_critic import _Encoder, _episode, _rollout

from hoi4_arena.learning import (
    GAMMA,
    INFO_CLIP,
    adaptive_clip,
    gae,
    information_density,
    ppo_loss,
)
from hoi4_arena.models import Policy


def test_information_time_discounts_decisions_not_waiting():
    rewards, done, valid, elapsed = _episode()
    values = torch.zeros(12)
    ticks, _ = gae(rewards, values, torch.tensor(0.0), done, valid, elapsed)
    # Every step fully uncertain: information time is tick time.
    same, _ = gae(rewards, values, torch.tensor(0.0), done, valid, elapsed, density=torch.ones(12))
    assert torch.allclose(same, ticks)
    # Nine confident waits and three decisions: only the decisions discount the win.
    density = torch.zeros(12)
    density[[2, 5, 8]] = 1.0
    informed, _ = gae(rewards, values, torch.tensor(0.0), done, valid, elapsed, density=density)
    assert informed[0] == pytest.approx(GAMMA**3)
    assert ticks[0] == pytest.approx(GAMMA**11)


def test_the_density_is_normalized_over_the_whole_update():
    first, second = information_density([torch.tensor([0.5, 2.0]), torch.tensor([4.0, 0.0])])
    assert torch.equal(first, torch.tensor([0.125, 0.5])) and torch.equal(
        second, torch.tensor([1.0, 0.0])
    )
    assert information_density([torch.zeros(3)])[0].eq(0).all()


def test_the_adaptive_clip_widens_with_uncertainty_and_closes_when_certain():
    low, high = adaptive_clip(torch.tensor([0.0, 0.01, 1.0]))
    assert torch.allclose(low[0], torch.tensor(1.0)) and torch.allclose(high[0], torch.tensor(1.0))
    assert (low.diff() < 0).all() and (high.diff() > 0).all()
    assert high[2] == pytest.approx(1 + torch.log1p(torch.tensor(INFO_CLIP[1])).item())
    # A certain step cannot be pushed up; an unsure one can, up to its bound.
    logp, old = torch.tensor([0.3, 0.3], requires_grad=True), torch.zeros(2)
    ppo_loss(logp, old, None, None, torch.ones(2), torch.zeros(2), bounds=adaptive_clip(
        torch.tensor([0.0, 1.0])
    )).backward()  # fmt: skip
    assert logp.grad[0] == 0 and logp.grad[1] < 0


def test_ppo_runs_on_the_information_clock_with_the_adaptive_clip(tmp_path, monkeypatch):
    import numpy as np

    import hoi4_arena.runner as runner

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Policy(_Encoder(), memory_dim=16).to(device)
    monkeypatch.setattr(runner, "load_policy", lambda *a, **k: (policy, {"variant": "x"}, "me"))
    _rollout(tmp_path / "rollouts", policy, "me")
    kwargs = dict(epochs=1, sequence=4, burn_in=1, kl_limit=1e9, clock="information")
    with pytest.raises(ValueError, match="entropy"):
        runner.train_ppo(tmp_path / "rollouts", "c.pt", tmp_path / "old.pt", **kwargs)
    for i, path in enumerate(sorted((tmp_path / "rollouts/match/player-0").glob("*.npz"))):
        step = dict(np.load(path))
        np.savez(path, **step, old_entropy=np.float32(i % 3))
    result = runner.train_ppo(
        tmp_path / "rollouts", "c.pt", tmp_path / "out.pt", clip="adaptive", **kwargs
    )
    assert result["updates"] > 0 and (result["clock"], result["clip"]) == (
        "information",
        "adaptive",
    )
    saved = json.loads((tmp_path / "out.json").read_text())["provenance"]
    assert saved["info_clip"] == list(INFO_CLIP)
