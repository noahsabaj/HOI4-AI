import json

import numpy as np
import pytest
import torch
from test_dataset import _recording, needs_ffmpeg

from hoi4_arena.actions import SLOTS
from hoi4_arena.dataset import session_labels
from hoi4_arena.learning import (
    GAE_LAMBDA,
    GAMMA,
    RETURN_BOUND,
    critic_loss,
    gae,
    scale_return,
    value_estimate,
)
from hoi4_arena.models import Policy


def _episode(steps=12):
    """An outcome-only episode: nothing until a win at the last step, no discounting."""
    rewards = torch.zeros(steps)
    rewards[-1] = 1.0
    done = torch.zeros(steps, dtype=torch.bool)
    done[-1] = True
    return rewards, done, torch.ones(steps, dtype=torch.bool), torch.full((steps,), 0.2)


def test_lambda_one_leaves_only_each_steps_own_value_error():
    """PACT's argument for lambda = 1, on an outcome-only episode.

    A wrong value at one step moves the advantage of every earlier step when lambda is
    below one. At one, it moves only that step's own advantage.
    """
    assert GAE_LAMBDA == 1.0
    rewards, done, valid, elapsed = _episode()
    exact = torch.ones(12)  # The true value of a sure win.
    wrong = exact.clone()
    wrong[6] += 0.5
    for lam, leaks in ((1.0, False), (0.95, True)):
        clean, _ = gae(rewards, exact, torch.tensor(0.0), done, valid, elapsed, gamma=1, lam=lam)
        noisy, _ = gae(rewards, wrong, torch.tensor(0.0), done, valid, elapsed, gamma=1, lam=lam)
        moved = (noisy - clean).abs() > 1e-6
        assert moved[6], "a step's own value error always shows in its advantage"
        assert bool(moved[:6].any()) == leaks, f"lambda {lam}"


def test_the_critic_predicts_a_scaled_return_and_bce_recovers_its_mean():
    assert scale_return(torch.tensor(-RETURN_BOUND)) == 0 and scale_return(torch.tensor(0.0)) == 0.5
    assert value_estimate(torch.tensor(0.0)) == 0.0
    # Half wins, half losses: the fitted return is 0, and a 3:1 mix gives 0.5.
    for targets, expected in (
        (torch.tensor([1.0, -1.0]), 0.0),
        (torch.tensor([1.0] * 3 + [-1.0]), 0.5),
    ):
        logit = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.Adam([logit], lr=0.05)
        for _ in range(2000):
            optimizer.zero_grad()
            critic_loss(logit.expand(len(targets)), targets).mean().backward()
            optimizer.step()
        assert float(value_estimate(logit.detach())) == pytest.approx(expected, abs=1e-2)


def test_an_importance_weight_scales_the_target_the_critic_learns():
    """E[w * target] is what the weighted loss recovers, the change of measure PACT uses."""
    logit = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([logit], lr=0.05)
    target = torch.tensor([RETURN_BOUND, RETURN_BOUND])  # Scaled, both are 1.
    weight = torch.tensor([0.5, 0.9])
    for _ in range(2000):
        optimizer.zero_grad()
        critic_loss(logit.expand(2), target, weight).mean().backward()
        optimizer.step()
    assert float(torch.sigmoid(logit.detach())) == pytest.approx(0.7, abs=1e-2)


def test_a_recorded_ai_game_gives_every_decision_its_discounted_result(tmp_path):
    for name, winner, sign in (("blue", "BLU", 1), ("red", "RED", -1), ("none", None, 0)):
        _recording(tmp_path / name, [1, 1], source="ai")
        manifest = json.loads((tmp_path / name / "manifest.json").read_text())
        manifest["winner"] = winner
        (tmp_path / name / "manifest.json").write_text(json.dumps(manifest))
        outcome = session_labels(tmp_path / name, sources=("ai",))["outcome"]
        if not sign:
            assert np.isnan(outcome).all()
            continue
        # Frames end at 3.9 s and decisions start at 1.8 s: 10.5 intervals to go.
        assert outcome[0] == pytest.approx(sign * GAMMA**10.5)
        assert (np.diff(sign * outcome) > 0).all(), "closer to the end, less discounted"


class _Encoder(torch.nn.Module):
    dim = 8

    def forward(self, clip, quadrants=None):
        batch = clip.shape[0]
        pooled = clip.mean((1, 2, 3, 4))[:, None].expand(batch, self.dim)
        return pooled, pooled[:, :, None, None].expand(batch, self.dim, 14, 14)


def _rollout(root, policy, digest, steps=10):
    """One stored self-play episode for `policy`, as collect_pair writes it."""
    player = root / "match" / "player-0"
    player.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for t in range(steps):
        np.savez(
            player / f"{t:06d}.npz",
            clip=rng.integers(0, 255, (8, 16, 16, 3), dtype=np.uint8),
            quadrants=rng.integers(0, 255, (4, 32, 32, 3), dtype=np.uint8),
            fovea=rng.integers(0, 255, (16, 16, 3), dtype=np.uint8),
            speed=np.int64(4),
            hidden=np.zeros(policy.memory_dim, np.float32),
            previous=np.zeros((SLOTS, 3), np.int64),
            action=np.zeros((SLOTS, 3), np.int64),
            old_logp=np.float32(-0.5),
            old_value=np.float32(0.0),
            noise=np.zeros(16, np.float32),
            reward=np.float32(1.0 if t == steps - 1 else 0.0),
            terminal=np.bool_(t == steps - 1),
            valid=np.bool_(True),
            elapsed=np.float32(0.2),
        )
    (root / "match" / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "valid": True,
                "clip_frames": 8,
                "observation": 2,
                "checkpoint_sha256": [digest, "opponent"],
                "game_speed": 4,
            }
        )
    )


@pytest.mark.parametrize("mode", ["pact", "joint"])
def test_ppo_trains_the_critic_after_the_actor_or_with_it(tmp_path, monkeypatch, mode):
    import hoi4_arena.runner as runner

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Policy(_Encoder(), memory_dim=16).to(device)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    monkeypatch.setattr(runner, "load_policy", lambda *a, **k: (policy, {"variant": "x"}, "me"))
    _rollout(tmp_path / "rollouts", policy, "me")
    result = runner.train_ppo(
        tmp_path / "rollouts", "ckpt.pt", tmp_path / "out.pt", epochs=1, sequence=4, burn_in=1,
        kl_limit=1e9, critic=mode,
    )  # fmt: skip
    after = policy.state_dict()
    assert not torch.equal(before["value.weight"], after["value.weight"])
    assert not torch.equal(before["actor.kinds.weight"], after["actor.kinds.weight"])
    if mode == "pact":
        assert result["critic_updates"] > 0 and result["critic_ratio_kept"] > 0
    else:
        assert result["critic_updates"] == 0
    saved = json.loads((tmp_path / "out.json").read_text())["provenance"]
    assert saved["critic"] == mode and saved["gae_lambda"] == 1.0


@needs_ffmpeg
def test_critic_pretraining_moves_only_the_value_head_and_reports_the_winner(tmp_path, monkeypatch):
    from test_dataset import needs_ffmpeg

    import hoi4_arena.runner as runner
    from hoi4_arena.train import train_critic

    needs_ffmpeg(lambda: None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)
    policy = Policy(_Encoder(), memory_dim=16).to(device)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    monkeypatch.setattr(runner, "load_policy", lambda *a, **k: (policy, {"variant": "x"}, "me"))
    (tmp_path / "games").mkdir()
    _recording(tmp_path / "games" / "blue-wins", [8, 6], source="ai")
    manifest = tmp_path / "games" / "blue-wins" / "manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "winner": "BLU"}))
    report = train_critic(tmp_path / "games", "ckpt.pt", tmp_path / "critic.pt", sequence=2)
    after = policy.state_dict()
    changed = {k for k in after if not torch.equal(before[k], after[k])}
    assert changed == {"value.weight", "value.bias"}
    assert report["train_steps"] > 0 and report["validation_loss"] is None  # No validation game.
    assert json.loads((tmp_path / "critic.json").read_text())["provenance"]["parent"] == "me"
