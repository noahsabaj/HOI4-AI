import io

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from hoi4_arena.actions import SLOTS, decode, encode_interval
from hoi4_arena.desktop import DesktopError, read_reply
from hoi4_arena.environment import ArenaPair
from hoi4_arena.learning import League, gae, paired_evaluation, ppo_loss, save_checkpoint
from hoi4_arena.models import ActionHead, PredictiveAuxiliary, rdmreg, reprelu
from hoi4_arena.recording import split_for_session
from hoi4_arena.vision import ScreenRules, add_template


def test_drag_edges_preserve_order_when_events_collide():
    events = [
        {"kind": "move", "x": 0.1, "y": 0.2},
        {"kind": "button", "button": 0, "down": True},
        {"kind": "move", "x": 0.8, "y": 0.9},
        {"kind": "button", "button": 0, "down": False},
    ]
    encoded = encode_interval([{"t_ns": i * 1000000, "event": e} for i, e in enumerate(events)], 0)
    decoded = [e for t in encoded for e in decode(t)]
    assert [e["kind"] for e in decoded] == ["move", "button", "move", "button"]
    assert decoded[1]["down"] and not decoded[3]["down"]
    assert decoded[2]["x"] == pytest.approx(0.8, abs=0.001)


def test_input_overflow_and_unsupported_keys_are_quarantined():
    with pytest.raises(ValueError, match="eight"):
        encode_interval(
            [
                {"t_ns": i, "event": {"kind": "button", "button": 0, "down": bool(i % 2)}}
                for i in range(9)
            ],
            0,
        )
    with pytest.raises(ValueError, match="unsupported"):
        encode_interval([{"t_ns": 0, "event": {"kind": "key", "vk": 0x20, "down": True}}], 0)


def test_framed_transport_handles_binary_newlines_and_truncation():
    reply = read_reply(io.BytesIO(b'{"bytes":4}\n\x00\n\xff\x03'))
    assert reply["payload"] == b"\x00\n\xff\x03"
    with pytest.raises(DesktopError, match="Truncated"):
        read_reply(io.BytesIO(b'{"bytes":5}\n123'))
    with pytest.raises(DesktopError, match="Invalid"):
        read_reply(io.BytesIO(b'{"bytes":-1}\n'))


def test_actor_likelihood_replays_with_same_latent_and_ignores_inactive_xy():
    torch.manual_seed(1)
    actor = ActionHead(memory_dim=16)
    memory = torch.randn(2, 16)
    noise = torch.randn(2, 16)
    action, old, _ = actor(memory, noise=noise)
    _, new, entropy = actor(memory, action, noise)
    assert torch.allclose(old, new)
    assert torch.isfinite(entropy).all()
    assert (action[:, :, 1:][action[:, :, 0] != 1] == 0).all()
    (-new.mean()).backward()
    assert actor.init.weight.grad.abs().sum() > 0


def test_reprelu_is_relu_with_gelu_gradient():
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = reprelu(x)
    assert torch.allclose(y, x.relu())
    y.sum().backward()
    other = x.detach().requires_grad_()
    torch.nn.functional.gelu(other).sum().backward()
    assert torch.allclose(x.grad, other.grad)


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_auxiliary_propagates_to_memory_features_and_predictor(mode):
    aux = PredictiveAuxiliary(memory_dim=16, feature_dim=8, latent_dim=12, mode=mode)
    memories = torch.randn(2, 3, 16, requires_grad=True)
    features = torch.randn(2, 3, 8, requires_grad=True)
    loss = aux(
        memories,
        features,
        torch.zeros(2, 3, SLOTS, 3, dtype=torch.long),
        torch.ones(2, 3, dtype=torch.bool),
    )
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert memories.grad.abs().sum() > 0 and features.grad.abs().sum() > 0
    assert aux.predictor[0].weight.grad.abs().sum() > 0


def test_rdm_needs_independent_batch_samples():
    with pytest.raises(ValueError, match="independent"):
        rdmreg(torch.randn(1, 4, 8), True)


def test_gae_terminal_never_bootstraps_and_invalid_episode_rejected():
    reward = torch.tensor([0.0, 1.0])
    values = torch.zeros(2)
    valid = torch.ones(2, dtype=torch.bool)
    done = torch.tensor([False, True])
    advantage, returns = gae(
        reward, values, torch.tensor(999.0), done, valid, torch.tensor([0.2, 0.2]), gamma=0.9, lam=1
    )
    assert torch.allclose(returns, torch.tensor([0.9, 1.0]))
    valid[0] = False
    with pytest.raises(ValueError, match="Invalid"):
        gae(reward, values, torch.tensor(0.0), done, valid, torch.ones(2))


def test_ppo_has_finite_gradients():
    lp = torch.tensor([-0.5, -1.0], requires_grad=True)
    value = torch.tensor([0.1, 0.2], requires_grad=True)
    loss = ppo_loss(
        lp, lp.detach(), value, torch.tensor([0.0, 1.0]), torch.tensor([-0.3, 0.3]), torch.ones(2)
    )
    loss.backward()
    assert torch.isfinite(lp.grad).all() and torch.isfinite(value.grad).all()


def test_checkpoint_immutability_and_league_integrity(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, nn.Linear(2, 2), {})
    with pytest.raises(FileExistsError):
        save_checkpoint(path, nn.Linear(2, 2), {})
    league = League(tmp_path / "league.json")
    league.add(path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        league.sample()


def test_pairs_exclude_incomplete_or_invalid_results():
    rows = [
        {
            "pair_id": i,
            "side": side,
            "valid": i != 2,
            "scenario": "arena",
            "outcome": "win" if side == "left" else "loss",
        }
        for i in range(3)
        for side in ["left", "right"]
    ]
    rows.append(
        {"pair_id": 3, "side": "left", "valid": True, "scenario": "arena", "outcome": "win"}
    )
    report = paired_evaluation(rows)
    assert report["pairs"] == 2 and report["excluded_pairs"] == 2 and report["score"] == 0.5
    assert not report["acceptance_sample_complete"]


def test_visual_outcomes_require_repeated_screen_evidence(tmp_path):
    screen = tmp_path / "screen.png"
    Image.new("RGB", (64, 32), (200, 10, 10)).save(screen)
    path = tmp_path / "rules.json"
    add_template(screen, path, "win", [0, 0, 10, 10])
    rules = ScreenRules(path)
    rgb = np.asarray(Image.open(screen))
    assert rules.outcome(rgb) is None
    assert rules.outcome(rgb) is None
    assert rules.outcome(rgb) == "win"
    assert rules.outcome(np.zeros_like(rgb)) is None
    with pytest.raises(ValueError, match="resolution"):
        rules.matches("win", rgb[:10])


def test_split_is_stable_for_whole_sessions():
    assert split_for_session("a") == split_for_session("a")
    assert set(split_for_session(str(i)) for i in range(1000)) == {"train", "validation", "test"}


def test_pair_cleanup_releases_other_worker_even_if_first_fails():
    from unittest.mock import Mock

    first, second = Mock(), Mock()
    first.close.side_effect = OSError("disconnected")
    pair = ArenaPair(first, second)
    with pytest.raises(OSError, match="disconnected"):
        pair.close()
    second.close.assert_called_once()


def test_all_wins_still_have_sampling_uncertainty():
    report = paired_evaluation(
        [
            {"pair_id": i, "side": side, "valid": True, "scenario": "arena", "outcome": "win"}
            for i in range(50)
            for side in ["left", "right"]
        ]
    )
    assert report["win_rate"] == 1.0
    assert 0 < report["win_rate_ci95"][0] < 1
