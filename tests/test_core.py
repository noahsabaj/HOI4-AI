import io
import json
from collections import deque

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from hoi4_arena.actions import GRID, SLOTS, decode, encode_interval
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


def _rules_with(tmp_path, names, clock=True):
    """Calibrate a rules file whose templates are disjoint and mutually exclusive.

    Every template is a saturated colour in its own non-overlapping ROI, and _screen
    fills unlisted ROIs with the opposite colour. A zero-filled ROI would sit within
    add_template's max_mae=5 of a near-black template and match by accident, which would
    silently weaken every test built on this helper.
    """
    from hoi4_arena.vision import set_clock_rect

    tmp_path.mkdir(parents=True, exist_ok=True)
    screen = tmp_path / "screen.png"
    Image.new("RGB", (64, 32), (0, 0, 0)).save(screen)
    path = tmp_path / "rules.json"
    for index, name in enumerate(names):
        shot = tmp_path / f"{name}-src.png"
        base = Image.new("RGB", (64, 32), (0, 0, 0))
        base.paste(Image.new("RGB", (8, 8), _COLOURS[index]), (8 * index, 0))
        base.save(shot)
        add_template(shot, path, name, [8 * index, 0, 8, 8])
    if clock:
        set_clock_rect(screen, path, [0, 24, 8, 8])
    return ScreenRules(path)


_COLOURS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]


def _screen(rules, active):
    """Build a frame that matches exactly the named templates and no others."""
    frame = np.zeros((rules.height, rules.width, 3), np.uint8)
    for name, rule in rules.rules.items():
        x, y, w, h = rule["rect"]
        # Fill every ROI with the bitwise complement of its template so it cannot match,
        # then overwrite the ones that should.
        frame[y : y + h, x : x + w] = 255 - rules.templates[name].astype(np.uint8)
    for name in active:
        x, y, w, h = rules.rules[name]["rect"]
        frame[y : y + h, x : x + w] = rules.templates[name]
    return frame


def test_screen_helper_matches_only_the_named_templates(tmp_path):
    names = ["ready", "healthy", "running_speed_two", "win", "loss", "disconnect", "desync"]
    rules = _rules_with(tmp_path, names)
    for wanted in ([], ["healthy"], ["win"], ["healthy", "running_speed_two"]):
        frame = _screen(rules, wanted)
        matched = {name for name in names if rules.matches(name, frame)}
        assert matched == set(wanted), f"expected {set(wanted)}, matched {matched}"


def test_clock_rect_is_calibratable_and_bounds_checked(tmp_path):
    from hoi4_arena.vision import set_clock_rect

    screen = tmp_path / "screen.png"
    Image.new("RGB", (64, 32), (5, 5, 5)).save(screen)
    path = tmp_path / "rules.json"
    assert set_clock_rect(screen, path, [1, 2, 4, 4])["clock_rect"] == [1, 2, 4, 4]
    assert ScreenRules(path).clock_rect == [1, 2, 4, 4]
    set_clock_rect(screen, path, [1, 2, 4, 4])  # Idempotent re-calibration is allowed.
    with pytest.raises(ValueError, match="remove the old clock calibration"):
        set_clock_rect(screen, path, [0, 0, 4, 4])
    with pytest.raises(ValueError, match="Clock rectangle outside"):
        set_clock_rect(screen, tmp_path / "other.json", [60, 0, 16, 4])


def test_require_match_rules_is_satisfiable_from_calibration_alone(tmp_path):
    names = ["ready", "healthy", "running_speed_two", "win", "loss", "disconnect", "desync"]
    _rules_with(tmp_path, names).require_match_rules()
    with pytest.raises(ValueError, match="clock_rect"):
        _rules_with(tmp_path / "no-clock", names, clock=False).require_match_rules()


def test_unhealthy_screen_cannot_buy_unbounded_grace_from_a_flickering_outcome(tmp_path):
    from unittest.mock import Mock

    from hoi4_arena.desktop import Frame
    from hoi4_arena.environment import ArenaEnv
    from hoi4_arena.vision import TERMINAL_GRACE_FRAMES

    rules = _rules_with(
        tmp_path, ["ready", "healthy", "running_speed_two", "win", "loss", "disconnect", "desync"]
    )
    # 'win' and 'loss' alternate, so ScreenRules.outcome never reaches its debounce and
    # rules.last is never None -- the exact state that used to skip every liveness gate.
    frames = [_screen(rules, ["win" if i % 2 == 0 else "loss"]) for i in range(40)]
    desktop = Mock()
    desktop.capture.side_effect = [
        Frame(f, {"seq": i}, i) for i, f in enumerate([_screen(rules, ["ready"])] + frames)
    ]
    desktop.apply.return_value = {"applied": 1}
    env = ArenaEnv(desktop, rules, [])
    env.reset()
    action = np.zeros((SLOTS, 3), dtype=np.int64)
    infos = []
    for _ in range(TERMINAL_GRACE_FRAMES + 2):
        infos.append(env.step(action)[4])
        if not infos[-1]["valid"]:
            break
    assert not infos[-1]["valid"], "an unhealthy flickering screen must invalidate the episode"
    assert "terminal_screen_never_confirmed" in infos[-1]["error"]
    # Upper bound: the grace is finite. The lower bound is pinned separately by
    # test_legitimate_terminal_screen_is_confirmed_through_its_transition_frames, so a
    # grace of zero cannot pass both.
    assert len(infos) <= TERMINAL_GRACE_FRAMES + 1
    assert not env.active


def test_step_invalidates_on_screenrules_value_errors_instead_of_escaping(tmp_path):
    from unittest.mock import Mock

    from hoi4_arena.desktop import Frame
    from hoi4_arena.environment import ArenaEnv

    rules = _rules_with(
        tmp_path, ["ready", "healthy", "running_speed_two", "win", "loss", "disconnect", "desync"]
    )
    desktop = Mock()
    # reset() captures twice: run_setup's final ready check, then the first observation.
    # A frame at the wrong resolution makes ScreenRules raise ValueError, not DesktopError.
    desktop.capture.side_effect = [
        Frame(_screen(rules, ["ready"]), {}, 0),
        Frame(_screen(rules, ["ready"]), {}, 1),
        Frame(np.zeros((8, 8, 3), np.uint8), {}, 2),
    ]
    desktop.apply.return_value = {}
    env = ArenaEnv(desktop, rules, [])
    env.reset()
    info = env.step(np.zeros((SLOTS, 3), dtype=np.int64))[4]
    assert not info["valid"] and info["outcome"] == "invalid"
    assert not env.active


def test_pair_reset_joins_both_sides_before_disarming():
    import threading
    from unittest.mock import Mock

    started = threading.Event()
    finished = threading.Event()

    def slow_reset(**_):
        started.set()
        finished.wait(5)
        return "ok", {}

    first, second = Mock(), Mock()
    first.reset.side_effect = RuntimeError("left recipe failed")
    second.reset.side_effect = slow_reset
    pair = ArenaPair(first, second)
    try:
        raised = None

        def run():
            nonlocal raised
            try:
                pair.reset()
            except Exception as error:
                raised = error

        thread = threading.Thread(target=run)
        thread.start()
        assert started.wait(5)
        # The slow side is still inside its recipe; reset must not have returned yet.
        thread.join(0.2)
        assert thread.is_alive(), "reset returned while a setup recipe was still running"
        finished.set()
        thread.join(5)
        assert isinstance(raised, RuntimeError)
        second.desktop.release.assert_called()
    finally:
        finished.set()
        pair.pool.shutdown(wait=True)


def test_deterministic_actor_takes_the_argmax():
    torch.manual_seed(0)
    actor = ActionHead(memory_dim=8)
    memory = torch.randn(2, 8)
    greedy = [actor(memory, deterministic=True)[0] for _ in range(3)]
    assert all(torch.equal(greedy[0], other) for other in greedy[1:])
    sampled = torch.stack([actor(memory)[0] for _ in range(12)])
    assert not torch.equal(sampled[0], sampled[-1]), "sampling must still be stochastic"


def test_legitimate_terminal_screen_is_confirmed_through_its_transition_frames(tmp_path):
    """Lower bound on the grace window: a real win must survive its transition frames.

    Pairs with the flicker test above, which pins the upper bound. A grace of zero passes
    that one and fails this one; an unbounded grace does the reverse.
    """
    from unittest.mock import Mock

    from hoi4_arena.desktop import Frame
    from hoi4_arena.environment import ArenaEnv
    from hoi4_arena.vision import OUTCOME_FRAMES

    rules = _rules_with(
        tmp_path, ["ready", "healthy", "running_speed_two", "win", "loss", "disconnect", "desync"]
    )
    # The HUD vanishes two frames before the victory panel renders, then it debounces.
    sequence = [[], []] + [["win"]] * OUTCOME_FRAMES
    desktop = Mock()
    desktop.capture.side_effect = [
        Frame(_screen(rules, ["ready", "healthy", "running_speed_two"]), {}, 0),
        Frame(_screen(rules, ["healthy", "running_speed_two"]), {}, 1),
        *[Frame(_screen(rules, active), {}, 2 + i) for i, active in enumerate(sequence)],
    ]
    desktop.apply.return_value = {}
    env = ArenaEnv(desktop, rules, [])
    env.reset()
    action = np.zeros((SLOTS, 3), dtype=np.int64)
    for index in range(len(sequence)):
        _, reward, done, _, info = env.step(action)
        assert info["valid"], f"frame {index} of a legitimate victory was invalidated"
        if done:
            assert info["outcome"] == "win" and reward == 1.0
            return
    raise AssertionError("a legitimate victory was never confirmed")


def test_act_noise_pins_the_latent_when_deterministic():
    from hoi4_arena.runner import act_noise

    # Taking the argmax is not enough: ActionHead conditions its start state on this
    # latent, so an xm actor that keeps drawing one stays stochastic.
    assert torch.equal(act_noise("xm", 16, True, "cpu"), torch.zeros(1, 16))
    assert torch.equal(act_noise("bc", 16, False, "cpu"), torch.zeros(1, 16))
    torch.manual_seed(0)
    draws = torch.stack([act_noise("xm", 16, False, "cpu") for _ in range(4)])
    assert not torch.equal(draws[0], draws[1]), "xm sampling must stay stochastic"


def test_actor_threads_deterministic_into_the_policy_and_is_reproducible():
    """Covers runner.Actor, not models.ActionHead, which already honored the flag."""
    from hoi4_arena.runner import Actor

    class TinyPolicy(torch.nn.Module):
        memory_dim = 8

        def __init__(self):
            super().__init__()
            self.actor = ActionHead(memory_dim=8)
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, clip, tiles, previous, hidden=None):
            batch = clip.shape[0]
            hidden = self.linear(clip.float().mean((1, 2, 3, 4))[:, None].expand(batch, 8))
            return hidden, hidden.sum(-1), hidden

    def build(deterministic, objective, stream):
        actor = Actor.__new__(Actor)
        torch.manual_seed(7)  # Identical weights every time...
        actor.policy = TinyPolicy().eval().requires_grad_(False)
        torch.manual_seed(stream)  # ...but a different RNG stream for sampling.
        actor.config = {"objective": objective}
        actor.deterministic = deterministic
        actor.device = "cpu"
        actor.hidden = None
        actor.previous = np.zeros((SLOTS, 3), dtype=np.int64)
        actor.history = deque(maxlen=64)
        return actor

    rgb = np.full((32, 32, 3), 120, np.uint8)
    # Same weights, same observation, different RNG streams: a deterministic actor must
    # be unaffected by the stream, for both objectives. This fails if either the argmax
    # or the pinned latent is missing.
    for objective in ("bc", "xm"):
        greedy = [build(True, objective, s).act(rgb, 10.0)[0] for s in range(4)]
        assert all(np.array_equal(greedy[0], other) for other in greedy[1:]), (
            f"deterministic actor is not reproducible for objective={objective}"
        )
    sampled = [build(False, "bc", s).act(rgb, 10.0)[0] for s in range(8)]
    assert any(not np.array_equal(sampled[0], other) for other in sampled[1:]), (
        "a non-deterministic actor must still sample"
    )


def test_deterministic_action_equals_the_head_argmax():
    torch.manual_seed(3)
    actor = ActionHead(memory_dim=8)
    memory = torch.randn(2, 8)
    noise = torch.zeros(2, actor.noise_dim)
    action, _, _ = actor(memory, noise=noise, deterministic=True)
    # Recompute the head distributions by hand and check the greedy path took the mode.
    state = torch.tanh(actor.init(torch.cat([memory, noise], -1)))
    previous = torch.zeros(2, 64)
    for slot in range(SLOTS):
        state = actor.cell(previous, state)
        expected_type = actor.type_head(state).float().argmax(-1)
        assert torch.equal(action[:, slot, 0], expected_type)
        expected = torch.stack(
            [expected_type, actor.x_head(state).argmax(-1), actor.y_head(state).argmax(-1)], -1
        )
        move = (expected_type == 1).long()
        expected = expected.clone()
        expected[:, 1:] *= move[:, None]
        assert torch.equal(action[:, slot], expected)
        previous = actor.embedding(expected[:, 0]) + actor.xy(expected[:, 1:].float() / (GRID - 1))


def test_ppo_excludes_deterministic_evaluation_rollouts():
    from hoi4_arena.runner import ppo_exclusion

    on_policy = {"complete": True, "valid": True, "deterministic": False}
    assert ppo_exclusion(on_policy) is None
    assert ppo_exclusion({**on_policy, "deterministic": True}) is not None
    assert ppo_exclusion({**on_policy, "valid": False}) is not None
    assert ppo_exclusion({**on_policy, "complete": False}) is not None
    # A manifest written before the deterministic key existed is still on-policy.
    assert ppo_exclusion({"complete": True, "valid": True}) is None


def test_clock_rect_must_be_integral(tmp_path):
    from hoi4_arena.vision import set_clock_rect

    screen = tmp_path / "screen.png"
    Image.new("RGB", (64, 32), (5, 5, 5)).save(screen)
    path = tmp_path / "rules.json"
    set_clock_rect(screen, path, [1, 2, 4, 4])
    spec = json.loads(path.read_text())
    spec["clock_rect"] = [1.0, 2.0, 4.0, 4.0]
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="four integers"):
        ScreenRules(path)


def test_close_records_release_failure_instead_of_raising_from_exit():
    from unittest.mock import Mock

    from hoi4_arena.desktop import Desktop

    desktop = Desktop.__new__(Desktop)
    desktop.close_error = None
    desktop.process = Mock()
    desktop.process.poll.return_value = None
    desktop.release = Mock(side_effect=DesktopError("worker gone"))
    desktop._shutdown = Mock()
    desktop.close()
    desktop._shutdown.assert_called_once()
    assert "worker gone" in desktop.close_error, "collect_pair needs this to invalidate the run"


def test_seed_is_reproducible_but_salted_per_match():
    from hoi4_arena.runner import seed_everything

    assert seed_everything(42, salt="pair-1") == seed_everything(42, salt="pair-1")
    assert seed_everything(42, salt="pair-1") != seed_everything(42, salt="pair-2")
    assert seed_everything(42) == 42


def test_screen_matching_nothing_is_rejected_fast_not_on_the_terminal_budget(tmp_path):
    """An unknown screen has nothing in flight, so it uses the short budget."""
    from unittest.mock import Mock

    from hoi4_arena.desktop import Frame
    from hoi4_arena.environment import ArenaEnv
    from hoi4_arena.vision import TERMINAL_GRACE_FRAMES, UNKNOWN_FRAMES

    rules = _rules_with(
        tmp_path, ["ready", "healthy", "running_speed_two", "win", "loss", "disconnect", "desync"]
    )
    blank = _screen(rules, [])
    desktop = Mock()
    desktop.capture.side_effect = [
        Frame(_screen(rules, ["ready", "healthy", "running_speed_two"]), {}, 0),
        Frame(_screen(rules, ["healthy", "running_speed_two"]), {}, 1),
        *[Frame(blank, {}, 2 + i) for i in range(TERMINAL_GRACE_FRAMES + 4)],
    ]
    desktop.apply.return_value = {}
    env = ArenaEnv(desktop, rules, [])
    env.reset()
    action = np.zeros((SLOTS, 3), dtype=np.int64)
    steps = 0
    while True:
        steps += 1
        info = env.step(action)[4]
        if not info["valid"]:
            break
    assert "unrecognized_match_screen" in info["error"]
    assert steps <= UNKNOWN_FRAMES + 1, (
        f"an unknown screen took {steps} steps; it must not consume the terminal budget"
    )
