import io
import json
import re
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


# Identical constants to crates/desktop-worker/src/main.rs::downscale_matches_the_training_resize.
# Both sides assert the same bytes, so a filter change in either language fails a test here.
_DOWNSCALE_GOLDEN = [
    (
        12,
        8,
        3,
        [
            124,
            133,
            134,
            126,
            104,
            79,
            159,
            144,
            150,
            144,
            135,
            132,
            126,
            131,
            128,
            167,
            142,
            133,
            112,
            146,
            125,
            119,
            129,
            154,
            148,
            117,
            118,
        ],
    ),
    (
        7,
        5,
        3,
        [
            96,
            103,
            140,
            172,
            105,
            110,
            210,
            58,
            109,
            94,
            130,
            129,
            149,
            110,
            99,
            162,
            116,
            141,
            144,
            159,
            102,
            149,
            138,
            164,
            151,
            186,
            218,
        ],
    ),
    (
        16,
        9,
        4,
        [
            116,
            167,
            123,
            160,
            96,
            116,
            145,
            123,
            130,
            154,
            106,
            105,
            144,
            137,
            177,
            162,
            130,
            107,
            124,
            149,
            104,
            132,
            117,
            128,
            144,
            122,
            167,
            135,
            137,
            98,
            137,
            149,
            142,
            125,
            122,
            109,
            135,
            121,
            158,
            121,
            137,
            112,
            164,
            142,
            117,
            131,
            142,
            133,
        ],
    ),
]


def _lcg(n):
    out, s = np.empty(n, np.uint8), 12345
    for i in range(n):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        out[i] = (s >> 16) & 0xFF
    return out


def test_worker_downscale_matches_training_resize():
    """The worker downscales before transport; it must produce the training pixels.

    A plain box filter over fractional boundaries differs from this by a mean of 11/255,
    and unfiltered bilinear by 72/255. Either would silently feed the policy different
    pixels at deployment than it saw during training.
    """
    import torch.nn.functional as F

    for w, h, size, expected in _DOWNSCALE_GOLDEN:
        rgb = _lcg(w * h * 4).reshape(h, w, 4)[:, :, [2, 1, 0]].astype(np.float32)
        t = torch.as_tensor(rgb).permute(2, 0, 1)[None]
        out = F.interpolate(t, (size, size), mode="area").round().clamp(0, 255).to(torch.uint8)
        got = out[0].permute(1, 2, 0).numpy().ravel().tolist()
        assert got == expected, f"{w}x{h}->{size} drifted from the worker's golden vector"


def test_views_uses_the_pinned_resampler_and_tiles_the_frame():
    from hoi4_arena.dataset import quadrants, views

    # Same source bytes as the golden case: BGRA from the LCG, swizzled to RGB.
    rgb = _lcg(16 * 9 * 4).reshape(9, 16, 4)[:, :, [2, 1, 0]].copy()
    g, tiles = views(rgb, size=4)
    assert g.shape == (4, 4, 3) and tiles.shape == (4, 4, 4, 3)
    assert g.dtype == torch.uint8
    # The global view of this frame is the third golden case, computed the same way.
    _, _, _, expected = _DOWNSCALE_GOLDEN[2]
    assert g.numpy().ravel().tolist() == expected
    assert sum(bh * bw for _, _, bh, bw in quadrants(9, 16)) == 9 * 16


def test_observation_from_worker_crops_answers_exactly_like_a_full_frame(tmp_path):
    """The two capture paths must be indistinguishable to the match loop."""
    from hoi4_arena.desktop import Frame

    names = ["ready", "healthy", "running_speed_two", "win", "loss", "disconnect", "desync"]
    rules = _rules_with(tmp_path, names)
    order, rects = rules.capture_regions()
    assert order == sorted(names) and len(rects) == len(names) + 1  # + clock_rect

    for active in ([], ["healthy"], ["win"], ["healthy", "running_speed_two"], ["win", "loss"]):
        rgb = _screen(rules, active)
        crops = [rgb[y : y + h, x : x + w].copy() for x, y, w, h in rects]
        full = rules.observe(Frame(rgb, {}, 0))
        cropped = rules.observe(Frame(None, {}, 0, crops=crops))
        assert [full.matches(n) for n in names] == [cropped.matches(n) for n in names]
        assert np.array_equal(full.clock_pixels(), cropped.clock_pixels())
        rules.last, rules.count = None, 0
        a = [rules.observe(Frame(rgb, {}, 0)).outcome() for _ in range(4)]
        rules.last, rules.count = None, 0
        b = [rules.observe(Frame(None, {}, 0, crops=crops)).outcome() for _ in range(4)]
        assert a == b, f"debounced outcome differs between capture paths for {active}"


def test_observation_rejects_a_capture_it_cannot_read(tmp_path):
    from hoi4_arena.desktop import Frame

    rules = _rules_with(
        tmp_path, ["ready", "healthy", "running_speed_two", "win", "loss", "disconnect", "desync"]
    )
    with pytest.raises(ValueError, match="neither a full frame nor calibrated crops"):
        rules.observe(Frame(None, {}, 0))
    _, rects = rules.capture_regions()
    with pytest.raises(ValueError, match="different region set"):
        rules.observe(Frame(None, {}, 0, crops=[np.zeros((2, 2, 3), np.uint8)] * (len(rects) - 1)))


def test_capture_splits_a_downscaled_worker_payload():
    """Protocol layout: full frame, then views, then region crops, in request order."""
    from unittest.mock import Mock

    from hoi4_arena.desktop import Desktop

    size, regions = 4, [[1, 2, 3, 2], [0, 0, 2, 2]]
    views_block = _lcg(5 * size * size * 3)
    crop_blocks = [_lcg(3 * 2 * 4), _lcg(2 * 2 * 4)]
    payload = bytes(views_block) + b"".join(bytes(c) for c in crop_blocks)

    desktop = Desktop.__new__(Desktop)
    desktop.request = Mock(
        return_value={
            "height": 8,
            "width": 8,
            "encoding": "raw",
            "overflow": False,
            "stopped": False,
            "full_bytes": 0,
            "view_size": size,
            "views_bytes": len(views_block),
            "region_bytes": [len(c) for c in crop_blocks],
            "payload": payload,
        }
    )
    frame = desktop.capture(views=size, regions=regions)
    assert frame.rgb is None, "a views-only capture must not carry the full frame"
    g, tiles = frame.views
    assert g.shape == (size, size, 3) and tiles.shape == (4, size, size, 3)
    assert np.array_equal(g.ravel(), views_block[: size * size * 3])
    assert [c.shape for c in frame.crops] == [(2, 3, 3), (2, 2, 3)]
    # Crops arrive BGRA and must be swizzled to RGB like the full frame is.
    assert np.array_equal(frame.crops[0], crop_blocks[0].reshape(2, 3, 4)[:, :, [2, 1, 0]])
    assert desktop.request.call_args.kwargs["regions"] == regions


def test_capture_rejects_a_payload_that_contradicts_its_header():
    from unittest.mock import Mock

    from hoi4_arena.desktop import Desktop

    desktop = Desktop.__new__(Desktop)
    desktop.request = Mock(
        return_value={
            "height": 4,
            "width": 4,
            "encoding": "raw",
            "overflow": False,
            "stopped": False,
            "full_bytes": 0,
            "view_size": 2,
            "views_bytes": 5 * 2 * 2 * 3,
            "region_bytes": [],
            "payload": b"\x00" * 7,
        }
    )
    with pytest.raises(DesktopError, match="declared layout"):
        desktop.capture(views=2)


def test_recorder_records_the_pixels_the_policy_saw_when_the_worker_downscaled():
    from hoi4_arena.desktop import Frame
    from hoi4_arena.recording import audit_pixels

    g = np.full((224, 224, 3), 7, np.uint8)
    downscaled = Frame(
        None, {"foreground": True}, 0, views=(g, np.zeros((4, 224, 224, 3), np.uint8))
    )
    assert np.array_equal(audit_pixels(downscaled), g)
    native = np.full((8, 8, 3), 3, np.uint8)
    assert np.array_equal(audit_pixels(Frame(native, {}, 0)), native)
    with pytest.raises(ValueError, match="no pixels"):
        audit_pixels(Frame(None, {}, 0))


def test_views_rounds_ties_to_even_like_the_worker():
    """Every channel here averages exactly x.5; half-away-from-zero would give 1,3,5."""
    from hoi4_arena.dataset import views

    bgra = np.array([[[4, 2, 0, 0], [5, 3, 1, 0]]], np.uint8)
    g, _ = views(bgra[:, :, [2, 1, 0]].copy(), size=1)
    assert g.numpy().ravel().tolist() == [0, 2, 4]


def test_views_orders_quadrants_top_left_top_right_bottom_left_bottom_right():
    """Each quadrant is a distinct constant, so any permutation changes the bytes."""
    from hoi4_arena.dataset import views

    src = np.zeros((4, 4, 3), np.uint8)
    for idx, (top, left) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)]):
        src[top : top + 2, left : left + 2] = [
            10 * (idx + 1),
            10 * (idx + 1) + 1,
            10 * (idx + 1) + 2,
        ]
    g, tiles = views(src, size=1)
    assert g.numpy().ravel().tolist() == [25, 26, 27]
    assert tiles.reshape(4, 3).numpy().tolist() == [
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32],
        [40, 41, 42],
    ]


def test_views_accumulates_in_float32_so_the_worker_can_match_it():
    """A 32x reduction averages ~1024 values; float16 cannot hold that sum exactly.

    The worker sums in u32 and divides in f32. If this side accumulated in half
    precision the two would disagree on a real 4K frame, which no small golden vector
    would reveal.
    """
    from hoi4_arena.dataset import views

    rng = np.random.default_rng(7)
    big = rng.integers(0, 256, (288, 512, 3), dtype=np.uint8)
    g, _ = views(big, size=16)
    assert int(g.sum()) == 97902, "resize no longer accumulates in float32"


def _cadence_env(tmp_path, capture_ms=0.0):
    """A match environment over a fake desktop whose capture costs `capture_ms`."""
    import time as _time
    from unittest.mock import Mock

    from hoi4_arena.desktop import Frame
    from hoi4_arena.environment import ArenaEnv

    rules = _rules_with(
        tmp_path, ["ready", "healthy", "running_speed_two", "win", "loss", "disconnect", "desync"]
    )
    live = _screen(rules, ["ready", "healthy", "running_speed_two"])
    ticking = [0]

    def capture(**_):
        _time.sleep(capture_ms / 1000)
        # The clock ROI must change or the stall gate fires.
        ticking[0] += 1
        frame = live.copy()
        x, y, w, h = rules.clock_rect
        frame[y : y + h, x : x + w] = ticking[0] % 200
        return Frame(frame, {"foreground": True}, 0)

    desktop = Mock()
    desktop.capture.side_effect = capture
    desktop.apply.return_value = {"applied": 1}
    env = ArenaEnv(desktop, rules, [], downscale=False)
    env.reset()
    return env, desktop


def test_tick_holds_cadence_while_a_slow_policy_thinks(tmp_path):
    """The whole point of the pipelined tick: inference overlaps input dispatch.

    Serial dispatch-then-capture made a tick cost PERIOD + capture + inference, so 5 Hz
    was unreachable by construction. With a 60 ms policy and a 15 ms capture the tick
    must still be one PERIOD, not PERIOD + 75 ms.
    """
    import time as _time

    from hoi4_arena.actions import PERIOD

    env, _ = _cadence_env(tmp_path, capture_ms=15)
    action = np.zeros((SLOTS, 3), dtype=np.int64)
    stamps = []
    try:
        for _ in range(6):
            stamps.append(_time.monotonic())
            _, _, _, _, info = env.step(action)
            assert info["valid"], info.get("error")
            _time.sleep(0.060)  # a policy deciding the next action
    finally:
        env.close()
    periods = np.diff(stamps)[1:]  # drop the first, which has no interval in flight
    assert periods.max() < PERIOD * 1.25, f"tick overran: {periods.tolist()}"
    # Tight on the low side too: dropping the deadline would let ticks run at whatever
    # rate the dispatch join allows (~175 ms), which is not the cadence that was asked for.
    assert np.median(periods) > PERIOD * 0.9, f"tick undershot: {periods.tolist()}"
    serial = PERIOD + 0.060 + 0.015
    assert periods.mean() < serial * 0.9, (
        f"mean tick {periods.mean():.3f}s is no better than serial {serial:.3f}s"
    )


def test_input_slots_still_span_the_whole_interval(tmp_path):
    """Overlapping must not compress the eight slots into a burst."""
    import time as _time

    from hoi4_arena.actions import PERIOD

    env, desktop = _cadence_env(tmp_path)
    when = []
    desktop.apply.side_effect = lambda events: when.append(_time.monotonic()) or {"applied": 1}
    action = np.zeros((SLOTS, 3), dtype=np.int64)
    try:
        env.step(action)
        _time.sleep(0.030)
        env.step(action)
    finally:
        env.close()
    assert len(when) >= SLOTS, f"only {len(when)} slots dispatched"
    gaps = np.diff(when[:SLOTS])
    slot = PERIOD / SLOTS
    assert when[SLOTS - 1] - when[0] > PERIOD * 0.5, "slots bursted instead of spanning"
    assert np.median(gaps) > slot * 0.5, f"slots too tightly packed: {gaps.tolist()}"
    assert gaps.max() < slot * 2.5, f"a slot was dropped or stalled: {gaps.tolist()}"


def test_terminal_and_fault_stop_the_interval_in_flight(tmp_path):
    """A dispatch thread must not keep injecting into a finished or faulted match."""
    import time as _time
    from unittest.mock import Mock

    from hoi4_arena.desktop import Frame
    from hoi4_arena.environment import ArenaEnv, disarm

    rules = _rules_with(
        tmp_path, ["ready", "healthy", "running_speed_two", "win", "loss", "disconnect", "desync"]
    )
    for terminal in (["win"] * 4, ["disconnect"] * 4):
        frames = [_screen(rules, ["ready", "healthy", "running_speed_two"])] + [
            _screen(rules, active) for active in ([t] for t in terminal)
        ]
        desktop = Mock()
        desktop.capture.side_effect = [
            Frame(f, {"foreground": True}, i) for i, f in enumerate(frames)
        ]
        applied = []
        desktop.apply.side_effect = lambda events: applied.append(_time.monotonic()) or {}
        rules.last, rules.count = None, 0
        env = ArenaEnv(desktop, rules, [], downscale=False)
        env.reset()
        action = np.zeros((SLOTS, 3), dtype=np.int64)
        try:
            for _ in range(len(terminal)):
                _, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            assert env.dispatch is None, "dispatch still in flight after the match ended"
            count = len(applied)
            _time.sleep(0.25)
            assert len(applied) == count, "input was still applied after the match ended"
        finally:
            disarm(env)
            env.close()


def test_a_failure_inside_the_dispatch_thread_invalidates_the_episode(tmp_path):
    """The join exists to surface this; without it the thread's error is swallowed.

    The worker disarms itself immediately on focus loss, so the safety action does not
    depend on this path. What depends on it is the episode being marked invalid instead
    of silently continuing with input that never landed.
    """
    import time as _time

    from hoi4_arena.desktop import DesktopError

    env, desktop = _cadence_env(tmp_path)
    action = np.zeros((SLOTS, 3), dtype=np.int64)
    try:
        assert env.step(action)[4]["valid"]
        desktop.apply.side_effect = DesktopError("focus_lost_during_batch")
        _time.sleep(0.030)
        info = env.step(action)[4]
        if info["valid"]:  # the failure lands during this interval, so at the latest next
            info = env.step(action)[4]
        assert not info["valid"], "a failed dispatch must invalidate the episode"
        assert "focus_lost_during_batch" in info["error"]
        assert not env.active
    finally:
        env.close()


@pytest.fixture(scope="module")
def arena(tmp_path_factory):
    """A freshly generated arena, built against a game directory holding only palettes."""
    from hoi4_arena.mapgen import generate

    game = tmp_path_factory.mktemp("game")
    (game / "map").mkdir()
    for name in ["provinces.bmp", "terrain.bmp", "rivers.bmp", "trees.bmp", "cities.bmp"]:
        palette = Image.new("P", (1, 1))
        palette.putpalette(bytes(range(256)) * 3)
        palette.save(game / "map" / name)
    root = tmp_path_factory.mktemp("mods") / "arena"
    generate(game, root)
    return root


def _audit_with(root, name, text):
    """Audit the arena with one file replaced, then put the original back."""
    from hoi4_arena.mapgen import audit

    path = root / name
    original = path.read_text()
    path.write_text(text)
    try:
        return audit(root)["problems"]
    finally:
        path.write_text(original)


def test_generated_arena_resolves_every_reference_the_engine_looks_up(arena):
    from hoi4_arena.mapgen import audit

    assert audit(arena)["problems"] == []


def test_audit_rejects_a_coast_only_the_land_side_admits(arena):
    rows = []
    for row in (arena / "map/definition.csv").read_text().splitlines():
        cells = row.split(";")
        if len(cells) > 5 and cells[4] == "sea":
            cells[5] = "false"
        rows.append(";".join(cells))
    problems = _audit_with(arena, "map/definition.csv", "\n".join(rows) + "\n")
    assert any("no sea province is marked coastal" in p for p in problems), problems


def test_audit_rejects_a_province_the_engine_can_build_on_but_cannot_place_a_model_for(arena):
    kept = [
        row
        for row in (arena / "map/buildings.txt").read_text().splitlines()
        if ";supply_node;" not in row
    ]
    problems = _audit_with(arena, "map/buildings.txt", "\n".join(kept) + "\n")
    assert any("supply_node placements" in p for p in problems), problems


def test_audit_rejects_a_port_pointing_at_no_sea_province(arena):
    rows = []
    for row in (arena / "map/buildings.txt").read_text().splitlines():
        cells = row.split(";")
        if len(cells) > 6 and cells[1] == "naval_base_spawn":
            cells[6] = "0"
        rows.append(";".join(cells))
    problems = _audit_with(arena, "map/buildings.txt", "\n".join(rows) + "\n")
    assert any("naval_base_spawn with no adjacent sea province" in p for p in problems), problems


def test_audit_rejects_counter_anchors_the_engine_expects_for_every_province(arena):
    only_first = [
        row
        for row in (arena / "map/unitstacks.txt").read_text().splitlines()
        if row.split(";")[1:2] == ["0"]
    ]
    problems = _audit_with(arena, "map/unitstacks.txt", "\n".join(only_first) + "\n")
    assert any("counter anchors" in p for p in problems), problems


def test_audit_rejects_one_weather_period_stretched_over_the_year(arena):
    name = "map/strategicregions/1-arena.txt"
    text = (arena / name).read_text()
    single = "period = { between = { 0.0 30.11 } temperature = { 15.0 20.0 } no_phenomenon = 1.0 }"
    patched = re.sub(r"weather = \{.*\}\s*\}$", f"weather = {{ {single} }} }}", text)
    problems = _audit_with(arena, name, patched)
    assert any("weather periods, not 12" in p for p in problems), problems


def test_province_adjacency_follows_shared_edges_in_the_bitmap():
    from hoi4_arena.mapgen import adjacency

    ids = np.array([[1, 1, 2], [1, 3, 2], [3, 3, 2]])
    neighbours = adjacency(ids, 3)
    assert neighbours[1] == {2, 3}
    assert neighbours[2] == {1, 3}
    assert neighbours[3] == {1, 2}


def test_generated_terrain_paints_only_the_indices_the_arena_defines(arena):
    """Palette 19 is the perm_snow plains variant and 13 is urban with spawn_city, so a
    stray index is a province rendered as something the definition never names.
    """
    from hoi4_arena.mapgen import OCEAN_INDEX, TERRAIN_INDEX

    drawn = set(np.unique(np.array(Image.open(arena / "map/terrain.bmp"))).tolist())
    assert drawn == set(TERRAIN_INDEX.values()) | {OCEAN_INDEX}, sorted(drawn)


def test_audit_rejects_a_tree_map_that_is_not_seventy_five_two_hundred_fifty_sixths(arena):
    """The engine fixes trees.bmp at 75/256 of the province bitmap; stock is 1650x600."""
    from hoi4_arena.mapgen import audit

    path = arena / "map/trees.bmp"
    original = path.read_bytes()
    Image.open(io.BytesIO(original)).resize((512, 384)).save(path)
    try:
        assert any("75/256" in p for p in audit(arena)["problems"])
    finally:
        path.write_bytes(original)


def test_audit_rejects_an_adjacency_file_with_no_end_marker(arena):
    """Removing the -1 row hangs the loader; the stock file carries one too."""
    header = (arena / "map/adjacencies.csv").read_text().splitlines()[0]
    problems = _audit_with(arena, "map/adjacencies.csv", header + chr(10))
    assert any("end marker" in p for p in problems), problems


def test_audit_rejects_a_missing_colour_map(arena):
    """Every map-shaped texture the arena does not ship is a picture of the stock Earth."""
    from hoi4_arena.mapgen import audit

    path = arena / "map/terrain/colormap_water_1.dds"
    original = path.read_bytes()
    path.unlink()
    try:
        assert any("colormap_water_1" in p and "stock Earth" in p for p in audit(arena)["problems"])
    finally:
        path.write_bytes(original)


def test_audit_rejects_a_tag_with_no_character_name_list(arena):
    """No name list means the random-character path has an empty pool and returns null."""
    text = (arena / "common/names/01_arena_names.txt").read_text()
    problems = _audit_with(arena, "common/names/01_arena_names.txt", text.replace("BLU =", "XXX ="))
    assert any("BLU" in p and "nameless" in p for p in problems), problems


def test_audit_rejects_a_tag_with_no_country_leader(arena):
    text = (arena / "common/characters/arena.txt").read_text()
    problems = _audit_with(
        arena, "common/characters/arena.txt", text.replace("RED_commander", "RED_unused")
    )
    assert any("RED" in p and "random-character" in p for p in problems), problems


def test_audit_rejects_a_victory_point_the_stock_game_would_name(arena):
    """An unnamed victory point shows whatever the stock localisation calls that id."""
    text = (arena / "localisation/english/arena_l_english.yml").read_text(encoding="utf-8-sig")
    stripped = chr(10).join(r for r in text.splitlines() if "VICTORY_POINTS_" not in r)
    problems = _audit_with(arena, "localisation/english/arena_l_english.yml", stripped)
    assert any("has no name" in p for p in problems), problems


def test_generated_coast_is_a_ramp_rather_than_a_cliff(arena):
    """Land must sit above byte 95 and sea below it, without a step no stock coast has."""
    heights = np.array(Image.open(arena / "map/heightmap.bmp")).astype(np.int16)
    step = max(np.abs(np.diff(heights, axis=0)).max(), np.abs(np.diff(heights, axis=1)).max())
    assert step <= 4, f"coast step of {step} bytes"
    assert heights.min() < 95 < heights.max()


def _stub_desktop(reply):
    """A Desktop whose worker reply is fixed, so capture's contract can be checked."""
    from hoi4_arena.desktop import Desktop

    desktop = object.__new__(Desktop)
    desktop.request = lambda op, **kwargs: dict(reply)
    return desktop


def test_capture_rejects_a_worker_that_ignores_the_requested_views():
    """A worker predating worker-side downscaling accepts views and sends 33 MB anyway."""
    stale = {
        "payload": b"",
        "width": 3840,
        "height": 2160,
        "overflow": False,
        "stopped": False,
    }
    with pytest.raises(DesktopError, match="predates worker-side"):
        _stub_desktop(stale).capture(views=224, full=False)


def test_capture_rejects_a_reply_with_the_wrong_number_of_crops():
    reply = {
        "payload": b"",
        "width": 3840,
        "height": 2160,
        "overflow": False,
        "stopped": False,
        "full_bytes": 0,
        "views_bytes": 0,
        "region_bytes": [10],
    }
    with pytest.raises(DesktopError, match="number of crops"):
        _stub_desktop(reply).capture(regions=[[0, 0, 2, 2], [0, 0, 3, 3]], full=False)


def test_audit_rejects_a_map_colour_without_a_named_colour_space(arena):
    """A bare `color = { }` is not the map colour, so Blue gets painted whatever the
    engine picks. Only an rgb-tagged entry in colors.txt decides it.
    """
    text = (arena / "common/countries/colors.txt").read_text()
    problems = _audit_with(
        arena, "common/countries/colors.txt", text.replace("color = rgb {", "color = {")
    )
    assert any("no rgb map colour" in p for p in problems), problems


def test_flag_pixels_and_map_colour_come_from_one_source(arena):
    """The flags were right while the map was wrong because they read different values."""
    from hoi4_arena.mapgen import COUNTRY_COLOUR

    colours = (arena / "common/countries/colors.txt").read_text()
    for tag, rgb in COUNTRY_COLOUR.items():
        assert f"color = rgb {{ {' '.join(map(str, rgb))} }}" in colours
        flag = np.array(Image.open(arena / f"gfx/flags/{tag}.tga").convert("RGB"))
        assert tuple(flag[0, 0]) == rgb, f"{tag} flag {tuple(flag[0, 0])} != {rgb}"


def test_every_province_gets_a_colour_of_its_own(arena):
    """definition.csv is how provinces.bmp is read back, so a shared colour merges two
    provinces into one and the loss is silent. The previous scheme took each channel
    modulo 251, which repeats every 251 ids and only held while there were 192 of them.
    """
    rows = [r.split(";") for r in (arena / "map/definition.csv").read_text().splitlines() if r]
    colours = [(r[1], r[2], r[3]) for r in rows]
    assert len(set(colours)) == len(colours)
    painted = np.array(Image.open(arena / "map/provinces.bmp").convert("RGB"))
    assert len(np.unique(painted.reshape(-1, 3), axis=0)) == len(rows) - 1


def test_states_are_a_grid_rather_than_one_state_a_side(arena):
    """A state is the unit the engine builds, supplies and garrisons in, and theatre
    generation has a documented three-state minimum that one state a side cannot meet.
    """
    from hoi4_arena.mapgen import STATE_COLUMNS, STATE_ROWS

    files = sorted((arena / "history/states").glob("*-arena.txt"))
    assert len(files) == 2 * STATE_COLUMNS * STATE_ROWS
    owners = [re.search(r"owner\s*=\s*(\w+)", f.read_text()).group(1) for f in files]
    assert owners.count("BLU") == owners.count("RED") == STATE_COLUMNS * STATE_ROWS


def test_each_state_holds_a_supply_hub(arena):
    """Supply flow falls off per province travelled and runs out after about two hops, so
    one hub a country left the ends of the border column out of supply, which caps a
    division's organisation below the level the AI needs before it will attack with it.
    """
    hubs = {
        int(line.split()[1])
        for line in (arena / "map/supply_nodes.txt").read_text().split("\n")
        if line
    }
    for path in (arena / "history/states").glob("*-arena.txt"):
        provinces = {
            int(p) for p in re.search(r"provinces = \{([^}]*)\}", path.read_text()).group(1).split()
        }
        assert provinces & hubs, f"{path.name} has no supply hub"


def test_both_countries_march_at_the_arena_rate(arena):
    """Province size is what makes a crossing cost days; marching speed is what pays for
    it. The spirit is useless if it is defined and never added.
    """
    from hoi4_arena.mapgen import ARMY_SPEED_FACTOR

    idea = (arena / "common/ideas/arena.txt").read_text()
    assert f"army_speed_factor = {ARMY_SPEED_FACTOR}" in idea
    for tag in ["BLU", "RED"]:
        assert (
            "add_ideas = arena_march_speed"
            in (arena / f"history/countries/{tag} - Arena.txt").read_text()
        )
