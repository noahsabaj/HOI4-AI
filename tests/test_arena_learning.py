"""Synthetic unit checks. These cannot establish learning in real HOI4."""
# ruff: noqa: E402 -- optional Torch must be skipped before importing its consumers.
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hoi4_agent.arena.actions import choices, choice_index
from hoi4_agent.arena.contracts import ArenaError, Country, Intent, Order, UnitView, Verb
from hoi4_agent.arena.policy import RecurrentPolicy
from hoi4_agent.arena.returns import advantages, objective_potential, transition_reward
from test_arena_bridge import observation


def test_mask_has_no_enemy_dependent_validity() -> None:
    own_view = observation()
    contact_view = replace(own_view, units=own_view.units + (
        UnitView(2, Country.RED, 2, None, None, None),))
    assert choices(own_view) == choices(contact_view)
    assert choices(replace(own_view, units=()))[0].verb == Verb.NOOP
    # Without units only noop, pause and the five speeds remain.
    assert len(choices(replace(own_view, units=()))) == 7
    for i, action in enumerate(choices(own_view)):
        assert choice_index(own_view, action.order(own_view, str(i))) == i


def test_graph_policy_is_order_invariant_and_trainable() -> None:
    torch.manual_seed(4)
    torch.set_num_threads(1)
    policy = RecurrentPolicy(width=32)
    view = observation()
    first = policy(view)
    reordered = policy(replace(view, provinces=tuple(reversed(view.provinces))))
    torch.testing.assert_close(first.distribution.logits, reordered.distribution.logits)
    assert sum(parameter.numel() for parameter in policy.parameters()) < 10_000_000
    assert torch.isfinite(first.distribution.probs).all()
    assert first.distribution.probs.sum().item() == pytest.approx(1)
    old = policy.value_head.weight.detach().clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    loss = -first.distribution.log_prob(torch.tensor(1)) + (first.value - 1).square()
    loss.backward()
    optimizer.step()
    assert not torch.equal(old, policy.value_head.weight)
    # The caller must discard recurrent state on reset, even for identical maps.
    reset = policy(view, policy.initial_memory())
    implicit_reset = policy(view, None)
    torch.testing.assert_close(reset.memory, implicit_reset.memory)
    continuing = policy(view, reset.memory)
    assert not torch.equal(reset.memory, continuing.memory)


def test_empty_army_and_terminal_policy_have_finite_noop() -> None:
    policy = RecurrentPolicy(32)
    output = policy(replace(observation(), units=(), terminal=True))
    assert len(output.actions) == 1
    assert output.distribution.probs.item() == 1
    assert torch.isfinite(output.value)


def test_hour_discounting_and_terminal_gae_boundaries() -> None:
    adv, targets = advantages([0, 1], [0.2, 0.3], [2, 1], [False, True],
                              bootstrap_value=999, gamma_per_hour=0.9, lambda_per_hour=1)
    np.testing.assert_allclose(targets, [0.81, 1])
    np.testing.assert_allclose(adv, [0.61, 0.7])
    # A following episode must never enter a terminal return.
    _, separated = advantages([1, -1], [0, 0], [1, 1], [True, True], gamma_per_hour=0.9)
    np.testing.assert_allclose(separated, [1, -1])
    # A paused decision sees no game time pass: undiscounted, but never negative.
    paused, _ = advantages([0, 1], [0, 0], [0, 1], [False, True], gamma_per_hour=0.9, lambda_per_hour=1)
    np.testing.assert_allclose(paused, [1, 1])
    with pytest.raises(ArenaError, match="negative"):
        advantages([0], [0], [-1], [False])


def test_potential_shaping_telescopes_and_zeroes_terminal_potential() -> None:
    before = observation()
    middle = replace(before, game_hour=2, sequence=1,
                     provinces=tuple(replace(p, controller=Country.BLUE) for p in before.provinces))
    after = replace(middle, game_hour=5, sequence=2, terminal=True, winner=Country.BLUE)
    gamma, weight = 0.9, 0.3
    reward_1 = transition_reward(before, middle, gamma, weight)
    reward_2 = transition_reward(middle, after, gamma, weight)
    shaped_return = reward_1 + gamma ** 2 * reward_2
    assert shaped_return == pytest.approx(gamma ** 2 - weight * objective_potential(before))
    assert objective_potential(after) == 0
    draw = replace(after, winner=None)
    assert transition_reward(middle, draw, gamma, 0) == 0
    assert transition_reward(middle, after, gamma, 0) == 1


def test_policy_sees_own_orders_unit_kind_and_river_crossings() -> None:
    torch.manual_seed(2)
    policy = RecurrentPolicy(32)
    view = observation()
    base = policy(view).distribution.logits
    ordered = replace(view, units=(replace(view.units[0], order_target_province_id=2, kind="armor",
                                           entrenchment=0.5),))
    river = replace(view, provinces=(replace(view.provinces[0], river_neighbors=(2,)), view.provinces[1]))
    assert not torch.equal(base, policy(ordered).distribution.logits)
    assert not torch.equal(base, policy(river).distribution.logits)
    assert type(view).from_dict(ordered.to_dict()) == ordered
    assert type(view).from_dict(river.to_dict()) == river
    with pytest.raises(ArenaError, match="river"):
        replace(view.provinces[0], river_neighbors=(9,))
    with pytest.raises(ArenaError, match="unknown province"):
        replace(view, units=(replace(view.units[0], order_target_province_id=9),))
    # Moving into a held province is the attack; the only other targeted verb is support.
    assert {action.verb for action in choices(view)} == {Verb.NOOP, Verb.PAUSE, Verb.SET_SPEED, Verb.CANCEL, Verb.MOVE, Verb.SUPPORT_ATTACK}


def test_default_discount_keeps_full_match_terminal_signal() -> None:
    from hoi4_agent.arena.learner import PPOConfig
    config = PPOConfig()
    assert config.gamma_per_hour ** 2160 > 0.25
    assert (config.gamma_per_hour * config.lambda_per_hour) ** 24 > 0.9


def test_intent_speed_and_stack_inputs_condition_the_policy() -> None:
    torch.manual_seed(3)
    policy = RecurrentPolicy(32)
    view = observation()
    view = replace(view, provinces=tuple(replace(p, sector="north") for p in view.provinces))
    base = policy(view)
    assert set(base.intent_logits) == {"north"} and base.intent_logits["north"].shape == (3,)
    attack = Intent.of({"north": "attack"}, produced_game_hour=0, source="jev", confidence=0.9)
    assert not torch.equal(base.distribution.logits, policy(view, None, attack).distribution.logits)
    stale = replace(view, game_hour=500, sequence=1)
    assert attack.age_hours(stale.game_hour) == 500
    assert not torch.equal(base.distribution.logits, policy(replace(view, game_speed=5)).distribution.logits)
    stack = replace(view, units=view.units + (UnitView(7, Country.RED, 2, 0.4, 0.8, None, count=6, confidence=0.7),))
    assert torch.isfinite(policy(stack).distribution.logits).all()
    with pytest.raises(ArenaError, match="stance"):
        Intent.of({"north": "charge"}, 0)
    with pytest.raises(ArenaError, match="speed"):
        Order("o", view.episode_id, 0, Country.BLUE, Verb.SET_SPEED)
    assert Order("o", view.episode_id, 0, Country.BLUE, Verb.SET_SPEED, speed=5).speed == 5


def test_intent_head_distills_teacher_probabilities() -> None:
    from hoi4_agent.arena.learner import PPOConfig, PPOLearner
    torch.manual_seed(5)
    view = observation()
    view = replace(view, provinces=tuple(replace(p, sector="north") for p in view.provinces))
    learner = PPOLearner(RecurrentPolicy(32), PPOConfig(learning_rate=0.01))
    teacher = {"north": {"attack": 0.9, "hold": 0.1, "retreat": 0.0}}
    first = learner.distill_intents([(view, teacher)], epochs=1)
    last = learner.distill_intents([(view, teacher)], epochs=40)
    assert last < first
    probabilities = learner.policy(view).intent_logits["north"].softmax(-1)
    assert probabilities.argmax().item() == 0
    with pytest.raises(ArenaError, match="distribution"):
        learner.distill_intents([(view, {"north": {"attack": 0.9, "hold": 0.9, "retreat": 0.0}})])
