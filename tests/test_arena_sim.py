"""SIMULATOR checks. Nothing here says anything about real HOI4."""
# ruff: noqa: E402 -- optional Torch must be skipped before importing its consumers.
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hoi4_agent.arena.contracts import ArenaError, ArenaSpec, Country, Order, Verb
from hoi4_agent.arena.learner import PPOConfig, PPOLearner
from hoi4_agent.arena.policy import RecurrentPolicy
from hoi4_agent.arena.scripted import ScriptedPolicy
from hoi4_agent.arena.sim.engine import Engine
from hoi4_agent.arena.sim.maps import SIZES, VARIANTS, scenario, scenario_id, scenario_ids, scenario_split
from hoi4_agent.arena.sim.session import (ENEMY_STACK_ID_BASE, SIM_FINGERPRINT, NoiseConfig, SimConfig, SimSession,
                                          SpeedModel)
from hoi4_agent.arena.train import PANEL, SimTask, TrainConfig, Trainer, evaluate, play_episode, scripted_opponent, wilson

BOTH = (Country.BLUE, Country.RED)
PACED = SpeedModel(decision_latency_ms=1000, decision_interval_hours=12)


def spec(identifier: str, seed: int = 0, days: int = 90) -> ArenaSpec:
    return ArenaSpec(identifier, SIM_FINGERPRINT, seed, days * 24)


def order(view, verb: Verb, unit: int | None = None, target: int | None = None, speed: int | None = None,
          name: str = "o") -> Order:
    return Order(name, view.episode_id, view.sequence, view.country, verb, () if unit is None else (unit,), target, speed)


def scripted_match(identifier: str, blue: str, red: str, seed: int = 0, days: int = 45, noise: bool = True):
    session = SimSession(SimConfig(PACED, NoiseConfig() if noise else None, max_decisions=1000))
    session.reset(spec(identifier, seed, days), BOTH)
    players = {Country.BLUE: ScriptedPolicy(blue, seed), Country.RED: ScriptedPolicy(red, seed)}
    memory = dict.fromkeys(BOTH)
    views = []
    while not session.terminal:
        for country in BOTH:
            decision, memory[country] = players[country].act(session.observe(country), memory[country])
            assert session.submit(decision).accepted or blue == "random" or red == "random"
        session.step()
        views.append(session.observe(Country.BLUE))
    return session, views


def test_maps_are_deterministic_mirrored_and_split_before_training() -> None:
    assert len(scenario(scenario_id("full", "open", 1)).layout.provinces) == 48
    assert scenario(scenario_id("full", "open", 1)).layout.sectors == ("center", "north", "south")
    assert {len(scenario(scenario_id(size, "open", 0)).layout.provinces) for size in ("tiny", "small")} == {8, 12}
    for size in SIZES:
        for variant in VARIANTS:
            identifier = scenario_id(size, variant, 3, armor=1)
            built = scenario(identifier)
            assert built == scenario(identifier)
            assert built.layout.source == "synthetic"
            by_id = {p.id: p for p in built.layout.provinces}
            swap = {"BLU": "RED", "RED": "BLU", "": ""}
            for province in built.layout.provinces:
                twin = by_id[built.mirror(province.id)]
                assert (twin.terrain, twin.victory_points, twin.sector, twin.y) == (
                    province.terrain, province.victory_points, province.sector, province.y)
                assert twin.x == pytest.approx(1 - province.x)
                assert twin.initial_controller == swap[province.initial_controller]
                assert twin.capital_of == swap[province.capital_of]
                assert set(twin.neighbors) == {built.mirror(n) for n in province.neighbors}
                assert set(twin.river_neighbors) == {built.mirror(n) for n in province.river_neighbors}
            blue = sorted((built.mirror(d.province_id), d.kind, d.organization) for d in built.deployments
                          if d.country == "BLU")
            red = sorted((d.province_id, d.kind, d.organization) for d in built.deployments if d.country == "RED")
            assert blue == red and any(d.kind == "armor" for d in built.deployments)
    assert scenario(scenario_id("small", "open", 1)) != scenario(scenario_id("small", "open", 2))
    assert all(scenario(scenario_id("small", "crossing", 5)).layout.province(p).river_neighbors for p in (3, 4))
    splits = {split: scenario_ids(split, "tiny", 10) for split in ("train", "validation", "held_out")}
    assert not set(splits["train"]) & set(splits["validation"]) and not set(splits["validation"]) & set(splits["held_out"])
    assert all(scenario_split(i) == split for split, ids in splits.items() for i in ids)
    assert splits["train"] == scenario_ids("train", "tiny", 10)
    with pytest.raises(ArenaError):
        scenario("sim-tiny-nonsense-d2a0-s0")


def test_same_seed_same_match_and_other_seed_differs() -> None:
    identifier = scenario_id("small", "open", 2)
    first = scripted_match(identifier, "advance", "random", seed=5)[1]
    assert first == scripted_match(identifier, "advance", "random", seed=5)[1]
    assert first != scripted_match(identifier, "advance", "random", seed=6)[1]
    assert first[-1].terminal


def test_fog_hides_far_enemies_and_is_invariant_to_hidden_changes() -> None:
    identifier = scenario_id("full", "open", 0)
    sessions = [SimSession(SimConfig(PACED)) for _ in range(2)]
    for session in sessions:
        session.reset(spec(identifier, 3), BOTH)
    start = sessions[0].observe(Country.BLUE)
    assert all(unit.country == Country.BLUE and unit.count == 1 for unit in start.units)  # RED starts out of sight
    assert all(province.controller is not None for province in start.provinces)  # control is public, units are not
    engine = sessions[1].engine
    assert engine is not None
    hidden = [d for d in engine.army(Country.RED) if d.province not in sessions[1].visible_provinces(Country.BLUE)]
    assert len(hidden) == 12
    for division in hidden:  # mutate only state BLUE cannot see, staying in provinces BLUE cannot see
        division.organization, division.strength, division.kind = 0.31, 0.52, "armor"
    hidden[0].province = engine.capital[Country.RED]
    hidden[1].alive = False
    for _ in range(3):
        assert sessions[0].observe(Country.BLUE) == sessions[1].observe(Country.BLUE)
        assert sessions[0].observe(Country.RED) != sessions[1].observe(Country.RED)
        for session in sessions:
            session.step()


def test_enemy_stacks_are_aggregated_with_ephemeral_ids() -> None:
    session = SimSession(SimConfig(PACED, noise=None))
    session.reset(spec(scenario_id("tiny", "breakthrough", 0, divisions=4)), BOTH)
    engine = session.engine
    assert engine is not None
    for division, organization in zip(engine.army(Country.RED), (0.2, 0.4, 0.6, 0.8)):
        division.id += 500  # engine IDs must not surface in the enemy view
        division.organization = organization
    view = session.observe(Country.BLUE)
    stacks = [unit for unit in view.units if unit.country == Country.RED]
    assert len(stacks) == 2 and all(stack.count == 2 and stack.id >= ENEMY_STACK_ID_BASE for stack in stacks)
    assert sorted(stack.organization for stack in stacks) == pytest.approx([0.4, 0.6])
    assert all(stack.supply is None and stack.order_target_province_id is None for stack in stacks)
    session.step()
    assert [u.id for u in session.observe(Country.BLUE).units if u.country == Country.RED] == [s.id for s in stacks]
    own = [unit for unit in view.units if unit.country == Country.BLUE]
    assert len(own) == 4 and all(unit.count == 1 and unit.confidence == 1.0 for unit in own)


def test_order_rejections_are_accounted_and_valid_orders_apply() -> None:
    session = SimSession(SimConfig(PACED, noise=None, speed_authority=Country.BLUE))
    episode = session.reset(spec(scenario_id("tiny", "breakthrough", 0)), BOTH)
    view = session.observe(Country.BLUE)
    unit = next(u for u in view.units if u.country == Country.BLUE)
    neighbor = next(p for p in view.provinces if p.id == unit.province_id).neighbors[0]
    far = next(p.id for p in view.provinces if p.id != unit.province_id and p.id not in
               next(q for q in view.provinces if q.id == unit.province_id).neighbors)
    enemy_stack = next(u for u in view.units if u.country == Country.RED)
    cases = [(order(view, Verb.MOVE, 99, neighbor), "unknown unit"),
             (order(view, Verb.MOVE, unit.id, far), "target not adjacent"),
             (order(view, Verb.MOVE, enemy_stack.id, neighbor), "not your unit"),
             (Order("x", "other-episode", view.sequence, Country.BLUE, Verb.NOOP), "wrong episode"),
             (Order("y", episode, view.sequence + 7, Country.BLUE, Verb.NOOP), "stale observation"),
             (order(session.observe(Country.RED), Verb.PAUSE), "country does not control speed and pause")]
    for submitted, reason in cases:
        receipt = session.submit(submitted)
        assert (receipt.accepted, receipt.reason, receipt.applied_game_hour) == (False, reason, None)
    friendly = next(n for n in next(p for p in view.provinces if p.id == unit.province_id).neighbors
                    if next(p for p in view.provinces if p.id == n).controller == Country.BLUE)
    assert session.submit(order(view, Verb.SUPPORT_ATTACK, unit.id, friendly)).reason == "support attack target is friendly"
    assert sum(session.rejections.values()) == 7 and session.accepted == 0
    accepted = session.submit(order(view, Verb.MOVE, unit.id, neighbor))
    assert accepted.accepted and accepted.applied_game_hour == 0 and session.accepted == 1
    session.step()
    after = next(u for u in session.observe(Country.BLUE).units if u.id == unit.id)
    assert after.order_target_province_id == neighbor and after.entrenchment == 0
    assert session.submit(order(session.observe(Country.BLUE), Verb.CANCEL, unit.id)).accepted
    session.step()
    assert next(u for u in session.observe(Country.BLUE).units if u.id == unit.id).order_target_province_id is None
    with pytest.raises(ArenaError):
        SimSession().reset(spec(scenario_id("tiny", "open", 0)), (Country.BLUE, Country.BLUE))
    solo = SimSession()
    solo.reset(spec(scenario_id("tiny", "open", 0)), (Country.BLUE,))
    with pytest.raises(ArenaError, match="not model-controlled"):
        solo.observe(Country.RED)


def test_combat_resolves_retreats_encircles_and_terminates() -> None:
    built = scenario(scenario_id("tiny", "breakthrough", 0, divisions=4))
    engine = Engine(built, seed=1)
    # Four BLUE divisions from one province attack a single RED division next door.
    red = engine.army(Country.RED)
    target = red[0].province
    for division in engine.occupants(target, Country.RED)[1:]:
        division.alive = False
    attackers = engine.army(Country.BLUE)
    origin = next(d.province for d in attackers if target in engine.neighbors[d.province])
    for division in attackers:
        division.province = origin
    for division in attackers:
        assert engine.order(Country.BLUE, division.id, target) is None
    engine.advance(6)
    assert all(d.in_combat for d in attackers) and all(d.in_combat for d in engine.occupants(target, Country.RED))
    assert all(d.province != target for d in attackers)  # attackers stay put while the battle runs
    before = [d.organization for d in engine.occupants(target, Country.RED)]
    assert all(value < 1 for value in before)
    engine.advance(24 * 20)
    assert engine.control[target] == Country.BLUE  # defenders broke and the attackers walked in
    assert all(0 <= d.organization <= 1 and 0 <= d.strength <= 1 for d in engine.divisions)
    # Encirclement: a broken defender with no friendly province to retreat to is destroyed.
    pocket = Engine(scenario(scenario_id("tiny", "encirclement", 0)), seed=1)
    trapped = next(d for d in pocket.army(Country.RED)
                   if all(pocket.control[n] == Country.BLUE for n in pocket.neighbors[d.province]))
    trapped.organization = 0.01
    besieger = pocket.army(Country.BLUE)[-1]
    besieger.province = next(iter(pocket.neighbors[trapped.province]))
    pocket.order(Country.BLUE, besieger.id, trapped.province)
    pocket.advance(24)
    assert not trapped.alive
    # Every scripted pairing ends by itself.
    session, views = scripted_match(scenario_id("tiny", "open", 1), "advance", "advance")
    assert views[-1].terminal and session.engine is not None and session.engine.reason


def test_capital_capture_and_victory_point_timeout_rule() -> None:
    built = scenario(scenario_id("tiny", "open", 0))
    idle = Engine(built, horizon_hours=48)
    idle.advance(47)
    assert not idle.terminal
    idle.advance(10)
    assert idle.terminal and idle.winner is None and idle.hour == 48 and idle.reason == "victory points"
    ahead = Engine(built, horizon_hours=48)
    prize = next(p.id for p in built.layout.provinces if p.victory_points and p.initial_controller == "RED"
                 and not p.capital_of)
    ahead.control[prize] = Country.BLUE
    ahead.advance(48)
    assert ahead.winner == Country.BLUE and ahead.points(Country.BLUE) > ahead.points(Country.RED)
    taken = Engine(built)
    taken.control[taken.capital[Country.BLUE]] = Country.RED
    taken.advance(1)
    assert (taken.winner, taken.reason) == (Country.RED, "capital captured")
    gone = Engine(built)
    for division in gone.army(Country.RED):
        division.alive = False
    gone.advance(1)
    assert (gone.winner, gone.reason) == (Country.BLUE, "army eliminated")


def test_pause_speed_and_decision_cap_semantics() -> None:
    model = SpeedModel(hours_per_second=(1, 2, 4, 8, 24), decision_latency_ms=500, decision_interval_hours=0)
    session = SimSession(SimConfig(model, noise=None, initial_speed=5, max_decisions=12))
    session.reset(spec(scenario_id("tiny", "open", 0)), (Country.BLUE,))
    assert session.step() == 12 and session.observe(Country.BLUE).game_speed == 5
    view = session.observe(Country.BLUE)
    assert session.submit(order(view, Verb.SET_SPEED, speed=1)).accepted
    assert [session.step() for _ in range(4)] == [0, 1, 0, 1]  # half an hour per decision accumulates
    assert session.observe(Country.BLUE).game_speed == 1
    assert session.submit(order(session.observe(Country.BLUE), Verb.PAUSE)).accepted
    before = session.observe(Country.BLUE)
    assert session.step() == 0
    after = session.observe(Country.BLUE)
    assert after.paused and after.game_hour == before.game_hour and after.sequence == before.sequence + 1
    assert after.captured_monotonic_ns > before.captured_monotonic_ns  # wall time still passes
    # A paused loop cannot run forever: the decision cap ends it as a truncated draw.
    while not session.terminal:
        session.step()
    final = session.observe(Country.BLUE)
    assert session.truncated and final.terminal and final.winner is None and session.decisions == 12
    assert SpeedModel(decision_interval_hours=12, decision_latency_ms=1000).hours(5) == 24
    assert SpeedModel(decision_interval_hours=12, decision_latency_ms=1000).hours(1) == 12


def test_noise_stays_within_bounds_and_spares_own_units() -> None:
    identifier = scenario_id("tiny", "breakthrough", 0, divisions=6)
    noisy = SimSession(SimConfig(PACED, NoiseConfig(stack_dropout=0.3, count_misread=0.3)))
    exact = SimSession(SimConfig(PACED, None))
    for session in (noisy, exact):
        session.reset(spec(identifier, 4), BOTH)
    seen = dropped = misread = 0
    for _ in range(60):
        view, truth = noisy.observe(Country.BLUE), exact.observe(Country.BLUE)
        true_stacks = {u.province_id: u for u in truth.units if u.country == Country.RED}
        stacks = {u.province_id: u for u in view.units if u.country == Country.RED}
        assert set(stacks) <= set(true_stacks)
        dropped += len(true_stacks) - len(stacks)
        for province, stack in stacks.items():
            seen += 1
            assert abs(stack.count - true_stacks[province].count) <= 1 and stack.count >= 1
            misread += stack.count != true_stacks[province].count
            assert 0.6 <= stack.confidence <= 1.0
            assert abs(stack.organization - true_stacks[province].organization) <= 0.5 / 30 + 1e-9
        for unit, true_unit in zip((u for u in view.units if u.country == Country.BLUE),
                                   (u for u in truth.units if u.country == Country.BLUE)):
            assert (unit.id, unit.province_id, unit.count, unit.confidence) == (true_unit.id, true_unit.province_id, 1, 1.0)
            assert abs(unit.organization - true_unit.organization) <= 0.5 / 30 + 1e-9
            assert abs(unit.organization * 30 - round(unit.organization * 30)) < 1e-9
        for session in (noisy, exact):
            session.step()
    assert dropped > 0 and misread > 0 and seen > 0
    assert noisy.observe(Country.BLUE) is noisy.observe(Country.BLUE)  # stable until the next step


@pytest.mark.parametrize("style", ScriptedPolicy.STYLES)
def test_scripted_styles_finish_and_mirror_matches_are_fair(style: str) -> None:
    for variant in ("open", "encirclement", "supply"):
        session, views = scripted_match(scenario_id("tiny", variant, 2), style, "hold", days=30)
        assert views[-1].terminal and not session.truncated
    # hold against hold is symmetric on every mirrored map: a draw on victory points.
    session, views = scripted_match(scenario_id("small", "crossing", 1), "hold", "hold", days=20)
    assert views[-1].winner is None and session.engine is not None and session.engine.reason == "victory points"
    assert session.engine.points(Country.BLUE) == session.engine.points(Country.RED)


def test_collector_output_trains_without_fixture_flag(tmp_path) -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    task = SimTask(horizon_days=6, max_decisions=30)
    policy = RecurrentPolicy(16)
    episodes = [play_episode(policy, scripted_opponent(PANEL[i % 4], i), task, task.scenarios("train")[i], i,
                             tuple(Country)[i % 2], 1 + i % 5, policy_id="sim-learner-u0", opponent_id=PANEL[i % 4])
                for i in range(3)]
    for episode in episodes:
        trajectory = episode.trajectory
        assert trajectory is not None and trajectory.complete
        assert (trajectory.provenance.source, trajectory.provenance.scenario_split) == ("simulator", "train")
        assert episode.decisions == len(trajectory.transitions) <= 30
        trajectory.save(tmp_path / "episode.json")
        assert type(trajectory).load(tmp_path / "episode.json") == trajectory
    paused = [t for e in episodes if e.trajectory for t in e.trajectory.transitions
              if t.observation.paused and t.order.verb is not Verb.PAUSE]
    assert all(t.elapsed_hours == 0 for t in paused)
    before = [p.detach().clone() for p in policy.parameters()]
    stats = PPOLearner(policy, PPOConfig(epochs=1)).update([e.trajectory for e in episodes if e.trajectory])
    assert stats["transitions"] == sum(e.decisions for e in episodes)
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))


def test_trainer_checkpoints_resume_and_evaluation_intervals(tmp_path) -> None:
    torch.set_num_threads(1)
    task = SimTask(horizon_days=4, max_decisions=16, variants=("open",))
    config = TrainConfig(str(tmp_path / "run"), width=16, updates=1, episodes_per_update=2, freeze_every=1,
                         device="cpu", ppo=PPOConfig(epochs=1), task=task)
    summary = Trainer(config, log=lambda _: None).run()
    assert summary["updates"] == 1 and summary["episodes"] == 2 and summary["games_per_hour"] > 0
    resumed = Trainer(TrainConfig(**{**config.__dict__, "updates": 2}), log=lambda _: None)
    assert resumed.learner.updates == 1 and resumed.league.evaluation_panel == PANEL
    assert len([o for o in resumed.league.opponents.values() if o.kind == "historical"]) == 2
    assert resumed.run()["episodes"] == 4
    rows = [row for row in (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and '"source": "simulator"' in rows[0]
    report = evaluate(resumed.learner.policy, task, scenarios=1, speeds=(1, 5), opponents=("hold-v1",))
    assert [(row["speed"], row["games"]) for row in report["rows"]] == [("1", 2), ("5", 2), ("all", 4)]
    low, high = wilson(8, 10)
    assert 0 < low < 0.8 < high < 1 and wilson(0, 0) == (0.0, 1.0)
