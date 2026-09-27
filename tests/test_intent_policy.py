import json

import numpy as np
import pytest
from test_intents import setup_game

from hoi4_arena import intent_policy as ip
from hoi4_arena import intents


def acts_of(events):
    return ip.macro_segments(intents.relabel(events))


def test_a_redraw_and_its_pauses_are_one_act():
    acts = acts_of(setup_game())
    assert [a["intent"] for a in acts] == [
        "form_army",
        "assign_general",
        "draw_front",
        "draw_offensive",
        "run",
        "set_law",
        "redraw",
        "execute",
    ]
    redraw = acts[6]
    assert redraw["t1"] > redraw["t0"]


def test_each_second_takes_the_act_begun_in_it_or_is_busy():
    events = setup_game()
    acts = acts_of(events)
    times = np.arange(events[0]["t_ns"] - 10**8, events[-1]["t_ns"] + 2 * 10**9, 2 * 10**8)
    steps = ip.strategy_steps(acts, times)
    labels = [ip.STRATEGY[k] for k in steps["labels"] if k]
    assert labels == [a["intent"] for a in acts]
    # While an act is being carried out, the seconds after its start are the hand's.
    assert steps["busy"].any() and not (steps["busy"] & (steps["labels"] > 0)).any()
    # The history counts acts begun before each second.
    first_run = int(np.flatnonzero(steps["labels"] == ip.STRATEGY.index("run"))[0])
    before, after = steps["history"][first_run], steps["history"][first_run + 1]
    k = ip.ACTS.index("run")
    assert before[k] == 0 and after[k] > 0
    # No manifest, so no order says it was done: the game is not known to run.
    assert after[-1] == 0.0


def test_held_out_games_are_the_splits_and_a_fixed_share():
    names = [f"game-{i}" for i in range(400)]
    held = [ip.held_out(n, {"game-3": "test"}) for n in names]
    assert held[3]
    assert 0.12 < np.mean(held) < 0.3
    assert held == [ip.held_out(n, {"game-3": "test"}) for n in names]


def test_bfloat16_bits_read_back():
    import torch

    values = torch.tensor([1.5, -40.25, 0.0078125], dtype=torch.bfloat16)
    bits = values.view(torch.int16).numpy()
    assert np.array_equal(ip.bfloat16_bits(bits), values.float().numpy())


def fake_data(tmp_path, games=6, steps=40):
    rng = np.random.default_rng(0)
    labels, busy, summary, history, game, seconds = [], [], [], [], [], []
    meta = []
    for g in range(games):
        lab = np.zeros(steps, np.int64)
        lab[5], lab[10] = ip.STRATEGY.index("form_army"), ip.STRATEGY.index("run")
        s = rng.normal(size=(steps, 16)).astype(np.float16)
        s[10, 0] = 5.0  # The screen tells when to run.
        labels.append(lab)
        busy.append(np.zeros(steps, bool))
        summary.append(s)
        history.append(np.zeros((steps, ip.HISTORY), np.float32))
        game.append(np.full(steps, g, np.int32))
        seconds.append(np.arange(steps, dtype=np.int32))
        meta.append({"name": f"g{g}", "steps": steps, "held_out": g >= games - 2})
    path = tmp_path / "steps.npz"
    np.savez(
        path,
        summary=np.concatenate(summary),
        history=np.concatenate(history),
        labels=np.concatenate(labels),
        busy=np.concatenate(busy),
        game=np.concatenate(game),
        seconds=np.concatenate(seconds),
    )
    path.with_suffix(".json").write_text(json.dumps({"recordings": meta}))
    return path


def test_an_intent_policy_trains_and_reports_on_held_out_games(tmp_path):
    torch = pytest.importorskip("torch")
    data = fake_data(tmp_path)
    report = ip.train(data, tmp_path / "run", epochs=3, batch=2, threads=1)
    held = report["held_out"]
    assert held["teacher_acts"] == 4
    assert 0 <= held["act_recall"] <= 1 and held["free_seconds"] == 80
    net = ip.load(tmp_path / "run" / "intent-policy.pt")
    logits, _ = net(torch.zeros(1, 2, 16), torch.zeros(1, 2, ip.HISTORY))
    assert logits.shape == (1, 2, len(ip.STRATEGY))


def test_the_intent_planner_decides_once_a_second_and_the_hand_acts(monkeypatch):
    torch = pytest.importorskip("torch")
    import hoi4_arena.ai_games as ai_games

    class Net(torch.nn.Module):
        def forward(self, summary, history, state=None):
            # Run first, then wait: act on the likeliest act when wait is unlikely.
            logits = torch.full((1, summary.shape[1], len(ip.STRATEGY)), -9.0)
            logits[..., ip.STRATEGY.index("run")] = 5.0
            return logits, state

    monkeypatch.setattr(ai_games, "screen", lambda desk: np.zeros((4, 4, 3), np.uint8))
    planner_class = ip.make_intent_planner(Net(), lambda rgb: torch.zeros(16), threshold=0.5)
    planner = planner_class("BLU", {"attack": "broad"}, {}, None, 5, frame=lambda: 7)
    done = []

    class Hand:
        def execute(self, desk, intent):
            done.append(intent.name)
            planner.running = True
            return True

    planner.hand = Hand()
    planner.setup(desk=None)
    assert done == ["run"]
    assert planner.intent_log[0]["act"] == "run" and planner.intent_log[0]["done"]
    assert planner.history.running == 1.0
    # The next decision is a second after this one.
    assert planner.next_at > 0 and len(planner.intent_log) == 1
