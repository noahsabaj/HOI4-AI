"""DAgger's expert: the scripted player's setup, asked one decision at a time."""

import json
import random

import numpy as np
from test_learned import _scripted

from hoi4_arena import dagger, practice
from hoi4_arena.actions import VOCAB, encode_interval
from hoi4_arena.dataset import DAGGER_LABELS, session_labels
from hoi4_arena.scripted import STOP_BUTTON, Planner, best_plan

S = 1_000_000_000
ALERT = (825 / 1920, 56 / 1080)
SHIFT = {"kind": "key", "vk": 0x10}


class _Rules:
    def __init__(self, paused=True):
        self.paused = paused

    def matches(self, name, rgb):
        return name == "paused" and self.paused


def _expert(shown, paused=True, country="BLU"):
    """An expert whose template search finds what `shown` names, where it says."""
    planner = Planner(country, best_plan(random.Random(0)), {}, None, 5, lambda: 0)
    expert = dagger.Expert(country, rules=_Rules(paused), templates={}, planner=planner)
    expert.find = lambda rgb, name: shown.get(name)
    return expert


def _screen(army=False, general=False, plan=False, land=False, plus=False):
    rgb = np.zeros((1080, 1920, 3), np.uint8)
    if land:
        # Blue's land on the left, Red's on the right, meeting at x = 960.
        rgb[200:800, 400:960] = (100, 130, 150)
        rgb[200:800, 960:1500] = (170, 140, 120)
    if army:
        x0, y0, x1, y1 = practice.CARD
        rgb[y0:y1, x0:x1] = (200, 150, 110) if general else (90, 90, 90)
    if plan:
        x0, y0, x1, y1 = STOP_BUTTON
        rgb[y0:y1, x0:x1] = (200, 20, 20)
    if plus:
        x0, y0, x1, y1 = dagger.PLUS_BOX
        rgb[1005:1020, 980:996] = (40, 200, 40)
    return rgb


def _feed(expert, t, *events):
    for i, event in enumerate(events):
        expert.feed({"t_ns": t + i, "event": event})


def _label(expert, rgb, t):
    events, _, phase = expert.label(rgb, t)
    return events, phase


def test_the_army_is_formed_one_decision_at_a_time_looking_before_each_click():
    expert = _expert({"unassigned": ALERT})
    rgb = _screen()
    events, phase = _label(expert, rgb, 0)
    assert phase == "army:alert" and events == [{"kind": "move", "x": ALERT[0], "y": ALERT[1]}]
    _feed(expert, 1, {"kind": "move", "x": ALERT[0], "y": ALERT[1]})
    events, _ = _label(expert, rgb, S // 5)
    assert events == [{**SHIFT, "down": True}, {"kind": "button", "button": 0, "down": True}]
    _feed(expert, S // 5, *events)
    events, phase = _label(expert, rgb, 2 * S // 5)
    assert phase == "finish"
    assert events == [{"kind": "button", "button": 0, "down": False}, {**SHIFT, "down": False}]
    _feed(expert, 2 * S // 5, *events)
    assert _label(expert, rgb, 3 * S // 5) == ([], "settle"), "the screen has not caught up"
    events, phase = _label(expert, _screen(plus=True), 2 * S)
    assert phase == "army:plus" and events[0]["kind"] == "move"
    assert abs(events[0]["x"] * 1920 - 988) < 10 and abs(events[0]["y"] * 1080 - 1012) < 10


def test_an_input_left_down_is_let_go_and_one_down_too_long_counts_as_up():
    expert = _expert({})
    _feed(expert, 0, {"kind": "key", "vk": 0x27, "down": True})
    events, phase = _label(expert, _screen(), S // 5)
    assert phase == "finish" and events == [{"kind": "key", "vk": 0x27, "down": False}]
    # Past the harness's hold limit it went up (desk.release leaves no event).
    assert expert.label(_screen(), 3 * S)[2] != "finish"
    expert.inputs.let_go()
    _feed(expert, 4 * S, {"kind": "key", "vk": 0x20, "down": True})
    assert expert.label(_screen(), 4 * S + S // 5)[2] != "finish", "space is the harness's"


def test_the_general_is_assigned_through_the_portrait_then_the_list():
    expert = _expert({"plans_bar": (0.4, 0.8)})
    rgb = _screen(army=True)
    portrait = dagger.pixels(dagger.COMMANDER_SLOT)
    events, phase = _label(expert, rgb, 3 * S)
    assert phase == "general:portrait" and events[0]["kind"] == "move"
    _feed(expert, 3 * S, {"kind": "move", "x": portrait[0], "y": portrait[1]})
    events, _ = _label(expert, rgb, 3 * S + S // 5)
    assert [e["kind"] for e in events] == ["button", "button"]
    _feed(expert, 3 * S + S // 5, *events)
    events, phase = _label(expert, rgb, 4 * S + S // 5)
    assert phase == "general:commander", "the list is open after a click on the portrait"
    assert _label(_expert({}), rgb, 3 * S)[1] == "general:select", "the army is selected first"


def test_the_front_is_drawn_with_the_tool_on_the_enemy_side_of_the_border():
    expert = _expert({"plans_bar": (0.4, 0.8)})
    rgb = _screen(army=True, general=True, land=True)
    events, phase = _label(expert, rgb, 5 * S)
    assert phase == "front:tool" and events[0] == {"kind": "key", "vk": 0x5A, "down": True}
    _feed(expert, 5 * S, *events)
    expert.find = lambda rgb, name: {"plans_bar": (0.4, 0.8), "front_tool": (0.6, 0.8)}.get(name)
    events, phase = _label(expert, rgb, 6 * S)
    assert phase == "front:click" and events[0]["kind"] == "move"
    x, y = events[0]["x"] * 1920, events[0]["y"] * 1080
    assert 955 <= x <= 975 and 200 < y < 800, "just across the border, in Red's land"
    assert (
        _label(_expert({"plans_bar": (0.4, 0.8)}), _screen(army=True, general=True), 5 * S)[1]
        == "front:zoom-out"
    ), "no border on screen: zoom out"


def test_the_offensive_then_the_run_follow_the_front_and_nothing_after():
    expert = _expert({"plans_bar": (0.4, 0.8)})
    rgb = _screen(army=True, general=True, plan=True, land=True)
    events, phase = _label(expert, rgb, 10 * S)
    assert phase == "offensive:tool" and events[0]["vk"] == 0x58
    _feed(expert, 10 * S, *events)
    start = expert.line[0]
    events, phase = _label(expert, rgb, 11 * S)
    assert phase == "offensive:start" and events == [{"kind": "move", "x": start[0], "y": start[1]}]
    _feed(expert, 11 * S, *events)
    events, phase = _label(expert, rgb, 12 * S)
    assert phase == "offensive:press" and events[0] == {"kind": "button", "button": 1, "down": True}
    _feed(expert, 12 * S, *events)
    t = 12 * S
    for _ in range(40):
        t += S // 5
        events, phase = _label(expert, rgb, t)
        _feed(expert, t, *events)
        if events and events[-1] == {"kind": "button", "button": 1, "down": False}:
            break
    else:
        raise AssertionError("the drag never ended")
    assert expert.offensive_at is not None
    events, phase = _label(expert, rgb, t + 2 * S)
    assert phase == "run" and events[0]["kind"] == "move"
    expert.rules.paused = False
    expert.paused_at = None
    assert _label(expert, rgb, t + 9 * S) == (None, "done"), "running: nothing to add"


def test_a_label_is_an_action_the_policy_can_take():
    events = [{**SHIFT, "down": True}, {"kind": "button", "button": 0, "down": True}]
    action = dagger.as_action(events, [0.0, 150.0], 10 * S)
    kinds = [VOCAB[int(k)] for k in action[:, 0] if k]
    assert kinds == [{**SHIFT, "down": True}, {"kind": "button", "button": 0, "down": True}]
    assert action[5:7, 0].any() and not action[1:5, 0].any(), "150 ms in: late in the 200"


def _practice_game(root):
    _scripted(root, coached=[{"from_frame": 20, "to_frame": 25, "step": "army"}])
    path = root / "manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "source": "policy"}))


def test_a_practice_game_trains_on_the_expert_s_labels_with_the_policy_s_own_as_its_past(
    tmp_path,
):
    game = tmp_path / "game"
    _practice_game(game)
    plain = session_labels(game, sources=("policy",), lead_in=0)
    decisions, times = plain["decisions"], plain["times"]
    labels = np.zeros((len(decisions), 8, 3), np.int64)
    up = {"kind": "key", "vk": 0x51, "down": True}
    labels[1] = encode_interval([{"t_ns": int(decisions[1]), "event": up}], int(decisions[1]))
    coached = (decisions >= times[20]) & (decisions <= times[25])
    inside = int(np.flatnonzero(coached)[0])
    labels[inside] = labels[1]
    valid = np.zeros(len(decisions), bool)
    valid[[1, 4, inside]] = True
    np.savez(game / DAGGER_LABELS, decisions=decisions, actions=labels, valid=valid,
             phase=np.array(["x"] * len(decisions)))  # fmt: skip
    taught = session_labels(game, sources=("policy",), lead_in=0, dagger=2.0)
    assert (taught["actions"][1] == labels[1]).all() and (taught["actions"][4] == 0).all()
    assert not coached[[1, 4]].any()
    assert (taught["actions"][inside] == plain["actions"][inside]).all(), "the coach's own"
    expected = np.where(coached, 1.0, 0.0)
    expected[[1, 4]] = 2.0
    assert np.allclose(taught["weight"], expected)
    assert (taught["previous"] == plain["previous"]).all(), "what the policy did, as it saw"
    off = session_labels(game, sources=("policy",), lead_in=0)
    assert (off["weight"] == np.where(coached, 1.0, 0.0)).all(), "without --dagger as before"


def test_agreement_counts_a_press_once_and_a_move_onto_the_click_as_aiming_at_it():
    def action(*events):
        return encode_interval([{"t_ns": i, "event": e} for i, e in enumerate(events)], 0)

    click = {"kind": "button", "button": 0, "down": True}
    onto = {"kind": "move", "x": 0.5, "y": 0.5}
    recorded = np.zeros((10, 8, 3), np.int64)
    recorded[2] = action(onto, click)
    recorded[6] = action({"kind": "key", "vk": 0x5A, "down": True})
    expert = np.zeros_like(recorded)
    expert[1] = action(onto)
    expert[7] = expert[8] = action({"kind": "key", "vk": 0x5A, "down": True})
    result = dagger.agreement(recorded, expert, np.ones(10, bool))
    assert (result["recall"], result["aim_recall"], result["precision"]) == (0.5, 1.0, 1.0)
    assert result["expert_presses"] == 1, "a run of the same press is one"


def test_the_aggregate_links_every_labelled_game_and_holds_some_out(tmp_path):
    for i in range(20):
        game = tmp_path / "practice" / f"practice-peer-{i:02d}"
        game.mkdir(parents=True)
        (game / "manifest.json").write_text("{}")
        if i != 3:
            (game / DAGGER_LABELS).write_bytes(b"")
    made = dagger.aggregate(tmp_path / "dagger", [tmp_path / "practice"])
    splits = json.loads((tmp_path / "dagger" / "splits.json").read_text())
    assert made["games"] == len(splits) == 19 and "practice-peer-03" not in splits
    assert (tmp_path / "dagger" / "practice-peer-00" / "manifest.json").exists()
    assert 0 < made["validation"] < 8 and set(splits.values()) == {"train", "validation"}
    again = dagger.aggregate(tmp_path / "dagger", [tmp_path / "practice"])
    kept = json.loads((tmp_path / "dagger" / "splits.json").read_text())
    assert again["added"] == 0 and kept == splits


def test_train_bc_takes_the_dagger_data_and_its_weight(monkeypatch, tmp_path):
    import sys

    import pytest

    from hoi4_arena import cli, train

    seen = {}
    monkeypatch.setattr(train, "train_bc", lambda data, model, out, **k: seen.update(k))
    monkeypatch.setattr("hoi4_arena.models.configure_precision", lambda tf32: None)
    monkeypatch.setattr("hoi4_arena.models.limit_gpu_memory", lambda fraction: None)
    argv = ["hoi4-arena", "train-bc", "base", "out", "--sources", "scripted", "policy",
            "--dagger", "2", "--dagger-data", "d1", "d2"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert seen["dagger"] == 2.0 and seen["dagger_data"] == ["d1", "d2"]
    with pytest.raises(ValueError, match="policy"):
        train._train_bc("base", "model", tmp_path / "out", dagger=1.0, sources=("scripted",))
