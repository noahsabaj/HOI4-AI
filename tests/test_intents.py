import numpy as np
import pytest

from hoi4_arena import intents
from hoi4_arena.intents import Intent


class Script:
    """Events as the scripted player's desktop logs them, 0.15 s apart unless waited."""

    def __init__(self):
        self.t = 1_000_000_000
        self.events = []

    def add(self, event, skill=None, gap=0.15):
        self.t += int(gap * 1e9)
        item = {"t_ns": self.t, "event": event}
        if skill:
            item["skill"] = skill
        self.events.append(item)

    def wait(self, seconds):
        self.t += int(seconds * 1e9)

    def move(self, x, y, skill=None):
        self.add({"kind": "move", "x": x, "y": y}, skill)

    def click(self, x, y, button=0, shift=False, skill=None):
        self.move(x, y, skill)
        self.wait(0.3)
        if shift:
            self.add({"kind": "key", "vk": 0x10, "down": True}, skill)
        self.add({"kind": "button", "button": button, "down": True}, skill)
        self.add({"kind": "button", "button": button, "down": False}, skill)
        if shift:
            self.add({"kind": "key", "vk": 0x10, "down": False}, skill)
        self.wait(0.8)

    def tap(self, vk, skill=None):
        self.add({"kind": "key", "vk": vk, "down": True}, skill)
        self.add({"kind": "key", "vk": vk, "down": False}, skill)

    def wheel(self, notches, x=0.5, y=0.5, skill=None):
        self.move(x, y, skill)
        for _ in range(abs(notches)):
            self.add({"kind": "wheel", "delta": 120 if notches > 0 else -120}, skill, gap=0.03)

    def survey(self, skill="survey"):
        self.wheel(-30, skill=skill)
        self.wait(0.5)
        self.move(0.65, 0.012, skill)
        self.wait(1.0)

    def drag(self, points, skill=None):
        self.move(*points[0], skill)
        self.wait(0.3)
        self.add({"kind": "button", "button": 1, "down": True}, skill)
        for p in points[1:]:
            self.move(*p, skill)
        self.add({"kind": "button", "button": 1, "down": False}, skill)
        self.wait(0.8)


def setup_game():
    """A scripted game's setup, a camera spell, a law step, and a paused redraw."""
    s = Script()
    s.move(0.5, 0.5, "form_army")
    s.click(0.43, 0.053, shift=True, skill="form_army")
    s.click(0.517, 0.936, skill="form_army")
    s.click(30 / 1920, 140 / 1080, skill="assign_general")
    s.click(950 / 1920, 352 / 1080, skill="assign_general")
    s.survey("draw_front")
    s.tap(0x5A, "draw_front")
    s.click(0.502, 0.58, skill="draw_front")
    s.tap(0x5A, "draw_front")
    s.click(0.49, 0.47, skill="draw_front")
    s.survey("draw_offensive")
    s.tap(0x58, "draw_offensive")
    s.drag([(0.41, 0.36), (0.41, 0.45), (0.41, 0.64)], "draw_offensive")
    for _ in range(3):
        s.click(1789 / 1920, 20 / 1080, skill="run")
    s.move(0.5, 0.75, "run")
    s.tap(0x20, "run")
    s.wait(2)
    s.click(1789 / 1920, 20 / 1080, skill="run")
    s.wait(4)
    # The camera: in on the front, a pan, a look.
    s.wheel(12, 0.49, 0.5)
    s.wait(1)
    s.add({"kind": "key", "vk": 0x26, "down": True})
    s.add({"kind": "key", "vk": 0x26, "down": False}, gap=0.3)
    s.wait(1)
    s.move(0.48, 0.45)
    s.wait(4)
    # A law step.
    s.tap(0x51, "set_law")
    s.wait(1)
    s.click(0.031, 0.549, skill="set_law")
    s.click(703 / 1920, 322 / 1080, skill="set_law")
    s.click(1054 / 1920, 677 / 1080, skill="set_law")
    s.tap(0x51, "set_law")
    s.wait(5)
    # A popup's Ok.
    s.click(0.47, 0.66, skill="popup")
    s.wait(5)
    # A paused redraw.
    s.move(0.5, 0.5, "pause")
    s.tap(0x20, "pause")
    s.wait(1)
    s.click(947 / 1920, 1010 / 1080, skill="clear_orders")
    s.move(0.5, 0.5, "clear_orders")
    s.click(1265 / 1920, 885 / 1080, button=1, skill="clear_orders")
    s.click(1054 / 1920, 677 / 1080, skill="clear_orders")
    s.survey("draw_front")
    s.tap(0x5A, "draw_front")
    s.click(0.45, 0.5, skill="draw_front")
    s.survey("draw_offensive")
    s.tap(0x58, "draw_offensive")
    s.drag([(0.38, 0.36), (0.38, 0.64)], "draw_offensive")
    s.move(0.5, 0.5, "pause")
    s.tap(0x20, "pause")
    s.wait(6)
    s.click(947 / 1920, 1010 / 1080, skill="execute")
    s.move(0.5, 0.5, "execute")
    s.click(979 / 1920, 958 / 1080, skill="execute")
    return s.events


def test_gestures_join_moves_to_presses_and_zooms():
    s = Script()
    s.click(0.43, 0.05, shift=True)
    s.wheel(-30)
    s.drag([(0.4, 0.4), (0.4, 0.5)])
    s.tap(0x5A)
    found = intents.gestures(s.events)
    assert [g.kind for g in found] == ["click", "zoom", "drag", "tap"]
    assert found[0].shift and found[0].first == 0 and found[0].last == 4
    assert found[1].notches == -30 and found[1].at == (0.5, 0.5)
    assert found[2].button == 1 and len(found[2].path) == 2
    assert found[3].vk == 0x5A


def test_a_scripted_game_relabels_into_the_planner_s_skills():
    events = setup_game()
    segments = intents.relabel(events)
    planner = [s["skill"] for s in segments if s["skill"] != "camera"]
    assert planner == [
        "form_army",
        "assign_general",
        "draw_front",
        "draw_offensive",
        "run",
        "set_law",
        "popup",
        "pause",
        "clear_orders",
        "draw_front",
        "draw_offensive",
        "pause",
        "execute",
    ]
    by = {s["skill"]: s for s in segments}
    assert by["draw_front"]["args"]["clicks"] == 1  # the redraw's; setup's had 2
    assert by["set_law"]["args"] == {"law": "limited", "answer": "ok"}
    assert by["set_law"]["intent_args"] == {"law": "limited"}
    # The redraw's front and offensive join its clear_orders.
    redraw = [s for s in segments if s["intent"] == "redraw"]
    assert [s["skill"] for s in redraw] == ["clear_orders", "draw_front", "draw_offensive"]
    assert len({s["group"] for s in redraw}) == 1
    pauses = [s["intent_args"]["paused"] for s in segments if s["skill"] == "pause"]
    assert pauses == [True, False]
    assert not any(s["skill"] == "unknown" for s in segments)


def test_the_relabel_agrees_with_the_recorder_s_tags():
    events = setup_game()
    segments = intents.relabel(events)
    share, count, wrong = intents.agreement(events, segments)
    assert count == len(events)
    assert share == 1.0, wrong


def test_orders_are_matched_to_the_segment_they_ended():
    events = setup_game()
    times = np.array([e["t_ns"] for e in events], np.int64)
    by_index = {id(e): i for i, e in enumerate(events)}

    def frame_after(k):
        return by_index[id(events[k])] + 2

    segments = intents.relabel(events)
    army = next(s for s in segments if s["skill"] == "form_army")
    law = next(s for s in segments if s["skill"] == "set_law")
    manifest = {
        "orders": [
            {"frame": frame_after(army["last_event"]), "order": "army"},
            # Stamped when OK closed the question, before the Q that closes the screen.
            {"frame": frame_after(law["last_event"] - 2), "order": "law", "law": "limited"},
            {"frame": frame_after(law["last_event"]), "order": "guard", "held": 0.2},
        ]
    }
    matched, named = intents.match_orders(segments, manifest, times)
    assert (matched, named) == (2, 2)
    assert army["order"] == "army" and law["order"] == "law"


def test_decisions_take_the_intent_in_progress():
    events = setup_game()
    segments = intents.relabel(events)
    period = 200_000_000
    decisions = np.arange(events[0]["t_ns"], events[-1]["t_ns"], period)
    kinds, skills, starts = intents.decision_intents(segments, decisions, period)
    assert kinds[0] == intents.INTENTS.index("form_army")
    assert intents.INTENTS.index("redraw") in kinds
    assert starts.sum() >= len([s for s in segments if s["skill"] != "camera"])
    assert set(skills.tolist()) <= set(range(-1, len(intents.SKILLS)))


def test_intents_check_their_arguments():
    assert Intent("draw_offensive", {"attack": "broad", "target_state": 12}).to_json() == {
        "intent": "draw_offensive",
        "attack": "broad",
        "target_state": 12,
    }
    assert Intent.from_json({"intent": "set_law", "law": "service"}).arg("law") == "service"
    with pytest.raises(ValueError):
        Intent("fly")
    with pytest.raises(ValueError):
        Intent("execute", {"law": "service"})
    with pytest.raises(ValueError):
        Intent("draw_offensive", {"target_state": 17})
    with pytest.raises(ValueError):
        Intent("execute", {"army": -1})
    assert set(intents.SKILL_INTENT.values()) <= set(intents.INTENTS)


def test_the_logged_desktop_tags_each_input_with_the_skill_in_hand():
    from hoi4_arena.ai_games import Logged

    class Desk:
        def apply(self, events):
            return {"t_ns": 5}

    class Player:
        @intents.tagged("draw_front")
        def draw_front(self, desk):
            desk.apply([{"kind": "key", "vk": 0x5A, "down": True}])
            self.look(desk)

        @intents.tagged("survey")
        def look(self, desk):
            desk.apply([{"kind": "move", "x": 0.65, "y": 0.012}])

    logged = Logged(Desk())
    Player().look(logged)
    Player().draw_front(logged)
    logged.apply([{"kind": "move", "x": 0.5, "y": 0.5}])
    assert [e.get("skill") for e in logged.take()] == ["survey", "draw_front", "draw_front", None]
    # A desktop that keeps no tags is left alone.
    Player().draw_front(Desk())


def test_the_scripted_hand_carries_intents_to_the_planner():
    from hoi4_arena.hand import ScriptedHand

    calls = []

    class Planner:
        plan = {"attack": "broad", "conscription": "service", "pause_redraw": True}
        running, law_step, speed, defending = True, 0, 5, False

        def draw_offensive(self, desk, attack=None, target_state=None):
            calls.append(("offensive", attack, target_state))

        def draw_front(self, desk, guard=None, front_state=None):
            calls.append(("front", guard, front_state))
            return False

        def clear_orders(self, desk):
            calls.append(("clear",))
            return True

        def pause(self, desk, paused):
            calls.append(("pause", paused))
            return True

        def guard_share(self):
            return 0.15

        def raise_conscription(self, desk):
            self.law_step += 1

    planner = Planner()
    hand = ScriptedHand(planner)
    assert hand.execute(None, Intent("draw_offensive", {"target_state": 11}))
    assert hand.execute(None, {"intent": "redraw", "target_state": 12, "guard": False})
    assert calls[1:] == [
        ("pause", True),
        ("clear",),
        ("front", None, None),
        ("offensive", None, 12),
        ("pause", False),
    ]
    assert hand.execute(None, Intent("set_law", {"law": "extensive"}))
    assert planner.plan["conscription"] == "extensive"
    with pytest.raises(NotImplementedError):
        hand.execute(None, Intent("execute", {"army": 1}))


def test_a_state_s_centre_lies_in_that_state():
    from hoi4_arena.scripted import state_at, state_centre

    for state in range(1, 17):
        u, v = state_centre(state)
        assert state_at(u, v) == state
    assert state_centre(99) is None
