import json
import math

import numpy as np

from hoi4_arena import scripted
from hoi4_arena.scripted import Planner, coarse_layout
from hoi4_arena.strategist import ArenaState, Strategist, day_number, strategist_plan

DAY = (
    "day  1:00, {d} February, 1936 {tag} states 8 owned {owned} divisions 8 surrender 0.1 "
    "strength 1.2 casualties 300 manpower 9.5 deployed 102 rifles 8.6 needed 8.6 at 1=3 2=5"
)


class Log:
    def __init__(self):
        self.lines, self.days = [], {}


def test_the_state_reader_gives_the_latest_day_and_every_state_changing_hands():
    from hoi4_arena.arena_log import parse

    assert day_number("12:00, 1 January, 1936") == 0.5
    assert day_number("1:00, 2 February, 1936") == round(32 + 1 / 24, 2)
    assert day_number(None) is None
    log = Log()
    reader = ArenaState(log, "BLU")
    assert reader.snapshot()["day"] is None
    for line in (DAY.format(d=2, tag="BLU", owned=8), DAY.format(d=3, tag="RED", owned=7)):
        log.lines.append(line)
        event = parse(line)
        log.days[event["tag"]] = event
    log.lines.append("control BLU from RED East 2 12:00, 3 February, 1936")
    snap = reader.snapshot()
    assert snap["date"].startswith("1:00, 3 February") and snap["day"] > 33
    assert snap["own"]["owned"] == 8 and snap["enemy"]["owned"] == 7
    assert snap["own"]["at"] == {1: 3, 2: 5}
    assert snap["controls"] == [
        {"date": "12:00, 3 February, 1936", "day": 33.5, "tag": "BLU", "from": "RED",
         "state": "East 2"}
    ]  # fmt: skip
    # Lines already read are not read twice.
    assert len(reader.snapshot()["controls"]) == 1


def test_a_decision_is_asked_for_by_file_and_kept_with_the_game(tmp_path):
    folder, game = tmp_path / "ask", tmp_path / "game"
    seen = []

    def sleep(seconds):
        # The strategist answers once it has seen the request.
        pending = json.loads((folder / "pending.json").read_text())
        seen.append(pending)
        (folder / "g-001.decision.json").write_text(json.dumps({"intents": ["execute"]}))

    strategist = Strategist(folder, timeout=5, sleep=sleep)
    rgb = np.zeros((4, 6, 3), np.uint8)
    decision = strategist.ask("g-001", {"n": 1}, rgb)
    assert decision == {"intents": ["execute"]}
    assert seen[0]["n"] == 1 and seen[0]["image"].endswith("g-001.png")
    assert seen[0]["answer"].endswith("g-001.decision.json")
    assert not (folder / "pending.json").exists()
    strategist.keep("g-001", game, {"n": 1, "decision": decision})
    assert {p.name for p in game.iterdir()} == {
        "g-001.png", "g-001.request.json", "g-001.decision.json", "log.jsonl"
    }  # fmt: skip
    assert not list(folder.glob("g-001*"))


def test_no_decision_in_time_lets_the_game_go_on(tmp_path):
    clock = [0.0]
    strategist = Strategist(
        tmp_path, timeout=3, sleep=lambda s: clock.__setitem__(0, clock[0] + s),
        clock=lambda: clock[0],
    )  # fmt: skip
    assert strategist.ask("g-002", {"n": 2}) is None
    assert not (tmp_path / "pending.json").exists()


def planner_with_stubs(monkeypatch, clock):
    monkeypatch.setattr(scripted.time, "monotonic", lambda: clock[0])
    planner = Planner("BLU", strategist_plan(), {}, None, 5, frame=lambda: 0)
    calls = []
    planner.clear_orders = lambda desk: calls.append("clear")
    planner.draw_front = lambda desk, guard=None, front_state=None: (
        calls.append(("front", guard, front_state)) and False
    )
    planner.draw_offensive = lambda desk, attack=None, target_state=None: calls.append(
        ("offensive", attack or planner.plan["attack"], target_state)
    )
    return planner, calls


def test_a_decision_sets_the_plan_and_carries_out_its_intents_in_order(monkeypatch):
    clock = [100.0]
    planner, calls = planner_with_stubs(monkeypatch, clock)
    planner.running = True
    applied, errors = planner.decide(None, {
        "plan": {"guard": 0.2, "redraw": 25, "bogus": 1},
        "intents": [
            {"intent": "set_law", "law": "all_adults"},
            {"intent": "draw_offensive", "target_state": 12},
            "execute",
            {"intent": "camera", "kind": "front"},
            {"intent": "fly"},
        ],
    })  # fmt: skip
    assert planner.plan["guard"] == 0.2 and planner.plan["redraw"] == 25
    assert planner.plan["conscription"] == "all_adults" and planner.law_at == 100.0
    assert planner.plan["attack"] == "broad" and planner.plan["target_state"] == 12
    assert calls == ["clear", ("front", 0.2, None), ("offensive", "broad", 12)]
    # A new plan builds its bonus before it executes.
    assert planner.activate_at == 100.0 + scripted.PLANNING
    assert applied == ["plan.guard=0.2", "plan.redraw=25", "set_law", "draw_offensive", "execute"]
    assert errors[0] == "unknown plan setting bogus"
    assert errors[1] == "camera is the hand's, not the strategist's"
    assert "unknown intent 'fly'" in errors[2] and len(errors) == 3
    # A front alone halts the attack: nothing executes, nothing is redrawn.
    planner.attacking = planner.active = True
    planner.redraw_at = 130.0
    calls[:] = []
    planner.decide(
        None, {"intents": [{"intent": "draw_front", "stance": "hold", "front_state": 7}]}
    )
    assert calls == ["clear", ("front", 0.2, 7)] and planner.plan["attack"] == "none"
    assert not planner.attacking and planner.activate_at == planner.redraw_at == math.inf
    # A broad offensive again forgets the target; the plan pushes only the rows asked for,
    # and a redraw keeps its target until told otherwise.
    calls[:] = []
    planner.decide(None, {"plan": {"rows": [0, 0.5]}, "intents": ["draw_offensive"]})
    assert planner.plan["attack"] == "broad" and planner.plan["target_state"] is None
    assert planner.plan["rows"] == [0, 0.5] and calls[-1] == ("offensive", "broad", None)
    planner.decide(None, {"intents": [{"intent": "redraw", "target_state": 11, "attack": "deep"}]})
    planner.decide(None, {"intents": [{"intent": "redraw", "guard": 0.3}]})
    assert calls[-1] == ("offensive", "deep", 11) and planner.plan["guard"] == 0.3
    # Arguments are checked by the shared vocabulary.
    _, errors = planner.decide(None, {"intents": [{"intent": "set_law", "law": "everyone"}]})
    assert "law must be one of" in errors[0]


def test_decision_points_come_every_few_game_days_and_on_events(monkeypatch):
    clock = [0.0]
    planner, _ = planner_with_stubs(monkeypatch, clock)
    snap = {"day": 10.0, "controls": []}
    planner.strategist, planner.state = object(), lambda: snap
    assert planner.decision_due() is None  # Not running yet.
    planner.running = True
    assert planner.decision_due() == "periodic"  # The first decision day is 0.
    planner.decided_day, planner.decide_day = 10.0, 55.0
    assert planner.decision_due() is None
    # A state lost soon after a decision waits for the gap, then calls one.
    snap["controls"] = [{"day": 12.0, "tag": "RED", "from": "BLU", "state": "West 1"}]
    snap["day"] = 12.0
    assert planner.decision_due() is None
    snap["day"] = 21.0
    assert planner.decision_due() == "lost_state"
    assert planner.due()
    planner.pending_event = None  # consult clears it.
    planner.decided_day = 21.0
    # An executing attack that takes nothing for STALL_DAYS.
    planner.attacking = planner.active = True
    snap["day"] = 21.0 + 39
    planner.decide_day = 200.0
    assert planner.decision_due() is None
    snap["day"] = 21.0 + 41
    assert planner.decision_due() == "stalled"


def test_a_consult_pauses_asks_carries_out_and_moves_the_timers_on(monkeypatch, tmp_path):
    clock = [50.0]
    planner, calls = planner_with_stubs(monkeypatch, clock)
    planner.running, planner.debug_dir = True, tmp_path / "game-1"
    planner.check_at, planner.redraw_at = 60.0, math.inf
    planner.pause = lambda desk, paused: calls.append(("pause", paused)) or True
    blue = np.zeros((10, 10), bool)
    blue[:, :5] = True
    planner.home = blue.copy()
    planner.overview = lambda desk: (np.zeros((4, 4, 3), np.uint8), blue, ~blue, (0, 0, 10, 10))
    planner.state = lambda: {"day": 40.0, "date": "x", "controls": []}

    class Answer:
        def ask(self, stem, request, rgb):
            assert stem == "game-1-001" and request["reason"] == "periodic"
            assert request["incursion"] == 0.0 and request["planner"]["law"] == "volunteer"
            clock[0] += 120  # Two minutes to think, the game paused.
            return {"intents": ["execute"], "next_days": 30}

        def keep(self, stem, archive, entry):
            calls.append(("keep", entry["n"], entry["waited_s"]))

    planner.strategist = Answer()
    planner.consult(None, "periodic")
    assert calls == [("pause", True), ("keep", 1, 120.0), ("pause", False)]
    assert planner.check_at == 180.0  # Moved on by the wait.
    assert planner.activate_at == 170.0 and planner.idle() == 120.0
    assert planner.decide_day == 70.0 and planner.decided_day == 40.0
    assert planner.orders[-1]["order"] == "decision"


def test_a_broad_offensive_can_push_part_of_the_front():
    planner = Planner("BLU", strategist_plan(), {}, None, 5, frame=lambda: 0)
    front = [(100, y) for y in range(10, 90)]
    whole = planner.broad_line(front, (10, 10, 90, 190))
    top = planner.broad_line(front, (10, 10, 90, 190), rows=[0, 0.5])
    assert len(whole) == 9 and len(top) == 5
    assert min(y for _, y in top) >= 10 and max(y for _, y in top) <= 50
    layout = np.array([[1, 1, 9, 9], [2, 2, 10, 10]])
    assert coarse_layout(layout, columns=4, rows=2) == [" 1  1  9  9", " 2  2 10 10"]
