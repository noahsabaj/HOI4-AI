import random

import numpy as np
import pytest

from hoi4_arena.scripted import ATTACKS, Planner, choose_plan, state_at, wilson, win_rate


def test_plans_cover_every_attack_and_always_redraw():
    rng = random.Random(0)
    plans = [choose_plan(rng) for _ in range(400)]
    assert {p["attack"] for p in plans} == set(ATTACKS)
    assert all(6 <= p["wait"] <= 240 for p in plans)
    assert sum(p["wait"] >= 90 for p in plans) > 0.7 * len(plans)
    assert all(30 <= p["redraw"] <= 90 for p in plans)


def test_points_map_to_the_states_mapgen_numbers():
    # Blue's states run down each column pair from the west; Red's are Blue's turned half
    # a turn, so its border states 15 and 16 face Blue's 7 and 8 across the seam.
    assert state_at(0.0, 0.0) == 1 and state_at(0.0, 0.99) == 2
    assert state_at(0.49, 0.1) == 7 and state_at(0.49, 0.9) == 8
    assert state_at(0.51, 0.9) == 15 and state_at(0.51, 0.1) == 16
    assert state_at(0.99, 0.9) == 9 and state_at(0.99, 0.1) == 10
    assert sorted({state_at(u / 48, v / 16) for u in range(48) for v in range(16)}) == list(
        range(1, 17)
    )


def test_a_layout_names_states_off_the_arena_itself(tmp_path):
    import json

    from hoi4_arena.scripted import arena_layout

    # A 4x2 layout with water (0) in its top right corner, as a bay leaves.
    (tmp_path / "generation.json").write_text(
        json.dumps({"layout": {"box": [0, 0, 8, 4], "states": ["1 7 16 0", "2 8 15 9"]}})
    )
    layout = arena_layout(tmp_path)
    assert state_at(0.1, 0.1, layout) == 1 and state_at(0.6, 0.9, layout) == 15
    # Over water the nearest land's state: the bay's corner is next to state 16 and 9.
    assert state_at(0.95, 0.1, layout) in (16, 9)
    # An arena from before the layouts has none, and the grid is used.
    assert arena_layout(tmp_path / "missing") is None


def test_the_front_is_where_the_two_countries_touch():
    blue = np.zeros((10, 20), bool)
    red = np.zeros((10, 20), bool)
    blue[:, :10], red[:, 10:] = True, True
    planner = Planner("BLU", choose_plan(random.Random(1)), {}, None, 5, frame=lambda: 0)
    front = planner.front(blue, red)
    # On the enemy's side: Red's columns next to Blue's for Blue, Blue's for Red.
    assert front and {x for x, _ in front} <= {10, 11, 12}
    red_front = Planner("RED", choose_plan(random.Random(1)), {}, None, 5, frame=lambda: 0)
    assert {x for x, _ in red_front.front(blue, red)} <= {7, 8, 9}
    assert planner.box_point((0, 0, 10, 20), 10, 5) == [0.5, 0.5]


def test_thin_lines_the_map_draws_are_not_land():
    from hoi4_arena.scripted import clean

    blue = np.zeros((100, 200), bool)
    red = np.zeros((100, 200), bool)
    blue[:, :100], red[:, 100:190] = True, True
    # The blue glow along Red's outer coast, and an offensive's red arrow inside Blue.
    blue[:, 190:192] = True
    red[50:52, 20:90] = True
    planner = Planner("BLU", choose_plan(random.Random(1)), {}, None, 5, frame=lambda: 0)
    assert max(x for x, _ in planner.front(blue, red)) > 180  # Taken for the front.
    blue, red = clean(blue), clean(red)
    assert {x for x, _ in planner.front(blue, red)} <= {100, 101, 102}
    red_side = Planner("RED", choose_plan(random.Random(1)), {}, None, 5, frame=lambda: 0)
    assert {x for x, _ in red_side.front(blue, red)} <= {97, 98, 99}


def test_a_broad_offensive_runs_the_front_s_length_a_third_of_the_way_in():
    front = [(100, y) for y in range(5, 96)]
    box = (0, 0, 100, 190)
    blue = Planner("BLU", choose_plan(random.Random(1)), {}, None, 5, frame=lambda: 0)
    line = blue.broad_line(front, box)
    assert len(line) == 9 and line[0][1] < 10 and line[-1][1] > 90
    assert all(x == pytest.approx(130) for x, _ in line)
    red = Planner("RED", choose_plan(random.Random(1)), {}, None, 5, frame=lambda: 0)
    assert all(x == pytest.approx(100 - 100 / 3) for x, _ in red.broad_line(front, box))


def test_the_win_rate_counts_decided_games_by_side_and_attack():
    games = [
        {"started_as": "BLU", "winner": "BLU", "plan": {"attack": "deep"}},
        {"started_as": "BLU", "winner": "RED", "seconds": 150, "plan": {"attack": "near"}},
        {"started_as": "RED", "winner": "RED", "plan": {"attack": "near"}},
        {"started_as": "RED", "winner": "timeout", "plan": {"attack": "broad"}},
        {"started_as": "BLU", "error": "RuntimeError: no army"},
    ]
    report = win_rate(games)
    assert report["errors"] == 1
    assert report["all"] == {
        "games": 4, "decided": 3, "wins": 2, "rate": 0.667,
        "interval95": [round(x, 3) for x in wilson(2, 3)], "lost_after_s": 150,
    }  # fmt: skip
    assert report["BLU"]["wins"] == 1 and report["RED"]["decided"] == 1
    assert report["near"]["decided"] == 2 and report["broad"]["decided"] == 0
    low, high = wilson(50, 100)
    assert low == pytest.approx(0.404, abs=1e-3) and high == pytest.approx(0.596, abs=1e-3)


def test_the_create_army_plus_is_found_by_its_green_however_brightly_it_glows():
    from hoi4_arena.scripted import green_plus

    screen = np.full((1080, 1920, 3), 40, np.uint8)
    assert green_plus(screen) is None  # Grey: nothing selected.
    for glow in (120, 200):
        lit = screen.copy()
        lit[1000:1030, 985:992] = (60, glow, 60)
        lit[1012:1018, 975:1002] = (60, glow, 60)
        x, y = green_plus(lit)
        assert abs(x * 1920 - 988.5) < 2 and abs(y * 1080 - 1015) < 3


def test_the_console_types_an_event_id_with_its_period():
    from unittest.mock import Mock

    from hoi4_arena import ai_games

    desk = Mock()
    desk.focus.return_value = True
    ai_games.console(desk, "event arena.1")
    keys = [
        call.args[0][0]["vk"]
        for call in desk.apply.call_args_list
        if call.args[0][0]["kind"] == "key" and call.args[0][0]["down"]
    ]
    # Grave, then e v e n t, space, a r e n a, period, 1, enter, grave.
    assert keys == [0xC0, *b"EVENT", 0x20, *b"ARENA", 0xBE, ord("1"), 0x0D, 0xC0]


def test_the_conscription_law_is_read_from_its_slot_alone():
    from hoi4_arena.scripted import CONSCRIPTION, LAW_SLOT

    generator = np.random.default_rng(0)
    limited = generator.integers(0, 255, (44, 44, 3), dtype=np.uint8)
    volunteer = generator.integers(0, 255, (44, 44, 3), dtype=np.uint8)
    planner = Planner(
        "BLU", choose_plan(random.Random(1)), {"law_limited": limited, "law_volunteer": volunteer},
        None, 5, frame=lambda: 0,
    )  # fmt: skip
    screen = np.zeros((1080, 1920, 3), np.uint8)
    # The open list shows every law's icon; only the slot's own says which is in force.
    screen[300:344, 600:644] = limited
    x0, y0, x1, y1 = LAW_SLOT
    screen[y0:y1, x0:x1] = volunteer
    assert planner.law(screen) == "volunteer"
    screen[y0:y1, x0:x1] = limited
    assert planner.law(screen) == "limited"
    plans = [choose_plan(random.Random(i)) for i in range(300)]
    assert {p["conscription"] for p in plans} == set(CONSCRIPTION)


def test_the_political_screen_is_opened_and_closed_by_looking(monkeypatch):
    from hoi4_arena import scripted

    noise = np.random.default_rng(0)
    title = noise.integers(0, 255, (20, 60, 3), dtype=np.uint8)
    background = noise.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    state = {"open": False, "presses": 0}

    def fake_screen(desk):
        rgb = background.copy()
        if state["open"]:
            rgb[94:114, 40:100] = title
        return rgb

    def fake_act(desk, events, pause=0.0):
        # Q toggles the screen, as in the game.
        if any(e.get("vk") == scripted.POLITICS and e.get("down") for e in events):
            state["open"] = not state["open"]
            state["presses"] += 1

    monkeypatch.setattr(scripted, "screen", fake_screen)
    monkeypatch.setattr(scripted, "act", fake_act)
    monkeypatch.setattr(scripted.time, "sleep", lambda seconds: None)
    templates = {"political_title": title}
    planner = Planner("BLU", choose_plan(random.Random(1)), templates, None, 5, frame=lambda: 0)
    assert planner.politics(None, True) and state == {"open": True, "presses": 1}
    assert planner.politics(None, True) and state["presses"] == 1  # Open already: no press.
    assert planner.politics(None, False) and state == {"open": False, "presses": 2}


def test_the_conscription_ladder_climbs_one_paid_step_at_a_time(monkeypatch):
    from hoi4_arena import scripted

    noise = np.random.default_rng(1)
    shapes = {
        name: noise.integers(0, 255, size, dtype=np.uint8)
        for name, size in [
            ("political_title", (18, 50, 3)),
            ("law_list", (18, 50, 3)),
            ("confirm_ok", (18, 50, 3)),
            ("law_volunteer", (30, 30, 3)),
            ("law_limited", (30, 30, 3)),
        ]
    }
    background = noise.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    game = {"political": False, "list": False, "confirm": None, "law": 0, "power": 150}
    pointer = [0.0, 0.0]

    def fake_screen(desk):
        rgb = background.copy()

        def put(name, y, x):
            h, w = shapes[name].shape[:2]
            rgb[y : y + h, x : x + w] = shapes[name]

        if game["political"]:
            put("political_title", 94, 40)
            if game["law"] < 2:
                put(("law_volunteer", "law_limited")[game["law"]], 575, 42)
        if game["list"]:
            put("law_list", 86, 700)
        if game["confirm"] is not None:
            put("confirm_ok", 660, 1030)
        return rgb

    def near(x, y, at):
        return abs(x - at[0]) < 5 and abs(y - at[1]) < 5

    def fake_act(desk, events, pause=0.0):
        for e in events:
            if e["kind"] == "move":
                pointer[:] = e["x"] * 1920, e["y"] * 1080
            elif e["kind"] == "key" and e["vk"] == scripted.POLITICS and e["down"]:
                game["political"], game["list"] = not game["political"], False
            elif e["kind"] == "button" and e["down"]:
                x, y = pointer
                if game["confirm"] is not None:
                    # OK takes only with the power to pay; the list then closes itself.
                    if near(x, y, scripted.CONFIRM) and game["power"] >= 150:
                        game["law"], game["confirm"], game["list"] = game["confirm"], None, False
                        game["power"] -= 150
                    elif near(x, y, scripted.CANCEL):
                        game["confirm"] = None
                elif game["list"]:
                    for i, law in enumerate(scripted.LAWS[1:], 1):
                        if near(x, y, scripted.LAW_ROWS[law]):
                            game["confirm"] = i
                    if near(x, y, scripted.CLOSE_LIST):
                        game["list"] = False
                elif game["political"] and 38 <= x <= 82 and 571 <= y <= 615:
                    game["list"] = True

    monkeypatch.setattr(scripted, "screen", fake_screen)
    monkeypatch.setattr(scripted, "act", fake_act)
    monkeypatch.setattr(scripted.time, "sleep", lambda seconds: None)
    plan = {**choose_plan(random.Random(1)), "conscription": "service"}
    planner = Planner("BLU", plan, shapes, None, 5, frame=lambda: 0)
    assert planner.raise_conscription(None) is False and game["law"] == 1
    # No power left: the step is cancelled, and every panel is closed again.
    assert planner.raise_conscription(None) is False and game["law"] == 1
    assert game == {"political": False, "list": False, "confirm": None, "law": 1, "power": 0}
    game["power"] = 300
    assert planner.raise_conscription(None) is False and game["law"] == 2
    assert planner.raise_conscription(None) is True and game["law"] == 3
    assert [o["law"] for o in planner.orders] == ["limited", "extensive", "service"]
    assert not game["political"]


def test_a_plan_shows_as_the_army_card_s_red_stop_button():
    from hoi4_arena.scripted import STOP_BUTTON, plan_shown

    screen = np.full((1080, 1920, 3), 30, np.uint8)
    assert not plan_shown(screen)
    x0, y0, x1, y1 = STOP_BUTTON
    for red in (160, 220):  # Executing (darker) and idle.
        screen[y0:y1, x0:x1] = (red, 40, 40)
        assert plan_shown(screen)


def test_a_plan_executes_only_once_its_arrow_is_neither_idle_nor_ready(monkeypatch):
    from hoi4_arena import scripted

    noise = np.random.default_rng(2)
    shapes = {n: noise.integers(0, 255, (20, 32, 3), dtype=np.uint8) for n in ("activate", "ready")}
    background = noise.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    looks = {"arrow": None, "plan": True}

    def fake_screen(desk):
        rgb = background.copy()
        x0, y0, x1, y1 = scripted.STOP_BUTTON
        rgb[y0:y1, x0:x1] = (200, 40, 40) if looks["plan"] else (30, 30, 30)
        if looks["arrow"]:
            rgb[949:969, 953:985] = shapes[looks["arrow"]]
        return rgb

    monkeypatch.setattr(scripted, "screen", fake_screen)
    planner = Planner("BLU", choose_plan(random.Random(1)), shapes, None, 5, frame=lambda: 0)
    for arrow, executing in (("activate", False), ("ready", False), (None, True)):
        looks["arrow"] = arrow
        assert planner.lit(None) is executing
    looks["plan"] = False
    assert planner.lit(None) is False
