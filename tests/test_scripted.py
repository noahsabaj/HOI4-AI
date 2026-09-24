import random

import numpy as np
import pytest

from hoi4_arena.scripted import ATTACKS, Planner, choose_plan, state_at, wilson, win_rate


def test_plans_cover_every_attack_and_never_redraw_without_one():
    rng = random.Random(0)
    plans = [choose_plan(rng) for _ in range(400)]
    assert {p["attack"] for p in plans} == set(ATTACKS)
    assert all(p["redraw"] is None for p in plans if p["attack"] == "none")
    assert all(p["wait"] == 0 or 5 <= p["wait"] <= 60 for p in plans)
    assert any(p["wait"] == 0 for p in plans) and any(p["redraw"] for p in plans)


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


def test_the_win_rate_counts_decided_games_by_side_and_attack():
    games = [
        {"started_as": "BLU", "winner": "BLU", "plan": {"attack": "deep"}},
        {"started_as": "BLU", "winner": "RED", "plan": {"attack": "near"}},
        {"started_as": "RED", "winner": "RED", "plan": {"attack": "near"}},
        {"started_as": "RED", "winner": "timeout", "plan": {"attack": "none"}},
        {"started_as": "BLU", "error": "RuntimeError: no army"},
    ]
    report = win_rate(games)
    assert report["errors"] == 1
    assert report["all"] == {
        "games": 4, "decided": 3, "wins": 2, "rate": 0.667,
        "interval95": [round(x, 3) for x in wilson(2, 3)],
    }  # fmt: skip
    assert report["BLU"]["wins"] == 1 and report["RED"]["decided"] == 1
    assert report["near"]["decided"] == 2 and report["none"]["decided"] == 0
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
