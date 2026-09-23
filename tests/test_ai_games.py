import random

import numpy as np
import pytest

from hoi4_arena import ai_games
from hoi4_arena.ai_games import MAP_BOTTOM, MAP_TOP, OK_MATCH, Popups, arena_offset, pick_country
from hoi4_arena.vision import find_template

# Land tints measured at 1080p (vision.country_pixels) and the sea around them.
BLUE_LAND, RED_LAND, SEA = (120, 134, 145), (168, 145, 131), (33, 43, 61)


def _button():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (22, 95, 3), dtype=np.uint8)


def test_a_template_is_found_wherever_it_opens_and_nowhere_else():
    frame = np.full((1080, 1920, 3), 40, np.uint8)
    button = _button()
    frame[590:612, 1085:1180] = button
    x, y = find_template(frame, button, OK_MATCH)
    assert x == pytest.approx((1085 + 95 / 2) / 1920)
    assert y == pytest.approx((590 + 22 / 2) / 1080)
    assert find_template(np.full_like(frame, 40), button, OK_MATCH) is None


def test_a_popup_is_clicked_only_after_its_reading_time_and_only_once():
    now = [0.0]
    frame = np.full((540, 960, 3), 40, np.uint8)
    button = _button()
    frame[100:122, 200:295] = button
    popups = Popups([button], rng=random.Random(1), clock=lambda: now[0])
    popups.look(frame)
    assert popups.due() is None, "a player reads the popup first"
    now[0] = 4.0
    at = popups.due()
    assert at == pytest.approx(((200 + 47.5) / 960, 111 / 540))
    assert popups.due() is None


def test_the_arena_offset_points_from_the_screen_centre_to_the_arena_centre():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:] = SEA
    # Blue's half left of the seam, Red's right, both shifted right of centre by 192 px.
    frame[300:700, 700:1152] = BLUE_LAND
    frame[300:700, 1152:1604] = RED_LAND
    down, right = arena_offset(frame)
    assert right == pytest.approx((1152 - 960) / 1920, abs=1e-3)
    assert down == pytest.approx((500 - 540) / 1080, abs=1e-3)
    # The HUD rows are ignored: land drawn only there is not an arena.
    chrome = np.zeros_like(frame)
    chrome[: MAP_TOP - 1] = BLUE_LAND
    chrome[1080 - MAP_BOTTOM + 1 :] = RED_LAND
    assert arena_offset(chrome) is None


def test_picking_a_country_clicks_its_land_until_its_flag_shows(monkeypatch):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:] = SEA
    frame[300:700, 700:1152] = BLUE_LAND
    frame[300:700, 1152:1604] = RED_LAND
    x0, y0, x1, y1 = ai_games.PICKER_FLAG
    frame[y0:y1, x0:x1] = (40, 60, 200)  # Blue, the default, is selected.
    clicks = []

    def click(desk, x, y):
        clicks.append((x, y))
        if 1152 <= x * 1920 < 1604:
            frame[y0:y1, x0:x1] = (200, 40, 40)

    monkeypatch.setattr(ai_games, "screen", lambda desk: frame)
    monkeypatch.setattr(ai_games, "click", click)
    monkeypatch.setattr(ai_games.time, "sleep", lambda s: None)
    monkeypatch.setattr(ai_games, "act", lambda desk, events, pause=0.15: None)
    assert ai_games.picked(frame) == "BLU"
    assert pick_country(None, "BLU") and not clicks, "already selected"
    assert pick_country(None, "RED")
    x, y = clicks[-1]
    assert 1152 <= x * 1920 < 1604 and 300 <= y * 1080 < 700
    assert ai_games.picked(frame) == "RED"
    # No Blue land and Blue not selected: the pick never takes.
    frame[300:700, 700:1152] = SEA
    assert not pick_country(None, "BLU")


def test_the_front_is_where_blue_land_meets_red():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:] = SEA
    frame[300:700, 700:1152] = BLUE_LAND
    frame[300:700, 1152:1604] = RED_LAND
    xs = [x * 1920 for x, _ in ai_games.front_points(frame)]
    ys = [y * 1080 for _, y in ai_games.front_points(frame)]
    assert xs and min(xs) >= 1150 and max(xs) <= 1156
    assert min(ys) >= 300 and max(ys) < 700
    frame[300:700, 1152:1604] = SEA
    assert ai_games.front_points(frame) == []
    assert len(ai_games.land_points(frame)) == 400 * 452


def test_each_speed_is_played_as_both_countries_and_the_pcs_are_out_of_step():
    here = [ai_games.game_plan("here", i, [4, 5]) for i in range(4)]
    peer = [ai_games.game_plan("peer", i, [4, 5]) for i in range(4)]
    assert here == [("BLU", 4), ("RED", 4), ("BLU", 5), ("RED", 5)]
    assert peer == [("RED", 4), ("BLU", 4), ("RED", 5), ("BLU", 5)]
