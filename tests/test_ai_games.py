import random

import numpy as np
import pytest

from hoi4_arena.ai_games import MAP_BOTTOM, MAP_TOP, OK_MATCH, Popups, arena_offset
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
