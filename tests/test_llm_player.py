import numpy as np
import pytest

from hoi4_arena.llm_player import VIEW, changed, cost, parse_turn, to_events, view


def test_parse_turn_reads_json_inside_words():
    turn = parse_turn(
        'here:\n```json\n{"actions": [{"type": "key", "key": "z"}], "run_seconds": 2}\n```'
    )
    assert turn["actions"] == [{"type": "key", "key": "z"}]
    assert turn["run_seconds"] == 2


def test_parse_turn_without_json_asks_for_no_game_time():
    assert parse_turn("<tool call markup>")["run_seconds"] == 0
    assert parse_turn("{not json}")["run_seconds"] == 0


def test_click_maps_view_pixels_to_screen_fractions():
    events = to_events({"type": "click", "x": VIEW[0] / 2, "y": VIEW[1] / 4, "button": "right"})
    assert events[0] == {"kind": "move", "x": 0.5, "y": 0.25}
    assert events[1:] == [
        {"kind": "button", "button": 1, "down": True},
        {"kind": "button", "button": 1, "down": False},
    ]


def test_modifiers_wrap_the_action():
    events = to_events({"type": "key", "key": "z", "mods": ["shift", "ctrl"]})
    assert [e["vk"] for e in events] == [0x10, 0x11, ord("Z"), ord("Z"), 0x11, 0x10]
    assert [e["down"] for e in events] == [True, True, True, False, False, False]


def test_drag_holds_the_button_across_moves():
    events = to_events({"type": "drag", "x1": 0, "y1": 0, "x2": 672, "y2": 378, "button": "right"})
    downs = [i for i, e in enumerate(events) if e["kind"] == "button"]
    assert len(downs) == 2 and events[downs[0]]["down"] and not events[downs[1]]["down"]
    assert events[downs[1] - 1] == {"kind": "move", "x": 0.5, "y": 0.5}


@pytest.mark.parametrize(
    "action",
    [
        {"type": "key", "key": "space"},
        {"type": "key", "key": "delete"},
        {"type": "click", "x": 10, "y": 10, "mods": ["alt"]},
        {"type": "click", "x": VIEW[0] + 5, "y": 10},
        {"type": "teleport"},
    ],
)
def test_what_the_harness_keeps_is_refused(action):
    with pytest.raises((ValueError, KeyError)):
        to_events(action)


def test_cost_counts_cache_hits_apart_and_doubles_at_peak():
    usage = {"prompt_cache_hit_tokens": 1_000_000, "prompt_cache_miss_tokens": 0,
             "completion_tokens": 1_000_000}  # fmt: skip
    assert cost(usage) == pytest.approx(0.603)
    assert cost(usage, at_peak=True) == pytest.approx(1.206)


def test_changed_ignores_a_still_screen_and_sees_a_new_one():
    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    screen[:, :960] = 200
    assert changed(screen, screen) == 0
    assert changed(screen, 255 - screen) > 0.9


def test_view_is_the_models_size():
    assert view(np.zeros((1080, 1920, 3), dtype=np.uint8)).size == VIEW
