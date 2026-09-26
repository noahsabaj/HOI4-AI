import json

import numpy as np
import pytest
from test_learned import _scripted

from hoi4_arena import practice
from hoi4_arena.dataset import session_labels

LAND = np.zeros((1080, 1920, 3), np.uint8)
# What the coach's screenshot shows, as a test sets it.
SCREEN = {"rgb": LAND}


class _Planner:
    """The scripted player's steps and its template search, as the test sets them."""

    def __init__(self):
        self.steps = []
        self.fail = set()
        self.shown = set()

    def find(self, rgb, name, top=0.0):
        return (0.5, 0.5) if name in self.shown else None

    def form_army(self, desk):
        self._do("army", desk)

    def assign_general(self, desk):
        self._do("general", desk)
        return True

    def draw_front(self, desk):
        self._do("front", desk)

    def _do(self, step, desk):
        if step in self.fail:
            raise RuntimeError(f"no {step}")
        self.steps.append(step)
        desk.apply([{"kind": "move", "x": 0.5, "y": 0.5}])


class _Desk:
    def __init__(self):
        self.applied = []

    def apply(self, events):
        self.applied.extend(events)
        return {"t_ns": 7}


def _card(general):
    """A screen with the first army's card lit, and a portrait on it if `general`."""
    rgb = LAND.copy()
    x0, y0, x1, y1 = practice.CARD
    rgb[y0:y1, x0:x1] = (200, 150, 110) if general else (90, 90, 90)
    return rgb


def test_the_army_card_says_whether_there_is_an_army_and_a_general():
    assert practice.army_card(LAND) == (False, False), "the dark gap between two + slots"
    assert practice.army_card(_card(False)) == (True, False)
    assert practice.army_card(_card(True)) == (True, True)


@pytest.fixture
def coach(monkeypatch):
    """A coach whose screen shows land (no drift), no army and no plan, over a fake planner."""
    SCREEN["rgb"] = LAND
    monkeypatch.setattr("hoi4_arena.ai_games.screen", lambda desk: SCREEN["rgb"])
    monkeypatch.setattr("hoi4_arena.scripted.plan_shown", lambda rgb: False)
    monkeypatch.setattr("hoi4_arena.vision.country_pixels", lambda crop: (np.ones(1), None))
    made = practice.Coach("BLU", None, planner=_Planner(), plan={"attack": "broad"})
    frames = iter(range(0, 1000, 10))
    made.frames = lambda: next(frames)
    return made


def test_the_coach_takes_over_a_late_step_in_order_and_records_it(coach):
    desk = _Desk()
    assert coach.look(desk, 5.0, running=False) is None, "the army is not late yet"
    assert coach.look(desk, 21.0, running=False) == "army"
    taken = coach.take_over(desk, "army", 21.0)
    assert [e["by"] for e in taken] == ["coach"] and taken[0]["t_ns"] == 7
    assert coach.done["army"] == {"at": 21.0, "by": "coach", "confirmed": None}
    # The general is due at 35 s; the front waits for it.
    assert coach.look(desk, 30.0, running=False) is None
    assert coach.look(desk, 61.0, running=False) == "general"


def test_what_the_policy_did_itself_is_its_own_and_never_taken_over(coach):
    SCREEN["rgb"] = _card(True)  # an army, with a general
    assert coach.look(_Desk(), 4.0, running=False) is None
    assert coach.done["army"]["by"] == coach.done["general"]["by"] == "policy"
    assert coach.look(_Desk(), 70.0, running=True) == "front"
    score = coach.score()
    assert score["own"] == 3 and score["steps"]["front"] is None, "army, general, running"


def test_a_camera_off_the_arena_is_brought_back(coach, monkeypatch):
    monkeypatch.setattr("hoi4_arena.vision.country_pixels", lambda crop: (None, None))
    recentred = []
    monkeypatch.setattr("hoi4_arena.ai_games.recentre", lambda desk: recentred.append(1))
    assert coach.look(_Desk(), 3.0, running=False) is None, "lost for 0 s"
    assert coach.look(_Desk(), 9.0, running=False) == "camera", "lost for 6 s"
    coach.take_over(_Desk(), "camera", 9.0)
    assert recentred == [1] and coach.lost_since is None
    assert coach.coached[-1]["step"] == "camera"


def test_an_army_counts_only_once_no_division_is_left_out(coach):
    SCREEN["rgb"] = _card(False)
    coach.planner.shown = {"unassigned"}  # an army of one division; the alert stays up
    assert coach.look(_Desk(), 5.0, running=False) is None and "army" not in coach.done
    coach.planner.shown = set()
    coach.look(_Desk(), 9.0, running=False)
    assert coach.done["army"] == {"at": 9.0, "by": "policy"}


def test_a_step_the_coach_cannot_do_is_given_up_not_retried(coach):
    coach.planner.fail = {"army"}
    assert coach.look(_Desk(), 21.0, running=False) == "army"
    coach.take_over(_Desk(), "army", 21.0)
    assert coach.failures[0]["step"] == "army" and coach.done["army"]["by"] == "nobody"
    assert coach.look(_Desk(), 40.0, running=False) == "general"


def test_the_summary_counts_the_policy_s_own_steps():
    own = {"at": 1.0, "by": "policy"}
    episodes = [
        {
            "setup": {
                "steps": {"army": own, "general": None, "front": None, "running": own},
                "own": 2,
            }
        },
        {"setup": {"steps": {"army": {"at": 21.0, "by": "coach"}}, "own": 0}},
        {"error": "the game did not start"},
    ]
    summary = practice.summary(episodes)
    assert summary == {"episodes": 2, "army": "1/2", "general": "0/2", "front": "0/2",
                       "running": "1/2", "coach_army": "1/1", "own_steps_mean": 1.0,
                       "temperature": 1.0, "pointer_temperature": 1.0}  # fmt: skip


def test_a_practice_game_teaches_only_what_the_coach_did(tmp_path):
    _scripted(tmp_path / "game", coached=[{"from_frame": 10, "to_frame": 20, "step": "army"}])
    path = tmp_path / "game" / "manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "source": "policy"}))
    labels = session_labels(tmp_path / "game", sources=("policy",), lead_in=0)
    times, decisions = labels["times"], labels["decisions"]
    taught = (decisions >= times[10]) & (decisions <= times[20])
    assert taught.any() and (~taught).any()
    assert (labels["weight"][taught] == 1).all() and (labels["weight"][~taught] == 0).all()


def test_drills_stay_on_an_arena_for_a_block_with_the_countries_alternating():
    order = practice.drill_order(["a", "b"], ["BLU", "RED"], 6, 2)
    assert order == [
        ("a", "BLU"),
        ("a", "RED"),
        ("b", "BLU"),
        ("b", "RED"),
        ("a", "BLU"),
        ("a", "RED"),
    ]


def test_every_arena_with_a_start_save_can_be_drilled(tmp_path):
    registry = tmp_path / "saves.json"
    registry.write_text(
        json.dumps({"arena-bay-v6": {"BLU": "arenabayv6blu", "RED": "arenabayv6red"}})
    )
    saves = practice.drill_saves(registry)
    assert saves[("arena-bay-v6", "RED")] == "arenabayv6red"
    assert saves[(practice.MAIN_ARENA, "BLU")] == "arenav4blu", "the main arena's own"


def test_a_scramble_goes_straight_to_the_desktop_and_says_what_it_did(monkeypatch):
    import random

    done = []
    monkeypatch.setattr(
        "hoi4_arena.ai_games.kick_camera", lambda desk, rng, zoom: done.append("camera")
    )
    monkeypatch.setattr("hoi4_arena.ai_games.click", lambda desk, x, y: done.append("click"))
    monkeypatch.setattr("hoi4_arena.ai_games.act", lambda desk, events: done.append(events))
    monkeypatch.setattr("hoi4_arena.practice.time.sleep", lambda s: None)
    for seed in range(20):
        done.clear()
        kinds = practice.scramble(object(), random.Random(seed))
        assert 1 <= len(kinds) <= 3 and len(done) == len(kinds)
        assert set(kinds) <= set(practice.SCRAMBLES)


def test_a_takeover_counts_once_the_screen_shows_the_step_done():
    coach = practice.Coach("BLU", None, planner=_Planner(), plan={"attack": "broad"})
    coach.seen("front", 60.0, by="coach")
    assert coach.done["front"] == {"at": 60.0, "by": "coach", "confirmed": None}
    assert not practice.coach_managed(coach.done["front"])
    coach.seen("front", 64.0)
    assert coach.done["front"] == {"at": 60.0, "by": "coach", "confirmed": 64.0}
    assert practice.coach_managed(coach.done["front"])
    assert practice.coach_managed({"at": 60.0, "by": "coach"}), "a record from before"
    assert not practice.coach_managed({"at": 60.0, "by": "nobody"})


@pytest.mark.parametrize(
    "front, teaches",
    [
        ({"at": 60.0, "by": "coach", "confirmed": 64.0}, True),
        ({"at": 60.0, "by": "coach"}, True),
        ({"at": 60.0, "by": "coach", "confirmed": None}, False),
        ({"at": 60.0, "by": "nobody"}, False),
    ],
)
def test_a_takeover_that_left_its_step_undone_teaches_nothing(tmp_path, front, teaches):
    _scripted(tmp_path / "game", coached=[{"from_frame": 10, "to_frame": 20, "step": "front"}])
    path = tmp_path / "game" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(source="policy", setup={"steps": {"front": front}, "own": 0})
    path.write_text(json.dumps(manifest))
    labels = session_labels(tmp_path / "game", sources=("policy",), lead_in=0)
    assert (labels["weight"] > 0).any() == teaches
