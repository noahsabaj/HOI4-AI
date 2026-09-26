import json

import numpy as np
import pytest
from test_learned import _scripted

from hoi4_arena import practice
from hoi4_arena.dataset import session_labels

LAND = np.zeros((1080, 1920, 3), np.uint8)


class _Planner:
    """The scripted player's screen checks and steps, as the test sets them."""

    def __init__(self):
        self.shown = {"unassigned"}
        self.steps = []
        self.fail = set()

    def find(self, rgb, name, top=0.0):
        return (0.5, 0.5) if name in self.shown else None

    def form_army(self, desk):
        self._do("army", desk)
        self.shown.discard("unassigned")

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


@pytest.fixture
def coach(monkeypatch):
    """A coach whose screen shows land (no drift) and no plan, over a fake planner."""
    monkeypatch.setattr("hoi4_arena.ai_games.screen", lambda desk: LAND)
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
    assert coach.done["army"] == {"at": 21.0, "by": "coach"}
    # The general is due at 35 s; the front waits for it.
    assert coach.look(desk, 30.0, running=False) is None
    assert coach.look(desk, 61.0, running=False) == "general"


def test_what_the_policy_did_itself_is_its_own_and_never_taken_over(coach):
    coach.planner.shown = {"plans_bar"}  # the army formed and selected, with a commander
    assert coach.look(_Desk(), 4.0, running=False) is None
    assert coach.done["army"]["by"] == coach.done["general"]["by"] == "policy"
    assert coach.look(_Desk(), 70.0, running=True) == "front"
    score = coach.score()
    assert score["own"] == 3 and score["steps"]["front"] is None, "army, general, running"


def test_a_camera_off_the_arena_is_brought_back(coach, monkeypatch):
    monkeypatch.setattr("hoi4_arena.vision.country_pixels", lambda crop: (None, None))
    recentred = []
    monkeypatch.setattr("hoi4_arena.ai_games.recentre", lambda desk: recentred.append(1))
    coach.planner.shown = set()
    assert coach.look(_Desk(), 3.0, running=False) is None, "lost for 0 s"
    assert coach.look(_Desk(), 9.0, running=False) == "camera", "lost for 6 s"
    coach.take_over(_Desk(), "camera", 9.0)
    assert recentred == [1] and coach.lost_since is None
    assert coach.coached[-1]["step"] == "camera"


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
                       "running": "1/2", "own_steps_mean": 1.0}  # fmt: skip


def test_a_practice_game_teaches_only_what_the_coach_did(tmp_path):
    _scripted(tmp_path / "game", coached=[{"from_frame": 10, "to_frame": 20, "step": "army"}])
    path = tmp_path / "game" / "manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "source": "policy"}))
    labels = session_labels(tmp_path / "game", sources=("policy",), lead_in=0)
    times, decisions = labels["times"], labels["decisions"]
    taught = (decisions >= times[10]) & (decisions <= times[20])
    assert taught.any() and (~taught).any()
    assert (labels["weight"][taught] == 1).all() and (labels["weight"][~taught] == 0).all()
