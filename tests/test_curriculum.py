"""The curriculum that skips the setup (curriculum.py): rung saves' names and registry, the
cut of recorded games at a rung, the scoring of a rung game, the promotion rule and the
adaptive frontier schedule."""

import json
import random

import numpy as np
import pytest
from test_learned import _scripted

from hoi4_arena import curriculum, practice
from hoi4_arena.dataset import session_labels
from hoi4_arena.scripted import ARROW, STOP_BUTTON


def test_rung_saves_are_named_after_their_last_step_and_none_contains_another():
    names = {
        curriculum.rung_save(start, rung)
        for start in ("arenav4blu", "arenav4red", "arenamarshv6blu", "arenaplainsv6red")
        for rung in curriculum.RUNGS
    }
    assert curriculum.rung_save("arenav4blu", "S0") == "arenav4blu"
    assert curriculum.rung_save("arenav4blu", "S1") == "armyv4blu"
    assert curriculum.rung_save("arenav4red", "S2") == "generalv4red"
    assert curriculum.rung_save("arenamarshv6blu", "S3") == "frontmarshv6blu"
    # The load dialog's templates match a name anywhere in a row: none may hold another.
    assert not [(a, b) for a in names for b in names if a != b and a in b]
    assert all(n.isalnum() for n in names)
    with pytest.raises(ValueError):
        curriculum.rung_save("arenav4blu", "S4")


def test_the_ladder_lists_every_rung_with_the_start_saves_as_s0(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ladder = curriculum.ladder("here")
    assert ladder[("arena-12x8-v4", "BLU", "S0")] == "arenav4blu"
    assert ("arena-12x8-v4", "BLU", "S3") not in ladder
    path = curriculum.registry_path("here")
    curriculum.remember_rung(path, "arena-12x8-v4", "BLU", "S3", "frontv4blu")
    curriculum.remember_rung(path, "arena-12x8-v4", "BLU", "S1", "armyv4blu")
    ladder = curriculum.ladder("here")
    assert ladder[("arena-12x8-v4", "BLU", "S3")] == "frontv4blu"
    assert ladder[("arena-12x8-v4", "BLU", "S1")] == "armyv4blu"
    assert json.loads(path.read_text())["arena-12x8-v4"]["BLU"] == {
        "S3": "frontv4blu",
        "S1": "armyv4blu",
    }


def test_the_worker_s_save_listing_gives_the_names():
    listing = """
name           MB LastWriteTime
----           -- -------------
frontv4blu    1.2 9/26/2026 6:40:01 PM
arenav4blu    1.1 9/26/2026 6:30:12 PM
"""
    assert curriculum.parse_saves(listing) == ["frontv4blu", "arenav4blu"]
    assert curriculum.parse_saves("saves: none") == []


def test_a_recorded_game_is_cut_where_its_rung_would_start(tmp_path):
    times = np.arange(40) * 200_000_000
    orders = [
        {"order": "army", "frame": 3},
        {"order": "general", "frame": 6},
        {"order": "front", "frame": 9},
        {"order": "offensive", "frame": 12},
        {"order": "run", "frame": 14},
    ]
    manifest = {"orders": orders}
    assert curriculum.rung_start(manifest, times, None) == times[0]
    assert curriculum.rung_start(manifest, times, "S0") == times[0]
    assert curriculum.rung_start(manifest, times, "S1") == times[3]
    assert curriculum.rung_start(manifest, times, "S2") == times[6]
    assert curriculum.rung_start(manifest, times, "S3") == times[12]
    # A plan with no offensive: its front completes S3.
    assert curriculum.rung_start({"orders": orders[:3]}, times, "S3") == times[9]
    # A game played from a rung save starts at its rung already.
    assert curriculum.rung_start({"orders": orders, "rung": "S3"}, times, "S3") == times[0]

    _scripted(tmp_path / "game", orders=[{"order": "offensive", "frame": 20}])
    whole = session_labels(tmp_path / "game", sources=("scripted",), lead_in=0)
    cut = session_labels(tmp_path / "game", sources=("scripted",), lead_in=0, rung="S3")
    early = cut["decisions"] < cut["times"][20]
    assert early.any() and (~early).any()
    assert (cut["weight"][early] == 0).all()
    np.testing.assert_array_equal(cut["weight"][~early], whole["weight"][~early])
    # The windows stay: only the weights change.
    np.testing.assert_array_equal(cut["valid"], whole["valid"])


class _Planner:
    def __init__(self, shown=()):
        self.shown = set(shown)

    def find(self, rgb, name, top=0.0):
        return (0.5, 0.5) if name in self.shown else None


def _screen(army=False, general=False, plan=False, executing=False):
    rgb = np.zeros((1080, 1920, 3), np.uint8)
    x0, y0, x1, y1 = practice.CARD
    if army:
        rgb[y0:y1, x0:x1] = (200, 150, 110) if general else (90, 90, 90)
    if plan:
        x0, y0, x1, y1 = STOP_BUTTON
        rgb[y0:y1, x0:x1] = (200, 20, 20)
    if executing:
        x0, y0, x1, y1 = ARROW
        rgb[y0:y1, x0:x1] = (40, 250, 40)
    return rgb


def test_a_rung_game_scores_what_the_rung_held_and_what_the_policy_did():
    watch = curriculum.RungWatch("BLU", None, "S3", planner=_Planner())
    assert watch.score()["steps"]["army"] == {"at": 0.0, "by": "rung"}
    assert watch.score()["steps"]["execute"] is None
    # A plan that shows but waits (not executing) is not executed, nor is one before running.
    watch.read(_screen(True, True, True, executing=True), running=False, seconds=4)
    assert "execute" not in watch.done
    watch.read(_screen(True, True, True, executing=True), running=True, seconds=40)
    score = watch.score()
    assert score["steps"]["execute"] == {"at": 40, "by": "policy"}
    assert score["steps"]["front"]["by"] == "rung"
    assert score["own"] == 1 and score["rung"] == "S3"

    waiting = curriculum.RungWatch("BLU", None, "S1", planner=_Planner({"ready"}))
    waiting.read(_screen(True, True, True, executing=True), running=True, seconds=9)
    assert waiting.done["general"]["by"] == "policy"
    assert "execute" not in waiting.done, "a ready (green check) plan is not executing"


def _game(rung, winner, started_as="BLU", execute=True, arena="arena-12x8-v4"):
    steps = {s: {"by": "rung"} for s in curriculum.DONE[rung] if s in curriculum.WATCHED}
    for step in curriculum.WATCHED:
        steps.setdefault(step, {"by": "policy"} if execute or step != "execute" else None)
    return {"rung": rung, "winner": winner, "started_as": started_as, "arena": arena,
            "seconds": 200, "setup": {"steps": steps}}  # fmt: skip


def test_setup_completion_needs_the_plan_executed():
    assert curriculum.setup_complete(_game("S3", "RED"))
    assert not curriculum.setup_complete(_game("S3", "RED", execute=False))


def test_the_summary_counts_wins_and_setups_per_rung():
    games = [
        _game("S3", "BLU"),
        _game("S3", "RED", execute=False),
        _game("S3", "timeout"),
        _game("S0", "RED"),
        {"rung": "S3", "started_as": "BLU", "error": "no launch"},
    ]
    out = curriculum.summary(games)
    assert out["S3"]["games"] == 3 and out["S3"]["wins"] == 1
    assert out["S3"]["setup_complete"] == 2 and out["S3"]["timeouts"] == 1
    assert out["S3"]["execute"] == "2/3" and out["S3"]["execute_own"] == 2
    assert out["S3"]["by_arena"] == {"arena-12x8-v4:BLU": "1/3"}
    assert out["S0"]["games"] == 1 and out["S0"]["wins"] == 0


def test_a_checkpoint_moves_down_a_rung_after_six_wins_in_its_last_ten():
    games = [_game("S3", "RED")] * 5 + [_game("S3", "BLU")] * 5
    assert curriculum.promote(games, "S3") == "S3", "5 of 10 is under 60%"
    games += [_game("S3", "BLU")]
    assert curriculum.promote(games, "S3") == "S2"
    assert curriculum.promote(games[:9], "S3") == "S3", "fewer than 10 games"
    assert curriculum.promote([_game("S0", "BLU")] * 10, "S0") == "S0", "S0 is the bottom"


def test_the_fixed_order_keeps_arenas_in_blocks_and_interleaves_rungs():
    order = curriculum.game_order(["a", "b"], ["BLU", "RED"], ["S3", "S0"], 8, block=4)
    assert [o[0] for o in order] == ["a"] * 4 + ["b"] * 4
    assert [o[1] for o in order] == ["BLU", "RED"] * 4
    assert [o[2] for o in order] == ["S3", "S3", "S0", "S0"] * 2


def test_the_frontier_prefers_cells_near_even_odds_and_untried_ones():
    rng = random.Random(0)
    cells = [("S3", "a", "BLU"), ("S0", "a", "BLU"), ("S2", "a", "BLU")]
    # S3 is won half the time, S0 never, S2 always, each over 12 games with no change.
    history = (
        [_game("S3", w, arena="a") for w in ["BLU", "RED"] * 6]
        + [_game("S0", "RED", execute=False, arena="a")] * 12
        + [_game("S2", "BLU", arena="a")] * 12
    )
    picks = [curriculum.frontier(history, cells, rng=rng)[0] for _ in range(50)]
    assert picks.count(("S3", "a", "BLU")) > 40
    cell, scores = curriculum.frontier(history, cells + [("S1", "a", "BLU")], rng=rng)
    assert cell == ("S1", "a", "BLU"), "an untried cell counts as full progress"
    assert scores[("S1", "a", "BLU")]["games"] == 0
    assert scores[("S0", "a", "BLU")]["mean"] < 0.1


def test_success_credits_a_setup_without_a_win_a_little():
    assert curriculum.success(_game("S3", "BLU")) == 1.0
    assert curriculum.success(_game("S3", "RED")) == 0.25
    assert curriculum.success(_game("S3", "RED", execute=False)) == 0.0
    assert curriculum.success({"rung": "S3", "error": "x"}) is None


def test_the_scripted_player_skips_the_steps_its_rung_save_holds():
    from hoi4_arena import scripted

    planner = scripted.Planner.__new__(scripted.Planner)
    calls = []
    planner.done = frozenset(curriculum.DONE["S3"])
    planner.plan = {"attack": "broad"}
    for name in ("form_army", "assign_general", "draw_front", "draw_offensive", "overview"):
        setattr(planner, name, lambda desk, name=name: calls.append(name))
    planner.rules, planner.speed = None, 5
    planner.orders, planner.frame = [], lambda: 0
    planner.start = lambda now: calls.append("start")
    run = []
    original = scripted.run_at
    scripted.run_at = lambda desk, rules, speed: run.append(speed)
    try:
        planner.setup(None)
    finally:
        scripted.run_at = original
    assert calls == ["overview", "start"] and run == [5]
    assert planner.running and planner.orders[-1]["order"] == "run"


def test_the_scripted_setup_is_read_from_its_orders():
    manifest = {"orders": [{"order": "run"}, {"order": "activate"}]}
    setup = curriculum.scripted_setup(manifest, "S3")
    assert setup["steps"]["front"]["by"] == "rung"
    assert setup["steps"]["execute"]["by"] == "scripted"
    assert curriculum.setup_complete({"setup": setup})
    assert not curriculum.setup_complete({"setup": curriculum.scripted_setup({}, "S3")})
