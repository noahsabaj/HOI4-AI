import json
import random
import types

import pytest

from hoi4_arena import tuning

DAYS = (
    "declare RED\n"
    "day  24:00, 21 April, 1938 BLU states 8 owned 7 divisions 4 surrender 0.05 strength 2\n"
    "day  23:00, 21 April, 1938 RED states 8 owned 7 divisions 8 surrender 0.06 strength 0.5\n"
)


def best(**changes):
    plan = {
        "best": True,
        "variant": "best",
        "conscription": "all_adults",
        "attack": "broad",
        "recruit": 0,
        "wait": 175,
        "redraw": 83,
        "pause_redraw": True,
        "guard": 0.15,
    }
    return {**plan, **changes}


def game(root, name, plan, winner="RED", side="RED", seconds=300, arena="arena-12x8-v4"):
    folder = root / f"scripted-peer-{name}"
    folder.mkdir(parents=True)
    manifest = {"plan": plan, "winner": winner, "started_as": side, "seconds": seconds}
    manifest["arena"] = arena
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "arena-log.txt").write_text(DAYS)
    return folder


def in_process(monkeypatch):
    """Tuner.propose asks in this process, not a subprocess."""

    def run(command, **options):
        return types.SimpleNamespace(stdout=json.dumps(tuning.ask(command[-2])))

    monkeypatch.setattr(tuning.subprocess, "run", run)


def test_a_game_scores_the_surrender_progress_between_the_two_sides(tmp_path):
    (tmp_path / "arena-log.txt").write_text(DAYS)
    # Cut short by the cap: Red had lost a little more than Blue.
    assert tuning.game_score(tmp_path, "RED", "timeout") == pytest.approx(-0.01)
    assert tuning.game_score(tmp_path, "BLU", "timeout") == pytest.approx(0.01)
    # A capitulation is full progress for the side that gave in.
    assert tuning.game_score(tmp_path, "RED", "RED") == pytest.approx(0.94)
    assert tuning.game_score(tmp_path, "RED", "BLU") == pytest.approx(-0.95)
    # Without daily lines, the result alone.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert tuning.game_score(empty, "BLU", "BLU") == 1.0
    assert tuning.game_score(empty, "BLU", "timeout") == 0.0


def test_the_tuner_asks_for_plans_takes_their_scores_and_fails_the_lost(tmp_path, monkeypatch):
    from optuna.trial import TrialState

    in_process(monkeypatch)
    path = tmp_path / "study.db"
    tuner = tuning.Tuner(path)
    plan = tuner.propose(random.Random(0))
    # The best plan with the asked settings, each inside the space.
    assert plan["variant"] == "tuned" and not plan["best"] and plan["trial"] == 0
    assert plan["attack"] == "broad" and plan["pause_redraw"]
    for name, distribution in tuning.space().items():
        value = plan[name]
        if hasattr(distribution, "choices"):
            assert value in distribution.choices
        else:
            assert distribution.low <= value <= distribution.high
    tuner.report(plan, 0.5)
    lost = tuner.propose(random.Random(1))
    # A recorder stopped mid-game: its trial is failed when the study is next opened.
    reopened = tuning.Tuner(path)
    states = {t.number: t.state for t in reopened.study.get_trials(deepcopy=False)}
    assert states == {0: TrialState.COMPLETE, 1: TrialState.FAIL}
    assert reopened.study.trials[0].value == 0.5
    # A game with no result fails its trial.
    failed = reopened.propose(random.Random(2))
    reopened.report(failed, None)
    assert reopened.study.trials[failed["trial"]].state == TrialState.FAIL
    assert lost["trial"] == 1


def test_seeding_adds_the_best_plan_s_finished_games_once(tmp_path):
    runs = tmp_path / "runs"
    game(runs, "1", best())  # A win as Red.
    game(runs, "2", best(wait=130, redraw=40), winner="BLU", side="BLU", arena="arena-plains-v6")
    game(runs, "3", {"variant": "explore", "attack": "near", "wait": 100, "redraw": 50})
    game(runs, "4", best(), seconds=5)  # A broken load.
    game(runs, "5", best(), arena="arena-marsh-v6")  # Skipped.
    game(runs, "6", {**best(), "variant": "tuned", "trial": 0})  # Already in the study.
    path = tmp_path / "study.db"
    assert tuning.seed(path, [runs], skip={"arena-marsh-v6"}) == 2
    assert tuning.seed(path, [runs], skip={"arena-marsh-v6"}) == 0
    shown = tuning.show(path)
    assert shown["trials"]["complete"] == 2 and shown["tuned_mean"] is None
    assert shown["seeded_mean"] == pytest.approx((0.94 + 0.95) / 2)
    # The best plan's settings where it never varied them: the broad line's third.
    assert shown["best"][0]["depth"] == pytest.approx(1 / 3)


def test_a_plan_outside_the_space_is_not_seeded():
    assert tuning.plan_settings(best()) is not None
    assert tuning.plan_settings(best(wait=60)) is None  # A hold shorter than any tried.
    assert tuning.plan_settings(best(conscription="limited")) is None
    assert tuning.plan_settings(best(attack="deep")) is None
    assert tuning.plan_settings({**best(), "guard": None}) is None


def test_exploring_games_ask_the_tuner_and_report_their_scores(tmp_path, monkeypatch):
    from hoi4_arena import ai_games, scripted

    asked, told = [], []

    class Fake:
        def propose(self, rng):
            asked.append(True)
            return {**best(), "best": False, "variant": "tuned", "trial": 3}

        def report(self, plan, score):
            told.append((plan["trial"], score))

    settings = {"tuner": Fake(), "tune_skip": {"arena-marsh-v6"}}
    monkeypatch.setattr(scripted, "choose_plan", lambda rng: {**best(), "variant": "explore"})
    rng = random.Random(0)
    assert ai_games.pick_plan(rng, "arena-12x8-v4", settings)["variant"] == "tuned"
    # Not on a skipped arena, nor for an arena test, which plays the best plan.
    assert ai_games.pick_plan(rng, "arena-marsh-v6", settings)["variant"] == "explore"
    assert ai_games.pick_plan(rng, "arena-12x8-v4", settings, request={"name": "x"})["best"]
    assert len(asked) == 1

    def broken(rng):
        raise RuntimeError("no torch")

    settings["tuner"].propose = broken
    assert ai_games.pick_plan(rng, "arena-12x8-v4", settings)["variant"] == "explore"
    # The score goes to the tuner, and into the results; a game that failed scores None.
    folder = game(tmp_path, "7", best())
    entry = {"plan": {**best(), "trial": 3}, "winner": "RED", "started_as": "RED", "reason": None}
    ai_games.report_score(settings, entry, folder)
    failed = {"plan": {**best(), "trial": 4}, "started_as": "RED", "error": "start failed"}
    ai_games.report_score(settings, failed, folder)
    assert told == [(3, pytest.approx(0.94)), (4, None)]
    assert entry["score"] == pytest.approx(0.94) and failed["score"] is None
