from unittest.mock import Mock

import numpy as np
import pytest

from hoi4_arena.actions import SLOTS
from hoi4_arena.arena_log import STATE_WEIGHT, ArenaLog, parse

# Copied from a live game.log on 2026-09-23, with the "ARENA " prefix the worker removes.
WEEK = "week  1:00, 4 January, 1936 BLU states 8 owned 8 divisions 8 surrender 0"


def test_every_kind_of_mod_line_parses_including_the_padded_hour():
    assert parse("start  12:00, 1 January, 1936") == {
        "kind": "start",
        "date": "12:00, 1 January, 1936",
    }
    week = parse(WEEK)
    assert (
        week["kind"] == "week" and week["tag"] == "BLU" and week["date"] == "1:00, 4 January, 1936"
    )
    assert (week["states"], week["owned"], week["divisions"], week["surrender"]) == (8, 8, 8, 0.0)
    assert parse(WEEK.replace("surrender 0", "surrender 0.35"))["surrender"] == 0.35
    control = parse("control RED from BLU West 3  1:00, 9 March, 1936")
    assert control["state"] == "West 3" and control["previous"] == "BLU"
    capitulated = parse("capitulated RED winner BLU 12:00, 2 June, 1936")
    assert (capitulated["loser"], capitulated["winner"]) == ("RED", "BLU")
    assert parse("peace RED BLU 12:00, 3 June, 1936")["other"] == "BLU"
    assert parse("declare RED") == {"kind": "declare", "tag": "RED"}
    assert parse("player BLU") == {"kind": "player", "tag": "BLU"}
    # A line from a newer mod is skipped, not an error.
    assert parse("supply BLU 3") is None


def _log(*batches):
    desktop = Mock()
    desktop.game_log.side_effect = [(list(batch), 100 * (i + 1)) for i, batch in enumerate(batches)]
    return desktop


def _week(tag, states, surrender):
    return (
        f"week 12:00, 1 May, 1936 {tag} states {states} owned 8 divisions 8 surrender {surrender}"
    )


def test_potential_weighs_surrender_progress_and_states_for_each_side():
    log = ArenaLog(
        _log([_week("BLU", 8, 0), _week("RED", 8, 0)], [_week("BLU", 7, 0.1), _week("RED", 9, 0.4)])
    )
    log.poll()
    assert log.potential("BLU") == 0.0
    log.poll()
    # Red is 0.4 of the way to surrender and Blue 0.1; Red holds one of Blue's 8 states.
    assert log.potential("BLU") == pytest.approx(0.4 - 0.1 + STATE_WEIGHT * (7 - 9) / 8)
    assert log.potential("RED") == pytest.approx(-log.potential("BLU"))
    assert log.desktop.game_log.call_args.args == (100,)


def test_a_surrender_names_the_winner_and_a_silent_log_is_a_stopped_clock():
    now = [0.0]
    log = ArenaLog(
        _log([_week("BLU", 8, 0)], ["capitulated RED winner BLU 12:00, 2 June, 1936"], []),
        silence=90,
        clock=lambda: now[0],
    )
    log.poll()
    assert log.outcome("BLU") is None
    log.poll()
    assert (log.outcome("BLU"), log.outcome("RED")) == ("win", "loss")
    assert log.surrendered == "12:00, 2 June, 1936"
    now[0] = 91
    with pytest.raises(RuntimeError, match="clock stopped"):
        log.poll()


def test_a_match_scored_from_the_log_pays_the_change_in_potential_and_ends_on_the_surrender(
    tmp_path,
):
    from test_core import _MATCH, _rules_with, _screen

    from hoi4_arena.desktop import Frame
    from hoi4_arena.environment import LOG_EVERY, ArenaEnv

    rules = _rules_with(tmp_path, _MATCH)
    running = _screen(rules, ["ready", "healthy", "speed"])
    desktop = Mock()
    desktop.capture.side_effect = [Frame(running, {}, i) for i in range(3 * LOG_EVERY + 2)]
    desktop.apply.return_value = {}
    desktop.game_log.side_effect = [
        ([_week("BLU", 8, 0), _week("RED", 8, 0)], 1),
        ([_week("BLU", 8, 0), _week("RED", 7, 0.25)], 2),
        (["capitulated RED winner BLU 12:00, 2 June, 1936"], 3),
    ]
    env = ArenaEnv(desktop, rules, [], downscale=False, reward="log")
    env.reset()
    action = np.zeros((SLOTS, 3), dtype=np.int64)
    rewards, done = [], False
    while not done:
        _, reward, done, timeout, info = env.step(action)
        assert info["valid"], info.get("error")
        assert not timeout
        rewards.append(reward)
    # Nothing between readings, the change at the first, and the surrender at the second.
    assert len(rewards) == 2 * LOG_EVERY
    assert rewards[LOG_EVERY - 1] == pytest.approx(0.25 + STATE_WEIGHT / 8)
    assert rewards[-1] == pytest.approx(1.0)
    assert sum(rewards) == pytest.approx(1.0 + 0.25 + STATE_WEIGHT / 8)
    assert info["outcome"] == "win" and info["reward_source"] == "log"
    env.close()


def test_an_unknown_reward_source_is_refused(tmp_path):
    from test_core import _MATCH, _rules_with

    from hoi4_arena.environment import ArenaEnv

    with pytest.raises(ValueError, match="reward"):
        ArenaEnv(Mock(), _rules_with(tmp_path, _MATCH), [], reward="pixels")


def test_the_declarer_and_the_players_are_kept():
    log = ArenaLog(_log(["declare RED", "start  12:00, 1 January, 1936", "player BLU", WEEK]))
    log.poll()
    assert (log.declarer, log.players) == ("RED", ["BLU"])
