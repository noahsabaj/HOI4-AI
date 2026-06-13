from pathlib import Path

import pytest

from hoi4_agent.enums import GermanState, PreconditionKind, ToolName
from hoi4_agent.errors import ConfigError
from hoi4_agent.playbook.loader import load_playbook, parse_goals
from hoi4_agent.playbook.select import all_done, next_pending_goal
from hoi4_agent.playbook.state import load_state, save_state
from hoi4_agent.schemas import Goal, PlaybookState, Precondition, WorldState

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAYBOOK = REPO_ROOT / "config" / "playbooks" / "germany_1936.toml"


def test_load_real_playbook():
    goals = load_playbook(PLAYBOOK)
    assert len(goals) == 8
    assert goals[0].tool is ToolName.ASSIGN_RESEARCH
    build = next(g for g in goals if g.id == "build_ruhr")
    assert build.tool is ToolName.BUILD_IN_STATE
    assert build.state is GermanState.RUHR
    assert build.precondition.kind is PreconditionKind.FREE_CIV_SLOT


def test_parse_goals_errors():
    with pytest.raises(ConfigError):
        parse_goals({"goal": [{"tool": "observe"}]})  # missing id
    with pytest.raises(ConfigError):
        parse_goals({"goal": [{"id": "x", "tool": "not_a_tool"}]})  # bad enum
    with pytest.raises(ConfigError):
        parse_goals({})  # no goals
    with pytest.raises(ConfigError):
        parse_goals({"goal": [{"id": "x", "tool": "build_in_state", "precondition": "date_before"}]})  # needs date


def test_next_pending_goal_strict_order():
    goals = [
        Goal(id="a", tool=ToolName.OBSERVE),
        Goal(id="b", tool=ToolName.BUILD_IN_STATE, state=GermanState.RUHR,
             precondition=Precondition(PreconditionKind.FREE_CIV_SLOT)),
        Goal(id="c", tool=ToolName.CLOSE_PANELS),
    ]
    state = PlaybookState()
    # 'a' is first, ALWAYS precond
    assert next_pending_goal(goals, state, WorldState()).id == "a"
    state = state.with_completed("a")
    # 'b' gates: no free slot -> None (do NOT skip to 'c')
    assert next_pending_goal(goals, state, WorldState(free_civ_slots=0)) is None
    # free slot -> 'b'
    assert next_pending_goal(goals, state, WorldState(free_civ_slots=2)).id == "b"
    state = state.with_completed("b")
    assert next_pending_goal(goals, state, WorldState()).id == "c"
    state = state.with_completed("c")
    assert next_pending_goal(goals, state, WorldState()) is None
    assert all_done(goals, state)


def test_state_persistence(tmp_path):
    p = tmp_path / "state.json"
    assert load_state(p) == PlaybookState()  # missing file -> empty
    s = PlaybookState(completed_goal_ids=("a",), cycle_count=3)
    save_state(p, s)
    assert load_state(p) == s
