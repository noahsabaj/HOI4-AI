from pathlib import Path

import pytest

from hoi4_agent.enums import GermanState, PreconditionKind, Tech, ToolName
from hoi4_agent.errors import ConfigError
from hoi4_agent.playbook.loader import (
    load_playbook,
    load_research_days,
    parse_goals,
    parse_research_days,
)
from hoi4_agent.playbook.select import all_done, expired_goal, next_pending_goal
from hoi4_agent.playbook.state import load_state, save_state
from hoi4_agent.schemas import GameDate, Goal, PlaybookState, Precondition, WorldState

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAYBOOK = REPO_ROOT / "config" / "playbooks" / "germany_1936.toml"


def test_load_real_playbook():
    goals = load_playbook(PLAYBOOK)
    assert len(goals) == 9
    assert goals[0].tool is ToolName.ASSIGN_RESEARCH
    build = next(g for g in goals if g.id == "build_ruhr")
    assert build.tool is ToolName.BUILD_IN_STATE
    assert build.state is GermanState.RUHR
    assert build.precondition.kind is PreconditionKind.FREE_CIV_SLOT
    refill = next(g for g in goals if g.id == "research_refill")
    assert refill.repeatable and refill.needs_judgment and refill.tech is None


def test_parse_goals_errors():
    with pytest.raises(ConfigError):
        parse_goals({"goal": [{"tool": "observe"}]})  # missing id
    with pytest.raises(ConfigError):
        parse_goals({"goal": [{"id": "x", "tool": "not_a_tool"}]})  # bad enum
    with pytest.raises(ConfigError):
        parse_goals({})  # no goals
    with pytest.raises(ConfigError):
        parse_goals({"goal": [{"id": "x", "tool": "build_in_state", "precondition": "date_before"}]})  # needs date


def test_parse_goals_rejects_unknown_keys():
    # A typo'd key must fail loudly, not silently load as a default.
    with pytest.raises(ConfigError, match="needs_judgement"):
        parse_goals({"goal": [{"id": "x", "tool": "observe", "needs_judgement": True}]})


def test_research_days_table():
    days = load_research_days(PLAYBOOK)
    assert days[Tech.CONSTRUCTION_1] == 170
    assert all(v > 0 for v in days.values())
    assert parse_research_days({}) == {}  # optional table
    with pytest.raises(ConfigError):
        parse_research_days({"research_days": {"not_a_tech": 100}})
    with pytest.raises(ConfigError):
        parse_research_days({"research_days": {"industry_1": 0}})


def test_parse_goals_rejects_invalid_precondition_date():
    with pytest.raises(ConfigError, match="precondition_date"):
        parse_goals({"goal": [{"id": "x", "tool": "observe",
                               "precondition": "date_after", "precondition_date": "1936.13.1"}]})


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


def test_next_pending_goal_attempts_uncertain_slot_precondition():
    goals = [Goal(id="b", tool=ToolName.BUILD_IN_STATE, state=GermanState.RUHR,
                  precondition=Precondition(PreconditionKind.FREE_CIV_SLOT))]
    # Slot facts are unreadable at home view -> attempt the goal; the tool opens
    # the panel and decides (NotReady if the slot isn't actually free).
    assert next_pending_goal(goals, PlaybookState(), WorldState()).id == "b"
    # A definitely-absent slot still waits.
    assert next_pending_goal(goals, PlaybookState(), WorldState(free_civ_slots=0)) is None


def _repeat_and_build():
    return [
        Goal(id="refill", tool=ToolName.ASSIGN_RESEARCH, repeatable=True, needs_judgment=True,
             precondition=Precondition(PreconditionKind.IDLE_RESEARCH_SLOT)),
        Goal(id="b", tool=ToolName.BUILD_IN_STATE, state=GermanState.RUHR),
    ]


def test_repeatable_goal_never_blocks_successors():
    goals = _repeat_and_build()
    # Definitely no idle slot: the repeatable is stepped PAST, not returned None.
    world = WorldState(idle_research_slots=0)
    assert next_pending_goal(goals, PlaybookState(), world).id == "b"
    # Slot available: the repeatable fires first (priority order).
    world = WorldState(idle_research_slots=1)
    assert next_pending_goal(goals, PlaybookState(), world).id == "refill"
    # skip= steps past a repeatable that already reported NotReady this cycle.
    world = WorldState()  # uncertain -> would be attempted
    assert next_pending_goal(goals, PlaybookState(), world).id == "refill"
    assert next_pending_goal(goals, PlaybookState(), world, skip={"refill"}).id == "b"
    # A repeatable is re-offered even after appearing in completed ids.
    state = PlaybookState(completed_goal_ids=("refill", "b"))
    assert next_pending_goal(goals, state, WorldState(idle_research_slots=1)).id == "refill"
    assert all_done(goals, state)


def test_expired_goal_detects_missed_date_window():
    stale = Goal(id="old", tool=ToolName.BUILD_IN_STATE, state=GermanState.RUHR,
                 precondition=Precondition(PreconditionKind.DATE_BEFORE, GameDate(1936, 1, 1)))
    goals = [stale, Goal(id="b", tool=ToolName.BUILD_IN_STATE, state=GermanState.SAXONY)]
    # Window passed and the date is READABLE -> expired.
    assert expired_goal(goals, PlaybookState(), WorldState(date=GameDate(1936, 6, 1))) is stale
    # Window still open, or date unreadable -> nothing expires.
    assert expired_goal(goals, PlaybookState(), WorldState(date=GameDate(1935, 6, 1))) is None
    assert expired_goal(goals, PlaybookState(), WorldState()) is None
    # Already completed -> not expired; and only the queue HEAD can expire.
    done = PlaybookState(completed_goal_ids=("old",))
    assert expired_goal(goals, done, WorldState(date=GameDate(1936, 6, 1))) is None


def test_state_persistence(tmp_path):
    p = tmp_path / "state.json"
    assert load_state(p) == PlaybookState()  # missing file -> empty
    s = PlaybookState(completed_goal_ids=("a",), cycle_count=3)
    save_state(p, s)
    assert load_state(p) == s
