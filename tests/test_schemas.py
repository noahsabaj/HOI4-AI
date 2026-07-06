import pytest

from hoi4_agent.enums import BuildingType, GermanState, Tech, ToolName
from hoi4_agent.errors import IntentValidationError
from hoi4_agent.schemas import (
    GameDate,
    Intent,
    PlaybookState,
    TraceRecord,
    validate_intent,
)


def test_validate_intent_ok():
    validate_intent(Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR))
    validate_intent(Intent(ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                           state=GermanState.RUHR))
    validate_intent(Intent(ToolName.ASSIGN_RESEARCH, tech=Tech.INDUSTRY_1))
    validate_intent(Intent(ToolName.ENSURE_PAUSED, paused=True))
    validate_intent(Intent(ToolName.SET_SPEED, speed=4))
    validate_intent(Intent(ToolName.OBSERVE))


@pytest.mark.parametrize(
    "intent",
    [
        Intent(ToolName.BUILD_IN_STATE),  # missing state
        Intent(ToolName.ASSIGN_RESEARCH),  # missing tech
        Intent(ToolName.SET_SPEED, speed=9),  # out of range
        Intent(ToolName.ENSURE_PAUSED, paused=None),  # missing bool
    ],
)
def test_validate_intent_rejects(intent):
    with pytest.raises(IntentValidationError):
        validate_intent(intent)


def test_gamedate_order_and_str():
    assert GameDate(1936, 1, 1) < GameDate(1936, 2, 1) < GameDate(1937, 1, 1)
    assert GameDate.from_str("1936.1.1") == GameDate(1936, 1, 1)
    assert GameDate.from_str("1936-01-01") == GameDate(1936, 1, 1)
    assert GameDate(1936, 1, 1).to_str() == "1936.01.01"


def test_gamedate_from_ui_text_ymd_and_hoi4_topbar():
    # numeric Y.M.D (the original design assumption)
    assert GameDate.from_ui_text("1936.1.1") == GameDate(1936, 1, 1)
    assert GameDate.from_ui_text("1936. 1. 14") == GameDate(1936, 1, 14)
    # HOI4's actual top-bar rendering: clock, day, month NAME, year
    assert GameDate.from_ui_text("12:00, 1 Jan, 1936") == GameDate(1936, 1, 1)
    assert GameDate.from_ui_text("04:00, 28 Dec, 1943") == GameDate(1943, 12, 28)
    assert GameDate.from_ui_text("12:00,1Jan,1936") == GameDate(1936, 1, 1)  # glyph text: no spaces
    assert GameDate.from_ui_text("11 September 1939") == GameDate(1939, 9, 11)  # full month word
    # never guess
    assert GameDate.from_ui_text("1936.13.1") is None  # month out of range
    assert GameDate.from_ui_text("1 Foo 1936") is None  # unknown month word
    assert GameDate.from_ui_text("1936.1") is None
    assert GameDate.from_ui_text("gibberish") is None


def test_gamedate_plus_days_rolls_over():
    assert GameDate(1936, 1, 25).plus_days(10) == GameDate(1936, 2, 5)
    assert GameDate(1936, 12, 25).plus_days(10) == GameDate(1937, 1, 5)


def test_playbook_state_roundtrip():
    s = PlaybookState(completed_goal_ids=("a", "b"), last_seen_date=GameDate(1936, 6, 1), cycle_count=4)
    assert PlaybookState.from_dict(s.to_dict()) == s


def test_playbook_state_transitions():
    s = PlaybookState()
    s2 = s.with_completed("g1").with_completed("g1").advance_cycle()
    assert s2.completed_goal_ids == ("g1",)
    assert s2.cycle_count == 1


def test_pending_etas_roundtrip_and_ops():
    s = PlaybookState().with_eta("industry_1", GameDate(1936, 4, 1))
    s = s.with_eta("industry_1", GameDate(1936, 5, 1))  # replaces, not appends
    s = s.with_eta("construction_1", GameDate(1936, 3, 1))
    assert dict(s.pending_etas) == {"industry_1": "1936.05.01", "construction_1": "1936.03.01"}
    assert PlaybookState.from_dict(s.to_dict()) == s
    spent = s.drop_etas_through(GameDate(1936, 3, 15))
    assert dict(spent.pending_etas) == {"industry_1": "1936.05.01"}
    assert s.drop_etas_through(None) is s  # unreadable date: nothing changes


def test_trace_record_roundtrip():
    r = TraceRecord(cycle=1, ts=1.0, verdict="ok", plan_step="x", actions=({"tool": "observe"},))
    assert TraceRecord.from_dict(r.to_dict()) == r
