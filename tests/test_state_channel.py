import re

import pytest

from hoi4_arena import arena_log
from hoi4_arena.state_channel import (
    ARMY,
    CONSCRIPTION,
    COUNTRY,
    Channel,
    daily_effect,
    mod_states,
    parse,
    startup_effect,
    upgrade,
)

STATES = {1: [11, 12, 13], 2: [21, 22], 3: [31, 32, 33], 4: [41]}

# An arena's on_actions as mapgen wrote them before the state channel (v4 and the v6
# presets), cut down to what upgrade reads.
OLD = (
    "on_actions = {\n"
    '\ton_startup = { effect = { log = "ARENA start [GetDateText]" every_country = { limit ='
    ' { is_ai = no } log = "ARENA player [THIS.GetTag]" } } }\n'
    '\ton_weekly = { effect = { log = "ARENA week [GetDateText]" } }\n'
    "\ton_daily_BLU = { effect = { if = { limit = { NOT = { has_global_flag = arena_declared }"
    ' } } log = "ARENA day [GetDateText] [ROOT.GetTag]" } }\n'
    '\ton_daily_RED = { effect = { log = "ARENA day [GetDateText] [ROOT.GetTag]" } }\n'
    '\ton_capitulation = { effect = { log = "ARENA capitulated" } }\n'
    "}"
)


def _arena(root):
    for state, provinces in STATES.items():
        path = root / "history" / "states" / f"{state}-arena.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'state = {{ id = {state} name = "ARENA_STATE_{state}" history = {{ owner = BLU }}'
            f" provinces = {{ {' '.join(map(str, provinces))} }} }}"
        )
    on_actions = root / "common" / "on_actions" / "arena.txt"
    on_actions.parent.mkdir(parents=True)
    on_actions.write_text(OLD)
    return on_actions


def test_the_daily_effect_checks_every_province_and_logs_one_line_a_side():
    effect = daily_effect(STATES)
    assert effect.count("{") == effect.count("}")
    for state, provinces in STATES.items():
        assert f"{state} = {{ is_controlled_by = ROOT }}" in effect
        assert f"add_to_temp_variable = {{ arena_held = {2 ** (state - 1)} }}" in effect
        for bit, province in enumerate(provinces):
            assert (
                f"controls_province = {province} }} add_to_temp_variable ="
                f" {{ arena_m{state} = {2**bit} }}"
            ) in effect
        assert f"arena_a{state} = num_units_in_state@{state}" in effect
    for idea in CONSCRIPTION:
        assert f"has_idea = {idea}" in effect
    # One log for the side, one per leader in command, and every log is the mod's.
    assert effect.count('log = "') == effect.count('log = "ARENA ') == 2
    assert "every_army_leader = { limit = { is_assigned = yes }" in effect
    # It reads and logs: no effect that changes the game.
    words = set(effect.replace("{", " ").replace("}", " ").split())
    changes = {w for w in words if w.startswith(("add_", "set_", "remove_", "create_"))}
    assert changes == {"set_temp_variable", "add_to_temp_variable"}


def test_startup_logs_each_states_provinces_in_mask_order():
    effect = startup_effect(STATES)
    assert 'log = "ARENA provinces 2 21 22"' in effect
    assert effect.index("provinces 1 ") < effect.index("provinces 4 ")


def test_upgrade_adds_the_channel_once_and_leaves_the_rest_as_it_was(tmp_path):
    on_actions = _arena(tmp_path)
    assert mod_states(tmp_path) == STATES
    assert upgrade(tmp_path)
    text = on_actions.read_text()
    assert text.count("{") == text.count("}")
    daily = daily_effect(STATES)
    for tag in ("BLU", "RED"):
        line = next(ln for ln in text.splitlines() if f"on_daily_{tag}" in ln)
        assert line.endswith(daily + " } }")
        # The old report comes first, unchanged.
        assert 'log = "ARENA day [GetDateText] [ROOT.GetTag]"' in line
    startup = next(ln for ln in text.splitlines() if "on_startup" in ln)
    assert startup.endswith(startup_effect(STATES) + " } }")
    assert 'log = "ARENA player [THIS.GetTag]" }' in startup
    # Everything else is untouched.
    untouched = [ln for ln in OLD.splitlines() if "on_daily" not in ln and "on_startup" not in ln]
    assert [ln for ln in text.splitlines() if ln in untouched] == untouched
    assert not upgrade(tmp_path)
    assert on_actions.read_text() == text


def test_upgrade_refuses_an_on_actions_it_does_not_recognise(tmp_path):
    on_actions = _arena(tmp_path)
    on_actions.write_text(OLD.replace("} } }\n\ton_weekly", "}\n}\n}\n\ton_weekly"))
    with pytest.raises(ValueError, match="unexpected"):
        upgrade(tmp_path)


# Built from the formats HOI4 logged in the live probe (see STATE and ARMY below), with the
# "ARENA " prefix the worker removes.
STATE = (
    "state  24:00, 4 January, 1936 BLU day 3 held 3 mask 7 3 0 0 law 1 economy -1 trade -1"
    " pp 8.5 command 0 stability 0.5 war 0.5 manpower 9.5 max 12.3 queue 0 stock 49000"
    " orders 1 battalions 72"
)
ARMY_LINE = (
    "army  24:00, 4 January, 1936 BLU units 8 plans 2 orders 1 planning 0.3 ready 1 combat 2"
    " attack 1 defend 1 progress 0.25 entrench 0.1 rifles 4.8 needed 4800 group 0"
    " at 1=0 2=8 3=0 4=0 name BLU_general_1"
)


def test_state_and_army_lines_parse_into_numbers():
    state = parse(STATE)
    assert (state["kind"], state["tag"], state["date"]) == (
        "state",
        "BLU",
        "24:00, 4 January, 1936",
    )
    assert (state["day"], state["held"], state["mask"]) == (3, 3, [7, 3, 0, 0])
    assert (state["law"], state["economy"], state["trade"]) == (1, -1, -1)
    assert {name for name, _ in COUNTRY} <= state.keys()
    assert state["stock"] == 49000 and state["pp"] == 8.5
    army = parse(ARMY_LINE)
    assert {name for name, _ in ARMY} <= army.keys()
    assert army["at"] == {1: 0, 2: 8, 3: 0, 4: 0} and army["name"] == "BLU_general_1"
    assert army["units"] == 8 and army["plans"] == 2 and army["progress"] == 0.25
    assert parse("provinces 2 21 22") == {"kind": "provinces", "state": 2, "provinces": [21, 22]}
    # The mod's older lines are arena_log's, and each reader skips the other's.
    assert parse("week  1:00, 4 January, 1936 BLU states 8 owned 8 divisions 8 surrender 0") is None
    assert arena_log.parse(STATE) is None and arena_log.parse(ARMY_LINE) is None


def test_the_channel_keeps_each_sides_latest_state_and_armies():
    channel = Channel()
    for state, provinces in STATES.items():
        channel.feed(f"ARENA provinces {state} {' '.join(map(str, provinces))}")
    assert channel.snapshot("BLU") is None
    channel.feed("ARENA " + STATE)
    channel.feed("ARENA " + ARMY_LINE)
    assert channel.feed("ARENA start 12:00, 1 January, 1936") is None
    snap = channel.snapshot("BLU")
    assert snap["held"] == [True, True, False, False]
    assert snap["provinces"] == 5 and snap["armies"] == 1 and snap["plans"] == 2
    assert snap["commanded"] == 8 and snap["fighting"] == 2
    assert channel.controllers("BLU") == {
        11: True, 12: True, 13: True, 21: True, 22: True, 31: False, 32: False, 33: False,
        41: False,
    }  # fmt: skip
    # A marshal's army group counts its armies' divisions again, so only its plans count.
    channel.feed("ARENA " + ARMY_LINE.replace("group 0", "group 1").replace("plans 2", "plans 1"))
    snap = channel.snapshot("BLU")
    assert (snap["armies"], snap["groups"], snap["commanded"], snap["plans"]) == (1, 1, 8, 3)
    # The next day's report replaces the armies: the general left command.
    channel.feed("ARENA " + STATE.replace("day 3", "day 4"))
    assert channel.snapshot("BLU")["armies"] == 0 and channel.snapshot("BLU")["day"] == 4


def test_held_is_logged_sixteen_states_a_number_past_sixteen_states():
    """A game variable holds about 21 bits: 32 states' held bits in one would overflow."""
    states = {s: [100 + s] for s in range(1, 33)}
    effect = daily_effect(states)
    assert "add_to_temp_variable = { arena_held = 32768 }" in effect
    assert "add_to_temp_variable = { arena_held2 = 1 }" in effect
    assert "add_to_temp_variable = { arena_held2 = 32768 }" in effect
    assert "held [?arena_held] [?arena_held2] mask" in effect
    assert max(int(v) for v in re.findall(r"arena_held\d? = (\d+)", effect)) < 2**21
    line = STATE.replace("held 3", "held 3 32768")
    assert parse(line)["held"] == 3 + (32768 << 16)
    # Sixteen states log one number, as before.
    assert "[?arena_held2]" not in daily_effect(STATES)
