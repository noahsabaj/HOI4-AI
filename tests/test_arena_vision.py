"""Offline tests for the arena vision runtime.

Only the two files under tests/data/arena are real HOI4 pixels. Every other frame here is
SYNTHETIC (``vision.synthetic``) and proves that the reader inverts the renderer, not that it
reads the live game.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from hoi4_agent import clausewitz
from hoi4_agent.arena.actions import choice_index
from hoi4_agent.arena.contracts import ArenaError, ArenaSpec, BuildFingerprint, Country, Order, Verb
from hoi4_agent.arena.layout import ArenaLayout, LayoutProvince
from hoi4_agent.arena.vision import add_commands
from hoi4_agent.arena.vision.audit import audit_observation, counter_truth_report, extract_divisions
from hoi4_agent.arena.vision.calibration import (ArenaVisionCalibration, PanelCalibration, anchor_provinces,
                                                 assemble, build_steps, dump_vision_toml, fit_affine,
                                                 load_vision_calibration)
from hoi4_agent.arena.vision.counters import (BUILTIN_DIGITS, MEASURED_DIGITS, CounterDigits, CounterStyle,
                                              find_counters)
from hoi4_agent.arena.vision.executor import OrderExecutor, line_change
from hoi4_agent.arena.vision.observe import (ENEMY_BASE, OWN_STACK_BASE, ArenaObserver, EnemyTracker, LogTail,
                                             OwnDivisionTracker, absolute_hour, parse_outcome)
from hoi4_agent.arena.vision.panel import read_panel, row_point, row_rect
from hoi4_agent.arena.vision.record import InputEvent, RecordedFrame, invert_events
from hoi4_agent.arena.vision.session import PROVENANCE_SOURCE, VisionSession
from hoi4_agent.arena.vision.synthetic import SyntheticStack, render_arena_frame
from hoi4_agent.geometry import WindowGeometry
from hoi4_agent.io.backends import FakeCapture, InputRecorder, OrderInputBackend, RecordingInput, StubLocator
from hoi4_agent.perception.templates import TemplateStore

DATA = Path(__file__).parent / "data" / "arena"
TEMPLATES = Path(__file__).parent.parent / "templates"
SIZE = (1280, 720)
MAP_COLORS = {"BLU": (60, 90, 200), "RED": (120, 60, 80)}  # like the real tint: far from any frame colour
ROIS = {"date": (0.46, 0.012, 0.52, 0.034), "speed": (0.415, 0.010, 0.46, 0.032),
        "pause": (0.40, 0.010, 0.415, 0.032)}


def grid_layout(cols: int = 4, rows: int = 3) -> ArenaLayout:
    provinces = []
    for r in range(rows):
        for c in range(cols):
            pid = r * cols + c + 1
            near = [pid - 1] * (c > 0) + [pid + 1] * (c < cols - 1) + [pid - cols] * (r > 0) + \
                [pid + cols] * (r < rows - 1)
            provinces.append(LayoutProvince(pid, (c + 0.5) / cols, (r + 0.5) / rows, "plains", tuple(near),
                                            sector="centre", initial_controller="BLU" if c < cols / 2 else "RED",
                                            capital_of="BLU" if pid == 1 else "RED" if pid == cols * rows else ""))
    return ArenaLayout("vision-test-grid", tuple(provinces))


LAYOUT = grid_layout()


def calibration(size: tuple[int, int] = SIZE, scale: float = 1.0, **changes) -> ArenaVisionCalibration:
    base = ArenaVisionCalibration(width=size[0], height=size[1], map_colors=dict(MAP_COLORS), rois=dict(ROIS))
    return replace(base, counter_style=base.counter_style.scaled(scale), **changes)


def observer(cal: ArenaVisionCalibration, **kwargs) -> ArenaObserver:
    kwargs.setdefault("templates", TemplateStore())
    result = ArenaObserver(LAYOUT, cal, Country.BLUE, **kwargs)
    result.begin_episode("episode-1")
    return result


STACKS = [SyntheticStack(1, "own", 5, 1.0, 0.5), SyntheticStack(6, "own", 12, 0.35, 0.8),
          SyntheticStack(7, "enemy", 7, 0.6, 1.0), SyntheticStack(12, "enemy", 3, 0.0, 0.1)]


# --- real pixels --------------------------------------------------------------
def test_real_sample_map_has_exactly_two_counters_reading_5_and_7():
    readings = find_counters(Image.open(DATA / "sample_map_600x400.png"))
    assert [r.count for r in readings] == [5, 7]
    assert [r.bbox for r in readings] == [(128, 220, 181, 242), (223, 294, 276, 316)]
    for reading in readings:
        assert reading.relation == "own" and reading.count_measured and reading.has_army_plate
        assert reading.scale == 1.0 and reading.count_score > 0.95 and reading.confidence > 0.9
        assert reading.organization is not None and 0.9 <= reading.organization <= 1.0
        assert reading.strength is not None and 0.9 <= reading.strength <= 1.0
        assert reading.in_combat is None  # nothing on a counter identifies combat: unknown, not False


def test_real_counter_crop_reads_at_a_different_scale():
    (reading,) = find_counters(Image.open(DATA / "sample_counter_crop.png"))
    assert reading.count == 5 and 1.4 < reading.scale < 1.6
    assert reading.organization == 1.0 and reading.strength == 1.0


def test_guessed_digit_templates_lower_confidence_and_calibrated_ones_replace_them(tmp_path):
    cal = calibration()
    frame = render_arena_frame(LAYOUT, [SyntheticStack(6, "own", 6, 1.0, 1.0)], cal)
    (guessed,) = find_counters(frame)
    assert guessed.count == 6 and not guessed.count_measured  # 0, 6, 8, 9 are still hand-drawn
    glyph = (np.array([[ch == "#" for ch in row] for row in BUILTIN_DIGITS["6"]]) * 225 + 30).astype(np.uint8)
    Image.fromarray(glyph).save(tmp_path / "counter_glyph_6.png")
    (measured,) = find_counters(frame, digits=CounterDigits.load(tmp_path))
    assert measured.count == 6 and measured.count_measured and measured.confidence > guessed.confidence
    assert MEASURED_DIGITS == frozenset("123457")


def test_blank_image_has_no_counters():
    assert find_counters(Image.new("RGB", (400, 300), (74, 80, 88))) == []
    assert find_counters(Image.new("RGB", (400, 300), CounterStyle().own_frame_rgb)) == []


# --- synthetic round trip -----------------------------------------------------
@pytest.mark.parametrize("size,scale", [((1280, 720), 1.0), ((2560, 1440), 2.0), ((1920, 1080), 1.5)])
def test_synthetic_round_trip(size, scale):
    cal = calibration(size, scale)
    templates = TemplateStore.load_dir(TEMPLATES)
    frame = render_arena_frame(LAYOUT, STACKS, cal, speed=3, paused=True, templates=templates,
                               controllers={p.id: Country(p.initial_controller) for p in LAYOUT.provinces})
    watcher = observer(cal, templates=templates)
    observation = watcher.observe(frame)
    quantum = 1.0 / (17 * scale) + 1e-6
    assert len(observation.units) == len(STACKS)
    for stack, unit in zip(STACKS, sorted(observation.units, key=lambda u: u.province_id)):
        assert unit.province_id == stack.province_id and unit.count == stack.count
        assert unit.country is (Country.BLUE if stack.relation == "own" else Country.RED)
        assert abs(unit.organization - stack.organization) <= quantum
        assert abs(unit.strength - stack.strength) <= quantum
        assert unit.supply is None and unit.in_combat is None and 0 < unit.confidence <= 1
    assert [p.controller.value for p in observation.provinces] == [p.initial_controller for p in LAYOUT.provinces]
    assert observation.game_speed == 3 and observation.paused is True
    assert watcher.detail.confidence["game_speed"] > 0.9 and watcher.detail.confidence["controllers"] == 1.0
    assert watcher.detail.hour_estimated and watcher.detail.confidence["game_hour"] == 0.2


def test_counter_on_land_of_its_own_hue_is_still_found():
    # Same hue as the frames, so the hue mask alone would fuse border and land. (Land within
    # edge_tolerance of the border's exact colour would still defeat the anchor: a known limit.)
    cal = calibration(map_colors={"BLU": (40, 95, 50), "RED": (110, 37, 37)})
    frame = render_arena_frame(LAYOUT, [SyntheticStack(2, "own", 5, 0.5, 0.5), SyntheticStack(11, "enemy", 7, 1.0, 1.0)],
                               cal, controllers={2: Country.BLUE, 11: Country.RED})
    assert [(r.relation, r.count) for r in find_counters(frame)] == [("own", 5), ("enemy", 7)]


def test_unknown_controller_and_unreadable_top_bar_stay_unknown():
    cal = replace(calibration(), map_colors={}, rois={})
    watcher = observer(cal)
    observation = watcher.observe(render_arena_frame(LAYOUT, STACKS[:1], cal))
    assert all(p.controller is None for p in observation.provinces)
    assert watcher.detail.confidence["game_speed"] == 0.0 and watcher.detail.confidence["paused"] == 0.0
    assert observation.game_speed == 1 and observation.paused is False  # last known defaults, confidence 0


def test_province_assignment_uses_the_affine_and_drops_far_counters():
    shifted = replace(calibration(), affine=(0.5, 0.0, 0.25, 0.0, 0.5, 0.3))
    frame = render_arena_frame(LAYOUT, [SyntheticStack(7, "own", 5, 1.0, 1.0)], shifted)
    observation = observer(shifted).observe(frame)
    assert [u.province_id for u in observation.units] == [7]
    tight = replace(calibration(), max_assign_distance=0.01)
    watcher = observer(tight)  # identity mapping: the counter is now ~20% of the screen from province 7
    assert watcher.observe(frame).units == () and watcher.detail.unassigned == 1


def test_sequence_and_timestamps_are_monotonic_and_ids_stable():
    cal = calibration()
    ticks = iter(range(1_000, 10_000_000, 1_000))
    watcher = observer(cal, clock_ns=lambda: next(ticks))
    frame = render_arena_frame(LAYOUT, STACKS, cal)
    first, second = watcher.observe(frame), watcher.observe(frame)
    assert (first.sequence, second.sequence) == (0, 1)
    assert second.captured_monotonic_ns > first.captured_monotonic_ns and second.game_hour >= first.game_hour
    assert [u.id for u in first.units] == [u.id for u in second.units]
    assert {u.id for u in first.units if u.country is Country.BLUE} == {OWN_STACK_BASE + 10, OWN_STACK_BASE + 60}


def test_enemy_ids_follow_province_continuity():
    tracker = EnemyTracker(LAYOUT)
    first = tracker.assign([7, 12])
    assert first == [ENEMY_BASE, ENEMY_BASE + 1]
    assert tracker.assign([12, 7]) == [first[1], first[0]]  # order of detection is irrelevant
    assert tracker.assign([6, 12]) == first  # 7 -> 6 is adjacent: same contact
    moved = tracker.assign([1, 12])  # 6 -> 1 is not adjacent: a new contact
    assert moved[1] == first[1] and moved[0] not in first
    assert tracker.assign([]) == [] and tracker.assign([12])[0] not in first  # lost contacts do not come back


def test_enemy_ids_through_the_observer():
    cal = calibration()
    watcher = observer(cal)
    a = watcher.observe(render_arena_frame(LAYOUT, [SyntheticStack(7, "enemy", 4, 1.0, 1.0)], cal))
    b = watcher.observe(render_arena_frame(LAYOUT, [SyntheticStack(6, "enemy", 4, 0.8, 1.0)], cal))
    assert a.units[0].id == b.units[0].id and b.units[0].province_id == 6


def test_clock_outcome_and_log_tail(tmp_path):
    assert absolute_hour(1936, 1, 2, 12) - absolute_hour(1936, 1, 1, 12) == 24
    assert absolute_hour(1936, 3, 1, 0) - absolute_hour(1936, 2, 28, 0) == 24  # no leap day in HOI4
    assert parse_outcome(["[effect] ARENA_OUTCOME winner=RED day=90"]) == (True, Country.RED)
    assert parse_outcome(["ARENA_OUTCOME DRAW"]) == (True, None) and parse_outcome(["nothing"]) == (False, None)
    log = tmp_path / "game.log"
    log.write_text("old ARENA_OUTCOME BLU\n", encoding="utf-8")
    tail = LogTail(log)
    assert tail.poll() == []  # history before the tail was opened is ignored
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("ARENA_OUTCOME winner=BLU\npartial")
    assert tail.poll() == ["ARENA_OUTCOME winner=BLU"]
    cal = calibration()
    watcher = observer(cal, log=tail)
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(" line\nARENA_OUTCOME winner=RED\n")
    observation = watcher.observe(render_arena_frame(LAYOUT, STACKS, cal))
    assert observation.terminal and observation.winner is Country.RED


def test_own_division_tracker_and_panel_give_per_division_views():
    panel = PanelCalibration(enabled=True, first_row=(0.0, 0.30, 0.15, 0.33), row_pitch=0.04, max_rows=6)
    cal = replace(calibration(), panel=panel, points={"army_select": (500, 990), "deselect": (990, 990)})
    frame = render_arena_frame(LAYOUT, [SyntheticStack(1, "own", 2, 1.0, 1.0), SyntheticStack(6, "own", 1, 1.0, 1.0)], cal)
    pixels = np.array(frame)
    for index, (org, strength) in enumerate([(1.0, 0.5), (0.25, 1.0), (0.5, 0.75)]):  # SYNTHETIC panel rows
        x0, y0, x1, y1 = (int(v * s) for v, s in zip(row_rect(panel, index), SIZE * 2))
        pixels[y0:y1, x0:x1] = (40, 40, 44)
        for box, value, color in ((panel.org_bar, org, (105, 195, 115)), (panel.strength_bar, strength, (195, 145, 75))):
            bx0, bx1 = x0 + round(box[0] * (x1 - x0)), x0 + round(box[2] * (x1 - x0))
            by0, by1 = y0 + round(box[1] * (y1 - y0)), y0 + round(box[3] * (y1 - y0))
            pixels[by0:by1, bx0:bx0 + round(value * (bx1 - bx0))] = color
    frame = Image.fromarray(pixels)
    rows = read_panel(frame, panel)
    assert [(r.index, round(r.organization, 1), round(r.strength, 2)) for r in rows] == [(0, 1.0, 0.5), (1, 0.2, 1.0),
                                                                                         (2, 0.5, 0.75)]
    assert read_panel(frame, PanelCalibration()) == ()  # disabled until calibrated
    watcher = observer(cal, initial_divisions=(1, 1, 6))
    observation = watcher.observe(frame)
    assert watcher.detail.per_division and [(u.id, u.province_id, u.count) for u in observation.units] == [
        (1, 1, 1), (2, 1, 1), (3, 6, 1)]
    assert watcher.detail.handles[2].panel_row == 1 and watcher.detail.handles[2].stack_count == 2

    tracker = OwnDivisionTracker((1, 1, 6))
    tracker.note_order(2, 2)
    assert tracker.update({1: 2, 6: 1}) and tracker.location[2] == 1  # not arrived yet
    assert tracker.update({1: 1, 2: 1, 6: 1}) and tracker.location[2] == 2 and 2 not in tracker.target
    assert not tracker.update({1: 1, 5: 1, 6: 1})  # an unordered move (retreat): give up, do not guess
    assert not tracker.update({1: 1, 2: 1, 6: 1})  # and stay given up until reset


# --- executor -----------------------------------------------------------------
GEO = WindowGeometry(1, 0, 0, *SIZE)


def order(observation, verb, units=(), target=None, speed=None, order_id="o1") -> Order:
    return Order(order_id, observation.episode_id, observation.sequence, Country.BLUE, verb, units, target, speed)


def with_arrow(frame: Image.Image, cal: ArenaVisionCalibration, origin: int, target: int) -> Image.Image:
    out = frame.copy()
    a, b = LAYOUT.province(origin), LAYOUT.province(target)
    ImageDraw.Draw(out).line([cal.layout_to_pixel(a.x, a.y, *SIZE), cal.layout_to_pixel(b.x, b.y, *SIZE)],
                             fill=(90, 230, 110), width=5)
    return out


def executor_setup(frames_after_observe, **points):
    cal = replace(calibration(), points={"deselect": (990, 990), **points})
    base = render_arena_frame(LAYOUT, STACKS, cal)
    watcher = observer(cal)
    observation = watcher.observe(base)
    pad = RecordingInput()
    assert isinstance(pad, OrderInputBackend)
    frames = [base if f == "base" else with_arrow(base, cal, *f) for f in frames_after_observe]
    clock = {"t": 0.0}
    executor = OrderExecutor(LAYOUT, cal, FakeCapture(frames), pad, clock=lambda: clock["t"],
                             sleep=lambda s: clock.__setitem__("t", clock["t"] + s), on_accepted=watcher.note_order)
    return cal, watcher, observation, pad, executor


def test_move_selects_right_clicks_confirms_and_deselects():
    cal, watcher, observation, pad, executor = executor_setup(["base", (1, 2)])
    unit = OWN_STACK_BASE + 10
    receipt = executor.execute(order(observation, Verb.MOVE, (unit,), 2), GEO, observation, watcher.detail)
    assert receipt.accepted and "arrow" in receipt.reason and receipt.applied_game_hour == observation.game_hour
    counter, target = cal.layout_to_point(0.125, 1 / 6), cal.layout_to_point(0.375, 1 / 6)
    assert pad.calls == [("focus",), ("click", *counter), ("right_click", *target, ()), ("click", 990, 990)]
    assert executor.timings[0].confirmed is True and executor.timings[0].inputs == 3
    again = watcher.observe(render_arena_frame(LAYOUT, STACKS, cal))
    assert {u.id: u.order_target_province_id for u in again.units}[unit] == 2  # reported back by the observer


def test_support_attack_holds_ctrl_and_cancel_presses_halt():
    _, watcher, observation, pad, executor = executor_setup(["base", (6, 7)])
    unit = OWN_STACK_BASE + 60
    assert executor.execute(order(observation, Verb.SUPPORT_ATTACK, (unit,), 7), GEO, observation,
                            watcher.detail).accepted
    assert [c for c in pad.calls if c[0] == "right_click"][0][3] == ("ctrl",)
    pad.calls.clear()
    receipt = executor.execute(order(observation, Verb.CANCEL, (unit,), order_id="o2"), GEO, observation, watcher.detail)
    assert receipt.accepted and "not visually confirmed" in receipt.reason
    assert [c[0] for c in pad.calls] == ["focus", "click", "key", "click"] and pad.keys == ["h"]


def test_failed_confirmation_is_a_rejected_receipt_and_still_deselects():
    _, watcher, observation, pad, executor = executor_setup(["base"])  # the frame never changes
    receipt = executor.execute(order(observation, Verb.MOVE, (OWN_STACK_BASE + 10,), 2), GEO, observation,
                               watcher.detail)
    assert not receipt.accepted and receipt.applied_game_hour is None and "no order arrow" in receipt.reason
    assert pad.calls[-1] == ("click", 990, 990) and executor.timings[0].confirmed is False
    assert watcher.stack_targets == {}


def test_budget_bounds_blocking_and_unknown_units_are_rejected():
    cal, watcher, observation, pad, executor = executor_setup(["base"])
    executor.calibration = replace(cal, executor=replace(cal.executor, budget_ms=200, settle_ms=120))
    receipt = executor.execute(order(observation, Verb.MOVE, (OWN_STACK_BASE + 10,), 2), GEO, observation,
                               watcher.detail)
    assert not receipt.accepted and "budget exhausted" in receipt.reason and executor.timings[0].elapsed_ms <= 200
    enemy = next(u.id for u in observation.units if u.country is Country.RED)
    rejected = executor.execute(order(observation, Verb.CANCEL, (enemy,), order_id="o3"), GEO, observation,
                                watcher.detail)
    assert not rejected.accepted and "not an own unit" in rejected.reason


def test_speed_pause_and_noop_inputs():
    cal, watcher, observation, pad, executor = executor_setup(["base"])
    assert executor.execute(order(observation, Verb.NOOP), GEO, observation, watcher.detail).accepted
    assert pad.calls == []
    assert executor.execute(order(observation, Verb.PAUSE, order_id="p"), GEO, observation, watcher.detail).accepted
    assert pad.keys == ["space"]
    # speed unreadable and no speed point: refuse instead of stepping blind
    refused = executor.execute(order(observation, Verb.SET_SPEED, speed=4, order_id="s0"), GEO, observation,
                               watcher.detail)
    assert not refused.accepted and "unreadable" in refused.reason
    watcher.detail.confidence["game_speed"] = 0.99
    fast = replace(observation, game_speed=2)
    pad.calls.clear()
    assert executor.execute(order(fast, Verb.SET_SPEED, speed=5, order_id="s1"), GEO, fast, watcher.detail).accepted
    assert pad.keys == ["+", "+", "+"]
    pad.calls.clear()
    assert executor.execute(order(fast, Verb.SET_SPEED, speed=1, order_id="s2"), GEO, fast, watcher.detail).accepted
    assert pad.keys == ["-"]
    for speed in range(1, 6):  # all five speeds through calibrated speed-step points
        executor.calibration = replace(cal, points={**cal.points, f"speed_{speed}": (400 + speed, 20)})
        pad.calls.clear()
        assert executor.execute(order(fast, Verb.SET_SPEED, speed=speed, order_id=f"c{speed}"), GEO, fast,
                                watcher.detail).accepted
        assert pad.clicks == [(400 + speed, 20)]
    executor.calibration = replace(cal, executor=replace(cal.executor, speed_mode="direct", pause_mode="click"))
    pad.calls.clear()
    assert executor.execute(order(fast, Verb.SET_SPEED, speed=3, order_id="d"), GEO, fast, watcher.detail).accepted
    assert pad.keys == ["3"]
    assert not executor.execute(order(fast, Verb.PAUSE, order_id="pc"), GEO, fast, watcher.detail).accepted
    assert len(executor.timings) == 12  # every order timed; nothing caps how many there are


def test_division_in_a_shared_province_is_selected_through_the_army_panel():
    cal, watcher, observation, pad, executor = executor_setup(["base", (1, 2)], army_select=(500, 990))
    panel = PanelCalibration(enabled=True)
    executor.calibration = replace(executor.calibration, panel=panel)
    handle = watcher.detail.handles[OWN_STACK_BASE + 10]
    watcher.detail.handles[2] = replace(handle, unit_id=2, panel_row=1)
    assert executor.execute(order(observation, Verb.MOVE, (2,), 2), GEO, observation, watcher.detail).accepted
    assert pad.clicks[:2] == [(500, 990), row_point(panel, 1)]
    executor.calibration = replace(executor.calibration, panel=PanelCalibration())
    refused = executor.execute(order(observation, Verb.MOVE, (2,), 2, order_id="o9"), GEO, observation, watcher.detail)
    assert not refused.accepted and "army panel is not calibrated" in refused.reason


def test_line_change_ignores_the_counter_and_unrelated_changes():
    base = Image.new("RGB", (300, 100), (70, 70, 70))
    arrow = base.copy()
    ImageDraw.Draw(arrow).line([(20, 50), (280, 50)], fill=(255, 255, 0), width=5)
    elsewhere = base.copy()
    ImageDraw.Draw(elsewhere).rectangle((100, 5, 200, 20), fill=(255, 255, 0))
    assert line_change(base, arrow, (20, 50), (280, 50), 30, 30) == 1.0
    assert line_change(base, elsewhere, (20, 50), (280, 50), 30, 30) == 0.0
    assert line_change(base, base, (20, 50), (280, 50), 30, 30) == 0.0


def test_input_recorder_journals_right_clicks():
    inner = RecordingInput()
    journal = InputRecorder(inner)
    journal.right_click(GEO, GEO.full_crop(), 10, 20, ("ctrl",))
    assert inner.calls == [("right_click", 10, 20, ("ctrl",))]
    assert journal.drain() == [{"kind": "right_click", "nx": 10, "ny": 20, "modifiers": ["ctrl"]}]


# --- session ------------------------------------------------------------------
def make_session(tmp_path, frames, reset_line="ARENA_RESET seed=0\n"):
    log = tmp_path / "game.log"
    log.write_text("startup\n", encoding="utf-8")
    cal = replace(calibration(), log_path=str(log),
                  points={"decisions_open": (100, 20), "reset_decision": (150, 300), "decisions_close": (300, 100),
                          "deselect": (990, 990)})
    base = render_arena_frame(LAYOUT, STACKS, cal)
    clock = {"t": 0.0, "written": False}

    def sleep(seconds: float) -> None:
        clock["t"] += seconds
        if reset_line and not clock["written"] and clock["t"] > 0.3:  # the "game" logs the reset a bit later
            clock["written"] = True
            with open(log, "a", encoding="utf-8") as handle:
                handle.write(reset_line)

    pad = RecordingInput()
    rendered = [base if f == "base" else with_arrow(base, cal, *f) for f in frames]
    session = VisionSession(LAYOUT, cal, StubLocator(GEO), FakeCapture(rendered), pad, sleep=sleep,
                            clock=lambda: clock["t"],
                            observer_factory=lambda country, tail: ArenaObserver(LAYOUT, cal, country, log=tail,
                                                                                 templates=TemplateStore()))
    return session, pad, clock, log


SPEC = ArenaSpec("vision-test-grid", BuildFingerprint("a" * 64, "1.0", "b" * 64, "c" * 64))


def test_session_reset_observe_submit_close(tmp_path):
    session, pad, clock, log = make_session(tmp_path, ["base", "base", "base", (1, 2)])
    episode = session.reset(SPEC, (Country.BLUE,))
    assert pad.clicks == [(100, 20), (150, 300), (300, 100)] and pad.calls[0] == ("focus",)
    assert PROVENANCE_SOURCE == "hoi4_vision"
    observation = session.observe(Country.BLUE)
    assert observation.episode_id == episode and observation.sequence == 1 and len(observation.units) == 4
    move = Order("m1", episode, observation.sequence, Country.BLUE, Verb.MOVE, (OWN_STACK_BASE + 10,), 2)
    choice_index(observation, move)  # the order is inside the policy's action vocabulary
    receipt = session.submit(move)
    assert receipt.accepted and receipt.order_id == "m1"
    with pytest.raises(ArenaError, match="duplicate"):
        session.submit(move)
    with pytest.raises(ArenaError, match="latest"):
        session.submit(Order("m2", episode, 0, Country.BLUE, Verb.NOOP))
    with pytest.raises(ArenaError):
        session.observe(Country.RED)
    latest = session.observe(Country.BLUE)
    clock["t"] += 1.0  # older than max_observation_age_ms: refused without touching the game
    before = len(pad.calls)
    stale = session.submit(Order("m3", episode, latest.sequence, Country.BLUE, Verb.PAUSE))
    assert not stale.accepted and "old" in stale.reason and len(pad.calls) == before
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("ARENA_OUTCOME winner=BLU\n")
    final = session.observe(Country.BLUE)
    assert final.terminal and final.winner is Country.BLUE
    with pytest.raises(ArenaError, match="nonterminal"):
        session.submit(Order("m4", episode, final.sequence, Country.BLUE, Verb.NOOP))
    session.close()
    with pytest.raises(ArenaError):
        session.observe(Country.BLUE)


def test_session_reset_refuses_without_confirmation(tmp_path):
    session, _, _, _ = make_session(tmp_path, ["base"], reset_line="")
    with pytest.raises(ArenaError, match="ARENA_RESET"):
        session.reset(SPEC, (Country.BLUE,))
    with pytest.raises(ArenaError, match="exactly one"):
        session.reset(SPEC, (Country.BLUE, Country.RED))
    session.calibration = replace(session.calibration, points={})
    with pytest.raises(ArenaError, match="calibrated point"):
        session.reset(SPEC, (Country.BLUE,))


# --- audit --------------------------------------------------------------------
SAVE_SNIPPET = """HOI4txt
date="1936.1.5.12"
countries={
    BLU={
        units={
            division={ id={ id=1 type=41 } name="1st" location=1 organisation=30.0 max_organisation=60.0 strength=1.000 }
            division={ id={ id=2 type=41 } name="2nd" location=1 organisation=60.0 max_organisation=60.0 strength=0.500 }
            division={ name="3rd" location=6 organisation=45.2 strength=0.8 }
            navy={ location=999 }
        }
        capital=1
    }
    RED={
        units={
            division={ name="r1" location=7 organisation=0.6 strength=1.0 }
            division={ name="r2" location=7 organisation=0.6 strength=1.0 }
        }
    }
}
"""


def test_audit_against_a_hand_written_save_snippet():
    divisions, missing = extract_divisions(clausewitz.parse(SAVE_SNIPPET))
    assert missing == [] and len(divisions) == 5  # the navy block is outside any division container
    first, _, third = (d for d in divisions if d.country == "BLU")
    assert (first.province_id, first.organization, first.strength) == (1, 0.5, 1.0)
    assert third.organization is None and third.strength == 0.8  # absolute org without its maximum: unknown
    cal = calibration()
    stacks = [SyntheticStack(1, "own", 2, 0.75, 0.75), SyntheticStack(6, "own", 2, 1.0, 0.3),
              SyntheticStack(7, "enemy", 2, 0.6, 1.0), SyntheticStack(12, "enemy", 3, 1.0, 1.0)]
    report = audit_observation(observer(cal).observe(render_arena_frame(LAYOUT, stacks, cal)), divisions, missing)
    assert report["own_province_recall"] == 1.0 and report["own_province_precision"] == 1.0
    assert report["own_count_accuracy"] == 0.5  # province 6 holds one division, perception said two
    assert report["own_organization_accuracy"] == 1.0 and report["own_organization_mae"] < 0.05
    assert report["own_strength_accuracy"] == 0.5  # 0.3 perceived against 0.8 in the save
    assert report["enemy_reported_exists"] == 0.5 and report["enemy_count_accuracy"] == 1.0
    assert report["enemy_plausibly_visible"] == 0.5 and report["save_divisions"] == {"BLU": 3, "RED": 2}
    empty, gaps = extract_divisions(clausewitz.parse("date=1\ncountries={ BLU={ capital=1 } }"))
    assert empty == [] and gaps == ["divisions of BLU", "country block RED"]
    mapped = audit_observation(observer(cal).observe(render_arena_frame(LAYOUT, stacks[:1], cal)),
                               [replace(divisions[0], province_id=5001), replace(divisions[1], province_id=5001)],
                               province_map={5001: 1})
    assert mapped["own_count_accuracy"] == 1.0


# --- recorder -----------------------------------------------------------------
def test_recorder_inverse_mapping_recovers_orders_aligned_with_the_preceding_observation():
    cal = replace(calibration(), points={"speed_4": (440, 20)})
    ticks = iter((0, 0, 1_000, 5_000))  # construction, begin_episode, then the two observations
    watcher = observer(cal, clock_ns=lambda: next(ticks))
    frames = []
    for _ in range(2):
        observation = watcher.observe(render_arena_frame(LAYOUT, STACKS, cal))
        frames.append(RecordedFrame(observation, dict(watcher.detail.handles)))
    unit = OWN_STACK_BASE + 10
    here, east, south = (cal.layout_to_pixel(LAYOUT.province(p).x, LAYOUT.province(p).y, *SIZE) for p in (1, 2, 5))
    events = [
        InputEvent(500, "right", *east),  # before any observation: dropped
        InputEvent(2_000, "right", *east),  # nothing selected: dropped
        InputEvent(2_500, "left", here[0] + 6, here[1] - 3),  # select the counter in province 1
        InputEvent(3_000, "right", east[0] + 9, east[1] + 7),  # -> MOVE 1 -> 2 against observation 0
        InputEvent(6_000, "right", *south, ctrl=True),  # -> SUPPORT_ATTACK against observation 1
        InputEvent(6_100, "right", *here),  # right-click on its own province: not an order
        InputEvent(6_200, "right", 5.0, 700.0),  # far from every centre: dropped
        InputEvent(6_500, "key", key="h"),  # -> CANCEL
        InputEvent(6_600, "key", key="space"),  # -> PAUSE
        InputEvent(6_700, "key", key="+"),  # -> SET_SPEED 2 (observed speed 1)
        InputEvent(6_800, "left", 440 / 1000 * SIZE[0], 20 / 1000 * SIZE[1]),  # speed step click -> SET_SPEED 4
        InputEvent(7_000, "left", 640.0, 400.0),  # click on empty map: deselect
        InputEvent(7_100, "key", key="h"),  # nothing selected: dropped
    ]
    orders = invert_events(events, frames, LAYOUT, cal, SIZE)
    assert [(o.verb, o.unit_ids, o.target_province_id, o.speed, o.observation_sequence) for o in orders] == [
        (Verb.MOVE, (unit,), 2, None, 0), (Verb.SUPPORT_ATTACK, (unit,), 5, None, 1),
        (Verb.CANCEL, (unit,), None, None, 1), (Verb.PAUSE, (), None, None, 1),
        (Verb.SET_SPEED, (), None, 2, 1), (Verb.SET_SPEED, (), None, 4, 1)]
    for recovered in orders:  # every recovered order is a legal demonstration for its observation
        choice_index(frames[recovered.observation_sequence].observation, recovered)


# --- calibration and CLI ---------------------------------------------------------
def test_affine_fit_toml_round_trip_and_wizard_assembly(tmp_path):
    truth = (0.8, 0.02, 0.1, -0.01, 0.7, 0.15)
    anchors = anchor_provinces(LAYOUT, 5)
    assert anchors[:2] == (1, 12) and len(set(anchors)) == 5
    hovered = {pid: (truth[0] * LAYOUT.province(pid).x + truth[1] * LAYOUT.province(pid).y + truth[2],
                     truth[3] * LAYOUT.province(pid).x + truth[4] * LAYOUT.province(pid).y + truth[5])
               for pid in anchors}
    fitted, residual = calibration().with_affine_from(LAYOUT, hovered)
    assert residual < 1e-9 and np.allclose(fitted.affine, truth)
    with pytest.raises(ArenaError, match="collinear"):
        fit_affine([(0, 0), (0.5, 0.5), (1, 1)], [(0, 0), (1, 1), (2, 2)])
    with pytest.raises(ArenaError, match="three"):
        fit_affine([(0, 0), (1, 1)], [(0, 0), (1, 1)])
    results = {"map:tl": (0.1, 0.1), "map:br": (0.9, 0.9), **{f"province:{p}": v for p, v in hovered.items()},
               "roi:date:tl": (0.4, 0.0), "roi:date:br": (0.5, 0.03), "point:reset_decision": (0.25, 0.5),
               "point:deselect": None, "map_color:BLU": (10, 20, 200), "panel:row1:tl": (0.0, 0.3),
               "panel:row1:br": (0.2, 0.32), "panel:row2:tl": (0.0, 0.33)}
    assert {s.id for s in build_steps(LAYOUT)} >= set(results)
    assembled, fit = assemble(ArenaVisionCalibration(), LAYOUT, results, 1920, 1080)
    assert fit is not None and fit < 1e-9 and (assembled.width, assembled.map_rect) == (1920, (0.1, 0.1, 0.9, 0.9))
    assert assembled.points == {"reset_decision": (250, 500)} and assembled.map_colors == {"BLU": (10, 20, 200)}
    assert assembled.panel.enabled and assembled.panel.row_pitch == pytest.approx(0.03)
    path = tmp_path / "vision.toml"
    path.write_text(dump_vision_toml(assembled), encoding="utf-8")
    loaded = load_vision_calibration(path)
    assert np.allclose(loaded.affine, assembled.affine, atol=1e-6)  # the TOML keeps six decimals
    assert replace(loaded, affine=None) == replace(assembled, affine=None)
    shipped = load_vision_calibration(Path(__file__).parent.parent / "config" / "arena" / "vision.toml")
    assert shipped.points == {} and not shipped.panel.enabled and shipped.hotkeys.halt == "h"
    with pytest.raises(ArenaError):
        load_vision_calibration(tmp_path / "missing.toml")


def test_cli_vision_read_and_audit(tmp_path, capsys):
    parser = argparse.ArgumentParser()
    handlers = add_commands(parser.add_subparsers(dest="command", required=True))
    assert set(handlers) == {"vision-read", "vision-observe", "calibrate-arena", "vision-audit"}
    args = parser.parse_args(["vision-read", "--image", str(DATA / "sample_map_600x400.png")])
    assert handlers[args.command](args) == 0
    assert [c["count"] for c in json.loads(capsys.readouterr().out)["counters"]] == [5, 7]
    cal = calibration()
    observation = observer(cal).observe(render_arena_frame(LAYOUT, STACKS, cal))
    (tmp_path / "observation.json").write_text(json.dumps({"observation": observation.to_dict()}), encoding="utf-8")
    (tmp_path / "save.hoi4").write_text(SAVE_SNIPPET, encoding="utf-8")
    args = parser.parse_args(["vision-audit", "--observation", str(tmp_path / "observation.json"),
                              "--save", str(tmp_path / "save.hoi4")])
    assert handlers[args.command](args) == 0
    assert json.loads(capsys.readouterr().out)["own_province_recall"] == 1.0


# --- second real sample: a busy 1942 front at 1680x1050 -------------------------
def hard_sample():
    truth = json.loads((DATA / "sample_front_1680x1050.truth.json").read_text(encoding="utf-8"))
    return find_counters(Image.open(DATA / "sample_front_1680x1050.png")), truth


def test_hard_real_sample_recall_precision_and_counts_against_hand_labels():
    readings, truth = hard_sample()
    report = counter_truth_report(readings, truth)
    for relation in ("own", "enemy", "other"):
        assert report[relation]["recall_full"] == 1.0  # every fully visible counter
        assert report[relation]["precision"] == 1.0 and report[relation]["false_positives"] == 0
        assert report[relation]["count_accuracy"] == 1.0
    assert (report["own"]["found"], report["other"]["found"]) == (12, 8)
    # 14 full + 4 of 7 partial labels; misses: one under the top-bar UI, two cut by the screen edge
    assert report["enemy"]["found"] == 18 and report["enemy"]["recall"] >= 0.85
    assert report["enemy"]["count_labelled"] == 15 and report["naval_misread_as_land"] == 0
    assert all(r.count_measured for r in readings if r.count is not None)  # only measured glyphs were needed
    assert {r.count for r in readings} == {None, 1, 2, 3, 4, 5}


def test_hard_real_sample_layouts_stacks_and_occlusion():
    readings, _ = hard_sample()
    assert all(0.95 < r.scale <= 1.0 for r in readings)  # the 62 px no-army frame is not a bigger counter
    column = sorted((r for r in readings if r.relation == "other"), key=lambda r: r.bbox[1])
    assert [r.count for r in column] == [1, 1, 2, 1, 2, 2, 1, 2]
    assert {b.bbox[1] - a.bbox[1] for a, b in zip(column, column[1:])} == {24}  # stacked 2 px apart
    hidden = [r for r in readings if r.occluded]
    assert [(r.bbox[:2], r.count) for r in hidden] == [((895, 361), None), ((1526, 634), None),
                                                       ((1542, 638), 1), ((950, 668), None)]
    # a covered counter never reports bars it cannot see, nor the count of the counter drawn over it
    assert all(r.organization is None and r.strength is None and r.confidence < 0.9 for r in hidden)
    assert all(r.organization is not None and r.strength is not None for r in readings if not r.occluded)


def test_hard_real_sample_bars_vary_and_the_empty_part_is_the_dark_box():
    """Bar VALUES cannot be verified without a save. What can: fills vary, and they rest on real
    partly filled bars whose empty part is the measured dark background, not a guess."""
    readings, _ = hard_sample()
    visible = [r for r in readings if not r.occluded]
    assert len({round(r.organization, 2) for r in visible}) >= 8
    assert min(r.organization for r in visible) < 0.1 and max(r.organization for r in visible) == 1.0
    assert {round(r.strength, 2) for r in visible} >= {0.76, 0.88, 1.0}
    pixels = np.asarray(Image.open(DATA / "sample_front_1680x1050.png").convert("RGB"))
    low = next(r for r in visible if r.bbox[:2] == (549, 606))  # own 4-stack with one lit org column
    assert low.organization == pytest.approx(1 / 17) and low.strength == 1.0
    assert tuple(pixels[620, 562]) == CounterStyle().org_rgb  # the lit column
    empty = pixels[620, 563:576].astype(int)
    assert (empty == (0x29, 0x25, 0x22)).all() and empty.max() < CounterStyle().bar_lit_value


def test_synthetic_other_relation_full_frame_and_touching_stack():
    from hoi4_agent.arena.vision.synthetic import draw_counter
    style = CounterStyle()
    pixels = np.full((200, 300, 3), (74, 80, 88), dtype=np.uint8)
    stacks = [SyntheticStack(1, "other", 2, 0.5, 1.0, army_plate=False),
              SyntheticStack(1, "enemy", 14, 1.0, 0.25, army_plate=False),
              SyntheticStack(1, "own", 3, 0.75, 0.5)]
    for index, stack in enumerate(stacks):  # pitch 22: each frame touches the next, no gap at all
        draw_counter(pixels, 40, 30 + 22 * index, stack, style)
    readings = find_counters(pixels)
    assert [(r.relation, r.count, r.bbox[1], r.bbox[2] - r.bbox[0], r.has_army_plate) for r in readings] == [
        ("other", 2, 30, 62, False), ("enemy", 14, 52, 62, False), ("own", 3, 74, 53, True)]
    assert [round(r.organization * 17) for r in readings] == [8, 17, 13]
    cal = calibration()
    observation = observer(cal).observe(render_arena_frame(
        LAYOUT, [SyntheticStack(6, "other", 2, 1.0, 1.0, army_plate=False), SyntheticStack(7, "enemy", 4, 1.0, 1.0)], cal))
    assert [(u.country, u.province_id) for u in observation.units] == [(Country.RED, 7)]  # "other" is no combatant
