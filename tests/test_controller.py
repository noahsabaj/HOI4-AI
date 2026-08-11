from pathlib import Path

import pytest

from hoi4_agent.brain.decide import Brain
from hoi4_agent.brain.llm import ScriptedBackend
from hoi4_agent.calibration import default_calibration
from hoi4_agent.context import AgentContext
from hoi4_agent.controller import cadence, recovery
from hoi4_agent.controller.loop import run
from hoi4_agent.enums import BuildingType, GermanState, PanelId, PreconditionKind, Tech, ToolName, Verdict
from hoi4_agent.errors import BackendUnavailableError, HaltAndFlag, ResetFailedError
from hoi4_agent.geometry import WindowGeometry
from hoi4_agent.io.backends import InputRecorder, RecordingInput
from hoi4_agent.perception.templates import TemplateStore
from hoi4_agent.playbook.loader import load_playbook
from hoi4_agent.playbook.select import all_done
from hoi4_agent.schemas import GameDate, Goal, Intent, PlaybookState, Precondition, WorldState
from hoi4_agent.testing import FakeGame
from hoi4_agent.trace.writer import JsonlTraceWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAYBOOK = REPO_ROOT / "config" / "playbooks" / "germany_1936.toml"


def _fakegame_ctx(cfg, fg: FakeGame | None = None):
    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = fg or FakeGame(calibration=calib)
    ctx = AgentContext(
        config=cfg, geometry=fg.geometry, input=fg, capture=fg,
        calibration=fg.calib, templates=TemplateStore(),
        # canned judgment for the playbook's repeatable research_refill goal
        brain=Brain(ScriptedBackend(['{"tech": "industry_2"}'])),
        mode=cfg.mode, perceive=fg.perceive, sleep=lambda _s: None,
    )
    return ctx, fg


def test_full_run_completes(cfg, tmp_path):
    ctx, fg = _fakegame_ctx(cfg)
    goals = load_playbook(PLAYBOOK)
    final = run(ctx, goals, PlaybookState(), state_path=str(tmp_path / "s.json"), sleep=lambda _s: None)
    assert all_done(goals, final)
    assert fg.queue == 4  # four civilian factories queued
    assert fg.idle_research == 0  # research slots filled


def test_persisted_state_resumes(cfg, tmp_path):
    ctx, fg = _fakegame_ctx(cfg)
    goals = load_playbook(PLAYBOOK)
    sp = str(tmp_path / "s.json")
    run(ctx, goals, PlaybookState(), state_path=sp, sleep=lambda _s: None)
    from hoi4_agent.playbook.state import load_state

    restored = load_state(sp)
    assert all_done(goals, restored)


def _const_ctx(cfg, world):
    ctx = AgentContext(
        config=cfg, geometry=WindowGeometry(1, 0, 0, cfg.display.width, cfg.display.height),
        input=RecordingInput(), capture=None,
        calibration=default_calibration(cfg.display.width, cfg.display.height),
        templates=TemplateStore(), brain=None, mode=cfg.mode,
        perceive=lambda read_numbers=True, fields=None: world,
        sleep=lambda _s: None,
    )
    return ctx


def test_persistent_failure_halts(cfg):
    # Panel always open but queue never grows AND reset never reaches home -> halt.
    world = WorldState(open_panel=PanelId.CONSTRUCTION, construction_queue_len=5,
                       free_civ_slots=3, paused=True, confidence={"panel": 1.0, "pause": 1.0})
    ctx = _const_ctx(cfg, world)
    goal = Goal(id="b", tool=ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                state=GermanState.RUHR, precondition=Precondition(PreconditionKind.FREE_CIV_SLOT))
    with pytest.raises(HaltAndFlag):
        run(ctx, [goal], PlaybookState(), max_failures=2, sleep=lambda _s: None)


def _con_ws(q=None, f=None):
    return WorldState(open_panel=PanelId.CONSTRUCTION, construction_queue_len=q,
                      free_civ_slots=f, confidence={"panel": 1.0})


def _state_clicks(ctx, inp, state=GermanState.RUHR):
    """Clicks on the state's calibrated point (the camera-anchor click is a pan,
    not an action — it must not count as 'queued a building')."""
    return [c for c in inp.clicks if c == ctx.calibration.state_point(state)]


def test_act_with_retry_retries_only_before_acting(cfg, scripted_ctx):
    # Attempt 1: baseline queue unreadable across all read-retries -> UNCERTAIN
    # with NO click. Attempt 2 (safe re-run): baseline 1, click, post 2 -> OK.
    states = [
        _con_ws(), _con_ws(), _con_ws(), _con_ws(),   # panel check + 3 unreadable baseline reads
        _con_ws(), _con_ws(1, 2), _con_ws(2),         # panel check, baseline, post-click
    ]
    ctx, inp = scripted_ctx(states)
    result = recovery.act_with_retry(ctx, Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), max_retries=2)
    assert result.verdict is Verdict.OK
    assert result.retries == 1
    assert len(_state_clicks(ctx, inp)) == 1  # exactly ONE building queued across both attempts


def test_act_with_retry_never_reexecutes_after_click(cfg, scripted_ctx):
    # Baseline readable -> click -> post unreadable after all read-retries:
    # UNCERTAIN but retry-UNSAFE; act_with_retry must NOT run the handler again.
    states = [_con_ws(), _con_ws(1, 2), _con_ws(), _con_ws(), _con_ws()]
    ctx, inp = scripted_ctx(states)
    result = recovery.act_with_retry(ctx, Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), max_retries=2)
    assert result.verdict is Verdict.UNCERTAIN
    assert result.retry_safe is False
    assert result.retries == 0
    assert len(_state_clicks(ctx, inp)) == 1  # the one click that may have landed — never repeated


def test_not_ready_advances_time_instead_of_failing(cfg, tmp_path):
    # free_civ starts at 0: the build tool reports NotReady, the loop advances
    # time (FakeGame regenerates capacity), and the build then succeeds.
    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = FakeGame(calibration=calib, free_civ=0, max_free_civ=1)
    ctx, fg = _fakegame_ctx(cfg, fg)
    goal = Goal(id="b", tool=ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                state=GermanState.RUHR, precondition=Precondition(PreconditionKind.FREE_CIV_SLOT))
    trace = tmp_path / "t.jsonl"
    with JsonlTraceWriter(trace) as w:
        final = run(ctx, [goal], PlaybookState(), writer=w, sleep=lambda _s: None)
    assert all_done([goal], final)
    assert fg.queue == 1
    records = JsonlTraceWriter.read(trace)
    assert any(r.kind == "advance" for r in records)  # the wait was a wait, not a failure


def test_recover_true_false(cfg):
    ctx, _ = _fakegame_ctx(cfg)
    assert recovery.recover(ctx) is True  # FakeGame escapes/f1 -> home
    stuck = _const_ctx(cfg, WorldState(open_panel=PanelId.CONSTRUCTION, confidence={"panel": 1.0}))
    assert recovery.recover(stuck) is False


def test_run_to_date_advances_and_repauses(cfg):
    ctx, fg = _fakegame_ctx(cfg)
    target = GameDate(1936, 1, 8)
    last = cadence.run_to_date(ctx, target, sleep=lambda _s: None)
    assert last is not None and last >= target
    assert fg.paused is True


def test_run_to_date_blind_advance_is_bounded(cfg):
    # target=None must advance only BLIND_POLLS polls even when the date is
    # perfectly readable (previously this ran the full 240-poll budget).
    ctx, fg = _fakegame_ctx(cfg)
    sleeps: list[float] = []
    last = cadence.run_to_date(ctx, None, sleep=lambda s: sleeps.append(s))
    assert fg.paused is True
    assert len(sleeps) <= cadence.BLIND_POLLS + 1
    assert last is not None  # dates were readable throughout


def test_infra_error_is_caught_traced_and_halts(cfg, tmp_path):
    # A dead model endpoint mid-perceive must become a recovered/halted failure
    # with an "error" trace record — never a raw crash out of run().
    def raising_perceive(read_numbers=True, fields=None):
        raise BackendUnavailableError("endpoint down")

    ctx = AgentContext(
        config=cfg, geometry=WindowGeometry(1, 0, 0, cfg.display.width, cfg.display.height),
        input=RecordingInput(), capture=None,
        calibration=default_calibration(cfg.display.width, cfg.display.height),
        templates=TemplateStore(), brain=None, mode=cfg.mode,
        perceive=raising_perceive, sleep=lambda _s: None,
    )
    trace = tmp_path / "t.jsonl"
    with JsonlTraceWriter(trace) as w:
        with pytest.raises(HaltAndFlag):
            run(ctx, [Goal(id="g", tool=ToolName.OBSERVE)], PlaybookState(), writer=w, sleep=lambda _s: None)
    records = JsonlTraceWriter.read(trace)
    assert any(r.kind == "error" and r.error for r in records)


def test_pause_failure_never_acts(cfg):
    # If the game cannot be confirmed paused, no goal action may run.
    world = WorldState(paused=None, confidence={"panel": 1.0})
    ctx = _const_ctx(cfg, world)
    goal = Goal(id="b", tool=ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                state=GermanState.RUHR)
    with pytest.raises(HaltAndFlag):
        run(ctx, [goal], PlaybookState(), max_failures=2, sleep=lambda _s: None)
    # Recovery may click the harmless minimap anchor; no ACTION click is allowed.
    anchor = ctx.calibration.ui_points["minimap_anchor"]
    assert all(c == anchor for c in ctx.input.clicks)


def test_trace_records_frames_actions_and_replays(cfg, tmp_path):
    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = FakeGame(calibration=calib)
    ctx, fg = _fakegame_ctx(cfg, fg)
    ctx.input = InputRecorder(fg)  # journal inputs like the live CLI does
    goals = load_playbook(PLAYBOOK)
    trace = tmp_path / "t.jsonl"
    with JsonlTraceWriter(trace, screenshot_dir=tmp_path / "frames") as w:
        run(ctx, goals, PlaybookState(), writer=w, state_path=str(tmp_path / "s.json"), sleep=lambda _s: None)

    records = JsonlTraceWriter.read(trace)
    actions = [r for r in records if r.kind == "action"]
    assert actions, "no action records written"
    for r in actions:
        assert r.pre_screenshot and Path(r.pre_screenshot).is_file()
        assert r.post_screenshot and Path(r.post_screenshot).is_file()
    assert any(r.actions for r in actions), "no input events journaled"
    key_events = [a for r in actions for a in r.actions if a.get("kind") == "key"]
    assert key_events, "expected at least one hotkey in the journaled actions"

    # The saved frames make the trace replayable end-to-end.
    from hoi4_agent.eval.replay import replay

    out = replay(trace, probe=lambda img: img.size)
    assert out and all("replayed" in o for o in out)


def test_frames_are_unique_when_several_goals_act_in_one_cycle(cfg, tmp_path):
    # A repeatable goal that reports NotReady is stepped past and the NEXT goal
    # acts in the SAME cycle. Frame names used to key on cycle_count alone, so
    # the second goal's frames overwrote the first's and the first record ended
    # up pointing at pixels from a different action.
    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = FakeGame(calibration=calib, idle_research=0, free_civ=2, max_free_civ=2)
    ctx, fg = _fakegame_ctx(cfg, fg)
    refill = Goal(id="refill", tool=ToolName.ASSIGN_RESEARCH, repeatable=True,
                  needs_judgment=True,
                  precondition=Precondition(PreconditionKind.IDLE_RESEARCH_SLOT))
    build = Goal(id="b", tool=ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                 state=GermanState.RUHR, precondition=Precondition(PreconditionKind.FREE_CIV_SLOT))
    trace = tmp_path / "t.jsonl"
    with JsonlTraceWriter(trace, screenshot_dir=tmp_path / "frames") as w:
        run(ctx, [refill, build], PlaybookState(), writer=w, sleep=lambda _s: None)

    actions = [r for r in JsonlTraceWriter.read(trace) if r.kind == "action"]
    in_cycle_0 = [r for r in actions if r.cycle == 0]
    assert {r.plan_step for r in in_cycle_0} == {"refill", "b"}, "need two goals in one cycle"
    for kind in ("pre_screenshot", "post_screenshot"):
        paths = [getattr(r, kind) for r in actions]
        assert all(paths) and all(Path(p).is_file() for p in paths)
        assert len(set(paths)) == len(paths), f"{kind} collided across records"


def test_trace_kinds_declared_matches_the_loop():
    # schemas.TRACE_KINDS is the documented set of record kinds. Checking it
    # against the loop's actual build_record(kind=...) literals catches drift in
    # BOTH directions — an emitted-but-undeclared kind (what the stale comment
    # had) and a declared-but-never-emitted one.
    import ast
    import inspect

    from hoi4_agent.controller import loop as loop_mod
    from hoi4_agent.schemas import TRACE_KINDS, TraceRecord

    tree = ast.parse(inspect.getsource(loop_mod))
    emitted = {
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "build_record"
        for kw in node.keywords
        if kw.arg == "kind" and isinstance(kw.value, ast.Constant)
    }
    assert emitted, "found no build_record(kind=...) literals to check"
    # The action record passes no kind at all and takes the dataclass default.
    default_kind = TraceRecord.__dataclass_fields__["kind"].default
    assert emitted | {default_kind} == set(TRACE_KINDS)


def test_run_to_date_makes_exactly_blind_polls_reads(cfg):
    # With no target the contract is "advance exactly BLIND_POLLS polls"; the
    # counter used to be incremented after the check, giving BLIND_POLLS + 1.
    ctx, fg = _fakegame_ctx(cfg)
    reads = []
    inner = ctx.perceive

    def counting(read_numbers=True, fields=None):
        if fields == {"date"}:
            reads.append(1)
        return inner(read_numbers=read_numbers, fields=fields)

    ctx.perceive = counting
    cadence.run_to_date(ctx, None, sleep=lambda _s: None)
    assert len(reads) == cadence.BLIND_POLLS
    assert fg.paused is True  # and it re-pauses afterwards


def test_click_required_popup_is_dismissed_and_run_completes(cfg, tmp_path):
    # A popup escape can't close (most HOI4 events) blocks a build; recovery
    # clicks the calibrated event_option and the playbook still completes.
    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = FakeGame(calibration=calib)
    fg.spawn_popup(needs_click=True)
    ctx, fg = _fakegame_ctx(cfg, fg)
    goal = Goal(id="b", tool=ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                state=GermanState.RUHR, precondition=Precondition(PreconditionKind.FREE_CIV_SLOT))
    trace = tmp_path / "t.jsonl"
    with JsonlTraceWriter(trace) as w:
        final = run(ctx, [goal], PlaybookState(), writer=w, sleep=lambda _s: None)
    assert all_done([goal], final)
    assert fg.queue == 1
    assert fg.event_popup is False  # dismissed by the option click, not escape
    records = JsonlTraceWriter.read(trace)
    assert any(r.verdict == "failed" and "popup" in (r.verification_question or "") for r in records)


def test_pause_menu_is_escaped_and_run_completes(cfg):
    # The escape/game menu swallows panel hotkeys; perception must see it and
    # recovery must escape it (it used to be a completely invisible state).
    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = FakeGame(calibration=calib)
    fg.pause_menu = True
    ctx, fg = _fakegame_ctx(cfg, fg)
    goal = Goal(id="b", tool=ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                state=GermanState.RUHR)
    final = run(ctx, [goal], PlaybookState(), sleep=lambda _s: None)
    assert all_done([goal], final)
    assert fg.pause_menu is False
    assert fg.queue == 1


def test_repeatable_refill_fires_when_slots_free_and_never_blocks(cfg, tmp_path):
    # One research slot now, another frees while time passes for the build.
    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = FakeGame(calibration=calib, idle_research=1, free_civ=0, max_free_civ=1,
                  research_frees_every=2)
    ctx, fg = _fakegame_ctx(cfg, fg)
    refill = Goal(id="refill", tool=ToolName.ASSIGN_RESEARCH, repeatable=True,
                  needs_judgment=True,
                  precondition=Precondition(PreconditionKind.IDLE_RESEARCH_SLOT))
    build = Goal(id="b", tool=ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                 state=GermanState.RUHR, precondition=Precondition(PreconditionKind.FREE_CIV_SLOT))
    trace = tmp_path / "t.jsonl"
    with JsonlTraceWriter(trace) as w:
        final = run(ctx, [refill, build], PlaybookState(), writer=w, sleep=lambda _s: None)
    assert all_done([refill, build], final)  # repeatable never gates completion
    assert fg.queue == 1  # the build was never starved by the earlier repeatable
    refill_oks = [r for r in JsonlTraceWriter.read(trace)
                  if r.plan_step == "refill" and r.verdict == "ok"]
    assert len(refill_oks) >= 2  # initial fill + at least one re-fire after a slot freed


def test_expired_date_window_is_skipped_not_waited_on(cfg, tmp_path):
    ctx, fg = _fakegame_ctx(cfg)
    stale = Goal(id="old", tool=ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                 state=GermanState.SAXONY,
                 precondition=Precondition(PreconditionKind.DATE_BEFORE, GameDate(1935, 1, 1)))
    build = Goal(id="b", tool=ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                 state=GermanState.RUHR)
    trace = tmp_path / "t.jsonl"
    with JsonlTraceWriter(trace) as w:
        final = run(ctx, [stale, build], PlaybookState(), writer=w, sleep=lambda _s: None)
    assert all_done([stale, build], final)
    assert "old" in final.completed_goal_ids  # skipped counts as done
    assert fg.queue == 1  # only the live goal actually built
    records = JsonlTraceWriter.read(trace)
    skipped = [r for r in records if r.kind == "skipped"]
    assert len(skipped) == 1 and skipped[0].plan_step == "old"
    assert "expired" in skipped[0].verification_question


def test_advance_target_backoff_and_eta(cfg):
    d = GameDate(1936, 1, 1)
    t0, r0 = cadence.compute_advance_target(d, (), 7, 56, 0)
    assert (t0, r0) == (d.plus_days(7), "backoff 7d")
    _, r3 = cadence.compute_advance_target(d, (), 7, 56, 3)  # 7 * 2^3 = 56
    assert r3 == "backoff 56d"
    _, r9 = cadence.compute_advance_target(d, (), 7, 56, 9)
    assert r9 == "backoff 56d"  # capped
    t_blind, _ = cadence.compute_advance_target(None, (), 7, 56, 0)
    assert t_blind is None
    # An ETA may only SHORTEN the hop...
    etas = (("construction_1", "1936.02.01"),)
    t, r = cadence.compute_advance_target(d, etas, 7, 56, 9)  # backoff would be 56d
    assert t == GameDate(1936, 2, 1) and r.startswith("eta")
    # ...never extend past the backoff target, and never fire once passed.
    t, r = cadence.compute_advance_target(d, etas, 7, 56, 0)  # backoff 7d < eta
    assert t == d.plus_days(7) and r.startswith("backoff")
    t, _ = cadence.compute_advance_target(GameDate(1936, 6, 1), etas, 7, 56, 0)
    assert t == GameDate(1936, 6, 1).plus_days(7)


def test_research_eta_recorded_on_assign(cfg):
    ctx, fg = _fakegame_ctx(cfg)
    goal = Goal(id="r", tool=ToolName.ASSIGN_RESEARCH, tech=Tech.INDUSTRY_1,
                precondition=Precondition(PreconditionKind.IDLE_RESEARCH_SLOT))
    final = run(ctx, [goal], PlaybookState(),
                research_days={Tech.INDUSTRY_1: 100}, sleep=lambda _s: None)
    expected_eta = GameDate(1936, 1, 1).plus_days(100).to_str()
    assert dict(final.pending_etas) == {"industry_1": expected_eta}


def test_backoff_reflected_in_advance_records(cfg, tmp_path):
    # A build that can never proceed (no civ regen): advances back off 7->14->28.
    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = FakeGame(calibration=calib, free_civ=0, max_free_civ=0)
    ctx, fg = _fakegame_ctx(cfg, fg)
    goal = Goal(id="b", tool=ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                state=GermanState.RUHR, precondition=Precondition(PreconditionKind.FREE_CIV_SLOT))
    trace = tmp_path / "t.jsonl"
    with JsonlTraceWriter(trace) as w:
        with pytest.raises(HaltAndFlag):  # max_cycles cap fires; we only care about pacing
            run(ctx, [goal], PlaybookState(), writer=w, max_cycles=4, sleep=lambda _s: None)
    reasons = [r.verification_question for r in JsonlTraceWriter.read(trace) if r.kind == "advance"]
    assert any("backoff 7d" in x for x in reasons)
    assert any("backoff 14d" in x for x in reasons)


def test_recovery_consult_unblocks_after_hint(cfg, monkeypatch):
    from hoi4_agent.controller import recovery as rec

    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = FakeGame(calibration=calib)
    ctx, fg = _fakegame_ctx(cfg, fg)
    ctx.brain = Brain(ScriptedBackend(['{"action": "press_escape"}']))

    calls = {"n": 0}
    real_reset = rec.macros.reset_to_home

    def flaky_reset(c):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ResetFailedError("transiently stuck")
        return real_reset(c)

    monkeypatch.setattr(rec.macros, "reset_to_home", flaky_reset)
    recovered, info = rec.recover_with_consult(ctx)
    assert recovered is True
    assert info is not None and info["hint"] == "press_escape"
    assert ("key", "escape") in fg.input_calls  # the hint was actually applied


def test_recovery_consult_none_or_disabled_goes_to_halt(cfg, monkeypatch):
    import dataclasses

    from hoi4_agent.controller import recovery as rec

    monkeypatch.setattr(rec.macros, "reset_to_home",
                        lambda c: (_ for _ in ()).throw(ResetFailedError("stuck")))
    ctx, _ = _fakegame_ctx(cfg)
    ctx.brain = Brain(ScriptedBackend(['{"action": "none"}']))
    recovered, info = rec.recover_with_consult(ctx)
    assert recovered is False and info is not None and info["hint"] == "none"

    ctx2, _ = _fakegame_ctx(cfg)
    ctx2.brain = None
    assert rec.recover_with_consult(ctx2) == (False, None)

    ctx3, _ = _fakegame_ctx(cfg)
    ctx3.config = dataclasses.replace(cfg, recovery_vlm_consult=False)
    assert rec.recover_with_consult(ctx3) == (False, None)


def test_consult_record_written_when_recovery_fails(cfg, tmp_path):
    from hoi4_agent.io.backends import FakeCapture
    from PIL import Image

    # Recovery can never reach home (panel stuck open) -> consult fires, is
    # traced, hint doesn't help -> halt.
    world = WorldState(open_panel=PanelId.CONSTRUCTION, construction_queue_len=5,
                       free_civ_slots=3, paused=True, confidence={"panel": 1.0, "pause": 1.0})
    ctx = _const_ctx(cfg, world)
    ctx.brain = Brain(ScriptedBackend(['{"action": "none"}']))
    ctx.capture = FakeCapture(Image.new("RGB", (8, 8)))
    goal = Goal(id="b", tool=ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                state=GermanState.RUHR)
    trace = tmp_path / "t.jsonl"
    with JsonlTraceWriter(trace) as w:
        with pytest.raises(HaltAndFlag):
            run(ctx, [goal], PlaybookState(), writer=w, max_failures=2, sleep=lambda _s: None)
    consults = [r for r in JsonlTraceWriter.read(trace) if r.kind == "consult"]
    assert consults and consults[0].vlm_used
    assert "none" in consults[0].verification_question
    assert consults[0].raw_model_output == '{"action": "none"}'


def test_judgment_prompt_and_raw_output_are_traced(cfg, tmp_path):
    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = FakeGame(calibration=calib)
    brain = Brain(ScriptedBackend(['{"state": "ruhr"}']))
    ctx = AgentContext(
        config=cfg, geometry=fg.geometry, input=fg, capture=fg,
        calibration=calib, templates=TemplateStore(), brain=brain, mode=cfg.mode,
        perceive=fg.perceive, sleep=lambda _s: None,
    )
    goal = Goal(id="bj", tool=ToolName.BUILD_IN_STATE, building=BuildingType.CIVILIAN_FACTORY,
                state=None, needs_judgment=True,
                precondition=Precondition(PreconditionKind.FREE_CIV_SLOT))
    trace = tmp_path / "t.jsonl"
    with JsonlTraceWriter(trace) as w:
        final = run(ctx, [goal], PlaybookState(), writer=w, sleep=lambda _s: None)
    assert all_done([goal], final)

    rec = next(r for r in JsonlTraceWriter.read(trace) if r.plan_step == "bj" and r.kind == "action")
    assert rec.vlm_used is True
    assert rec.prompt and "state" in rec.prompt
    assert rec.raw_model_output == '{"state": "ruhr"}'
    assert rec.parsed_intent["state"] == "ruhr"  # the RESOLVED intent, not the template
