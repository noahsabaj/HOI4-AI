from pathlib import Path

import pytest

from hoi4_agent.brain.decide import Brain
from hoi4_agent.brain.llm import ScriptedBackend
from hoi4_agent.calibration import default_calibration
from hoi4_agent.context import AgentContext
from hoi4_agent.controller import cadence, recovery
from hoi4_agent.controller.loop import run
from hoi4_agent.enums import BuildingType, GermanState, PanelId, PreconditionKind, ToolName, Verdict
from hoi4_agent.errors import BackendUnavailableError, HaltAndFlag
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
