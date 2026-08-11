import dataclasses

from hoi4_agent.enums import GermanState, PanelId, ResearchTab, Tech, ToolName, Verdict
from hoi4_agent.errors import BuildInStateError, IntentValidationError, NotReadyError, PreconditionError
from hoi4_agent.schemas import Intent, WorldState
from hoi4_agent.tools.executor import execute
from hoi4_agent.tools.grounding import map_crop

CON = PanelId.CONSTRUCTION
RES = PanelId.RESEARCH


def _ws(**kw):
    kw.setdefault("confidence", {"panel": 1.0, "pause": 1.0})
    return WorldState(**kw)


def _state_clicks(ctx, inp, state=GermanState.RUHR):
    """Clicks that landed on the state's calibrated point (excludes the harmless
    minimap camera-anchor click that precedes every state click)."""
    return [c for c in inp.clicks if c == ctx.calibration.state_point(state)]


def test_build_in_state_ok(scripted_ctx):
    # ensure_panel(read_numbers=False), pre(), post()
    states = [
        _ws(open_panel=CON),
        _ws(open_panel=CON, construction_queue_len=1),
        _ws(open_panel=CON, construction_queue_len=2),
    ]
    ctx, inp = scripted_ctx(states)
    r = execute(Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), ctx)
    assert r.verdict is Verdict.OK
    assert "1->2" in r.assertion
    assert len(_state_clicks(ctx, inp)) == 1  # one state click (no building arg)
    assert len(inp.clicks) == 2  # camera anchor + state


def test_build_in_state_failed_queue_unchanged(scripted_ctx):
    states = [
        _ws(open_panel=CON),
        _ws(open_panel=CON, construction_queue_len=1),
        _ws(open_panel=CON, construction_queue_len=1),
    ]
    ctx, _ = scripted_ctx(states)
    r = execute(Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), ctx)
    assert r.verdict is Verdict.FAILED
    assert isinstance(r.error, BuildInStateError)


def test_build_in_state_panel_fails_to_open(scripted_ctx):
    # ensure_panel: first read NONE, after hotkey still NONE with high conf -> FAILED
    states = [_ws(open_panel=PanelId.NONE), _ws(open_panel=PanelId.NONE)]
    ctx, _ = scripted_ctx(states)
    r = execute(Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), ctx)
    assert r.verdict is Verdict.FAILED


def test_build_in_state_unreadable_baseline_never_clicks(scripted_ctx):
    # Queue baseline unreadable through every read-retry: UNCERTAIN, retry-safe,
    # and — the point — ZERO clicks were sent.
    states = [
        _ws(open_panel=CON),                                  # ensure_panel
        _ws(open_panel=CON, construction_queue_len=None),     # baseline read
        _ws(open_panel=CON, construction_queue_len=None),     # read-retry 1
        _ws(open_panel=CON, construction_queue_len=None),     # read-retry 2
    ]
    ctx, inp = scripted_ctx(states)
    r = execute(Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), ctx)
    assert r.verdict is Verdict.UNCERTAIN
    assert r.retry_safe is True
    assert inp.clicks == []  # not even the camera anchor — nothing was attempted


def test_build_in_state_post_unreadable_is_retry_unsafe(scripted_ctx):
    # Click happened, effect unreadable: UNCERTAIN but marked NOT retry-safe.
    states = [
        _ws(open_panel=CON),
        _ws(open_panel=CON, construction_queue_len=1, free_civ_slots=2),
        _ws(open_panel=CON, construction_queue_len=None),
        _ws(open_panel=CON, construction_queue_len=None),
        _ws(open_panel=CON, construction_queue_len=None),
    ]
    ctx, inp = scripted_ctx(states)
    r = execute(Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), ctx)
    assert r.verdict is Verdict.UNCERTAIN
    assert r.retry_safe is False
    assert len(_state_clicks(ctx, inp)) == 1


def test_build_in_state_no_free_civ_is_not_ready(scripted_ctx):
    states = [
        _ws(open_panel=CON),
        _ws(open_panel=CON, construction_queue_len=1, free_civ_slots=0),
    ]
    ctx, inp = scripted_ctx(states)
    r = execute(Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), ctx)
    assert r.verdict is Verdict.FAILED
    assert isinstance(r.error, NotReadyError)
    assert inp.clicks == []  # waiting, not acting


def test_assign_research_ok_and_precondition(scripted_ctx):
    ok_states = [
        _ws(open_panel=RES),
        _ws(open_panel=RES, idle_research_slots=2),
        _ws(open_panel=RES, idle_research_slots=1),
    ]
    ctx, _ = scripted_ctx(ok_states)
    r = execute(Intent(ToolName.ASSIGN_RESEARCH, tech=Tech.INDUSTRY_1), ctx)
    assert r.verdict is Verdict.OK

    no_slot = [_ws(open_panel=RES), _ws(open_panel=RES, idle_research_slots=0)]
    ctx, _ = scripted_ctx(no_slot)
    r = execute(Intent(ToolName.ASSIGN_RESEARCH, tech=Tech.INDUSTRY_1), ctx)
    assert r.verdict is Verdict.FAILED
    assert isinstance(r.error, PreconditionError)
    assert isinstance(r.error, NotReadyError)  # a wait-state, not a defect


def test_assign_research_selects_tab_before_clicking_tech(scripted_ctx):
    # tech points are tab-relative: the tab must be clicked before the tech
    states = [
        _ws(open_panel=RES),
        _ws(open_panel=RES, idle_research_slots=2),
        _ws(open_panel=RES, idle_research_slots=1),
    ]
    ctx, inp = scripted_ctx(states)
    r = execute(Intent(ToolName.ASSIGN_RESEARCH, tech=Tech.ELECTRONICS_1), ctx)
    assert r.verdict is Verdict.OK
    tab_pt = ctx.calibration.research_tab_point(ResearchTab.ENGINEERING)  # electronics -> engineering
    tech_pt = ctx.calibration.tech_point(Tech.ELECTRONICS_1)
    assert tab_pt in inp.clicks and tech_pt in inp.clicks
    assert inp.clicks.index(tab_pt) < inp.clicks.index(tech_pt)


def test_assign_research_missing_tab_point_fails_without_clicking_tech(scripted_ctx):
    states = [_ws(open_panel=RES), _ws(open_panel=RES, idle_research_slots=2)]
    ctx, inp = scripted_ctx(states)
    ctx.calibration = dataclasses.replace(ctx.calibration, research_tabs={})
    r = execute(Intent(ToolName.ASSIGN_RESEARCH, tech=Tech.INDUSTRY_1), ctx)
    assert r.verdict is Verdict.FAILED
    assert ctx.calibration.tech_point(Tech.INDUSTRY_1) not in inp.clicks  # never reached the tech


def test_ensure_paused(scripted_ctx):
    ctx, inp = scripted_ctx([_ws(paused=True)])
    assert execute(Intent(ToolName.ENSURE_PAUSED, paused=True), ctx).verdict is Verdict.OK
    assert inp.keys == []  # observed-not-assumed: no toggle needed

    ctx, inp = scripted_ctx([_ws(paused=False), _ws(paused=True)])
    assert execute(Intent(ToolName.ENSURE_PAUSED, paused=True), ctx).verdict is Verdict.OK
    assert inp.keys == ["space"]


def test_open_construction_idempotent(scripted_ctx):
    ctx, inp = scripted_ctx([_ws(open_panel=CON)])
    r = execute(Intent(ToolName.OPEN_CONSTRUCTION), ctx)
    assert r.verdict is Verdict.OK and inp.keys == []


def test_close_panels_resets(scripted_ctx):
    # _close_panels: pre() then reset_to_home [perceive CON -> escape -> perceive NONE -> f1 -> perceive NONE]
    states = [
        _ws(open_panel=CON),            # pre in _close_panels
        _ws(open_panel=CON),            # reset_to_home first read -> still open
        _ws(open_panel=PanelId.NONE),   # after escape
        _ws(open_panel=PanelId.NONE),   # after f1
    ]
    ctx, inp = scripted_ctx(states)
    r = execute(Intent(ToolName.CLOSE_PANELS), ctx)
    assert r.verdict is Verdict.OK
    assert "escape" in inp.keys and "f1" in inp.keys


def test_invalid_intent_is_failed_not_raised(scripted_ctx):
    ctx, _ = scripted_ctx([])
    r = execute(Intent(ToolName.BUILD_IN_STATE), ctx)  # missing state
    assert r.verdict is Verdict.FAILED
    assert isinstance(r.error, IntentValidationError)


def test_focus_failure_sends_no_input(scripted_ctx):
    # If the game window won't take focus, keystrokes would land in whatever IS
    # focused — the handler must fail before sending anything.
    ctx, inp = scripted_ctx([_ws(paused=False)], focus_ok=False)
    r = execute(Intent(ToolName.ENSURE_PAUSED, paused=True), ctx)
    assert r.verdict is Verdict.FAILED
    assert isinstance(r.error, PreconditionError)
    assert inp.keys == [] and inp.clicks == []

    ctx, inp = scripted_ctx([_ws(open_panel=CON, construction_queue_len=1)], focus_ok=False)
    r = execute(Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), ctx)
    assert r.verdict is Verdict.FAILED
    assert inp.keys == [] and inp.clicks == []


def _grounded_ctx(cfg, states, calib, locator_response):
    import dataclasses  # noqa: F401  (kept for symmetry with callers)

    from PIL import Image

    from hoi4_agent.brain.decide import Brain
    from hoi4_agent.brain.llm import ScriptedBackend
    from hoi4_agent.calibration import default_calibration  # noqa: F401
    from hoi4_agent.context import AgentContext
    from hoi4_agent.geometry import WindowGeometry
    from hoi4_agent.io.backends import FakeCapture, RecordingInput
    from hoi4_agent.perception.templates import TemplateStore

    it = iter(states)
    inp = RecordingInput()
    ctx = AgentContext(
        config=cfg,
        geometry=WindowGeometry(1, 0, 0, cfg.display.width, cfg.display.height),
        input=inp,
        capture=FakeCapture(Image.new("RGB", (64, 64))),
        calibration=calib,
        templates=TemplateStore(),
        brain=None,
        mode=cfg.mode,
        perceive=lambda read_numbers=True, fields=None: next(it),
        sleep=lambda _s: None,
        locator=Brain(ScriptedBackend([locator_response])) if locator_response else None,
    )
    return ctx, inp


def test_build_grounds_state_point_when_uncalibrated(cfg):
    import dataclasses

    from hoi4_agent.calibration import default_calibration

    calib = dataclasses.replace(
        default_calibration(cfg.display.width, cfg.display.height), state_points={}
    )
    states = [
        _ws(open_panel=CON),
        _ws(open_panel=CON, construction_queue_len=1, free_civ_slots=2),
        _ws(open_panel=CON, construction_queue_len=2),
    ]
    ctx, inp = _grounded_ctx(cfg, states, calib, '{"x": 640, "y": 320}')
    r = execute(Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), ctx)
    assert r.verdict is Verdict.OK  # grounded click still verified by queue +1
    assert (640, 320) in inp.clicks

    # Without a locator, the missing point is the same hard failure as before.
    ctx2, inp2 = _grounded_ctx(cfg, [_ws(open_panel=CON)], calib, None)
    r2 = execute(Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), ctx2)
    assert r2.verdict is Verdict.FAILED
    assert inp2.clicks == []


def test_reset_dismisses_popup_via_grounded_click(cfg):
    import dataclasses

    from hoi4_agent.calibration import default_calibration
    from hoi4_agent.tools.macros import reset_to_home

    calib = dataclasses.replace(
        default_calibration(cfg.display.width, cfg.display.height), ui_points={}
    )
    states = [
        _ws(event_popup=True),  # reset's first look
        _ws(),                  # after the grounded option click
        _ws(),                  # final assert after map-mode key
    ]
    ctx, inp = _grounded_ctx(cfg, states, calib, '{"x": 500, "y": 800}')
    reset_to_home(ctx)
    assert (500, 800) in inp.clicks  # popup dismissed by locate, not escape
    assert "escape" not in inp.keys


def test_build_blocked_by_event_popup(scripted_ctx):
    # A popup over the open panel swallows clicks: fail (recovery dismisses it),
    # never click through it.
    states = [_ws(open_panel=CON, event_popup=True)]
    ctx, inp = scripted_ctx(states)
    r = execute(Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), ctx)
    assert r.verdict is Verdict.FAILED
    assert isinstance(r.error, PreconditionError)
    assert "popup" in r.assertion
    assert inp.clicks == []


def test_map_crop_clears_the_whole_construction_panel(scripted_ctx):
    # The panel's right edge is not calibrated directly. Deriving the crop from
    # the "construction_panel" ROI alone was wrong once the wizard started
    # asking for "a small static region (its header)" — that box ends well
    # inside the panel, so the "map only" crop still contained most of it.
    ctx, _ = scripted_ctx([_ws()])
    rois = {
        "construction_panel": (0.0, 0.05, 0.09, 0.08),   # header only
        "construction_queue": (0.0, 0.12, 0.16, 0.99),   # the panel's real reach
        "free_civ_slots": (0.0, 0.08, 0.10, 0.12),
    }
    ctx.calibration = dataclasses.replace(ctx.calibration, rois=rois)
    crop = map_crop(ctx)
    widest = max(r[2] for r in rois.values())
    assert crop.client_x0 == round(widest * ctx.geometry.client_w)
    assert crop.client_x0 > round(rois["construction_panel"][2] * ctx.geometry.client_w)
    assert crop.crop_w == ctx.geometry.client_w - crop.client_x0
    assert crop.crop_h == ctx.geometry.client_h


def test_map_crop_falls_back_to_the_full_frame_when_uncalibrated(scripted_ctx):
    ctx, _ = scripted_ctx([_ws()])
    ctx.calibration = dataclasses.replace(ctx.calibration, rois={})
    assert map_crop(ctx) == ctx.geometry.full_crop()
