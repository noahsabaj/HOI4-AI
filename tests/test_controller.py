import itertools

import pytest

from hoi4_agent.calibration import default_calibration
from hoi4_agent.context import AgentContext
from hoi4_agent.controller import cadence, recovery
from hoi4_agent.controller.loop import run
from hoi4_agent.enums import BuildingType, GermanState, PanelId, PreconditionKind, ToolName, Verdict
from hoi4_agent.errors import HaltAndFlag
from hoi4_agent.io.backends import RecordingInput
from hoi4_agent.perception.templates import TemplateStore
from hoi4_agent.playbook.loader import load_playbook
from hoi4_agent.playbook.select import all_done
from hoi4_agent.schemas import GameDate, Goal, Intent, PlaybookState, Precondition, WorldState
from hoi4_agent.testing import FakeGame
from hoi4_agent.geometry import WindowGeometry

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAYBOOK = REPO_ROOT / "config" / "playbooks" / "germany_1936.toml"


def _fakegame_ctx(cfg):
    calib = default_calibration(cfg.display.width, cfg.display.height)
    fg = FakeGame(calibration=calib)
    ctx = AgentContext(
        config=cfg, geometry=fg.geometry, input=fg, capture=fg,
        calibration=calib, templates=TemplateStore(), brain=None, mode=cfg.mode,
        perceive=fg.perceive,
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
        perceive=lambda read_numbers=True: world,
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


def test_act_with_retry_retries_uncertain(cfg, scripted_ctx):
    # First execute UNCERTAIN (queue unreadable), second OK.
    def ws(q):
        return WorldState(open_panel=PanelId.CONSTRUCTION, construction_queue_len=q,
                          confidence={"panel": 1.0})
    states = [ws(None), ws(None), ws(None),  # attempt 1 -> UNCERTAIN
              ws(None), ws(1), ws(2)]         # attempt 2 -> OK (1->2)
    ctx, _ = scripted_ctx(states)
    result = recovery.act_with_retry(ctx, Intent(ToolName.BUILD_IN_STATE, state=GermanState.RUHR), max_retries=2)
    assert result.verdict is Verdict.OK
    assert result.retries == 1


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
