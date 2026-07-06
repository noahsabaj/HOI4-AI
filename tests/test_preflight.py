import dataclasses
from pathlib import Path

import numpy as np

from hoi4_agent.calibration import REQUIRED_TEMPLATES, default_calibration
from hoi4_agent.enums import ToolName
from hoi4_agent.perception.templates import TemplateStore
from hoi4_agent.playbook.loader import load_playbook
from hoi4_agent.preflight import preflight
from hoi4_agent.schemas import Goal

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAYBOOK = REPO_ROOT / "config" / "playbooks" / "germany_1936.toml"


def _full_templates() -> TemplateStore:
    store = TemplateStore()
    pat = np.arange(12, dtype=np.float32).reshape(3, 4)
    for name in REQUIRED_TEMPLATES:
        store.add(name, pat)
    for s in range(1, 6):
        store.add(f"speed_{s}", pat)
    store.add("glyph_0", pat)
    return store


def test_empty_templates_block_a_live_run(cfg):
    goals = load_playbook(PLAYBOOK)
    calib = default_calibration(cfg.display.width, cfg.display.height)
    errors, warnings = preflight(cfg, calib, TemplateStore(), goals)
    for name in REQUIRED_TEMPLATES:
        assert any(name in e for e in errors), f"no error names missing template {name}"
    assert any("glyph" in w for w in warnings)
    assert any("speed" in w for w in warnings)


def test_fully_provisioned_is_clean(cfg):
    goals = load_playbook(PLAYBOOK)
    calib = default_calibration(cfg.display.width, cfg.display.height)
    errors, warnings = preflight(cfg, calib, _full_templates(), goals)
    assert errors == []
    assert warnings == []


def test_resolution_mismatch_is_error(cfg):
    goals = load_playbook(PLAYBOOK)
    calib = default_calibration(1920, 1080)
    errors, _ = preflight(cfg, calib, _full_templates(), goals)
    assert any("recalibrate" in e for e in errors)


def test_missing_roi_and_click_point_reported(cfg):
    goals = load_playbook(PLAYBOOK)
    calib = default_calibration(cfg.display.width, cfg.display.height)
    calib = dataclasses.replace(
        calib,
        state_points={},
        rois={k: v for k, v in calib.rois.items() if k != "date"},
    )
    errors, _ = preflight(cfg, calib, _full_templates(), goals)
    assert any("state:ruhr" in e for e in errors)
    assert any("'date'" in e and "ROI" in e for e in errors)


def test_invalid_goal_and_empty_judgment_options(cfg):
    calib = default_calibration(cfg.display.width, cfg.display.height)
    bad = Goal(id="bad", tool=ToolName.ASSIGN_RESEARCH)  # tech missing, no judgment
    errors, _ = preflight(cfg, calib, _full_templates(), [bad])
    assert any("'bad'" in e and "intent" in e for e in errors)

    judged = Goal(id="j", tool=ToolName.BUILD_IN_STATE, needs_judgment=True)
    no_states = dataclasses.replace(calib, state_points={})
    errors, _ = preflight(cfg, no_states, _full_templates(), [judged])
    assert any("'j'" in e and "judgment" in e for e in errors)
