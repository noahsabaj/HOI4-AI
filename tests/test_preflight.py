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
    """Everything a live run wants: T0 templates, all five speeds, and a glyph
    set covering the WHOLE date strip — a digits-only set reads numbers but not
    the date, which preflight now says out loud."""
    from hoi4_agent.perception.digits import DATE_STRIP_CHARS, template_name_for

    store = TemplateStore()
    pat = np.arange(12, dtype=np.float32).reshape(3, 4)
    for name in REQUIRED_TEMPLATES:
        store.add(name, pat)
    for s in range(1, 6):
        store.add(f"speed_{s}", pat)
    for ch in DATE_STRIP_CHARS:
        store.add(template_name_for(ch), pat)
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


def test_grounding_enabled_warns_to_validate_first(cfg):
    import dataclasses

    goals = load_playbook(PLAYBOOK)
    calib = default_calibration(cfg.display.width, cfg.display.height)
    grounded_cfg = dataclasses.replace(cfg, grounding=cfg.llm)
    errors, warnings = preflight(grounded_cfg, calib, _full_templates(), goals)
    assert errors == []
    assert any("grounding" in w and "M0" in w for w in warnings)


def test_tech_tabs_covers_every_tech():
    # invariant: a tech with no tab mapping would be unreachable (executor + preflight
    # both key off TECH_TABS) — so every Tech member must be present
    from hoi4_agent.enums import TECH_TABS, Tech

    assert set(TECH_TABS) == set(Tech)


def test_uncalibrated_research_tab_blocks_run(cfg):
    goals = load_playbook(PLAYBOOK)
    calib = default_calibration(cfg.display.width, cfg.display.height)
    # drop the engineering tab point while electronics_1/radar_1 stay calibrated
    no_eng = dataclasses.replace(
        calib, research_tabs={k: v for k, v in calib.research_tabs.items() if k != "engineering"}
    )
    errors, _ = preflight(cfg, no_eng, _full_templates(), goals)
    assert any("engineering" in e and "tab" in e for e in errors)
    # and it names the fix
    assert any("--only tabs" in e for e in errors)


def test_invalid_goal_and_empty_judgment_options(cfg):
    calib = default_calibration(cfg.display.width, cfg.display.height)
    bad = Goal(id="bad", tool=ToolName.ASSIGN_RESEARCH)  # tech missing, no judgment
    errors, _ = preflight(cfg, calib, _full_templates(), [bad])
    assert any("'bad'" in e and "intent" in e for e in errors)

    judged = Goal(id="j", tool=ToolName.BUILD_IN_STATE, needs_judgment=True)
    no_states = dataclasses.replace(calib, state_points={})
    errors, _ = preflight(cfg, no_states, _full_templates(), [judged])
    assert any("'j'" in e and "judgment" in e for e in errors)


def test_undecodable_glyph_file_is_not_reported_as_a_glyph_tier(cfg):
    # preflight used to pattern-match the "glyph_" file-name prefix while the
    # reader decodes the name to a character. A glyph_*.png the reader can't
    # decode must not be reported as a working deterministic tier.
    from hoi4_agent.perception.digits import GlyphReader

    goals = load_playbook(PLAYBOOK)
    calib = default_calibration(cfg.display.width, cfg.display.height)
    templates = _full_templates()

    junk = TemplateStore()
    pat = np.arange(12, dtype=np.float32).reshape(3, 4)
    for name in REQUIRED_TEMPLATES:
        junk.add(name, pat)
    junk.add("glyph_not_a_character", pat)

    assert GlyphReader(junk).available() is False
    _, warnings = preflight(cfg, calib, junk, goals)
    assert any("glyph_" in w for w in warnings), "silent about an unusable glyph tier"

    # ...and a real one is not warned about
    _, warnings = preflight(cfg, calib, templates, goals)
    assert not any("glyph_" in w for w in warnings)


def test_partial_glyph_set_warns_that_dates_cannot_be_read(cfg):
    # A digits-only set reads numbers fine but silently cannot read the date
    # (colon, commas, month NAME). Time advance polls the date every couple of
    # seconds, so a quiet fallback here is a VLM call per poll.
    from hoi4_agent.perception.digits import (
        DATE_STRIP_CHARS,
        missing_date_glyphs,
        template_name_for,
    )

    goals = load_playbook(PLAYBOOK)
    calib = default_calibration(cfg.display.width, cfg.display.height)
    pat = np.arange(12, dtype=np.float32).reshape(3, 4)

    def _store(chars):
        store = TemplateStore()
        for name in REQUIRED_TEMPLATES:
            store.add(name, pat)
        for ch in chars:
            store.add(template_name_for(ch), pat)
        return store

    # Exactly the set the repo shipped with: digits from one 1936 capture.
    digits_only = _store("012369")
    missing = missing_date_glyphs(digits_only)
    assert ":" in missing and "," in missing and "J" in missing
    _, warnings = preflight(cfg, calib, digits_only, goals)
    assert any("cannot read the in-game date" in w for w in warnings)
    assert any("--only glyphs" in w for w in warnings)

    # A complete set is silent.
    complete = _store(DATE_STRIP_CHARS)
    assert missing_date_glyphs(complete) == []
    _, warnings = preflight(cfg, calib, complete, goals)
    assert not any("date" in w for w in warnings)


def test_date_strip_alphabet_covers_every_month_name():
    from hoi4_agent.perception.digits import DATE_STRIP_CHARS
    from hoi4_agent.schemas import MONTH_ABBREVS

    assert len(MONTH_ABBREVS) == 12
    for name in MONTH_ABBREVS:
        assert all(ch in DATE_STRIP_CHARS for ch in name), name
    assert all(d in DATE_STRIP_CHARS for d in "0123456789")
    assert ":" in DATE_STRIP_CHARS and "," in DATE_STRIP_CHARS
    assert " " not in DATE_STRIP_CHARS  # spaces leave no ink to segment
