"""Pure parts of the calibration wizard: engine navigation, step composition,
--only filtering, and ROI->template staleness. The run loop itself needs the
live game + a human and stays untested by design."""

import pytest

from hoi4_agent.cli.calibrate import (
    TEMPLATE_SPECS,
    build_steps,
    parse_only,
    stale_templates_for,
)
from hoi4_agent.cli.wizard import Step, Wizard
from hoi4_agent.errors import ConfigError

# 10 ROIs x 2 corners + 2 buildings + 3 tabs + 8 techs + 8 states + 2 ui + 11 templates + glyphs
FULL_STEPS = 20 + 2 + 3 + 8 + 8 + 2 + 11 + 1


def _wiz(*ids, skippable=()):
    return Wizard([Step(i, f"do {i}", skippable=i in skippable) for i in ids])


def test_wizard_records_and_finishes():
    w = _wiz("a", "b")
    assert w.position == (1, 2) and not w.done
    w.record(1)
    w.record(2)
    assert w.done and w.current is None
    assert w.results == {"a": 1, "b": 2}


def test_wizard_back_keep_and_overwrite():
    w = _wiz("a", "b", "c")
    w.record(1)
    w.record(2)
    assert w.back() and w.back()  # back at "a"
    assert not w.back()  # already first
    assert w.has_value()
    w.record(10)  # re-capture overwrites
    assert w.keep()  # "b" keeps its earlier answer
    assert not w.has_value()  # "c" never answered
    assert not w.keep()  # keep without a value refuses
    w.record(3)
    assert w.done and w.results == {"a": 10, "b": 2, "c": 3}


def test_wizard_skip_only_where_allowed():
    w = _wiz("a", "b", skippable=("b",))
    assert not w.skip()
    w.record(1)
    assert w.skip()
    assert w.done and w.results["b"] is None


def test_wizard_rejects_bad_step_lists():
    with pytest.raises(ValueError):
        Wizard([])
    with pytest.raises(ValueError):
        _wiz("a", "a")


def test_build_steps_full_composition():
    steps = build_steps(None)
    assert len(steps) == FULL_STEPS
    ids = [s.id for s in steps]
    assert len(set(ids)) == len(ids)
    # ROIs first, glyphs last (templates depend on ROIs; glyphs on the date ROI)
    assert ids[0] == "roi:date:tl" and ids[-1] == "glyphs"
    glyph_step = steps[-1]
    assert glyph_step.textual and glyph_step.skippable


def test_build_steps_section_and_narrow_filters():
    assert {s.id for s in build_steps(parse_only("glyphs"))} == {"glyphs"}
    techs = build_steps(parse_only("points:techs"))
    assert len(techs) == 8 and all(s.id.startswith("tech:") for s in techs)
    one = build_steps(parse_only("points:industry_1"))
    assert [s.id for s in one] == ["tech:industry_1"]
    combo = {s.id for s in build_steps(parse_only("rois:date,templates:pause_on,glyphs"))}
    assert combo == {"roi:date:tl", "roi:date:br", "template:pause_on", "glyphs"}


def test_build_steps_tabs_section():
    tabs = build_steps(parse_only("tabs"))
    assert {s.id for s in tabs} == {"tab:industry", "tab:engineering", "tab:land_doctrine"}
    one = build_steps(parse_only("tabs:engineering"))
    assert [s.id for s in one] == ["tab:engineering"]
    # tabs sit between buildings and techs in the full flow (both research-panel)
    ids = [s.id for s in build_steps(None)]
    assert ids.index("tab:industry") < ids.index("tech:construction_1")
    assert ids.index("building:civilian_factory") < ids.index("tab:industry")


def test_parse_only_validates_loudly():
    assert parse_only(None) is None
    with pytest.raises(ConfigError):
        parse_only("bogus_section")
    with pytest.raises(ConfigError):
        parse_only("points:bogus_name")
    with pytest.raises(ConfigError):
        parse_only("glyphs:anything")  # glyphs has no narrow names
    with pytest.raises(ConfigError):
        parse_only("tabs:bogus_tab")
    with pytest.raises(ConfigError):
        parse_only(" , ")


def test_unknown_section_error_lists_every_real_section():
    # The message was a hand-written literal and had already gone stale (it
    # omitted "tabs"), telling an operator a valid section did not exist.
    with pytest.raises(ConfigError) as e:
        parse_only("bogus_section")
    for section in ("rois", "points", "tabs", "templates", "glyphs"):
        assert section in str(e.value)
        parse_only(section)  # and each one really is accepted


def test_stale_templates_track_roi_dependencies():
    # redoing the speed ROI invalidates all five speed templates...
    assert stale_templates_for(["speed"], set()) == [f"speed_{i}" for i in range(1, 6)]
    # ...unless they were recaptured in the same run
    assert stale_templates_for(["speed"], {f"speed_{i}" for i in range(1, 6)}) == []
    # the date ROI owns the glyph set
    assert stale_templates_for(["date"], set()) == ["glyph_*"]
    assert stale_templates_for(["date"], {"glyph_*"}) == []
    # number ROIs have no dependent templates
    assert stale_templates_for(["free_civ_slots"], set()) == []


def test_template_specs_cover_all_required_and_speed():
    names = {n for n, _, _ in TEMPLATE_SPECS}
    from hoi4_agent.calibration import REQUIRED_TEMPLATES, SPEED_TEMPLATES

    assert set(REQUIRED_TEMPLATES) <= names
    assert set(SPEED_TEMPLATES) <= names
