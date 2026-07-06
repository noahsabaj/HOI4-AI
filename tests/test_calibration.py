from hoi4_agent.calibration import default_calibration, dump_toml, load_calibration
from hoi4_agent.enums import ResearchTab


def test_toml_roundtrip_preserves_research_tabs(tmp_path):
    calib = default_calibration(3840, 2160)
    assert set(calib.research_tabs) == {t.value for t in ResearchTab}  # default covers all tabs
    p = tmp_path / "calibration.toml"
    p.write_text(dump_toml(calib), encoding="utf-8")
    back = load_calibration(p)
    assert back.research_tabs == calib.research_tabs
    assert back == calib  # nothing else lost in the round-trip


def test_research_tab_point_lookup():
    calib = default_calibration(3840, 2160)
    assert calib.research_tab_point(ResearchTab.INDUSTRY) == calib.research_tabs["industry"]
    import dataclasses

    empty = dataclasses.replace(calib, research_tabs={})
    assert empty.research_tab_point(ResearchTab.INDUSTRY) is None  # uncalibrated -> None
