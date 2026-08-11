import pytest

from hoi4_agent import clausewitz
from hoi4_agent.errors import ConfigError
from hoi4_agent.eval.savefile import read_save_facts
from hoi4_agent.gui_import import draft_calibration, load_elements

# --- clausewitz parser --------------------------------------------------------

SNIPPET = """
# a comment
date="1936.03.15"
name = "Weimar \\"Republic\\""
count = 42
ratio = 0.5
flag = yes
block = {
    nested = { a = 1 b = 2 }
    line = { id = 1 }
    line = { id = 2 }     # duplicate key -> list
}
list = { 10 20 30 }
empty = { }
"""


def test_clausewitz_parses_scalars_blocks_and_duplicates():
    d = clausewitz.parse(SNIPPET)
    assert d["date"] == "1936.03.15"
    assert d["name"].startswith("Weimar")
    assert d["count"] == 42 and d["ratio"] == 0.5 and d["flag"] == "yes"
    assert d["block"]["nested"] == {"a": 1, "b": 2}
    assert d["block"]["line"] == [{"id": 1}, {"id": 2}]  # duplicates collected
    assert d["list"] == [10, 20, 30]
    assert d["empty"] == {}


def test_clausewitz_magic_prefix_and_tolerance():
    d = clausewitz.parse('HOI4txt\ndate="1936.01.01"\nopen_block = { a = 1 ')  # EOF in block
    assert d["date"] == "1936.01.01"
    assert d["open_block"] == {"a": 1}
    with pytest.raises(ConfigError):
        clausewitz.parse("= 5")  # dangling '='
    with pytest.raises(ConfigError):
        clausewitz.parse_file("/no/such/file.hoi4")


# --- save-file facts -----------------------------------------------------------

SAVE = """HOI4txt
date="1936.03.15"
countries={
    GER={
        technology={ slots={ construction_1={ points=100 } industry_1={} } }
        civilian_factories=42
        construction={ line={ id=1 } line={ id=2 } line={ id=3 } }
    }
    FRA={ civilian_factories=30 }
}
"""


def test_read_save_facts(tmp_path):
    p = tmp_path / "auto.hoi4"
    p.write_text(SAVE, encoding="utf-8")
    facts = read_save_facts(p, country="GER")
    assert facts.date == "1936.03.15"
    assert facts.researching == ("construction_1", "industry_1")
    assert facts.civilian_factories == 42
    assert facts.construction_lines == 3
    assert facts.missing == ()


def test_read_save_facts_reports_missing_not_guesses(tmp_path):
    p = tmp_path / "auto.hoi4"
    p.write_text('HOI4txt\ndate="1936.01.01"\ncountries={ FRA={} }\n', encoding="utf-8")
    facts = read_save_facts(p, country="GER")
    assert facts.date == "1936.01.01"
    assert facts.researching is None and facts.civilian_factories is None
    assert any("GER" in m for m in facts.missing)


def test_dates_agree_normalizes_both_formats():
    from hoi4_agent.eval.savefile import dates_agree

    # A trace carries GameDate.to_str()'s zero-padded form; a save carries
    # year.month.day.HOUR. Comparing the raw strings reported MISMATCH for the
    # same day, which made the save cross-check useless.
    assert dates_agree("1936.01.01", "1936.1.1.12") is True
    assert dates_agree("1936.01.01", "1936.1.1") is True
    assert dates_agree("1943.12.28", "1943.12.28.06") is True
    assert dates_agree("1936.01.01", "1936.1.2.12") is False
    # undecidable is None, never a silent False
    assert dates_agree(None, "1936.1.1.12") is None
    assert dates_agree("1936.01.01", None) is None
    assert dates_agree("1936.01.01", "not a date") is None


# --- gui-import ----------------------------------------------------------------

GUI = """
guiTypes = {
    containerWindowType = {
        name = "topbar"
        position = { x = 0 y = 0 }
        size = { width = 2560 height = 40 }
        instantTextBoxType = {
            name = "DateText"
            position = { x = 1178 y = 17 }
            size = { width = 154 height = 32 }
        }
        iconType = {
            name = "anchored_thing"
            position = { x = -50 y = 0 }
            size = { width = 20 height = 20 }
        }
    }
}
"""


def test_gui_import_drafts_recognized_rois(tmp_path):
    p = tmp_path / "topbar.gui"
    p.write_text(GUI, encoding="utf-8")
    elements = load_elements(tmp_path)
    names = {e.name for e in elements}
    assert {"topbar", "DateText", "anchored_thing"} <= names

    calib, report = draft_calibration(elements, 2560, 1440)
    assert "date" in calib.rois
    fx0, fy0, fx1, fy1 = calib.rois["date"]
    assert fx0 == round(1178 / 2560, 4) and fy1 == round((17 + 32) / 1440, 4)
    assert "date" in report["mapped"]
    assert "pause" in report["unmapped_rois"]  # nothing matched it


def test_gui_import_cli_defaults_to_the_configured_display(tmp_path, cfg, capsys):
    # A draft calibration computed against a resolution the agent isn't running
    # at is worse than useless, so the dimensions default to [display].
    from hoi4_agent.calibration import load_calibration
    from hoi4_agent.cli.main import main

    src = tmp_path / "topbar.gui"
    src.write_text(GUI, encoding="utf-8")
    out = tmp_path / "draft.toml"
    assert main(["gui-import", str(src), "--out", str(out)]) == 0

    drafted = load_calibration(out)
    assert (drafted.width, drafted.height) == (cfg.display.width, cfg.display.height)
    assert f"{cfg.display.width}x{cfg.display.height}" in capsys.readouterr().out
    # and an explicit override still wins
    assert main(["gui-import", str(src), "--out", str(out), "--width", "1920",
                 "--height", "1080"]) == 0
    assert (load_calibration(out).width, load_calibration(out).height) == (1920, 1080)
