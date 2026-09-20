"""Arena mod generator tests. The "install" here is a tiny SYNTHETIC fixture, not HOI4 data;
only ``test_default_region_on_real_install`` reads the real game, and it never launches it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from PIL import Image

from hoi4_agent.arena.contracts import ArenaError, Country
from hoi4_agent.arena.fingerprint import tree_hash
from hoi4_agent.arena.layout import ArenaLayout
from hoi4_agent.arena.mod.generate import (
    HANDICAPS,
    VARIANTS,
    ModConfig,
    add_commands,
    build_mod,
    choose_tags,
    exposed_vps,
    placements,
    plan_states,
    validate_mod,
)
from hoi4_agent.arena.mod.region import DEFAULT_GAME, DEFAULT_REGION, RegionSpec, asymmetry, extract_region
from hoi4_agent.clausewitz import parse

CELL = 4  # pixels per province edge: 4 columns x 3 rows of provinces, IDs 1..12 row-major


def _province(row: int, col: int) -> int:
    return row * 4 + col + 1


SPEC = RegionSpec("synthetic_grid", (1, 2), tuple(_province(r, c) for r in range(3) for c in (0, 1)),
                  _province(1, 0), _province(1, 3),
                  ((_province(1, 0), 10.0), (_province(1, 3), 10.0), (_province(1, 1), 3.0), (_province(1, 2), 3.0)),
                  margin=0.1, view_aspect=1.0)


@pytest.fixture()
def fake_game(tmp_path: Path) -> Path:
    game = tmp_path / "fake_install"
    (game / "map").mkdir(parents=True)
    (game / "history" / "states").mkdir(parents=True)
    (game / "common" / "country_tags").mkdir(parents=True)
    image = Image.new("RGB", (4 * CELL + 2, 3 * CELL + 2), (0, 0, 200))  # one pixel of sea all round
    rivers = Image.new("P", image.size, 255)
    rivers.putpalette([v for i in range(256) for v in (i, i, i)])  # Pillow cannot re-read a palette-less BMP
    rows = ["0;0;0;0;land;false;unknown;0", "13;0;0;200;sea;true;ocean;0", "14;9;9;9;land;false;plains;1"]
    image.paste((9, 9, 9), (0, 0, 1, image.size[1]))  # an outside land strip west of the arena (state 3)
    for r in range(3):
        for c in range(4):
            pid = _province(r, c)
            color = (10 * pid, 50 + r, 100 + c)
            rows.append(f"{pid};{color[0]};{color[1]};{color[2]};land;false;{'forest' if r == 0 else 'plains'};1")
            image.paste(color, (1 + c * CELL, 1 + r * CELL, 1 + (c + 1) * CELL, 1 + (r + 1) * CELL))
    for y in range(1, 1 + 3 * CELL):  # a river down the first pixel column of province column 2
        rivers.putpixel((1 + 2 * CELL, y), 4)
    image.save(game / "map" / "provinces.bmp")
    rivers.save(game / "map" / "rivers.bmp")
    (game / "map" / "definition.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (game / "map" / "adjacencies.csv").write_text(
        "From;To;Type;Through;start_x;start_y;stop_x;stop_y;adjacency_rule_name;Comment\n"
        f"{_province(0, 0)};{_province(1, 0)};impassable;-1;-1;-1;-1;-1;;wall\n"
        f"{_province(0, 0)};{_province(2, 0)};sea;13;-1;-1;-1;-1;;ferry\n-1;-1;;-1;-1;-1;-1;-1;-1\n", encoding="utf-8")
    for state, cols in ((1, (0, 1)), (2, (2, 3))):
        members = " ".join(str(_province(r, c)) for r in range(3) for c in cols)
        (game / "history" / "states" / f"{state}-Synthetic.txt").write_text(
            f'﻿state={{\n id={state}\n name="STATE_{state}"\n manpower = 5\n state_category = rural\n'
            f" history={{ owner = AAA victory_points = {{ {_province(1, cols[0])} 2 }} }}\n"
            f" provinces={{ {members} }}\n}}\n", encoding="utf-8")
    (game / "history" / "states" / "3-Outside.txt").write_text(
        'state={\n id=3\n name="STATE_3"\n history={\n  owner = AAA\n }\n provinces={ 14 }\n}\n', encoding="utf-8")
    (game / "events").mkdir()
    (game / "events" / "News.txt").write_text(
        "add_namespace = news\nnews_event = { # first\n\tid = news.1\n\toption = { name = a }\n}\n"
        "country_event = {\n\tid = other.1\n}\nnews_event = {\n\tid = news.2\n}\n", encoding="utf-8")
    (game / "events" / "Other.txt").write_text("country_event = {\n\tid = other.2\n}\n", encoding="utf-8")
    (game / "common" / "country_tags" / "00_countries.txt").write_text('AAA = "countries/A.txt"\n', encoding="utf-8")
    # Two strategic regions that cut ACROSS both states (row 0 | rows 1-2), as the vanilla map does.
    (game / "map" / "strategicregions").mkdir()
    for rid, members in ((1, "1 2 3 4 13 14"), (2, "5 6 7 8 9 10 11 12")):
        (game / "map" / "strategicregions" / f"{rid}-Synthetic.txt").write_text(
            f'strategic_region={{\n\tid={rid}\n\tname="R{rid}"\n\tprovinces={{\n\t\t{members}\n\t}}\n'
            "\tweather={ period={ between={ 0.0 30.0 } } }\n}\n", encoding="utf-8")
    # buildings.txt: state;type;x;height;z;rotation;sea, with z counted from the BOTTOM bitmap row.
    bottom = 3 * CELL + 2 - 1
    (game / "map" / "buildings.txt").write_text(
        f"1;arms_factory;2.00;9.50;{bottom - 2}.00;0.00;0\n"  # pixel (2, 2): province 1 of state 1
        f"2;arms_factory;{1 + 3 * CELL}.00;9.50;{bottom - 2}.00;0.00;0", encoding="utf-8")  # province 4; like vanilla, no final newline
    return game


def test_region_extraction_and_adjacency(fake_game: Path) -> None:
    region = extract_region(fake_game, SPEC)
    assert [p.id for p in region.provinces] == list(range(1, 13))
    assert region.province(1).terrain == "forest" and region.province(5).terrain == "plains"
    assert region.neighbors[6] == (2, 5, 7, 10)  # 4-neighbour grid, no diagonals
    assert 5 not in region.neighbors[1] and 1 not in region.neighbors[5]  # impassable row removes the link
    assert 9 in region.neighbors[1] and 1 in region.neighbors[9]  # any other adjacency row adds one
    # The river runs along the column 1 | column 2 border; links crossed at a single pixel are not crossings.
    assert region.river_neighbors[6] == (7,) and region.river_neighbors[7] == (6,)
    assert region.river_neighbors[3] == (2,) and 7 not in region.river_neighbors[3]
    assert [s.victory_points for s in region.states] == [((5, 2.0),), ((7, 2.0),)]
    assert {p.id for p in region.front(Country.BLUE)} == {2, 6, 10}
    assert {p.sector for p in region.provinces if p.id <= 4} == {"north"}
    assert {p.sector for p in region.provinces if p.id >= 9} == {"south"}


def test_centroids_are_normalized_with_margin(fake_game: Path) -> None:
    region = extract_region(fake_game, SPEC)
    assert region.bbox == (1, 1, 1 + 4 * CELL, 1 + 3 * CELL)
    first, last = region.province(1), region.province(12)
    assert (first.map_x, first.map_y) == (3.0, 3.0)  # centre of pixels 1..4
    # 16 x 12 pixels fitted into a square view with a 0.1 margin: x spans the full 0.8, y is centred.
    assert first.x == pytest.approx(0.1 + 0.8 * 2 / 16) and last.x == pytest.approx(0.9 - 0.8 * 2 / 16)
    assert first.y == pytest.approx(0.2 + 0.6 * 2 / 12) and last.y == pytest.approx(0.8 - 0.6 * 2 / 12)
    assert all(0.1 <= p.x <= 0.9 and 0.1 <= p.y <= 0.9 for p in region.provinces)
    layout = region.layout()
    assert layout.source == "vanilla_region" and layout.sectors == ("center", "north", "south")
    assert layout.capital(Country.BLUE).id == 5 and layout.capital(Country.RED).id == 8
    assert layout.province(6).victory_points == 3.0 and layout.province(6).initial_controller == "BLU"


def test_region_rejects_bad_specs(fake_game: Path) -> None:
    with pytest.raises(ArenaError):
        extract_region(fake_game, RegionSpec("x", (1, 2), (1, 2), 5, 8, ()))  # capital not on its side
    with pytest.raises(ArenaError):
        extract_region(fake_game, RegionSpec("x", (1, 2), (1, 12), 1, 8, ()))  # blue side is not connected
    with pytest.raises(ArenaError):
        extract_region(fake_game, RegionSpec("x", (1, 99), (1,), 1, 8, ()))  # unknown state


def test_state_plan_and_placements(fake_game: Path) -> None:
    region = extract_region(fake_game, SPEC)
    plans = plan_states(region)
    assert [(p.id, p.owner, p.land) for p in plans] == [(1, "BLU", (1, 2, 5, 6, 9, 10)), (2, "RED", (3, 4, 7, 8, 11, 12))]
    for variant in VARIANTS:
        for level in HANDICAPS:
            blue, red = (placements(region, variant, c, level) for c in Country)
            assert len(blue) == len(red) >= 1
            assert {p for p, _ in blue} <= set(plans[0].land) and {p for p, _ in red} <= set(plans[1].land)
            assert sum(t == "Arena Armor" for _, t in blue) == sum(t == "Arena Armor" for _, t in red)
    line = next(v for v in VARIANTS if v.id == "inf4_line")
    assert [p for p, _ in placements(region, line, Country.BLUE)] == [6, 2, 10, 6]  # center, north, south, center
    assert {v.split for v in VARIANTS} == {"train", "validation", "held_out"}
    # Third live load: front provinces that carry victory points are garrisoned first, and what stays
    # exposed (an empty VP next to an enemy start province) is reported instead of looking like a bug.
    wide = RegionSpec("synthetic_vp", SPEC.state_ids, SPEC.blue_provinces, SPEC.blue_capital, SPEC.red_capital,
                      (*SPEC.victory_points, (10, 2.0), (11, 2.0)), margin=0.1, view_aspect=1.0)
    prized = extract_region(fake_game, wide)
    two = next(v for v in VARIANTS if v.id == "inf2_line")
    assert [p for p, _ in placements(prized, two, Country.BLUE)] == [6, 2]  # center VP first, then north
    assert exposed_vps(prized, two) == {"BLU": [], "RED": []}  # 10 and 11 are empty but so are 11 and 10
    assert exposed_vps(prized, next(v for v in VARIANTS if v.id == "inf12_blue_defends")) == {"BLU": [], "RED": [11]}
    # The grid is mirror symmetric except for terrain-neutral details: the river lies inside RED's column.
    score = asymmetry(region)
    assert score["components"]["provinces"] == 0 and score["components"]["terrain_mix"] == 0
    assert 0 <= score["score"] < 0.2


def test_generation_is_deterministic_and_valid(fake_game: Path, tmp_path: Path) -> None:
    config = ModConfig(width=1280, height=720, default_scenario="inf4_line")
    first = build_mod(fake_game, tmp_path / "a", SPEC, config)
    second = build_mod(fake_game, tmp_path / "b", SPEC, config)
    assert first == second and first["mod_sha256"] == tree_hash(tmp_path / "a" / "mod" / "hoi4_arena")
    files = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(tmp_path / "b") for p in (tmp_path / "b").rglob("*") if p.is_file())
    for relative in files:
        if relative.as_posix() != "mod/hoi4_arena.mod":  # the only file that carries an absolute path
            assert (tmp_path / "a" / relative).read_bytes() == (tmp_path / "b" / relative).read_bytes(), relative
    assert b"\r" not in (tmp_path / "a" / "mod" / "hoi4_arena" / "common" / "scripted_effects" /
                         "arena_effects.txt").read_bytes()
    assert "x=1280 y=720" in (tmp_path / "a" / "settings.txt").read_text()
    assert json.loads((tmp_path / "a" / "dlc_load.json").read_text())["enabled_mods"] == ["mod/hoi4_arena.mod"]
    report = validate_mod(tmp_path / "a", fake_game)
    assert report["ok"], report["problems"]
    # Regressions from the first live load: no state spans strategic regions, buildings follow their state.
    mod = tmp_path / "b" / "mod" / "hoi4_arena"
    moved = (mod / "map" / "strategicregions" / "2-Synthetic.txt").read_text(encoding="utf-8")
    assert "1 2 3 4 5 6 7 8 9 10 11 12" in moved and "weather={ period=" in moved
    assert "\t\t13 14\n" in (mod / "map" / "strategicregions" / "1-Synthetic.txt").read_text(encoding="utf-8")
    rows = (mod / "map" / "buildings.txt").read_text(encoding="utf-8")
    assert rows.startswith("1;arms_factory;2.00;9.50;")
    (mod / "map" / "strategicregions" / "1-Synthetic.txt").write_text(
        (fake_game / "map" / "strategicregions" / "1-Synthetic.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (mod / "map" / "buildings.txt").write_text(rows.replace("1;arms_factory", "2;arms_factory", 1), encoding="utf-8")
    assert not rows.endswith("\n") and all(line.count(";") == 6 for line in rows.split("\n"))  # like the fixture
    # Live matches: a news window covered the map. News events are hidden, popups are off, the world is inert.
    news = (mod / "events" / "News.txt").read_text(encoding="utf-8")
    assert news.count("hidden = yes") == 2 and "news_event = { # first\n\thidden = yes" in news
    assert not (mod / "events" / "Other.txt").exists()
    assert "popup_news=no" in (tmp_path / "b" / "settings.txt").read_text()
    assert first["quiet"]["replace_path"] == [] and first["quiet"]["inert_country"] == {
        "tag": "NEU", "color": [96, 96, 96], "seed_state": 3, "seed_state_file": "3-Outside.txt",
        "seed_state_vanilla_owner": "AAA"}
    assert "owner = NEU" in (mod / "history" / "states" / "3-Outside.txt").read_text(encoding="utf-8")
    freeze = parse((mod / "common" / "scripted_effects" / "arena_world_effects.txt").read_text(encoding="utf-8"))
    assert freeze["arena_freeze_world"]["if"]["every_country"]["NEU"]["annex_country"]["target"] == "PREV"
    (mod / "events" / "News.txt").write_text(news.replace("id = news.2", "id = news.3"), encoding="utf-8")
    assert any("News.txt" in problem for problem in validate_mod(tmp_path / "b", fake_game)["problems"])
    vanilla_world = build_mod(fake_game, tmp_path / "v", SPEC, ModConfig(world="vanilla"))
    assert "inert_country" not in vanilla_world["quiet"] and validate_mod(tmp_path / "v", fake_game)["ok"]
    assert not (tmp_path / "v" / "mod" / "hoi4_arena" / "common" / "on_actions" / "arena_world_on_actions.txt").exists()
    # Second live load: on_startup is GLOBAL scope, so every entry point enters a country scope first.
    actions = parse((mod / "common" / "on_actions" / "arena_on_actions.txt").read_text(encoding="utf-8"))["on_actions"]
    assert actions["on_startup"]["effect"]["BLU"] == {"arena_startup": "yes"}
    assert actions["on_startup"]["effect"]["log"].startswith("ARENA_STARTUP")
    assert actions["on_capitulation_immediate"]["effect"]["ROOT"] == {"arena_on_capitulation": "yes"}
    names = parse((mod / "common" / "names" / "zz_arena_names.txt").read_text(encoding="utf-8"))
    assert set(names) == {"BLU", "RED"} and names["RED"]["male"]["names"] and names["RED"]["surnames"]
    (mod / "common" / "on_actions" / "arena_on_actions.txt").write_text(
        "on_actions = { on_startup = { effect = { arena_startup = yes } } }\n", encoding="utf-8")
    found = " ".join(validate_mod(tmp_path / "b", fake_game)["problems"])
    assert "'arena_startup' is called outside an explicit country scope" in found
    assert "lose or duplicate provinces" in found and "is not inside a province of state 2" in found
    assert report["files_parsed"] > 100
    layout = ArenaLayout.load(tmp_path / "a" / "arena_layout.json")
    assert len(layout.provinces) == 12
    with pytest.raises(ArenaError):
        build_mod(fake_game, tmp_path / "a", SPEC, config)  # never silently overwrites
    assert build_mod(fake_game, tmp_path / "a", SPEC, config, force=True) == first


def test_generated_script_parses_and_carries_the_episode_logic(fake_game: Path, tmp_path: Path) -> None:
    build_mod(fake_game, tmp_path / "out", SPEC)
    mod = tmp_path / "out" / "mod" / "hoi4_arena"
    effects = parse((mod / "common" / "scripted_effects" / "arena_effects.txt").read_text(encoding="utf-8"))
    assert {"arena_startup", "arena_daily", "arena_reset_episode", "arena_spawn_units", "arena_count_vp"} <= set(effects)
    text = (mod / "common" / "scripted_effects" / "arena_effects.txt").read_text(encoding="utf-8")
    for needle in ("ARENA_OUTCOME episode=", "winner=BLU reason=capital", "winner=DRAW reason=timeout_vp",
                   "ARENA_RESET episode=", "ARENA_TICK episode=", "value = 90"):
        assert needle in text
    decisions = parse((mod / "common" / "decisions" / "arena_decisions.txt").read_text(encoding="utf-8"))
    assert "arena_reset" in decisions["arena_control"] and len(decisions["arena_scenario"]) == len(VARIANTS)
    state = parse((mod / "history" / "states" / "2-Synthetic.txt").read_text(encoding="utf-8"))["state"]
    assert state["history"]["owner"] == "RED" and state["provinces"] == [3, 4, 7, 8, 11, 12]
    units = parse((mod / "history" / "units" / "ARENA_07_BLU_even.txt").read_text(encoding="utf-8"))
    assert len(units["units"]["division"]) == 12
    assert (mod / "localisation" / "english" / "arena_l_english.yml").read_bytes().startswith(b"\xef\xbb\xbfl_english:")


def test_validation_reports_undocumented_names_and_tag_collisions(fake_game: Path, tmp_path: Path) -> None:
    (fake_game / "common" / "country_tags" / "00_countries.txt").write_text('BLU = "countries/B.txt"\n')
    assert choose_tags(fake_game) == {"BLU": "ABL", "RED": "RED"}
    manifest = build_mod(fake_game, tmp_path / "out", SPEC)
    assert manifest["tags"]["BLU"] == "ABL"
    assert validate_mod(tmp_path / "out", fake_game)["ok"]  # no documentation in the fixture: names unchecked
    docs = fake_game / "documentation"
    docs.mkdir()
    (docs / "effects_documentation.md").write_text("## if\n\n## log\n\n## set_variable\n", encoding="utf-8")
    report = validate_mod(tmp_path / "out", fake_game)
    assert not report["ok"] and any("'load_oob'" in problem for problem in report["problems"])
    assert not any("'log'" in problem for problem in report["problems"])
    # A division that starts on the other side's land (the suspicion after the third live load) is a finding.
    oob = tmp_path / "out" / "mod" / "hoi4_arena" / "history" / "units" / "ARENA_00_RED_even.txt"
    oob.write_text(oob.read_text(encoding="utf-8").replace("location = 7", "location = 6"), encoding="utf-8")
    assert any("starts outside its own land at 6" in problem for problem in validate_mod(tmp_path / "out", None)["problems"])
    # Tampering with the mod is caught by the recorded hash.
    (tmp_path / "out" / "mod" / "hoi4_arena" / "descriptor.mod").write_text("changed")
    assert any("hash" in problem for problem in validate_mod(tmp_path / "out", None)["problems"])


def test_cli_commands(fake_game: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.ArgumentParser()
    handlers = add_commands(parser.add_subparsers(dest="command"))
    assert set(handlers) == {"mod-build", "mod-validate"}
    # mod-build always uses the default (Hungarian) region, which the synthetic install does not contain.
    args = parser.parse_args(["mod-build", "--game", str(fake_game), "--output", str(tmp_path / "cli")])
    with pytest.raises(ArenaError):
        handlers["mod-build"](args)
    build_mod(fake_game, tmp_path / "cli", SPEC)
    args = parser.parse_args(["mod-validate", "--game", str(fake_game), "--output", str(tmp_path / "cli")])
    assert handlers["mod-validate"](args) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


@pytest.mark.skipif(not (DEFAULT_GAME / "map" / "provinces.bmp").is_file(), reason="HOI4 is not installed here")
def test_default_region_on_real_install() -> None:
    region = extract_region(DEFAULT_GAME, DEFAULT_REGION)
    layout = region.layout()
    assert 40 <= len(layout.provinces) <= 60 and layout.source == "vanilla_region"
    assert len(region.side(Country.BLUE)) == len(region.side(Country.RED))
    assert not any(p.coastal for p in region.provinces)
    assert set(layout.sectors) == {"north", "center", "south"}
    for country in Country:
        assert {p.sector for p in region.front(country)} == {"north", "center", "south"}
        assert layout.capital(country).sector == "center"
    blue = sum(p.victory_points for p in layout.provinces if p.initial_controller == "BLU")
    assert blue == sum(p.victory_points for p in layout.provinces if p.initial_controller == "RED")
    assert any(p.river_neighbors for p in layout.provinces)
    assert choose_tags(DEFAULT_GAME) == {"BLU": "BLU", "RED": "RED"}
