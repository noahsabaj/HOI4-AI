"""Arenas for three and four nations: the turn, equal shares, and the wars."""

import json
import re
import shutil
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from hoi4_arena import multination
from hoi4_arena.arenas import LAND_TYPES
from hoi4_arena.mapgen import _province_ids, adjacency, audit
from hoi4_arena.mapgen_multi import draw_provinces, generate_multi
from hoi4_arena.multination import NATION_PRESETS, NATIONS, SETUPS, wars


def _fixture_game(tmp_path):
    game = tmp_path / "fixture-game"
    if not game.exists():
        (game / "map").mkdir(parents=True)
        for name in ["provinces", "terrain", "rivers", "trees", "cities"]:
            palette = Image.new("P", (8, 8))
            palette.putpalette([v for v in range(256) for _ in range(3)])
            palette.save(game / "map" / f"{name}.bmp")
    return game


# Each generation takes about 40 s, so one arena of each kind is made once and shared: four
# nations in two factions, and three with a lone nation given the bonus.
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("nations")
    game = _fixture_game(base)
    made = {}
    for name, setup, bonus in (("quad-ridges", "2v2", 0.0), ("tri-plains", "2v1", 0.25)):
        report = generate_multi(game, base / name, preset=name, wars=setup, lone_bonus=bonus)
        made[name] = (base / name, report, audit(base / name))
    return made


def _definitions(root):
    rows = [r.split(";") for r in (root / "map/definition.csv").read_text().splitlines()]
    return {int(r[0]): r for r in rows[1:] if r[0]}


def _states(root):
    found = {}
    for path in sorted((root / "history/states").glob("*.txt")):
        text = path.read_text()
        state = int(re.search(r"id = (\d+)", text).group(1))
        owner = re.search(r"owner = (\w+)", text).group(1)
        provinces = [
            int(p) for p in re.search(r"provinces = \{ ([\d ]+) \}", text).group(1).split()
        ]
        points = {
            int(p): int(v) for p, v in re.findall(r"victory_points = \{ (\d+) (\d+) \}", text)
        }
        found[state] = (owner, provinces, points)
    return found


def _turn(report):
    n, orbits = report["nations"], report["symmetry"]["orbits"]

    def sigma(i, k=1):
        for _ in range(k % n):
            if i <= orbits * n:
                i = (i - 1) // n * n + ((i - 1) % n + 1) % n + 1
        return i

    return sigma


def test_every_design_names_its_nations_terrain_and_a_default_war():
    assert {design.nations for design in NATION_PRESETS.values()} == {3, 4}
    for name, design in NATION_PRESETS.items():
        for patch in design.patches:
            assert patch.terrain in LAND_TYPES, name
        plan = wars(design.wars, NATIONS[: design.nations])
        assert plan.fair, name
        assert design.states * 12 >= 90, "states of about a dozen provinces, as on 12x8"


def test_the_named_setups_split_the_nations_as_named_and_say_which_are_fair():
    fair = {
        "1v1v1": True,
        "2v1": False,
        "1v1+1": False,
        "1v1v1v1": True,
        "2v2": True,
        "2v2x": True,
        "3v1": False,
        "2v1v1": False,
        "1v1v1+1": False,
        "2v1+1": False,
        "1v1+2": False,
    }
    for n, names in SETUPS.items():
        tags = NATIONS[:n]
        for name in names:
            plan = wars(name, tags)
            sizes = re.match(r"[\dv]+", name).group(0).split("v")
            assert [len(side) for side in plan.sides] == [int(s) for s in sizes], name
            assert len(plan.neutral) == n - sum(int(s) for s in sizes), name
            assert plan.fair == fair[name], name
            assert ("unfair by design" in plan.note()) == (len(set(sizes)) > 1), name
    # Neighbours stand together; with x, allies stand apart.
    assert wars("2v2", NATIONS).sides == (("BLU", "RED"), ("PUR", "GRN"))
    assert wars("2v2x", NATIONS).sides == (("BLU", "PUR"), ("RED", "GRN"))
    assert wars("3v1", NATIONS).underdogs() == ("GRN",)
    assert wars("1v1+1", NATIONS[:3]).neutral == ("PUR",)
    explicit = wars("GRN+RED:BLU", NATIONS)
    assert explicit.sides == (("GRN", "RED"), ("BLU",)) and explicit.neutral == ("PUR",)
    for bad in ("2v2", "1v1v1v1", "BLU+BLU:RED", "BLU:ZZZ", "BLU", "0v3", "2v1x"):
        with pytest.raises(ValueError):
            wars(bad, NATIONS[:3])


def test_each_war_is_declared_once_by_the_side_with_the_first_say():
    plan = wars("1v1v1v1", NATIONS)
    effect = multination.setup_effect(plan, 0)
    declared = re.findall(r"(\w{3}) = \{ declare_war_on = \{ target = (\w{3})", effect)
    assert len(declared) == 6 and len({frozenset(pair) for pair in declared}) == 6
    # Blue, first, declares all three of its wars; Green, last, none.
    assert Counter(d for d, _ in declared) == {"BLU": 3, "RED": 2, "PUR": 1}
    later = re.findall(r"(\w{3}) = \{ declare_war_on", multination.setup_effect(plan, 3))
    assert Counter(later)["GRN"] == 3
    # A faction's leader declares on the other's, and its ally is added to the war.
    plan = wars("2v2", NATIONS)
    effect = multination.setup_effect(plan, 2)
    assert re.findall(r"(\w{3}) = \{ declare_war_on = \{ target = (\w{3})", effect) == [
        ("PUR", "BLU")
    ]
    assert "GRN = { if = { limit = { NOT = { has_war_with = BLU } } add_to_war = {" in effect
    assert "targeted_alliance = PUR enemy = BLU" in effect
    assert "RED = { if = { limit = { NOT = { has_war_with = PUR } } add_to_war = {" in effect
    # Neutrals are never at war.
    effect = multination.setup_effect(wars("1v1+2", NATIONS), 0)
    assert "PUR" not in effect and "GRN" not in effect
    assert effect.count("{") == effect.count("}")


def test_the_arenas_pass_the_audit(built):
    for name, (root, report, checked) in built.items():
        assert checked["problems"] == [], name
        assert report["preset"] == name
        assert report["countries"] == report["nations"] == NATION_PRESETS[name].nations
        assert checked["states"] == report["nations"] * report["states_per_country"]


def test_every_nation_holds_the_same_share(built):
    for name, (root, report, _) in built.items():
        rows = _definitions(root)
        states = _states(root)
        tags = report["tags"]
        shares = {}
        for tag in tags:
            mine = [s for s in states.values() if s[0] == tag]
            provinces = [p for _, listed, _ in mine for p in listed]
            shares[tag] = (
                len(mine),
                len(provinces),
                Counter(rows[p][6] for p in provinces),
                sorted(v for _, _, points in mine for v in points.values()),
                sorted(len(listed) for _, listed, _ in mine),
                sum(1 for p in provinces if rows[p][5] == "true"),
            )
        assert all(share == shares[tags[0]] for share in shares.values()), name
        assert shares[tags[0]][3] == [5, 5, 5, 20], "the capital's 20 and three cities of 5"
        assert shares[tags[0]][1] == report["land_provinces_per_country"]
        units = {
            tag: len(
                re.findall(r"division =", (root / f"history/units/{tag}_1936.txt").read_text())
            )
            for tag in tags
        }
        assert len(set(units.values())) == 1 and units[tags[0]] >= 8, name


def test_four_nations_are_an_exact_quarter_turn(built):
    root, report, _ = built["quad-ridges"]
    sigma = _turn(report)
    ids, _ = _province_ids(root)
    x0, _, side = report["symmetry"]["square"]
    lookup = np.array([sigma(i) for i in range(report["provinces"] + 1)])
    square = ids[:, x0 : x0 + side]
    assert (lookup[np.rot90(square)] == square).all()
    for name in ("terrain.bmp", "heightmap.bmp", "trees.bmp", "cities.bmp"):
        picture = np.array(Image.open(root / "map" / name))
        start = (picture.shape[1] - picture.shape[0]) // 2
        cut = picture[:, start : start + picture.shape[0]]
        assert (np.rot90(cut) == cut).all(), name
    # The turn fixes the centre, a lake no nation stands on.
    rows = _definitions(root)
    assert [rows[i][4] for i in report["symmetry"]["fixed"]] == ["lake"]


def test_three_nations_keep_their_province_graph_under_a_third_turn(built):
    root, report, _ = built["tri-plains"]
    sigma = _turn(report)
    rows = _definitions(root)
    ids, _ = _province_ids(root)
    neighbours = adjacency(ids, report["provinces"])
    land = [i for i, r in rows.items() if r[4] == "land"]
    assert land and all(i <= report["symmetry"]["orbits"] * 3 for i in land)
    for i in land:
        assert rows[sigma(i)][4:7] == rows[i][4:7]
        assert {sigma(j) for j in neighbours[i]} == neighbours[sigma(i)], i
    # Every nation touches both others, and the three meet at the centre.
    holder = {p: owner for owner, listed, _ in _states(root).values() for p in listed}
    for tag in report["tags"]:
        touched = {holder.get(j) for p, t in holder.items() if t == tag for j in neighbours[p]}
        assert set(report["tags"]) - {tag} <= touched


def test_factions_wars_and_the_lone_bonus_are_written_as_configured(built):
    root, report, _ = built["quad-ridges"]
    history = {t: (root / f"history/countries/{t} - Arena.txt").read_text() for t in NATIONS}
    assert "create_faction_from_template = { template = arena_faction" in history["BLU"]
    assert "add_to_faction = RED" in history["BLU"]
    assert "add_to_faction = GRN" in history["PUR"]
    assert "create_faction" not in history["RED"] + history["GRN"]
    assert "arena_faction = {" in (root / "common/factions/templates/arena.txt").read_text()
    events = (root / "events/arena.txt").read_text()
    assert [int(k) for k in re.findall(r"id = arena\.(\d+) ", events)] == [1, 2, 3, 4]
    on_actions = (root / "common/on_actions/arena.txt").read_text()
    assert 'log = "ARENA wars BLU+RED:PUR+GRN fair yes"' in on_actions
    daily = {t: re.search(rf"on_daily_{t} = .*", on_actions).group(0) for t in NATIONS}
    assert "random_list" in daily["BLU"] and daily["BLU"].count("declare_war_on") == 4
    assert all("random_list" not in daily[t] for t in NATIONS[1:])
    strategy = (root / "common/ai_strategy/arena.txt").read_text()
    assert strategy.count("type = front_control") == 12
    assert report["wars"]["fair"] and report["wars"]["underdogs"] == []
    assert "arena_underdog" not in (root / "common/ideas/arena.txt").read_text()

    root, report, _ = built["tri-plains"]
    assert report["wars"] == {**report["wars"], "setup": "2v1", "fair": False,
                              "underdogs": ["PUR"], "lone_bonus": 0.25}  # fmt: skip
    assert "unfair by design" in report["wars"]["note"]
    ideas = (root / "common/ideas/arena.txt").read_text()
    assert "arena_underdog" in ideas and "army_attack_factor = 0.25" in ideas
    for tag in ("BLU", "RED", "PUR"):
        text = (root / f"history/countries/{tag} - Arena.txt").read_text()
        assert ("add_ideas = arena_underdog" in text) == (tag == "PUR")
    on_actions = (root / "common/on_actions/arena.txt").read_text()
    assert 'log = "ARENA wars BLU+RED:PUR fair no"' in on_actions


def test_the_state_channel_splits_held_past_sixteen_states(built):
    root, report, _ = built["quad-ridges"]
    on_actions = (root / "common/on_actions/arena.txt").read_text()
    assert "held [?arena_held] [?arena_held2] mask" in on_actions
    assert "arena_held2 = 32768" in on_actions and "arena_held = 65536" not in on_actions


def test_a_preview_draws_every_nation(built, tmp_path):
    from hoi4_arena.mapgen import preview

    root, _, _ = built["tri-plains"]
    result = preview(root, tmp_path / "tri.png")
    assert (tmp_path / "tri.png").exists() and result["size"][0] == 1800


def test_the_audit_catches_a_nation_that_is_not_the_others_turned(built, tmp_path):
    root, report, _ = built["quad-ridges"]
    broken = tmp_path / "broken"
    shutil.copytree(root, broken)
    rows = (broken / "map/definition.csv").read_text().splitlines()
    # A plains province of nation 0 turned to forest.
    first = next(
        i for i, r in enumerate(rows) if r.split(";")[4] == "land" and r.split(";")[6] == "plains"
    )
    cells = rows[first].split(";")
    cells[6] = "forest"
    rows[first] = ";".join(cells)
    (broken / "map/definition.csv").write_text("\n".join(rows) + "\n", newline="\r\n")
    # And the railways across one border taken up.
    holder = {p: o for o, listed, _ in _states(broken).values() for p in listed}
    rails = (broken / "map/railways.txt").read_text().splitlines()
    kept = [r for r in rails if {holder.get(int(p)) for p in r.split()[2:]} != {"BLU", "RED"}]
    (broken / "map/railways.txt").write_text("\n".join(kept) + "\n")
    problems = audit(broken)["problems"]
    assert any("differ from their turn" in p for p in problems), problems
    assert any("railways cross between BLU and RED" in p for p in problems), problems
    assert any("railways are not the same turned" in p for p in problems), problems


def test_a_seed_draws_the_same_provinces_every_time_and_another_seed_others():
    design = NATION_PRESETS["tri-ridges"]
    first, again = draw_provinces(design, 5), draw_provinces(design, 5)
    assert (first.ids == again.ids).all() and first.total == again.total
    other = draw_provinces(design, 6)
    assert (other.ids != first.ids).any()
    # The same design all the same: the same land share, turned.
    assert (first.kinds == 1).sum() == (other.kinds == 1).sum()


def test_the_cli_refuses_two_country_options_on_a_multi_nation_preset(tmp_path):
    from hoi4_arena import cli

    with pytest.raises(ValueError, match="two-country"):
        cli._dispatch(
            "generate-map",
            {
                "game": str(_fixture_game(tmp_path)),
                "output": str(tmp_path / "x"),
                "preset": "quad-plains",
                "seed": None,
                "wars": None,
                "lone_bonus": 0.0,
                "undefended": "BLU",
                "victory_points_on_border": False,
                "columns_per_half": None,
                "rows": None,
                "state_columns": None,
                "state_rows": None,
                "land_columns": None,
                "land_rows": None,
            },
        )
    with pytest.raises(ValueError, match="multi-nation"):
        cli._dispatch(
            "generate-map",
            {
                "game": str(_fixture_game(tmp_path)),
                "output": str(tmp_path / "y"),
                "preset": "plains",
                "seed": None,
                "wars": "2v2",
                "lone_bonus": 0.0,
                "undefended": None,
                "victory_points_on_border": False,
                "columns_per_half": None,
                "rows": None,
                "state_columns": None,
                "state_rows": None,
                "land_columns": None,
                "land_rows": None,
            },
        )
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


def test_the_arena_tags_come_from_generation_json(tmp_path):
    assert multination.arena_tags(tmp_path / "missing") == ("BLU", "RED")
    (tmp_path / "generation.json").write_text(json.dumps({"tags": ["BLU", "RED", "PUR"]}))
    assert multination.arena_tags(tmp_path) == ("BLU", "RED", "PUR")
