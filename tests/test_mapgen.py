import re

import numpy as np
import pytest
from PIL import Image

from hoi4_arena.mapgen import (
    COLUMNS_PER_HALF,
    OCEAN_RINGS,
    ROWS,
    STATE_COLUMNS,
    STATE_ROWS,
    generate,
)


def _fixture_game(tmp_path):
    game = tmp_path / "fixture-game"
    if not game.exists():
        (game / "map").mkdir(parents=True)
        for name in ["provinces", "terrain", "rivers", "trees", "cities"]:
            palette = Image.new("P", (8, 8))
            palette.putpalette([v for v in range(256) for _ in range(3)])
            palette.save(game / "map" / f"{name}.bmp")
    return game


def _victory_points(output, tag):
    """Every province the given country holds a victory point in."""
    found = set()
    for path in sorted((output / "history/states").glob("*.txt")):
        text = path.read_text()
        if re.search(rf"owner = {tag}\b", text):
            found |= {int(p) for p in re.findall(r"victory_points = \{ (\d+)", text)}
    return found


def _garrison(output, tag):
    """Every province the given country starts a division in."""
    units = (output / "history/units" / f"{tag}_1936.txt").read_text()
    return {int(p) for p in re.findall(r"location = (\d+)", units)}


def test_map_has_balanced_forces_and_nonempty_engine_placement_files(tmp_path):
    game = _fixture_game(tmp_path)
    output = tmp_path / "arena"
    report = generate(game, output)
    # Derived from the grid rather than written out, so changing the province size is a
    # one-line edit instead of a hunt for the numbers it was spelled into.
    land_columns, land_rows = COLUMNS_PER_HALF - OCEAN_RINGS, ROWS - 2 * OCEAN_RINGS
    assert report["land_provinces_per_country"] == land_columns * land_rows
    assert report["states_per_country"] == STATE_COLUMNS * STATE_ROWS
    assert report["provinces"] == 2 * COLUMNS_PER_HALF * ROWS
    assert (report["land_columns"], report["land_rows"]) == (land_columns, land_rows)
    assert not report["gameplay_verified"]
    # The land grid has to divide into whole states, or a state straddles the map edge.
    assert land_columns % STATE_COLUMNS == 0 and land_rows % STATE_ROWS == 0
    for tag in ["BLU", "RED"]:
        units = (output / "history/units" / f"{tag}_1936.txt").read_text()
        # One division per row of the border column, so a side holds its own front.
        assert len(re.findall(r"division =", units)) == land_rows
    weather = (output / "map/weatherpositions.txt").read_text()
    for region in [1, 2]:
        for kind in ["small", "big"]:
            assert re.search(rf"^{region};.*;{kind}$", weather, re.M)
    assert "supply_node" in (output / "map/buildings.txt").read_text()


def test_an_undefended_country_fields_nothing_and_its_enemy_still_does(tmp_path):
    """The diagnostic that tells an idle AI apart from one whose attacks all fail.

    A balanced arena that never moves reads the same either way. Take one side's army
    away and the readings separate: an AI that is attacking walks into the empty front,
    and an AI that is not leaves it alone.
    """
    output = generate(_fixture_game(tmp_path), tmp_path / "arena", undefended="RED")
    root = tmp_path / "arena"
    assert output["undefended"] == "RED"
    assert not _garrison(root, "RED")
    assert _garrison(root, "BLU")
    # The template, the equipment and the generals stay: only the deployment is empty, so
    # nothing else about the side is being changed by accident.
    units = (root / "history/units/RED_1936.txt").read_text()
    assert "division_template" in units
    recruited = (root / "history/countries/RED - Arena.txt").read_text()
    assert "recruit_character = RED_marshal" in recruited
    assert "add_equipment_to_stockpile" in recruited


def test_the_harness_masses_the_whole_surrender_weight_on_one_border_province(tmp_path):
    """One province holds the whole victory-point weight, on the column the army starts on.

    That is the measurement that showed victory points do not decide a surrender. The
    weight stays massed so the same fixture can be compared with the spread arena.
    """
    harness = tmp_path / "harness"
    plain = tmp_path / "plain"
    generate(_fixture_game(tmp_path), harness, victory_points_on_border=True)
    generate(_fixture_game(tmp_path), plain)
    for tag in ["BLU", "RED"]:
        points = _victory_points(harness, tag)
        assert len(points) == 1, "the whole weight sits on one province"
        assert points <= _garrison(harness, tag), "and on the column the army starts on"
        # The playable arena keeps them spread and out of reach of a single crossing.
        assert len(_victory_points(plain, tag)) == 4
        assert not _victory_points(plain, tag) <= _garrison(plain, tag)


def test_the_harness_keeps_the_same_thirty_five_points_it_would_otherwise_spread(tmp_path):
    """Massing the weight must not change how much of it there is."""
    harness = tmp_path / "harness"
    plain = tmp_path / "plain"
    generate(_fixture_game(tmp_path), harness, victory_points_on_border=True)
    generate(_fixture_game(tmp_path), plain)

    def total(root, tag):
        found = 0
        for path in sorted((root / "history/states").glob("*.txt")):
            text = path.read_text()
            if re.search(rf"owner = {tag}\b", text):
                found += sum(int(v) for v in re.findall(r"victory_points = \{ \d+ (\d+)", text))
        return found

    for tag in ["BLU", "RED"]:
        assert total(harness, tag) == total(plain, tag) == 35


def test_the_harness_capital_moves_with_its_victory_point(tmp_path):
    """A capital left behind the front would keep the 20 points five hops away."""
    root = tmp_path / "harness"
    generate(_fixture_game(tmp_path), root, victory_points_on_border=True)
    for tag in ["BLU", "RED"]:
        capital = int(
            re.search(
                r"capital = (\d+)", (root / f"history/countries/{tag} - Arena.txt").read_text()
            ).group(1)
        )
        state = (root / f"history/states/{capital}-arena.txt").read_text()
        held = {int(p) for p in re.findall(r"victory_points = \{ (\d+)", state)}
        assert held & _garrison(root, tag), f"{tag} capital state holds no front-line point"


def test_a_small_country_keeps_every_province_arena_sized(tmp_path):
    """A small country is a land block in the full grid, not a smaller grid.

    Centering a smaller lattice on the bitmap left the margin to a few sea provinces up
    to 2245x769 px, and the engine crashed loading them. Every province, sea included,
    must stay near the playable arena's cell size.
    """
    from hoi4_arena.mapgen import MAP_SIZE, audit

    output = tmp_path / "island"
    report = generate(
        _fixture_game(tmp_path),
        output,
        undefended="RED",
        state_columns=2,
        state_rows=2,
        land_columns=6,
        land_rows=4,
    )
    assert report["states_per_country"] == 4
    assert report["land_provinces_per_country"] == 24
    assert report["provinces"] == 2 * COLUMNS_PER_HALF * ROWS
    assert (report["land_columns"], report["land_rows"]) == (6, 4)
    assert not _garrison(output, "RED")
    assert _garrison(output, "BLU")
    assert audit(output)["problems"] == []
    # Stock events and on_actions crashed the game on its first daily tick.
    descriptor = (output / "descriptor.mod").read_text()
    assert 'replace_path = "events"' in descriptor
    assert 'replace_path = "common/on_actions"' in descriptor
    # The arena reports its surrenders and weekly counts in game.log, for the worker's
    # game_log request, and every line it writes starts with the prefix that finds them.
    on_actions = (output / "common/on_actions/arena.txt").read_text()
    for hook in ("on_capitulation", "on_weekly", "on_state_control_changed"):
        assert hook in on_actions
    assert on_actions.count('log = "') == on_actions.count('log = "ARENA ')
    # A coin flip picks who declares the war, and the side and each human player are logged.
    for tag, enemy in (("BLU", "RED"), ("RED", "BLU")):
        assert f"{tag} = {{ declare_war_on = {{ target = {enemy} " in on_actions
        assert f'log = "ARENA declare {tag}"' in on_actions
    assert "random_list = { 50 = {" in on_actions
    assert on_actions.count("{") == on_actions.count("}")
    assert 'log = "ARENA player [THIS.GetTag]"' in on_actions
    # Since v4 the war starts from the recorder's own coin flip, through the console, and
    # the game's flip is a fallback after the first day that neither event has fired.
    startup = on_actions.split("on_startup", 1)[1].splitlines()[0]
    assert "declare_war_on" not in startup
    events = (output / "events/arena.txt").read_text()
    assert events.startswith("add_namespace = arena")
    for number, (tag, enemy) in enumerate((("BLU", "RED"), ("RED", "BLU")), 1):
        event = events.split(f"id = arena.{number} ", 1)[1].splitlines()[0]
        assert f"{tag} = {{ declare_war_on = {{ target = {enemy} " in event
        assert "set_global_flag = arena_declared" in event and "is_triggered_only = yes" in event
    assert events.count("{") == events.count("}")
    blue_daily = on_actions.split("on_daily_BLU", 1)[1].splitlines()[0]
    assert "NOT = { has_global_flag = arena_declared }" in blue_daily
    assert "random_list" not in on_actions.split("on_daily_RED", 1)[1].splitlines()[0]
    # Since v3 each side also reports its state every day, with its divisions in each state.
    for tag in ("BLU", "RED"):
        assert f"on_daily_{tag} = {{ effect = {{" in on_actions
    daily = on_actions.split("on_daily_BLU", 1)[1].splitlines()[0]
    assert "ARENA day [GetDateText] [ROOT.GetTag]" in daily
    for state in range(1, 2 * report["states_per_country"] + 1):
        assert f"arena_d{state} = num_armies_in_state@{state} }}" in daily
        assert f" {state}=[?arena_d{state}]" in daily
    bitmap = np.asarray(Image.open(output / "map/provinces.bmp"))
    packed = (
        bitmap[:, :, 0].astype(np.uint32) << 16
        | bitmap[:, :, 1].astype(np.uint32) << 8
        | bitmap[:, :, 2].astype(np.uint32)
    )
    _, counts = np.unique(packed, return_counts=True)
    cell = (MAP_SIZE[0] // (2 * COLUMNS_PER_HALF)) * (MAP_SIZE[1] // ROWS)
    assert len(counts) == report["provinces"]
    assert max(counts) < cell * 3


def test_an_undivided_or_undersized_state_grid_is_refused(tmp_path):
    game = _fixture_game(tmp_path)
    with pytest.raises(ValueError, match="whole states"):
        generate(
            game, tmp_path / "uneven", columns_per_half=8, rows=8, state_columns=5, state_rows=2
        )
    with pytest.raises(ValueError, match="three states"):
        generate(game, tmp_path / "one", columns_per_half=8, rows=8, state_columns=1, state_rows=1)
    with pytest.raises(ValueError, match="line up"):
        generate(
            game, tmp_path / "offset", state_columns=2, state_rows=1, land_columns=6, land_rows=3
        )
    with pytest.raises(ValueError, match="ocean rings"):
        generate(game, tmp_path / "wide", land_columns=COLUMNS_PER_HALF, land_rows=4)
    assert not (tmp_path / "uneven").exists()
    assert not (tmp_path / "one").exists()


# The named arenas. Each full generation takes seconds, so a few representative presets
# are generated once per module and shared by the tests that read them.


@pytest.fixture(scope="module")
def preset_arenas(tmp_path_factory):
    from hoi4_arena.mapgen import audit

    base = tmp_path_factory.mktemp("presets")
    game = _fixture_game(base)
    built = {}
    # marsh: lakes and a large river; salient: a bent border; bay: sea cut into the land.
    for name in ("marsh", "salient", "bay"):
        # The salient is drawn from another seed: the same design, another map.
        report = generate(game, base / name, preset=name, seed=11 if name == "salient" else None)
        built[name] = (base / name, report, audit(base / name))
    return built


def _definitions(root):
    rows = [r.split(";") for r in (root / "map/definition.csv").read_text().splitlines()]
    return {int(r[0]): r for r in rows[1:] if r[0]}


def _state_provinces(root, state):
    text = (root / f"history/states/{state}-arena.txt").read_text()
    return {int(p) for p in re.search(r"provinces = \{ ([\d ]+) \}", text).group(1).split()}


def test_every_preset_is_a_twelve_by_eight_arena_with_four_cities_a_side():
    from hoi4_arena.arenas import DESIGN_COLUMNS, DESIGN_ROWS, LAND_TYPES, PRESETS

    assert len(PRESETS) >= 5
    for name, design in PRESETS.items():
        assert (design.land_columns, design.land_rows) == (12, 8), name
        assert (design.state_columns, design.state_rows) == (4, 2), name
        # The capital's 20 points and three cities of 5: the 35 a side the plain arena has.
        assert len(design.cities) == 4, name
        for x, y in design.cities:
            assert 0 < x < DESIGN_COLUMNS / 2 and 0 < y < DESIGN_ROWS, name
        # The capital stands mid-country: the country picker opens centred on Blue's
        # capital, and the recorder picks Red by clicking Red's land on that screen.
        assert 5.5 <= design.cities[0][0] <= 7.5, name
        for patch in design.patches:
            assert patch.terrain in LAND_TYPES, name
        for number, river in enumerate(design.rivers):
            assert river.joins is None or river.joins < number, name
        # A trade names a cell on Red's side; the twin it gives back is Blue's.
        for x, _ in design.trades:
            assert x > DESIGN_COLUMNS / 2, name


def test_an_unknown_preset_is_refused_with_the_names(tmp_path):
    with pytest.raises(ValueError, match="plains"):
        generate(_fixture_game(tmp_path), tmp_path / "nowhere", preset="moon")
    assert not (tmp_path / "nowhere").exists()


def test_a_preset_arena_passes_the_audit_and_keeps_the_grid(preset_arenas):
    for name, (root, report, checked) in preset_arenas.items():
        assert checked["problems"] == [], name
        assert report["preset"] == name
        assert report["land_provinces_per_country"] <= 96
        assert report["states_per_country"] == 8
        assert report["provinces"] == 2 * COLUMNS_PER_HALF * ROWS
        # Sixteen states, Blue's 1 to 8 and Red's 9 to 16, named as the logs expect.
        names = (root / "localisation/english/arena_l_english.yml").read_text(encoding="utf-8-sig")
        for state in range(1, 17):
            side, number = ("West", state) if state <= 8 else ("East", state - 8)
            assert f' ARENA_STATE_{state}:0 "{side} {number}"' in names
            owner = "BLU" if state <= 8 else "RED"
            text = (root / f"history/states/{state}-arena.txt").read_text()
            assert f"owner = {owner} " in text
        for tag in ("BLU", "RED"):
            assert len(_victory_points(root, tag)) == 4


def test_a_preset_mirrors_every_province_to_its_twin(preset_arenas):
    """Fairness: province i and its twin have the same kind and terrain, each state's
    provinces turned round are its twin state's, and the victory points and starting
    divisions sit on twins."""
    for name, (root, report, _) in preset_arenas.items():
        rows = _definitions(root)
        half = report["provinces"] // 2

        def twin(province):
            return (province + half - 1) % (2 * half) + 1

        for i, row in rows.items():
            assert row[4:7] == rows[twin(i)][4:7], (name, i)
        for state in range(1, 9):
            turned = {twin(p) for p in _state_provinces(root, state)}
            assert turned == _state_provinces(root, state + 8), (name, state)
        assert {twin(p) for p in _victory_points(root, "BLU")} == _victory_points(root, "RED")
        assert {twin(p) for p in _garrison(root, "BLU")} == _garrison(root, "RED")


def test_a_capital_is_a_city_and_its_name_overrides_the_stock_one(preset_arenas):
    root, _, _ = preset_arenas["marsh"]
    rows = _definitions(root)
    for tag in ("BLU", "RED"):
        for province in _victory_points(root, tag):
            assert rows[province][6] == "urban"
    # VICTORY_POINTS_<id> keys exist for stock provinces too (564 is Kassel), and only
    # the replace folder is loaded after them.
    replaced = root / "localisation/english/replace/arena_victory_points_l_english.yml"
    labels = replaced.read_text(encoding="utf-8-sig")
    assert "West Capital" in labels and "East Capital" in labels
    main = (root / "localisation/english/arena_l_english.yml").read_text(encoding="utf-8-sig")
    assert "VICTORY_POINTS" not in main


def test_lakes_are_their_own_class_and_make_no_coast(preset_arenas):
    root, report, _ = preset_arenas["marsh"]
    rows = _definitions(root)
    lakes = [i for i, r in rows.items() if r[4] == "lake"]
    assert len(lakes) == 2 * report["design"]["lakes"] > 0
    for i in lakes:
        # As every stock lake: terrain lakes, never coastal, continent 0.
        assert rows[i][5:8] == ["false", "lakes", "0"]
    anchored = {line.split(";")[0] for line in (root / "map/unitstacks.txt").read_text().split()}
    assert not {str(i) for i in lakes} & anchored
    # In the land strategic region, as all 126 stock lakes are in land regions.
    land_region = (root / "map/strategicregions/1-arena.txt").read_text()
    listed = set(re.search(r"provinces = \{ ([\d ]+) \}", land_region).group(1).split())
    assert {str(i) for i in lakes} <= listed


def test_a_bent_border_trades_cells_one_for_one(preset_arenas):
    root, report, _ = preset_arenas["salient"]
    assert report["design"]["traded"] == 3
    assert report["land_provinces_per_country"] == 96
    half = report["provinces"] // 2
    blue = set().union(*(_state_provinces(root, state) for state in range(1, 9)))
    # Blue holds three provinces of the eastern half, and Red their twins in the west.
    assert len({p for p in blue if p > half}) == 3
    # Railways cross the border on a preset, so a captured hub can serve its captor.
    rails = [line.split() for line in (root / "map/railways.txt").read_text().splitlines()]
    assert any((int(a) in blue) != (int(b) in blue) for _, _, a, b in rails)


def test_a_bay_is_sea_inside_the_land_block(preset_arenas):
    root, report, _ = preset_arenas["bay"]
    rows = _definitions(root)
    land = [i for i, r in rows.items() if r[4] == "land"]
    assert len(land) == 2 * report["land_provinces_per_country"]
    assert report["land_provinces_per_country"] == 96 - report["design"]["bays"]
    # The bay's shore is a coast, so ports and naval placements follow it.
    assert report["coastal_land_provinces"] > 60


def test_rivers_run_on_province_borders_from_a_source(preset_arenas):
    from hoi4_arena.arenas import RIVER_JOIN, RIVER_SOURCE

    root, report, _ = preset_arenas["marsh"]
    rivers = np.asarray(Image.open(root / "map/rivers.bmp"))
    assert (rivers == RIVER_SOURCE).sum() == 2 * report["design"]["rivers"]
    assert (rivers == RIVER_JOIN).sum() == 0
    # The marsh's river is large: crossings over indices 7 to 11 cost 60% of an attack.
    assert ((rivers >= 7) & (rivers <= 11)).sum() > 0.5 * report["design"]["river_pixels"]


def test_the_river_audit_catches_what_the_engine_cannot_trace():
    from hoi4_arena.mapgen import _audit_rivers

    # Two provinces side by side: a river may run down the border between them.
    ids = np.ones((12, 12), dtype=np.int64)
    ids[:, 6:] = 2
    wet = np.zeros((12, 12), bool)

    def check(pixels):
        rivers = np.full((12, 12), 255, np.uint8)
        for (y, x), value in pixels.items():
            rivers[y, x] = value
        return " ".join(_audit_rivers(rivers, ids, wet))

    good = {(y, 5): 3 for y in range(2, 10)} | {(2, 5): 0}
    assert check(good) == ""
    assert "no source" in check({(y, 5): 3 for y in range(2, 10)})
    assert "inside a province" in check({(y, 2): 3 for y in range(2, 10)} | {(2, 2): 0})
    assert "corner" in check(good | {(10, 6): 3})
    assert "free end" in check(good | {(6, 5): 0})


def test_the_warp_turns_with_the_map_and_keeps_the_mirror():
    from hoi4_arena.arenas import mirror_ids, voronoi, warp_field

    rng = np.random.default_rng(0)
    shape = (256, 512)
    wx, wy = warp_field(rng, shape, 20.0)
    # w(half turn of p) = -w(p): the displacement turns with the map.
    assert np.allclose(wx[::-1, ::-1], -wx, atol=1e-4)
    assert np.allclose(wy[::-1, ::-1], -wy, atol=1e-4)
    left = np.array([(x, y) for x in range(16, 256, 32) for y in range(16, 256, 32)], float)
    left += rng.uniform(-6, 6, left.shape)
    points = np.concatenate([left, [511, 255] - left])
    ids = voronoi(points, shape, (wx, wy), len(left))
    assert (mirror_ids(ids[::-1, ::-1], len(left)) == ids).all()


def test_a_preview_draws_the_arena(preset_arenas, tmp_path):
    from hoi4_arena.mapgen import preview

    root, _, _ = preset_arenas["bay"]
    result = preview(root, tmp_path / "bay.png", width=600)
    picture = Image.open(tmp_path / "bay.png")
    assert picture.size == tuple(result["size"]) and picture.width == 600


def test_the_layout_names_a_bulge_for_the_scripted_player(preset_arenas):
    """The scripted player names the state under a point of the land box it reads off the
    screen. On the grid a point just east of the middle is Red's; on the salient arena the
    north of it is Blue's bulge, and generation.json's layout says so."""
    from hoi4_arena.scripted import arena_layout, state_at

    root, _, _ = preset_arenas["salient"]
    layout = arena_layout(root)
    assert layout.shape == (32, 96)
    assert set(np.unique(layout)) <= set(range(17))
    assert state_at(0.535, 0.25) >= 9
    assert 1 <= state_at(0.535, 0.25, layout) <= 8
    assert 9 <= state_at(0.465, 0.75, layout) <= 16
    # Every state of both sides shows somewhere in the layout.
    assert set(range(1, 17)) <= set(np.unique(layout))


def _rail_links(root):
    links = set()
    for line in (root / "map/railways.txt").read_text().splitlines():
        cells = [int(c) for c in line.split()][2:]
        links |= {tuple(sorted(pair)) for pair in zip(cells, cells[1:])}
    return links


def test_a_preset_railway_is_a_mirrored_trunk_that_reaches_every_hub(preset_arenas):
    """A trunk network, not a line on every adjacency: taking a junction cuts off the
    hubs beyond it. Every hub still reaches its capital, two lines cross the border,
    and Red's network is Blue's turned round."""
    for name, (root, report, checked) in preset_arenas.items():
        assert checked["problems"] == [], name
        links = _rail_links(root)
        rows = _definitions(root)
        land = {i for i, r in rows.items() if r[4] == "land"}
        half = report["provinces"] // 2

        def twin(province):
            return (province + half - 1) % (2 * half) + 1

        assert {tuple(sorted((twin(a), twin(b)))) for a, b in links} == links, name
        assert all(a in land and b in land for a, b in links), name
        blue = set().union(*(_state_provinces(root, s) for s in range(1, 9)))
        across = [(a, b) for a, b in links if (a in blue) != (b in blue)]
        assert len(across) >= 2, name
        # Far fewer than the plain arena's line on every adjacency (513 on these maps).
        assert len(links) < 150, (name, len(links))


def test_the_audit_catches_a_hub_cut_off_from_its_capital(preset_arenas):
    from hoi4_arena.mapgen import audit

    root, _, _ = preset_arenas["bay"]
    path = root / "map/railways.txt"
    original = path.read_text()
    hubs = [int(c) for c in (root / "map/supply_nodes.txt").read_text().split()[1::2]]
    # Lift every line into one hub: it is still a hub, but nothing joins it any more.
    lonely = hubs[0]
    kept = [line for line in original.splitlines() if str(lonely) not in line.split()[2:]]
    path.write_text("\n".join(kept) + "\n")
    try:
        problems = audit(root)["problems"]
    finally:
        path.write_text(original)
    assert any("no railway joins" in p for p in problems), problems


def test_ground_colours_stay_bright_and_neutral_enough_to_read_as_land():
    """The scripted player tells land by tint at full zoom-out, and only above a
    brightness sum of 250 on screen. Forest and marsh at a colour-map sum near 180 drew
    at about 220 and read as holes; the screen sum came out at about 1.8 times the colour
    map's minus 84. A red cast would also blunt Blue's tint (blue minus red above 10)."""
    from hoi4_arena.arenas import GROUND

    for index, (r, g, b) in GROUND.items():
        assert 1.8 * (r + g + b) - 84 >= 300, index
        # No warmer than the plain arena's grass (red minus blue 16), which it was
        # calibrated on.
        assert r - b <= 18, index


def test_a_seed_redraws_a_preset_and_the_report_measures_its_front(preset_arenas):
    """The report says what attacking across each arena's border costs: the bay leaves
    a narrow isthmus, the marsh puts much of its front at -40% or worse, and the bent
    border of the salient is longer than a straight one."""
    from hoi4_arena.arenas import PRESETS

    fronts = {name: report["design"]["front"] for name, (_, report, _) in preset_arenas.items()}
    assert preset_arenas["salient"][1]["design"]["seed"] == 11 != PRESETS["salient"].seed
    assert preset_arenas["marsh"][1]["design"]["seed"] == PRESETS["marsh"].seed
    assert fronts["bay"]["pairs"] < fronts["marsh"]["pairs"] < fronts["salient"]["pairs"]
    assert fronts["marsh"]["share_at_40_or_worse"] >= 0.25
    assert fronts["bay"]["mean_attack"] > fronts["marsh"]["mean_attack"]
