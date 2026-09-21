import re

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
    """One order must take 100% of a side's victory points, or no capitulation is on offer.

    Spread across the half, the 28 of 35 points a surrender needs sit about five province
    hops behind the front, and reaching them means steering twenty divisions across twenty
    map rows. Massed on a single border province the enemy's own starting division is
    already adjacent to it, so the run is one click and its outcome is unambiguous: if
    taking every victory point does not capitulate, victory points are not what decides.
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
