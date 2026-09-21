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


def test_map_has_balanced_forces_and_nonempty_engine_placement_files(tmp_path):
    game = tmp_path / "fixture-game"
    (game / "map").mkdir(parents=True)
    for name in ["provinces", "terrain", "rivers", "trees", "cities"]:
        palette = Image.new("P", (8, 8))
        palette.putpalette([v for v in range(256) for _ in range(3)])
        palette.save(game / "map" / f"{name}.bmp")
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
