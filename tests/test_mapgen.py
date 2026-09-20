import re

from PIL import Image

from hoi4_arena.mapgen import generate


def test_map_has_balanced_forces_and_nonempty_engine_placement_files(tmp_path):
    game = tmp_path / "fixture-game"
    (game / "map").mkdir(parents=True)
    for name in ["provinces", "terrain", "rivers", "trees", "cities"]:
        palette = Image.new("P", (8, 8))
        palette.putpalette([v for v in range(256) for _ in range(3)])
        palette.save(game / "map" / f"{name}.bmp")
    output = tmp_path / "arena"
    report = generate(game, output)
    assert report["land_provinces_per_country"] == 48
    assert not report["gameplay_verified"]
    for tag in ["BLU", "RED"]:
        units = (output / "history/units" / f"{tag}_1936.txt").read_text()
        assert len(re.findall(r"division =", units)) == 12
    weather = (output / "map/weatherpositions.txt").read_text()
    for region in [1, 2]:
        for kind in ["small", "big"]:
            assert re.search(rf"^{region};.*;{kind}$", weather, re.M)
    assert "supply_node" in (output / "map/buildings.txt").read_text()
