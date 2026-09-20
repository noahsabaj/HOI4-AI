"""From-scratch arena map tests. Everything is built at 512x256 WITHOUT the game install (built-in
palette entries); only ``test_small_build_against_real_install`` reads the install, and nothing launches it.
A passing validator here means "our files obey the documented rules", never "the game loads this"."""
from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from hoi4_agent.arena.contracts import ArenaError, Country
from hoi4_agent.arena.fingerprint import tree_hash
from hoi4_agent.arena.layout import ArenaLayout
from hoi4_agent.arena.mod import bmpio
from hoi4_agent.arena.mod.custommap import (
    CLOSING_LINE,
    REPLACE_PATHS,
    add_commands,
    build_custom_map,
    check_bitmaps,
    check_provinces,
    check_routes,
    province_raster,
    validate_custom_map,
)
from hoi4_agent.arena.mod.mapdesign import PRESETS, MapDesign, components, line_pixels, rasterize, rect, x_crossings
from hoi4_agent.arena.mod.region import DEFAULT_GAME

SMALL = (512, 256, 16)


def _small(name: str) -> MapDesign:
    return PRESETS[name].sized(*SMALL)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("custommap") / "three_lanes"
    build_custom_map(_small("three_lanes"), output, game=None)
    return output


def _copy(built: Path, tmp_path: Path) -> tuple[Path, Path]:
    output = tmp_path / "copy"
    shutil.copytree(built, output)
    return output, output / "mod" / "hoi4_arena_three_lanes" / "map"


def _problems(output: Path) -> str:
    return "\n".join(validate_custom_map(output)["problems"])


def test_build_validates_and_is_deterministic(built: Path, tmp_path: Path) -> None:
    report = validate_custom_map(built)
    assert report["ok"], report["problems"]
    assert report["evidence_kind"] == "static_validation_not_a_game_load" and report["files_parsed"] > 100
    again = tmp_path / "again"
    manifest = build_custom_map(_small("three_lanes"), again, game=None, previews=False)
    first = json.loads((built / "arena_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mod_sha256"] == first["mod_sha256"] == tree_hash(built / first["mod_dir"])
    for name in ("arena_layout.json", "arena_map.json", "arena_manifest.json", "FIRST_LAUNCH.md"):
        assert (again / name).read_bytes() == (built / name).read_bytes()
    assert first["evidence_kind"] == "generated_static_never_loaded_by_the_game"
    assert first["dataset_role"]["suggestion"] == "train_validation" and first["expected_routes"] == 3
    assert {s["split"] for s in first["scenarios"]} == {"train", "validation", "held_out"}
    assert all(s["scenario_id"].startswith("custom_three_lanes_v1.") for s in first["scenarios"])
    with pytest.raises(ArenaError):
        build_custom_map(_small("three_lanes"), again, game=None)  # no silent overwrite


def test_bitmap_headers_and_text_conventions(built: Path) -> None:
    folder = built / "mod" / "hoi4_arena_three_lanes" / "map"
    provinces = bmpio.read_bmp(folder / "provinces.bmp")
    assert (provinces.bits, provinces.header_size, provinces.compression) == (24, 40, 0)
    assert (provinces.width, provinces.height) == SMALL[:2]
    for name in ("heightmap", "terrain", "rivers", "trees", "cities"):
        image = bmpio.read_bmp(folder / f"{name}.bmp")
        assert (image.bits, image.header_size, len(image.palette)) == (8, 40, 256)
    assert bmpio.read_bmp(folder / "trees.bmp").width == 512 * 75 // 256
    assert set(np.unique(bmpio.read_bmp(folder / "heightmap.bmp").pixels).tolist()) == {85, 100}
    raw = (folder / "definition.csv").read_bytes()
    assert raw.startswith(b"0;0;0;0;land;false;unknown;0\r\n") and raw.count(b"\n") == raw.count(b"\r\n")
    lines = (folder / "adjacencies.csv").read_text(encoding="utf-8").splitlines()
    assert lines[-1] == CLOSING_LINE and any(";impassable;" in line for line in lines)
    assert (folder / "buildings.txt").stat().st_size > 0 and (folder / "unitstacks.txt").stat().st_size > 0
    head = bmpio.read_dds_header(folder / "terrain" / "colormap_water_0.dds")
    assert (head["width"], head["height"], head["fourcc"]) == (256, 128, "DXT5")
    descriptor = (folder.parent / "descriptor.mod").read_text(encoding="utf-8")
    assert all(f'replace_path="{path}"' in descriptor for path in REPLACE_PATHS)
    assert check_bitmaps(folder) == []


def test_layout_matches_the_game_files(built: Path) -> None:
    layout = ArenaLayout.load(built / "arena_layout.json")
    meta = json.loads((built / "arena_map.json").read_text(encoding="utf-8"))
    ids, problems = province_raster(built / "mod" / "hoi4_arena_three_lanes" / "map")
    assert problems == [] and layout.source == "custom_map"
    assert sorted(np.unique(ids).tolist()) == list(range(1, len(meta["provinces"]) + 1))  # sequential, no gaps
    kinds = {row["id"]: row["kind"] for row in meta["provinces"]}
    assert {p.id for p in layout.provinces} == {i for i, kind in kinds.items() if kind == "play"}
    partner = {row["id"]: row["partner"] for row in meta["provinces"]}
    by_id = {p.id: p for p in layout.provinces}
    for p in layout.provinces:  # exact mirror symmetry of the gameplay graph
        other = by_id[partner[p.id]]
        assert sorted(partner[n] for n in p.neighbors) == sorted(other.neighbors)
        assert (p.sector, p.terrain, p.victory_points) == (other.sector, other.terrain, other.victory_points)
        assert abs(p.x + other.x - 1) < 1e-5 and abs(p.y - other.y) < 1e-5 and p.initial_controller != other.initial_controller
        cx, cy = next((row["cx"], row["cy"]) for row in meta["provinces"] if row["id"] == p.id)
        assert ids[cy, cx] == p.id
    mirrored = np.array([0] + [partner[i] for i in range(1, len(partner) + 1)])[ids[:, ::-1]]
    differ = {(int(x), int(y)) for y, x in np.argwhere(mirrored != ids).tolist()}
    allowed = {(x, y) for x, y in meta["repairs"]} | {(511 - x, y) for x, y in meta["repairs"]}
    assert differ <= allowed and len(meta["repairs"]) > 0  # only the declared centre-line repair pixels differ
    assert layout.capital(Country.BLUE).initial_controller == "BLU" and set(layout.sectors) == {"north", "center", "south"}
    assert meta["river_crossings"]["links"] and all(set(p.river_neighbors) <= set(p.neighbors) for p in layout.provinces)


def test_three_routes_property(built: Path) -> None:
    layout = ArenaLayout.load(built / "arena_layout.json")
    assert check_routes(layout, 3) == []
    assert any("expected 1 separate routes, found 3" in problem for problem in check_routes(layout, 1))
    arena = rasterize(_small("three_lanes"))
    assert len(arena.routes()) == 3
    assert {arena.province(route[0]).sector for route in arena.routes()} == {"north", "center", "south"}


@pytest.mark.parametrize("name", [n for n in PRESETS if n != "three_lanes"])
def test_every_preset_builds_and_validates(name: str, tmp_path: Path) -> None:
    manifest = build_custom_map(_small(name), tmp_path / name, game=None, previews=False)
    report = validate_custom_map(tmp_path / name)
    assert report["ok"], report["problems"]
    assert manifest["symmetry"] == PRESETS[name].symmetry and manifest["dataset_role"]["suggestion"] in (
        "train_validation", "held_out_map")
    layout = ArenaLayout.load(tmp_path / name / "arena_layout.json")
    assert layout.scenario_id == f"custom_{name}_v1" and len(layout.provinces) == 32


def test_options_lake_infill_and_player_box(tmp_path: Path) -> None:
    from dataclasses import replace
    design = replace(_small("three_lanes"), blocked_kind="lake", player_box=True)
    manifest = build_custom_map(design, tmp_path / "lake", game=None, previews=False)
    assert validate_custom_map(tmp_path / "lake")["ok"]
    assert manifest["provinces"]["box"] == 2 and manifest["blocked_infill"] == "lake"
    definition = (tmp_path / "lake" / manifest["mod_dir"] / "map" / "definition.csv").read_text(encoding="ascii")
    assert ";lake;false;lakes;1" in definition
    states = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "lake" / manifest["mod_dir"] / "history" / "states").glob("*.txt"))
    assert "impassable" not in states


def test_previews_are_written(built: Path) -> None:
    from PIL import Image

    from hoi4_agent.arena.mod.preview import contact_sheet
    names = ["preview_political.png", "preview_terrain_rivers.png", "preview_supply.png", "preview_provinces_raw.png",
             "preview_start_positions_inf12_line.png", "preview_start_positions_mix12_mass_south.png"]
    for name in names:
        with Image.open(built / name) as image:
            assert image.width == 1600
    sheet = contact_sheet(built.parent)
    assert sheet is not None and sheet.name == "preview_all_designs.png"


def test_raster_helpers() -> None:
    lab = np.array([[1, 1, 2], [3, 1, 2], [3, 3, 1]])
    comp, labels, sizes = components(lab)
    assert sorted(zip(labels, sizes, strict=True)) == [(1, 1), (1, 3), (2, 2), (3, 3)] and comp.shape == lab.shape
    assert len(x_crossings(np.array([[1, 2], [3, 4]]))) == 1 and len(x_crossings(np.array([[1, 2], [1, 3]]))) == 0
    path = line_pixels(0, 0, 5, 3)
    assert path[0] == (0, 0) and path[-1] == (5, 3) and len(path) == 9  # 4-connected: |dx| + |dy| + 1
    with pytest.raises(ArenaError):
        MapDesign("bad", "", 1, (rect(0.3, 0.3, 0.5, 0.7),), (0.35, 0.5), (), width=500)
    with pytest.raises(ArenaError):  # land that never reaches the centre line has no front
        rasterize(MapDesign("island", "", 1, (rect(0.2, 0.3, 0.4, 0.7),), (0.3, 0.5), ()).sized(*SMALL))


# --- negative tests: every validator must catch a deliberately broken file -------------------------------

def test_catches_x_crossing_box_and_tiny_province() -> None:
    ids = np.ones((256, 512), dtype=np.int32)
    ids[100:, :200], ids[:100, 200:], ids[100:, 200:] = 2, 3, 4
    ids[0, 0:3] = 5
    text = "\n".join(check_provinces(ids))
    assert "2 X crossings" in text and "x=199 y=99" in text  # the inner one and the one on the wrap seam
    assert "smaller than 8 pixels: [5]" in text and "too wide" in text and "too tall" in text


def test_catches_x_crossing_in_the_file(built: Path, tmp_path: Path) -> None:
    output, folder = _copy(built, tmp_path)
    image = bmpio.read_bmp(folder / "provinces.bmp")
    ids, _ = province_raster(folder)
    pixels = image.pixels.copy()
    y, x = next((y, x) for y, x in np.argwhere((ids[:-1, :-1] != ids[:-1, 1:]) & (ids[:-1, :-1] != ids[1:, :-1])
                                               & (ids[1:, :-1] == ids[1:, 1:])).tolist() if ids[y, x + 1] != ids[y + 1, x])
    others = [tuple(pixels[r, c]) for r in range(256) for c in (0,) if ids[r, c] not in ids[y:y + 2, x:x + 2]]
    pixels[y + 1, x + 1] = others[0]
    bmpio.write_bmp24(folder / "provinces.bmp", pixels, ppm=3780)
    assert "X crossings" in _problems(output)


def test_catches_gap_in_ids_and_lf_endings(built: Path, tmp_path: Path) -> None:
    output, folder = _copy(built, tmp_path)
    original = (folder / "definition.csv").read_bytes()
    lines = original.split(b"\r\n")
    (folder / "definition.csv").write_bytes(b"\r\n".join(lines[:5] + lines[6:]))
    assert "sequential from 0 without gaps" in _problems(output)
    (folder / "definition.csv").write_bytes(original.replace(b"\r\n", b"\n"))
    assert "CRLF" in _problems(output)


def test_catches_32_bit_and_wrong_palette_bitmaps(built: Path, tmp_path: Path) -> None:
    output, folder = _copy(built, tmp_path)
    image = bmpio.read_bmp(folder / "provinces.bmp")
    bgra = np.dstack([image.pixels[::-1, :, ::-1], np.full(image.pixels.shape[:2], 255, dtype=np.uint8)])
    header = b"BM" + struct.pack("<IHHI", 54 + bgra.size, 0, 0, 54) + struct.pack(
        "<IiiHHIIiiII", 40, image.width, image.height, 1, 32, 0, bgra.size, 2834, 2834, 0, 0)
    (folder / "provinces.bmp").write_bytes(header + bgra.tobytes())
    assert "must be 24-bit" in _problems(output)
    rivers = bmpio.read_bmp(folder / "rivers.bmp")
    bmpio.write_bmp8(folder / "rivers.bmp", rivers.pixels, bmpio.grey_palette())
    assert "rivers.bmp palette differs from vanilla" in _problems(output)


def test_catches_membership_supply_and_closing_line(built: Path, tmp_path: Path) -> None:
    output, folder = _copy(built, tmp_path)
    region = next(iter(sorted((folder / "strategicregions").glob("1-*.txt"))))
    region.write_text(region.read_text(encoding="utf-8").replace("\t\t1 ", "\t\t", 1), encoding="utf-8", newline="\n")
    adjacencies = folder / "adjacencies.csv"
    adjacencies.write_text(adjacencies.read_text(encoding="utf-8").replace(CLOSING_LINE + "\n", ""), encoding="utf-8",
                           newline="\n")
    (folder / "railways.txt").write_text("3 2 1 32 \n", encoding="utf-8", newline="\n")
    state = next(iter(sorted((folder.parent / "history" / "states").glob("1-*.txt"))))
    state.write_text(state.read_text(encoding="utf-8").replace("provinces = { ", "provinces = { 2 ", 1), encoding="utf-8",
                     newline="\n")
    text = _problems(output)
    assert "provinces without a strategic region (crash risk): [1]" in text
    assert "must end with the closing line" in text and "do not touch" in text
    assert "province 2 is in states" in text and "does not match the manifest hash" in text


def test_catches_river_and_symmetry_damage(built: Path, tmp_path: Path) -> None:
    output, folder = _copy(built, tmp_path)
    rivers = bmpio.read_bmp(folder / "rivers.bmp")
    pixels = rivers.pixels.copy()
    ys, xs = np.nonzero(pixels <= 11)
    pixels[ys[-1], xs[-1]] = 0  # a second green source on the same river
    pixels[ys[3], xs[3] + 1] = 6  # a side branch without a flow-in marker
    bmpio.write_bmp8(folder / "rivers.bmp", pixels, list(rivers.palette))
    image = bmpio.read_bmp(folder / "provinces.bmp")
    meta = json.loads((output / "arena_map.json").read_text(encoding="utf-8"))
    one, two = meta["provinces"][0], meta["provinces"][1]
    painted = image.pixels.copy()
    painted[one["cy"] - 1:one["cy"] + 2, one["cx"] - 1:one["cx"] + 2] = painted[two["cy"], two["cx"]]
    bmpio.write_bmp24(folder / "provinces.bmp", painted, ppm=3780)
    text = _problems(output)
    assert "2 source pixels" in text and "not a simple orthogonal path" in text
    assert "18 asymmetric pixels outside the declared repairs" in text  # the 3x3 patch and its mirror image


def test_cli_registers_commands() -> None:
    parser = argparse.ArgumentParser()
    handlers = add_commands(parser.add_subparsers(dest="command"))
    assert {"custom-map-build", "custom-map-validate", "custom-map-preview"} <= set(handlers)
    args = parser.parse_args(["custom-map-build", "--design", "all", "--no-install"])
    assert args.design == "all" and args.no_install


@pytest.mark.skipif(not (DEFAULT_GAME / "map" / "rivers.bmp").is_file(), reason="needs the HOI4 install (read only)")
def test_small_build_against_real_install(tmp_path: Path) -> None:
    build_custom_map(_small("open_field"), tmp_path / "real", game=DEFAULT_GAME, previews=False)
    report = validate_custom_map(tmp_path / "real", DEFAULT_GAME)
    assert report["ok"], report["problems"]
    assert report["install_checks"] and report["names_checked"] > 500  # effects/triggers exist in this build's docs
    terrain = bmpio.read_bmp(tmp_path / "real" / "mod" / "hoi4_arena_open_field" / "map" / "terrain.bmp")
    assert list(terrain.palette) == bmpio.vanilla_palette(DEFAULT_GAME / "map" / "terrain.bmp")
    manifest = json.loads((tmp_path / "real" / "arena_manifest.json").read_text(encoding="utf-8"))
    blank = manifest["neutralised_vanilla_files"]["blank"]  # derived from the install, see neutralise.py
    assert "common/ideas/GER.txt" in blank and "common/achievements.txt" in blank and len(blank) > 300
    assert not {"common/ideas/_economic.txt", "common/ideas/_manpower.txt", "common/ideas/army_spirits.txt"} & set(blank)
    assert not any(name.startswith(("common/technologies/", "common/units/equipment/")) for name in blank)
    assert "common/scripted_triggers/00_scripted_triggers.txt" in manifest["neutralised_vanilla_files"]["rescued_bound_files"]


def test_wrap_seam_and_bisect_builds(built: Path, tmp_path: Path) -> None:
    ids = np.ones((256, 512), dtype=np.int32)  # the map wraps: x = 511 touches x = 0 (seen live, launch 2)
    ids[100:, :200], ids[:100, 200:], ids[100:, 200:] = 2, 3, 4
    ids[:100, 300:], ids[100:, 300:] = 5, 6
    assert "3 X crossings" in "\n".join(check_provinces(ids[:, 100:400]))  # two inside, one where 5|6 wraps to 1|2
    full, _ = province_raster(built / "mod" / "hoi4_arena_three_lanes" / "map")
    assert len(x_crossings(np.concatenate([full, full[:, :1]], axis=1))) == 0
    meta = json.loads((built / "arena_map.json").read_text(encoding="utf-8"))
    assert any(x == 511 for x, _ in meta["repairs"]) and any(x == 255 for x, _ in meta["repairs"])
    manifest = build_custom_map(_small("three_lanes"), tmp_path / "debug", game=None, previews=False,
                                bisect=("rails", "keep:common/country_tags", "dynamic_tags"))
    assert manifest["debug_bisect"] and 'replace_path="common/country_tags"' not in (
        tmp_path / "debug" / manifest["mod_dir"] / "descriptor.mod").read_text(encoding="utf-8")
    assert "DEBUG bisect build" in _problems(tmp_path / "debug")
    with pytest.raises(ArenaError):
        build_custom_map(_small("three_lanes"), tmp_path / "bad", game=None, previews=False, bisect=("nonsense",))
