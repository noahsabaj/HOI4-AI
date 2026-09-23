"""Generate an original, rotationally mirrored training map. Never read by the policy.

Every constant here is measured from the stock game or stated in the map-modding
documentation, and `audit` re-checks the written files against the same rules. The engine
does not report bad map data: `CProvinceProvider::GetProvince` returns null below id 1 and
the match-start callers dereference the result, so a wrong value ends the process with an
access violation and no log line.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

# Graphical terrain palette indices. These are the two the stock terrain.bmp actually
# paints most land with: index 0 covers 9.8% of the stock map as terrain_0 (type plains)
# and index 1 covers 5.7% as terrain_1 (type forest), both with no side effects. The
# similarly named plains_17 (index 19) carries perm_snow and forest_13 (index 13) is
# type urban with spawn_city, so neither can stand in for ordinary ground.
TERRAIN_INDEX = {"plains": 0, "forest": 1}
OCEAN_INDEX = 15
# Stock cities.txt claims index 0 for the asia_city group at density 0.00001, so an
# all-zero cities.bmp sprinkles a few buildings across the whole map, ocean included.
# Index 4 is claimed by no city_group, which is how you ask for no cities at all.
CITY_INDEX = 4

# Trees are one model per pixel of trees.bmp, and the engine fixes its resolution at
# 75/256 of the province bitmap: the stock 5632x2048 map ships a 1650x600 tree map.
TREES_NUMERATOR, TREES_DENOMINATOR = 75, 256

# Heightmap bytes. Sea level is 9.5 world units, i.e. byte 95, so land must sit above 95
# and sea below it. The stock sea floor has a median of 89, stock land a median of 107,
# and no neighbouring pair anywhere on it differs by more than 48. Land at exactly 100
# puts the ground at Y=10.00, which is the height every model placement below is written
# at. The shore ramp exists because a hard step at every coast is both a visible cliff
# and steeper than anything the stock map contains.
SEA_FLOOR, LAND_HEIGHT, SHORE_PIXELS = 89, 100, 8

# Unit-counter anchors, taken from the indices the stock unitstacks.txt supplies for all
# land and all sea provinces. 0 standstill, 1-2 moving, 9 attacking, 10 defending,
# 21-23 the regrouping variants, 11-12 and 30-31 disembarking, 38 the victory point.
LAND_STACKS = (0, 1, 2, 9, 10, 21, 22, 23, 38)
SEA_STACKS = (0, 1, 2, 9, 10, 11, 12, 21, 22, 23, 30, 31, 38)
PORT_STACKS = (19, 20)  # ship in port, ship in port moving

# Model placements, keyed by state, that the stock buildings.txt supplies for every state,
# every land province and every coastal land province. A naval placement also carries the
# adjacent sea province: without it the engine has no sea zone for the port.
STATE_BUILDINGS = {
    "air_base": 1,
    "synthetic_refinery": 1,
    "nuclear_reactor_spawn": 1,
    "rocket_site_spawn": 1,
    "radar_station": 1,
    "fuel_silo": 1,
    "stronghold_network": 1,
    "anti_air_building": 3,
    "arms_factory": 6,
    "industrial_complex": 6,
}
PROVINCE_BUILDINGS = ("supply_node", "bunker", "special_project_facility_spawn")
COASTAL_BUILDINGS = (
    "naval_base_spawn",
    "naval_supply_hub",
    "naval_headquarters",
    "coastal_bunker",
    "floating_harbor",
)
PORT_BUILDINGS = ("naval_base_spawn", "floating_harbor")
# Zero-based last day of each month, matching the twelve stock weather periods.
MONTH_LAST_DAY = (30, 27, 30, 29, 30, 29, 30, 30, 29, 30, 29, 30)

# One source of truth for each side's colour: the flag pixels, the country file and the
# country colour database all read from here. The map colour comes from
# common/countries/colors.txt, where the colour space is named explicitly. A bare
# `color = { ... }` in a country file is not the map colour, which is why BLU rendered
# green and RED rendered pale cyan while their flags were right: same numbers, and only
# the flag path read them as RGB.
COUNTRY_COLOUR = {"BLU": (40, 100, 220), "RED": (220, 60, 60)}
COUNTRY_COLOUR_UI = {"BLU": (70, 130, 255), "RED": (255, 90, 90)}

# One field marshal to hold an army group and enough generals to hold armies under it.
# The portrait is the only generic land-commander sprite the stock interface defines; a
# character with none renders an empty frame.
COMMANDER_PORTRAIT = "GFX_portrait_europe_generic_land_13"
GENERALS_PER_COUNTRY = 3

# The map is the size of the stock one, and for the same reason: the camera's zoom-out
# limit is fixed in world units, not fitted to the map. A 2048x1536 arena left the camera
# able to see past the top and bottom edges and more than one map width across, and since
# HOI4 wraps horizontally that showed the same two countries two and a half times over,
# only one copy carrying a name. Both dimensions must be multiples of 256 and the area
# must stay under 13238272 pixels; 5632x2048 is the stock map exactly.
MAP_SIZE = (5632, 2048)
# 32 columns and 24 rows per half, mirrored, so 1536 provinces of about 88x85 px. The
# earlier 8x12 grid gave 352x170 cells: 150 times the area of a mean stock land province
# and 14.6 times its linear size, which made a single border crossing cost 26 in-game days
# and put each capital about five hops and 130 days behind its own front. No match could
# reach a decision, and the 1800-second soak at speed one covered barely one crossing.
COLUMNS_PER_HALF, ROWS, OCEAN_RINGS = 32, 24, 2
# Each half's land is cut into a grid of states rather than held as one. A state is the
# unit the engine builds, supplies and garrisons in, one state per country left both
# countries below the documented three-state minimum for theatre generation, and a single
# supply hub cannot reach the ends of a front column. Both counts must divide the land
# grid exactly: land is COLUMNS_PER_HALF - OCEAN_RINGS columns by ROWS - 2 * OCEAN_RINGS
# rows, so 30 by 20, cut into 6 by 5 states of 5 by 4 provinces each.
STATE_COLUMNS, STATE_ROWS = 6, 5
# Even at 88 px a crossing is 626 km, which the infantry archetype's 4 km/h walks in 6.5
# days against the roughly one day a stock province takes. The rest of the gap is closed
# with a country spirit rather than by shrinking the provinces further, because province
# count is what costs generation time and engine load, and marching speed is free.
ARMY_SPEED_FACTOR = 4.0

# Colours sampled from the stock colour maps, so the arena's water and ground read the way
# the game's own do. The alpha of the RGB colour map is the city-light mask and the alpha
# of the fog-of-war map is the water specular.
OCEAN_COLOUR = (79, 102, 141, 0)
GROUND_COLOUR = {0: (78, 86, 62, 0), 1: (58, 68, 48, 0)}
WATER_COLOUR = (20, 28, 63, 255)
FOG_SEA, FOG_LAND = (68, 69, 68, 44), (131, 132, 131, 27)
MINIMAP_SEA, MINIMAP_LAND = (32, 52, 84, 255), (86, 104, 70, 255)
# The stock minimap widgets, which are pictures of the Earth at the stock map's aspect.
MINIMAP_SIZES = {"gfx/minimap/minimap.dds": (268, 97), "gfx/interface/minimap.dds": (268, 98)}


def write_dds(path, pixels):
    """Write an uncompressed 8.8.8.8 ARGB DDS with no mip chain, as colour maps need."""
    rows, columns = pixels.shape[:2]
    header = struct.pack(
        "<4s7I44x2I4s5I5I",
        b"DDS ",
        124,
        0x100F,  # caps | height | width | pitch | pixelformat
        rows,
        columns,
        columns * 4,
        0,
        0,  # depth, mipmap count
        32,
        0x41,  # pixelformat size, alphapixels | rgb
        b"\0\0\0\0",
        32,
        0x00FF0000,
        0x0000FF00,
        0x000000FF,
        0xFF000000,
        0x1000,  # DDSCAPS_TEXTURE
        0,
        0,
        0,
        0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # The masks above put blue in the low byte, so the rows are stored BGRA.
    path.write_bytes(header + pixels[..., [2, 1, 0, 3]].astype(np.uint8).tobytes())


def adjacency(ids, count):
    """Province neighbours as the engine reads them: shared edges in provinces.bmp."""
    neighbours = {i: set() for i in range(1, count + 1)}
    for first, second in [(ids[:-1, :], ids[1:, :]), (ids[:, :-1], ids[:, 1:])]:
        mask = first != second
        for a, b in np.unique(np.stack([first[mask], second[mask]], 1), axis=0).tolist():
            neighbours[a].add(b)
            neighbours[b].add(a)
    return neighbours


def shore_heights(ground):
    """A smooth coast: byte 100 inland, 89 offshore, crossing sea level at the border."""
    signed = distance_transform_edt(ground) - distance_transform_edt(~ground)
    ramp = np.clip(signed / SHORE_PIXELS, -1, 1)
    middle, half = (LAND_HEIGHT + SEA_FLOOR) / 2, (LAND_HEIGHT - SEA_FLOOR) / 2
    return np.round(middle + half * ramp).astype(np.uint8)


def generate(
    game,
    output,
    *,
    undefended=None,
    victory_points_on_border=False,
    columns_per_half=COLUMNS_PER_HALF,
    rows=ROWS,
    state_columns=STATE_COLUMNS,
    state_rows=STATE_ROWS,
    land_columns=None,
    land_rows=None,
):
    """Write an arena. The keyword arguments build diagnostics, not playable arenas.

    `undefended` fields no divisions for one side. `victory_points_on_border` moves every
    victory point onto the border column. That does not produce a surrender: capitulation
    is territorial, and taking the whole victory-point weight was measured to leave the
    country in the war. Both exist to make something happen on screen that a balanced
    arena cannot be asked to produce on demand, and `generation.json` records which were
    used.

    `land_columns` and `land_rows` shrink each country to a block of the full grid: the
    block touches the seam and is centered vertically, and every other cell is sea. Every
    province keeps the playable arena's size. Centering a smaller lattice on the bitmap
    instead left the margin to a handful of sea provinces up to 2245x769 px, and the
    engine crashed loading them. Fewer than three states a side is below the theatre
    minimum, and a land block that does not divide into the state grid is rejected here
    rather than left for the engine.
    """
    game, root = Path(game), Path(output).resolve()
    if not (game / "map/provinces.bmp").exists():
        raise ValueError("Point --game at the installed HOI4 directory")
    if undefended is not None and undefended not in COUNTRY_COLOUR:
        raise ValueError("undefended names a country tag: BLU or RED")
    if columns_per_half <= OCEAN_RINGS or rows <= 2 * OCEAN_RINGS:
        raise ValueError("grid leaves no land after the ocean rings")
    full_columns, full_rows = columns_per_half - OCEAN_RINGS, rows - 2 * OCEAN_RINGS
    land_columns = full_columns if land_columns is None else int(land_columns)
    land_rows = full_rows if land_rows is None else int(land_rows)
    if not (0 < land_columns <= full_columns and 0 < land_rows <= full_rows):
        raise ValueError("land block must fit inside the ocean rings")
    if (rows - land_rows) % 2:
        # The right half is the left rotated, so an off-centre block would meet its
        # mirror a row out of step along the seam.
        raise ValueError("rows minus land rows must be even so the two fronts line up")
    if land_columns % state_columns or land_rows % state_rows:
        raise ValueError("land grid does not divide into whole states")
    if state_columns * state_rows < 3:
        raise ValueError("a country needs at least three states for theatre generation")

    def write(name, text):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # definition.csv loses every province's continent if the rows end in bare LF, so
        # every generated text file is written with CRLF whatever the host platform is.
        path.write_text(
            text, encoding="utf-8-sig" if name.endswith(".yml") else "utf8", newline="\r\n"
        )

    # Both dimensions must be a multiple of 256 and the area must stay under 13238272 px.
    width, height = MAP_SIZE
    # Stretch across the bitmap. The remainder stays on the last row, as it always has:
    # centering it would move every province and invalidate the measured arena.
    step_x, step_y = width // (2 * columns_per_half), height // rows
    # The land block of the left half. The default is everything inside the ocean rings.
    column0, row0 = columns_per_half - land_columns, (rows - land_rows) // 2
    root.mkdir(parents=True, exist_ok=False)
    half_count = columns_per_half * rows
    total_provinces = 2 * half_count
    left = np.array(
        [
            (
                x * step_x + step_x // 2,
                y * step_y + step_y // 2 + (x % 2) * (step_y // 4),
            )
            for x in range(columns_per_half)
            for y in range(rows)
        ]
    )
    points = np.concatenate([left, [width - 1, height - 1] - left])
    yy, xx = np.mgrid[:height, :width]
    ids = cKDTree(points).query(np.stack([xx.ravel(), yy.ravel()], 1))[1].reshape(height, width) + 1
    # At least two rings of provinces on the outer edges are sea, so the land sits in open
    # water rather than running off the edge of the world. A smaller land block leaves
    # more of the grid as sea, at the same province size. The right half is the left half
    # rotated, so it shares the left half's land flags.
    column, row = np.divmod(np.arange(half_count), rows)
    half_land = (column >= column0) & (row >= row0) & (row < row0 + land_rows)
    land = np.concatenate([half_land, half_land])
    # Break pixel-only four-way contacts, preserving rotational symmetry. The map wraps
    # horizontally, so the seam between the last and first column is a contact too.
    for _ in range(3):
        wrapped = np.concatenate([ids, ids[:, :1]], axis=1)
        a, b = wrapped[:-1, :-1], wrapped[:-1, 1:]
        c, d = wrapped[1:, :-1], wrapped[1:, 1:]
        crossing = (a != b) & (a != c) & (a != d) & (b != c) & (b != d) & (c != d)
        for y, x in np.argwhere(crossing):
            if x < width // 2:
                ids[y, (x + 1) % width] = ids[y, x]
                mirror = (ids[y, x] + half_count - 1) % total_provinces + 1
                ids[height - 1 - y, (width - 2 - x) % width] = mirror
    # definition.csv is read back by colour, so two provinces sharing one is a map that
    # silently loses provinces. The old scheme took each channel modulo 251, which repeats
    # every 251 ids and was only safe while there were 192 of them. Multiplying by an odd
    # constant and keeping the low 24 bits is injective for every id the map can hold, and
    # only id 0 lands on black.
    colors = np.array(
        [
            [(i * 2654435761 >> shift) & 255 for shift in (16, 8, 0)]
            for i in range(total_provinces + 1)
        ],
        dtype=np.uint8,
    )
    (root / "map").mkdir()
    Image.fromarray(colors[ids]).save(root / "map/provinces.bmp")
    ground = land[ids - 1]
    # Forest on every sixth row, so the same one-in-six share of the map as the 8x12 grid
    # painted, which is close to the share the stock terrain.bmp gives palette index 1.
    terrain_types = ["forest" if i % rows % 6 == 3 else "plains" for i in range(half_count)] * 2
    terrain_ids = np.array([TERRAIN_INDEX[t] for t in terrain_types], dtype=np.uint8)
    neighbours = adjacency(ids, len(points))
    # A coast is a shared edge between the two classes, so it belongs to both provinces.
    # Since 1.11 the bitmap decides and definition.csv only has to agree with it, but a
    # disagreement is one MAP_ERROR per province in an already long startup log.
    coastal = {i: any(land[i - 1] != land[j - 1] for j in sides) for i, sides in neighbours.items()}
    ports = {
        i: min((j for j in neighbours[i] if not land[j - 1]), default=0)
        for i in neighbours
        if land[i - 1] and coastal[i]
    }

    def indexed(name, pixels):
        original = Image.open(game / "map" / name)
        image = Image.fromarray(pixels.astype(np.uint8)).convert("P")
        palette = original.getpalette()
        if palette is not None:
            image.putpalette(palette)
        image.save(root / "map" / name)

    graphical = np.where(ground, terrain_ids[ids - 1], OCEAN_INDEX)
    indexed("terrain.bmp", graphical)
    indexed("rivers.bmp", np.where(ground, 255, 254))
    Image.fromarray(shore_heights(ground)).save(root / "map/heightmap.bmp")
    Image.new("RGB", (width // 2, height // 2), (128, 128, 255)).save(root / "map/world_normal.bmp")
    trees = (
        width * TREES_NUMERATOR // TREES_DENOMINATOR,
        height * TREES_NUMERATOR // TREES_DENOMINATOR,
    )
    indexed("trees.bmp", np.zeros((trees[1], trees[0]), dtype=np.uint8))
    indexed("cities.bmp", np.full((height, width), CITY_INDEX, dtype=np.uint8))

    # The map-shaped textures. Every one of these is a painting of the stock Earth at the
    # stock map's aspect, so leaving them out leaves the Earth's coastline drawn over the
    # arena's ocean, its biome colours over the arena's land and its cities glowing at
    # night. They are half the province bitmap's resolution, uncompressed, without mips.
    half = (slice(None, None, 2), slice(None, None, 2))
    land_half, terrain_half = ground[half], graphical[half]
    colour = np.zeros((*land_half.shape, 4), dtype=np.uint8)
    colour[...] = OCEAN_COLOUR
    for index, value in GROUND_COLOUR.items():
        colour[land_half & (terrain_half == index)] = value
    write_dds(root / "map/terrain/colormap_rgb_cityemissivemask_a.dds", colour)
    fog = np.where(land_half[..., None], np.array(FOG_LAND), np.array(FOG_SEA)).astype(np.uint8)
    write_dds(root / "map/terrain/fow_rgb_waterspec_a.dds", fog)
    for level in range(3):
        step = 2 ** (level + 1)
        water = np.empty((height // step, width // step, 4), dtype=np.uint8)
        water[...] = WATER_COLOUR
        write_dds(root / f"map/terrain/colormap_water_{level}.dds", water)
    for name, (widget_columns, widget_rows) in MINIMAP_SIZES.items():
        shrunk = np.array(
            Image.fromarray(ground.astype(np.uint8) * 255).resize(
                (widget_columns, widget_rows), Image.BILINEAR
            )
        )
        picture = np.where(shrunk[..., None] > 127, np.array(MINIMAP_LAND), np.array(MINIMAP_SEA))
        write_dds(root / name, picture.astype(np.uint8))

    definitions = ["0;0;0;0;land;false;unknown;0"]
    for i in range(1, len(points) + 1):
        is_land = bool(land[i - 1])
        r, g, b = colors[i]
        definitions.append(
            f"{i};{r};{g};{b};{'land' if is_land else 'sea'};{str(coastal[i]).lower()};"
            f"{terrain_types[i - 1] if is_land else 'ocean'};{1 if is_land else 0}"
        )
    write("map/definition.csv", "\n".join(definitions) + "\n")
    write(
        "map/default.map",
        "\n".join(
            f'{key} = "{value}"'
            for key, value in {
                "definitions": "definition.csv",
                "provinces": "provinces.bmp",
                "positions": "positions.txt",
                "terrain": "terrain.bmp",
                "rivers": "rivers.bmp",
                "heightmap": "heightmap.bmp",
                "tree_definition": "trees.bmp",
                "continent": "continent.txt",
                "adjacency_rules": "adjacency_rules.txt",
                "adjacencies": "adjacencies.csv",
                "ambient_object": "ambient_object.txt",
                "seasons": "seasons.txt",
            }.items()
        )
        + "\ntree = { 3 4 7 10 }\n",
    )
    write(
        "map/continent.txt",
        "continents = { europe north_america south_america australia africa asia middle_east }\n",
    )
    # The stock ambient objects are the world frame, positioned for a 5632x2048 map, and
    # the stock adjacency rules name provinces this map does not have. seasons.txt is left
    # alone deliberately: the stock one is real content and nothing here invalidates it.
    for name in ["positions.txt", "adjacency_rules.txt", "ambient_object.txt"]:
        write(f"map/{name}", "# Generated arena: no custom entries.\n")
    # The stock tutorial names state 550 and provinces 5010, 5091 and 12766, and the
    # in-game hint loader resolves them at match start whether or not anyone asked for a
    # tutorial. On a 2-state, 192-province map that lookup returns null and is
    # dereferenced, which is a crash with no log line before it. replace_path does not
    # unload this folder, so the file itself has to be overridden. It cannot be emptied
    # either: the loader finishes by marking the last entry of the list it just built, and
    # on an empty list that indexes element -1. One block naming no province and no state
    # is the smallest thing that both loads and resolves.
    write(
        "tutorial/tutorial.txt",
        'tutorial = {\n\twindow = "tutorial_screen_1"\n\tuse_mil_fac = { textbox = "obj_1" }\n}\n',
    )
    # The trailing -1 row is the engine's end-of-file marker, not a stray sentinel: the
    # stock file has one and the documentation requires it even when there are no rows.
    write(
        "map/adjacencies.csv",
        "From;To;Type;Through;start_x;start_y;stop_x;stop_y;adjacency_rule_name;Comment\n"
        "-1;-1;;-1;-1;-1;-1;-1;-1\n",
    )
    stacks = []
    for i, (x, y) in enumerate(points, 1):
        slots = LAND_STACKS if land[i - 1] else SEA_STACKS
        if land[i - 1] and coastal[i]:
            slots += PORT_STACKS
        for slot in slots:
            stacks.append(f"{i};{slot};{x}.00;10.00;{height - y}.00;0.00;0.30")
    write("map/unitstacks.txt", "\n".join(stacks) + "\n")
    weather = " ".join(
        f"period = {{ between = {{ 0.{month} {last}.{month} }} "
        f"temperature = {{ 15.0 20.0 }} no_phenomenon = 1.0 }}"
        for month, last in enumerate(MONTH_LAST_DAY)
    )
    for region, mask in [(1, land), (2, ~land)]:
        listed = (np.flatnonzero(mask) + 1).tolist()
        # A sea region takes its provincial terrain from the region, not definition.csv.
        naval = "" if region == 1 else "naval_terrain = water_shallow_sea "
        write(
            f"map/strategicregions/{region}-arena.txt",
            f'strategic_region = {{ id = {region} name = "ARENA_REGION_{region}" '
            f"provinces = {{ {' '.join(map(str, listed))} }} {naval}"
            f"weather = {{ {weather} }} }}",
        )
    # Weather objects belong over their own region, so each one is anchored on provinces
    # that region actually contains.
    positions = []
    for region, mask in [(1, land), (2, ~land)]:
        listed = (np.flatnonzero(mask) + 1).tolist()
        chosen = [listed[len(listed) // 4], listed[3 * len(listed) // 4]]
        for kind in ["small", "big"]:
            for province in chosen:
                x, y = points[province - 1]
                positions.append(f"{region};{x}.00;10.00;{height - y}.00;{kind}")
    write("map/weatherpositions.txt", "\n".join(positions) + "\n")

    left_land = (np.flatnonzero(land[:half_count]) + 1).tolist()
    right_land = [i + half_count for i in left_land]
    state_width, state_height = land_columns // state_columns, land_rows // state_rows
    states_per_country = state_columns * state_rows
    # One division per row of the border column, so a side actually holds its own front
    # instead of leaving gaps an opponent can walk through unopposed.
    divisions_per_country = land_rows

    def state_cell(province):
        """Which state of its own half a land province falls in, counted from zero.

        Province ids run down each column, and the right half is the left half rotated,
        so the same column-and-row arithmetic places both and the state grid comes out
        rotationally symmetric for free.
        """
        index = (province - 1) % half_count
        column = index // rows - column0
        row = index % rows - row0
        return (column // state_width) * state_rows + row // state_height

    states, state_owner, capitals, capital_states, victory_points = {}, {}, [], [], {}
    for half, (tag, province_list) in enumerate([("BLU", left_land), ("RED", right_land)]):
        if (land_columns, land_rows) != (full_columns, full_rows):
            centre = points[np.array(province_list) - 1].mean(axis=0)
        elif half == 0:
            # Kept exactly as measured, so the playable arena's capital does not move.
            centre = [width // 4, height // 2]
        else:
            centre = [width - 1 - width // 4, height - 1 - height // 2]
        # The border column: one province per land row, nearest the vertical seam. The
        # starting divisions stand here, and the harness puts every victory point here
        # too, so it is computed once and shared.
        border = sorted(province_list, key=lambda i: abs(points[i - 1, 0] - width / 2))[
            :divisions_per_country
        ]
        capital = (
            min(border, key=lambda i: abs(points[i - 1, 1] - height / 2))
            if victory_points_on_border
            else min(province_list, key=lambda i: np.linalg.norm(points[i - 1] - centre))
        )
        capitals.append(capital)
        for province in province_list:
            state = half * states_per_country + state_cell(province) + 1
            states.setdefault(state, []).append(province)
            state_owner[state] = tag
        capital_states.append(half * states_per_country + state_cell(capital) + 1)
        # Victory points are not the surrender threshold. A measured match gave Red a
        # single border province carrying all 35 of them; Blue took it on 13 January
        # and Red had not capitulated by May 1940. Surrender is occupation.
        # BASE_SURRENDER_LIMIT is that fraction, and BASE_SURRENDER_LEVEL is the level
        # that has to be reached. The spread below is so the tooltip is not one tile.
        # The harness puts the whole weight on the border anyway, because that was the
        # measurement that separated the two.
        if victory_points_on_border:
            outposts = []
        else:
            spread = sorted(
                province_list,
                key=lambda i: -np.linalg.norm(points[i - 1] - points[capital - 1]),
            )
            outposts = [spread[0], spread[len(spread) // 2], spread[-2]]
        victory_points[tag] = (
            {capital: 35}
            if victory_points_on_border
            else {capital: 20, **{p: 5 for p in outposts if p != capital}}
        )
        write(
            f"common/countries/{tag}.txt",
            f"graphical_culture = western_european_gfx\ngraphical_culture_2d = western_european_2d\n"
            f"color = rgb {{ {' '.join(map(str, COUNTRY_COLOUR[tag]))} }}",
        )
        write(
            f"history/countries/{tag} - Arena.txt",
            f'capital = {capital_states[half]}\noob = "{tag}_1936"\nrecruit_character = {tag}_commander\n'
            + f"recruit_character = {tag}_marshal\n"
            + "".join(
                f"recruit_character = {tag}_general_{n}\n"
                for n in range(1, GENERALS_PER_COUNTRY + 1)
            )
            + f"set_politics = {{ ruling_party = neutrality elections_allowed = no }}\nset_popularities = {{ neutrality = 100 }}\nset_stability = 1\nset_war_support = 1\nadd_ideas = arena_march_speed\nset_technology = {{ infantry_weapons = 1 infantry_weapons1 = 1 basic_train = 1 }}\nadd_equipment_to_stockpile = {{ type = infantry_equipment_1 amount = 50000 producer = {tag} }}\nadd_equipment_to_stockpile = {{ type = train_equipment_1 amount = 50 producer = {tag} }}\n",
        )
        regiments = " ".join(
            f"infantry = {{ x = {x} y = {y} }}" for x in range(2) for y in range(3)
        )
        # An undefended side keeps its template, its equipment and its generals and
        # fields nothing. That separates the two readings a motionless map cannot tell
        # apart: an AI that never attacks leaves an empty front untouched, and an AI
        # whose every attack fails does not.
        deployed = [] if tag == undefended else border
        divisions = "\n".join(
            f'division = {{ name = "Infantry {n}" location = {p} division_template = "Arena Infantry" start_experience_factor = 0.3 start_equipment_factor = 1 }}'
            for n, p in enumerate(deployed, 1)
        )
        write(
            f"history/units/{tag}_1936.txt",
            f'division_template = {{ name = "Arena Infantry" regiments = {{ {regiments} }} }}\nunits = {{ {divisions} }}',
        )
        for sub, size in [("", (82, 52)), ("medium/", (41, 26)), ("small/", (10, 7))]:
            flag = root / f"gfx/flags/{sub}{tag}.tga"
            flag.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", size, (*COUNTRY_COLOUR[tag], 255)).save(flag)
    # A country's manpower is split across its states rather than repeated in each, so the
    # total stays the roughly one million that twenty divisions can actually draw on.
    for state, province_list in sorted(states.items()):
        tag = state_owner[state]
        points_block = " ".join(
            f"victory_points = {{ {province} {value} }}"
            for province, value in victory_points[tag].items()
            if province in set(province_list)
        )
        write(
            f"history/states/{state}-arena.txt",
            f'state = {{ id = {state} name = "ARENA_STATE_{state}" manpower = {1000000 // states_per_country} state_category = rural history = {{ owner = {tag} add_core_of = {tag} {points_block} buildings = {{ infrastructure = 4 }} }} provinces = {{ {" ".join(map(str, province_list))} }} }}',
        )
    # Marching speed, not province size, is what closes the gap between an 88 px cell and
    # the roughly one day a stock province takes to cross. common/ideas is not replaced,
    # so this file merges with the stock ones rather than shadowing them.
    write(
        "common/ideas/arena.txt",
        "ideas = {\n\tcountry = {\n\t\tarena_march_speed = {\n"
        "\t\t\tallowed = { always = no }\n\t\t\tremoval_cost = -1\n"
        f"\t\t\tmodifier = {{ army_speed_factor = {ARMY_SPEED_FACTOR} }}\n"
        "\t\t}\n\t}\n}\n",
    )
    # Every country that exists at game start has a name list. Without one the engine
    # still takes the random-character path for leaders, advisors and unit commanders,
    # fails to name them, and dereferences the result. common/names is not replaced, so a
    # differently named file merges with the stock one rather than shadowing it.
    write(
        "common/names/01_arena_names.txt",
        "\n".join(
            f"{tag} = {{\n\tmale = {{ names = {{ {male} }} }}\n"
            f"\tfemale = {{ names = {{ {female} }} }}\n"
            f"\tsurnames = {{ {surnames} }}\n\tcallsigns = {{ }}\n}}"
            for tag, male, female, surnames in [
                (
                    "BLU",
                    "Alan Arthur Bernard Charles David Edward Francis George Harold Henry "
                    "James John Leonard Martin Michael Norman Oliver Philip Richard Robert "
                    "Samuel Stephen Thomas Victor Walter William",
                    "Alice Barbara Catherine Dorothy Edith Eleanor Frances Grace Helen Irene "
                    "Joan Katherine Louise Margaret Marion Nancy Olive Rachel Ruth Sarah "
                    "Sylvia Vera Violet Winifred",
                    "Ashton Baker Bennett Carter Chapman Clarke Cooper Dawson Ellis Fletcher "
                    "Gibson Hale Harper Hayes Hudson Kent Lawson Marsh Newton Osborne Palmer "
                    "Reed Sinclair Stanton Thornton Vance Warren Whitfield Wilkins Young",
                ),
                (
                    "RED",
                    "Adrian Alexis Anton Boris Dimitri Fedor Gregor Ivan Konstantin Leonid "
                    "Maksim Mikhail Nikolai Oleg Pavel Roman Sergei Stepan Timur Valentin "
                    "Vasili Viktor Vladimir Yakov Yuri Zakhar",
                    "Anna Daria Ekaterina Elena Galina Inna Irina Klavdia Larisa Lidia "
                    "Lyudmila Marina Nadezhda Natalia Nina Olga Polina Raisa Svetlana Tamara "
                    "Tatiana Valentina Yelena Zoya",
                    "Agapov Belov Chernov Dorokhov Ermakov Gorelik Ivashov Kalinin Komarov "
                    "Lapin Maslov Nesterov Orlov Panov Rodin Savelev Shestakov Sokolov "
                    "Tarasov Ustinov Vlasov Volkov Yudin Zaitsev Zhukov",
                ),
            ]
        )
        + "\n",
    )

    # A country leader is not a general. Both tags had only the former, so neither could
    # form an army group and the audit never looked. Stock 1936 countries do exist with no
    # commander at all and still fight, so this is not on its own why Red never moved, but
    # a front the AI is meant to populate needs someone to command it.
    def commander(key, role):
        return (
            f'\t{key} = {{\n\t\tname = "{key}"\n'
            f'\t\tportraits = {{ army = {{ small = "{COMMANDER_PORTRAIT}_small" }} '
            f'army = {{ large = "{COMMANDER_PORTRAIT}" }} }}\n'
            f"\t\t{role} = {{ traits = {{ }} skill = 3 attack_skill = 3 defense_skill = 3 "
            f"planning_skill = 3 logistics_skill = 3 }}\n\t}}\n"
        )

    write(
        "common/characters/arena.txt",
        "characters = {\n"
        + "".join(
            f'\t{tag}_commander = {{\n\t\tname = "{name}"\n'
            f'\t\tcountry_leader = {{ ideology = despotism expire = "1965.1.1.1" id = -1 }}\n\t}}\n'
            + commander(f"{tag}_marshal", "field_marshal")
            + "".join(
                commander(f"{tag}_general_{n}", "corps_commander")
                for n in range(1, GENERALS_PER_COUNTRY + 1)
            )
            for tag, name in [("BLU", "Blue Command"), ("RED", "Red Command")]
        )
        + "}\n",
    )
    # The engine draws the front on its own, but nothing here ever told either AI to
    # execute an order across it, and Red held position for three months of game time
    # against a stationary Blue. front_control is the documented override: execute_order
    # forces the execute-or-not decision and execution_type overrides the stance the AI
    # would otherwise pick from a front-strength comparison that two identical armies can
    # never move. common/ai_strategy is not replaced, so this merges with the stock files.
    write(
        "common/ai_strategy/arena.txt",
        "".join(
            f"{tag}_arena_offensive = {{\n"
            f"\tallowed = {{ original_tag = {tag} }}\n"
            f"\tenable = {{ has_war_with = {enemy} }}\n"
            f"\tabort = {{ always = no }}\n\n"
            f"\tai_strategy = {{\n\t\ttype = front_control\n\t\ttag = {enemy}\n"
            f"\t\tratio = 0.1\n\t\tpriority = 100\n\t\tordertype = front\n"
            f"\t\texecution_type = rush\n\t\texecute_order = yes\n\t\tmanual_attack = yes\n\t}}\n"
            f"}}\n"
            for tag, enemy in [("BLU", "RED"), ("RED", "BLU")]
        ),
    )
    # Ordinary supply hubs and rail lines following actual bitmap adjacency.
    rails = [
        f"1 2 {a} {b}"
        for a, sides in sorted(neighbours.items())
        for b in sorted(sides)
        if a < b and land[a - 1] and land[b - 1] and (a <= half_count) == (b <= half_count)
    ]
    write("map/railways.txt", "\n".join(rails) + "\n")

    def state_centre(province_list):
        """The province nearest the middle of a state, used to anchor its hub and slots."""
        middle = points[np.array(province_list) - 1].mean(axis=0)
        return min(province_list, key=lambda i: np.linalg.norm(points[i - 1] - middle))

    centres = {state: state_centre(listed) for state, listed in states.items()}
    # A hub in every state rather than one per country. Supply flow falls off per province
    # travelled and runs out after about two hops, so a single mid-front hub left the ends
    # of the border column out of supply, which caps a division's organisation below the
    # level the AI requires before it will attack with it at all.
    hubs = sorted(set(centres.values()) | set(capitals))
    write("map/supply_nodes.txt", "\n".join(f"1 {hub}" for hub in hubs) + "\n")
    # One placement wherever the stock database supplies one: per state slot, per land
    # province and per coastal province. A building the engine can place but has no
    # position for leaves it holding province 0, which is the null province.
    buildings = []
    for state, province_list in sorted(states.items()):
        cx, cy = points[centres[state] - 1]
        for kind, slots in STATE_BUILDINGS.items():
            for slot in range(slots):
                buildings.append(
                    f"{state};{kind};{cx + slot * 2}.00;10.00;{height - cy + slot * 2}.00;0.00;0"
                )
        if any(coastal[i] for i in province_list):
            buildings.append(f"{state};dockyard;{cx}.00;10.00;{height - cy}.00;0.00;0")
        for i in province_list:
            x, y = points[i - 1]
            for kind in PROVINCE_BUILDINGS:
                buildings.append(f"{state};{kind};{x}.00;10.00;{height - y}.00;0.00;0")
            for kind in COASTAL_BUILDINGS if coastal[i] else ():
                port = ports[i] if kind in PORT_BUILDINGS else 0
                buildings.append(f"{state};{kind};{x}.00;10.00;{height - y}.00;0.00;{port}")
    write("map/buildings.txt", "\n".join(buildings) + "\n")
    write(
        "common/national_focus/arena.txt",
        "focus_tree = { id = arena_focus default = yes country = { factor = 1 } "
        "focus = { id = arena_training icon = GFX_goal_generic_army_doctrines "
        "x = 0 y = 0 cost = 1000 completion_reward = { } } }",
    )
    write(
        "common/country_tags/00_arena.txt", 'BLU = "countries/BLU.txt"\nRED = "countries/RED.txt"\n'
    )
    # The map colour comes from this database, not from the country file, and every entry
    # names its colour space. Replacing the stock file drops the colours of the 351 stock
    # tags, which is harmless here: their history files are replaced away, so none of them
    # owns a province to paint.
    write(
        "common/countries/colors.txt",
        "#reload countrycolors\n"
        + "".join(
            f"{tag} = {{\n"
            f"\tcolor = rgb {{ {' '.join(map(str, COUNTRY_COLOUR[tag]))} }}\n"
            f"\tcolor_ui = rgb {{ {' '.join(map(str, COUNTRY_COLOUR_UI[tag]))} }}\n"
            f"}}\n"
            for tag in COUNTRY_COLOUR
        ),
    )
    write(
        "common/bookmarks/arena.txt",
        'bookmarks = { bookmark = { name = "ARENA_BOOKMARK" desc = "ARENA_DESC" '
        'date = 1936.1.1.12 picture = "GFX_select_date_1936" default_country = "BLU" default = yes '
        "effect = { randomize_weather = 22 } "
        'BLU = { history = "ARENA_BLU_HISTORY" ideology = neutrality } '
        'RED = { history = "ARENA_RED_HISTORY" ideology = neutrality } } }',
    )
    # Besides the war, the arena reports itself in game.log, which changes no rule: each
    # line starts "ARENA " so the worker's game_log request can pick them out. A surrender
    # names both sides, so a match ends without reading pixels, and the weekly counts
    # give an exact territory figure. Checked on 2026-09-22: the dynamic variables
    # resolve in log strings, and a state changing hands logs its name.
    write(
        "common/on_actions/arena.txt",
        "on_actions = {\n"
        "\ton_startup = { effect = { BLU = { declare_war_on = { target = RED type = annex_everything } }"
        ' log = "ARENA start [GetDateText]" } }\n'
        '\ton_weekly = { effect = { log = "ARENA week [GetDateText] [ROOT.GetTag] states'
        " [?num_controlled_states] owned [?num_owned_controlled_states] divisions"
        ' [?num_divisions] surrender [?surrender_progress]" } }\n'
        '\ton_capitulation = { effect = { log = "ARENA capitulated [ROOT.GetTag] winner'
        ' [FROM.GetTag] [GetDateText]" } }\n'
        '\ton_state_control_changed = { effect = { log = "ARENA control [ROOT.GetTag] from'
        ' [FROM.GetTag] [FROM.FROM.GetName] [GetDateText]" } }\n'
        '\ton_peaceconference_ended = { effect = { log = "ARENA peace [ROOT.GetTag]'
        ' [FROM.GetTag] [GetDateText]" } }\n'
        "}",
    )
    # Without an adjective and an ideology-qualified name every string the game builds
    # from the tag renders a raw key, starting with the name of the war.
    localisation = [
        "l_english:",
        ' ARENA_BOOKMARK:0 "Infantry Arena"',
        ' ARENA_DESC:0 "Equal infantry armies. Multiple routes. Normal supply and fog of war."',
        ' ARENA_BLU_HISTORY:0 "Blue holds the western half of the arena."',
        ' ARENA_RED_HISTORY:0 "Red holds the eastern half of the arena."',
        ' ARENA_REGION_1:0 "Arena"',
        ' ARENA_REGION_2:0 "Ocean"',
        ' arena_focus:0 "Arena"',
        ' arena_training:0 "Army Training"',
        ' arena_training_desc:0 ""',
        ' arena_march_speed:0 "Arena March Rate"',
        ' arena_march_speed_desc:0 "Provinces here are far larger than a stock one, so'
        ' armies march proportionally faster."',
    ]
    # Every state needs a name or the engine falls back to a stock string, which is how
    # Red's capital came to be labelled Kargopol.
    for state in sorted(states):
        side = "West" if state_owner[state] == "BLU" else "East"
        localisation.append(
            f' ARENA_STATE_{state}:0 "{side} {(state - 1) % states_per_country + 1}"'
        )
    for tag, name in [("BLU", "Blue"), ("RED", "Red")]:
        localisation += [
            f' {tag}:0 "{name}"',
            f' {tag}_DEF:0 "{name}"',
            f' {tag}_ADJ:0 "{name}"',
            f' {tag}_neutrality:0 "{name}"',
            f' {tag}_neutrality_DEF:0 "{name}"',
            f' {tag}_neutrality_ADJ:0 "{name}"',
            f' {tag}_commander:0 "{name} Command"',
            f' {tag}_marshal:0 "{name} Marshal"',
            *(
                f' {tag}_general_{n}:0 "{name} General {n}"'
                for n in range(1, GENERALS_PER_COUNTRY + 1)
            ),
        ]
    for tag, listed in victory_points.items():
        side = "West" if tag == "BLU" else "East"
        for order, province in enumerate(listed):
            label = f"{side} Capital" if order == 0 else f"{side} Outpost {order}"
            localisation.append(f' VICTORY_POINTS_{province}:0 "{label}"')
    write("localisation/english/arena_l_english.yml", "\n".join(localisation) + "\n")
    replacements = [
        # The stock tutorial hard-codes state 550 and provinces 5010, 5091 and 12766, and
        # the in-game hint loader resolves them at match start whether or not anyone asked
        # for a tutorial. On a 2-state, 192-province map that lookup returns null and is
        # dereferenced immediately, which is a crash with no log line before it.
        "tutorial",
        "history/countries",
        "history/states",
        "history/units",
        "common/bookmarks",
        "common/national_focus",
        "common/ai_focuses",
        "common/ai_strategy_plans",
        "common/decisions",
        "common/decisions/categories",
        "common/strategic_locations",
        # Stock events and on_actions refer to states and countries this map lacks. Left
        # in, the game crashed on the first daily tick (1936-01-02, 2026-09-22). Replaced,
        # a capitulation still ran all the way through its peace conference.
        "events",
        "common/on_actions",
        "map/strategicregions",
        "map/supplyareas",
    ]
    descriptor = (
        'name = "HOI4 Visual Infantry Arena"\nversion = "0.1"\nsupported_version = "1.19.*"\n'
        + "".join(f'replace_path = "{p}"\n' for p in replacements)
    )
    for directory in replacements:
        (root / directory).mkdir(parents=True, exist_ok=True)
    write("descriptor.mod", descriptor)
    (root.with_suffix(".mod")).write_text(
        descriptor + f'path = "{root.as_posix()}"\n', newline="\r\n"
    )
    report = {
        "width": width,
        "height": height,
        "countries": 2,
        "provinces": total_provinces,
        "land_provinces_per_country": len(left_land),
        "states_per_country": states_per_country,
        "divisions_per_country": divisions_per_country,
        "columns_per_half": columns_per_half,
        "rows": rows,
        "state_columns": state_columns,
        "state_rows": state_rows,
        "land_columns": land_columns,
        "land_rows": land_rows,
        "undefended": undefended,
        "victory_points_on_border": victory_points_on_border,
        "army_speed_factor": ARMY_SPEED_FACTOR,
        "coastal_land_provinces": sum(1 for i in neighbours if land[i - 1] and coastal[i]),
        "victory_points_per_country": len(victory_points["BLU"]),
        "rotational_mirror": True,
        "gameplay_verified": False,
        "engine_load_verified": False,
    }
    write("generation.json", json.dumps(report, indent=2))
    return report


def _block(text, key):
    found = re.search(key + r"\s*=\s*\{([^}]*)\}", text)
    return [int(value) for value in found.group(1).split()] if found else []


def _dds_size(path):
    header = path.open("rb").read(20)
    rows, columns = struct.unpack_from("<II", header, 12)
    return columns, rows


def audit(root):
    """Report references a generated arena asks the engine to resolve and it cannot.

    Every finding is a province the engine looks up and does not find, a placement it
    looks for and does not have, or a stock asset the arena leaves in place that is a
    picture of a different world. This reads the written files, so a hand-edited mod is
    checked too, and it is the only check that can run without the game.
    """
    root = Path(root)
    problems = []
    rows = [r.split(";") for r in (root / "map/definition.csv").read_text().splitlines() if r]
    ids = [int(r[0]) for r in rows]
    kind = {int(r[0]): r[4] for r in rows}
    coastal = {int(r[0]): r[5] == "true" for r in rows}
    count = max(ids)
    valid = set(range(1, count + 1))
    if ids != list(range(count + 1)):
        problems.append("definition.csv province ids are not contiguous from 0")
    if b"\r\n" not in (root / "map/definition.csv").read_bytes():
        problems.append("definition.csv does not use CRLF, so every continent reads as missing")

    def check(name, found):
        outside = sorted({p for p in found if p not in valid})
        if outside:
            problems.append(f"{name} references provinces outside 1..{count}: {outside[:8]}")

    adjacencies = (root / "map/adjacencies.csv").read_text().splitlines()
    columns = []
    for row in adjacencies[1:]:
        cells = row.split(";")
        if not row.startswith("#") and len(cells) > 3 and not row.startswith("-1;"):
            columns += [int(c) for c in (cells[0], cells[1], cells[3]) if re.fullmatch(r"-?\d+", c)]
    check("adjacencies.csv", columns)
    if not any(r.startswith("-1;-1;") for r in adjacencies):
        problems.append("adjacencies.csv has no -1 end marker, which hangs the loader")
    hubs = (root / "map/supply_nodes.txt").read_text().split()
    check("supply_nodes.txt", [int(p) for p in hubs[1::2]])
    check(
        "railways.txt",
        [
            int(p)
            for line in (root / "map/railways.txt").read_text().splitlines()
            for p in line.split()[2:]
        ],
    )

    stacks = {}
    for row in (root / "map/unitstacks.txt").read_text().splitlines():
        if row.strip():
            cells = row.split(";")
            stacks.setdefault(int(cells[0]), set()).add(int(cells[1]))
    check("unitstacks.txt", stacks)
    bare = [
        p
        for p in sorted(valid)
        if not set(LAND_STACKS if kind[p] == "land" else SEA_STACKS) <= stacks.get(p, set())
    ]
    if bare:
        problems.append(f"{len(bare)} provinces lack counter anchors, first {bare[0]}")
    portless = [
        p
        for p in sorted(valid)
        if kind[p] == "land" and coastal[p] and not set(PORT_STACKS) <= stacks.get(p, set())
    ]
    if portless:
        problems.append(f"{len(portless)} ports lack ship-in-port anchors, first {portless[0]}")

    regions, periods, naval = {}, {}, {}
    for path in sorted((root / "map/strategicregions").glob("*.txt")):
        text = path.read_text()
        region = int(re.search(r"id\s*=\s*(\d+)", text).group(1))
        regions[region] = _block(text, "provinces")
        periods[region] = text.count("period = {")
        naval[region] = "naval_terrain" in text
    placed = [p for listed in regions.values() for p in listed]
    check("strategicregions", placed)
    homeless = sorted(valid - set(placed))
    if homeless:
        problems.append(
            f"{len(homeless)} provinces are in no strategic region, first {homeless[0]}"
        )
    for region, found in sorted(periods.items()):
        if found != len(MONTH_LAST_DAY):
            problems.append(f"strategic region {region} has {found} weather periods, not 12")
    for region, listed in sorted(regions.items()):
        if all(kind.get(p) == "sea" for p in listed) and not naval[region]:
            problems.append(f"sea strategic region {region} has no naval_terrain")
    sizes = {}
    for row in (root / "map/weatherpositions.txt").read_text().splitlines():
        if row.strip():
            cells = row.split(";")
            sizes.setdefault(int(cells[0]), set()).add(cells[4])
    for region in regions:
        if not {"small", "big"} <= sizes.get(region, set()):
            problems.append(f"strategic region {region} lacks a small or big weather object")

    states, owner_points = {}, {}
    for path in sorted((root / "history/states").glob("*.txt")):
        text = path.read_text()
        states[int(re.search(r"id\s*=\s*(\d+)", text).group(1))] = _block(text, "provinces")
        owner = re.search(r"owner\s*=\s*(\w+)", text)
        if owner:
            owner_points.setdefault(owner.group(1), 0)
            owner_points[owner.group(1)] += len(
                re.findall(r"victory_points\s*=\s*\{\s*\d+\s+\d+", text)
            )
    # Per country, not per state: most stock states hold no victory point at all, but a
    # country with none can never be made to capitulate, so the match has no way to end.
    for tag, owned in sorted(owner_points.items()):
        if not owned:
            problems.append(f"{tag} owns no victory point, so it can never capitulate")
    owned = [p for listed in states.values() for p in listed]
    check("history/states", owned)
    stateless = sorted(p for p in valid if kind[p] == "land" and p not in owned)
    if stateless:
        problems.append(f"{len(stateless)} land provinces have no state, first {stateless[0]}")
    wet = sorted(p for p in owned if kind.get(p) != "land")
    if wet:
        problems.append(f"{len(wet)} sea provinces belong to a state, first {wet[0]}")

    models = {}
    for row in (root / "map/buildings.txt").read_text().splitlines():
        if row.strip():
            cells = row.split(";")
            models.setdefault((int(cells[0]), cells[1]), []).append(int(cells[6]))
    for state, listed in sorted(states.items()):
        shore = [p for p in listed if coastal[p]]
        wanted = dict(STATE_BUILDINGS)
        wanted.update(dict.fromkeys(PROVINCE_BUILDINGS, len(listed)))
        wanted.update(dict.fromkeys(COASTAL_BUILDINGS, len(shore)))
        for building, needed in wanted.items():
            if len(models.get((state, building), [])) < needed:
                problems.append(f"state {state} has fewer than {needed} {building} placements")
        for building in PORT_BUILDINGS:
            sea = models.get((state, building), [])
            if any(p not in valid or kind[p] != "sea" for p in sea):
                problems.append(f"state {state} has a {building} with no adjacent sea province")

    shore_land = [p for p in valid if kind[p] == "land" and coastal[p]]
    shore_sea = [p for p in valid if kind[p] == "sea" and coastal[p]]
    if shore_land and not shore_sea:
        problems.append("coastal land provinces exist but no sea province is marked coastal")

    locations = [
        int(found)
        for path in sorted((root / "history/units").glob("*.txt"))
        for found in re.findall(r"location\s*=\s*(\d+)", path.read_text())
    ]
    check("history/units", locations)
    afloat = sorted(p for p in locations if kind.get(p) != "land")
    if afloat:
        problems.append(f"{len(afloat)} divisions start on water, first province {afloat[0]}")

    # Graphics that are a picture of the world rather than a texture.
    provinces = Image.open(root / "map/provinces.bmp").size
    trees = Image.open(root / "map/trees.bmp").size
    wanted = tuple(side * TREES_NUMERATOR // TREES_DENOMINATOR for side in provinces)
    if trees != wanted:
        problems.append(f"trees.bmp is {trees}, but the engine fixes it at 75/256: {wanted}")
    painted = set(np.unique(np.array(Image.open(root / "map/terrain.bmp"))).tolist())
    stray = painted - set(TERRAIN_INDEX.values()) - {OCEAN_INDEX}
    if stray:
        problems.append(f"terrain.bmp paints unintended palette indices {sorted(stray)}")
    heights = np.array(Image.open(root / "map/heightmap.bmp")).astype(np.int16)
    step = max(np.abs(np.diff(heights, axis=0)).max(), np.abs(np.diff(heights, axis=1)).max())
    if step > 48:
        problems.append(f"heightmap has a {step}-byte neighbour step, steeper than any stock coast")
    for name, divisor in [
        ("map/terrain/colormap_rgb_cityemissivemask_a.dds", 2),
        ("map/terrain/fow_rgb_waterspec_a.dds", 2),
        ("map/terrain/colormap_water_0.dds", 2),
        ("map/terrain/colormap_water_1.dds", 4),
        ("map/terrain/colormap_water_2.dds", 8),
    ]:
        path = root / name
        if not path.exists():
            problems.append(f"{name} is missing, so the stock Earth is drawn over the arena")
        elif _dds_size(path) != tuple(side // divisor for side in provinces):
            problems.append(f"{name} is {_dds_size(path)}, not provinces/{divisor}")
    for name, size in MINIMAP_SIZES.items():
        if not (root / name).exists():
            problems.append(f"{name} is missing, so the minimap still shows the stock world")
        elif _dds_size(root / name) != size:
            problems.append(f"{name} is {_dds_size(root / name)}, not {size}")

    # Stock content the arena must unload rather than inherit, because it names ids that
    # only exist on the stock map and the engine resolves them without checking.
    tutorial = root / "tutorial/tutorial.txt"
    text = tutorial.read_text() if tutorial.exists() else ""
    if text.count("tutorial = {") != 1:
        problems.append("tutorial.txt must hold exactly one block: the loader indexes the last")
    if re.search(r"\b(state|target|direction_pointer|highlight_states_trigger)\b", text):
        problems.append("tutorial.txt names a state or province, which the hint loader resolves")

    # Country data the engine needs before it can generate anything for a tag.
    tags = sorted(
        re.findall(r"^(\w+)\s*=", (root / "common/country_tags/00_arena.txt").read_text(), re.M)
    )
    colours = root / "common/countries/colors.txt"
    colour_text = colours.read_text() if colours.exists() else ""
    for tag in tags:
        # Without an rgb-tagged entry here the engine picks its own colour, whatever the
        # country file says, and the side that calls itself Blue is painted green.
        if not re.search(rf"^{tag}\s*=\s*{{[^}}]*color\s*=\s*rgb", colour_text, re.M | re.S):
            problems.append(f"{tag} has no rgb map colour in common/countries/colors.txt")
    names = root / "common/names/01_arena_names.txt"
    characters = root / "common/characters/arena.txt"
    localised = (root / "localisation/english/arena_l_english.yml").read_text(encoding="utf-8-sig")
    for index, tag in enumerate(tags):
        if not names.exists() or not re.search(rf"^{tag}\s*=\s*{{", names.read_text(), re.M):
            problems.append(
                f"{tag} has no character name list, so generated characters are nameless"
            )
        if not characters.exists() or f"{tag}_commander" not in characters.read_text():
            problems.append(f"{tag} has no country leader, forcing the random-character path")
        # A country leader is not a general, and without one no army group can form.
        written = characters.read_text() if characters.exists() else ""
        recruited = (root / f"history/countries/{tag} - Arena.txt").read_text()
        for key, role in [(f"{tag}_marshal", "field_marshal")] + [
            (f"{tag}_general_{n}", "corps_commander") for n in range(1, GENERALS_PER_COUNTRY + 1)
        ]:
            # Split on the block rather than matching inside it: a character carries
            # nested braces for its portraits, so any [^}]* stops at the wrong one.
            parts = written.split(f"\t{key} = {{", 1)
            if len(parts) < 2 or role not in parts[1].split("\n\t}", 1)[0]:
                problems.append(f"{tag} has no {role} named {key}")
            if f"recruit_character = {key}" not in recruited:
                problems.append(f"{key} is defined but never recruited, so it does not exist")
        strategy = root / "common/ai_strategy/arena.txt"
        if not strategy.exists() or f"tag = {tags[1 - index]}" not in strategy.read_text():
            problems.append(f"{tag} has no front_control strategy, so the AI may never attack")
        for suffix in ["", "_DEF", "_ADJ"]:
            if f" {tag}{suffix}:" not in localised:
                problems.append(f"localisation has no {tag}{suffix} key")
    for state, listed in sorted(states.items()):
        text = (root / f"history/states/{state}-arena.txt").read_text()
        for province in re.findall(r"victory_points\s*=\s*\{\s*(\d+)", text):
            if f" VICTORY_POINTS_{province}:" not in localised:
                problems.append(f"victory point {province} has no name, so a stock name shows")
    return {"provinces": count, "states": len(states), "problems": problems}
