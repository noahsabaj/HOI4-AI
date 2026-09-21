"""Generate an original, rotationally mirrored training map. Never read by the policy."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

# Palette indices in the stock terrain.bmp: 19 is plains, 13 forest, 15 ocean. Indices
# 0 and 1 are the blend terrains terrain_0 and terrain_1, so writing those gave every
# land pixel a graphical terrain that the gameplay terrain in definition.csv never names.
TERRAIN_INDEX = {"plains": 19, "forest": 13}
OCEAN_INDEX = 15

# Unit-counter anchors the engine expects for every province, taken from the indices the
# stock unitstacks.txt supplies for all land and for all sea provinces respectively.
LAND_STACKS = (0, 1, 2, 9, 10, 21, 22, 23, 38)
SEA_STACKS = (0, 1, 2, 9, 10, 11, 12, 21, 22, 23, 30, 31, 38)

# Model placements, keyed by state, that the stock buildings.txt supplies for every
# state, every land province and every coastal land province. The port kinds also carry
# the adjacent sea province in the last column.
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


def adjacency(ids, count):
    """Province neighbours as the engine reads them: shared edges in provinces.bmp."""
    neighbours = {i: set() for i in range(1, count + 1)}
    for first, second in [(ids[:-1, :], ids[1:, :]), (ids[:, :-1], ids[:, 1:])]:
        mask = first != second
        for a, b in np.unique(np.stack([first[mask], second[mask]], 1), axis=0).tolist():
            neighbours[a].add(b)
            neighbours[b].add(a)
    return neighbours


def generate(game, output):
    game, root = Path(game), Path(output).resolve()
    if not (game / "map/provinces.bmp").exists():
        raise ValueError("Point --game at the installed HOI4 directory")
    root.mkdir(parents=True, exist_ok=False)

    def write(name, text):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8-sig" if name.endswith(".yml") else "utf8")

    width, height = 2048, 1536
    left = np.array(
        [(x * 128 + 64, y * 128 + 48 + (x % 2) * 32) for x in range(8) for y in range(12)]
    )
    points = np.concatenate([left, [width - 1, height - 1] - left])
    yy, xx = np.mgrid[:height, :width]
    ids = cKDTree(points).query(np.stack([xx.ravel(), yy.ravel()], 1))[1].reshape(height, width) + 1
    land = (
        (points[:, 0] >= 256)
        & (points[:, 0] < width - 256)
        & (points[:, 1] >= 256)
        & (points[:, 1] < height - 256)
    )
    # Break pixel-only four-way contacts, preserving rotational symmetry.
    for _ in range(3):
        a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
        crossing = (a != b) & (a != c) & (a != d) & (b != c) & (b != d) & (c != d)
        for y, x in np.argwhere(crossing):
            if x < width // 2:
                ids[y, x + 1] = ids[y, x]
                mirror = (ids[y, x] + 95) % 192 + 1
                ids[height - 1 - y, width - 2 - x] = mirror
    colors = np.array(
        [[0, 0, 0]]
        + [[(i * 67) % 251 + 1, (i * 101) % 251 + 1, (i * 149) % 251 + 1] for i in range(1, 193)],
        dtype=np.uint8,
    )
    (root / "map").mkdir()
    Image.fromarray(colors[ids]).save(root / "map/provinces.bmp")
    ground = land[ids - 1]
    terrain_types = ["plains" if i % 12 not in (3, 7) else "forest" for i in range(96)] * 2
    terrain_ids = np.array([TERRAIN_INDEX[t] for t in terrain_types], dtype=np.uint8)
    neighbours = adjacency(ids, len(points))
    # A coast is a shared edge between the two classes, so it belongs to both provinces.
    # Marking only the land side left every port facing a sea zone the engine does not
    # consider coastal, and so no sea province to resolve.
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

    indexed("terrain.bmp", np.where(ground, terrain_ids[ids - 1], OCEAN_INDEX))
    indexed("rivers.bmp", np.where(ground, 255, 254))
    Image.fromarray(np.where(ground, 100, 50).astype(np.uint8)).save(root / "map/heightmap.bmp")
    Image.new("RGB", (width // 2, height // 2), (128, 128, 255)).save(root / "map/world_normal.bmp")
    indexed("trees.bmp", np.zeros((height // 4, width // 4), dtype=np.uint8))
    indexed("cities.bmp", np.zeros((height, width), dtype=np.uint8))
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
    for name in ["positions.txt", "adjacency_rules.txt", "ambient_object.txt", "seasons.txt"]:
        write(f"map/{name}", "# Generated arena: no custom entries.\n")
    # The stock adjacencies.csv ends on a comment. The -1 sentinel other Paradox titles
    # use is read here as a real row, and province -1 does not resolve.
    write(
        "map/adjacencies.csv",
        "From;To;Type;Through;start_x;start_y;stop_x;stop_y;adjacency_rule_name;Comment\n"
        "#Generated arena: no manual adjacencies.\n",
    )
    stacks = []
    for i, (x, y) in enumerate(points, 1):
        for slot in LAND_STACKS if land[i - 1] else SEA_STACKS:
            stacks.append(f"{i};{slot};{x}.00;10.00;{height - y}.00;0.00;0.30")
    write("map/unitstacks.txt", "\n".join(stacks) + "\n")
    weather = " ".join(
        f"period = {{ between = {{ 0.{month} {last}.{month} }} "
        f"temperature = {{ 15.0 20.0 }} no_phenomenon = 1.0 }}"
        for month, last in enumerate(MONTH_LAST_DAY)
    )
    for region, mask in [(1, land), (2, ~land)]:
        provinces = " ".join(map(str, np.flatnonzero(mask) + 1))
        write(
            f"map/strategicregions/{region}-arena.txt",
            f'strategic_region = {{ id = {region} name = "ARENA_REGION_{region}" '
            f"provinces = {{ {provinces} }} weather = {{ {weather} }} }}",
        )
    left_land = (np.flatnonzero(land[:96]) + 1).tolist()
    right_land = [i + 96 for i in left_land]
    capitals = []
    for state, tag, province_list in [(1, "BLU", left_land), (2, "RED", right_land)]:
        capital = min(
            province_list,
            key=lambda i: np.linalg.norm(
                points[i - 1] - ([448, 768] if state == 1 else [width - 449, height - 769])
            ),
        )
        capitals.append(capital)
        write(
            f"common/countries/{tag}.txt",
            f"graphical_culture = western_european_gfx\ngraphical_culture_2d = western_european_2d\ncolor = {{ {'40 100 220' if state == 1 else '220 60 60'} }}",
        )
        write(
            f"history/states/{state}-arena.txt",
            f'state = {{ id = {state} name = "ARENA_STATE_{state}" manpower = 1000000 state_category = rural history = {{ owner = {tag} add_core_of = {tag} victory_points = {{ {capital} 50 }} buildings = {{ infrastructure = 4 }} }} provinces = {{ {" ".join(map(str, province_list))} }} }}',
        )
        write(
            f"history/countries/{tag} - Arena.txt",
            f'capital = {state}\noob = "{tag}_1936"\nset_politics = {{ ruling_party = neutrality elections_allowed = no }}\nset_popularities = {{ neutrality = 100 }}\nset_stability = 1\nset_war_support = 1\nset_technology = {{ infantry_weapons = 1 infantry_weapons1 = 1 basic_train = 1 }}\nadd_equipment_to_stockpile = {{ type = infantry_equipment_1 amount = 50000 producer = {tag} }}\nadd_equipment_to_stockpile = {{ type = train_equipment_1 amount = 50 producer = {tag} }}\n',
        )
        front = sorted(province_list, key=lambda i: abs(points[i - 1, 0] - width / 2))[:12]
        regiments = " ".join(
            f"infantry = {{ x = {x} y = {y} }}" for x in range(2) for y in range(3)
        )
        divisions = "\n".join(
            f'division = {{ name = "Infantry {n}" location = {p} division_template = "Arena Infantry" start_experience_factor = 0.3 start_equipment_factor = 1 }}'
            for n, p in enumerate(front, 1)
        )
        write(
            f"history/units/{tag}_1936.txt",
            f'division_template = {{ name = "Arena Infantry" regiments = {{ {regiments} }} }}\nunits = {{ {divisions} }}',
        )
        for sub, size in [("", (82, 52)), ("medium/", (41, 26)), ("small/", (10, 7))]:
            flag = root / f"gfx/flags/{sub}{tag}.tga"
            flag.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", size, (40, 100, 220) if state == 1 else (220, 60, 60)).save(flag)
    # Ordinary supply hubs and rail lines following actual bitmap adjacency.
    rails = [
        f"1 2 {a} {b}"
        for a, sides in sorted(neighbours.items())
        for b in sorted(sides)
        if a < b and land[a - 1] and land[b - 1] and (a <= 96) == (b <= 96)
    ]
    write("map/railways.txt", "\n".join(rails) + "\n")
    hubs = []
    for provinces, capital in zip([left_land, right_land], capitals, strict=True):
        front = min(
            provinces,
            key=lambda i: abs(points[i - 1, 0] - width / 2) + abs(points[i - 1, 1] - height / 2),
        )
        hubs += [f"1 {capital}", f"1 {front}"]
    write("map/supply_nodes.txt", "\n".join(hubs) + "\n")
    # One placement wherever the stock database supplies one: per state slot, per land
    # province and per coastal province. A building the engine can place but has no
    # position for leaves it holding province 0, which is the null province.
    buildings = []
    for state, province_list in [(1, left_land), (2, right_land)]:
        cx, cy = points[capitals[state - 1] - 1]
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
    # Two of each phenomenon size per region, as the stock database supplies.
    write(
        "map/weatherpositions.txt",
        "\n".join(
            f"{region};{x}.00;10.00;{height // 2}.00;{kind}"
            for region in [1, 2]
            for kind in ["small", "big"]
            for x in [width // 4, 3 * width // 4]
        )
        + "\n",
    )
    write(
        "common/national_focus/arena.txt",
        "focus_tree = { id = arena_focus default = yes country = { factor = 1 } "
        "focus = { id = arena_training icon = GFX_goal_generic_army_doctrines "
        "x = 0 y = 0 cost = 1000 completion_reward = { } } }",
    )
    write(
        "common/country_tags/00_arena.txt", 'BLU = "countries/BLU.txt"\nRED = "countries/RED.txt"\n'
    )
    write(
        "common/bookmarks/arena.txt",
        'bookmarks = { bookmark = { name = "ARENA_BOOKMARK" desc = "ARENA_DESC" date = 1936.1.1.12 picture = "GFX_select_date_1936" default_country = "BLU" default = yes BLU = { ideology = neutrality } RED = { ideology = neutrality } } }',
    )
    write(
        "common/on_actions/arena.txt",
        "on_actions = { on_startup = { effect = { BLU = { declare_war_on = { target = RED type = annex_everything } } } } }",
    )
    write(
        "localisation/english/arena_l_english.yml",
        'l_english:\n BLU:0 "Blue"\n BLU_DEF:0 "Blue"\n RED:0 "Red"\n RED_DEF:0 "Red"\n ARENA_BOOKMARK:0 "Infantry Arena"\n ARENA_DESC:0 "Equal infantry armies. Multiple routes. Normal supply and fog of war."\n ARENA_STATE_1:0 "West"\n ARENA_STATE_2:0 "East"\n ARENA_REGION_1:0 "Arena"\n ARENA_REGION_2:0 "Ocean"\n',
    )
    replacements = [
        "history/countries",
        "history/states",
        "history/units",
        "common/bookmarks",
        "common/on_actions",
        "common/national_focus",
        "common/ai_focuses",
        "common/ai_strategy",
        "common/ai_strategy_plans",
        "common/decisions",
        "common/strategic_locations",
        "events",
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
    root.with_suffix(".mod").write_text(descriptor + f'path = "{root.as_posix()}"\n')
    report = {
        "width": width,
        "height": height,
        "countries": 2,
        "land_provinces_per_country": len(left_land),
        "divisions_per_country": 12,
        "coastal_land_provinces": sum(1 for i in neighbours if land[i - 1] and coastal[i]),
        "rotational_mirror": True,
        "gameplay_verified": False,
        "engine_load_verified": False,
    }
    write("generation.json", json.dumps(report, indent=2))
    return report


def _block(text, key):
    found = re.search(key + r"\s*=\s*\{([^}]*)\}", text)
    return [int(value) for value in found.group(1).split()] if found else []


def audit(root):
    """Report references a generated arena asks the engine to resolve and it cannot.

    Every finding is a province the engine looks up and does not find, or a placement it
    looks for and does not have. `CProvinceProvider::GetProvince` returns null below id
    1, and the match-start callers do not check, so an unset id crashes the process
    rather than logging. This reads the written files, so a hand-edited mod is checked
    too, and it is the only check that can run without the game.
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

    def check(name, found):
        outside = sorted({p for p in found if p not in valid})
        if outside:
            problems.append(f"{name} references provinces outside 1..{count}: {outside[:8]}")

    columns = []
    for row in (root / "map/adjacencies.csv").read_text().splitlines()[1:]:
        cells = row.split(";")
        if not row.startswith("#") and len(cells) > 3:
            columns += [int(c) for c in (cells[0], cells[1], cells[3]) if re.fullmatch(r"-?\d+", c)]
    check("adjacencies.csv", columns)
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

    regions, periods = {}, {}
    for path in sorted((root / "map/strategicregions").glob("*.txt")):
        text = path.read_text()
        region = int(re.search(r"id\s*=\s*(\d+)", text).group(1))
        regions[region] = _block(text, "provinces")
        periods[region] = text.count("period = {")
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

    states = {}
    for path in sorted((root / "history/states").glob("*.txt")):
        text = path.read_text()
        states[int(re.search(r"id\s*=\s*(\d+)", text).group(1))] = _block(text, "provinces")
        if not re.search(r"victory_points\s*=\s*\{\s*\d+\s+\d+", text):
            problems.append(f"{path.name} has no victory point province")
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
    return {"provinces": count, "states": len(states), "problems": problems}
