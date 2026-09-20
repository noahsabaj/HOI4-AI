"""Generate an original, rotationally mirrored training map. Never read by the policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


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
    terrain_ids = np.array([0 if t == "plains" else 1 for t in terrain_types], dtype=np.uint8)

    def indexed(name, pixels):
        original = Image.open(game / "map" / name)
        image = Image.fromarray(pixels.astype(np.uint8)).convert("P")
        palette = original.getpalette()
        if palette is not None:
            image.putpalette(palette)
        image.save(root / "map" / name)

    indexed("terrain.bmp", np.where(ground, terrain_ids[ids - 1], 15))
    indexed("rivers.bmp", np.where(ground, 255, 254))
    Image.fromarray(np.where(ground, 100, 50).astype(np.uint8)).save(root / "map/heightmap.bmp")
    Image.new("RGB", (width // 2, height // 2), (128, 128, 255)).save(root / "map/world_normal.bmp")
    indexed("trees.bmp", np.zeros((height // 4, width // 4), dtype=np.uint8))
    indexed("cities.bmp", np.zeros((height, width), dtype=np.uint8))
    definitions = ["0;0;0;0;land;false;unknown;0"]
    for i, (x, y) in enumerate(points, 1):
        is_land = bool(land[i - 1])
        r, g, b = colors[i]
        coast = False
        if is_land:
            mask = ids == i
            coast = any(
                np.any(mask & np.roll(~ground, k, axis)) for k in [-1, 1] for axis in [0, 1]
            )
        definitions.append(
            f"{i};{r};{g};{b};{'land' if is_land else 'sea'};{str(coast).lower()};{terrain_types[i - 1] if is_land else 'ocean'};{1 if is_land else 0}"
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
    for name in [
        "positions.txt",
        "adjacency_rules.txt",
        "ambient_object.txt",
        "seasons.txt",
        "buildings.txt",
        "airports.txt",
        "rocketsites.txt",
        "weatherpositions.txt",
    ]:
        write(f"map/{name}", "# Generated arena: no custom entries.\n")
    write(
        "map/adjacencies.csv",
        "From;To;Type;Through;start_x;start_y;stop_x;stop_y;adjacency_rule_name;Comment\n-1;-1;;-1;-1;-1;-1;-1;;\n",
    )
    write(
        "map/unitstacks.txt",
        "\n".join(
            f"{i};0;{x}.00;10.00;{height - y}.00;0.00;0.30"
            for i, (x, y) in enumerate(points, 1)
            if land[i - 1]
        ),
    )
    for region, mask in [(1, land), (2, ~land)]:
        provinces = " ".join(map(str, np.flatnonzero(mask) + 1))
        write(
            f"map/strategicregions/{region}-arena.txt",
            f'strategic_region = {{ id = {region} name = "ARENA_REGION_{region}" provinces = {{ {provinces} }} weather = {{ period = {{ between = {{ 0.0 30.11 }} temperature = {{ 15.0 20.0 }} no_phenomenon = 1.0 }} }} }}',
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
    edges = set()
    for first, second in [(ids[:-1, :], ids[1:, :]), (ids[:, :-1], ids[:, 1:])]:
        mask = first != second
        edges.update(
            tuple(sorted(pair))
            for pair in np.unique(np.stack([first[mask], second[mask]], 1), axis=0).tolist()
        )
    rails = [
        f"1 2 {a} {b}"
        for a, b in sorted(edges)
        if land[a - 1] and land[b - 1] and (a <= 96) == (b <= 96)
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
    # Supply hubs require actual model placements on the replacement map.
    # Never leave the building-position database entirely empty.
    buildings = []
    for state, capital in enumerate(capitals, 1):
        x, y = points[capital - 1]
        for kind in ["supply_node", "industrial_complex", "arms_factory"]:
            buildings.append(f"{state};{kind};{x}.00;10.00;{height - y}.00;0.00;0")
    write("map/buildings.txt", "\n".join(buildings) + "\n")
    write(
        "map/weatherpositions.txt",
        "\n".join(
            f"{region};1024.00;10.00;768.00;{kind}"
            for region in [1, 2]
            for kind in ["small", "big"]
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
        "rotational_mirror": True,
        "gameplay_verified": False,
        "engine_load_verified": False,
    }
    write("generation.json", json.dumps(report, indent=2))
    return report
