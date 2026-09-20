"""Preview PNGs of a generated arena map, so a human can follow along without launching the game.

These are DRAWINGS OF THE GENERATOR'S DATA (province raster, layout, supply plan, orders of battle), not
screenshots: they show what was written, not what the game renders. ``preview_provinces_raw.png`` is the
one exception that is read back from disk (the real ``provinces.bmp``, scaled down).
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..contracts import Country
from . import bmpio
from .generate import ARMOR, VARIANTS, StatePlan, placements
from .mapdesign import CustomMap
from .region import Region

WIDTH = 1600
START_VARIANTS = ("inf4_line", "inf12_line", "mix12_mass_south", "inf12_reserve")
SIDE_FILL = {"BLU": {"north": (150, 190, 250), "center": (110, 160, 240), "south": (80, 130, 225)},
             "RED": {"north": (250, 170, 160), "center": (240, 130, 120), "south": (225, 100, 95)}}
TERRAIN_FILL = {"plains": (176, 204, 120), "forest": (70, 130, 70), "hills": (190, 170, 110),
                "mountain": (150, 140, 135), "marsh": (120, 170, 160), "lakes": (90, 140, 200), "ocean": (28, 52, 96)}
OCEAN, BLOCKED, BOX = (28, 52, 96), (92, 88, 84), (200, 200, 120)


def _font(size: int) -> Any:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


class _Canvas:
    """The arena cropped to its land (plus a margin), scaled to ``WIDTH`` pixels."""

    def __init__(self, arena: CustomMap, fill: dict[int, tuple[int, int, int]], title: str) -> None:
        land = [p for p in arena.provinces if p.kind != "sea"]
        height, width = arena.ids.shape
        pad = max(width // 40, 8)
        self.x0 = max(min(p.bbox[0] for p in land) - pad, 0)
        self.y0 = max(min(p.bbox[1] for p in land) - pad, 0)
        x1 = min(max(p.bbox[2] for p in land) + pad, width)
        y1 = min(max(p.bbox[3] for p in land) + pad, height)
        self.scale = WIDTH / (x1 - self.x0)
        crop = arena.ids[self.y0:y1, self.x0:x1]
        table = np.zeros((len(arena.provinces) + 1, 3), dtype=np.uint8)
        for province_id, color in fill.items():
            table[province_id] = color
        rgb = table[crop]
        edge = np.zeros(crop.shape, dtype=bool)
        edge[:, :-1] |= crop[:, :-1] != crop[:, 1:]
        edge[:-1] |= crop[:-1] != crop[1:]
        thick = edge.copy()
        thick[:, 1:] |= edge[:, :-1]
        thick[1:] |= edge[:-1]
        rgb[thick] = (rgb[thick] * 0.35).astype(np.uint8)
        size = (WIDTH, round((y1 - self.y0) * self.scale))
        self.image = Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS)
        self.draw = ImageDraw.Draw(self.image)
        self.arena = arena
        self.font, self.small = _font(max(11, round(15 * self.scale))), _font(max(10, round(11 * self.scale)))
        self.draw.rectangle((0, 0, WIDTH, 26), fill=(0, 0, 0))
        self.draw.text((8, 4), title, fill=(255, 255, 255), font=_font(16))

    def at(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x0) * self.scale, (y - self.y0) * self.scale

    def center(self, province_id: int) -> tuple[float, float]:
        p = self.arena.province(province_id)
        return self.at(p.cx + 0.5, p.cy + 0.5)

    def label(self, xy: tuple[float, float], text: str, fill: tuple[int, int, int] = (0, 0, 0), font: Any = None,
              box: tuple[int, int, int] | None = None) -> None:
        font = font or self.font
        left, top, right, bottom = self.draw.textbbox((0, 0), text, font=font)
        x, y = xy[0] - (right - left) / 2, xy[1] - (bottom - top) / 2
        if box:
            self.draw.rectangle((x - 3, y - 1, x + right - left + 3, y + bottom - top + 4), fill=box, outline=(0, 0, 0))
        self.draw.text((x, y - top), text, fill=fill, font=font)

    def star(self, xy: tuple[float, float], radius: float, fill: tuple[int, int, int]) -> None:
        points = []
        for k in range(10):
            r = radius if k % 2 == 0 else radius * 0.45
            angle = -math.pi / 2 + k * math.pi / 5
            points.append((xy[0] + r * math.cos(angle), xy[1] + r * math.sin(angle)))
        self.draw.polygon(points, fill=fill, outline=(0, 0, 0))

    def save(self, path: Path) -> Path:
        self.image.save(path, format="PNG", optimize=False)
        return path


def _owner_fill(arena: CustomMap, pale: bool = False) -> dict[int, tuple[int, int, int]]:
    fill: dict[int, tuple[int, int, int]] = {}
    for p in arena.provinces:
        color = (OCEAN if p.kind == "sea" else BLOCKED if p.kind == "blocked" else BOX if p.kind == "box"
                 else SIDE_FILL[p.side][p.sector])
        r, g, b = color
        fill[p.id] = ((r + 255) // 2, (g + 255) // 2, (b + 255) // 2) if pale and p.kind == "play" else color
    return fill


def _political(arena: CustomMap, title: str, pale: bool = False) -> _Canvas:
    canvas = _Canvas(arena, _owner_fill(arena, pale), title)
    points = dict(arena.victory_points)
    capitals = set(arena.capitals.values())
    for p in arena.of_kind("play"):
        x, y = canvas.center(p.id)
        if p.id in capitals:
            canvas.star((x, y - 16 * canvas.scale - 8), 13, (255, 215, 0))
        elif p.id in points:
            canvas.draw.ellipse((x - 9, y - 16 * canvas.scale - 17, x + 9, y - 16 * canvas.scale + 1), fill=(255, 215, 0),
                                outline=(0, 0, 0))
        if p.id in points:
            offset = 22 if p.id in capitals else 0
            canvas.label((x + offset, y - 16 * canvas.scale - 8), f"{points[p.id]:g}", font=canvas.small)
        canvas.label((x, y), str(p.id))
    return canvas


def write_previews(arena: CustomMap, plans: tuple[StatePlan, ...], region: Region, output: Path) -> list[Path]:
    design = arena.design
    written = []
    head = f"{design.name} ({design.symmetry}, {len(arena.of_kind('play'))} playable provinces)"
    canvas = _political(arena, f"{head}: owners, province ids, capitals (star), victory points, sectors by shade. "
                               "Generator data, not a screenshot.")
    height = arena.ids.shape[0]
    for index, cut in enumerate(arena.sector_cuts):
        _, y = canvas.at(0, cut * height)
        canvas.draw.line((0, y, WIDTH, y), fill=(255, 255, 255), width=1)
        canvas.label((60, y + (-12 if index else 12)), "center", fill=(255, 255, 255), font=canvas.small)
    canvas.label((60, canvas.at(0, arena.sector_cuts[0] * height)[1] - 12), "north", fill=(255, 255, 255), font=canvas.small)
    canvas.label((60, canvas.at(0, arena.sector_cuts[1] * height)[1] + 12), "south", fill=(255, 255, 255), font=canvas.small)
    for number, route in enumerate(arena.routes(), start=1):
        xs, ys = zip(*(canvas.center(p) for p in route), strict=True)
        canvas.label((sum(xs) / len(xs), max(ys) + 30 * canvas.scale + 10), f"route {number}", fill=(255, 255, 255),
                     font=canvas.small, box=(40, 40, 40))
    written.append(canvas.save(output / "preview_political.png"))

    fill = {p.id: TERRAIN_FILL[p.terrain] if p.kind != "blocked" else BLOCKED for p in arena.provinces}
    canvas = _Canvas(arena, fill, f"{head}: terrain (definition.csv), rivers (cyan), river-crossing links (dashed). "
                                  "Green plains, dark forest, tan hills, grey mountain/blocked.")
    wet_y, wet_x = np.nonzero(arena.rivers <= 11)
    for x, y in zip(wet_x.tolist(), wet_y.tolist(), strict=True):
        px, py = canvas.at(x + 0.5, y + 0.5)
        source = arena.rivers[y, x] == 0
        canvas.draw.rectangle((px - 2, py - 2, px + 2, py + 2), fill=(0, 255, 0) if source else (0, 230, 255))
    for a, others in arena.river_neighbors.items():
        for b in others:
            if a < b:
                (x0, y0), (x1, y1) = canvas.center(a), canvas.center(b)
                for k in range(0, 10, 2):
                    canvas.draw.line((x0 + (x1 - x0) * k / 10, y0 + (y1 - y0) * k / 10, x0 + (x1 - x0) * (k + 1) / 10,
                                      y0 + (y1 - y0) * (k + 1) / 10), fill=(200, 0, 60), width=3)
    for p in arena.of_kind("play"):
        canvas.label(canvas.center(p.id), f"{p.id} {p.terrain[:2]}", font=canvas.small)
    written.append(canvas.save(output / "preview_terrain_rivers.png"))

    canvas = _political(arena, f"{head}: supply hubs (white squares), railways (black), capitals (star).", pale=True)
    for rail in arena.railways:
        canvas.draw.line([canvas.center(p) for p in rail], fill=(0, 0, 0), width=4)
    for hub in arena.hubs:
        hx, hy = canvas.center(hub)
        canvas.draw.rectangle((hx - 9, hy + 10, hx + 9, hy + 28), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        canvas.label((hx, hy + 19), "H", font=canvas.small)
    written.append(canvas.save(output / "preview_supply.png"))

    for variant in (v for v in VARIANTS if v.id in START_VARIANTS):
        canvas = _political(arena, f"{head}: start positions of scenario {variant.id} ({variant.divisions} divisions, "
                                   f"{variant.armor} armour, even handicap). I = infantry, A = armour.", pale=True)
        for country in Country:
            counts: Counter[tuple[int, str]] = Counter(placements(region, variant, country))
            where: dict[int, list[str]] = {}
            for (province, template), count in sorted(counts.items()):
                where.setdefault(province, []).append(f"{count}{'A' if template == ARMOR else 'I'}")
            color = (30, 70, 200) if country is Country.BLUE else (200, 30, 30)
            for province, parts in where.items():
                ux, uy = canvas.center(province)
                canvas.label((ux, uy + 20), "+".join(parts), fill=(255, 255, 255), box=color)
        written.append(canvas.save(output / f"preview_start_positions_{variant.id}.png"))

    folder = next(iter(sorted((output / "mod").glob("*/map"))), None)
    if folder is not None and (folder / "provinces.bmp").is_file():
        raw = Image.fromarray(bmpio.read_bmp(folder / "provinces.bmp").pixels)
        raw = raw.resize((WIDTH, WIDTH * raw.height // raw.width), Image.Resampling.NEAREST)
        raw.save(output / "preview_provinces_raw.png", format="PNG")
        written.append(output / "preview_provinces_raw.png")
    del plans
    return written


def contact_sheet(root: Path) -> Path | None:
    """``preview_all_designs.png``: the political preview of every design built under ``root``."""
    tiles = [(path.parent.name, path) for path in sorted(root.glob("*/preview_political.png"))]
    if not tiles:
        return None
    tile_w, columns = 800, 2
    images = []
    for _, path in tiles:
        with Image.open(path) as handle:
            images.append(handle.convert("RGB").resize((tile_w, tile_w * handle.height // handle.width),
                                                       Image.Resampling.LANCZOS))
    tile_h = max(image.height for image in images) + 30
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, ((name, _), image) in enumerate(zip(tiles, images, strict=True)):
        x, y = index % columns * tile_w, index // columns * tile_h
        sheet.paste(image, (x, y + 30))
        draw.text((x + 8, y + 5), name, fill=(255, 255, 255), font=_font(18))
    path = root / "preview_all_designs.png"
    sheet.save(path, format="PNG")
    return path
