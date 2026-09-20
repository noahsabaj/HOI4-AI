"""Find and read HOI4 unit counters on a map frame (numpy/Pillow, no OpenCV, no network).

Everything geometric lives in ``CounterStyle``. Its defaults were MEASURED with numpy on
``tests/data/arena/sample_map_600x400.png`` (two own counters showing 5 and 7):

    frame 53x22 px, 1 px border in the relation colour (top edge #3f8b4c, darkening downward)
    icon box   x 1..30,  y 1..12   neutral dark grey with the unit-type icon
    org bar    x 14..30, y 13..15  #69c373 (upper, green)
    str bar    x 14..30, y 17..19  #c3914b (lower, orange)  -- "strength" is the architecture
                                   doc's reading; confirming it is not supply is an open question
    flag       x 3..12,  y 14..20
    count plate x 32..52, y 1..20  dark relation colour, pure white 5x8 px digits at x 41..45, y 7..14
    army plate x 53..61            army colour + insignia, present only for units in an army

Detection is scale-free: a counter is anchored on its top border, a long horizontal run of
the relation hue, and every inner box is a fraction of that run. So one style serves any
resolution or UI scale; ``CounterStyle.scale`` only drives the synthetic renderer.

Honest limits, all UNVERIFIED against the live game:
- Own (green), enemy (red #b23b3b) and "other" (blue #3d7fb8: allied or another relation)
  frames are measured, the last two on ``sample_front_1680x1050.png``. A unit outside an army
  has a 62 px frame (the frame colour runs on where the army plate would be, with status icons
  drawn inside); both layouts share the inner geometry.
- Bar VALUES are unverified: no save file accompanies the samples. What is measured is that
  the empty part of a bar is the dark box background, so "bright column = filled" holds.
- Digits 1, 2, 3, 4, 5, 7 are pixel-exact. 0, 6, 8, 9 are hand-drawn in the same 5x8 style and
  reads that used one are reported with reduced confidence. ``calibrate-arena`` captures real
  ones, which replace the built-ins. No two-digit count has been seen for real.
- Counters partly hidden by another counter or the screen edge are reported ``occluded`` when
  at least the icon box is visible; one hidden from the left or the top is not found at all.
- Naval counters (two text rows, taller) fail the frame test and are deliberately not reported.
- No in-combat mark was identifiable on a counter (HOI4 draws battles as bubbles on the
  province border), so ``in_combat`` is always None here.
- The anchor is the border's exact colour (``edge_tolerance``). Land of the same HUE is handled;
  land within that RGB distance of the border colour itself would hide the counter.
- A selected counter's highlight has not been seen; the executor deselects after each order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ...perception.digits import DELTA_LADDER, GLYPH_PREFIX, glyph_boxes
from ...perception.ncc import match_resized, to_gray_f32
from ...perception.templates import TemplateStore

RGB = tuple[int, int, int]
Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in base pixels, ends exclusive

RELATIONS = ("own", "enemy", "other")
COUNTER_GLYPH_PREFIX = "counter_glyph_"  # file names calibrate-arena writes
MEASURED_DIGITS = frozenset("123457")
# 1-5 are pixel-exact from sample_front_1680x1050.png (each seen on two or more counters) and 5, 7
# from sample_map_600x400.png. 0, 6, 8 and 9 are still GUESSED in the same 5x8 style.
BUILTIN_DIGITS: dict[str, tuple[str, ...]] = {
    "0": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "1": (".#.", "##.", ".#.", ".#.", ".#.", ".#.", ".#.", "###"),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    "3": ("#####", "...#.", "..#..", "..##.", "....#", "....#", "#...#", "####."),
    "4": ("...#.", "...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": (".####", ".#...", ".#...", ".###.", "....#", "....#", "....#", "####."),
    "6": (".###.", "#....", "#....", "####.", "#...#", "#...#", "#...#", ".###."),
    "7": ("#####", "#...#", "....#", "...#.", "...#.", "...#.", "..#..", "..#.."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", "#...#", ".####", "....#", "....#", ".###."),
}
_INK, _PAPER = 255.0, 30.0


@dataclass(frozen=True)
class CounterStyle:
    """Counter geometry in base pixels (scale 1.0 = the measured sample) and its colours."""
    frame_w: float = 53.0  # frame beside an army plate (first sample)
    full_w: float = 62.0  # frame of a unit outside an army: it runs on where the plate would be
    separator_x: float = 31.0  # frame-coloured column between the icon box and the count plate
    frame_h: float = 22.0
    icon_box: Box = (1, 1, 31, 13)
    # Bars are 17 px long. They start at x=14 in the first sample and at x=13 in the second (a
    # one-pixel layout difference between the two captures), so the box spans both and the
    # fill is the lit run over ``bar_length``.
    org_bar: Box = (13, 13, 31, 16)
    strength_bar: Box = (13, 17, 31, 20)
    bar_length: float = 17.0
    flag_box: Box = (3, 14, 13, 21)
    plate_box: Box = (32, 1, 53, 21)
    plate_text_box: Box = (34, 3, 52, 19)
    army_plate_box: Box = (53, 0, 62, 22)
    digit_top: float = 7.0
    digit_h: float = 8.0
    own_frame_rgb: RGB = (0x3F, 0x8B, 0x4C)  # measured
    enemy_frame_rgb: RGB = (0xB2, 0x3B, 0x3B)  # measured on sample_front_1680x1050.png
    other_frame_rgb: RGB = (0x3D, 0x7F, 0xB8)  # measured: blue frames (allied / other relation)
    org_rgb: RGB = (0x69, 0xC3, 0x73)  # measured
    strength_rgb: RGB = (0xC3, 0x91, 0x4B)  # measured
    icon_bg_rgb: RGB = (0x3F, 0x3C, 0x3A)  # measured
    bar_bg_rgb: RGB = (0x1E, 0x1A, 0x17)  # measured around full bars; empty-bar look is a guess
    hue_cos: float = 0.985  # min cosine between a pixel and the frame colour
    min_saturation: float = 0.35
    min_value: float = 18.0
    edge_tolerance: float = 45.0  # RGB distance to the frame colour that still counts as the top border
    plate_dark_value: float = 90.0  # plates peak at 57 (#391211); frame borders start at 139
    ink_value: float = 170.0
    plate_foreign_share: float = 0.15  # anti-aliased glyph edges stay well under this
    bar_lit_value: float = 90.0  # a bar column is filled when its brightest channel reaches this
    min_scale: float = 0.75
    max_scale: float = 3.0
    scale: float = 1.0  # synthetic rendering only; detection measures scale per counter

    def scaled(self, factor: float) -> CounterStyle:
        return replace(self, scale=self.scale * factor)

    def frame_rgb(self, relation: str) -> RGB:
        return {"own": self.own_frame_rgb, "enemy": self.enemy_frame_rgb}.get(relation, self.other_frame_rgb)


@dataclass(frozen=True)
class CounterReading:
    bbox: tuple[int, int, int, int]  # frame only (army plate excluded), image pixels, ends exclusive
    relation: str  # "own" | "enemy" | "other" (blue frame: allied or other relation)
    count: int | None  # None when a digit could not be classified; never a guess
    count_score: float
    count_measured: bool  # False when a hand-drawn (guessed) digit template was used
    organization: float | None
    strength: float | None
    in_combat: bool | None
    has_army_plate: bool
    flag_rgb: RGB
    scale: float
    structure_score: float
    confidence: float
    occluded: bool = False  # only part of the top border was visible (overlap or screen edge)

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "center": self.center}


def digit_template(rows: tuple[str, ...]) -> np.ndarray:
    """Bitmap rows -> grayscale template with a 1 px paper margin.

    The margin keeps a bar-shaped glyph such as "1" from having zero variance, which NCC
    scores as 0 against everything.
    """
    ink = np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)
    return np.pad(np.where(ink, _INK, _PAPER).astype(np.float32), 1, constant_values=_PAPER)


class CounterDigits:
    """Counter-font digit templates: built-ins overlaid by whatever calibration captured."""

    def __init__(self, store: TemplateStore | None = None, measured: frozenset[str] = MEASURED_DIGITS,
                 threshold: float = 0.6) -> None:
        self.store = store or TemplateStore({f"{GLYPH_PREFIX}{d}": digit_template(rows)
                                             for d, rows in BUILTIN_DIGITS.items()})
        self.measured = measured
        self.threshold = threshold

    @classmethod
    def load(cls, directory: str | Path | None, threshold: float = 0.6) -> CounterDigits:
        digits = cls(threshold=threshold)
        measured = set(digits.measured)
        if directory is not None and Path(directory).is_dir():
            for path in sorted(Path(directory).glob(f"{COUNTER_GLYPH_PREFIX}[0-9].png")):
                digit = path.stem[len(COUNTER_GLYPH_PREFIX):]
                gray = to_gray_f32(Image.open(path))
                # Captured glyphs are tight crops of white ink: give them the same 1 px margin of
                # plate (their darkest value) that the built-ins and the read path use.
                digits.store.add(f"{GLYPH_PREFIX}{digit}", np.pad(gray, 1, constant_values=float(gray.min())))
                measured.add(digit)
        digits.measured = frozenset(measured)
        return digits

    def _classify(self, glyph: Image.Image) -> tuple[str | None, float]:
        best: tuple[float, str | None] = (-1.0, None)
        aspect = glyph.width / max(1, glyph.height)
        for name in self.store.names():
            template = self.store.get(name)
            template_aspect = template.shape[1] / template.shape[0]
            if not 0.6 <= aspect / template_aspect <= 1.6:  # a stretched "1" matches anything
                continue
            score = match_resized(glyph, template)
            if score > best[0]:
                best = (score, name[len(GLYPH_PREFIX):])
        return (best[1], best[0]) if best[0] >= self.threshold else (None, max(best[0], 0.0))

    def read(self, plate: Image.Image) -> tuple[int | None, float, bool]:
        """(count, weakest glyph score, all glyphs from measured templates).

        All-or-nothing like ``GlyphReader.read_text``: an unknown glyph makes the read None.
        """
        gray = to_gray_f32(plate)
        worst = 0.0
        for delta in DELTA_LADDER:
            boxes = glyph_boxes(gray, delta)
            if not boxes or len(boxes) > 3:
                continue
            results = []
            for x0, y0, x1, y1 in boxes:
                padded = (max(0, x0 - 1), max(0, y0 - 1), min(plate.width, x1 + 1), min(plate.height, y1 + 1))
                results.append(self._classify(plate.crop(padded)))
            worst = max(worst, min(score for _, score in results))
            if all(char is not None for char, _ in results):
                text = "".join(char for char, _ in results if char is not None)
                if int(text) >= 1:
                    return (int(text), min(score for _, score in results),
                            all(char in self.measured for char in text))
        return None, worst, False


def hue_mask(rgb: np.ndarray, reference: RGB, style: CounterStyle) -> np.ndarray:
    """Pixels sharing the reference colour's hue at any brightness (the frame darkens downward).

    cos(pixel, reference) > hue_cos, written without a square root.
    """
    channels = np.asarray(rgb).astype(np.int32)
    red, green, blue = channels[..., 0], channels[..., 1], channels[..., 2]
    dot = (red * reference[0] + green * reference[1] + blue * reference[2]).astype(np.float32)
    norm2 = (red * red + green * green + blue * blue).astype(np.float32)
    limit = np.float32(style.hue_cos ** 2 * sum(v * v for v in reference))
    high = np.maximum(np.maximum(red, green), blue)
    low = np.minimum(np.minimum(red, green), blue)
    return ((dot * dot > limit * norm2) & ((high - low) > style.min_saturation * high)
            & (high >= style.min_value))


def edge_mask(rgb: np.ndarray, reference: RGB, style: CounterStyle) -> np.ndarray:
    """Pixels close to the frame's TOP-BORDER colour itself, not merely its hue.

    The anchor has to be this strict: a counter standing on land of its own hue (a red
    frame on a red country) would otherwise fuse its border into one long background run.
    A per-channel uint8 box test runs first so the exact distance is only computed on the
    few pixels that survive it; this is the only full-frame pass of the detector.
    """
    pixels = np.asarray(rgb, dtype=np.uint8)
    reach = int(style.edge_tolerance)
    near = np.ones(pixels.shape[:2], dtype=bool)
    for channel, value in enumerate(reference):
        plane = pixels[..., channel]
        near &= (plane >= max(0, value - reach)) & (plane <= min(255, value + reach))
    rows, columns = np.nonzero(near)
    delta = pixels[rows, columns].astype(np.float32) - np.asarray(reference, dtype=np.float32)
    near[rows, columns] = (delta * delta).sum(axis=1) < style.edge_tolerance ** 2
    return near


def _row_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int, int]]:
    """All horizontal True runs of at least ``min_len`` as (y, x0, x1), top to bottom."""
    padded = np.zeros((mask.shape[0], mask.shape[1] + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    edges = np.diff(padded, axis=1)
    start_y, start_x = np.nonzero(edges == 1)
    end_x = np.nonzero(edges == -1)[1]  # row-major order pairs the nth start with the nth end
    keep = (end_x - start_x) >= min_len
    return [(int(y), int(x0), int(x1)) for y, x0, x1 in zip(start_y[keep], start_x[keep], end_x[keep])]


def _box(x: int, y: int, scale: float, box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0 = x + int(round(box[0] * scale)), y + int(round(box[1] * scale))
    x1, y1 = x + int(round(box[2] * scale)), y + int(round(box[3] * scale))
    x0, y0 = min(max(x0, 0), width - 1), min(max(y0, 0), height - 1)
    return x0, y0, min(max(x1, x0 + 1), width), min(max(y1, y0 + 1), height)


def _mean(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    region = mask[box[1]:box[3], box[0]:box[2]]
    return float(region.mean()) if region.size else 0.0


def _pixels(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    return rgb[box[1]:box[3], box[0]:box[2]].reshape(-1, 3)


def bar_fill(rgb: np.ndarray, box: tuple[int, int, int, int], style: CounterStyle, scale: float = 1.0
             ) -> float | None:
    """Filled fraction of a bar: the lit run from its left end over the bar's full length.

    The empty part is the box's dark neutral background (measured #292522..#1e1b19 on partly
    filled bars in sample_front_1680x1050.png), so a column is filled when it is bright. The run
    may start up to two base pixels into the box, which covers both observed bar positions.
    """
    region = rgb[box[1]:box[3], box[0]:box[2]]
    if region.size == 0:
        return None
    lit = region.max(axis=2).mean(axis=0) >= style.bar_lit_value
    run = 0  # the longest run that starts in the window: a blurred flag edge can light column 0
    for start in range(min(lit.size, int(np.ceil(2 * scale)) + 1)):
        if lit[start] and (start == 0 or not lit[start - 1]):
            dark = np.nonzero(~lit[start:])[0]
            run = max(run, int(dark[0]) if dark.size else lit.size - start)
    return float(min(1.0, run / max(1.0, round(style.bar_length * scale))))


def plate_is_clean(rgb: np.ndarray, box: tuple[int, int, int, int], style: CounterStyle) -> bool:
    """A count plate holds only dark plate and white ink. Bright colour inside it means another
    counter is drawn over the plate, and reading digits there would report that counter's."""
    region = rgb[box[1]:box[3], box[0]:box[2]].astype(np.int16)
    if region.size == 0:
        return False
    foreign = (region.max(axis=2) >= style.plate_dark_value) & (region.min(axis=2) < style.ink_value)
    return float(foreign.mean()) <= style.plate_foreign_share


def _covered(mask: np.ndarray, x0: int, y: int, scale: float, style: CounterStyle) -> int:
    """Rows below the top border, on the icon-box side, that still look like this counter.

    Used for a full-length top border whose body is partly hidden by a counter drawn over it.
    """
    height, width = mask.shape
    inner = slice(min(width - 1, x0 + int(round(2 * scale))), min(width, x0 + int(round(29 * scale))))
    rows = 0
    while (y + rows + 1 < height and rows < int(round(style.frame_h * scale)) and mask[y + rows + 1, x0]
           and float(mask[y + rows + 1, inner].mean()) < 0.3):
        rows += 1
    return rows


def _refine_left(mask: np.ndarray, edge: np.ndarray, y: int, x0: int, rough: float) -> int:
    """The corner pixel is darker than the border and can miss the strict edge mask, so the left
    column is re-found from inside: walk left out of the icon box until the border's hue."""
    height, width = mask.shape
    probe_y, column = min(height - 1, y + int(round(3 * rough))), min(width - 1, x0 + int(round(3 * rough)))
    if mask[probe_y, column]:
        return x0  # not inside an icon box (something is drawn over this counter): keep the run's start
    while column > max(0, x0 - 3) and not mask[probe_y, column]:
        column -= 1
    # A scaled border is several pixels thick: take its outer column. The strict mask keeps this
    # from running into same-hue land left of the counter.
    floor = max(0, x0 - int(np.ceil(rough)) - 1)
    while column > floor and edge[probe_y, column - 1]:
        column -= 1
    return column if mask[probe_y, column] and abs(column - x0) <= 2 * rough + 1 else x0


def _structure(mask: np.ndarray, x0: int, y: int, scale: float, visible: int,
               style: CounterStyle) -> tuple[int, float] | None:
    """(bottom row, score) when a counter of this scale has its top-left corner at (x0, y).

    ``visible`` is how many columns of the top border were seen; a counter drawn over this one
    hides the rest, so only the visible part is tested. The separator test is what tells the two
    frame layouts apart: the column at 31 must be frame-coloured and the one before it must not.
    """
    height, width = mask.shape
    if not style.min_scale <= scale <= style.max_scale:
        return None
    expected = y + int(round(style.frame_h * scale)) - 1
    slack = max(1, int(round(0.1 * style.frame_h * scale)))
    span = x0 + max(4, min(visible, int(round(style.separator_x * scale))))
    bottom = next((row for row in range(max(y + 2, expected - slack), min(height, expected + slack + 1))
                   if float(mask[row, x0:span].mean()) >= 0.85), None)
    if bottom is None:
        return None
    scores = [float(mask[y:bottom + 1, x0].mean())]
    icon = _box(x0, y, scale, style.icon_box, min(width, x0 + visible), height)
    scores.append(1.0 - _mean(mask, icon))
    if scores[0] < 0.7 or scores[1] < 0.5:
        return None
    rows = slice(y + int(round(2 * scale)), y + max(int(round(2 * scale)) + 1, int(round(12 * scale))))
    sep_x = x0 + int(round((style.separator_x + 0.5) * scale - 0.5))
    if sep_x + 1 < min(width, x0 + visible):
        before = x0 + int(round((style.separator_x - 1.5) * scale))
        separator = float(mask[rows, max(x0, sep_x - 1):sep_x + 2].mean(axis=0).max())
        if separator < 0.7 or float(mask[rows, before].mean()) > 0.3:
            return None
        scores.append(separator)
    if visible >= int(style.frame_w * scale) - 1:
        plate = _mean(mask, _box(x0, y, scale, style.plate_box, width, height))
        if plate < 0.5:
            return None
        scores.append(plate)
    return bottom, float(np.mean(scores))


def _read(pil: Image.Image, rgb: np.ndarray, relation: str, x0: int, y: int, x1: int, bottom: int,
          scale: float, structure: float, occluded: bool, style: CounterStyle,
          digits: CounterDigits, bars: bool = True) -> CounterReading:
    height, width = rgb.shape[:2]
    count, score, measured = None, 0.0, False
    text_box = _box(x0, y, scale, style.plate_text_box, width, height)
    if x1 - x0 >= int(style.frame_w * scale) - 1 and plate_is_clean(rgb, text_box, style):
        count, score, measured = digits.read(pil.crop(text_box))
    organization = strength = None
    if bars:
        organization = bar_fill(rgb, _box(x0, y, scale, style.org_bar, width, height), style, scale)
        strength = bar_fill(rgb, _box(x0, y, scale, style.strength_bar, width, height), style, scale)
    # An army plate is a differently coloured block right of a 53-wide frame; without an army the
    # frame itself runs on to 62 and status icons are drawn inside that extension.
    has_plate = not occluded and (x1 - x0) < (style.frame_w + style.full_w) / 2 * scale
    flag = _pixels(rgb, _box(x0, y, scale, style.flag_box, width, height)).mean(axis=0)
    digit_confidence = 0.0 if count is None else score * (1.0 if measured else 0.6)
    confidence = float(np.clip(structure * (0.5 + 0.5 * digit_confidence) * (0.8 if occluded else 1.0), 0, 1))
    return CounterReading(
        (x0, y, x1, bottom + 1), relation, count, round(score, 4), measured, organization, strength,
        None, has_plate, (int(flag[0]), int(flag[1]), int(flag[2])), round(scale, 3),
        round(structure, 4), round(confidence, 4), occluded)


def find_counters(image: Image.Image | np.ndarray, style: CounterStyle | None = None,
                  digits: CounterDigits | None = None) -> list[CounterReading]:
    """Every land counter on the frame, top-to-bottom then left-to-right.

    Pass 1 accepts top-border runs that are exactly one counter long in either layout (53 wide
    beside an army plate, 62 wide without one) and so measures the scale. Pass 2 revisits the
    shorter runs at that scale: counters partly hidden by one drawn over them, or cut by the
    screen edge. Those are reported ``occluded`` and read a count only if the plate is visible.
    Naval counters are two text rows tall and fail the bottom-row test, so they are not reported.
    """
    style = style or CounterStyle()
    digits = digits or CounterDigits()
    pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8))
    pil = pil.convert("RGB")
    rgb = np.asarray(pil)
    height, width = rgb.shape[:2]
    found: list[CounterReading] = []
    leftovers: list[tuple[str, np.ndarray, tuple[int, int, int]]] = []

    def claimed(x0: int, y: int, x1: int, bottom: int) -> bool:
        return any(x0 < c.bbox[2] and c.bbox[0] < x1 and y < c.bbox[3] and c.bbox[1] <= bottom
                   and abs(c.bbox[1] - y) < 3 for c in found)

    def window(y: int, x0: int, x1: int, reach: float) -> tuple[int, int, int, int]:
        return max(0, x0 - 6), max(0, y - 1), min(width, x1 + 6), min(height, y + int(reach) + 6)

    for relation in RELATIONS:
        reference = style.frame_rgb(relation)
        edge = edge_mask(rgb, reference, style)
        for y, run_x0, x1 in _row_runs(edge, int(style.separator_x * style.min_scale)):
            if y > 0 and float(edge[y - 1, run_x0:x1].mean()) >= 0.5:
                continue  # not an edge: the border colour continues upward
            wx0, wy0, wx1, wy1 = window(y, run_x0, x1, 0.6 * (x1 - run_x0))
            mask, local = hue_mask(rgb[wy0:wy1, wx0:wx1], reference, style), edge[wy0:wy1, wx0:wx1]
            x0 = _refine_left(mask, local, y - wy0, run_x0 - wx0, (x1 - run_x0) / style.full_w) + wx0
            fits = [(fit, (x1 - x0) / layout) for layout in (style.full_w, style.frame_w)
                    if (fit := _structure(mask, x0 - wx0, y - wy0, (x1 - x0) / layout, x1 - x0, style))]
            if not fits:
                leftovers.append((relation, edge, (y, run_x0, x1)))
                continue
            (bottom, structure), scale = max(fits, key=lambda item: item[0][1])
            if not claimed(x0, y, x1, bottom + wy0):
                found.append(_read(pil, rgb, relation, x0, y, x1, bottom + wy0, scale, structure, False,
                                   style, digits))
    if found:
        scale = float(np.median([c.scale for c in found]))
        body = int(round(style.frame_h * scale))
        for relation, edge, (y, run_x0, x1) in leftovers:
            if x1 - run_x0 > style.full_w * scale + 2:
                continue
            wx0, wy0, wx1, wy1 = window(y, run_x0, x1, style.frame_h * scale)
            mask = hue_mask(rgb[wy0:wy1, wx0:wx1], style.frame_rgb(relation), style)
            x0 = _refine_left(mask, edge[wy0:wy1, wx0:wx1], y - wy0, run_x0 - wx0, scale) + wx0
            if claimed(x0, y, x1, y + body - 1):
                continue
            fit = _structure(mask, x0 - wx0, y - wy0, scale, x1 - x0, style)
            if fit is not None:  # the right part is hidden; bars need the whole icon side
                found.append(_read(pil, rgb, relation, x0, y, x1, fit[0] + wy0, scale, fit[1], True, style,
                                   digits, bars=x1 - x0 >= style.separator_x * scale))
                continue
            whole = any(abs((x1 - x0) - layout * scale) <= 2 for layout in (style.full_w, style.frame_w))
            rows = _covered(mask, x0 - wx0, y - wy0, scale, style) if whole else 0
            if rows < 2:
                continue
            seen = 0.5 + 0.5 * min(1.0, rows / body)
            reading = _read(pil, rgb, relation, x0, y, x1, y + body - 1, scale, seen, True, style, digits,
                            bars=rows >= int(round(20 * scale)))
            if rows >= 3 or reading.count is not None:  # two rows alone are not evidence enough
                found.append(reading)
    return sorted(found, key=lambda c: (c.bbox[1], c.bbox[0]))
