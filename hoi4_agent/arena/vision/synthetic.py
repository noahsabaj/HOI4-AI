"""SYNTHETIC arena frames. Not HOI4 output and never evidence about the real game.

Draws a fake frame from a layout and a unit list using the geometry and colours of
``CounterStyle`` (measured on the real sample for own counters, guessed for the rest), so
tests can run render -> perceive and the simulator can be pushed through real perception
code. Every frame is stamped "SYNTHETIC". A passing round trip proves the reader inverts
this renderer; how well it reads the live game is a separate, unverified question.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from ...perception.templates import TemplateStore
from ..contracts import Country
from ..layout import ArenaLayout
from .calibration import ArenaVisionCalibration
from .counters import BUILTIN_DIGITS, RGB, Box, CounterStyle

BACKGROUND: RGB = (74, 80, 88)
TOP_BAR: RGB = (28, 28, 30)
_FRAME_BOTTOM, _PLATE = 0.48, 0.31  # measured brightness of the frame's last row / count plate vs its top


@dataclass(frozen=True)
class SyntheticStack:
    province_id: int
    relation: str  # "own" | "enemy" | "other"
    count: int
    organization: float
    strength: float
    army_plate: bool = True


def _shade(color: RGB, factor: float) -> RGB:
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))


def _rect(x: int, y: int, scale: float, box: Box) -> tuple[int, int, int, int]:
    """Same rounding as ``counters._box`` so rendered and read boxes coincide."""
    return (x + int(round(box[0] * scale)), y + int(round(box[1] * scale)),
            x + int(round(box[2] * scale)), y + int(round(box[3] * scale)))


def _fill(pixels: np.ndarray, rect: tuple[int, int, int, int], color: RGB) -> None:
    pixels[max(0, rect[1]):max(0, rect[3]), max(0, rect[0]):max(0, rect[2])] = color


def draw_counter(pixels: np.ndarray, x: int, y: int, stack: SyntheticStack, style: CounterStyle) -> None:
    """Paint one counter with its frame's top-left corner at (x, y)."""
    scale = style.scale
    frame = style.frame_rgb(stack.relation)
    # Beside an army plate the frame is 53 wide; without one it runs on to 62 (both measured).
    layout = style.frame_w if stack.army_plate else style.full_w
    width, height = int(round(layout * scale)), int(round(style.frame_h * scale))
    for row in range(height):  # the border colour darkens downward, as measured
        factor = 1.0 - (1.0 - _FRAME_BOTTOM) * row / max(1, height - 1)
        pixels[y + row, x:x + width] = _shade(frame, factor)
    _fill(pixels, _rect(x, y, scale, (1, 1, 31, style.frame_h - 1)), style.bar_bg_rgb)
    _fill(pixels, _rect(x, y, scale, style.icon_box), style.icon_bg_rgb)
    _fill(pixels, _rect(x, y, scale, style.plate_box), _shade(frame, _PLATE))
    for box, color, value in ((style.org_bar, style.org_rgb, stack.organization),
                              (style.strength_bar, style.strength_rgb, stack.strength)):
        x0, y0, _, y1 = _rect(x, y, scale, box)
        _fill(pixels, (x0, y0, x0 + int(round(value * round(style.bar_length * scale))), y1), color)
    flag: RGB = (150, 84, 83) if stack.relation == "own" else (70, 90, 160)
    _fill(pixels, _rect(x, y, scale, style.flag_box), flag)
    if stack.army_plate:
        _fill(pixels, _rect(x, y, scale, style.army_plate_box), (10, 7, 148))
    glyphs = [np.array([[ch == "#" for ch in row] for row in BUILTIN_DIGITS[d]], dtype=np.uint8) * 255
              for d in str(stack.count)]
    images = [Image.fromarray(g).resize((max(1, int(round(g.shape[1] * scale))),
                                         max(1, int(round(style.digit_h * scale)))), Image.Resampling.NEAREST)
              for g in glyphs]
    gap = max(1, int(round(scale)))
    total = sum(image.width for image in images) + gap * (len(images) - 1)
    text = _rect(x, y, scale, style.plate_text_box)
    cursor = (text[0] + text[2] - total) // 2
    top = y + int(round(style.digit_top * scale))
    for image in images:
        ink = np.asarray(image) > 127
        pixels[top:top + image.height, cursor:cursor + image.width][ink] = (255, 255, 255)
        cursor += image.width + gap


def render_arena_frame(layout: ArenaLayout, stacks: list[SyntheticStack], calibration: ArenaVisionCalibration,
                       size: tuple[int, int] | None = None, *, controllers: dict[int, Country] | None = None,
                       speed: int | None = None, paused: bool | None = None,
                       templates: TemplateStore | None = None) -> Image.Image:
    """A full-client SYNTHETIC frame. Counters are centred on their province's calibrated centre;
    several stacks in one province are fanned out vertically."""
    width, height = size or (calibration.width, calibration.height)
    style = calibration.counter_style
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = BACKGROUND
    pixels[:int(calibration.map_rect[1] * height)] = TOP_BAR
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    radius = 0.03 * width
    for province in layout.provinces:
        owner = (controllers or {}).get(province.id)
        if owner is not None and owner.value in calibration.map_colors:
            cx, cy = calibration.layout_to_pixel(province.x, province.y, width, height)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                         fill=calibration.map_colors[owner.value])
    draw.text((4, 2), "SYNTHETIC - not HOI4", fill=(235, 235, 235))
    if templates is not None:
        for roi, name in (("speed", f"speed_{speed}" if speed else ""),
                          ("pause", "" if paused is None else ("pause_on" if paused else "pause_off"))):
            if name and roi in calibration.rois and templates.has(name):
                fx0, fy0, fx1, fy1 = calibration.rois[roi]
                box = (int(round(fx0 * width)), int(round(fy0 * height)),
                       int(round(fx1 * width)), int(round(fy1 * height)))
                patch = Image.fromarray(templates.get(name).astype(np.uint8)).convert("RGB")
                image.paste(patch.resize((max(1, box[2] - box[0]), max(1, box[3] - box[1])),
                                         Image.Resampling.BILINEAR), box[:2])
    pixels = np.array(image)
    frame_w, frame_h = int(round(style.frame_w * style.scale)), int(round(style.frame_h * style.scale))
    seen: dict[int, int] = {}
    for stack in stacks:
        province = layout.province(stack.province_id)
        cx, cy = calibration.layout_to_pixel(province.x, province.y, width, height)
        slot = seen.get(stack.province_id, 0)
        seen[stack.province_id] = slot + 1
        x = int(round(cx - frame_w / 2))
        y = int(round(cy - frame_h / 2)) + slot * (frame_h + max(2, int(2 * style.scale)))
        x = min(max(x, 1), width - int(round(style.army_plate_box[2] * style.scale)) - 1)
        y = min(max(y, 1), height - frame_h - 1)
        draw_counter(pixels, x, y, stack, style)
    return Image.fromarray(pixels)
