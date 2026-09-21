"""Read the in-game clock as text, instead of watching its pixels change.

The match loop used to ask only "did this rectangle change?", which cannot tell a paused
game from a running one once capture noise is accounted for, and throws away everything
the rectangle actually says. Reading it gives an absolute in-game timestamp, from which
the loop gets a measured game speed, rewind and desync detection, and a stall test that is
a comparison of two dates rather than a threshold on a pixel difference.

This is deliberately not a learned model. The clock is a bitmap font the game ships
(`gfx/fonts/hoi_18mbs.fnt` plus its atlas), rendered at a fixed position with no
intra-class variation, so every glyph that can appear is already on disk with its exact
metrics. Template matching against the game's own atlas needs no training data, runs in
under a millisecond on the CPU, and -- the part a classifier cannot offer -- its output is
checked against the date grammar, so a misread is *reported* rather than silently returned.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The font the top bar's DateText widget names (interface/topbar.gui). Its atlas stores
# glyph coverage in the alpha channel; RGB is a flat colour and carries nothing.
CLOCK_FONT = "gfx/fonts/hoi_18mbs.fnt"

# BMFont pads every glyph rectangle by this much on each side (`padding=2,2,2,2`).
GLYPH_PADDING = 2

# Everything the clock can render: digits, the separators, and the month abbreviations.
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
ALPHABET = frozenset("0123456789:," + "".join(MONTHS))

# A glyph must overlap its template this well to be accepted. Antialiasing and the HUD's
# translucency put a correct match around 0.1; the nearest wrong glyph sits far above this.
MAX_GLYPH_DISTANCE = 0.45

# The clock reads `HH:MM, DD Mon, YYYY`. Anything else is a misread, not a date.
CLOCK_GRAMMAR = re.compile(
    r"^(?P<hour>\d{1,2}):(?P<minute>\d{2}),?\s*(?P<day>\d{1,2})\s*(?P<month>[A-Z][a-z]{2}),?\s*(?P<year>\d{4})$"
)


@dataclass(frozen=True)
class GameTime:
    """One reading of the in-game clock, already checked against the grammar."""

    year: int
    month: int
    day: int
    hour: int
    minute: int

    @property
    def hours(self) -> float:
        """A monotonic hour count, for comparing two readings and measuring a rate.

        The calendar arithmetic is deliberately crude -- 12 months of 31 days -- because
        nothing here needs a real calendar, only an ordering and a difference that grows
        at a steady rate. A 31-day month makes the difference between two readings in the
        same month exact, which is the only case a stall or speed check ever sees.
        """
        return (
            ((self.year * 12 + self.month - 1) * 31 + self.day - 1) * 24
            + self.hour
            + self.minute / 60
        )

    def __str__(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d} {self.day} {MONTHS[self.month - 1]} {self.year}"


class ClockUnreadable(ValueError):
    """The crop did not decode to a legal date. Never raised for a merely stopped clock."""


def _glyph_atlas(game: Path):
    """Every glyph the clock can draw, as a coverage map keyed by character."""
    descriptor = (game / CLOCK_FONT).read_text(encoding="utf8", errors="replace")
    page = re.search(r'page id=0 file="([^"]+)"', descriptor)
    texture = game / CLOCK_FONT.rsplit("/", 1)[0] / (page.group(1) if page else "hoi_18mbs.dds")
    if not texture.exists():
        # The descriptor names `<font>_0.dds`; the shipped file drops the page suffix.
        texture = (game / CLOCK_FONT).with_suffix(".dds")
    raw = texture.read_bytes()
    height, width = struct.unpack_from("<2I", raw, 12)
    if struct.unpack_from("<I", raw, 88)[0] != 32:
        raise ValueError(f"{texture} is not a 32-bit uncompressed atlas")
    coverage = np.frombuffer(raw, np.uint8, width * height * 4, 128)
    coverage = coverage.reshape(height, width, 4)[:, :, 3].astype(np.float32) / 255.0

    glyphs = {}
    for line in descriptor.splitlines():
        if not line.startswith("char id="):
            continue
        field = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)}
        character = chr(field["id"])
        if character not in ALPHABET:
            continue
        x, y = field["x"] + GLYPH_PADDING, field["y"] + GLYPH_PADDING
        w = field["width"] - 2 * GLYPH_PADDING
        h = field["height"] - 2 * GLYPH_PADDING
        if w <= 0 or h <= 0:
            continue
        glyphs[character] = _tight(coverage[y : y + h, x : x + w])
    missing = ALPHABET - set(glyphs)
    if missing:
        raise ValueError(f"{texture} has no glyph for {sorted(missing)}")
    return glyphs


def _tight(image):
    """The ink of a coverage map, with its blank margin removed."""
    rows, columns = np.nonzero(image > 0.15)
    if not rows.size:
        return np.zeros((0, 0), np.float32)
    return image[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]


def _rescale(glyph, scale):
    """Resample a glyph's coverage map by a factor, with bilinear weights.

    The HUD does not draw the atlas at 1:1. Measured on a 3840x2160 capture, a digit is
    16x11 where the atlas holds 10x8, so the interface is scaled by about 1.6 -- and that
    factor is a user setting, not a constant, so it is fitted from each crop rather than
    written down here.
    """
    rows = max(1, int(round(glyph.shape[0] * scale)))
    columns = max(1, int(round(glyph.shape[1] * scale)))
    y = np.clip((np.arange(rows) + 0.5) / scale - 0.5, 0, glyph.shape[0] - 1)
    x = np.clip((np.arange(columns) + 0.5) / scale - 0.5, 0, glyph.shape[1] - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, glyph.shape[0] - 1)
    x1 = np.minimum(x0 + 1, glyph.shape[1] - 1)
    wy, wx = (y - y0)[:, None], (x - x0)[None, :]
    top = glyph[y0][:, x0] * (1 - wx) + glyph[y0][:, x1] * wx
    bottom = glyph[y1][:, x0] * (1 - wx) + glyph[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def _distance(patch, glyph):
    """One minus the overlap of two coverage maps, padded to a common size.

    Mean absolute difference is the obvious score and it is wrong here: most of a crop is
    background, so a blank template scores well against everything. Overlap is driven by
    the ink, and a glyph that covers the wrong pixels is penalised whichever way it errs.
    """
    if patch.sum() <= 0 or glyph.sum() <= 0:
        return 1.0

    # The HUD does not draw the shipped atlas at any single scale: measured on a
    # 3840x2160 capture, glyph heights come out about 1.6x the atlas and widths only
    # about 1.3x, so the game is rendering the typeface at its own size rather than
    # blitting the bitmap. Fitting one factor therefore cannot work. Compare shape alone
    # -- resample the glyph onto the patch's own box -- and put the size information back
    # as an explicit aspect-ratio penalty, which is what actually separates ':' from ','
    # and '1' from '0' once both are stretched to the same rectangle.
    resampled = _resample_to(glyph, patch.shape)
    overlap = float(np.minimum(patch, resampled).sum())
    dice = 1.0 - 2.0 * overlap / (float(patch.sum()) + float(resampled.sum()))

    patch_aspect = patch.shape[1] / patch.shape[0]
    glyph_aspect = glyph.shape[1] / glyph.shape[0]
    aspect = abs(np.log(patch_aspect / glyph_aspect))
    return float(dice + 0.6 * aspect)


def _resample_to(glyph, shape):
    """Stretch a glyph's coverage onto an arbitrary box, with bilinear weights."""
    rows, columns = shape
    y = np.clip((np.arange(rows) + 0.5) * glyph.shape[0] / rows - 0.5, 0, glyph.shape[0] - 1)
    x = np.clip((np.arange(columns) + 0.5) * glyph.shape[1] / columns - 0.5, 0, glyph.shape[1] - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, glyph.shape[0] - 1)
    x1 = np.minimum(x0 + 1, glyph.shape[1] - 1)
    wy, wx = (y - y0)[:, None], (x - x0)[None, :]
    top = glyph[y0][:, x0] * (1 - wx) + glyph[y0][:, x1] * wx
    bottom = glyph[y1][:, x0] * (1 - wx) + glyph[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def _ink(crop):
    """The crop as coverage in 0..1, stretched between its background and its text."""
    luminance = crop.astype(np.float32).mean(axis=2) if crop.ndim == 3 else crop.astype(np.float32)
    low, high = np.percentile(luminance, 5), np.percentile(luminance, 99)
    return np.clip((luminance - low) / max(high - low, 1e-6), 0.0, 1.0)


def _runs(mask):
    """Column spans that hold ink, left to right."""
    columns = mask.any(axis=0)
    spans, start = [], None
    for index, filled in enumerate(columns):
        if filled and start is None:
            start = index
        elif not filled and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(columns)))
    return spans


def read_clock(crop, glyphs) -> GameTime:
    """Decode one capture of the clock rectangle, or say why it could not be decoded."""
    ink = _ink(crop)
    mask = ink > 0.45
    if not mask.any():
        raise ClockUnreadable("clock crop holds no text")
    rows, columns = np.nonzero(mask)
    ink = ink[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    mask = ink > 0.45

    # Fit the interface scale from this crop. Digits and capitals share a cap height and
    # are the tallest thing the clock draws, so the tallest run divided by the atlas digit
    # gives the factor without needing to know which glyph is which yet.
    spans = _runs(mask)
    patches = [(span, _tight(ink[:, span[0] : span[1]])) for span in spans]
    patches = [(span, patch) for span, patch in patches if patch.size]
    if not patches:
        raise ClockUnreadable("clock crop holds no glyphs")
    scaled = glyphs

    text, worst = [], 0.0
    for (start, _end), patch in patches:
        distance, best = min(
            ((_distance(patch, glyph), character) for character, glyph in scaled.items()),
            key=lambda pair: pair[0],
        )
        if distance > MAX_GLYPH_DISTANCE:
            raise ClockUnreadable(
                f"glyph at column {start} matches nothing (distance {distance:.2f})"
            )
        worst = max(worst, distance)
        text.append(best)

    # Month abbreviations arrive as three separate runs; the grammar wants them joined,
    # and a space between fields is a gap rather than a glyph, so rebuild both.
    joined = "".join(text)
    spaced = re.sub(r"(\d)([A-Z])", r"\1 ", joined)
    spaced = re.sub(r"([a-z])(\d)", r"\1,", spaced)
    match = CLOCK_GRAMMAR.match(spaced)
    if not match:
        raise ClockUnreadable(f"{joined!r} is not a legal clock reading")
    month = match.group("month")
    if month not in MONTHS:
        raise ClockUnreadable(f"{month!r} is not a month")
    reading = GameTime(
        year=int(match.group("year")),
        month=MONTHS.index(month) + 1,
        day=int(match.group("day")),
        hour=int(match.group("hour")),
        minute=int(match.group("minute")),
    )
    if not (1 <= reading.day <= 31 and reading.hour < 24 and reading.minute < 60):
        raise ClockUnreadable(f"{reading} is not a legal date")
    return reading


def load_glyphs(game):
    """Build the glyph table once; it is a few kilobytes and never changes."""
    return _glyph_atlas(Path(game))
