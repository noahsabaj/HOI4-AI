import numpy as np
import pytest
from PIL import Image

from hoi4_agent.perception.digits import FallbackReader, GlyphReader, glyph_boxes
from hoi4_agent.perception.templates import TemplateStore
from hoi4_agent.schemas import GameDate

# A tiny synthetic font: distinct 4x3 patterns (dot is 2x2). Values are ink (1).
FONT = {
    "0": [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
    "1": [[0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1]],
    "3": [[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]],
    "6": [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1]],
    "9": [[1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
    ".": [[1, 0], [1, 1]],
    # checkerboard "unknown" glyph — deliberately unlike every digit
    "X": [[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]],
}


def _glyph_array(ch: str) -> np.ndarray:
    return np.asarray(FONT[ch], dtype=np.float32) * 255.0


def _strip(text: str) -> Image.Image:
    """Render `text` as a black strip with white glyphs, 2-px gaps + margins."""
    height = 6
    arrays = [_glyph_array(ch) for ch in text]
    width = 2 + sum(a.shape[1] + 2 for a in arrays)
    out = np.zeros((height, width), dtype=np.float32)
    x = 2
    for a in arrays:
        h, w = a.shape
        y = height - 1 - h  # bottom-aligned, 1-px bottom margin
        out[y : y + h, x : x + w] = a
        x += w + 2
    return Image.fromarray(out.astype(np.uint8), mode="L")


def _store(chars: str) -> TemplateStore:
    store = TemplateStore()
    for ch in chars:
        name = "glyph_dot" if ch == "." else f"glyph_{ch}"
        store.add(name, _glyph_array(ch))
    return store


def test_glyph_boxes_segments_each_glyph():
    from hoi4_agent.perception.ncc import to_gray_f32

    boxes = glyph_boxes(to_gray_f32(_strip("1936")))
    assert len(boxes) == 4
    assert boxes == sorted(boxes)  # left-to-right


def test_read_number_exact():
    reader = GlyphReader(_store("01369"), threshold=0.9)
    assert reader.read_number(_strip("36"), "free_civ_slots") == 36
    assert reader.read_number(_strip("1936"), "x") == 1936
    assert reader.read_number(_strip("0"), "x") == 0


def test_read_text_delta_ladder_splits_antialiased_bridges():
    # faint ink (60/255) bridging the glyph gaps: at delta 40 everything merges
    # into one box; the ladder's stricter deltas recover the real segmentation
    from hoi4_agent.perception.ncc import to_gray_f32

    strip = np.asarray(_strip("1936"), dtype=np.float32).copy()
    bridge_row = strip.shape[0] - 2
    strip[bridge_row, strip[bridge_row] == 0] = 60.0
    img = Image.fromarray(strip.astype(np.uint8), mode="L")
    assert len(glyph_boxes(to_gray_f32(img))) == 1  # default delta: one blob
    reader = GlyphReader(_store("01369"), threshold=0.9)
    assert reader.read_number(img, "x") == 1936  # ladder recovers it


def test_read_number_unknown_glyph_is_none():
    reader = GlyphReader(_store("0136"), threshold=0.9)  # no "9", no "X"
    assert reader.read_number(_strip("196"), "x") is None
    assert reader.read_number(_strip("1X6"), "x") is None
    assert reader.read_number(Image.new("L", (20, 6), 0), "x") is None  # blank crop


def test_read_date_and_range_validation():
    reader = GlyphReader(_store("01369."), threshold=0.9)
    assert reader.read_date(_strip("1936.1.1")) == GameDate(1936, 1, 1)
    assert reader.read_date(_strip("1936.13.1")) is None  # month 13 -> rejected
    assert reader.read_date(_strip("1936.1")) is None  # not three groups
    assert reader.read_date(_strip("1936")) is None


def test_glyph_reader_availability():
    assert GlyphReader(TemplateStore(), 0.9).available() is False
    assert GlyphReader(_store("1"), 0.9).available() is True


class _SpyReader:
    def __init__(self, number=7, date=GameDate(1940, 5, 10)):
        self.calls = []
        self._number, self._date = number, date

    def read_number(self, crop, field):
        self.calls.append(("number", field))
        return self._number

    def read_date(self, crop):
        self.calls.append(("date", None))
        return self._date


def test_fallback_reader_prefers_primary():
    primary = GlyphReader(_store("01369."), threshold=0.9)
    spy = _SpyReader()
    fr = FallbackReader(primary, spy)
    assert fr.read_number(_strip("36"), "x") == 36
    assert fr.read_date(_strip("1936.1.1")) == GameDate(1936, 1, 1)
    assert spy.calls == []  # fallback untouched while glyphs read cleanly


def test_fallback_reader_falls_back_per_call():
    primary = GlyphReader(_store("0136"), threshold=0.9)  # can't read "9"
    spy = _SpyReader(number=42)
    fr = FallbackReader(primary, spy)
    assert fr.read_number(_strip("99"), "x") == 42
    assert ("number", "x") in spy.calls
    # and with no primary at all
    assert FallbackReader(None, _SpyReader(number=5)).read_number(_strip("1"), "x") == 5
    assert FallbackReader(None, None).read_number(_strip("1"), "x") is None


def test_gamedate_rejects_out_of_range():
    with pytest.raises(ValueError):
        GameDate(1936, 13, 1)
    with pytest.raises(ValueError):
        GameDate(1936, 0, 1)
    with pytest.raises(ValueError):
        GameDate(1936, 1, 32)
